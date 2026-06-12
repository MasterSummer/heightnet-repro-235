from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def _torch_load_compat(
    path: str,
    map_location: str | torch.device,
    *,
    weights_only: bool,
):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-click from existing RGB dataset: "
            "rgb -> depth -> height/valid_mask/person_mask -> manifests -> train"
        )
    )
    parser.add_argument("--data-root", type=str, required=True, help="Existing dataroot with train/val/test/rgb")
    parser.add_argument("--bg-depth-root", type=str, required=True, help="Camera-wise background depth map root")
    parser.add_argument("--camera-height-m", type=float, default=-1.0, help="Used when --camera-height-mode=fixed")
    parser.add_argument(
        "--camera-height-mode",
        type=str,
        default="fixed",
        choices=["fixed", "from-sequence-cm"],
        help="Use fixed camera height or infer from sequence name (e.g. 350cm_* -> 3.5m).",
    )

    # Depth Anything
    parser.add_argument("--depthanything-root", type=str, default="", help="Depth-Anything-V2 root")
    parser.add_argument("--depth-encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--depth-checkpoint", type=str, default="", help="Depth Anything checkpoint .pth")
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--overwrite-depth", action="store_true")

    # Labels/manifests
    parser.add_argument("--out-dir", type=str, default="", help="Where train/val/test_manifest.csv are written")
    parser.add_argument("--matches-root", type=str, default="", help="Matches root in manifests")
    parser.add_argument("--fg-rel-thresh", type=float, default=0.06)
    parser.add_argument("--min-area", type=int, default=128)
    parser.add_argument("--person-mask-source", type=str, default="depth", choices=["depth", "seg"])
    parser.add_argument("--person-seg-model", type=str, default="")
    parser.add_argument("--person-seg-conf", type=float, default=0.25)
    parser.add_argument("--person-seg-iou", type=float, default=0.7)
    parser.add_argument("--person-seg-imgsz", type=int, default=640)
    parser.add_argument("--person-seg-strict-native", action="store_true")

    # Train
    parser.add_argument("--train-config-template", type=str, default="configs/default.yaml")
    parser.add_argument("--train-config-out", type=str, default="configs/auto_from_rgb.yaml")
    parser.add_argument("--run-name", type=str, default="auto_from_rgb")
    parser.add_argument("--epochs", type=int, default=-1, help="Override train.epochs when > 0")
    parser.add_argument("--batch-size", type=int, default=-1, help="Override train.batch_size when > 0")
    parser.add_argument("--num-workers", type=int, default=-1, help="Override train.num_workers when >= 0")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data/manifests/config; do not launch train")
    parser.add_argument(
        "--launcher",
        type=str,
        default="none",
        choices=["none", "torchrun"],
        help="Training launcher. Use torchrun for multi-GPU DDP training.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Processes/GPUs per node when launcher=torchrun")
    return parser.parse_args()


def resolve_depthanything_root(project_root: Path, arg_root: str) -> Path:
    if arg_root:
        root = Path(arg_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"depthanything root not found: {root}")
        return root

    cand1 = project_root.parent / "Depth-Anything-V2"
    if cand1.exists():
        return cand1
    cand2 = project_root.parent.parent / "Depth-Anything-V2"
    if cand2.exists():
        return cand2
    raise FileNotFoundError("Cannot auto-find Depth-Anything-V2 root. Please pass --depthanything-root.")


def resolve_checkpoint(depth_root: Path, encoder: str, ckpt_arg: str) -> Path:
    if ckpt_arg:
        p = Path(ckpt_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Depth checkpoint not found: {p}")
        return p
    cands = [
        depth_root / "checkpoints" / f"depth_anything_v2_{encoder}.pth",
        depth_root / "checkpoint" / f"depth_anything_v2_{encoder}.pth",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Depth checkpoint not found. Tried:\n"
        + "\n".join(str(x) for x in cands)
        + "\nPlease pass --depth-checkpoint."
    )


def build_depth_model(depth_root: Path, encoder: str, checkpoint: Path):
    if str(depth_root) not in sys.path:
        sys.path.insert(0, str(depth_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    state_dict = _torch_load_compat(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device


def rgb_to_depth_npy(data_root: Path, model, input_size: int, overwrite: bool) -> None:
    total = 0
    saved = 0
    skipped = 0
    for split in ("train", "val", "test"):
        rgb_files = sorted(glob.glob(str(data_root / split / "rgb" / "*" / "*.png")))
        for rgb in rgb_files:
            total += 1
            rgb_path = Path(rgb)
            seq = rgb_path.parent.name
            stem = rgb_path.stem
            out_dir = data_root / split / "depth" / seq
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}.npy"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            img = cv2.imread(str(rgb_path))
            if img is None:
                continue
            depth = model.infer_image(img, input_size).astype(np.float32)
            np.save(str(out_path), depth)
            saved += 1
            if saved % 500 == 0:
                print(f"[depth] saved {saved} files ...")
    print(f"[depth] total={total}, saved={saved}, skipped_existing={skipped}")
    if total == 0:
        raise RuntimeError(f"No RGB frames found under {data_root}/{{train,val,test}}/rgb")
    if saved + skipped == 0:
        raise RuntimeError(
            "Depth generation produced 0 npy files. "
            "Please check RGB decoding/readability and depth model setup."
        )


def write_train_config(
    project_root: Path,
    template_path: Path,
    config_out_path: Path,
    out_dir: Path,
    run_name: str,
    person_mask_source: str,
    person_seg_model: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> None:
    with template_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["train_manifest"] = str((out_dir / "train_manifest.csv").resolve())
    cfg["paths"]["val_manifest"] = str((out_dir / "val_manifest.csv").resolve())
    cfg["paths"]["test_manifest"] = str((out_dir / "test_manifest.csv").resolve())
    cfg["paths"]["output_dir"] = str((project_root / "runs" / run_name).resolve())

    if person_mask_source == "seg":
        if not person_seg_model:
            raise ValueError("--person-seg-model is required when person-mask-source=seg")
        cfg["runtime_seg"]["enabled"] = True
        cfg["runtime_seg"]["model_path"] = str(Path(person_seg_model).expanduser().resolve())
    else:
        cfg["runtime_seg"]["enabled"] = False
        cfg["runtime_seg"]["model_path"] = ""

    if epochs > 0:
        cfg["train"]["epochs"] = int(epochs)
    if batch_size > 0:
        cfg["train"]["batch_size"] = int(batch_size)
    if num_workers >= 0:
        cfg["train"]["num_workers"] = int(num_workers)

    config_out_path.parent.mkdir(parents=True, exist_ok=True)
    with config_out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    data_root = Path(args.data_root).expanduser().resolve()
    bg_depth_root = Path(args.bg_depth_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not bg_depth_root.exists():
        raise FileNotFoundError(f"bg depth root not found: {bg_depth_root}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (project_root / "data")
    matches_root = (
        Path(args.matches_root).expanduser().resolve() if args.matches_root else (project_root / "data" / "matches")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    matches_root.mkdir(parents=True, exist_ok=True)

    # 1) rgb -> depth
    depth_root = resolve_depthanything_root(project_root, args.depthanything_root)
    checkpoint = resolve_checkpoint(depth_root, args.depth_encoder, args.depth_checkpoint)
    model, device = build_depth_model(depth_root, args.depth_encoder, checkpoint)
    print(f"[depth] device={device}, checkpoint={checkpoint}")
    rgb_to_depth_npy(data_root=data_root, model=model, input_size=args.depth_input_size, overwrite=args.overwrite_depth)

    # 2) depth -> labels + manifests
    prepare_e2e = project_root / "tools" / "prepare_end2end_data.py"
    cmd = [
        sys.executable,
        str(prepare_e2e),
        "--data-root",
        str(data_root),
        "--bg-depth-root",
        str(bg_depth_root),
        "--camera-height-mode",
        args.camera_height_mode,
        "--out-dir",
        str(out_dir),
        "--matches-root",
        str(matches_root),
        "--fg-rel-thresh",
        str(args.fg_rel_thresh),
        "--min-area",
        str(args.min_area),
        "--person-mask-source",
        args.person_mask_source,
    ]
    if args.camera_height_mode == "fixed":
        if args.camera_height_m <= 0:
            raise ValueError("--camera-height-m must be > 0 when --camera-height-mode=fixed")
        cmd.extend(["--camera-height-m", str(args.camera_height_m)])
    if args.person_mask_source == "seg":
        if not args.person_seg_model:
            raise ValueError("--person-seg-model is required when person-mask-source=seg")
        cmd.extend(
            [
                "--person-seg-model",
                str(Path(args.person_seg_model).expanduser().resolve()),
                "--person-seg-conf",
                str(args.person_seg_conf),
                "--person-seg-iou",
                str(args.person_seg_iou),
                "--person-seg-imgsz",
                str(args.person_seg_imgsz),
            ]
        )
        if args.person_seg_strict_native:
            cmd.append("--person-seg-strict-native")
    run(cmd)

    # 3) write train config
    template = Path(args.train_config_template).expanduser()
    if not template.is_absolute():
        template = (project_root / template).resolve()
    config_out = Path(args.train_config_out).expanduser()
    if not config_out.is_absolute():
        config_out = (project_root / config_out).resolve()
    write_train_config(
        project_root=project_root,
        template_path=template,
        config_out_path=config_out,
        out_dir=out_dir,
        run_name=args.run_name,
        person_mask_source=args.person_mask_source,
        person_seg_model=args.person_seg_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[config] written: {config_out}")

    if args.prepare_only:
        print("[done] preparation finished. training not started (--prepare-only).")
        return

    # 4) launch training
    train_py = project_root / "train.py"
    if args.launcher == "torchrun":
        if args.nproc_per_node <= 1:
            raise ValueError("--nproc-per-node must be > 1 when --launcher torchrun")
        run(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(args.nproc_per_node),
                str(train_py),
                "--config",
                str(config_out),
            ]
        )
    else:
        run([sys.executable, str(train_py), "--config", str(config_out)])


if __name__ == "__main__":
    main()
