from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


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
            "One-click pipeline: videos -> rgb frames -> depth npy -> "
            "height/valid_mask/person_mask -> manifests"
        )
    )

    # Stage 1: video -> rgb
    parser.add_argument("--video-root", type=str, required=True, help="Input videos root: <video_root>/<person_id>/*.mp4")
    parser.add_argument("--data-root", type=str, required=True, help="Output dataroot used by HeightNet repro")
    parser.add_argument("--work-dir", type=str, default="data", help="Where to save split list files")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--exts", type=str, default=".mp4,.avi,.mov,.mkv")
    parser.add_argument("--overwrite-rgb", action="store_true")

    # Stage 2: rgb -> depth
    parser.add_argument("--depthanything-root", type=str, default="", help="Depth-Anything-V2 root path")
    parser.add_argument("--depth-encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--depth-checkpoint", type=str, default="", help="Depth Anything checkpoint .pth")
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--overwrite-depth", action="store_true")

    # Stage 3: depth -> labels/manifests
    parser.add_argument("--bg-depth-root", type=str, required=True)
    parser.add_argument("--camera-height-m", type=float, required=True)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--matches-root", type=str, default="data/matches")
    parser.add_argument("--fg-rel-thresh", type=float, default=0.06)
    parser.add_argument("--min-area", type=int, default=128)
    parser.add_argument("--person-mask-source", type=str, default="seg", choices=["seg", "depth"])
    parser.add_argument("--person-seg-model", type=str, default="")
    parser.add_argument("--person-seg-conf", type=float, default=0.25)
    parser.add_argument("--person-seg-iou", type=float, default=0.7)
    parser.add_argument("--person-seg-imgsz", type=int, default=640)
    parser.add_argument("--person-seg-strict-native", action="store_true")

    return parser.parse_args()


def resolve_depthanything_root(arg_root: str) -> Path:
    if arg_root:
        root = Path(arg_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"depthanything root not found: {root}")
        return root

    this_file = Path(__file__).resolve()
    # Try <...>/height/Depth-Anything-V2 first.
    cand1 = this_file.parents[2] / "Depth-Anything-V2"
    if cand1.exists():
        return cand1
    # Fallback to <...>/gait/Depth-Anything-V2.
    cand2 = this_file.parents[3] / "Depth-Anything-V2"
    if cand2.exists():
        return cand2

    raise FileNotFoundError(
        "Cannot auto-find Depth-Anything-V2 root. Please pass --depthanything-root."
    )


def resolve_checkpoint(depth_root: Path, encoder: str, ckpt_arg: str) -> Path:
    if ckpt_arg:
        ckpt = Path(ckpt_arg).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"depth checkpoint not found: {ckpt}")
        return ckpt

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


def rgb_to_depth_npy(
    data_root: Path,
    model,
    input_size: int,
    overwrite_depth: bool,
) -> None:
    total = 0
    done = 0
    skipped = 0

    for split in ("train", "val", "test"):
        rgb_files = sorted(glob.glob(str(data_root / split / "rgb" / "*" / "*.png")))
        for rgb_path_str in rgb_files:
            rgb_path = Path(rgb_path_str)
            sequence_id = rgb_path.parent.name
            stem = rgb_path.stem
            out_dir = data_root / split / "depth" / sequence_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}.npy"
            total += 1

            if out_path.exists() and not overwrite_depth:
                skipped += 1
                continue

            img = cv2.imread(str(rgb_path))
            if img is None:
                continue
            depth = model.infer_image(img, input_size).astype(np.float32)
            np.save(str(out_path), depth)
            done += 1

            if done % 500 == 0:
                print(f"[depth] saved {done} files ...")

    print(f"[depth] total={total}, saved={done}, skipped_existing={skipped}")
    if total == 0:
        raise RuntimeError(f"No RGB frames found under {data_root}/{{train,val,test}}/rgb")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    data_root = Path(args.data_root).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if not video_root.exists():
        raise FileNotFoundError(f"video root not found: {video_root}")

    # Stage 1: videos -> rgb
    prepare_midrun = project_root / "tools" / "prepare_midrun_dataset.py"
    run(
        [
            sys.executable,
            str(prepare_midrun),
            "--video-root",
            str(video_root),
            "--data-root",
            str(data_root),
            "--work-dir",
            str(work_dir),
            "--sample-size",
            str(args.sample_size),
            "--seed",
            str(args.seed),
            "--fps",
            str(args.fps),
            "--max-frames",
            str(args.max_frames),
            "--train-ratio",
            str(args.train_ratio),
            "--val-ratio",
            str(args.val_ratio),
            "--exts",
            args.exts,
            *(["--overwrite"] if args.overwrite_rgb else []),
        ]
    )

    # Stage 2: rgb -> depth npy
    depth_root = resolve_depthanything_root(args.depthanything_root)
    checkpoint = resolve_checkpoint(depth_root, args.depth_encoder, args.depth_checkpoint)
    model, device = build_depth_model(depth_root, args.depth_encoder, checkpoint)
    print(f"[depth] device={device}, root={depth_root}, checkpoint={checkpoint}")
    rgb_to_depth_npy(
        data_root=data_root,
        model=model,
        input_size=args.depth_input_size,
        overwrite_depth=args.overwrite_depth,
    )

    # Stage 3: depth -> height/valid_mask/person_mask + manifests
    prepare_e2e = project_root / "tools" / "prepare_end2end_data.py"
    cmd = [
        sys.executable,
        str(prepare_e2e),
        "--data-root",
        str(data_root),
        "--bg-depth-root",
        str(Path(args.bg_depth_root).expanduser().resolve()),
        "--camera-height-m",
        str(args.camera_height_m),
        "--out-dir",
        str(Path(args.out_dir).expanduser().resolve()),
        "--matches-root",
        str(Path(args.matches_root).expanduser().resolve()),
        "--fg-rel-thresh",
        str(args.fg_rel_thresh),
        "--min-area",
        str(args.min_area),
        "--person-mask-source",
        args.person_mask_source,
    ]
    if args.person_mask_source == "seg":
        if not args.person_seg_model:
            raise ValueError("--person-seg-model is required when --person-mask-source seg")
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
    print("[done] videos -> depth -> height -> manifest pipeline completed.")


if __name__ == "__main__":
    main()
