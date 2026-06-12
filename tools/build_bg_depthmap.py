from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from heightnet.runtime_depth import RuntimeDepthEstimator


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build camera-wise average background depth maps from background images."
    )
    parser.add_argument("--bg-image-root", type=str, required=True, help="Root directory containing per-camera background images.")
    parser.add_argument("--out-root", type=str, required=True, help="Output root for <camera>/<camera>_avg_depth.npy")
    parser.add_argument("--depthanything-root", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--save-preview", action="store_true", help="Also save normalized preview PNG per camera.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _iter_camera_dirs(bg_image_root: Path) -> list[Path]:
    return sorted([p for p in bg_image_root.iterdir() if p.is_dir()])


def _collect_images(camera_dir: Path) -> list[Path]:
    return sorted(
        [p for p in camera_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _normalize_preview(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.uint8)
    vals = x[valid]
    lo = np.percentile(vals, 2)
    hi = np.percentile(vals, 98)
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    bg_image_root = Path(args.bg_image_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = RuntimeDepthEstimator(
        depthanything_root=args.depthanything_root,
        encoder=args.encoder,
        checkpoint=args.checkpoint,
        input_size=args.input_size,
    ).to(device)

    camera_dirs = _iter_camera_dirs(bg_image_root)
    if not camera_dirs:
        raise RuntimeError(f"no camera directories found under: {bg_image_root}")

    for camera_dir in camera_dirs:
        camera = camera_dir.name
        image_paths = _collect_images(camera_dir)
        if not image_paths:
            print(f"[skip] {camera}: no images")
            continue

        camera_out = out_root / camera
        camera_out.mkdir(parents=True, exist_ok=True)
        out_npy = camera_out / f"{camera}_avg_depth.npy"
        out_png = camera_out / f"{camera}_avg_depth_preview.png"
        out_txt = camera_out / "summary.txt"
        if out_npy.exists() and not args.overwrite:
            print(f"[skip] {camera}: exists -> {out_npy}")
            continue

        acc = None
        count = 0
        target_hw = None
        for img_path in image_paths:
            rgb = _load_rgb(img_path)
            target_hw = rgb.shape[:2]
            image_raw = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            depth = estimator.infer_batch(image_raw)[0, 0].detach().cpu().numpy().astype(np.float32)
            if acc is None:
                acc = np.zeros_like(depth, dtype=np.float64)
            acc += depth
            count += 1

        if acc is None or count == 0 or target_hw is None:
            print(f"[skip] {camera}: no valid depths")
            continue

        avg_depth = (acc / float(count)).astype(np.float32)
        np.save(out_npy, avg_depth)
        if args.save_preview:
            cv2.imwrite(str(out_png), _normalize_preview(avg_depth))
        out_txt.write_text(
            f"camera={camera}\nimages_used={count}\nshape={avg_depth.shape[0]}x{avg_depth.shape[1]}\nencoder={args.encoder}\ninput_size={args.input_size}\n",
            encoding="utf-8",
        )
        print(f"[done] {camera}: images_used={count}, out={out_npy}")


if __name__ == "__main__":
    main()
