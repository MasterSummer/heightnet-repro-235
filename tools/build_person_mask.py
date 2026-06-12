from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np


def _infer_camera_id(sequence_id: str) -> str | None:
    m = re.search(r"\d+cm_(?:inside|outside|slantside|side|front|back)", sequence_id, re.IGNORECASE)
    if m:
        return m.group(0).lower()
    return None


def _load_bg_depth(bg_depth_root: str, camera_id: str, target_hw: tuple[int, int]) -> np.ndarray:
    path = os.path.join(bg_depth_root, camera_id, f"{camera_id}_avg_depth.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"background depth map not found: {path}")
    arr = np.load(path).astype(np.float32)
    h, w = target_hw
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return arr


def compute_person_mask(
    depth: np.ndarray,
    bg_depth: np.ndarray,
    fg_rel_thresh: float = 0.06,
    min_area: int = 128,
) -> np.ndarray:
    """
    Foreground mask from relative depth gap:
      rel = (D_b - D_f) / D_b
      person_mask = rel > fg_rel_thresh
    """
    d_f = depth.astype(np.float32)
    d_b = bg_depth.astype(np.float32)
    valid = np.isfinite(d_f) & np.isfinite(d_b) & (np.abs(d_b) > 1e-6)

    rel = np.zeros_like(d_f, dtype=np.float32)
    rel[valid] = (d_b[valid] - d_f[valid]) / np.maximum(d_b[valid], 1e-6)
    fg = (rel > fg_rel_thresh).astype(np.uint8)

    k = np.ones((3, 3), dtype=np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = np.zeros_like(fg, dtype=np.uint8)
    for lab in range(1, nlab):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == lab] = 1
    return out.astype(np.float32)


def process_split(
    data_root: str,
    split: str,
    bg_depth_root: str,
    fg_rel_thresh: float,
    min_area: int,
) -> int:
    split_root = os.path.join(data_root, split)
    depth_files = sorted(glob.glob(os.path.join(split_root, "depth", "*", "*.npy")))
    if not depth_files:
        raise RuntimeError(f"no depth files found in split={split}, root={os.path.join(split_root, 'depth')}")

    count = 0
    for depth_path in depth_files:
        rel = os.path.relpath(depth_path, os.path.join(split_root, "depth"))
        sequence_id = rel.split(os.sep)[0]
        frame_stem = os.path.splitext(os.path.basename(depth_path))[0]

        camera_id = _infer_camera_id(sequence_id)
        if not camera_id:
            raise ValueError(f"cannot infer camera id from sequence_id: {sequence_id}")

        depth = np.load(depth_path).astype(np.float32)
        bg = _load_bg_depth(bg_depth_root, camera_id, (depth.shape[0], depth.shape[1]))
        person_mask = compute_person_mask(depth, bg, fg_rel_thresh=fg_rel_thresh, min_area=min_area)

        out_dir = os.path.join(split_root, "person_mask", sequence_id)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{frame_stem}.npy"), person_mask)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build person masks from depth + camera background depth map.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--bg-depth-root", type=str, required=True)
    parser.add_argument("--fg-rel-thresh", type=float, default=0.06)
    parser.add_argument("--min-area", type=int, default=128)
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    total = 0
    for s in splits:
        n = process_split(
            data_root=args.data_root,
            split=s,
            bg_depth_root=args.bg_depth_root,
            fg_rel_thresh=args.fg_rel_thresh,
            min_area=args.min_area,
        )
        total += n
        print(f"[{s}] person_mask generated: {n}")
    print(f"Done. total={total}")


if __name__ == "__main__":
    main()
