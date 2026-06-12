from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.person_cache import person_bbox_cache_path, person_mask_cache_path


def draw_overlay(frame_path: str, out_path: Path) -> bool:
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    mask_path = person_mask_cache_path(frame_path)
    bbox_path = person_bbox_cache_path(frame_path)
    if not os.path.exists(mask_path) or not os.path.exists(bbox_path):
        return False
    mask = np.load(mask_path).astype(bool)
    bbox = np.load(bbox_path).astype(np.float32).reshape(-1)[:4]
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = img.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), overlay))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write bbox+mask overlay samples for jianzhi person-region cache QA.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    rows = frame.to_dict(orient="records")
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    out_dir = Path(args.out_dir).expanduser().resolve()
    written = 0
    for row in rows:
        frame_path = str(row.get("frame_path", "")).strip()
        if not frame_path:
            continue
        out_path = out_dir / f"{written:04d}_{Path(frame_path).stem}.jpg"
        if draw_overlay(frame_path, out_path):
            written += 1
            if written >= int(args.limit):
                break
    print(f"[VIS] written={written} out_dir={out_dir}")


if __name__ == "__main__":
    main()
