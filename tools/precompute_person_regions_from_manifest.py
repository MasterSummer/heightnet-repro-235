from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.config import load_config
from heightnet.person_cache import person_bbox_cache_path, person_mask_cache_path
from heightnet.runtime_seg import PersonSegmenter


def _load_rgb(frame_path: str) -> np.ndarray:
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"cannot open frame image: {frame_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# Cache for open VideoCapture objects: video_path -> cv2.VideoCapture
_video_cap_cache: dict[str, cv2.VideoCapture] = {}


def _get_video_cap(video_path: str) -> cv2.VideoCapture:
    if video_path not in _video_cap_cache:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        _video_cap_cache[video_path] = cap
    return _video_cap_cache[video_path]


def _load_rgb_from_video(video_path: str, frame_idx: int) -> np.ndarray:
    cap = _get_video_cap(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        raise FileNotFoundError(f"cannot read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _iter_unique_rows(manifest_paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        df = pd.read_csv(manifest_path)
        usecols = ["frame_path"]
        if "video_path" in df.columns:
            usecols.append("video_path")
        if "frame_idx" in df.columns:
            usecols.append("frame_idx")
        for row in df.to_dict(orient="records"):
            frame_path = str(row.get("frame_path", "")).strip()
            if not frame_path or frame_path in seen:
                continue
            seen.add(frame_path)
            rows.append(row)
    return rows


def _load_image_for_row(row: dict) -> np.ndarray | None:
    frame_path = str(row.get("frame_path", "")).strip()

    if frame_path and os.path.exists(frame_path):
        return _load_rgb(frame_path)

    video_path = str(row.get("video_path", "")).strip()
    frame_idx = row.get("frame_idx")
    if video_path and frame_idx is not None and os.path.exists(video_path):
        try:
            return _load_rgb_from_video(video_path, int(frame_idx))
        except (FileNotFoundError, ValueError):
            return None

    return None


def _process_batch(
    batch_rows: list[dict],
    segmenter: PersonSegmenter,
    device: torch.device,
) -> tuple[int, int]:
    images = []
    valid_rows = []
    skipped = 0
    for row in batch_rows:
        rgb = _load_image_for_row(row)
        if rgb is None:
            skipped += 1
            continue
        images.append(torch.from_numpy(np.transpose(rgb, (2, 0, 1))).contiguous())
        valid_rows.append(row)
    if not valid_rows:
        return 0, skipped

    image_batch = torch.stack(images, dim=0)
    masks, boxes = segmenter.infer_batch_regions(image_batch, device)
    masks = masks.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()

    for idx, row in enumerate(valid_rows):
        frame_path = str(row.get("frame_path", "")).strip()
        np.save(person_mask_cache_path(frame_path), masks[idx, 0].astype(np.uint8))
        np.save(person_bbox_cache_path(frame_path), boxes[idx].astype(np.float32))
    return len(valid_rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute YOLO person mask/bbox caches for frame manifests.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    segmenter = PersonSegmenter(
        model_path=cfg.runtime_seg.model_path,
        conf=cfg.runtime_seg.conf,
        iou=cfg.runtime_seg.iou,
        imgsz=cfg.runtime_seg.imgsz,
        strict_native=cfg.runtime_seg.strict_native,
    )

    unique_rows = _iter_unique_rows(args.manifest)
    pending: list[dict] = []
    skipped_existing = 0
    for row in unique_rows:
        frame_path = str(row.get("frame_path", "")).strip()
        mask_path = Path(person_mask_cache_path(frame_path))
        bbox_path = Path(person_bbox_cache_path(frame_path))
        if not args.overwrite_existing and mask_path.exists() and bbox_path.exists():
            skipped_existing += 1
            continue
        pending.append(row)

    generated = 0
    skipped_missing = 0
    batch_size = max(1, args.batch_size)
    for start in range(0, len(pending), batch_size):
        done, skipped = _process_batch(
            pending[start : start + batch_size],
            segmenter=segmenter,
            device=device,
        )
        generated += done
        skipped_missing += skipped
        if (start // batch_size) % 20 == 0:
            print(
                f"[PERSON_CACHE] processed={min(start + batch_size, len(pending))}/{len(pending)} "
                f"generated={generated} skipped_missing={skipped_missing}"
            )

    for cap in _video_cap_cache.values():
        cap.release()
    _video_cap_cache.clear()

    print(
        f"[PERSON_CACHE] unique_frames={len(unique_rows)} pending={len(pending)} "
        f"generated={generated} skipped_existing={skipped_existing} skipped_missing={skipped_missing}"
    )


if __name__ == "__main__":
    main()
