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
from heightnet.parsing_mask import build_full_frame_mask_from_parsing, resolve_extracted_parsing_path
from heightnet.person_cache import person_bbox_cache_path, person_mask_cache_path
from heightnet.runtime_seg import PersonSegmenter


def _load_rgb(frame_path: str, image_height: int, image_width: int) -> np.ndarray:
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"cannot open frame image: {frame_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (int(image_width), int(image_height)), interpolation=cv2.INTER_LINEAR)


def _iter_unique_rows(manifest_paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        frame = pd.read_csv(manifest_path)
        if "frame_path" not in frame.columns or "parsing_member" not in frame.columns:
            raise ValueError(f"manifest must contain frame_path and parsing_member: {manifest_path}")
        for row in frame.to_dict(orient="records"):
            frame_path = str(row.get("frame_path", "")).strip()
            if not frame_path or frame_path in seen:
                continue
            seen.add(frame_path)
            rows.append(row)
    return rows


def _process_batch(
    rows: list[dict],
    parsing_root: Path,
    segmenter: PersonSegmenter,
    device: torch.device,
    image_height: int,
    image_width: int,
) -> tuple[int, int]:
    images = []
    valid_rows = []
    skipped = 0
    for row in rows:
        frame_path = str(row["frame_path"])
        parsing_path = resolve_extracted_parsing_path(parsing_root, str(row["parsing_member"]))
        if not os.path.exists(frame_path) or not parsing_path.exists():
            skipped += 1
            continue
        images.append(
            torch.from_numpy(np.transpose(_load_rgb(frame_path, image_height, image_width), (2, 0, 1))).contiguous()
        )
        valid_rows.append(row)
    if not valid_rows:
        return 0, skipped

    image_batch = torch.stack(images, dim=0)
    _yolo_masks, boxes = segmenter.infer_batch_regions(image_batch, device)
    boxes_np = boxes.detach().cpu().numpy()

    generated = 0
    for idx, row in enumerate(valid_rows):
        frame_path = str(row["frame_path"])
        rgb = images[idx]
        height = int(rgb.shape[1])
        width = int(rgb.shape[2])
        parsing_path = resolve_extracted_parsing_path(parsing_root, str(row["parsing_member"]))
        parsing_mask = cv2.imread(str(parsing_path), cv2.IMREAD_GRAYSCALE)
        if parsing_mask is None:
            skipped += 1
            continue
        full_mask = build_full_frame_mask_from_parsing(
            parsing_mask=parsing_mask,
            bbox_xyxy=boxes_np[idx],
            image_height=height,
            image_width=width,
        )
        np.save(person_mask_cache_path(frame_path), (full_mask > 0).astype(np.uint8))
        np.save(person_bbox_cache_path(frame_path), boxes_np[idx].astype(np.float32))
        generated += 1
    return generated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute person masks from parsing crops placed by YOLO bbox caches.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--parsing-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_height, image_width = cfg.data.image_size
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
        frame_path = str(row["frame_path"])
        mask_path = Path(person_mask_cache_path(frame_path))
        bbox_path = Path(person_bbox_cache_path(frame_path))
        if not args.overwrite_existing and mask_path.exists() and bbox_path.exists():
            skipped_existing += 1
            continue
        pending.append(row)

    generated = 0
    skipped_missing = 0
    batch_size = max(1, int(args.batch_size))
    parsing_root = Path(args.parsing_root).resolve()
    for start in range(0, len(pending), batch_size):
        done, skipped = _process_batch(
            pending[start : start + batch_size],
            parsing_root=parsing_root,
            segmenter=segmenter,
            device=device,
            image_height=int(image_height),
            image_width=int(image_width),
        )
        generated += done
        skipped_missing += skipped
        if (start // batch_size) % 20 == 0:
            print(
                f"[PARSING_PERSON_CACHE] processed={min(start + batch_size, len(pending))}/{len(pending)} "
                f"generated={generated} skipped_missing={skipped_missing}"
            )

    print(
        f"[PARSING_PERSON_CACHE] unique_frames={len(unique_rows)} pending={len(pending)} "
        f"generated={generated} skipped_existing={skipped_existing} skipped_missing={skipped_missing}"
    )


if __name__ == "__main__":
    main()
