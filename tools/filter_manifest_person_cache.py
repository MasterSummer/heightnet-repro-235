from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.person_cache import person_bbox_cache_path, person_mask_cache_path


def _cache_is_valid(frame_path: str, min_pixels: int) -> tuple[bool, str]:
    if not frame_path:
        return False, "missing_frame_path"
    bbox_path = person_bbox_cache_path(frame_path)
    mask_path = person_mask_cache_path(frame_path)
    if not os.path.exists(bbox_path):
        return False, "missing_bbox"
    if not os.path.exists(mask_path):
        return False, "missing_mask"
    try:
        bbox = np.load(bbox_path).astype(np.float32).reshape(-1)
        mask = np.load(mask_path)
    except Exception:
        return False, "unreadable_cache"
    if bbox.shape[0] < 4 or np.any(~np.isfinite(bbox[:4])) or float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
        return False, "invalid_bbox"
    if mask.ndim != 2 or int(np.count_nonzero(mask > 0)) < min_pixels:
        return False, "empty_mask"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only manifest rows with valid .person_bbox.npy and .person_mask.npy caches.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    keep = []
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for row in df.to_dict(orient="records"):
        frame_path = str(row.get("frame_path", ""))
        ok, reason = _cache_is_valid(frame_path, int(args.min_mask_pixels))
        counts[reason] = counts.get(reason, 0) + 1
        if ok:
            keep.append(row)
        elif len(examples.setdefault(reason, [])) < 10:
            examples[reason].append(frame_path)

    out_manifest = Path(args.out_manifest).expanduser().resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(keep).to_csv(out_manifest, index=False)

    report = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "out_manifest": str(out_manifest),
        "input_rows": int(len(df)),
        "kept_rows": int(len(keep)),
        "dropped_rows": int(len(df) - len(keep)),
        "counts": counts,
        "examples": examples,
        "min_mask_pixels": int(args.min_mask_pixels),
    }
    out_report = Path(args.out_report).expanduser().resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[FILTER_PERSON_CACHE] input={len(df)} kept={len(keep)} dropped={len(df) - len(keep)} -> {out_manifest}")


if __name__ == "__main__":
    main()
