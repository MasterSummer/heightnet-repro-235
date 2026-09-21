#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _video_stem(row: dict) -> str:
    video_filename = str(row.get("video_filename") or Path(str(row.get("video_path") or "")).name)
    return Path(video_filename).stem


def _person_id(row: dict) -> str:
    return str(row.get("person_id") or row.get("pid") or row.get("penson_id") or "")


def _old_bbox_to_training_schema(old_bbox: np.ndarray) -> np.ndarray:
    """Convert legacy [cx, cy, w, h, aspect, rel_h, area] rows to training schema."""
    old = np.asarray(old_bbox, dtype=np.float32)
    out = np.zeros_like(old, dtype=np.float32)
    cx = old[:, 0]
    cy = old[:, 1]
    w = old[:, 2]
    h = old[:, 3]
    area = old[:, 6]
    y1 = cy - 0.5 * h
    y2 = cy + 0.5 * h
    out[:, 0] = h
    out[:, 1] = w
    out[:, 2] = -y1
    out[:, 3] = y2
    out[:, 4] = cy
    out[:, 5] = area
    out[:, 6] = 1.0
    return out.astype(np.float32, copy=False)


def _valid_rows(bbox: np.ndarray, stats: np.ndarray, crops: np.ndarray) -> np.ndarray:
    bbox_ok = np.isfinite(bbox).all(axis=1) & (np.abs(bbox).sum(axis=1) > 1e-8)
    stats_ok = np.isfinite(stats).all(axis=1)
    crops_flat = crops.reshape(crops.shape[0], -1)
    crops_ok = np.isfinite(crops_flat).all(axis=1) & (np.abs(crops_flat).sum(axis=1) > 1e-6)
    return bbox_ok & stats_ok & crops_ok


def repack_one(src: Path, dst: Path) -> dict:
    with np.load(src, allow_pickle=False) as data:
        bbox = np.asarray(data["bbox_feats"], dtype=np.float32)
        stats = np.asarray(data["height_stats"], dtype=np.float32)
        crops = np.asarray(data["heightmap_crops"], dtype=np.float32)
        camera_id = np.asarray(data["camera_id"]) if "camera_id" in data.files else np.asarray([""])
        video_stem = np.asarray(data["video_stem"]) if "video_stem" in data.files else np.asarray([src.stem])

    if bbox.ndim != 2 or bbox.shape[1] != 7 or stats.ndim != 2 or crops.ndim != 4:
        return {"status": "invalid_schema", "src": str(src)}
    if len(bbox) != len(stats) or len(bbox) != len(crops):
        return {"status": "count_mismatch", "src": str(src)}

    keep = _valid_rows(bbox, stats, crops)
    if not bool(keep.any()):
        return {"status": "no_valid_rows", "src": str(src), "rows_in": int(len(bbox))}

    fixed_bbox = _old_bbox_to_training_schema(bbox[keep])
    fixed_stats = stats[keep].astype(np.float32, copy=False)
    fixed_crops = crops[keep].astype(np.float32, copy=False)
    payload = {
        "bbox_feats": fixed_bbox,
        "height_stats": fixed_stats,
        "heightmap_crops": fixed_crops,
        "valid_count": np.asarray([int(fixed_bbox.shape[0])], dtype=np.int32),
        "camera_id": camera_id,
        "video_stem": video_stem,
        "source_row_indices": np.flatnonzero(keep).astype(np.int32),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)
    return {
        "status": "ok",
        "src": str(src),
        "dst": str(dst),
        "rows_in": int(len(bbox)),
        "rows_out": int(fixed_bbox.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack exact-normal legacy NPZs into compact valid-frame training NPZs.")
    parser.add_argument("--manifest", type=Path, default=Path("/home/zyding/data/test.csv"))
    parser.add_argument("--parsing-json-root", type=Path, default=Path("/data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/scanner/main_card_out"))
    parser.add_argument("--source-feature-root", type=Path, default=Path("/home/zyding/height/jianzhi_2511_sequence/features"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    results: list[dict] = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            person_id = _person_id(row)
            stem = _video_stem(row)
            if not person_id or not stem:
                continue
            exact_json = args.parsing_json_root / f"{person_id}_{stem}.json"
            if not exact_json.exists():
                results.append({"status": "missing_exact_parsing", "person_id": person_id, "video_stem": stem})
                continue
            src = args.source_feature_root / person_id / f"{stem}.npz"
            if not src.exists():
                results.append({"status": "missing_source_npz", "person_id": person_id, "video_stem": stem, "src": str(src)})
                continue
            dst = args.output_root / person_id / f"{stem}.npz"
            item = repack_one(src, dst)
            item.update({"person_id": person_id, "video_stem": stem})
            results.append(item)

    counts = Counter(str(item["status"]) for item in results)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "parsing_json_root": str(args.parsing_json_root),
        "source_feature_root": str(args.source_feature_root),
        "output_root": str(args.output_root),
        "status_counts": dict(sorted(counts.items())),
        "results": results,
    }
    summary_out = args.summary_out or (args.output_root / "repack_exact_normal_summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status_counts": summary["status_counts"], "summary": str(summary_out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
