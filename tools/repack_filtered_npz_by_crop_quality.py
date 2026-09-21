#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _rank_info() -> tuple[int, int]:
    import os

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _finite_nonzero_crops(crops: np.ndarray) -> np.ndarray:
    flat = crops.reshape(crops.shape[0], -1)
    return np.isfinite(flat).all(axis=1) & (np.abs(flat).sum(axis=1) > 0)


def _finite_nonzero_bbox(bbox: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(bbox).all(axis=1)
        & (bbox[:, 0] > 0.0)
        & (bbox[:, 1] > 0.0)
        & (bbox[:, 5] > 0.0)
    )


def _height_stats_l2_mean(height_stats: np.ndarray) -> float:
    if height_stats.ndim != 2 or height_stats.shape[0] <= 1:
        return 0.0
    stats = np.asarray(height_stats, dtype=np.float32)
    if not np.isfinite(stats).all():
        return float("inf")
    centered = stats - stats.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(centered, axis=1)))


def _build_frame_bbox_map(payload: Any) -> dict[int, dict[str, Any]]:
    frame_map: dict[int, dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return frame_map
    for obj in payload.values():
        if not isinstance(obj, dict):
            continue
        sub_tracks = obj.get("sub_track")
        if not isinstance(sub_tracks, list):
            continue
        for sub_track in sub_tracks:
            if not isinstance(sub_track, dict):
                continue
            data = sub_track.get("data")
            if not isinstance(data, dict):
                continue
            for track_items in data.values():
                if not isinstance(track_items, list):
                    continue
                for item in track_items:
                    if not isinstance(item, dict):
                        continue
                    frame_id = item.get("frame_id")
                    rect = item.get("rect")
                    if frame_id is None or not isinstance(rect, (list, tuple)) or len(rect) < 4:
                        continue
                    x, y, w, h = [float(v) for v in rect[:4]]
                    score = float(item.get("score", 0.0))
                    prev = frame_map.get(int(frame_id))
                    if prev is None or score >= float(prev.get("score", 0.0)):
                        frame_map[int(frame_id)] = {"bbox_xyxy": (x, y, x + w, y + h), "score": score}
    return frame_map


def _sample_source_rows_from_bbox(frame_bbox_map: dict[int, dict[str, Any]], frames_per_video: int = 20) -> list[dict[str, Any]]:
    keys = sorted(frame_bbox_map)
    if not keys:
        return []
    frames_per_video = max(1, min(int(frames_per_video), len(keys)))
    if frames_per_video == len(keys):
        return [frame_bbox_map[key] for key in keys]
    if frames_per_video == 1:
        return [frame_bbox_map[keys[len(keys) // 2]]]
    positions = np.linspace(0, len(keys) - 1, num=frames_per_video)
    sampled_keys = sorted(dict.fromkeys(keys[int(round(float(pos)))] for pos in positions))
    return [frame_bbox_map[key] for key in sampled_keys]


def _edge_margins_for_source_rows(src_path: Path, source_bbox: np.ndarray, args: argparse.Namespace) -> np.ndarray | None:
    if args.bbox_json_root is None:
        return None
    person_id = src_path.parent.name
    json_path = Path(args.bbox_json_root) / f"{person_id}_{src_path.stem}.json"
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    entries = _sample_source_rows_from_bbox(_build_frame_bbox_map(payload), frames_per_video=int(args.frames_per_video))
    if not entries:
        return None
    count = min(len(entries), len(source_bbox))
    margins = np.ones((len(source_bbox),), dtype=np.float32)
    for idx in range(count):
        x1, y1, x2, y2 = [float(v) for v in entries[idx]["bbox_xyxy"]]
        bw = max(x2 - x1, 1e-6)
        bh = max(y2 - y1, 1e-6)
        frame_w = bw / max(float(source_bbox[idx, 1]), 1e-6)
        frame_h = bh / max(float(source_bbox[idx, 0]), 1e-6)
        margins[idx] = float(min(x1 / frame_w, 1.0 - x2 / frame_w, y1 / frame_h, 1.0 - y2 / frame_h))
    return margins


def _source_keep_indices(src_path: Path, payload: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, int]]:
    if args.source_root is None:
        bbox = np.asarray(payload["bbox_feats"])
        crops = np.asarray(payload["heightmap_crops"])
        stored_count = len(bbox)
    else:
        src_root = Path(args.source_root).resolve()
        rel = src_path.resolve().relative_to(Path(args.src_root).resolve())
        source_path = src_root / rel
        with np.load(source_path, allow_pickle=False) as source_data:
            bbox = np.asarray(source_data["bbox_feats"])
            crops = np.asarray(source_data["heightmap_crops"])
        stored_count = len(bbox)

    frame_valid = _finite_nonzero_bbox(bbox) & _finite_nonzero_crops(crops)
    valid_indices = np.flatnonzero(frame_valid)
    if valid_indices.size <= 0:
        return valid_indices, {"finite_nonzero_valid": 0}

    edge_margins = _edge_margins_for_source_rows(src_path, bbox, args)
    keep = (
        frame_valid
        & (bbox[:, 0] >= float(args.min_bbox_h_norm))
        & (bbox[:, 1] >= float(args.min_bbox_w_norm))
        & (bbox[:, 6] >= float(args.min_bbox_score))
    )
    dropped_edge = 0
    if edge_margins is not None and float(args.min_bbox_edge_margin_norm) > 0.0:
        edge_keep = edge_margins >= float(args.min_bbox_edge_margin_norm)
        dropped_edge = int((keep & ~edge_keep).sum())
        keep &= edge_keep
    keep_indices = np.flatnonzero(keep)
    return keep_indices, {
        "stored_count": int(stored_count),
        "finite_nonzero_valid": int(valid_indices.size),
        "edge_margin_available": int(edge_margins is not None),
        "dropped_edge": dropped_edge,
    }


def _filter_one(src_path: Path, dst_path: Path, args: argparse.Namespace) -> dict:
    try:
        with np.load(src_path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
    except Exception as exc:
        return {"status": "read_error", "path": str(src_path), "error": repr(exc)}

    required = {"bbox_feats", "height_stats", "heightmap_crops"}
    if not required.issubset(payload):
        return {"status": "missing_keys", "path": str(src_path)}

    bbox = np.asarray(payload["bbox_feats"])
    height_stats = np.asarray(payload["height_stats"])
    crops = np.asarray(payload["heightmap_crops"])
    if (
        bbox.ndim != 2
        or bbox.shape[1] < 7
        or height_stats.ndim != 2
        or crops.ndim < 2
        or len(bbox) != len(crops)
        or len(bbox) != len(height_stats)
    ):
        return {"status": "bad_shape", "path": str(src_path)}

    stored_count = len(bbox)
    declared_valid_count = int(np.asarray(payload.get("valid_count", [stored_count])).reshape(-1)[0])
    if declared_valid_count <= 0:
        return {"status": "no_valid_rows", "path": str(src_path), "stored_count": stored_count}

    try:
        source_keep_indices, source_stats = _source_keep_indices(src_path, payload, args)
    except Exception as exc:
        return {"status": "source_map_error", "path": str(src_path), "error": repr(exc)}

    if source_keep_indices.size <= 0:
        return {
            "status": "no_valid_rows",
            "path": str(src_path),
            "stored_count": stored_count,
            "declared_valid_count": declared_valid_count,
            **source_stats,
        }

    if args.source_root is None:
        candidate_indices = source_keep_indices
    else:
        candidate_indices = source_keep_indices[source_keep_indices < stored_count]
    bbox_valid = bbox[candidate_indices]
    height_stats_valid = height_stats[candidate_indices]
    valid_count = int(len(candidate_indices))
    h_median = float(np.median(bbox_valid[:, 0]))
    w_median = float(np.median(bbox_valid[:, 1]))
    h_cv = float(np.std(bbox_valid[:, 0]) / max(h_median, 1e-6))
    w_cv = float(np.std(bbox_valid[:, 1]) / max(w_median, 1e-6))
    low_score_frac = float(np.mean(bbox_valid[:, 6] < float(args.low_score_threshold)))
    height_stats_l2_mean = _height_stats_l2_mean(height_stats_valid)
    if h_cv > float(args.max_bbox_h_cv) or w_cv > float(args.max_bbox_w_cv):
        return {
            "status": "unstable_bbox_scale",
            "path": str(src_path),
            "valid_count": valid_count,
            "declared_valid_count": declared_valid_count,
            "bbox_h_cv": h_cv,
            "bbox_w_cv": w_cv,
            "low_score_frac": low_score_frac,
            "height_stats_l2_mean": height_stats_l2_mean,
            "median_h": h_median,
            "median_w": w_median,
            **source_stats,
        }
    if low_score_frac > float(args.max_low_score_frac):
        return {
            "status": "too_many_low_score_frames",
            "path": str(src_path),
            "valid_count": valid_count,
            "declared_valid_count": declared_valid_count,
            "bbox_h_cv": h_cv,
            "bbox_w_cv": w_cv,
            "low_score_frac": low_score_frac,
            "height_stats_l2_mean": height_stats_l2_mean,
            "median_h": h_median,
            "median_w": w_median,
            **source_stats,
        }
    if height_stats_l2_mean > float(args.max_height_stats_l2_mean):
        return {
            "status": "unstable_height_stats",
            "path": str(src_path),
            "valid_count": valid_count,
            "declared_valid_count": declared_valid_count,
            "bbox_h_cv": h_cv,
            "bbox_w_cv": w_cv,
            "low_score_frac": low_score_frac,
            "height_stats_l2_mean": height_stats_l2_mean,
            "median_h": h_median,
            "median_w": w_median,
            **source_stats,
        }

    kept = valid_count
    if kept < int(args.min_valid_count):
        return {
            "status": "filtered_video",
            "path": str(src_path),
            "valid_count": valid_count,
            "declared_valid_count": declared_valid_count,
            "kept_count": kept,
            "median_h": float(np.median(bbox_valid[:, 0])),
            "median_score": float(np.median(bbox_valid[:, 6])),
            "bbox_h_cv": h_cv,
            "bbox_w_cv": w_cv,
            "low_score_frac": low_score_frac,
            "height_stats_l2_mean": height_stats_l2_mean,
            **source_stats,
        }

    out_payload = {}
    kept_indices = candidate_indices
    for key, value in payload.items():
        arr = np.asarray(value)
        if key == "valid_count":
            out_payload[key] = np.asarray([kept], dtype=np.int32)
        elif arr.shape[:1] == (stored_count,):
            out_payload[key] = arr[kept_indices]
        else:
            out_payload[key] = arr

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **out_payload)
    return {
        "status": "ok",
        "path": str(src_path),
        "out_path": str(dst_path),
        "valid_count": valid_count,
        "declared_valid_count": declared_valid_count,
        "kept_count": kept,
        "median_h": float(np.median(bbox_valid[:, 0])),
        "median_score": float(np.median(bbox_valid[:, 6])),
        "bbox_h_cv": h_cv,
        "bbox_w_cv": w_cv,
        "low_score_frac": low_score_frac,
        "height_stats_l2_mean": height_stats_l2_mean,
        **source_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack NPZ features after frame/video crop-quality filtering.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--bbox-json-root", type=Path, default=None)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=20)
    parser.add_argument("--min-valid-count", type=int, default=8)
    parser.add_argument("--min-bbox-h-norm", type=float, default=0.04)
    parser.add_argument("--min-bbox-w-norm", type=float, default=0.01)
    parser.add_argument("--min-bbox-score", type=float, default=0.35)
    parser.add_argument("--min-bbox-edge-margin-norm", type=float, default=0.0)
    parser.add_argument("--max-bbox-h-cv", type=float, default=999.0)
    parser.add_argument("--max-bbox-w-cv", type=float, default=999.0)
    parser.add_argument("--low-score-threshold", type=float, default=0.4)
    parser.add_argument("--max-low-score-frac", type=float, default=1.0)
    parser.add_argument("--max-height-stats-l2-mean", type=float, default=999.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rank, world_size = _rank_info()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if not src_root.exists():
        raise FileNotFoundError(src_root)
    if dst_root.exists() and rank == 0 and args.overwrite:
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(src_root.glob("**/*.npz"))
    selected = [path for idx, path in enumerate(npz_paths) if idx % world_size == rank]
    records = []
    for src_path in selected:
        rel = src_path.relative_to(src_root)
        records.append(_filter_one(src_path, dst_root / rel, args))

    report_dir = dst_root / "_filter_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(record["status"] for record in records)
    summary = {
        "rank": rank,
        "world_size": world_size,
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "selected": len(selected),
        "status_counts": dict(status_counts),
        "args": {
            "min_valid_count": args.min_valid_count,
            "min_bbox_h_norm": args.min_bbox_h_norm,
            "min_bbox_w_norm": args.min_bbox_w_norm,
            "min_bbox_score": args.min_bbox_score,
            "min_bbox_edge_margin_norm": args.min_bbox_edge_margin_norm,
            "max_bbox_h_cv": args.max_bbox_h_cv,
            "max_bbox_w_cv": args.max_bbox_w_cv,
            "low_score_threshold": args.low_score_threshold,
            "max_low_score_frac": args.max_low_score_frac,
            "max_height_stats_l2_mean": args.max_height_stats_l2_mean,
            "source_root": str(args.source_root) if args.source_root else None,
            "bbox_json_root": str(args.bbox_json_root) if args.bbox_json_root else None,
        },
    }
    (report_dir / f"summary_rank{rank}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / f"records_rank{rank}.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
