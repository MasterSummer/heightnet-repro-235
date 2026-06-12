#!/usr/bin/env python3
# pyright: basic
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = ("bbox_feats", "height_stats", "heightmap_crops")
EXPECTED_CROP_SHAPE = (1, 128, 64)
P90_INDEX = 2
MAX_INDEX = 4


@dataclass(frozen=True)
class FileValidationResult:
    path: str
    valid: bool
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate recursive sequence NPZ feature schema.")
    p.add_argument("--feature-root", required=True, type=str, help="Root containing nested sequence NPZ files.")
    p.add_argument("--out", required=True, type=str, help="Path to write JSON validation summary.")
    return p.parse_args()


def _shape_text(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def validate_npz_file(path: Path) -> FileValidationResult:
    reasons: list[str] = []
    arrays: dict[str, np.ndarray] = {}

    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            for key in REQUIRED_KEYS:
                if key not in keys:
                    reasons.append(f"missing required key {key}")
                else:
                    arrays[key] = data[key]
    except Exception as exc:
        return FileValidationResult(path=str(path), valid=False, reasons=[f"failed to load npz: {type(exc).__name__}: {exc}"])

    bbox = arrays.get("bbox_feats")
    stats = arrays.get("height_stats")
    crops = arrays.get("heightmap_crops")

    counts: dict[str, int] = {}
    if bbox is not None:
        if bbox.ndim != 2:
            reasons.append(f"bbox_feats must be 2D with 7 columns, got shape {_shape_text(bbox.shape)}")
        else:
            counts["bbox_feats"] = int(bbox.shape[0])
            if bbox.shape[1] != 7:
                reasons.append(f"bbox_feats.shape[1] must be 7, got {bbox.shape[1]}")

    if stats is not None:
        if stats.ndim != 2:
            reasons.append(f"height_stats must be 2D with at least 5 columns, got shape {_shape_text(stats.shape)}")
        else:
            counts["height_stats"] = int(stats.shape[0])
            if stats.shape[1] < 5:
                reasons.append(f"height_stats.shape[1] must be >= 5, got {stats.shape[1]}")
            else:
                p90 = stats[:, P90_INDEX]
                max_values = stats[:, MAX_INDEX]
                p90_finite = np.isfinite(p90)
                max_finite = np.isfinite(max_values)
                if not bool(np.all(p90_finite)):
                    bad = np.flatnonzero(~p90_finite)[:5].tolist()
                    reasons.append(f"p90 height_stats[:,{P90_INDEX}] must be finite for all rows, bad rows {bad}")
                if not bool(np.all(max_finite)):
                    bad = np.flatnonzero(~max_finite)[:5].tolist()
                    reasons.append(f"max height_stats[:,{MAX_INDEX}] must be finite for all rows, bad rows {bad}")
                both = p90_finite & max_finite
                if bool(np.any(max_values[both] < p90[both])):
                    bad = np.flatnonzero(both & (max_values < p90))[:5].tolist()
                    reasons.append(f"max height_stats[:,{MAX_INDEX}] must be >= p90 height_stats[:,{P90_INDEX}], bad rows {bad}")

    if crops is not None:
        if crops.ndim != 4:
            reasons.append(f"heightmap_crops must be 4D with shape (N, 1, 128, 64), got shape {_shape_text(crops.shape)}")
        else:
            counts["heightmap_crops"] = int(crops.shape[0])
            if tuple(crops.shape[1:]) != EXPECTED_CROP_SHAPE:
                reasons.append(
                    "heightmap_crops must have shape (N, 1, 128, 64), "
                    f"got {_shape_text(crops.shape)}"
                )

    if len(set(counts.values())) > 1:
        detail = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        reasons.append(f"frame count mismatch across arrays: {detail}")

    return FileValidationResult(path=str(path), valid=not reasons, reasons=reasons)


def validate_feature_root(feature_root: Path) -> list[FileValidationResult]:
    return [validate_npz_file(path) for path in sorted(feature_root.rglob("*.npz"))]


def build_summary(feature_root: Path, results: list[FileValidationResult]) -> dict[str, Any]:
    invalid = [result for result in results if not result.valid]
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "feature_root": str(feature_root),
        "expected_crop_shape": [None, *EXPECTED_CROP_SHAPE],
        "p90_index": P90_INDEX,
        "max_index": MAX_INDEX,
        "required_keys": list(REQUIRED_KEYS),
        "total_files": len(results),
        "valid_files": len(results) - len(invalid),
        "invalid_files": len(invalid),
        "invalid_file_list": [
            {"path": result.path, "reasons": result.reasons}
            for result in invalid
        ],
    }


def main() -> int:
    args = parse_args()
    feature_root = Path(args.feature_root)
    out_path = Path(args.out)

    results = validate_feature_root(feature_root)
    summary = build_summary(feature_root, results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        "[schema] "
        f"total={summary['total_files']} valid={summary['valid_files']} invalid={summary['invalid_files']} "
        f"out={out_path}",
        flush=True,
    )
    if summary["invalid_files"]:
        for item in summary["invalid_file_list"][:10]:
            print(f"[invalid] {item['path']}: {'; '.join(item['reasons'])}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
