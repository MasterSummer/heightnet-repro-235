#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cross_camera_heightmap_fusion_core import (
    camera_height_m,
    load_height_labels,
    load_npz_frames,
    load_split_rows,
)


def _gpu_inventory() -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]
    except Exception as exc:
        return [f"nvidia-smi failed: {exc}"]


def _summarize_split(rows: list[dict]) -> dict:
    return {
        "videos": len(rows),
        "people": len({row["person_id"] for row in rows}),
        "cameras": sorted({row["camera_id"] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/zyding/data"))
    parser.add_argument("--feature-root", type=Path, default=Path("/home/zyding/height/jianzhi_2511_sequence/features"))
    parser.add_argument("--coverage-manifest", type=Path, default=Path("runs/home_data_heightmap_fusion/coverage_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/cross_camera_heightmap_fusion/input_audit.json"))
    parser.add_argument("--max-npz", type=int, default=0, help="0 scans every referenced scored NPZ")
    args = parser.parse_args()

    labels = load_height_labels(args.data_root / "label" / "rank.json")
    by_split = {
        split: load_split_rows(args.data_root / f"{split}.csv", args.feature_root, split)
        for split in ("train", "val", "test")
    }
    people_by_split = {split: {row["person_id"] for row in rows} for split, rows in by_split.items()}
    overlaps = {
        "train_val": sorted(people_by_split["train"] & people_by_split["val"]),
        "train_test": sorted(people_by_split["train"] & people_by_split["test"]),
        "val_test": sorted(people_by_split["val"] & people_by_split["test"]),
    }
    train_labeled = [row for row in by_split["train"] if row["person_id"] in labels]
    cameras_by_person: dict[str, set[str]] = defaultdict(set)
    for row in train_labeled:
        cameras_by_person[row["person_id"]].add(row["camera_id"])
    eligible_people = sorted(pid for pid, cameras in cameras_by_person.items() if len(cameras) >= 2)

    coverage = json.loads(args.coverage_manifest.read_text(encoding="utf-8"))
    candidates = coverage["candidates"]
    npz_paths: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.get("status") != "scored":
            continue
        path = str(candidate.get("expected_npz_path") or "")
        if path and path not in seen:
            seen.add(path)
            npz_paths.append(Path(path))
    if args.max_npz > 0:
        npz_paths = npz_paths[: args.max_npz]

    invalid: list[dict] = []
    excluded_zero_valid_count: list[dict] = []
    camera_counts: Counter[str] = Counter()
    for idx, path in enumerate(npz_paths, start=1):
        try:
            frames = load_npz_frames(path)
            camera_counts[frames.camera_id] += 1
            camera_height_m(frames.camera_id)
        except Exception as exc:
            item = {"path": str(path), "error": str(exc)}
            if "invalid valid_count=0" in str(exc):
                excluded_zero_valid_count.append(item)
            else:
                invalid.append(item)
        if idx % 2000 == 0:
            print(f"[AUDIT] scanned={idx}/{len(npz_paths)} invalid={len(invalid)}", flush=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu_inventory": _gpu_inventory(),
        "paths": {
            "data_root": str(args.data_root),
            "feature_root": str(args.feature_root),
            "coverage_manifest": str(args.coverage_manifest),
        },
        "labels": {"people": len(labels)},
        "splits": {split: _summarize_split(rows) for split, rows in by_split.items()},
        "split_overlap": {key: {"count": len(values), "people": values} for key, values in overlaps.items()},
        "cross_camera_consistency": {
            "train_labeled_videos": len(train_labeled),
            "eligible_people_count": len(eligible_people),
            "eligible_people": eligible_people,
        },
        "coverage_summary": coverage.get("summary", {}),
        "npz_scan": {
            "requested": len(npz_paths),
            "invalid_count": len(invalid),
            "invalid": invalid,
            "excluded_zero_valid_count_count": len(excluded_zero_valid_count),
            "excluded_zero_valid_count": excluded_zero_valid_count,
            "camera_counts": dict(sorted(camera_counts.items())),
        },
        "ok": not invalid and bool(eligible_people),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("labels", "splits", "cross_camera_consistency", "npz_scan", "ok")}, ensure_ascii=False, indent=2))
    print("[OUT]", args.out)
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
