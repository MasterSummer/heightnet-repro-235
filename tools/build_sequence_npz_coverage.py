#!/usr/bin/env python3
# pyright: basic
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CAMERA_RE = re.compile(r"_(?P<h>\d+d\d+)_(?P<a>\d+)_", re.IGNORECASE)


def infer_camera_id(text: str) -> str:
    match = CAMERA_RE.search(str(text))
    if not match:
        return "unknown_camera"
    return f"{match.group('h').lower()}_{match.group('a')}"


def load_height_labels(path: Path) -> dict[str, float]:
    labels: dict[str, float] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person_id = (row.get("penson_id") or row.get("person_id") or "").strip()
            raw_height = (row.get("height(cm)") or row.get("height_cm") or row.get("height") or "").strip()
            if not person_id or not raw_height:
                continue
            try:
                labels[person_id] = float(raw_height)
            except ValueError:
                continue
    return labels


def read_csv_records(path: Path, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            person_id = (row.get("person_id") or row.get("penson_id") or "").strip()
            video_filename = (row.get("video_filename") or "").strip()
            video_path = (row.get("video_path") or "").strip()
            split = (row.get("split") or source).strip() or source
            if not video_filename and video_path:
                video_filename = Path(video_path).name
            video_stem = Path(video_filename).stem if video_filename else Path(video_path).stem
            records.append(
                {
                    "source": source,
                    "split": split,
                    "person_id": person_id,
                    "group": None,
                    "video_path": video_path,
                    "video_filename": video_filename,
                    "video_stem": video_stem,
                    "row_idx": row_idx,
                }
            )
    return records


def discover_softlink_records(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    videos = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    for video in sorted(videos):
        rel = video.relative_to(root)
        person_id = rel.parts[0] if len(rel.parts) >= 1 else ""
        group = rel.parts[1] if len(rel.parts) >= 3 else (rel.parent.name if rel.parent != Path(".") else "")
        records.append(
            {
                "source": "softlink",
                "split": None,
                "person_id": person_id,
                "group": group,
                "video_path": str(video),
                "video_filename": video.name,
                "video_stem": video.stem,
                "row_idx": None,
            }
        )
    return records


def expected_npz_path(feature_root: Path, person_id: str, video_stem: str) -> Path:
    return feature_root / person_id / f"{video_stem}.npz"


def build_rect_index(rect_root: Path | None) -> dict[str, list[Path]]:
    if rect_root is None or not rect_root.exists():
        return {}
    index: dict[str, list[Path]] = {}
    for path in sorted(rect_root.glob("*.json")):
        parts = path.stem.split("_")
        person_id = "_".join(parts[:2]) if len(parts) >= 2 else path.stem
        index.setdefault(person_id, []).append(path)
    return index


def rect_json_path(
    rect_root: Path | None,
    rect_index: dict[str, list[Path]],
    person_id: str,
    video_stem: str,
    camera_id: str,
) -> tuple[str | None, str]:
    if rect_root is None:
        return None, "rect_root_not_provided"
    exact = rect_root / f"{person_id}_{video_stem}.json"
    if exact.exists():
        return str(exact), "exact"
    if rect_root.exists():
        candidates = []
        if camera_id != "unknown_camera":
            camera_token = f"_{camera_id}_"
            candidates = [p for p in rect_index.get(person_id, []) if camera_token in p.stem]
        if candidates:
            return str(candidates[0]), "camera_glob"
    return str(exact), "expected_missing"


def assess_status(video_path: str, npz_path: Path, rect_mode: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not video_path:
        reasons.append("missing_video_path")
    elif not Path(video_path).exists():
        reasons.append("missing_video_file")
    if not npz_path.exists():
        reasons.append("missing_expected_npz")
    if rect_mode in {"rect_root_not_provided", "expected_missing"}:
        reasons.append("missing_rect_json" if rect_mode == "expected_missing" else "rect_root_not_provided")

    fatal = {"missing_video_path", "missing_video_file", "missing_expected_npz"}
    if any(r in fatal for r in reasons):
        return "skipped", reasons
    if reasons:
        return "partial", reasons
    return "scored", []


def make_candidate(
    record: dict[str, Any],
    labels: dict[str, float],
    feature_root: Path,
    rect_root: Path | None,
    rect_index: dict[str, list[Path]],
) -> dict[str, Any]:
    person_id = record["person_id"]
    video_stem = record["video_stem"]
    camera_id = infer_camera_id(video_stem)
    npz_path = expected_npz_path(feature_root, person_id, video_stem)
    rect_path, rect_mode = rect_json_path(rect_root, rect_index, person_id, video_stem, camera_id)
    status, reasons = assess_status(record["video_path"], npz_path, rect_mode)
    has_label = person_id in labels
    if record["source"] == "softlink":
        candidate_id = f"softlink:{person_id}:{record.get('group') or ''}:{video_stem}"
    else:
        candidate_id = f"csv:{record['source']}:{person_id}:{video_stem}"
    return {
        "candidate_id": candidate_id,
        "source": record["source"],
        "split": record.get("split"),
        "person_id": person_id,
        "group": record.get("group"),
        "video_path": record["video_path"],
        "video_filename": record["video_filename"],
        "video_stem": video_stem,
        "expected_npz_path": str(npz_path),
        "rect_json_path": rect_path,
        "rect_json_heuristic": rect_mode,
        "camera_id": camera_id,
        "has_height_label": has_label,
        "height_cm": labels.get(person_id) if has_label else None,
        "status": status,
        "reasons": reasons,
    }


def build_manifest(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    rank_json: Path,
    softlink_root: Path,
    feature_root: Path,
    rect_root: Path | None,
    out_path: Path,
) -> dict[str, Any]:
    labels = load_height_labels(rank_json)
    labeled_records = []
    for source, path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        labeled_records.extend(read_csv_records(path, source))
    softlink_records = discover_softlink_records(softlink_root)
    rect_index = build_rect_index(rect_root)
    candidates = [make_candidate(r, labels, feature_root, rect_root, rect_index) for r in labeled_records + softlink_records]

    status_counts = Counter(c["status"] for c in candidates)
    camera_counts = Counter(c["camera_id"] for c in candidates)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_labeled_candidates": len(labeled_records),
        "total_softlink_candidates": len(softlink_records),
        "total_candidates": len(candidates),
        "status_counts": dict(sorted(status_counts.items())),
        "unlabeled_count": sum(1 for c in candidates if not c["has_height_label"]),
        "camera_counts": dict(sorted(camera_counts.items())),
        "output_path": str(out_path),
        "source_paths": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "rank_json": str(rank_json),
            "softlink_root": str(softlink_root),
            "feature_root": str(feature_root),
            "rect_root": str(rect_root) if rect_root is not None else None,
        },
    }
    manifest = {"summary": summary, "candidates": candidates}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build labeled/unlabeled sequence NPZ coverage manifest.")
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--val-csv", required=True, type=Path)
    p.add_argument("--test-csv", required=True, type=Path)
    p.add_argument("--rank-json", required=True, type=Path)
    p.add_argument("--softlink-root", required=True, type=Path)
    p.add_argument("--feature-root", required=True, type=Path)
    p.add_argument("--rect-root", default=None, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        rank_json=args.rank_json,
        softlink_root=args.softlink_root,
        feature_root=args.feature_root,
        rect_root=args.rect_root,
        out_path=args.out,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
