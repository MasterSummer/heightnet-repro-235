from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]
CAMERA_RE = re.compile(r"((?:\d+d\d+)_(?:0|30|90|150|330))(?:_\d+w)?", re.IGNORECASE)
PERSON_RE = re.compile(r"(\d{4}_(?:man|woman)\d+)", re.IGNORECASE)


def _infer_person_id(stem: str) -> str | None:
    match = PERSON_RE.search(stem)
    return match.group(1) if match else None


def _normalize_camera(stem: str) -> str:
    match = CAMERA_RE.search(stem)
    return match.group(1).lower() if match else "unknown_camera"


def _camera_height_m(camera_id: str) -> float:
    match = re.search(r"(\d+)d(\d+)_", camera_id)
    return float(f"{match.group(1)}.{match.group(2)}") if match else 0.0


def _video_stem_from_parsing_dir(dirname: str, person_id: str) -> str:
    prefix = person_id + "_"
    return dirname[len(prefix) :] if dirname.startswith(prefix) else dirname


def _find_video(video_root: Path, person_id: str, video_stem: str) -> Path | None:
    base = video_root / person_id
    for ext in VIDEO_EXTS:
        candidate = base / f"{video_stem}{ext}"
        if candidate.exists():
            return candidate.resolve()
    matches = sorted(base.rglob(f"{video_stem}.*")) if base.exists() else []
    for path in matches:
        if path.suffix.lower() in VIDEO_EXTS:
            return path.resolve()
    return None


def _frame_indices_from_parsing_dir(path: Path, max_frames: int, seed: int) -> list[int]:
    indices = sorted({int(p.stem) for p in path.glob("*/*.bmp") if p.stem.isdigit()})
    if max_frames > 0 and len(indices) > max_frames:
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_frames))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frame manifest from existing jianzhi parsing bmp positions.")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--parsing-root", required=True)
    parser.add_argument("--bg-depth-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--camera", default="2d5_0")
    parser.add_argument("--person-csv", default="", help="Optional CSV with person ids in the first column, e.g. rank.json from jianzhi split.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames-per-sequence", type=int, default=20)
    args = parser.parse_args()

    video_root = Path(args.video_root).expanduser().resolve()
    parsing_root = Path(args.parsing_root).expanduser().resolve()
    bg_depth_root = Path(args.bg_depth_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed_people: set[str] | None = None
    if args.person_csv:
        person_frame = pd.read_csv(Path(args.person_csv).expanduser().resolve())
        allowed_people = {str(x) for x in person_frame.iloc[:, 0].dropna().tolist()}

    video_rows = []
    frame_rows = []
    missing_videos = []
    for parsing_dir in sorted(p for p in parsing_root.iterdir() if p.is_dir()):
        person_id = _infer_person_id(parsing_dir.name)
        if not person_id:
            continue
        if allowed_people is not None and person_id not in allowed_people:
            continue
        video_stem = _video_stem_from_parsing_dir(parsing_dir.name, person_id)
        camera_id = _normalize_camera(video_stem)
        if camera_id != args.camera:
            continue
        video_path = _find_video(video_root, person_id, video_stem)
        if video_path is None:
            missing_videos.append({"person_id": person_id, "video_stem": video_stem})
            continue
        frame_indices = _frame_indices_from_parsing_dir(parsing_dir, args.max_frames_per_sequence, args.seed)
        if not frame_indices:
            continue
        sequence_id = f"{person_id}__{video_stem}"
        row = {
            "video_path": str(video_path),
            "frame_path": "",
            "sequence_id": sequence_id,
            "person_id": person_id,
            "camera_id": camera_id,
            "frame_start": int(min(frame_indices)),
            "frame_end": int(max(frame_indices)),
            "fps": 25.0,
            "valid_frames_path": "",
            "height_cache_path": "",
            "valid_mask_cache_path": "",
            "depth_cache_path": "",
            "bg_depth_path": str((bg_depth_root / camera_id / f"{camera_id}_avg_depth.npy").resolve()),
            "camera_height_m": _camera_height_m(camera_id),
        }
        video_rows.append(row)
        for frame_idx in frame_indices:
            item = dict(row)
            item["frame_idx"] = int(frame_idx)
            item["frame_start"] = int(frame_idx)
            item["frame_end"] = int(frame_idx)
            item["frame_path"] = str((out_dir / "frames" / person_id / f"{sequence_id}__frame{frame_idx:06d}.jpg").resolve())
            frame_rows.append(item)

    people = sorted({r["person_id"] for r in video_rows})
    rng = random.Random(args.seed)
    rng.shuffle(people)
    n_train = int(round(len(people) * args.train_ratio))
    train_people = set(people[:n_train])
    split_rows = {"train": [], "test": [], "val": []}
    for row in frame_rows:
        split = "train" if row["person_id"] in train_people else "test"
        split_rows[split].append(row)

    summary = {
        "camera": args.camera,
        "num_people": len(people),
        "person_csv": str(Path(args.person_csv).expanduser().resolve()) if args.person_csv else "",
        "train_people": sorted(train_people),
        "test_people": sorted(set(people) - train_people),
        "num_sequences": len(video_rows),
        "num_frames": len(frame_rows),
        "missing_videos": missing_videos[:50],
        "splits": {},
    }
    for split, rows in split_rows.items():
        pd.DataFrame(rows).to_csv(out_dir / f"{split}_manifest.csv", index=False)
        summary["splits"][split] = {
            "frames": len(rows),
            "sequences": len({r["sequence_id"] for r in rows}),
            "people": len({r["person_id"] for r in rows}),
        }
    with (out_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
