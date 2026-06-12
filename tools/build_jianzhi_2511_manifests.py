from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2
import pandas as pd


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
LEGACY_CAMERA_RE = re.compile(r"(\d{3}cm_(?:inside|outside|slantside|side|front|back))", re.IGNORECASE)
PIXEL_CAMERA_RE = re.compile(r"((?:\d+d\d+)_(?:0|30|90|150|330))(?:_\d+w)?", re.IGNORECASE)


def _infer_person_id(video_path: Path, video_root: Path) -> str:
    rel = video_path.relative_to(video_root)
    return rel.parts[0] if len(rel.parts) > 1 else "unknown_person"


def _infer_camera_id(video_path: Path) -> str:
    text = str(video_path)
    match = PIXEL_CAMERA_RE.search(text)
    if match:
        return match.group(1).lower()
    match = LEGACY_CAMERA_RE.search(text)
    return match.group(1).lower() if match else "unknown_camera"


def _infer_camera_height_m(camera_id: str) -> float:
    pixel_match = re.search(r"(\d+)d(\d+)_", camera_id)
    if pixel_match:
        return float(f"{pixel_match.group(1)}.{pixel_match.group(2)}")
    match = re.search(r"(\d+)cm", camera_id)
    return float(match.group(1)) / 100.0 if match else 0.0


def _video_meta(video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if n_frames <= 0:
        raise RuntimeError(f"video has no frames: {video_path}")
    return n_frames, fps if fps > 0 else 0.0


def _resolve_bg_depth_path(bg_depth_root: Path, camera_id: str) -> str:
    candidates = [
        bg_depth_root / camera_id / f"{camera_id}_avg_depth.npy",
        bg_depth_root / f"{camera_id}_avg_depth.npy",
    ]
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    if bg_depth_root.exists():
        for path in bg_depth_root.rglob("*_avg_depth.npy"):
            name = path.name.lower()
            if name.startswith(camera_id.lower()):
                return str(path.resolve())
    return ""


def scan_videos(video_root: Path, bg_depth_root: Path, include_camera: str = "") -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    bad: list[dict] = []
    videos = sorted(p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    for video_path in videos:
        person_id = _infer_person_id(video_path, video_root)
        camera_id = _infer_camera_id(video_path)
        if include_camera and camera_id != include_camera:
            continue
        sequence_id = f"{person_id}__{video_path.stem}"
        try:
            n_frames, fps = _video_meta(video_path)
        except Exception as exc:
            bad.append({"video_path": str(video_path), "error": str(exc)})
            continue
        bg_depth_path = _resolve_bg_depth_path(bg_depth_root, camera_id)
        rows.append(
            {
                "video_path": str(video_path.resolve()),
                "frame_path": "",
                "sequence_id": sequence_id,
                "person_id": person_id,
                "camera_id": camera_id,
                "frame_start": 0,
                "frame_end": n_frames - 1,
                "fps": fps,
                "valid_frames_path": "",
                "height_cache_path": "",
                "valid_mask_cache_path": "",
                "depth_cache_path": "",
                "bg_depth_path": bg_depth_path,
                "camera_height_m": _infer_camera_height_m(camera_id),
            }
        )
    return rows, bad


def _load_person_hints(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return set()
    if p.is_dir():
        return {x.name for x in p.iterdir() if x.is_dir()}
    people: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        people.add(Path(token).parts[0])
    return people


def split_rows(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    forced_test_people: set[str] | None = None,
) -> dict[str, list[dict]]:
    by_person: dict[str, list[dict]] = {}
    for row in rows:
        by_person.setdefault(str(row["person_id"]), []).append(row)
    people = sorted(by_person)
    rng = random.Random(seed)
    rng.shuffle(people)
    forced_test_people = set(forced_test_people or set()).intersection(people)
    remaining = [pid for pid in people if pid not in forced_test_people]
    n = len(people)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_people = set(remaining[:n_train])
    val_people = set(remaining[n_train : n_train + n_val])
    test_people = set(remaining[n_train + n_val :]) | forced_test_people
    out = {"train": [], "val": [], "test": []}
    for row in rows:
        pid = str(row["person_id"])
        split = "train" if pid in train_people else "val" if pid in val_people else "test"
        out[split].append(row)
    return out


def expand_frame_rows(rows: list[dict], sample_fps: float, frames_root: Path) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        fps = float(row["fps"])
        start = int(row["frame_start"])
        end = int(row["frame_end"])
        stride = max(int(round(fps / sample_fps)), 1) if fps > 0 and sample_fps > 0 else 1
        for frame_idx in range(start, end + 1, stride):
            item = dict(row)
            item["source_frame_start"] = start
            item["source_frame_end"] = end
            item["frame_idx"] = int(frame_idx)
            item["frame_start"] = int(frame_idx)
            item["frame_end"] = int(frame_idx)
            item["frame_path"] = str((frames_root / item["person_id"] / f"{item['sequence_id']}__frame{frame_idx:06d}.jpg").resolve())
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build jianzhi_2511 video and sampled frame manifests.")
    parser.add_argument("--video-root", type=str, default="/data2/dataset/jianzhi_2511/video2")
    parser.add_argument("--bg-depth-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--include-camera", type=str, default="", help="Normalized camera id such as 2d5_0. Empty means all cameras.")
    parser.add_argument("--test-person-list", type=str, default="", help="File or directory whose people are forced into test split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    video_root = Path(args.video_root).expanduser().resolve()
    bg_depth_root = Path(args.bg_depth_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not video_root.exists():
        raise FileNotFoundError(f"video root not found: {video_root}")
    if not bg_depth_root.exists():
        print(f"[warn] bg depth root not found yet: {bg_depth_root}", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, bad = scan_videos(video_root, bg_depth_root, include_camera=str(args.include_camera).strip())
    forced_test_people = _load_person_hints(args.test_person_list)
    splits = split_rows(
        rows,
        float(args.train_ratio),
        float(args.val_ratio),
        int(args.seed),
        forced_test_people=forced_test_people,
    )
    frames_root = out_dir / "frames"

    summary = {
        "video_root": str(video_root),
        "include_camera": str(args.include_camera).strip(),
        "forced_test_people": sorted(forced_test_people),
        "num_videos": len(rows),
        "bad_videos": bad,
        "splits": {},
    }
    for split, split_rows_ in splits.items():
        video_manifest = out_dir / f"{split}_video_manifest.csv"
        frame_manifest = out_dir / f"{split}_manifest.csv"
        pd.DataFrame(split_rows_).to_csv(video_manifest, index=False)
        frame_rows = expand_frame_rows(split_rows_, float(args.sample_fps), frames_root)
        pd.DataFrame(frame_rows).to_csv(frame_manifest, index=False)
        summary["splits"][split] = {"videos": len(split_rows_), "frames": len(frame_rows)}
        print(f"[MANIFEST] {split}: videos={len(split_rows_)} frames={len(frame_rows)}")

    with (out_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
