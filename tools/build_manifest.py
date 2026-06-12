from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd
import yaml

SPLITS = ("train", "val", "test")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def infer_camera_id_from_name(name: str) -> str:
    m = re.search(r"\d+cm_(?:inside|outside|slantside|side|front|back)", name, flags=re.IGNORECASE)
    return m.group(0).lower() if m else "unknown_camera"


def infer_camera_height_m(camera_id: str) -> float:
    m = re.search(r"(\d+)cm_", camera_id, flags=re.IGNORECASE)
    return float(m.group(1)) / 100.0 if m else -1.0


def video_name_is_excluded(name: str, patterns: List[str]) -> bool:
    lowered = name.lower()
    return any(p.strip().lower() in lowered for p in patterns if p.strip())


def video_meta(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if n_frames <= 0:
        raise RuntimeError(f"video has no frames: {path}")
    if fps <= 0:
        fps = 0.0
    return n_frames, fps


def split_person_ids(person_ids: List[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
    items = sorted(set(person_ids))
    if len(items) < 3:
        raise ValueError("need at least 3 unique person_ids to build train/val/test without leakage")

    rnd = random.Random(seed)
    rnd.shuffle(items)

    n_total = len(items)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    while n_test < 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        n_test = n_total - n_train - n_val

    return {
        "train": sorted(items[:n_train]),
        "val": sorted(items[n_train : n_train + n_val]),
        "test": sorted(items[n_train + n_val :]),
    }


def _bg_depth_path(bg_depth_root: str, camera_id: str) -> str:
    if not bg_depth_root or camera_id == "unknown_camera":
        return ""
    path = Path(bg_depth_root) / camera_id / f"{camera_id}_avg_depth.npy"
    return str(path.resolve()) if path.exists() else ""


def _cache_path(cache_root: Path, sequence_id: str, suffixes: List[str]) -> str:
    for suffix in suffixes:
        path = cache_root / f"{sequence_id}{suffix}"
        if path.exists():
            return str(path.resolve())
    return ""


def collect_rows(
    video_root: Path,
    bg_depth_root: str,
    allow_online_depth_supervision: bool,
    exclude_name_substrings: List[str] | None = None,
) -> tuple[List[dict], List[dict], List[dict]]:
    rows: List[dict] = []
    bad_videos: List[dict] = []
    missing_supervision: List[dict] = []
    excluded_patterns = list(exclude_name_substrings or [])

    for person_dir in sorted(video_root.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        for video_path in sorted(person_dir.iterdir()):
            if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTS:
                continue
            if video_name_is_excluded(video_path.name, excluded_patterns):
                continue
            sequence_id = f"{person_id}__{video_path.stem}"
            camera_id = infer_camera_id_from_name(video_path.name)
            try:
                frame_count, fps = video_meta(video_path)
            except Exception as exc:
                bad_videos.append(
                    {
                        "video_path": str(video_path.resolve()),
                        "person_id": person_id,
                        "sequence_id": sequence_id,
                        "error": str(exc),
                    }
                )
                continue

            height_cache = _cache_path(video_root / "height_cache", sequence_id, [".npz", ".npy"])
            valid_cache = _cache_path(video_root / "valid_mask_cache", sequence_id, [".npz", ".npy"])
            depth_cache = _cache_path(video_root / "depth_cache", sequence_id, [".npz", ".npy"])
            bg_depth_path = _bg_depth_path(bg_depth_root, camera_id)

            row = {
                "video_path": str(video_path.resolve()),
                "sequence_id": sequence_id,
                "person_id": person_id,
                "camera_id": camera_id,
                "frame_start": 0,
                "frame_end": frame_count - 1,
                "fps": fps,
                "valid_frames_path": "",
                "height_cache_path": height_cache,
                "valid_mask_cache_path": valid_cache,
                "depth_cache_path": depth_cache,
                "bg_depth_path": bg_depth_path,
                "camera_height_m": infer_camera_height_m(camera_id),
            }

            has_height_cache = bool(height_cache and valid_cache)
            has_depth_route = bool(depth_cache and bg_depth_path)
            has_online_depth_route = bool(allow_online_depth_supervision and bg_depth_path)
            if not (has_height_cache or has_depth_route or has_online_depth_route):
                missing_supervision.append(
                    {
                        "video_path": row["video_path"],
                        "person_id": person_id,
                        "sequence_id": sequence_id,
                        "camera_id": camera_id,
                        "reason": "missing supervision caches and bg depth route",
                    }
                )
                continue
            rows.append(row)

    return rows, bad_videos, missing_supervision


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_txt(path: Path, rows: List[dict], key: str = "video_path") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            if key in item:
                f.write(str(item[key]) + "\n")


def _check_no_leakage(split_people: Dict[str, List[str]]) -> None:
    train = set(split_people["train"])
    val = set(split_people["val"])
    test = set(split_people["test"])
    if train & val or train & test or val & test:
        raise RuntimeError("person-level leakage detected across train/val/test")


def _summary_for_split(rows: List[dict]) -> dict:
    cameras = Counter(row["camera_id"] for row in rows)
    people = sorted({row["person_id"] for row in rows})
    return {
        "num_videos": len(rows),
        "num_people": len(people),
        "people": people,
        "camera_distribution": dict(sorted(cameras.items())),
    }


def _sanitize_camera_id(camera_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", camera_id.strip())


def build_camera_splits(
    rows: List[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, List[dict]]]:
    rows_by_camera: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        rows_by_camera[row["camera_id"]].append(row)

    out: Dict[str, Dict[str, List[dict]]] = {}
    for camera_id, camera_rows in sorted(rows_by_camera.items()):
        people = sorted({row["person_id"] for row in camera_rows})
        split_people = split_person_ids(people, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        _check_no_leakage(split_people)
        rows_by_person: Dict[str, List[dict]] = defaultdict(list)
        for row in camera_rows:
            rows_by_person[row["person_id"]].append(row)
        out[camera_id] = {}
        for split in SPLITS:
            out[camera_id][split] = sorted(
                [row for person_id in split_people[split] for row in rows_by_person[person_id]],
                key=lambda x: (x["person_id"], x["sequence_id"]),
            )
    return out


def write_camera_configs(
    camera_ids: List[str],
    out_root: Path,
    template_config: Path,
    config_out_dir: Path,
    runs_root: str,
) -> Dict[str, str]:
    with template_config.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config_out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for camera_id in camera_ids:
        camera_dir = out_root / camera_id
        cfg = json.loads(json.dumps(raw))
        cfg["paths"]["train_manifest"] = str((camera_dir / "train_manifest.csv").resolve())
        cfg["paths"]["val_manifest"] = str((camera_dir / "val_manifest.csv").resolve())
        cfg["paths"]["test_manifest"] = str((camera_dir / "test_manifest.csv").resolve())
        cfg["paths"]["train_video_root"] = ""
        cfg["paths"]["val_video_root"] = ""
        cfg["paths"]["test_video_root"] = ""
        cfg["paths"]["output_dir"] = f"{runs_root.rstrip('/')}/{camera_id}"

        config_path = config_out_dir / f"{_sanitize_camera_id(camera_id)}.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written[camera_id] = str(config_path.resolve())
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test manifests from raw videos with person-level split.")
    parser.add_argument("--video-root", type=str, required=True, help="Root dir with layout <video_root>/<person_id>/*.mp4")
    parser.add_argument("--out-dir", type=str, required=True, help="Output dir for train/val/test manifest CSV files")
    parser.add_argument("--bg-depth-root", type=str, default="", help="Optional camera-wise background depth root")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-online-depth-supervision", action="store_true")
    parser.add_argument(
        "--exclude-name-substrings",
        nargs="*",
        default=[],
        help="Skip videos whose filename contains any of these case-insensitive substrings.",
    )
    parser.add_argument("--split-by-camera", action="store_true", help="Build an independent train/val/test manifest set for each camera_id.")
    parser.add_argument("--template-config", type=str, default="", help="Optional template YAML used to generate one config per camera.")
    parser.add_argument("--config-out-dir", type=str, default="", help="Directory for generated per-camera configs.")
    parser.add_argument("--runs-root", type=str, default="", help="Output root prefix used inside generated per-camera configs.")
    args = parser.parse_args()

    video_root = Path(args.video_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not video_root.exists():
        raise FileNotFoundError(f"video_root not found: {video_root}")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1")

    rows, bad_videos, missing_supervision = collect_rows(
        video_root=video_root,
        bg_depth_root=args.bg_depth_root,
        allow_online_depth_supervision=args.allow_online_depth_supervision,
        exclude_name_substrings=args.exclude_name_substrings,
    )
    if not rows:
        raise RuntimeError("no valid videos found after filtering bad videos and missing supervision")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_by_camera:
        camera_splits = build_camera_splits(rows, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
        summary = {
            "video_root": str(video_root),
            "out_dir": str(out_dir),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "allow_online_depth_supervision": bool(args.allow_online_depth_supervision),
            "exclude_name_substrings": list(args.exclude_name_substrings),
            "split_by_camera": True,
            "total_valid_videos": len(rows),
            "bad_videos": len(bad_videos),
            "missing_supervision": len(missing_supervision),
            "cameras": {},
        }
        for camera_id, split_rows in camera_splits.items():
            camera_dir = out_dir / _sanitize_camera_id(camera_id)
            camera_dir.mkdir(parents=True, exist_ok=True)
            for split in SPLITS:
                pd.DataFrame(split_rows[split]).to_csv(camera_dir / f"{split}_manifest.csv", index=False)
            summary["cameras"][camera_id] = {
                split: _summary_for_split(split_rows[split]) for split in SPLITS
            }
        if args.template_config:
            if not args.config_out_dir or not args.runs_root:
                raise ValueError("--config-out-dir and --runs-root are required when --template-config is set")
            config_map = write_camera_configs(
                camera_ids=sorted(camera_splits.keys()),
                out_root=out_dir,
                template_config=Path(args.template_config).resolve(),
                config_out_dir=Path(args.config_out_dir).resolve(),
                runs_root=args.runs_root,
            )
            summary["generated_configs"] = config_map
    else:
        people = sorted({row["person_id"] for row in rows})
        split_people = split_person_ids(people, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
        _check_no_leakage(split_people)

        rows_by_person: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            rows_by_person[row["person_id"]].append(row)

        split_rows: Dict[str, List[dict]] = {}
        for split in SPLITS:
            split_rows[split] = sorted(
                [row for person_id in split_people[split] for row in rows_by_person[person_id]],
                key=lambda x: (x["camera_id"], x["person_id"], x["sequence_id"]),
            )
            pd.DataFrame(split_rows[split]).to_csv(out_dir / f"{split}_manifest.csv", index=False)

        summary = {
            "video_root": str(video_root),
            "out_dir": str(out_dir),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "allow_online_depth_supervision": bool(args.allow_online_depth_supervision),
            "exclude_name_substrings": list(args.exclude_name_substrings),
            "split_by_camera": False,
            "total_valid_videos": len(rows),
            "bad_videos": len(bad_videos),
            "missing_supervision": len(missing_supervision),
            "splits": {split: _summary_for_split(split_rows[split]) for split in SPLITS},
        }

    _write_json(out_dir / "split_summary.json", summary)
    _write_json(out_dir / "bad_videos.json", bad_videos)
    _write_json(out_dir / "missing_supervision.json", missing_supervision)
    _write_txt(out_dir / "bad_videos.txt", bad_videos)
    _write_txt(out_dir / "missing_supervision.txt", missing_supervision)

    if args.split_by_camera:
        print(f"Done. built per-camera manifests for {len(summary['cameras'])} cameras under {out_dir}")
    else:
        print(
            "Done. "
            f"train={len(split_rows['train'])}, val={len(split_rows['val'])}, test={len(split_rows['test'])}, "
            f"people_train={len(split_people['train'])}, people_val={len(split_people['val'])}, people_test={len(split_people['test'])}"
        )


if __name__ == "__main__":
    main()
