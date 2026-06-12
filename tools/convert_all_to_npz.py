#!/usr/bin/env python3
# pyright: basic
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CAMERA_RE = re.compile(r"_(?P<h>\d+d\d+)_(?P<a>\d+)_", re.IGNORECASE)
RESOLUTION_RE = re.compile(r"_(?P<res>\d+w)_", re.IGNORECASE)


@dataclass(frozen=True)
class VideoRecord:
    person_id: str
    video_filename: str
    video_path: str
    split: str
    coat_type: str
    action: str

    @property
    def video_stem(self) -> str:
        return Path(self.video_filename).stem

    @property
    def camera_id(self) -> str:
        return infer_camera_id(self.video_stem)

    @property
    def camera_height_m(self) -> float:
        return infer_camera_height_m(self.camera_id)

    @property
    def output_path(self) -> Path:
        return Path(DEFAULTS.output_root) / self.person_id / f"{self.video_stem}.npz"


@dataclass(frozen=True)
class Defaults:
    video_root: str = "/data2/dataset/jianzhi_2511/video2"
    bg_depth_root: str = "/home/zyding/height/jianzhi_2511_spilt_coat/bg_depthmap"
    parsing_json_root: str = "/data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/scanner/main_card_out"
    manifest_paths: tuple[str, ...] = (
        "/home/zyding/data/train.csv",
        "/home/zyding/data/val.csv",
        "/home/zyding/data/test.csv",
    )
    depthanything_root: str = "/home/zyding/height/Depth-Anything-V2"
    checkpoint: str = "/home/zyding/height/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
    output_root: str = "/home/zyding/height/jianzhi_2511_sequence/features"


DEFAULTS = Defaults()


class ProgressLogger:
    def __init__(self, total: int, every: int = 50) -> None:
        self.total = max(int(total), 0)
        self.every = max(int(every), 1)
        self.start_time = time.time()

    def maybe_log(self, done: int, stats: dict[str, int]) -> None:
        if done <= 0:
            return
        if done % self.every != 0 and done != self.total:
            return
        elapsed = max(time.time() - self.start_time, 1e-6)
        rate = done / elapsed
        remaining = max(self.total - done, 0)
        eta = remaining / max(rate, 1e-6)
        print(
            "[progress] "
            f"processed={done}/{self.total} "
            f"ok={stats.get('ok', 0)} skipped={stats.get('skipped', 0)} failed={stats.get('failed', 0)} "
            f"missing_bbox={stats.get('missing_bbox', 0)} missing_bg={stats.get('missing_bg', 0)} "
            f"eta_sec={eta:.1f}",
            flush=True,
        )


class ParsingCache:
    def __init__(self) -> None:
        self._json_cache: dict[Path, Any] = {}
        self._frame_bbox_cache: dict[Path, dict[int, dict[str, Any]]] = {}

    def load_json(self, path: Path) -> Any:
        if path not in self._json_cache:
            with path.open("r", encoding="utf-8") as f:
                self._json_cache[path] = json.load(f)
        return self._json_cache[path]

    def frame_bbox_map(self, path: Path) -> dict[int, dict[str, Any]]:
        if path not in self._frame_bbox_cache:
            self._frame_bbox_cache[path] = build_frame_bbox_map(self.load_json(path))
        return self._frame_bbox_cache[path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert all videos to sequence-level NPZ height-ranking features.")
    p.add_argument("--video-root", default=DEFAULTS.video_root, type=str)
    p.add_argument("--bg-depth-root", default=DEFAULTS.bg_depth_root, type=str)
    p.add_argument("--parsing-json-root", default=DEFAULTS.parsing_json_root, type=str)
    p.add_argument("--manifest", nargs="*", default=list(DEFAULTS.manifest_paths))
    p.add_argument("--depthanything-root", default=DEFAULTS.depthanything_root, type=str)
    p.add_argument("--checkpoint", default=DEFAULTS.checkpoint, type=str)
    p.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"], type=str)
    p.add_argument("--input-size", default=518, type=int)
    p.add_argument("--output-root", default=DEFAULTS.output_root, type=str)
    p.add_argument("--frames-per-video", default=20, type=int)
    p.add_argument("--limit-videos", default=0, type=int)
    p.add_argument("--limit-per-person", default=0, type=int)
    p.add_argument("--persons", nargs="*", default=[])
    p.add_argument("--cache-persons", nargs="*", default=["1124_man1"])
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=True)
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--online-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite-summary", action="store_true")
    return p.parse_args()


def infer_camera_id(text: str) -> str:
    match = CAMERA_RE.search(str(text))
    if not match:
        return "unknown_camera"
    return f"{match.group('h').lower()}_{match.group('a')}"


def infer_resolution_token(text: str) -> str:
    match = RESOLUTION_RE.search(str(text))
    return match.group("res").lower() if match else ""


def infer_camera_height_m(camera_id: str) -> float:
    match = re.search(r"(\d+)d(\d+)", str(camera_id), re.IGNORECASE)
    if not match:
        return 0.0
    return float(f"{match.group(1)}.{match.group(2)}")


def get_rank_world() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        world_size = 1
    if not 0 <= rank < world_size:
        rank = 0
    return rank, world_size


def build_parsing_index(parsing_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(parsing_root.glob("*.json")):
        stem = path.stem
        parts = stem.split("_")
        person_id = "_".join(parts[:2]) if len(parts) >= 2 else stem
        index.setdefault(person_id, []).append(path)
    return index


def choose_best_parsing_json(person_id: str, video_stem: str, parsing_index: dict[str, list[Path]]) -> Path | None:
    candidates = parsing_index.get(person_id, [])
    if not candidates:
        return None
    target_cam = infer_camera_id(video_stem)
    target_res = infer_resolution_token(video_stem)
    target_tokens = set(re.split(r"[_\W]+", video_stem.lower()))

    best_path: Path | None = None
    best_score: tuple[int, int, int, str] | None = None
    for path in candidates:
        stem = path.stem
        stem_wo_person = stem[len(person_id) + 1 :] if stem.startswith(f"{person_id}_") else stem
        cand_cam = infer_camera_id(stem_wo_person)
        cand_res = infer_resolution_token(stem_wo_person)
        cand_tokens = set(re.split(r"[_\W]+", stem_wo_person.lower()))
        token_overlap = len([t for t in cand_tokens if t and t in target_tokens])
        score = (
            1 if cand_cam == target_cam else 0,
            1 if target_res and cand_res == target_res else 0,
            token_overlap,
            stem,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_path = path
    return best_path


def build_frame_bbox_map(payload: Any) -> dict[int, dict[str, Any]]:
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
                    bbox_xyxy = (x, y, x + w, y + h)
                    score = float(item.get("score", 0.0))
                    prev = frame_map.get(int(frame_id))
                    if prev is None or score >= float(prev.get("score", 0.0)):
                        frame_map[int(frame_id)] = {"bbox_xyxy": bbox_xyxy, "score": score}
    return frame_map


def compute_bbox_features(bbox_xyxy: tuple[float, float, float, float], frame_w: int, frame_h: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    aspect = w / h
    rel_h = h / max(float(frame_h), 1.0)
    area = (w * h) / max(float(frame_w * frame_h), 1.0)
    feat = np.asarray(
        [
            cx / max(float(frame_w), 1.0),
            cy / max(float(frame_h), 1.0),
            w / max(float(frame_w), 1.0),
            h / max(float(frame_h), 1.0),
            aspect,
            rel_h,
            area,
        ],
        dtype=np.float32,
    )
    return feat


def clamp_bbox(bbox_xyxy: tuple[float, float, float, float], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    xi1 = max(0, min(frame_w - 1, int(math.floor(x1)))) if frame_w > 0 else 0
    yi1 = max(0, min(frame_h - 1, int(math.floor(y1)))) if frame_h > 0 else 0
    xi2 = max(1, min(frame_w, int(math.ceil(x2)))) if frame_w > 0 else 1
    yi2 = max(1, min(frame_h, int(math.ceil(y2)))) if frame_h > 0 else 1
    if xi2 <= xi1:
        xi1, xi2 = 0, max(frame_w, 1)
    if yi2 <= yi1:
        yi1, yi2 = 0, max(frame_h, 1)
    return xi1, yi1, xi2, yi2


def compute_height_stats(height: np.ndarray, valid_mask: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    frame_h, frame_w = height.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox_xyxy, frame_w, frame_h)
    roi = height[y1:y2, x1:x2]
    roi_valid = valid_mask[y1:y2, x1:x2].astype(bool)
    vals = roi[np.isfinite(roi) & roi_valid]
    if vals.size == 0:
        return np.zeros((8,), dtype=np.float32)

    bg_vals = height[np.isfinite(height) & valid_mask.astype(bool)]
    bg_mean = float(bg_vals.mean()) if bg_vals.size > 0 else 0.0
    masked_avg = float((roi * roi_valid.astype(np.float32)).sum() / max(float(roi_valid.sum()), 1.0))
    stats = np.asarray(
        [
            float(vals.mean()),
            float(vals.std()),
            float(np.percentile(vals, 90)),
            float(np.percentile(vals, 10)),
            float(vals.max()),
            float(vals.min()),
            float(vals.mean() - bg_mean),
            masked_avg,
        ],
        dtype=np.float32,
    )
    return stats


def extract_heightmap_crop(
    height: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    out_h: int = 128,
    out_w: int = 64,
    context_ratio: float = 0.15,
) -> np.ndarray:
    frame_h, frame_w = height.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    x1 -= bw * context_ratio
    x2 += bw * context_ratio
    y1 -= bh * context_ratio
    y2 += bh * context_ratio
    xi1, yi1, xi2, yi2 = clamp_bbox((x1, y1, x2, y2), frame_w, frame_h)
    crop = height[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        crop = np.zeros((out_h, out_w), dtype=np.float32)
    else:
        crop = cv2.resize(crop.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return crop[np.newaxis, ...].astype(np.float32)


def load_bg_depth(path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    bg = np.load(path, allow_pickle=True)
    if hasattr(bg, "files"):
        files = list(getattr(bg, "files"))
        key = "depth" if "depth" in files else files[0]
        bg = bg[key]
    bg = np.asarray(bg, dtype=np.float32)
    if bg.ndim == 3:
        bg = bg[0]
    h, w = target_hw
    if bg.shape != (h, w):
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg.astype(np.float32)


def depth_to_height_np(depth: np.ndarray, bg_depth: np.ndarray, camera_height_m: float) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    bg_depth = np.asarray(bg_depth, dtype=np.float32)
    valid = np.isfinite(depth) & np.isfinite(bg_depth) & (np.abs(bg_depth) > 1e-6)
    height = np.zeros_like(depth, dtype=np.float32)
    height[valid] = float(camera_height_m) * (bg_depth[valid] - depth[valid]) / bg_depth[valid]
    height = np.clip(height, 0.0, float(camera_height_m) * 3.0)
    return height.astype(np.float32), valid.astype(bool)


def sample_frame_indices(num_frames: int, frames_per_video: int) -> list[int]:
    if num_frames <= 0:
        return []
    frames_per_video = max(1, min(int(frames_per_video), int(num_frames)))
    if frames_per_video == 1:
        return [min(num_frames // 2, num_frames - 1)]
    xs = np.linspace(0, num_frames - 1, num=frames_per_video)
    out = sorted({int(round(float(x))) for x in xs})
    if len(out) < frames_per_video:
        for idx in range(num_frames):
            if idx not in out:
                out.append(idx)
            if len(out) >= frames_per_video:
                break
    return sorted(out[:frames_per_video])


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def load_depth_cache(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=True)
        if hasattr(arr, "files"):
            files = list(getattr(arr, "files"))
            key = "depth" if "depth" in files else files[0]
            arr = arr[key]
    except Exception as exc:
        raise RuntimeError(f"failed to load depth cache {path}: {exc}") from exc
    depth = np.asarray(arr, dtype=np.float32)
    h, w = target_hw
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float32)


def build_da2_model(depthanything_root: Path, checkpoint: Path, encoder: str):
    import torch
    import importlib

    if str(depthanything_root) not in sys.path:
        sys.path.insert(0, str(depthanything_root))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dpt_module = importlib.import_module("depth_anything_v2.dpt")
        DepthAnythingV2 = getattr(dpt_module, "DepthAnythingV2")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        model_device = f"cuda:{local_rank}"
    else:
        model_device = "cpu"
    model = model.to(model_device).eval()
    return model, model_device


def deduplicate_records(manifest_paths: list[str]) -> list[VideoRecord]:
    seen: set[str] = set()
    records: list[VideoRecord] = []
    for manifest_path in manifest_paths:
        df = pd.read_csv(manifest_path)
        for row in df.to_dict(orient="records"):
            video_path = str(row.get("video_path", "")).strip()
            if not video_path or video_path in seen:
                continue
            seen.add(video_path)
            records.append(
                VideoRecord(
                    person_id=str(row.get("person_id", "")).strip(),
                    video_filename=str(row.get("video_filename", Path(video_path).name)).strip(),
                    video_path=video_path,
                    split=str(row.get("split", "")).strip(),
                    coat_type=str(row.get("coat_type", "")).strip(),
                    action=str(row.get("action", "")).strip(),
                )
            )
    return sorted(records, key=lambda x: (x.person_id, x.video_filename, x.video_path))


def filter_records(args: argparse.Namespace, records: list[VideoRecord]) -> list[VideoRecord]:
    selected = records
    persons = {p.strip() for p in args.persons if str(p).strip()}
    cache_people = {p.strip() for p in args.cache_persons if str(p).strip()}
    if persons:
        selected = [r for r in selected if r.person_id in persons]
    if args.cache_only:
        selected = [r for r in selected if r.person_id in cache_people]
    if args.online_only:
        selected = [r for r in selected if r.person_id not in cache_people]
    if args.limit_per_person > 0:
        kept: list[VideoRecord] = []
        counts: dict[str, int] = {}
        for record in selected:
            cnt = counts.get(record.person_id, 0)
            if cnt >= args.limit_per_person:
                continue
            kept.append(record)
            counts[record.person_id] = cnt + 1
        selected = kept
    if args.limit_videos > 0:
        selected = selected[: args.limit_videos]
    rank, world_size = get_rank_world()
    return selected[rank::world_size]


def video_mode(record: VideoRecord, cache_people: set[str]) -> str:
    return "cache" if record.person_id in cache_people else "online"


def resolve_bg_depth_path(bg_depth_root: Path, camera_id: str) -> Path | None:
    path = bg_depth_root / camera_id / f"{camera_id}_avg_depth.npy"
    return path if path.exists() else None


def process_video(
    record: VideoRecord,
    args: argparse.Namespace,
    parsing_index: dict[str, list[Path]],
    parsing_cache: ParsingCache,
    da2_model: Any,
    da2_device: Any,
) -> tuple[str, dict[str, Any]]:
    output_path = Path(args.output_root) / record.person_id / f"{record.video_stem}.npz"
    if args.skip_existing and output_path.exists():
        return "skipped", {"video": record.video_path, "reason": "existing"}

    video_path = Path(record.video_path)
    if not video_path.exists():
        return "failed", {"video": record.video_path, "reason": "missing_video"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "failed", {"video": record.video_path, "reason": "open_failed"}
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames <= 0 or frame_w <= 0 or frame_h <= 0:
            return "failed", {"video": record.video_path, "reason": "bad_video_meta"}

        bg_path = resolve_bg_depth_path(Path(args.bg_depth_root), record.camera_id)
        if bg_path is None:
            return "missing_bg", {"video": record.video_path, "camera_id": record.camera_id}
        bg_depth = load_bg_depth(bg_path, (frame_h, frame_w))
        if bg_depth is None:
            return "missing_bg", {"video": record.video_path, "camera_id": record.camera_id}

        parsing_path = choose_best_parsing_json(record.person_id, record.video_stem, parsing_index)
        frame_bbox_map: dict[int, dict[str, Any]] = {}
        if parsing_path is not None and parsing_path.exists():
            frame_bbox_map = parsing_cache.frame_bbox_map(parsing_path)

        mode = video_mode(record, {p.strip() for p in args.cache_persons if str(p).strip()})
        sample_indices = sample_frame_indices(total_frames, args.frames_per_video)
        bbox_feats = []
        height_stats = []
        heightmap_crops = []
        valid_count = 0
        corrupt_depth = 0
        missing_bbox = 0
        bad_frame = 0

        for frame_idx in sample_indices:
            frame = read_video_frame(cap, frame_idx)
            if frame is None:
                bad_frame += 1
                continue
            if mode == "cache":
                cache_path = video_path.parent / ".depth_cache" / f"{record.video_stem}.frame_{frame_idx:06d}.depth.npy"
                try:
                    depth = load_depth_cache(cache_path, (frame_h, frame_w))
                except Exception:
                    corrupt_depth += 1
                    continue
            else:
                if da2_model is None:
                    raise RuntimeError("online mode requested but DA2 model not initialized")
                depth = da2_model.infer_image(frame, args.input_size).astype(np.float32)
                if depth.shape != (frame_h, frame_w):
                    depth = cv2.resize(depth, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

            height, valid = depth_to_height_np(depth, bg_depth, record.camera_height_m)
            bbox_entry = frame_bbox_map.get(int(frame_idx))
            if bbox_entry is None:
                missing_bbox += 1
                bbox = (0.0, 0.0, float(frame_w), float(frame_h))
                bbox_feat = np.zeros((7,), dtype=np.float32)
                stats = np.zeros((8,), dtype=np.float32)
                crop = np.zeros((1, 128, 64), dtype=np.float32)
            else:
                bbox = bbox_entry["bbox_xyxy"]
                bbox_feat = compute_bbox_features(bbox, frame_w=frame_w, frame_h=frame_h)
                stats = compute_height_stats(height, valid, bbox)
                crop = extract_heightmap_crop(height, bbox, out_h=128, out_w=64)
                valid_count += 1

            bbox_feats.append(bbox_feat)
            height_stats.append(stats)
            heightmap_crops.append(crop)

        if not bbox_feats:
            return "failed", {
                "video": record.video_path,
                "reason": "no_valid_sampled_frames",
                "corrupt_depth": corrupt_depth,
                "bad_frame": bad_frame,
            }

        payload = {
            "bbox_feats": np.stack(bbox_feats, axis=0).astype(np.float32),
            "height_stats": np.stack(height_stats, axis=0).astype(np.float32),
            "heightmap_crops": np.stack(heightmap_crops, axis=0).astype(np.float32),
            "valid_count": np.asarray([int(valid_count)], dtype=np.int32),
            "camera_id": np.asarray([record.camera_id]),
            "video_stem": np.asarray([record.video_stem]),
        }
        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **payload)
        return (
            "ok",
            {
                "video": record.video_path,
                "mode": mode,
                "frames_requested": len(sample_indices),
                "valid_count": int(valid_count),
                "missing_bbox_frames": int(missing_bbox),
                "corrupt_depth_frames": int(corrupt_depth),
                "bad_frame_count": int(bad_frame),
                "output": str(output_path),
            },
        )
    finally:
        cap.release()


def print_summary(stats: dict[str, int], results: list[dict[str, Any]], out_path: Path) -> None:
    summary = {
        "stats": stats,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[summary] " + json.dumps(stats, ensure_ascii=False), flush=True)
    print(f"[summary_file] {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.cache_only and args.online_only:
        raise ValueError("cannot set both --cache-only and --online-only")
    if args.frames_per_video < 1:
        raise ValueError("--frames-per-video must be >= 1")

    rank, world_size = get_rank_world()
    print(f"[init] rank={rank} world_size={world_size}", flush=True)
    print(f"[init] output_root={args.output_root}", flush=True)

    records = deduplicate_records([str(x) for x in args.manifest])
    selected = filter_records(args, records)
    parsing_root = Path(args.parsing_json_root)
    parsing_index = build_parsing_index(parsing_root)
    parsing_cache = ParsingCache()

    cache_people = {p.strip() for p in args.cache_persons if str(p).strip()}
    need_online = any(video_mode(r, cache_people) == "online" for r in selected)
    da2_model = None
    da2_device = None
    if need_online:
        da2_model, da2_device = build_da2_model(Path(args.depthanything_root), Path(args.checkpoint), args.encoder)

    stats = {
        "selected": len(selected),
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "missing_bg": 0,
        "missing_bbox": 0,
    }
    results: list[dict[str, Any]] = []
    progress = ProgressLogger(total=len(selected), every=50)

    for idx, record in enumerate(selected, 1):
        status, payload = process_video(record, args, parsing_index, parsing_cache, da2_model, da2_device)
        stats[status] = stats.get(status, 0) + 1
        if int(payload.get("missing_bbox_frames", 0)) > 0:
            stats["missing_bbox"] += 1
        payload["status"] = status
        payload["person_id"] = record.person_id
        payload["camera_id"] = record.camera_id
        results.append(payload)
        if status != "ok":
            print(f"[warn] video={record.video_path} status={status} payload={json.dumps(payload, ensure_ascii=False)}", flush=True)
        progress.maybe_log(idx, stats)

    summary_path = Path(args.output_root) / f"convert_all_to_npz_summary.rank{rank}.json"
    if summary_path.exists() and not args.overwrite_summary:
        summary_path = Path(args.output_root) / f"convert_all_to_npz_summary.rank{rank}.{int(time.time())}.json"
    print_summary(stats, results, summary_path)


if __name__ == "__main__":
    main()
