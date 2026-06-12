from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PathLike:
    @staticmethod
    def dirname(path: str) -> str:
        return os.path.dirname(path)

    @staticmethod
    def basename(path: str) -> str:
        return os.path.basename(path)


@dataclass
class VideoRow:
    video_path: str
    frame_path: str
    sequence_id: str
    person_id: str
    camera_id: str
    frame_start: int
    frame_end: int
    fps: float
    valid_frames_path: str
    height_cache_path: str
    valid_mask_cache_path: str
    depth_cache_path: str
    bg_depth_path: str
    camera_height_m: float


def _infer_camera_height_m(camera_id: str) -> float:
    m = re.search(r"(\d+)cm_", camera_id, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot infer camera height from camera_id={camera_id}")
    return float(m.group(1)) / 100.0


def _clean_path_value(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


class HeightDataset(Dataset):
    """Video-level dataset: decode frames online and provide adjacent-frame sample."""

    def __init__(
        self,
        manifest_path: str,
        image_size: Tuple[int, int],
        normalize_rgb: bool = True,
        use_pair_consistency: bool = True,
        train_mode: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.image_h, self.image_w = image_size
        self.normalize_rgb = normalize_rgb
        self.use_pair_consistency = use_pair_consistency
        self.train_mode = train_mode

        frame = pd.read_csv(manifest_path)
        required_cols = {
            "video_path",
            "sequence_id",
            "person_id",
            "camera_id",
            "frame_start",
            "frame_end",
            "fps",
        }
        missing = required_cols - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest missing video-level columns: {sorted(missing)}")

        self.rows: List[VideoRow] = []
        for _, row in frame.iterrows():
            cam = str(row["camera_id"])
            cam_h = float(row["camera_height_m"]) if "camera_height_m" in frame.columns else _infer_camera_height_m(cam)
            self.rows.append(
                VideoRow(
                    video_path=str(row["video_path"]),
                    frame_path=_clean_path_value(row["frame_path"]) if "frame_path" in frame.columns else "",
                    sequence_id=str(row["sequence_id"]),
                    person_id=str(row["person_id"]),
                    camera_id=cam,
                    frame_start=int(row["frame_start"]),
                    frame_end=int(row["frame_end"]),
                    fps=float(row["fps"]),
                    valid_frames_path=_clean_path_value(row["valid_frames_path"]) if "valid_frames_path" in frame.columns else "",
                    height_cache_path=_clean_path_value(row["height_cache_path"]) if "height_cache_path" in frame.columns else "",
                    valid_mask_cache_path=_clean_path_value(row["valid_mask_cache_path"]) if "valid_mask_cache_path" in frame.columns else "",
                    depth_cache_path=_clean_path_value(row["depth_cache_path"]) if "depth_cache_path" in frame.columns else "",
                    bg_depth_path=_clean_path_value(row["bg_depth_path"]) if "bg_depth_path" in frame.columns else "",
                    camera_height_m=cam_h,
                )
            )

        self._cache_arrays: Dict[str, np.ndarray] = {}
        self._sequence_sampled_frames = self._build_sequence_sampled_frames()

    @classmethod
    def from_video_root(
        cls,
        video_root: str,
        bg_depth_root: str,
        image_size: Tuple[int, int],
        normalize_rgb: bool = True,
        use_pair_consistency: bool = True,
        train_mode: bool = True,
    ) -> "HeightDataset":
        ds = cls.__new__(cls)
        ds.manifest_path = ""
        ds.image_h, ds.image_w = image_size
        ds.normalize_rgb = normalize_rgb
        ds.use_pair_consistency = use_pair_consistency
        ds.train_mode = train_mode
        ds._cache_arrays = {}
        ds.rows = ds._build_rows_from_video_root(video_root=video_root, bg_depth_root=bg_depth_root)
        ds._sequence_sampled_frames = ds._build_sequence_sampled_frames()
        return ds

    def __len__(self) -> int:
        return len(self.rows)

    def _build_sequence_sampled_frames(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        grouped: Dict[str, List[int]] = {}
        for row in self.rows:
            grouped.setdefault(row.sequence_id, []).append(int(row.frame_start))
        for key, values in grouped.items():
            uniq = sorted(set(values))
            if len(uniq) > 1:
                out[key] = uniq
        return out

    def _infer_person_camera(self, video_path: str) -> tuple[str, str]:
        full_text = video_path
        person_id = PathLike.basename(PathLike.dirname(video_path)) if PathLike.dirname(video_path) else "unknown_person"
        camera_id = "unknown_camera"

        person_pat = re.compile(r"\d{4}_(?:man|woman)\d+", re.IGNORECASE)
        camera_pat = re.compile(r"\d+cm_(?:inside|outside|slantside|side|front|back)", re.IGNORECASE)

        m_person = person_pat.search(full_text)
        m_camera = camera_pat.search(full_text)
        if m_person:
            person_id = m_person.group(0)
        if m_camera:
            camera_id = m_camera.group(0).lower()
        return person_id, camera_id

    def _infer_bg_depth_path(self, bg_depth_root: str, camera_id: str) -> str:
        if not bg_depth_root or camera_id == "unknown_camera":
            return ""
        p = os.path.join(bg_depth_root, camera_id, f"{camera_id}_avg_depth.npy")
        return p if os.path.exists(p) else ""

    def _build_rows_from_video_root(self, video_root: str, bg_depth_root: str) -> List[VideoRow]:
        if not os.path.exists(video_root):
            raise FileNotFoundError(f"video_root not found: {video_root}")

        exts = {".mp4", ".avi", ".mov", ".mkv"}
        videos: List[str] = []
        for root, _, files in os.walk(video_root):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    videos.append(os.path.join(root, name))
        videos = sorted(videos)
        if not videos:
            raise RuntimeError(f"no videos found under {video_root}")

        rows: List[VideoRow] = []
        bad_videos: List[tuple[str, str]] = []
        for video_path in videos:
            sequence_stem = os.path.splitext(os.path.basename(video_path))[0]
            person_id, camera_id = self._infer_person_camera(video_path)
            sequence_id = f"{person_id}__{sequence_stem}"
            try:
                frame_count, fps = self._video_meta(video_path)
            except Exception as exc:
                bad_videos.append((video_path, str(exc)))
                continue
            bg_depth_path = self._infer_bg_depth_path(bg_depth_root, camera_id)
            camera_height_m = _infer_camera_height_m(camera_id)
            rows.append(
                VideoRow(
                    video_path=video_path,
                    frame_path="",
                    sequence_id=sequence_id,
                    person_id=person_id,
                    camera_id=camera_id,
                    frame_start=0,
                    frame_end=frame_count - 1,
                    fps=fps,
                    valid_frames_path="",
                    height_cache_path="",
                    valid_mask_cache_path="",
                    depth_cache_path="",
                    bg_depth_path=bg_depth_path,
                    camera_height_m=camera_height_m,
                )
            )
        if bad_videos:
            print(f"[warn] skipped {len(bad_videos)} bad videos under {video_root}")
            for path, err in bad_videos[:10]:
                print(f"[warn] bad video: {path} ({err})")
            if len(bad_videos) > 10:
                print(f"[warn] ... and {len(bad_videos) - 10} more bad videos")
        if not rows:
            raise RuntimeError(f"no valid videos found under {video_root}")
        return rows

    def _video_meta(self, path: str) -> tuple[int, float]:
        cap = cv2.VideoCapture(path)
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

    def _normalize_rgb(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        if self.normalize_rgb:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std
        return np.transpose(x, (2, 0, 1))

    def _load_rgb_file(self, frame_path: str) -> np.ndarray:
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"cannot open frame image: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        return frame

    def _load_video_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"failed to decode frame={frame_idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        return frame

    def _pick_frame(self, start: int, end: int) -> int:
        if end < start:
            raise ValueError(f"invalid frame range: [{start}, {end}]")
        if self.train_mode:
            return random.randint(start, end)
        return start

    def _load_valid_frames(self, row: VideoRow) -> List[int]:
        if not row.valid_frames_path:
            return []
        arr = self._load_cache_array(row.valid_frames_path)
        values = np.asarray(arr).reshape(-1)
        out = []
        for value in values:
            idx = int(value)
            if row.frame_start <= idx <= row.frame_end:
                out.append(idx)
        deduped = sorted(set(out))
        return deduped

    def _pick_frame_for_row(self, row: VideoRow) -> int:
        valid_frames = self._load_valid_frames(row)
        if valid_frames:
            if self.train_mode:
                return int(random.choice(valid_frames))
            return int(valid_frames[0])
        return self._pick_frame(row.frame_start, row.frame_end)

    def _pick_adjacent(self, idx: int, start: int, end: int) -> int:
        if idx + 1 <= end:
            return idx + 1
        if idx - 1 >= start:
            return idx - 1
        return idx

    def _pick_adjacent_for_row(self, row: VideoRow, idx: int) -> int:
        sampled_frames = self._sequence_sampled_frames.get(row.sequence_id, [])
        if sampled_frames:
            nearest = min(range(len(sampled_frames)), key=lambda k: abs(sampled_frames[k] - idx))
            if nearest + 1 < len(sampled_frames):
                return int(sampled_frames[nearest + 1])
            if nearest - 1 >= 0:
                return int(sampled_frames[nearest - 1])
            return int(sampled_frames[nearest])
        valid_frames = self._load_valid_frames(row)
        if valid_frames:
            if idx in valid_frames:
                pos = valid_frames.index(idx)
                if pos + 1 < len(valid_frames):
                    return int(valid_frames[pos + 1])
                if pos - 1 >= 0:
                    return int(valid_frames[pos - 1])
                return int(idx)
            nearest = min(valid_frames, key=lambda x: abs(x - idx))
            return int(nearest)
        return self._pick_adjacent(idx, row.frame_start, row.frame_end)

    def _frame_retry_order(self, frame_idx: int, start: int, end: int) -> List[int]:
        if end < start:
            raise ValueError(f"invalid frame range: [{start}, {end}]")
        out = [frame_idx]
        max_offset = max(frame_idx - start, end - frame_idx)
        for offset in range(1, max_offset + 1):
            right = frame_idx + offset
            left = frame_idx - offset
            if right <= end:
                out.append(right)
            if left >= start:
                out.append(left)
        return out

    def _single_frame_loose_retry_order(self, frame_idx: int, radius: int = 8) -> List[int]:
        out = [frame_idx]
        for offset in range(1, radius + 1):
            right = frame_idx + offset
            left = frame_idx - offset
            out.append(right)
            if left >= 0:
                out.append(left)
        return out

    def _load_cache_array(self, path: str) -> np.ndarray:
        if path in self._cache_arrays:
            return self._cache_arrays[path]
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        obj = np.load(path, mmap_mode="r")
        if isinstance(obj, np.ndarray):
            arr = obj
        else:
            keys = list(obj.keys())
            key = "height" if "height" in keys else "valid_mask" if "valid_mask" in keys else "depth" if "depth" in keys else keys[0]
            arr = obj[key]
        self._cache_arrays[path] = arr
        return arr

    def _load_bg_depth(self, path: str) -> np.ndarray:
        bg = self._load_cache_array(path).astype(np.float32)
        if bg.ndim == 3:
            bg = bg[0]
        if bg.shape != (self.image_h, self.image_w):
            bg = cv2.resize(bg, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        return bg.astype(np.float32)

    def _load_height_and_mask(self, row: VideoRow, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Returns: (height, valid_mask, bg_depth, camera_height_m, need_online_depth_flag)
        """
        # Prefer direct height/mask cache.
        if row.height_cache_path and row.valid_mask_cache_path:
            h_arr = self._load_cache_array(row.height_cache_path)
            m_arr = self._load_cache_array(row.valid_mask_cache_path)
            height = np.asarray(h_arr[frame_idx], dtype=np.float32)
            valid = np.asarray(m_arr[frame_idx], dtype=np.float32)
            bg = self._load_bg_depth(row.bg_depth_path) if row.bg_depth_path else np.zeros((self.image_h, self.image_w), dtype=np.float32)
            need_online = 0.0
        elif row.depth_cache_path and row.bg_depth_path:
            d_arr = self._load_cache_array(row.depth_cache_path)
            depth = np.asarray(d_arr[frame_idx], dtype=np.float32)
            bg = self._load_bg_depth(row.bg_depth_path)
            if depth.shape != (self.image_h, self.image_w):
                depth = cv2.resize(depth, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
            valid = (np.isfinite(depth) & np.isfinite(bg) & (np.abs(bg) > 1e-6)).astype(np.float32)
            height = np.zeros_like(depth, dtype=np.float32)
            vv = valid > 0.5
            height[vv] = row.camera_height_m * (bg[vv] - depth[vv]) / bg[vv]
            height = np.clip(height, 0.0, row.camera_height_m * 3.0)
            need_online = 0.0
        elif row.bg_depth_path:
            # Fallback: no cache exists, caller should run online depth and then derive height/mask.
            bg = self._load_bg_depth(row.bg_depth_path)
            height = np.zeros((self.image_h, self.image_w), dtype=np.float32)
            valid = np.zeros((self.image_h, self.image_w), dtype=np.float32)
            need_online = 1.0
        else:
            raise RuntimeError(
                f"manifest row lacks supervision cache for sequence={row.sequence_id}. "
                "Need (height_cache_path + valid_mask_cache_path) or (depth_cache_path + bg_depth_path)."
            )

        if height.shape != (self.image_h, self.image_w):
            height = cv2.resize(height, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)
        if valid.shape != (self.image_h, self.image_w):
            valid = cv2.resize(valid, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)
        if bg.shape != (self.image_h, self.image_w):
            bg = cv2.resize(bg, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        return height.astype(np.float32), (valid > 0.5).astype(np.float32), bg.astype(np.float32), float(row.camera_height_m), need_online

    def _build_sample(self, row: VideoRow, frame_idx: int) -> tuple[dict, int]:
        last_error: Exception | None = None
        actual_frame_idx = frame_idx
        img_raw: np.ndarray | None = None
        if row.frame_path:
            img_raw = self._load_rgb_file(row.frame_path)
            actual_frame_idx = row.frame_start
        else:
            for candidate_idx in self._frame_retry_order(frame_idx, row.frame_start, row.frame_end):
                try:
                    img_raw = self._load_video_frame(row.video_path, candidate_idx)
                    actual_frame_idx = candidate_idx
                    break
                except RuntimeError as exc:
                    last_error = exc
            if img_raw is None and row.frame_start == row.frame_end:
                # Frame manifests often pin a single sampled frame. If that decode fails,
                # search a small neighborhood in the source video instead of hard-failing.
                for candidate_idx in self._single_frame_loose_retry_order(frame_idx):
                    if candidate_idx == frame_idx:
                        continue
                    try:
                        img_raw = self._load_video_frame(row.video_path, candidate_idx)
                        actual_frame_idx = candidate_idx
                        break
                    except RuntimeError as exc:
                        last_error = exc
        if img_raw is None:
            if last_error is not None:
                raise RuntimeError(
                    f"failed to decode any frame in range [{row.frame_start}, {row.frame_end}] "
                    f"for requested frame={frame_idx} from {row.video_path}"
                ) from last_error
            raise RuntimeError(f"failed to decode frame={frame_idx} from {row.video_path}")
        height, valid, bg_depth, camera_height_m, need_online = self._load_height_and_mask(row, actual_frame_idx)
        sample = {
            "image": torch.from_numpy(self._normalize_rgb(img_raw)),
            "image_raw": torch.from_numpy(np.transpose(img_raw.copy(), (2, 0, 1))),
            "height": torch.from_numpy(height).unsqueeze(0),
            "mask": torch.from_numpy(valid).unsqueeze(0),
            "bg_depth": torch.from_numpy(bg_depth).unsqueeze(0),
            "camera_height_m": torch.tensor(camera_height_m, dtype=torch.float32),
            "need_online_depth": torch.tensor(need_online, dtype=torch.float32),
        }
        return sample, actual_frame_idx

    @staticmethod
    def _compute_depth_cache_path(row: VideoRow, frame_idx: int) -> str:
        """Compute DA2 cache path from the source video and sampled frame index."""
        if not row.video_path:
            return ""
        video_dir = os.path.dirname(row.video_path)
        video_stem = os.path.splitext(os.path.basename(row.video_path))[0]
        return os.path.join(video_dir, ".depth_cache", f"{video_stem}.frame_{frame_idx:06d}.depth.npy")

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        requested_frame_idx = self._pick_frame_for_row(row)
        sample, frame_idx = self._build_sample(row, requested_frame_idx)
        sample.update(
            {
                "person_id": row.person_id,
                "camera_id": row.camera_id,
                "sequence_id": row.sequence_id,
                "frame_idx": frame_idx,
                "frame_path": row.frame_path,
                "depth_cache_path": self._compute_depth_cache_path(row, frame_idx),
                "bg_depth_path": row.bg_depth_path,
            }
        )

        if self.use_pair_consistency:
            frame_idx_pair = self._pick_adjacent_for_row(row, frame_idx)
            pair, frame_idx_pair = self._build_sample(row, frame_idx_pair)
            sample["image_pair"] = pair["image"]
            sample["image_pair_raw"] = pair["image_raw"]
            sample["frame_idx_pair"] = frame_idx_pair

        return sample
