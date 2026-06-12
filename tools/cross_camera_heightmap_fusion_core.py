#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn


FEATURE_SCHEMA = [
    "bbox_h_norm",
    "bbox_w_norm",
    "bbox_y1_norm_neg",
    "bbox_y2_norm",
    "bbox_cy_norm",
    "area_norm",
    "rect_score",
    "height_p90",
    "height_max",
]
CAMERA_RE = re.compile(r"_(?P<h>\d+d\d+)_(?P<a>\d+)(?:_(?P<p>\d+w))?_", re.I)
CAMERA_ID_RE = re.compile(r"^(?P<h>\d+d\d+)_(?P<a>\d+)$", re.I)
CAMERA_GEOMETRY_FEATURE_DIM = 7


@dataclass(frozen=True)
class SequenceFrames:
    tabular: np.ndarray
    crops: np.ndarray
    camera_id: str
    video_stem: str

    @property
    def count(self) -> int:
        return int(self.tabular.shape[0])

    def aggregate(self) -> tuple[np.ndarray, np.ndarray]:
        return self.tabular.mean(axis=0), self.crops.mean(axis=0)


def base_camera_from_name(name: str) -> str | None:
    match = CAMERA_RE.search(name)
    if not match:
        return None
    return f"{match.group('h').lower()}_{match.group('a')}"


def camera_height_m(camera_id: str) -> float:
    match = CAMERA_ID_RE.match(str(camera_id))
    if not match:
        raise ValueError(f"cannot parse camera height from camera_id={camera_id!r}")
    return float(match.group("h").replace("d", "."))


def _camera_matrix(image_width: int, image_height: int, focal_px: float | None = None) -> np.ndarray:
    focal = float(focal_px) if focal_px is not None else float(max(image_width, image_height))
    return np.array(
        [[focal, 0.0, image_width / 2.0], [0.0, focal, image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_x(pitch_deg: float) -> np.ndarray:
    pitch_rad = math.radians(float(pitch_deg))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(pitch_rad), -math.sin(pitch_rad)], [0.0, math.sin(pitch_rad), math.cos(pitch_rad)]],
        dtype=np.float64,
    )


def image_point_to_world_ground(
    u: float,
    v: float,
    image_width: int,
    image_height: int,
    cam_height: float,
    pitch_deg: float,
    focal_px: float | None = None,
) -> np.ndarray:
    camera_matrix = _camera_matrix(image_width, image_height, focal_px)
    rotation = _rotation_x(pitch_deg)
    t_cam = -rotation @ np.array([[0.0], [0.0], [float(cam_height)]], dtype=np.float64)
    projection = camera_matrix @ np.hstack((rotation, t_cam))
    p_col1, p_col2, p_col4 = projection[:, 0], projection[:, 1], projection[:, 3]
    a = np.array(
        [[p_col1[0] - u * p_col1[2], p_col2[0] - u * p_col2[2]], [p_col1[1] - v * p_col1[2], p_col2[1] - v * p_col2[2]]],
        dtype=np.float64,
    )
    b = np.array([u * p_col4[2] - p_col4[0], v * p_col4[2] - p_col4[1]], dtype=np.float64)
    try:
        world_xy = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0], dtype=np.float32)
    if not np.all(np.isfinite(world_xy)):
        return np.array([0.0, 0.0], dtype=np.float32)
    return world_xy.astype(np.float32)


def pixels_per_meter_at_image_point(
    u: float,
    v: float,
    image_width: int,
    image_height: int,
    cam_height: float,
    pitch_deg: float,
    focal_px: float | None = None,
) -> float:
    camera_matrix = _camera_matrix(image_width, image_height, focal_px)
    world_xy = image_point_to_world_ground(u, v, image_width, image_height, cam_height, pitch_deg, focal_px)
    rotation = _rotation_x(pitch_deg)
    t_cam = -rotation @ np.array([[0.0], [0.0], [float(cam_height)]], dtype=np.float64)
    foot = np.array([[float(world_xy[0])], [float(world_xy[1])], [0.0]], dtype=np.float64)
    above = np.array([[float(world_xy[0])], [float(world_xy[1])], [1.0]], dtype=np.float64)
    foot_cam = rotation @ foot + t_cam
    above_cam = rotation @ above + t_cam
    if abs(float(foot_cam[2, 0])) < 1e-9 or abs(float(above_cam[2, 0])) < 1e-9:
        return 0.0
    foot_proj = (camera_matrix @ foot_cam).reshape(-1)
    above_proj = (camera_matrix @ above_cam).reshape(-1)
    foot_v = foot_proj[1] / foot_proj[2]
    above_v = above_proj[1] / above_proj[2]
    ppm = abs(float(above_v - foot_v))
    return ppm if math.isfinite(ppm) else 0.0


def camera_geometry_features(
    tabular: np.ndarray,
    camera_id: str,
    image_width: int = 800,
    image_height: int = 600,
    pitch_deg: float = 18.0,
    focal_px: float | None = None,
) -> np.ndarray:
    values = np.asarray(tabular, dtype=np.float32).reshape(-1)
    cam_height = camera_height_m(camera_id)
    bbox_h_norm = float(values[0]) if values.size else 0.0
    foot_v_norm = float(values[3]) if values.size > 3 else 0.0
    foot_u = image_width / 2.0
    foot_v = float(np.clip(foot_v_norm, 0.0, 1.2) * image_height)
    ppm = pixels_per_meter_at_image_point(foot_u, foot_v, image_width, image_height, cam_height, pitch_deg, focal_px)
    pixel_height = max(0.0, bbox_h_norm * image_height)
    geom_height_m = pixel_height / ppm if ppm > 1.0 else 0.0
    world_xy = image_point_to_world_ground(foot_u, foot_v, image_width, image_height, cam_height, pitch_deg, focal_px)
    pitch_rad = math.radians(float(pitch_deg))
    features = np.array(
        [
            math.sin(pitch_rad),
            math.cos(pitch_rad),
            foot_v_norm,
            math.log1p(max(0.0, ppm)) / 6.0,
            geom_height_m,
            float(world_xy[0]) / 10.0,
            float(world_xy[1]) / 10.0,
        ],
        dtype=np.float32,
    )
    features[~np.isfinite(features)] = 0.0
    return features


def _scalar_text(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    values = np.asarray(data[key]).reshape(-1)
    return str(values[0]) if values.size else default


@lru_cache(maxsize=1024)
def load_npz_frames(path: Path | str) -> SequenceFrames:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"bbox_feats", "height_stats", "heightmap_crops"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing NPZ fields {missing}")
        bbox = np.asarray(data["bbox_feats"], dtype=np.float32)
        stats = np.asarray(data["height_stats"], dtype=np.float32)
        crops = np.asarray(data["heightmap_crops"], dtype=np.float32)
        valid_count = int(np.asarray(data["valid_count"]).reshape(-1)[0]) if "valid_count" in data.files else len(bbox)
        camera_id = _scalar_text(data, "camera_id")
        video_stem = _scalar_text(data, "video_stem", path.stem)
    if bbox.ndim != 2 or bbox.shape[1] != 7:
        raise ValueError(f"{path}: bbox_feats must have shape (N, 7), got {bbox.shape}")
    if stats.ndim != 2 or stats.shape[1] < 5:
        raise ValueError(f"{path}: height_stats must have shape (N, >=5), got {stats.shape}")
    if crops.ndim != 4 or tuple(crops.shape[1:]) != (1, 128, 64):
        raise ValueError(f"{path}: heightmap_crops must have shape (N, 1, 128, 64), got {crops.shape}")
    if len(stats) != len(bbox) or len(crops) != len(bbox):
        raise ValueError(f"{path}: feature frame counts are not aligned")
    if valid_count <= 0 or valid_count > len(bbox):
        raise ValueError(f"{path}: invalid valid_count={valid_count} for stored rows={len(bbox)}")
    tabular = np.concatenate([bbox[:valid_count], stats[:valid_count, [2, 4]]], axis=1).astype(np.float32, copy=False)
    crops = crops[:valid_count].astype(np.float32, copy=False)
    if not np.all(np.isfinite(tabular)) or not np.all(np.isfinite(crops)):
        raise ValueError(f"{path}: non-finite selected features")
    if camera_id:
        camera_height_m(camera_id)
    return SequenceFrames(tabular=tabular, crops=crops, camera_id=camera_id, video_stem=video_stem)


def load_height_labels(path: Path | str) -> dict[str, float]:
    labels: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            person_id = row.get("person_id") or row.get("penson_id") or row.get("pid")
            height = row.get("height(cm)") or row.get("height_cm") or row.get("height")
            if person_id and height:
                labels[str(person_id)] = float(height)
    return labels


def load_split_rows(csv_path: Path | str, feature_root: Path | str, split: str) -> list[dict]:
    feature_root = Path(feature_root)
    rows: list[dict] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            person_id = raw.get("person_id") or raw.get("penson_id") or raw.get("pid")
            video_path = str(raw.get("video_path") or "")
            video_filename = str(raw.get("video_filename") or Path(video_path).name)
            camera_id = base_camera_from_name(video_filename)
            if not person_id or not video_filename or not camera_id:
                continue
            video_stem = Path(video_filename).stem
            rows.append({
                "split": split,
                "person_id": str(person_id),
                "video_path": video_path,
                "video_filename": video_filename,
                "video_stem": video_stem,
                "camera_id": camera_id,
                "camera_height_m": camera_height_m(camera_id),
                "npz_path": str(feature_root / str(person_id) / f"{video_stem}.npz"),
            })
    return rows


def filter_labeled_rows(rows: Iterable[dict], labels: dict[str, float]) -> list[dict]:
    return [dict(row, height_cm=float(labels[row["person_id"]])) for row in rows if row["person_id"] in labels]


def filter_strict_no_train_overlap(rows: Iterable[dict], train_people: set[str]) -> list[dict]:
    return [row for row in rows if row["person_id"] not in train_people]


def sample_cross_camera_triplet(rows: list[dict], rng: np.random.Generator) -> tuple[dict, dict, dict]:
    by_person: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_person[str(row["person_id"])].append(row)
    eligible = [
        pid for pid, items in by_person.items()
        if len({item["camera_id"] for item in items}) >= 2
    ]
    if not eligible:
        raise ValueError("no person has observations from at least two cameras")
    person_id = str(rng.choice(eligible))
    first = by_person[person_id][int(rng.integers(len(by_person[person_id])))]
    alternatives = [row for row in by_person[person_id] if row["camera_id"] != first["camera_id"]]
    second = alternatives[int(rng.integers(len(alternatives)))]
    others = [row for row in rows if row["person_id"] != person_id]
    if not others:
        raise ValueError("cross-camera consistency requires a third-person observation")
    third = others[int(rng.integers(len(others)))]
    return first, second, third


def sample_track_frame_pair(frame_count: int, rng: np.random.Generator) -> tuple[int, int]:
    if frame_count < 2:
        raise ValueError("track consistency requires at least two valid frames")
    pair = rng.choice(frame_count, size=2, replace=False)
    return int(pair[0]), int(pair[1])


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CropEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(1, 16), nn.ReLU(inplace=True), ResidualBlock(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 32), nn.ReLU(inplace=True), ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        return self.net(crop)


class CrossCameraFusionRanker(nn.Module):
    def __init__(self, camera_count: int, tabular_dim: int = 9, embedding_dim: int = 96, camera_geometry_dim: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.camera_geometry_dim = int(camera_geometry_dim)
        self.tabular = nn.Sequential(
            nn.Linear(tabular_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 48), nn.ReLU(inplace=True),
        )
        self.crop = CropEncoder(48)
        self.camera_embedding = nn.Embedding(camera_count, 12)
        self.camera_context = nn.Sequential(nn.Linear(13 + self.camera_geometry_dim, 16), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Linear(48 + 48 + 16, embedding_dim), nn.ReLU(inplace=True))
        self.score = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def encode(
        self,
        tabular: torch.Tensor,
        crop: torch.Tensor,
        camera_index: torch.Tensor,
        camera_height: torch.Tensor,
        camera_geometry: torch.Tensor | None = None,
    ) -> torch.Tensor:
        camera = torch.cat([self.camera_embedding(camera_index), camera_height.view(-1, 1)], dim=1)
        if self.camera_geometry_dim:
            if camera_geometry is None:
                camera_geometry = camera.new_zeros((camera.shape[0], self.camera_geometry_dim))
            camera = torch.cat([camera, camera_geometry.to(camera.device, dtype=camera.dtype)], dim=1)
        return self.fusion(torch.cat([self.tabular(tabular), self.crop(crop), self.camera_context(camera)], dim=1))

    def forward(
        self,
        tabular: torch.Tensor | None = None,
        crop: torch.Tensor | None = None,
        camera_index: torch.Tensor | None = None,
        camera_height: torch.Tensor | None = None,
        camera_geometry: torch.Tensor | None = None,
        *,
        op: str = "encode",
        encoded_a: torch.Tensor | None = None,
        encoded_b: torch.Tensor | None = None,
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if op == "compare_encoded":
            if encoded_a is None or encoded_b is None:
                raise ValueError("encoded_a and encoded_b are required for op='compare_encoded'")
            return self.compare_encoded(encoded_a, encoded_b)
        if op == "score":
            if embeddings is None:
                raise ValueError("embeddings is required for op='score'")
            return self.score(embeddings).squeeze(1)
        if tabular is None or crop is None or camera_index is None or camera_height is None:
            raise ValueError("tabular, crop, camera_index, and camera_height are required for op='encode'")
        return self.encode(tabular, crop, camera_index, camera_height, camera_geometry)

    def compare_encoded(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (self.score(a) - self.score(b)).squeeze(1)


def score_identity_consistency_loss(
    model: CrossCameraFusionRanker,
    embeddings: torch.Tensor,
    person_ids: list[str],
) -> tuple[torch.Tensor, int]:
    if embeddings.shape[0] != len(person_ids):
        raise ValueError(f"embeddings/person_ids length mismatch: {embeddings.shape[0]} vs {len(person_ids)}")
    scores = model.score(embeddings).squeeze(1)
    losses: list[torch.Tensor] = []
    for person_id in sorted(set(str(pid) for pid in person_ids)):
        indices = [idx for idx, pid in enumerate(person_ids) if str(pid) == person_id]
        if len(indices) < 2:
            continue
        group = scores[torch.tensor(indices, dtype=torch.long, device=scores.device)]
        losses.append(((group - group.mean()) ** 2).mean())
    if not losses:
        return scores.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def soft_copeland_scores(
    embeddings: torch.Tensor,
    compare_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    chunk_size: int = 256,
) -> torch.Tensor:
    n = int(embeddings.shape[0])
    scores = embeddings.new_zeros(n)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        left = embeddings[start:stop]
        subtotal = embeddings.new_zeros(stop - start)
        for other_start in range(0, n, chunk_size):
            other_stop = min(n, other_start + chunk_size)
            right = embeddings[other_start:other_stop]
            logits = compare_fn(
                left[:, None, :].expand(-1, len(right), -1).reshape(-1, embeddings.shape[1]),
                right[None, :, :].expand(len(left), -1, -1).reshape(-1, embeddings.shape[1]),
            ).reshape(len(left), len(right))
            subtotal += torch.sigmoid(logits).sum(dim=1)
        scores[start:stop] = subtotal - 0.5
    return scores


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and values[order[end + 1]] == values[order[pos]]:
            end += 1
        rank = (pos + end + 2.0) / 2.0
        for idx in order[pos:end + 1]:
            ranks[idx] = rank
        pos = end + 1
    return ranks


def correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denominator = math.sqrt(float((aa * aa).sum() * (bb * bb).sum()))
    return float((aa * bb).sum() / denominator) if denominator else None
