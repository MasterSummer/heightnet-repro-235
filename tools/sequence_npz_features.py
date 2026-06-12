#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

P90_INDEX = 2
MAX_INDEX = 4
EXPECTED_CROP_SHAPE = (1, 128, 64)


def _shape_text(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def load_sequence_npz_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        bbox_feats = np.asarray(data["bbox_feats"], dtype=np.float32)
        height_stats = np.asarray(data["height_stats"], dtype=np.float32)
        heightmap_crops = np.asarray(data["heightmap_crops"], dtype=np.float32)
        valid_count = int(np.asarray(data["valid_count"]).reshape(-1)[0]) if "valid_count" in data.files else bbox_feats.shape[0]

    if bbox_feats.ndim != 2 or bbox_feats.shape[1] != 7:
        raise ValueError(f"bbox_feats must have shape (N, 7), got {_shape_text(bbox_feats.shape)}")
    if height_stats.ndim != 2 or height_stats.shape[1] <= MAX_INDEX:
        raise ValueError(f"height_stats must have at least {MAX_INDEX + 1} columns, got {_shape_text(height_stats.shape)}")
    if heightmap_crops.ndim != 4 or tuple(heightmap_crops.shape[1:]) != EXPECTED_CROP_SHAPE:
        raise ValueError(f"heightmap_crops must have shape (N, 1, 128, 64), got {_shape_text(heightmap_crops.shape)}")

    n_rows = bbox_feats.shape[0]
    if height_stats.shape[0] != n_rows:
        raise ValueError(f"height_stats frame count {height_stats.shape[0]} does not match bbox_feats frame count {n_rows}")
    if heightmap_crops.shape[0] != n_rows:
        raise ValueError(f"heightmap_crops frame count {heightmap_crops.shape[0]} does not match selected feature frame count {n_rows}")
    if valid_count != n_rows:
        raise ValueError(f"valid_count {valid_count} does not match selected feature frame count {n_rows}")

    selected_stats = height_stats[:, [P90_INDEX, MAX_INDEX]].astype(np.float32, copy=False)
    features = np.concatenate([bbox_feats, selected_stats], axis=1).astype(np.float32, copy=False)
    return features, heightmap_crops
