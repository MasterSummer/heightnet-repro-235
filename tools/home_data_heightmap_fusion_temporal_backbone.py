#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from typing import NamedTuple

from torchvision import models


def _load_base_module():
    path = Path(__file__).with_name("home_data_heightmap_fusion.py")
    spec = importlib.util.spec_from_file_location("home_data_heightmap_fusion_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

CAM_RE = re.compile(r"_(?P<h>\d+d\d+)_(?P<a>\d+)_(?P<p>\d+w)_", re.I)
BBOX_FEATURES = base.BBOX_FEATURES
FEATURE_SCHEMA = base.FEATURE_SCHEMA
PRIMARY_METRIC_DEFINITION = base.PRIMARY_METRIC_DEFINITION
VARIANTS = (
    "resnet18_scratch_gated",
    "resnet18_pretrained_gated",
    "resnet34_scratch_gated",
    "resnet34_pretrained_gated",
    "vit_b16_scratch_gated",
    "vit_b16_pretrained_gated",
)
LOSS_FIELDS = [
    "camera_id",
    "variant",
    "seed",
    "epoch",
    "split",
    "total_loss",
    "mse_loss",
    "pair_loss",
    "ordinal_loss",
    "frame_consistency_loss",
    "video_frame_loss",
]


base_camera_from_name = base.base_camera_from_name
load_rank_csv = base.load_rank_csv
load_split_csv = base.load_split_csv
limit_rows_for_smoke = base.limit_rows_for_smoke
person_split_counts = base.person_split_counts
person_overlap_counts = base.person_overlap_counts
apply_person_split = base.apply_person_split
pair_acc = base.pair_acc
pair_acc_counts = base.pair_acc_counts
bisect = base.bisect
spearman = base.spearman
height_gap_bucket = base.height_gap_bucket
parse_near_bucket_weights = base.parse_near_bucket_weights
pair_weight_for_gap = base.pair_weight_for_gap
filter_target_cameras = base.filter_target_cameras


def _shape_text(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def load_sequence_npz_all_frames(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        bbox_feats = np.asarray(data["bbox_feats"], dtype=np.float32)
        height_stats = np.asarray(data["height_stats"], dtype=np.float32)
        crops = np.asarray(data["heightmap_crops"], dtype=np.float32)
    if bbox_feats.ndim != 2 or bbox_feats.shape[1] != 7:
        raise ValueError(f"bbox_feats must have shape (T, 7), got {_shape_text(bbox_feats.shape)}")
    if height_stats.ndim != 2 or height_stats.shape[1] <= 4:
        raise ValueError(f"height_stats must have at least 5 columns, got {_shape_text(height_stats.shape)}")
    if crops.ndim != 4 or tuple(crops.shape[1:]) != (1, 128, 64):
        raise ValueError(f"heightmap_crops must have shape (T, 1, 128, 64), got {_shape_text(crops.shape)}")
    n = bbox_feats.shape[0]
    if n == 0:
        raise ValueError(f"{path}: no feature frames")
    if height_stats.shape[0] != n or crops.shape[0] != n:
        raise ValueError(f"{path}: frame counts do not match")
    frame_features = np.concatenate([bbox_feats, height_stats[:, [2, 4]]], axis=1).astype(np.float32, copy=False)
    if not np.all(np.isfinite(frame_features)):
        raise ValueError(f"{path}: non-finite tabular sequence")
    if not np.all(np.isfinite(crops)):
        raise ValueError(f"{path}: non-finite crop sequence")
    return frame_features.astype(np.float32, copy=False), crops.astype(np.float32, copy=False)


def build_sequence_index(rows: list[dict], feature_root: Path) -> tuple[dict, dict]:
    meta: dict[str, dict] = {}
    missing: dict[str, str] = {}
    split_priority = {"train": 0, "val": 1, "test": 2}
    for row in rows:
        sid = row["sequence_id"]
        existing = meta.get(sid)
        if existing is not None:
            old_p = split_priority.get(existing.get("split", "test"), 99)
            new_p = split_priority.get(row.get("split", "test"), 99)
            if old_p <= new_p:
                continue
        npz_path = feature_root / row["person_id"] / f"{row['video_stem']}.npz"
        if not npz_path.exists():
            missing[sid] = str(npz_path)
            continue
        item = dict(row)
        item["npz_path"] = str(npz_path)
        meta[sid] = item
    return meta, missing


def load_sequences_for_ids(ids: list[str], meta: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    tabular: dict[str, np.ndarray] = {}
    crops: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for sid in ids:
        try:
            x, c = load_sequence_npz_all_frames(Path(meta[sid]["npz_path"]))
        except Exception as exc:
            errors[sid] = str(exc)
            continue
        tabular[sid] = x
        crops[sid] = c
    return tabular, crops, errors


def pad_sequence_batch(ids: list[str], tabular: dict[str, np.ndarray], crops: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_t = max(int(tabular[sid].shape[0]) for sid in ids)
    x = np.zeros((len(ids), max_t, len(FEATURE_SCHEMA)), dtype=np.float32)
    c = np.zeros((len(ids), max_t, 1, 128, 64), dtype=np.float32)
    mask = np.zeros((len(ids), max_t), dtype=bool)
    for i, sid in enumerate(ids):
        n = int(tabular[sid].shape[0])
        x[i, :n] = tabular[sid]
        c[i, :n] = crops[sid]
        mask[i, :n] = True
    return x, c, mask


def standardize_tabular_sequence(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)).astype(np.float32, copy=False)


class CandidateConfig(NamedTuple):
    backbone: str
    pretrained: bool
    fusion: str = "gated"


def parse_candidate(name: str) -> CandidateConfig:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; expected {VARIANTS}")
    if name.startswith("resnet18_"):
        backbone = "resnet18"
    elif name.startswith("resnet34_"):
        backbone = "resnet34"
    elif name.startswith("vit_b16_"):
        backbone = "vit_b16"
    else:
        raise ValueError(f"unknown backbone in {name!r}")
    return CandidateConfig(backbone=backbone, pretrained="_pretrained_" in name, fusion="gated")


class TorchvisionFrameEncoder(nn.Module):
    def __init__(self, cfg: CandidateConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.resize_to = 224 if cfg.backbone == "vit_b16" else None
        if cfg.backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            net = models.resnet18(weights=weights)
            in_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif cfg.backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            net = models.resnet34(weights=weights)
            in_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif cfg.backbone == "vit_b16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            net = models.vit_b_16(weights=weights)
            in_dim = net.heads.head.in_features
            net.heads.head = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone {cfg.backbone!r}")
        self.net = net
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, crops):
        b, t, ch, h, w = crops.shape
        flat = crops.reshape(b * t, ch, h, w)
        flat = flat.repeat(1, 3, 1, 1)
        if self.resize_to is not None:
            flat = torch.nn.functional.interpolate(flat, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        feat = self.proj(self.net(flat))
        return feat.reshape(b, t, -1)


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, features, mask):
        logits = self.score(features).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class FiLMConditioner(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, features, condition):
        scale_shift = self.net(condition)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return features * (1.0 + torch.tanh(scale)) + shift


class GatedFusion(nn.Module):
    def __init__(self, dim: int, n_inputs: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.gate = nn.Linear(dim * n_inputs, n_inputs)

    def forward(self, inputs: list[torch.Tensor]):
        merged = torch.cat(inputs, dim=-1)
        weights = torch.softmax(self.gate(merged), dim=-1)
        stacked = torch.stack(inputs, dim=-2)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=-2)
        return fused, weights


class TemporalBackboneRanker(nn.Module):
    def __init__(self, cfg: CandidateConfig, hidden: int = 128, n_bins: int = 16):
        super().__init__()
        self.cfg = cfg
        self.n_bins = n_bins
        self.bbox = nn.Sequential(nn.Linear(len(FEATURE_SCHEMA), hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.crop = TorchvisionFrameEncoder(cfg, hidden)
        self.attn_bbox = MaskedAttentionPool(hidden)
        self.attn_crop = MaskedAttentionPool(hidden)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1), nn.Sigmoid())
        self.score = nn.Linear(hidden, 1)
        self.height = nn.Linear(hidden, 1)
        self.ordinal = nn.Linear(hidden, n_bins)
        self.frame_score = nn.Linear(hidden, 1)

    def forward(self, tabular, crops, mask):
        bbox_feat = self.bbox(tabular)
        crop_feat = self.crop(crops)
        bbox_pool, wb = self.attn_bbox(bbox_feat, mask)
        crop_pool, wc = self.attn_crop(crop_feat, mask)
        gate = self.gate(torch.cat([bbox_pool, crop_pool], dim=1))
        pooled = gate * crop_pool + (1.0 - gate) * bbox_pool
        frame_scores = self.frame_score(crop_feat).squeeze(-1)
        frame_scores = frame_scores.masked_fill(~mask, 0.0)
        return {
            "score": self.score(pooled).squeeze(-1),
            "video_score": self.score(pooled).squeeze(-1),
            "height": self.height(pooled).squeeze(-1),
            "ordinal_logits": self.ordinal(pooled),
            "frame_scores": frame_scores,
            "mask": mask,
            "gate": gate,
            "attention_weights": {"bbox": wb, "crop": wc, "gate": gate},
        }


def ordinal_targets(heights: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(bin_edges, heights, side="right").clip(0, len(bin_edges)).astype(np.int64)


def build_height_bins(train_heights: np.ndarray, n_bins: int) -> np.ndarray:
    if len(set(float(x) for x in train_heights)) <= 1:
        return np.linspace(float(train_heights.min()) - 1, float(train_heights.max()) + 1, n_bins + 1)[1:-1].astype(np.float32)
    return np.linspace(float(train_heights.min()), float(train_heights.max()), n_bins + 1, dtype=np.float32)[1:-1]


def batch_pair_loss(pred, heights, pair_loss_weight: float, near_bucket_weights: dict[str, float]) -> torch.Tensor:
    pairs = []
    signs = []
    weights = []
    h = heights.detach().cpu().numpy()
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if h[i] == h[j]:
                continue
            pairs.append((i, j))
            signs.append(1.0 if h[i] > h[j] else -1.0)
            weights.append(pair_weight_for_gap(abs(float(h[i] - h[j])), near_bucket_weights))
    if not pairs:
        return pred.sum() * 0.0
    idx = torch.tensor(pairs, device=pred.device, dtype=torch.long)
    sign = torch.tensor(signs, device=pred.device, dtype=torch.float32)
    weight = torch.tensor(weights, device=pred.device, dtype=torch.float32)
    diff = pred[idx[:, 0]] - pred[idx[:, 1]]
    raw = torch.nn.functional.softplus(-sign * diff)
    return pair_loss_weight * (raw * weight).sum() / weight.sum().clamp_min(1e-6)


def masked_frame_consistency_loss(frame_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.float()
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (frame_scores * valid).sum(dim=1, keepdim=True) / denom
    var = (((frame_scores - mean) ** 2) * valid).sum(dim=1) / denom.squeeze(1)
    return var.mean()


def masked_frame_mean(frame_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.float()
    return (frame_scores * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.002):
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.best = None
        self.bad = 0
        self.should_stop = False

    def update(self, val_pairwise: float) -> bool:
        if self.best is None or float(val_pairwise) >= self.best + self.min_delta:
            self.best = float(val_pairwise)
            self.bad = 0
            self.should_stop = False
            return True
        self.bad += 1
        self.should_stop = self.bad >= self.patience
        return False


def compute_losses(
    outputs, heights, ordinal_y, pair_loss_weight, ordinal_loss_weight, near_bucket_weights,
    frame_consistency_weight=0.2, video_frame_weight=0.5,
):
    mse = torch.nn.functional.mse_loss(outputs["height"], heights) / 1000.0
    pair = batch_pair_loss(outputs["score"], heights, pair_loss_weight, near_bucket_weights)
    ordinal = torch.nn.functional.cross_entropy(outputs["ordinal_logits"], ordinal_y)
    frame_consistency = masked_frame_consistency_loss(outputs["frame_scores"], outputs["mask"])
    frame_mean = masked_frame_mean(outputs["frame_scores"], outputs["mask"])
    video_frame = torch.nn.functional.mse_loss(outputs["video_score"], frame_mean)
    total = mse + pair + float(ordinal_loss_weight) * ordinal + float(frame_consistency_weight) * frame_consistency + float(video_frame_weight) * video_frame
    return total, mse, pair, ordinal, frame_consistency, video_frame


def iter_batches(ids: list[str], batch_size: int, shuffle: bool, rng: random.Random):
    order = list(ids)
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield order[i : i + batch_size]


def tensors_for_ids(batch_ids, sid_to_idx, x, c, mask, y, ordinal_y, device):
    idx = [sid_to_idx[sid] for sid in batch_ids]
    return (
        torch.tensor(x[idx], device=device),
        torch.tensor(c[idx], device=device),
        torch.tensor(mask[idx], device=device, dtype=torch.bool),
        torch.tensor(y[idx], device=device),
        torch.tensor(ordinal_y[idx], device=device, dtype=torch.long),
    )


def predict_scores(model, ids, sid_to_idx, x, c, mask, device, batch_size):
    model.eval()
    scores = {}
    heights = {}
    with torch.no_grad():
        for batch_ids in iter_batches(ids, batch_size, False, random.Random(0)):
            idx = [sid_to_idx[sid] for sid in batch_ids]
            xx = torch.tensor(x[idx], device=device)
            cc = torch.tensor(c[idx], device=device)
            mm = torch.tensor(mask[idx], device=device, dtype=torch.bool)
            out = model(xx, cc, mm)
            pp = out["score"].detach().cpu().numpy()
            hh = out["height"].detach().cpu().numpy()
            for sid, score, height in zip(batch_ids, pp, hh):
                scores[sid] = float(score)
                heights[sid] = float(height)
    return scores, heights


def evaluate_loss(model, ids, sid_to_idx, x, c, mask, y, ordinal_y, device, batch_size, pair_loss_weight, ordinal_loss_weight, near_bucket_weights):
    if not ids:
        return {"total_loss": 0.0, "mse_loss": 0.0, "pair_loss": 0.0, "ordinal_loss": 0.0, "frame_consistency_loss": 0.0, "video_frame_loss": 0.0}
    model.eval()
    totals = []
    mses = []
    pairs = []
    ords = []
    frames = []
    video_frames = []
    with torch.no_grad():
        for batch_ids in iter_batches(ids, batch_size, False, random.Random(0)):
            xx, cc, mm, yy, oo = tensors_for_ids(batch_ids, sid_to_idx, x, c, mask, y, ordinal_y, device)
            out = model(xx, cc, mm)
            total, mse, pair, ordinal, frame_consistency, video_frame = compute_losses(out, yy, oo, pair_loss_weight, ordinal_loss_weight, near_bucket_weights)
            totals.append(float(total.detach().cpu()))
            mses.append(float(mse.detach().cpu()))
            pairs.append(float(pair.detach().cpu()))
            ords.append(float(ordinal.detach().cpu()))
            frames.append(float(frame_consistency.detach().cpu()))
            video_frames.append(float(video_frame.detach().cpu()))
    return {
        "total_loss": float(np.mean(totals)),
        "mse_loss": float(np.mean(mses)),
        "pair_loss": float(np.mean(pairs)),
        "ordinal_loss": float(np.mean(ords)),
        "frame_consistency_loss": float(np.mean(frames)),
        "video_frame_loss": float(np.mean(video_frames)),
    }


def select_best_seed_result(results: list[dict]) -> dict:
    def key(result):
        val_sp = result["val"].get("spearman")
        val_sp = -1e9 if val_sp is None else float(val_sp)
        val_bisect = result["val"].get("bisect", {}).get("hit_rate") or 0.0
        return (
            float(result["val_pairwise"]),
            val_sp,
            float(val_bisect),
            -int(result["best_epoch"]),
            -int(result["seed"]),
        )
    return max(results, key=key)


def bucket_pair_acc_counts(ids, scores, heights):
    counts = {b: {"correct": 0, "total": 0, "accuracy": None} for b in ("lt3", "3to5", "5to8", "ge8")}
    order = sorted(ids, key=lambda sid: (-scores[sid], sid))
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            dh = heights[order[i]] - heights[order[j]]
            if abs(dh) <= 0:
                continue
            b = height_gap_bucket(abs(float(dh)))
            counts[b]["total"] += 1
            counts[b]["correct"] += int(dh > 0)
    for b, row in counts.items():
        row["accuracy"] = row["correct"] / row["total"] if row["total"] else None
    return counts


def write_loss_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in LOSS_FIELDS})


def plot_loss_history(out_dir: Path, variant: str, rows: list[dict]) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cameras = sorted({row["camera_id"] for row in rows})
    for camera in cameras:
        cam_rows = [r for r in rows if r["camera_id"] == camera]
        _plot_rows(cam_rows, plot_dir / f"{camera}_{variant}_loss.png", f"{camera} {variant} loss")
    macro_rows = []
    for epoch in sorted({int(r["epoch"]) for r in rows}):
        for split in ["train", "val"]:
            subset = [r for r in rows if int(r["epoch"]) == epoch and r["split"] == split]
            if not subset:
                continue
            macro_rows.append({
                "epoch": epoch,
                "split": split,
                "total_loss": float(np.mean([float(r["total_loss"]) for r in subset])),
                "mse_loss": float(np.mean([float(r["mse_loss"]) for r in subset])),
                "pair_loss": float(np.mean([float(r["pair_loss"]) for r in subset])),
                "ordinal_loss": float(np.mean([float(r["ordinal_loss"]) for r in subset])),
            })
    _plot_rows(macro_rows, plot_dir / f"{variant}_macro_loss.png", f"{variant} macro loss")


def _plot_rows(rows: list[dict], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ["train", "val"]:
        subset = sorted([r for r in rows if r["split"] == split], key=lambda r: int(r["epoch"]))
        if not subset:
            continue
        xs = [int(r["epoch"]) for r in subset]
        ax.plot(xs, [float(r["total_loss"]) for r in subset], label=f"{split} total")
        ax.plot(xs, [float(r["pair_loss"]) for r in subset], linestyle="--", label=f"{split} pair")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train_camera(cam, ids, meta, person_heights, device, epochs, seeds, hidden, variant, ckpt_dir, limit_per_camera, pair_loss_weight, ordinal_loss_weight, near_bucket_weights, batch_size, log_loss_every):
    by_split = {"train": [], "val": [], "test": []}
    heights: dict[str, float] = {}
    for sid in ids:
        pid = meta[sid]["person_id"]
        split = meta[sid]["split"]
        if pid in person_heights and split in by_split:
            by_split[split].append(sid)
            heights[sid] = person_heights[pid]
    by_split = base.limit_split_ids(by_split, limit_per_camera)
    kept = set(by_split["train"] + by_split["val"] + by_split["test"])
    heights = {sid: h for sid, h in heights.items() if sid in kept}
    if len(by_split["train"]) < 2 or len(by_split["test"]) < 2:
        return {"skipped": True, "split_counts": {k: len(v) for k, v in by_split.items()}, "loss_history": []}

    all_ids = by_split["train"] + by_split["val"] + by_split["test"]
    raw_tabular, raw_crops, load_errors = load_sequences_for_ids(all_ids, meta)
    if load_errors:
        ok = set(raw_tabular)
        by_split = {k: [sid for sid in v if sid in ok] for k, v in by_split.items()}
        all_ids = by_split["train"] + by_split["val"] + by_split["test"]
        heights = {sid: h for sid, h in heights.items() if sid in ok}
    if len(by_split["train"]) < 2 or len(by_split["test"]) < 2:
        return {"skipped": True, "split_counts": {k: len(v) for k, v in by_split.items()}, "load_errors": load_errors, "loss_history": []}
    x_raw, crops, mask = pad_sequence_batch(all_ids, raw_tabular, raw_crops)
    sid_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    train_idx = [sid_to_idx[sid] for sid in by_split["train"]]
    train_frames = x_raw[train_idx].reshape(-1, len(FEATURE_SCHEMA))
    mu = train_frames.mean(0)
    sd = train_frames.std(0)
    sd[sd < 1e-6] = 1.0
    x = standardize_tabular_sequence(x_raw, mu, sd)
    y = np.array([heights[sid] for sid in all_ids], dtype=np.float32)
    bin_edges = build_height_bins(np.array([heights[sid] for sid in by_split["train"]], dtype=np.float32), n_bins=16)
    ordinal_y = ordinal_targets(y, bin_edges)

    results = []
    all_loss_rows = []
    cfg = parse_candidate(variant)
    for seed in seeds:
        rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = TemporalBackboneRanker(cfg, hidden=hidden, n_bins=16).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        best = None
        best_state = None
        seed_loss_rows = []
        stopper = EarlyStopper(patience=10, min_delta=0.002)
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch_ids in iter_batches(by_split["train"], batch_size, True, rng):
                xx, cc, mm, yy, oo = tensors_for_ids(batch_ids, sid_to_idx, x, crops, mask, y, ordinal_y, device)
                opt.zero_grad()
                out = model(xx, cc, mm)
                total, mse, pair, ordinal, frame_consistency, video_frame = compute_losses(out, yy, oo, pair_loss_weight, ordinal_loss_weight, near_bucket_weights)
                total.backward()
                opt.step()
                train_losses.append((float(total.detach().cpu()), float(mse.detach().cpu()), float(pair.detach().cpu()), float(ordinal.detach().cpu()), float(frame_consistency.detach().cpu()), float(video_frame.detach().cpu())))
            if epoch % log_loss_every == 0 or epoch == epochs:
                scores, _ = predict_scores(model, all_ids, sid_to_idx, x, crops, mask, device, batch_size)
                val_ids = by_split["val"] or by_split["train"]
                val_pa = pair_acc(val_ids, scores, heights)
                improved = stopper.update(val_pa)
                if best is None or improved:
                    best = {"epoch": epoch, "val_pairwise": val_pa, "scores": scores}
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                train_eval = evaluate_loss(model, by_split["train"], sid_to_idx, x, crops, mask, y, ordinal_y, device, batch_size, pair_loss_weight, ordinal_loss_weight, near_bucket_weights)
                val_eval = evaluate_loss(model, val_ids, sid_to_idx, x, crops, mask, y, ordinal_y, device, batch_size, pair_loss_weight, ordinal_loss_weight, near_bucket_weights)
                for split, metrics in [("train", train_eval), ("val", val_eval)]:
                    row = {
                        "camera_id": cam,
                        "variant": variant,
                        "seed": seed,
                        "epoch": epoch,
                        "split": split,
                        **metrics,
                    }
                    seed_loss_rows.append(row)
                    all_loss_rows.append(row)
                if stopper.should_stop:
                    break
        scores = best.pop("scores")
        item = {"seed": seed, "best_epoch": best["epoch"], "val_pairwise": best["val_pairwise"]}
        for split in ["train", "val", "test"]:
            sids = by_split[split]
            item[split] = {
                "n": len(sids),
                "pairwise_accuracy": pair_acc(sids, scores, heights),
                "pairwise_counts": pair_acc_counts(sids, scores, heights),
                "height_gap_buckets": bucket_pair_acc_counts(sids, scores, heights),
                "spearman": spearman([scores[s] for s in sids], [heights[s] for s in sids]) if len(sids) >= 2 else None,
                "bisect": bisect(sids, scores, heights),
            }
        item["_scores"] = scores
        item["_state"] = best_state
        item["_loss_history"] = seed_loss_rows
        results.append(item)
    best = select_best_seed_result(results)
    best_scores = best.pop("_scores")
    best_state = best.pop("_state")
    best_loss_history = best.pop("_loss_history")
    for r in results:
        r.pop("_scores", None)
        r.pop("_state", None)
        r.pop("_loss_history", None)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cam}_{variant}_best.pt"
    torch.save({
        "model_state_dict": best_state,
        "mu": mu.astype(np.float32),
        "sd": sd.astype(np.float32),
        "feature_schema": FEATURE_SCHEMA,
        "camera_id": cam,
        "best_seed": best["seed"],
        "best_epoch": best["best_epoch"],
        "variant": variant,
        "bin_edges": bin_edges.astype(np.float32),
    }, ckpt_path)
    ranking = sorted(best_scores, key=lambda sid: (-best_scores[sid], sid))
    video_ranking = [{
        "rank": i + 1,
        "sequence_id": sid,
        "person_id": meta[sid]["person_id"],
        "split": meta[sid]["split"],
        "video_filename": meta[sid]["video_filename"],
        "height_cm": float(heights[sid]),
        "score": float(best_scores[sid]),
    } for i, sid in enumerate(ranking)]
    return {
        "skipped": False,
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "features": FEATURE_SCHEMA,
        "variant": variant,
        "checkpoint_path": str(ckpt_path),
        "pair_loss_weight": float(pair_loss_weight),
        "ordinal_loss_weight": float(ordinal_loss_weight),
        "near_bucket_weights": near_bucket_weights,
        "batch_size": int(batch_size),
        "seeds": results,
        "best": best,
        "video_level_ranking": video_ranking,
        "load_errors": load_errors,
        "loss_history": best_loss_history,
    }


def summarize_candidate(results: dict) -> dict:
    vals = []
    correct = total = 0
    spears = []
    for res in results["camera_results"].values():
        if res.get("skipped"):
            continue
        test = res["best"]["test"]
        vals.append(test["pairwise_accuracy"])
        c = test.get("pairwise_counts", {})
        correct += c.get("correct", 0)
        total += c.get("total", 0)
        if test.get("spearman") is not None:
            spears.append(test["spearman"])
    return {
        "n_cameras": len(vals),
        "macro_pairacc": float(sum(vals) / len(vals)) if vals else None,
        "micro_pairacc": float(correct / total) if total else None,
        "median_pairacc": float(np.median(vals)) if vals else None,
        "macro_spearman": float(sum(spears) / len(spears)) if spears else None,
        "pairwise_counts": {"correct": correct, "total": total, "accuracy": float(correct / total) if total else None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/zyding/data")
    ap.add_argument("--feature-root", default="/home/zyding/height/jianzhi_2511_sequence/features")
    ap.add_argument("--out-dir", default="/home/zyding/height/heightnet_repro/runs/home_data_heightmap_fusion_v2")
    ap.add_argument("--variant", choices=VARIANTS, default="resnet18_scratch_gated")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit-per-camera", type=int, default=0)
    ap.add_argument("--person-split-json", default="")
    ap.add_argument("--pair-loss-weight", type=float, default=3.0)
    ap.add_argument("--ordinal-loss-weight", type=float, default=0.5)
    ap.add_argument("--near-bucket-weights", default="lt3=5,3to5=3,5to8=1.5,ge8=1")
    ap.add_argument("--target-cameras", nargs="+", default=[])
    ap.add_argument("--log-loss-every", type=int, default=20)
    ap.add_argument("--plot-loss", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    person_heights = load_rank_csv(data_root / "label" / "rank.json")
    rows = []
    for split in ["train", "val", "test"]:
        rows.extend(load_split_csv(data_root / f"{split}.csv", split))
    source_row_count = len(rows)
    split_counts = None
    overlap_counts = None
    if args.person_split_json:
        person_split = json.loads(Path(args.person_split_json).read_text(encoding="utf-8"))
        split_counts = person_split_counts(person_split)
        overlap_counts = person_overlap_counts(person_split)
        rows = apply_person_split(rows, person_split)
    rows = limit_rows_for_smoke(rows, args.limit_per_camera, person_heights)
    meta, missing = build_sequence_index(rows, Path(args.feature_root))
    cameras = filter_target_cameras(sorted({m["camera_id"] for m in meta.values()}), args.target_cameras)
    near_bucket_weights = parse_near_bucket_weights(args.near_bucket_weights)
    candidate = parse_candidate(args.variant)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {
        "data_root": str(data_root),
        "feature_root": str(args.feature_root),
        "device": device,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "seeds": args.seeds,
        "variant": args.variant,
        "candidate": {"backbone": candidate.backbone, "pretrained": candidate.pretrained, "fusion": candidate.fusion},
        "variants_supported": VARIANTS,
        "feature_schema": FEATURE_SCHEMA,
        "frame_input": "all_frames",
        "cross_frame_constraints": {
            "frame_consistency_loss": "variance of per-frame crop scores over valid frames",
            "video_frame_loss": "MSE between video score and masked mean frame score",
        },
        "camera_embedding": False,
        "single_camera_training": True,
        "limit_per_camera": args.limit_per_camera,
        "target_cameras": args.target_cameras,
        "person_split_json": args.person_split_json or None,
        "person_split_counts": split_counts,
        "person_overlap_counts": overlap_counts,
        "primary_metric_definition": PRIMARY_METRIC_DEFINITION,
        "pair_loss_weight": args.pair_loss_weight,
        "ordinal_loss_weight": args.ordinal_loss_weight,
        "near_bucket_weights": near_bucket_weights,
        "batch_size": args.batch_size,
        "log_loss_every": args.log_loss_every,
        "n_rows": len(rows),
        "source_n_rows": source_row_count,
        "n_sequences_with_npz": len(meta),
        "missing_npz_count": len(missing),
        "camera_results": {},
        "summary": {},
    }
    all_loss_rows = []
    for cam in cameras:
        ids = [sid for sid, m in meta.items() if m["camera_id"] == cam]
        print(f"[CAM] {cam} n={len(ids)} variant={args.variant}", flush=True)
        res = train_camera(
            cam, ids, meta, person_heights, device, args.epochs, args.seeds, args.hidden, args.variant,
            out_dir / "checkpoints", args.limit_per_camera, args.pair_loss_weight, args.ordinal_loss_weight,
            near_bucket_weights, args.batch_size, max(1, args.log_loss_every),
        )
        all_loss_rows.extend(res.get("loss_history", []))
        results["camera_results"][cam] = res
    results["summary"] = summarize_candidate(results)
    result_path = out_dir / f"base_camera_heightmap_fusion_temporal_backbone_{args.variant}_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    loss_path = out_dir / f"loss_history_{args.variant}.csv"
    write_loss_history(loss_path, all_loss_rows)
    if args.plot_loss:
        plot_loss_history(out_dir, args.variant, all_loss_rows)
    print("SUMMARY", results["summary"], flush=True)
    print("[OUT]", result_path, flush=True)
    print("[LOSS]", loss_path, flush=True)


if __name__ == "__main__":
    main()
