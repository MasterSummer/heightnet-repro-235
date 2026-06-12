#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

CAM_RE = re.compile(r"_(?P<h>\d+d\d+)_(?P<a>\d+)_(?P<p>\d+w)_", re.I)
BBOX_FEATURES = [
    "bbox_h_norm",
    "bbox_w_norm",
    "bbox_y1_norm_neg",
    "bbox_y2_norm",
    "bbox_cy_norm",
    "area_norm",
    "rect_score",
]
FEATURE_SCHEMA = BBOX_FEATURES + ["height_p90", "height_max"]
VARIANTS = ("bbox_only", "bbox_heightstats", "crop_only", "fused_concat", "shuffled_heightstats", "shuffled_crops")
NEGATIVE_CONTROL_VARIANTS = frozenset({"shuffled_heightstats", "shuffled_crops"})
PRIMARY_METRIC_DEFINITION = (
    "one record per usable video/NPZ; sequence_id=person_id__video_stem; "
    "no person-camera aggregation; no person aggregation; pairwise excludes equal-height pairs"
)


def variant_model_mode(variant: str) -> str:
    if variant in NEGATIVE_CONTROL_VARIANTS:
        return "fused_concat"
    return variant


def variant_result_label(variant: str) -> str:
    if variant in NEGATIVE_CONTROL_VARIANTS:
        return f"negative_control:{variant}"
    return variant


def apply_negative_control(variant: str, X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if variant not in NEGATIVE_CONTROL_VARIANTS:
        return X, C
    rng = np.random.default_rng(12345)
    if variant == "shuffled_heightstats":
        shuffled = X.copy()
        perm = rng.permutation(shuffled.shape[0])
        shuffled[:, 7:9] = shuffled[perm, 7:9]
        return shuffled, C
    if variant == "shuffled_crops":
        perm = rng.permutation(C.shape[0])
        return X, C[perm].copy()
    raise ValueError(f"unknown negative control variant {variant!r}")


def base_camera_from_name(name: str) -> str | None:
    m = CAM_RE.search(name)
    if not m:
        return None
    return f"{m.group('h').lower()}_{m.group('a')}"


def load_rank_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("penson_id") or row.get("person_id")
            h = row.get("height(cm)") or row.get("height_cm") or row.get("height")
            if pid and h:
                out[str(pid)] = float(h)
    return out


def load_split_csv(path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("person_id") or row.get("pid") or row.get("penson_id")
            video_path = row.get("video_path") or ""
            video_filename = row.get("video_filename") or Path(video_path).name
            if not pid or not video_filename:
                continue
            stem = Path(video_filename).stem
            cam = base_camera_from_name(video_filename)
            if not cam:
                continue
            rows.append({
                "split": split,
                "person_id": str(pid),
                "video_filename": video_filename,
                "video_path": video_path,
                "sequence_id": f"{pid}__{stem}",
                "video_stem": stem,
                "camera_id": cam,
            })
    return rows


def _shape_text(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def aggregate_sequence_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        bbox_feats = np.asarray(data["bbox_feats"], dtype=np.float32)
        height_stats = np.asarray(data["height_stats"], dtype=np.float32)
        frame_crops = np.asarray(data["heightmap_crops"], dtype=np.float32)
    if bbox_feats.ndim != 2 or bbox_feats.shape[1] != 7:
        raise ValueError(f"bbox_feats must have shape (N, 7), got {_shape_text(bbox_feats.shape)}")
    if height_stats.ndim != 2 or height_stats.shape[1] <= 4:
        raise ValueError(f"height_stats must have at least 5 columns, got {_shape_text(height_stats.shape)}")
    if frame_crops.ndim != 4 or tuple(frame_crops.shape[1:]) != (1, 128, 64):
        raise ValueError(f"heightmap_crops must have shape (N, 1, 128, 64), got {_shape_text(frame_crops.shape)}")
    n_rows = bbox_feats.shape[0]
    if n_rows == 0:
        raise ValueError(f"{path}: no feature frames")
    if height_stats.shape[0] != n_rows:
        raise ValueError(f"height_stats frame count {height_stats.shape[0]} does not match bbox_feats frame count {n_rows}")
    if frame_crops.shape[0] != n_rows:
        raise ValueError(f"heightmap_crops frame count {frame_crops.shape[0]} does not match bbox_feats frame count {n_rows}")
    frame_features = np.concatenate([bbox_feats, height_stats[:, [2, 4]]], axis=1).astype(np.float32, copy=False)
    tabular = np.nanmean(frame_features, axis=0).astype(np.float32, copy=False)
    crop = np.nanmean(frame_crops, axis=0).astype(np.float32, copy=False)
    if tabular.shape != (9,):
        raise ValueError(f"{path}: expected 9 tabular features, got {tabular.shape}")
    if crop.shape != (1, 128, 64):
        raise ValueError(f"{path}: expected crop shape (1, 128, 64), got {crop.shape}")
    if not np.all(np.isfinite(tabular)):
        raise ValueError(f"{path}: non-finite tabular aggregate")
    if not np.all(np.isfinite(crop)):
        raise ValueError(f"{path}: non-finite crop aggregate")
    return tabular.astype(np.float32, copy=False), crop.astype(np.float32, copy=False)


def limit_rows_for_smoke(rows: list[dict], limit_per_camera: int | None, person_heights: dict[str, float]) -> list[dict]:
    if not limit_per_camera or limit_per_camera <= 0:
        return rows
    counts: dict[tuple[str, str], int] = {}
    kept: list[dict] = []
    for row in rows:
        if row["person_id"] not in person_heights:
            continue
        key = (row["split"], row["camera_id"])
        if counts.get(key, 0) >= limit_per_camera:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(row)
    return kept


def person_split_counts(person_split: dict[str, list[str]]) -> dict[str, int]:
    return {split: len(set(person_split.get(split, []))) for split in ("train", "val", "test")}


def person_overlap_counts(person_split: dict[str, list[str]]) -> dict[str, int]:
    train = set(person_split.get("train", []))
    val = set(person_split.get("val", []))
    test = set(person_split.get("test", []))
    return {
        "train_val": len(train & val),
        "train_test": len(train & test),
        "val_test": len(val & test),
    }


def apply_person_split(rows: list[dict], person_split: dict[str, list[str]]) -> list[dict]:
    person_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for person_id in person_split.get(split, []):
            person_to_split[str(person_id)] = split
    reassigned: list[dict] = []
    seen_sequence_ids: set[str] = set()
    for row in rows:
        split = person_to_split.get(str(row.get("person_id")))
        if split is None:
            continue
        sequence_id = str(row.get("sequence_id") or "")
        if sequence_id in seen_sequence_ids:
            continue
        seen_sequence_ids.add(sequence_id)
        item = dict(row)
        item["split"] = split
        reassigned.append(item)
    return reassigned


def build_sequence_table(rows: list[dict], feature_root: Path) -> tuple[dict, dict, dict, dict]:
    meta: dict[str, dict] = {}
    tabular: dict[str, list[float]] = {}
    crops: dict[str, np.ndarray] = {}
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
        try:
            seq_tabular, seq_crop = aggregate_sequence_npz(npz_path)
        except Exception as exc:
            missing[sid] = f"{npz_path}: {exc}"
            continue
        item = dict(row)
        item["npz_path"] = str(npz_path)
        meta[sid] = item
        tabular[sid] = seq_tabular.tolist()
        crops[sid] = seq_crop
    return meta, tabular, crops, missing


def rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j + 2.0) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    if len(a) < 2:
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa -= aa.mean(); bb -= bb.mean()
    den = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    return float((aa * bb).sum() / den) if den > 0 else None


def spearman(pred, gt):
    return pearson(rankdata(pred), rankdata(gt))


def height_gap_bucket(gap: float) -> str:
    return "lt3" if gap < 3 else "3to5" if gap < 5 else "5to8" if gap < 8 else "ge8"


def parse_near_bucket_weights(text: str | None) -> dict[str, float]:
    weights = {"lt3": 1.0, "3to5": 1.0, "5to8": 1.0, "ge8": 1.0}
    if not text:
        return weights
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid near bucket weight {part!r}; expected key=value")
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise ValueError(f"invalid near bucket key {key!r}; expected one of {sorted(weights)}")
        weights[key] = float(value)
    return weights


def pair_weight_for_gap(gap: float, weights: dict[str, float]) -> float:
    return float(weights[height_gap_bucket(float(gap))])


def pair_acc(ids, scores, heights):
    metrics = pair_acc_counts(ids, scores, heights)
    return metrics["accuracy"]


def pair_acc_counts(ids, scores, heights):
    order = sorted(ids, key=lambda sid: (-scores[sid], sid))
    correct = total = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            dh = heights[order[i]] - heights[order[j]]
            if abs(dh) <= 0:
                continue
            total += 1
            correct += int(dh > 0)
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def bisect(ids, scores, heights, overlap_ratio=0.2):
    ranking = sorted(ids, key=lambda sid: (-scores[sid], sid))
    n = len(ranking)
    if n == 0:
        return {"hit_rate": 0.0, "hit_count": 0, "evaluated_count": 0}
    vals = sorted(heights[s] for s in ranking)
    thr = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    overlap_n = int(round(n * overlap_ratio))
    mid = n // 2
    start = max(0, mid - overlap_n // 2)
    end = min(n, start + overlap_n)
    overlap = set(ranking[start:end])
    high = {s for i, s in enumerate(ranking) if i < mid or s in overlap}
    low = {s for i, s in enumerate(ranking) if i >= mid or s in overlap}
    hit = 0; sizes = []
    for sid in ranking:
        gt = "high" if heights[sid] >= thr else "low"
        if sid in high and sid in low:
            ok = True; size = len(high | low)
        elif sid in high:
            ok = gt == "high"; size = len(high)
        elif sid in low:
            ok = gt == "low"; size = len(low)
        else:
            ok = False; size = 0
        hit += int(ok); sizes.append(size)
    avg = sum(sizes) / len(sizes)
    return {"hit_rate": hit / n, "hit_count": hit, "evaluated_count": n, "avg_candidate_size": avg, "search_space_reduction_ratio": 1 - avg / max(1, n)}


class TabularMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, max(8, hidden // 2)), nn.ReLU(),
        )
        self.out_dim = max(8, hidden // 2)

    def forward(self, x):
        return self.net(x)


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

    def forward(self, x):
        return self.act(x + self.net(x))


class CropResNetEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2, bias=False), nn.GroupNorm(1, 16), nn.ReLU(inplace=True),
            ResidualBlock(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(1, 32), nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.GroupNorm(1, 64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, crop):
        return self.net(crop)


class FusionRegressor(nn.Module):
    def __init__(self, variant: str, tabular_dim: int = 9, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant_model_mode(variant)
        self.tabular_dim = tabular_dim
        if variant == "bbox_only":
            self.tabular = TabularMLP(7, hidden, dropout)
            head_in = self.tabular.out_dim
        elif variant == "bbox_heightstats":
            self.tabular = TabularMLP(tabular_dim, hidden, dropout)
            head_in = self.tabular.out_dim
        elif variant == "crop_only":
            self.crop = CropResNetEncoder(max(16, hidden // 2))
            head_in = self.crop.out_dim
        else:
            self.tabular = TabularMLP(tabular_dim, hidden, dropout)
            self.crop = CropResNetEncoder(max(16, hidden // 2))
            head_in = self.tabular.out_dim + self.crop.out_dim
        self.score = nn.Linear(head_in, 1)

    def forward(self, tabular, crop):
        if self.variant == "bbox_only":
            feat = self.tabular(tabular[:, :7])
        elif self.variant == "bbox_heightstats":
            feat = self.tabular(tabular)
        elif self.variant == "crop_only":
            feat = self.crop(crop)
        else:
            feat = torch.cat([self.tabular(tabular), self.crop(crop)], dim=1)
        return self.score(feat).squeeze(-1)


def limit_split_ids(by_split: dict[str, list[str]], limit_per_camera: int | None) -> dict[str, list[str]]:
    if not limit_per_camera or limit_per_camera <= 0:
        return by_split
    return {k: sorted(v)[:limit_per_camera] for k, v in by_split.items()}


def filter_target_cameras(cameras: list[str], target_cameras: list[str] | None) -> list[str]:
    if not target_cameras:
        return cameras
    wanted = set(target_cameras)
    available = set(cameras)
    missing = sorted(wanted - available)
    if missing:
        raise ValueError(f"target camera(s) not found: {missing}; available={sorted(available)}")
    return [cam for cam in cameras if cam in wanted]


def train_camera(cam, ids, meta, tabular, crops, person_heights, device, epochs, seeds, hidden, variant, ckpt_dir, limit_per_camera=None, pair_loss_weight=0.5, near_bucket_weights=None):
    by_split = {"train": [], "val": [], "test": []}
    heights: dict[str, float] = {}
    for sid in ids:
        pid = meta[sid]["person_id"]
        split = meta[sid]["split"]
        if pid in person_heights and split in by_split:
            by_split[split].append(sid)
            heights[sid] = person_heights[pid]
    by_split = limit_split_ids(by_split, limit_per_camera)
    kept = set(by_split["train"] + by_split["val"] + by_split["test"])
    heights = {sid: h for sid, h in heights.items() if sid in kept}
    if len(by_split["train"]) < 2 or len(by_split["test"]) < 2:
        return {"skipped": True, "split_counts": {k: len(v) for k, v in by_split.items()}}
    all_ids = by_split["train"] + by_split["val"] + by_split["test"]
    X = np.array([tabular[sid] for sid in all_ids], dtype=np.float32)
    C = np.stack([crops[sid] for sid in all_ids], axis=0).astype(np.float32, copy=False)
    X, C = apply_negative_control(variant, X, C)
    y = np.array([heights[sid] for sid in all_ids], dtype=np.float32)
    sid_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    train_idx = [sid_to_idx[s] for s in by_split["train"]]
    mu = X[train_idx].mean(0); sd = X[train_idx].std(0); sd[sd < 1e-6] = 1.0
    X = (X - mu) / sd
    bucket_weights = near_bucket_weights or parse_near_bucket_weights("")
    pairs = []
    tr = by_split["train"]
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            a, b = tr[i], tr[j]
            if heights[a] == heights[b]:
                continue
            pairs.append((sid_to_idx[a], sid_to_idx[b], 1.0 if heights[a] > heights[b] else -1.0, pair_weight_for_gap(abs(heights[a] - heights[b]), bucket_weights)))
    results = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = FusionRegressor(variant, tabular_dim=len(FEATURE_SCHEMA), hidden=hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        x = torch.tensor(X, device=device)
        c = torch.tensor(C, device=device)
        yy = torch.tensor(y, device=device)
        train_tensor = torch.tensor(train_idx, device=device, dtype=torch.long)
        pair_idx = torch.tensor([[a, b] for a, b, _, _ in pairs], device=device, dtype=torch.long) if pairs else None
        pair_sign = torch.tensor([s for _, _, s, _ in pairs], device=device, dtype=torch.float32) if pairs else None
        pair_weight = torch.tensor([w for _, _, _, w in pairs], device=device, dtype=torch.float32) if pairs else None
        best = None
        best_seed_state = None
        for ep in range(1, epochs + 1):
            model.train(); opt.zero_grad()
            pred = model(x, c)
            mse = torch.nn.functional.mse_loss(pred[train_tensor], yy[train_tensor]) / 1000.0
            if pair_idx is not None:
                diff = pred[pair_idx[:, 0]] - pred[pair_idx[:, 1]]
                raw_pair_loss = torch.nn.functional.softplus(-pair_sign * diff)
                pair_loss = (raw_pair_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)
            else:
                pair_loss = torch.tensor(0.0, device=device)
            loss = mse + float(pair_loss_weight) * pair_loss
            loss.backward(); opt.step()
            if ep % 20 == 0 or ep == epochs:
                model.eval()
                with torch.no_grad():
                    pp = model(x, c).detach().cpu().numpy()
                scores = {sid: float(pp[sid_to_idx[sid]]) for sid in all_ids}
                val_ids = by_split["val"] or by_split["train"]
                val_pa = pair_acc(val_ids, scores, heights)
                if best is None or val_pa > best["val_pairwise"]:
                    best = {"epoch": ep, "val_pairwise": val_pa, "scores": scores}
                    best_seed_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scores = best.pop("scores")
        item = {"seed": seed, "best_epoch": best["epoch"], "val_pairwise": best["val_pairwise"]}
        for split in ["train", "val", "test"]:
            sids = by_split[split]
            item[split] = {
                "n": len(sids),
                "pairwise_accuracy": pair_acc(sids, scores, heights),
                "pairwise_counts": pair_acc_counts(sids, scores, heights),
                "spearman": spearman([scores[s] for s in sids], [heights[s] for s in sids]) if len(sids) >= 2 else None,
                "bisect": bisect(sids, scores, heights),
            }
        item["_scores"] = scores
        item["_state"] = best_seed_state
        results.append(item)
    best = max(results, key=lambda r: (
        r["val_pairwise"],
        r["val"].get("spearman") or -1e9,
        r["val"].get("bisect", {}).get("hit_rate") or 0.0,
        -int(r["best_epoch"]),
        -int(r["seed"]),
    ))
    best_scores = best.pop("_scores")
    best_state = best.pop("_state")
    for r in results:
        r.pop("_scores", None); r.pop("_state", None)
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
        "result_label": variant_result_label(variant),
        "negative_control": variant in NEGATIVE_CONTROL_VARIANTS,
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
        "result_label": variant_result_label(variant),
        "negative_control": variant in NEGATIVE_CONTROL_VARIANTS,
        "pair_loss_weight": float(pair_loss_weight),
        "near_bucket_weights": bucket_weights,
        "checkpoint_path": str(ckpt_path),
        "seeds": results,
        "best": best,
        "video_level_ranking": video_ranking,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/zyding/data")
    ap.add_argument("--feature-root", default="/home/zyding/height/jianzhi_2511_sequence/features")
    ap.add_argument("--out-dir", default="/home/zyding/height/heightnet_repro/runs/home_data_heightmap_fusion")
    ap.add_argument("--variant", choices=VARIANTS, default="fused_concat")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--limit-per-camera", type=int, default=0, help="limit labeled sequences per split and camera for smoke tests")
    ap.add_argument("--person-split-json", default="", help="optional strict person-disjoint split JSON with train/val/test person ids")
    ap.add_argument("--pair-loss-weight", type=float, default=0.5, help="weight for pairwise ranking loss; default preserves the original training objective")
    ap.add_argument("--near-bucket-weights", default="", help="optional comma-separated height-gap pair weights, e.g. lt3=2,3to5=1.5,5to8=1.2,ge8=1")
    ap.add_argument("--target-cameras", nargs="+", default=[], help="optional camera ids to train, e.g. 3d5_330 2d5_330")
    args = ap.parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    person_heights = load_rank_csv(data_root / "label" / "rank.json")
    rows = []
    for split in ["train", "val", "test"]:
        rows.extend(load_split_csv(data_root / f"{split}.csv", split))
    source_row_count = len(rows)
    near_bucket_weights = parse_near_bucket_weights(args.near_bucket_weights)
    person_split = None
    split_counts = None
    overlap_counts = None
    if args.person_split_json:
        person_split = json.loads(Path(args.person_split_json).read_text(encoding="utf-8"))
        split_counts = person_split_counts(person_split)
        overlap_counts = person_overlap_counts(person_split)
        rows = apply_person_split(rows, person_split)
    rows = limit_rows_for_smoke(rows, args.limit_per_camera, person_heights)
    meta, tabular, crops, missing = build_sequence_table(rows, Path(args.feature_root))
    feature_payload = {
        "meta": meta,
        "features": tabular,
        "missing_npz": missing,
        "features_used": FEATURE_SCHEMA,
        "variant": args.variant,
        "result_label": variant_result_label(args.variant),
        "negative_control": args.variant in NEGATIVE_CONTROL_VARIANTS,
        "person_split_json": args.person_split_json or None,
        "person_split_counts": split_counts,
        "person_overlap_counts": overlap_counts,
        "primary_metric_definition": PRIMARY_METRIC_DEFINITION,
        "checkpoint_selection": "validation-only: val PairAcc, val Spearman, val bisect, earlier epoch, smaller seed",
        "pair_loss_weight": args.pair_loss_weight,
        "near_bucket_weights": near_bucket_weights,
    }
    (out_dir / f"sequence_features_{args.variant}.json").write_text(json.dumps(feature_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cameras = filter_target_cameras(sorted({m["camera_id"] for m in meta.values()}), args.target_cameras)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {
        "data_root": str(data_root),
        "feature_root": str(args.feature_root),
        "device": device,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "seeds": args.seeds,
        "variant": args.variant,
        "result_label": variant_result_label(args.variant),
        "negative_control": args.variant in NEGATIVE_CONTROL_VARIANTS,
        "feature_schema": FEATURE_SCHEMA,
        "limit_per_camera": args.limit_per_camera,
        "target_cameras": args.target_cameras,
        "person_split_json": args.person_split_json or None,
        "person_split_counts": split_counts,
        "person_overlap_counts": overlap_counts,
        "primary_metric_definition": PRIMARY_METRIC_DEFINITION,
        "pair_loss_weight": args.pair_loss_weight,
        "near_bucket_weights": near_bucket_weights,
        "n_rows": len(rows),
        "source_n_rows": source_row_count,
        "n_sequences_with_npz": len(meta),
        "missing_npz_count": len(missing),
        "n_persons_with_height": len(person_heights),
        "camera_results": {},
    }
    for cam in cameras:
        ids = [sid for sid, m in meta.items() if m["camera_id"] == cam]
        print(f"[CAM] {cam} n={len(ids)} variant={args.variant}", flush=True)
        results["camera_results"][cam] = train_camera(
            cam, ids, meta, tabular, crops, person_heights, device, args.epochs, args.seeds,
            args.hidden, args.variant, out_dir / "checkpoints", args.limit_per_camera,
            args.pair_loss_weight, near_bucket_weights,
        )
    result_path = out_dir / f"base_camera_heightmap_fusion_baseline_valonly_{args.variant}_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    top = []
    for cam, res in results["camera_results"].items():
        if res.get("skipped"):
            continue
        b = res["best"]
        top.append((b["test"]["pairwise_accuracy"], b["test"].get("spearman"), b["test"]["bisect"].get("hit_rate"), cam, b["seed"], b["best_epoch"], res["split_counts"], res.get("checkpoint_path")))
    print("TOP_TEST")
    def _top_sort_key(row):
        return tuple(-1e9 if v is None else v for v in row[:3]) + row[3:6]

    for row in sorted(top, key=_top_sort_key, reverse=True)[:20]:
        print(row, flush=True)
    print("[OUT]", result_path)


if __name__ == "__main__":
    main()
