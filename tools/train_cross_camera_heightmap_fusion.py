#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cross_camera_heightmap_fusion_core import (
    CAMERA_GEOMETRY_FEATURE_DIM,
    FEATURE_SCHEMA,
    CrossCameraFusionRanker,
    camera_geometry_features,
    correlation,
    filter_labeled_rows,
    filter_strict_no_train_overlap,
    load_height_labels,
    load_npz_frames,
    load_split_rows,
    rankdata,
    sample_cross_camera_triplet,
    sample_track_frame_pair,
    score_identity_consistency_loss,
)


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _distributed_context_from_env(device_arg: str, environ: dict[str, str] | None = None) -> DistributedContext:
    env = os.environ if environ is None else environ
    world_size = int(env.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(device_arg),
        )
    rank = int(env.get("RANK", "0"))
    local_rank = int(env.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistributedContext(
        distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def _setup_distributed(ctx: DistributedContext) -> None:
    if not ctx.distributed:
        return
    if ctx.device.type == "cuda":
        torch.cuda.set_device(ctx.local_rank)
    backend = "nccl" if ctx.device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _model_core(model):
    return model.module if hasattr(model, "module") else model


def _compare_encoded_for_training(model, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "module"):
        return model(op="compare_encoded", encoded_a=a, encoded_b=b)
    return model.compare_encoded(a, b)


def _score_identity_consistency_loss_for_training(
    model,
    embeddings: torch.Tensor,
    person_ids: list[str],
) -> tuple[torch.Tensor, int]:
    if hasattr(model, "module"):
        if embeddings.shape[0] != len(person_ids):
            raise ValueError(f"embeddings/person_ids length mismatch: {embeddings.shape[0]} vs {len(person_ids)}")
        scores = model(op="score", embeddings=embeddings)
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
    return score_identity_consistency_loss(model, embeddings, person_ids)


def _load_rows(data_root: Path, feature_root: Path, limit_per_split_camera: int) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        rows = load_split_rows(data_root / f"{split}.csv", feature_root, split)
        if limit_per_split_camera > 0:
            counts: dict[str, int] = defaultdict(int)
            limited: list[dict] = []
            for row in rows:
                if counts[row["camera_id"]] >= limit_per_split_camera:
                    continue
                counts[row["camera_id"]] += 1
                limited.append(row)
            rows = limited
        output[split] = rows
    return output


def _usable(rows: list[dict]) -> list[dict]:
    usable = []
    for row in rows:
        if not Path(row["npz_path"]).exists():
            continue
        try:
            load_npz_frames(row["npz_path"])
        except ValueError as exc:
            if "invalid valid_count=0" in str(exc):
                continue
            raise
        usable.append(row)
    return usable


def _limit_per_camera(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return rows
    counts: dict[str, int] = defaultdict(int)
    output = []
    for row in rows:
        if counts[row["camera_id"]] >= limit:
            continue
        counts[row["camera_id"]] += 1
        output.append(row)
    return output


def _apply_person_split(rows_by_split: dict[str, list[dict]], person_split: dict[str, list[str]]) -> dict[str, list[dict]]:
    source_rows = sum(rows_by_split.values(), [])
    output: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        people = set(person_split[split])
        seen: set[tuple[str, str]] = set()
        kept = []
        for row in source_rows:
            if row["person_id"] not in people:
                continue
            key = (str(row["person_id"]), str(row.get("npz_path") or row.get("video_path") or row.get("video_filename")))
            if key in seen:
                continue
            seen.add(key)
            kept.append(row)
        output[split] = kept
    return output


def _camera_vocab(rows: list[dict]) -> dict[str, int]:
    return {camera_id: idx for idx, camera_id in enumerate(sorted({row["camera_id"] for row in rows}))}


def _scaler(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = [load_npz_frames(row["npz_path"]).aggregate()[0] for row in rows]
    x = np.stack(features).astype(np.float32)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def _geometry_kwargs(args) -> dict:
    return {
        "image_width": args.geometry_image_width,
        "image_height": args.geometry_image_height,
        "pitch_deg": args.geometry_pitch_deg,
        "focal_px": args.geometry_focal_px if args.geometry_focal_px > 0 else None,
    }


def _sequence_tensors(rows: list[dict], camera_vocab: dict[str, int], mu: np.ndarray, sd: np.ndarray, device: torch.device, geometry_kwargs: dict | None = None):
    tabular, crops, camera_index, camera_height, camera_geometry = [], [], [], [], []
    for row in rows:
        frames = load_npz_frames(row["npz_path"])
        tab, crop = frames.aggregate()
        tabular.append((tab - mu) / sd)
        crops.append(crop)
        camera_index.append(camera_vocab[row["camera_id"]])
        camera_height.append(row["camera_height_m"])
        if geometry_kwargs is not None:
            camera_geometry.append(camera_geometry_features(tab, row["camera_id"], **geometry_kwargs))
    tensors = [
        torch.tensor(np.stack(tabular), dtype=torch.float32, device=device),
        torch.tensor(np.stack(crops), dtype=torch.float32, device=device),
        torch.tensor(camera_index, dtype=torch.long, device=device),
        torch.tensor(camera_height, dtype=torch.float32, device=device),
    ]
    if geometry_kwargs is not None:
        tensors.append(torch.tensor(np.stack(camera_geometry), dtype=torch.float32, device=device))
    return tuple(tensors)


def _frame_tensors(row: dict, frame_indices: list[int], camera_vocab: dict[str, int], mu: np.ndarray, sd: np.ndarray, device: torch.device, geometry_kwargs: dict | None = None):
    frames = load_npz_frames(row["npz_path"])
    tensors = [
        torch.tensor((frames.tabular[frame_indices] - mu) / sd, dtype=torch.float32, device=device),
        torch.tensor(frames.crops[frame_indices], dtype=torch.float32, device=device),
        torch.full((len(frame_indices),), camera_vocab[row["camera_id"]], dtype=torch.long, device=device),
        torch.full((len(frame_indices),), float(row["camera_height_m"]), dtype=torch.float32, device=device),
    ]
    if geometry_kwargs is not None:
        geometry = np.stack([camera_geometry_features(frames.tabular[index], row["camera_id"], **geometry_kwargs) for index in frame_indices])
        tensors.append(torch.tensor(geometry, dtype=torch.float32, device=device))
    return tuple(tensors)


def _pick_supervised_pairs(rows: list[dict], batch_size: int, rng: np.random.Generator) -> tuple[list[dict], list[dict], torch.Tensor]:
    left, right, labels = [], [], []
    while len(left) < batch_size:
        a, b = rng.choice(rows, size=2, replace=False)
        if a["person_id"] == b["person_id"] or a["height_cm"] == b["height_cm"]:
            continue
        left.append(a)
        right.append(b)
        labels.append(1.0 if a["height_cm"] > b["height_cm"] else 0.0)
    return left, right, torch.tensor(labels, dtype=torch.float32)


def _sample_identity_consistency_rows(rows: list[dict], group_count: int, rng: np.random.Generator) -> list[dict]:
    if group_count <= 0:
        return []
    by_person: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_person[str(row["person_id"])].append(row)
    eligible = [person_id for person_id, items in by_person.items() if len(items) >= 2]
    if not eligible:
        return []
    replace_people = group_count > len(eligible)
    chosen_people = rng.choice(eligible, size=group_count, replace=replace_people)
    sampled: list[dict] = []
    for person_id in chosen_people:
        person_id = str(person_id)
        items = by_person[person_id]
        indices = rng.choice(len(items), size=2, replace=False)
        sampled.extend([items[int(indices[0])], items[int(indices[1])]])
    return sampled


def _height_gap_bucket(gap_cm: float) -> str:
    return "lt3" if gap_cm < 3 else "3to5" if gap_cm < 5 else "5to8" if gap_cm < 8 else "ge8"


def _parse_bucket_weights(spec: str) -> dict[str, float]:
    weights = {"lt3": 1.0, "3to5": 1.0, "5to8": 1.0, "ge8": 1.0}
    if not spec:
        return weights
    for item in spec.split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        if not sep or key not in weights:
            raise ValueError(f"invalid bucket weight item={item!r}")
        weights[key] = float(value)
    return weights


def _pair_targets_and_weights(
    left: list[dict],
    right: list[dict],
    label_mode: str,
    soft_temperature_cm: float,
    bucket_weights: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if label_mode not in {"hard", "soft"}:
        raise ValueError(f"unknown pair label mode={label_mode!r}")
    if soft_temperature_cm <= 0:
        raise ValueError("soft pair temperature must be positive")
    diffs = torch.tensor([float(a["height_cm"]) - float(b["height_cm"]) for a, b in zip(left, right)], dtype=torch.float32)
    if label_mode == "soft":
        targets = torch.sigmoid(diffs / float(soft_temperature_cm))
    else:
        targets = (diffs > 0).to(torch.float32)
    weights = torch.tensor([bucket_weights[_height_gap_bucket(abs(float(diff)))] for diff in diffs], dtype=torch.float32)
    return targets, weights


def _weighted_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


class HeightRegressionHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings).squeeze(1)


def _height_regression_loss(
    head: HeightRegressionHead,
    embeddings: torch.Tensor,
    rows: list[dict],
    height_mu: float,
    height_sd: float,
    device: torch.device,
) -> torch.Tensor:
    targets = torch.tensor(
        [(float(row["height_cm"]) - float(height_mu)) / float(height_sd) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    return F.smooth_l1_loss(head(embeddings), targets)


def _encode_rows(model, rows, camera_vocab, mu, sd, device, geometry_kwargs: dict | None = None):
    tensors = _sequence_tensors(rows, camera_vocab, mu, sd, device, geometry_kwargs)
    return model(*tensors)


def _person_camera_records(model, rows, camera_vocab, mu, sd, device, geometry_kwargs: dict | None = None) -> list[dict]:
    model.eval()
    grouped: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    heights: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            encoded = _encode_rows(model, batch, camera_vocab, mu, sd, device, geometry_kwargs).cpu()
            for row, embedding in zip(batch, encoded):
                grouped[(row["person_id"], row["camera_id"])].append(embedding)
                heights[row["person_id"]] = float(row["height_cm"])
    return [
        {"person_id": pid, "camera_id": camera, "height_cm": heights[pid], "embedding": torch.stack(values).mean(0)}
        for (pid, camera), values in sorted(grouped.items())
    ]


def _bisect_hit_rate(records: list[dict], scores: dict[str, float]) -> float | None:
    people = sorted(scores, key=lambda pid: (-scores[pid], pid))
    if len(people) < 2:
        return None
    heights = {record["person_id"]: record["height_cm"] for record in records}
    truth = sorted(people, key=lambda pid: (-heights[pid], pid))
    midpoint = len(people) // 2
    pred_high = set(people[:midpoint])
    truth_high = set(truth[:midpoint])
    return float(sum((pid in pred_high) == (pid in truth_high) for pid in people) / len(people))


def evaluate(model, rows, camera_vocab, mu, sd, device, geometry_kwargs: dict | None = None) -> dict:
    records = _person_camera_records(model, rows, camera_vocab, mu, sd, device, geometry_kwargs)
    correct = {"all": 0, "same_camera": 0, "cross_camera": 0}
    total = {"all": 0, "same_camera": 0, "cross_camera": 0}
    buckets = {"lt3": [0, 0], "3to5": [0, 0], "5to8": [0, 0], "ge8": [0, 0]}
    person_embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)
    person_heights: dict[str, float] = {}
    for record in records:
        person_embeddings[record["person_id"]].append(record["embedding"])
        person_heights[record["person_id"]] = record["height_cm"]
    model.eval()
    with torch.no_grad():
        for i, left in enumerate(records):
            for right in records[i + 1:]:
                if left["person_id"] == right["person_id"] or left["height_cm"] == right["height_cm"]:
                    continue
                logit = float(model.compare_encoded(left["embedding"][None].to(device), right["embedding"][None].to(device)).item())
                expected = left["height_cm"] > right["height_cm"]
                ok = (logit > 0) == expected
                relation = "same_camera" if left["camera_id"] == right["camera_id"] else "cross_camera"
                for key in ("all", relation):
                    total[key] += 1
                    correct[key] += int(ok)
                gap = abs(left["height_cm"] - right["height_cm"])
                bucket = "lt3" if gap < 3 else "3to5" if gap < 5 else "5to8" if gap < 8 else "ge8"
                buckets[bucket][1] += 1
                buckets[bucket][0] += int(ok)
        people = sorted(person_embeddings)
        embeddings = torch.stack([torch.stack(person_embeddings[pid]).mean(0) for pid in people]).to(device)
        latent = model.score(embeddings).squeeze(1).cpu().tolist()
    heights = [person_heights[pid] for pid in people]
    scores = dict(zip(people, latent))
    return {
        "people": len(people),
        "person_camera_records": len(records),
        "all_pairwise_accuracy": correct["all"] / total["all"] if total["all"] else None,
        "same_camera_pairwise_accuracy": correct["same_camera"] / total["same_camera"] if total["same_camera"] else None,
        "cross_camera_pairwise_accuracy": correct["cross_camera"] / total["cross_camera"] if total["cross_camera"] else None,
        "pair_counts": total,
        "spearman": correlation(rankdata(latent), rankdata(heights)),
        "kendall_tau": float(kendalltau(latent, heights).statistic) if len(people) >= 2 else None,
        "bisect_hit_rate": _bisect_hit_rate(records, scores),
        "height_gap_bucket_pairwise_accuracy": {
            key: {"accuracy": values[0] / values[1] if values[1] else None, "pairs": values[1]}
            for key, values in buckets.items()
        },
    }


def train(args) -> dict:
    started = time.time()
    ctx = _distributed_context_from_env(args.device)
    _setup_distributed(ctx)
    device = ctx.device
    worker_seed = int(args.seed) + ctx.rank * 100003
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
    data_root, feature_root = Path(args.data_root), Path(args.feature_root)
    labels = load_height_labels(data_root / "label" / "rank.json")
    split_rows = _load_rows(data_root, feature_root, 0)
    person_split = None
    if args.person_split_json:
        person_split = json.loads(Path(args.person_split_json).read_text(encoding="utf-8"))
        split_rows = _apply_person_split(split_rows, person_split)
    labeled = {
        split: _usable(_limit_per_camera(filter_labeled_rows(rows, labels), args.limit_per_split_camera))
        for split, rows in split_rows.items()
    }
    if len({row["person_id"] for row in labeled["train"]}) < 2:
        raise ValueError("training requires at least two labeled people")
    sample_cross_camera_triplet(labeled["train"], np.random.default_rng(worker_seed))
    all_rows = sum(labeled.values(), [])
    camera_vocab = _camera_vocab(all_rows)
    mu, sd = _scaler(labeled["train"])
    height_values = np.asarray([float(row["height_cm"]) for row in labeled["train"]], dtype=np.float32)
    height_mu = float(height_values.mean())
    height_sd = float(height_values.std())
    if height_sd < 1e-6:
        height_sd = 1.0
    rng = np.random.default_rng(worker_seed)
    bucket_weights = _parse_bucket_weights(args.near_bucket_weights)
    geometry_kwargs = _geometry_kwargs(args) if args.use_camera_geometry else None
    camera_geometry_dim = CAMERA_GEOMETRY_FEATURE_DIM if args.use_camera_geometry else 0
    model = CrossCameraFusionRanker(camera_count=len(camera_vocab), embedding_dim=args.embedding_dim, camera_geometry_dim=camera_geometry_dim).to(device)
    height_head = HeightRegressionHead(args.embedding_dim).to(device) if args.lambda_height > 0 else None
    if ctx.distributed:
        ddp_kwargs = {
            "device_ids": [ctx.local_rank],
            "output_device": ctx.local_rank,
        } if device.type == "cuda" else {}
        model = DDP(model, find_unused_parameters=False, **ddp_kwargs)
        if hasattr(model, "_set_static_graph"):
            model._set_static_graph()
        if height_head is not None:
            height_head = DDP(height_head, find_unused_parameters=False, **ddp_kwargs)
            if hasattr(height_head, "_set_static_graph"):
                height_head._set_static_graph()
    model_ref = _model_core(model)
    height_head_ref = _model_core(height_head) if height_head is not None else None
    params = list(model.parameters()) + (list(height_head.parameters()) if height_head is not None else [])
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    best = None
    out_dir = Path(args.out_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ctx.distributed:
        dist.barrier()
    train_people = {row["person_id"] for row in labeled["train"]}
    strict_test = filter_strict_no_train_overlap(labeled["test"], train_people)
    overlap = {
        "train_val_people": sorted(train_people & {row["person_id"] for row in labeled["val"]}),
        "train_test_people": sorted(train_people & {row["person_id"] for row in labeled["test"]}),
    }
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if height_head is not None:
            height_head.train()
        sums = defaultdict(float)
        for _ in range(args.steps_per_epoch):
            left, right, _ = _pick_supervised_pairs(labeled["train"], args.batch_size, rng)
            targets, pair_weights = _pair_targets_and_weights(left, right, args.pair_label_mode, args.pair_soft_temperature_cm, bucket_weights)
            za = _encode_rows(model, left, camera_vocab, mu, sd, device, geometry_kwargs)
            zb = _encode_rows(model, right, camera_vocab, mu, sd, device, geometry_kwargs)
            pair_loss = _weighted_bce_with_logits(_compare_encoded_for_training(model, za, zb), targets.to(device), pair_weights.to(device))
            cross_terms = []
            for _ in range(args.consistency_batch_size):
                a, ap, b = sample_cross_camera_triplet(labeled["train"], rng)
                z_a = _encode_rows(model, [a], camera_vocab, mu, sd, device, geometry_kwargs)
                z_ap = _encode_rows(model, [ap], camera_vocab, mu, sd, device, geometry_kwargs)
                z_b = _encode_rows(model, [b], camera_vocab, mu, sd, device, geometry_kwargs)
                cross_terms.append(F.mse_loss(torch.sigmoid(_compare_encoded_for_training(model, z_a, z_b)), torch.sigmoid(_compare_encoded_for_training(model, z_ap, z_b))))
            cross_loss = torch.stack(cross_terms).mean()
            id_rows = _sample_identity_consistency_rows(labeled["train"], args.consistency_batch_size, rng)
            if id_rows and args.lambda_id_score > 0:
                id_embeddings = _encode_rows(model, id_rows, camera_vocab, mu, sd, device, geometry_kwargs)
                id_person_ids = [str(row["person_id"]) for row in id_rows]
                id_score_loss, id_group_count = _score_identity_consistency_loss_for_training(model, id_embeddings, id_person_ids)
            else:
                id_score_loss = pair_loss.new_tensor(0.0)
                id_group_count = 0
            track_terms = []
            track_rows = rng.choice(labeled["train"], size=args.track_batch_size, replace=True)
            for row in track_rows:
                frames = load_npz_frames(row["npz_path"])
                if frames.count < 2:
                    continue
                i, j = sample_track_frame_pair(frames.count, rng)
                tensors = _frame_tensors(row, [i, j], camera_vocab, mu, sd, device, geometry_kwargs)
                encoded = model(*tensors)
                track_terms.append(F.mse_loss(encoded[0], encoded[1]))
            track_loss = torch.stack(track_terms).mean() if track_terms else pair_loss.new_tensor(0.0)
            if height_head is not None:
                height_loss = _height_regression_loss(height_head, torch.cat([za, zb], dim=0), left + right, height_mu, height_sd, device)
            else:
                height_loss = pair_loss.new_tensor(0.0)
            loss = (
                pair_loss
                + args.lambda_cross * cross_loss
                + args.lambda_track * track_loss
                + args.lambda_id_score * id_score_loss
                + args.lambda_height * height_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums["loss"] += float(loss.item())
            sums["pair_loss"] += float(pair_loss.item())
            sums["cross_relative_consistency_loss"] += float(cross_loss.item())
            sums["id_score_consistency_loss"] += float(id_score_loss.item())
            sums["id_score_consistency_groups"] += float(id_group_count)
            sums["track_embedding_consistency_loss"] += float(track_loss.item())
            sums["height_loss"] += float(height_loss.item())
        if ctx.distributed:
            dist.barrier()
        if ctx.is_main:
            val_metrics = evaluate(model_ref, labeled["val"], camera_vocab, mu, sd, device, geometry_kwargs)
            item = {"epoch": epoch, **{key: value / args.steps_per_epoch for key, value in sums.items()}, "val": val_metrics}
            history.append(item)
            print("[EPOCH]", json.dumps(item, ensure_ascii=False), flush=True)
            key = (val_metrics["cross_camera_pairwise_accuracy"] or -1.0, val_metrics["all_pairwise_accuracy"] or -1.0, val_metrics["kendall_tau"] or -1.0)
            if best is None or key > best["key"]:
                best = {"key": key, "epoch": epoch, "metrics": val_metrics, "state": {k: v.detach().cpu() for k, v in model_ref.state_dict().items()}}
        if ctx.distributed:
            dist.barrier()
    if not ctx.is_main:
        return {"distributed_worker_rank": ctx.rank}
    if best is None:
        raise RuntimeError("no best checkpoint was selected")
    model_ref.load_state_dict(best["state"])
    legacy_test = evaluate(model_ref, labeled["test"], camera_vocab, mu, sd, device, geometry_kwargs)
    strict_metrics = evaluate(model_ref, strict_test, camera_vocab, mu, sd, device, geometry_kwargs) if strict_test else {"people": 0, "reason": "no strict test rows"}
    checkpoint = {
        "model_state_dict": best["state"],
        "camera_vocab": camera_vocab,
        "feature_schema": FEATURE_SCHEMA,
        "mu": mu,
        "sd": sd,
        "lambda_cross": args.lambda_cross,
        "lambda_track": args.lambda_track,
        "lambda_id_score": args.lambda_id_score,
        "lambda_height": args.lambda_height,
        "pair_label_mode": args.pair_label_mode,
        "pair_soft_temperature_cm": args.pair_soft_temperature_cm,
        "near_bucket_weights": bucket_weights,
        "height_mu": height_mu,
        "height_sd": height_sd,
        "height_head_state_dict": {k: v.detach().cpu() for k, v in height_head_ref.state_dict().items()} if height_head_ref is not None else None,
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "best_val_metrics": best["metrics"],
        "split_overlap_summary": overlap,
        "person_split_json": args.person_split_json,
        "embedding_dim": args.embedding_dim,
        "camera_geometry_dim": camera_geometry_dim,
        "camera_geometry": {
            "enabled": bool(args.use_camera_geometry),
            "feature_dim": camera_geometry_dim,
            **(_geometry_kwargs(args) if args.use_camera_geometry else {}),
        },
    }
    checkpoint_path = out_dir / "checkpoint_best.pt"
    torch.save(checkpoint, checkpoint_path)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "checkpoint": str(checkpoint_path),
        "camera_vocab": camera_vocab,
        "split_counts": {split: len(rows) for split, rows in labeled.items()},
        "split_overlap_summary": overlap,
        "person_split_json": args.person_split_json,
        "primary_metric_definition": "video-level evaluation: one usable video/NPZ is one independent sample; no person-camera or person-level aggregation for primary metrics",
        "best_epoch": best["epoch"],
        "best_val_metrics": best["metrics"],
        "non_primary_legacy_person_camera_aggregated_split": legacy_test,
        "non_primary_strict_no_train_overlap_person_camera_aggregated": strict_metrics,
        "duration_seconds": time.time() - started,
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/zyding/data")
    parser.add_argument("--feature-root", default="/home/zyding/height/jianzhi_2511_sequence/features")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--person-split-json", default="")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--consistency-batch-size", type=int, default=8)
    parser.add_argument("--track-batch-size", type=int, default=8)
    parser.add_argument("--limit-per-split-camera", type=int, default=0)
    parser.add_argument("--lambda-cross", type=float, default=0.2)
    parser.add_argument("--lambda-track", type=float, default=0.1)
    parser.add_argument("--lambda-id-score", type=float, default=0.0)
    parser.add_argument("--lambda-height", type=float, default=0.0)
    parser.add_argument("--pair-label-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--pair-soft-temperature-cm", type=float, default=3.0)
    parser.add_argument("--near-bucket-weights", default="lt3=1.0,3to5=1.0,5to8=1.0,ge8=1.0")
    parser.add_argument("--use-camera-geometry", action="store_true")
    parser.add_argument("--geometry-image-width", type=int, default=800)
    parser.add_argument("--geometry-image-height", type=int, default=600)
    parser.add_argument("--geometry-pitch-deg", type=float, default=18.0)
    parser.add_argument("--geometry-focal-px", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    try:
        payload = train(args)
    finally:
        _cleanup_distributed()
    if "distributed_worker_rank" in payload:
        return
    print("[OUT]", Path(args.out_dir) / "results.json")
    summary_keys = (
        "best_epoch",
        "best_val_metrics",
        "non_primary_legacy_person_camera_aggregated_split",
        "non_primary_strict_no_train_overlap_person_camera_aggregated",
    )
    print(json.dumps({key: payload[key] for key in summary_keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
