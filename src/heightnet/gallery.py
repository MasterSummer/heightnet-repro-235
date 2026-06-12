from __future__ import annotations

import random
from typing import Callable, Dict, List

import torch

from .datasets import HeightDataset
from .losses import masked_avg_pool, person_mask_is_valid
from .metrics import binary_accuracy, binary_auc, binary_f1


def uniform_sample_indices(count: int, max_items: int) -> List[int]:
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []
    if max_items <= 0 or max_items >= count:
        return list(range(count))
    if max_items == 1:
        return [0]
    out = []
    for k in range(max_items):
        pos = round(((count - 1) * k) / float(max_items - 1))
        out.append(int(pos))
    deduped = []
    for idx in out:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    return deduped


def uniform_frame_indices(frame_start: int, frame_end: int, num_frames: int) -> List[int]:
    if frame_end < frame_start:
        raise ValueError(f"invalid frame range: [{frame_start}, {frame_end}]")
    if num_frames <= 1 or frame_start == frame_end:
        return [frame_start]
    span = frame_end - frame_start
    steps = min(num_frames, span + 1)
    out = []
    for k in range(steps):
        pos = frame_start + round((span * k) / float(steps - 1))
        out.append(int(pos))
    deduped = []
    for idx in out:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    return deduped


def select_video_frame_indices(
    dataset: HeightDataset,
    row,
    num_frames: int,
) -> List[int]:
    valid_frames = dataset._load_valid_frames(row)
    if valid_frames:
        picked = uniform_sample_indices(len(valid_frames), max(num_frames, 1))
        return [int(valid_frames[idx]) for idx in picked]
    return uniform_frame_indices(row.frame_start, row.frame_end, num_frames)


def aggregate_video_feature(frame_features: List[torch.Tensor]) -> torch.Tensor:
    if not frame_features:
        raise ValueError("frame_features must not be empty")
    stacked = torch.stack(frame_features, dim=0)
    return stacked.mean(dim=0)


def aggregate_video_scalar(frame_scores: List[float]) -> float:
    if not frame_scores:
        raise ValueError("frame_scores must not be empty")
    return float(sum(float(v) for v in frame_scores) / len(frame_scores))


def foreground_height_scores(pred_height: torch.Tensor, person_mask: torch.Tensor) -> Dict[str, float]:
    if pred_height.ndim != 4 or person_mask.ndim != 4:
        raise ValueError("pred_height and person_mask must be BCHW tensors")
    avg, cnt, _ = masked_avg_pool(pred_height, person_mask)
    if int(cnt[0].item()) <= 0:
        raise ValueError("person_mask has no valid foreground pixels")
    values = pred_height[person_mask > 0.5]
    return {
        "masked_avg": float(avg[0].item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def cross_split_pairwise_metrics(
    query_records: List[dict],
    gallery_records: List[dict],
    pairwise_labels: Dict[str, Dict[tuple[str, str], int]],
    prob_fn: Callable[[int, int], float],
) -> dict:
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []
    n_comparisons = 0

    for i, query in enumerate(query_records):
        cam = query["camera_id"]
        labels_cam = pairwise_labels.get(cam, {})
        for j, gallery in enumerate(gallery_records):
            if gallery["camera_id"] != cam:
                continue
            y = labels_cam.get((query["person_id"], gallery["person_id"]))
            if y is None:
                continue
            prob = float(prob_fn(i, j))
            pred = 1 if prob >= 0.5 else 0
            all_true.append(int(y))
            all_pred.append(pred)
            all_prob.append(prob)
            n_comparisons += 1

    return {
        "pairwise_accuracy": binary_accuracy(all_true, all_pred),
        "auc": binary_auc(all_true, all_prob),
        "f1": binary_f1(all_true, all_pred),
        "n_pairs_eval": len(all_true),
        "n_comparisons": n_comparisons,
    }


def build_scalar_prob_fn(
    query_records: List[dict],
    gallery_records: List[dict],
    score_key: str,
) -> Callable[[int, int], float]:
    def _prob_fn(i: int, j: int) -> float:
        q = float(query_records[i][score_key])
        g = float(gallery_records[j][score_key])
        diff = torch.tensor(q - g, dtype=torch.float32)
        return float(torch.sigmoid(diff).item())

    return _prob_fn


def sample_frame_row_indices_per_person(
    dataset: HeightDataset,
    max_per_person: int,
    seed: int,
) -> List[int]:
    by_person: Dict[str, List[int]] = {}
    for idx, row in enumerate(dataset.rows):
        by_person.setdefault(row.person_id, []).append(idx)

    rnd = random.Random(seed)
    picked: List[int] = []
    for person_id, indices in sorted(by_person.items()):
        choices = list(indices)
        rnd.shuffle(choices)
        picked.extend(sorted(choices[: max(max_per_person, 1)]))
    return picked


@torch.no_grad()
def build_frame_feature_gallery(
    model,
    dataset: HeightDataset,
    device: torch.device,
    segmenter,
    row_indices: List[int],
    min_valid_pixels: int,
    min_valid_ratio: float,
) -> List[dict]:
    model_ref = model.module if hasattr(model, "module") else model
    was_training = model_ref.training
    model_ref.eval()
    records: List[dict] = []

    for row_idx in row_indices:
        sample = dataset[row_idx]
        image = sample["image"].unsqueeze(0).to(device)
        image_raw = sample["image_raw"].unsqueeze(0)
        pred = model_ref(image)["pred_height_map"]
        person_mask, person_bbox = segmenter.infer_batch_regions(image_raw, device)
        keep = person_mask_is_valid(
            person_mask,
            min_valid_pixels=min_valid_pixels,
            min_valid_ratio=min_valid_ratio,
        )
        if not bool(keep[0].item()):
            continue
        feat = model_ref.encode_person(pred, person_mask, person_bbox)[0].detach().cpu()
        records.append(
            {
                "sequence_id": sample["sequence_id"],
                "person_id": sample["person_id"],
                "camera_id": sample["camera_id"],
                "frame_idx": int(sample["frame_idx"]),
                "feature": feat,
            }
        )

    model_ref.train(was_training)
    return records


@torch.no_grad()
def build_video_feature_gallery(
    model,
    dataset: HeightDataset,
    device: torch.device,
    segmenter,
    num_frames: int,
    min_valid_pixels: int,
    min_valid_ratio: float,
) -> List[dict]:
    model_ref = model.module if hasattr(model, "module") else model
    was_training = model_ref.training
    model_ref.eval()
    records: List[dict] = []

    for row in dataset.rows:
        frame_features: List[torch.Tensor] = []
        sampled_frames: List[int] = []
        frame_indices = select_video_frame_indices(dataset, row, num_frames)
        for frame_idx in frame_indices:
            sample, actual_frame_idx = dataset._build_sample(row, frame_idx)
            image = sample["image"].unsqueeze(0).to(device)
            image_raw = sample["image_raw"].unsqueeze(0)
            pred = model_ref(image)["pred_height_map"]
            person_mask, person_bbox = segmenter.infer_batch_regions(image_raw, device)
            keep = person_mask_is_valid(
                person_mask,
                min_valid_pixels=min_valid_pixels,
                min_valid_ratio=min_valid_ratio,
            )
            if not bool(keep[0].item()):
                continue
            feat = model_ref.encode_person(pred, person_mask, person_bbox)[0].detach().cpu()
            frame_features.append(feat)
            sampled_frames.append(int(actual_frame_idx))
        if not frame_features:
            continue
        records.append(
            {
                "sequence_id": row.sequence_id,
                "person_id": row.person_id,
                "camera_id": row.camera_id,
                "sampled_frames": sampled_frames,
                "feature": aggregate_video_feature(frame_features),
            }
        )

    model_ref.train(was_training)
    return records


@torch.no_grad()
def build_video_stat_gallery(
    model,
    dataset: HeightDataset,
    device: torch.device,
    segmenter,
    num_frames: int,
    min_valid_pixels: int,
    min_valid_ratio: float,
) -> List[dict]:
    model_ref = model.module if hasattr(model, "module") else model
    was_training = model_ref.training
    model_ref.eval()
    records: List[dict] = []

    for row in dataset.rows:
        score_lists: Dict[str, List[float]] = {
            "masked_avg": [],
            "p95": [],
            "p99": [],
            "max": [],
        }
        sampled_frames: List[int] = []
        frame_indices = select_video_frame_indices(dataset, row, num_frames)
        for frame_idx in frame_indices:
            sample, actual_frame_idx = dataset._build_sample(row, frame_idx)
            image = sample["image"].unsqueeze(0).to(device)
            image_raw = sample["image_raw"].unsqueeze(0)
            pred = model_ref(image)["pred_height_map"]
            person_mask, _ = segmenter.infer_batch_regions(image_raw, device)
            keep = person_mask_is_valid(
                person_mask,
                min_valid_pixels=min_valid_pixels,
                min_valid_ratio=min_valid_ratio,
            )
            if not bool(keep[0].item()):
                continue
            scores = foreground_height_scores(pred, person_mask)
            for key, value in scores.items():
                score_lists[key].append(value)
            sampled_frames.append(int(actual_frame_idx))
        if not sampled_frames:
            continue
        record = {
            "sequence_id": row.sequence_id,
            "person_id": row.person_id,
            "camera_id": row.camera_id,
            "sampled_frames": sampled_frames,
        }
        for key, values in score_lists.items():
            record[key] = aggregate_video_scalar(values)
        records.append(record)

    model_ref.train(was_training)
    return records


def save_video_feature_gallery(path: str, records: List[dict], num_frames: int) -> None:
    payload = {
        "num_records": len(records),
        "num_frames": int(num_frames),
        "records": records,
    }
    torch.save(payload, path)


def load_video_feature_gallery(path: str) -> List[dict]:
    payload = torch.load(path, map_location="cpu")
    return list(payload.get("records", []))
