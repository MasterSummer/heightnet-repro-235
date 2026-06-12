from __future__ import annotations

from typing import Callable, Dict, List

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = mask > 0.5
    if m.sum() == 0:
        return pred.new_tensor(0.0)
    err = pred[m] - target[m]
    return torch.sqrt(torch.mean(err * err) + eps)


def delta_threshold(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, thresh: float) -> torch.Tensor:
    m = mask > 0.5
    if m.sum() == 0:
        return pred.new_tensor(0.0)
    p = torch.clamp(pred[m], min=1e-6)
    t = torch.clamp(target[m], min=1e-6)
    ratio = torch.maximum(p / t, t / p)
    return (ratio < thresh).float().mean()


def pairwise_accuracy_from_ranked_lists(
    pred_scores: Dict[str, float],
    gt_ranking: List[str],
) -> float:
    """
    Compute pairwise ranking accuracy from a global ranking list.
    For any i<j in gt_ranking, prediction is correct iff score(i) > score(j).
    """
    ids = [x for x in gt_ranking if x in pred_scores]
    if len(ids) < 2:
        return 0.0
    correct = 0
    total = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            total += 1
            if pred_scores[ids[i]] > pred_scores[ids[j]]:
                correct += 1
    return float(correct / total) if total > 0 else 0.0


def binary_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    return float(sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1))


def binary_f1(y_true: List[int], y_pred: List[int], eps: float = 1e-6) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2.0 * precision * recall / (precision + recall + eps)


def binary_auc(y_true: List[int], y_prob: List[float], eps: float = 1e-6) -> float:
    pos = [p for t, p in zip(y_true, y_prob) if t == 1]
    neg = [p for t, p in zip(y_true, y_prob) if t == 0]
    if not pos or not neg:
        return 0.0
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif abs(p - n) <= eps:
                wins += 0.5
    return wins / float(total)


def _quicksort_indices(
    indices: List[int],
    compare_fn: Callable[[int, int], float],
) -> List[int]:
    if len(indices) <= 1:
        return indices
    pivot_pos = len(indices) // 2
    pivot = indices[pivot_pos]
    left: List[int] = []
    right: List[int] = []
    for idx in indices[:pivot_pos] + indices[pivot_pos + 1 :]:
        prob = compare_fn(idx, pivot)
        if prob >= 0.5:
            left.append(idx)
        else:
            right.append(idx)
    return _quicksort_indices(left, compare_fn) + [pivot] + _quicksort_indices(right, compare_fn)


def same_camera_quicksort_metrics(
    records: List[dict],
    pairwise_labels: Dict[str, Dict[tuple[str, str], int]],
    prob_fn: Callable[[int, int], float],
) -> dict:
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []
    n_comparisons = 0
    rankings: Dict[str, List[int]] = {}
    by_camera: Dict[str, List[int]] = {}
    for idx, rec in enumerate(records):
        by_camera.setdefault(rec["camera_id"], []).append(idx)

    for cam, indices in by_camera.items():
        cache: Dict[tuple[int, int], float] = {}

        def _compare(i: int, j: int) -> float:
            nonlocal n_comparisons
            key_ij = (i, j)
            if key_ij in cache:
                return cache[key_ij]
            prob = float(prob_fn(i, j))
            cache[(i, j)] = prob
            cache[(j, i)] = 1.0 - prob
            n_comparisons += 1

            y = pairwise_labels.get(cam, {}).get((records[i]["person_id"], records[j]["person_id"]))
            if y is not None:
                pred = 1 if prob >= 0.5 else 0
                all_true.append(int(y))
                all_pred.append(pred)
                all_prob.append(prob)
            return prob

        rankings[cam] = _quicksort_indices(indices, _compare)

    return {
        "pairwise_accuracy": binary_accuracy(all_true, all_pred),
        "auc": binary_auc(all_true, all_prob),
        "f1": binary_f1(all_true, all_pred),
        "n_pairs_eval": len(all_true),
        "n_comparisons": n_comparisons,
        "rankings": rankings,
    }
