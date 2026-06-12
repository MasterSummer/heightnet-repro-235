from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = valid_mask > 0.5
    if m.sum() == 0:
        return pred.new_tensor(0.0)
    err2 = (pred - target) ** 2
    num = err2[m].sum()
    den = m.float().sum() + eps
    return torch.sqrt(num / den + eps)


def masked_avg_pool(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = (mask > 0.5).float()
    cnt = m.flatten(1).sum(dim=1)
    val = (x * m).flatten(1).sum(dim=1)
    avg = val / (cnt + eps)
    ratio = cnt / float(x.shape[-2] * x.shape[-1])
    return avg, cnt, ratio


def person_mask_is_valid(
    mask: torch.Tensor,
    min_valid_pixels: int,
    min_valid_ratio: float,
) -> torch.Tensor:
    m = (mask > 0.5).float()
    cnt = m.flatten(1).sum(dim=1)
    ratio = cnt / float(mask.shape[-2] * mask.shape[-1])
    return (cnt >= float(min_valid_pixels)) & (ratio >= float(min_valid_ratio))


def build_rank_pairs(
    person_ids: List[str],
    camera_ids: List[str],
    person_masks: torch.Tensor,
    pairwise_labels: Dict[str, Dict[Tuple[str, str], int]],
    min_valid_pixels: int,
    min_valid_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_person = person_mask_is_valid(person_masks, min_valid_pixels=min_valid_pixels, min_valid_ratio=min_valid_ratio)
    idx_i: List[int] = []
    idx_j: List[int] = []
    y: List[float] = []

    n = len(person_ids)
    for i in range(n):
        if not bool(valid_person[i].item()):
            continue
        for j in range(i + 1, n):
            if camera_ids[i] != camera_ids[j]:
                continue
            if not bool(valid_person[j].item()):
                continue
            cam = camera_ids[i]
            labels_cam = pairwise_labels.get(cam, {})
            key = (person_ids[i], person_ids[j])
            if key not in labels_cam:
                continue
            idx_i.append(i)
            idx_j.append(j)
            y.append(float(labels_cam[key]))

    if not y:
        z = torch.empty((0,), dtype=torch.long, device=device)
        return z, z, torch.empty((0,), dtype=torch.float32, device=device)

    return (
        torch.tensor(idx_i, dtype=torch.long, device=device),
        torch.tensor(idx_j, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


class HeightNetLoss(nn.Module):
    def __init__(
        self,
        lambda_rmse: float,
        lambda_rank: float,
        lambda_cons: float,
        eps: float,
        min_valid_pixels: int,
        min_valid_ratio: float,
        consistency_mode: str = "map",
    ) -> None:
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_rank = lambda_rank
        self.lambda_cons = lambda_cons
        self.eps = eps
        self.min_valid_pixels = int(min_valid_pixels)
        self.min_valid_ratio = float(min_valid_ratio)
        self.consistency_mode = str(consistency_mode).lower()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    @staticmethod
    def load_pairwise_labels(path: str) -> Dict[str, Dict[Tuple[str, str], int]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(dict)
        for item in data:
            cam = str(item["camera"])
            i = str(item["id_i"])
            j = str(item["id_j"])
            y = int(item["y"])
            labels[cam][(i, j)] = y
            labels[cam][(j, i)] = 1 - y
        return dict(labels)

    def consistency_loss(
        self,
        pred_t: torch.Tensor,
        pred_t1: torch.Tensor,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if self.consistency_mode == "scalar":
            return self._scalar_consistency_loss(pred_t, pred_t1, mask_t, mask_t1)
        if self.consistency_mode == "map":
            return self._map_consistency_loss(pred_t, pred_t1, mask_t, mask_t1)
        raise ValueError(f"unsupported consistency_mode: {self.consistency_mode}")

    def _scalar_consistency_loss(
        self,
        pred_t: torch.Tensor,
        pred_t1: torch.Tensor,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        s_t, cnt_t, ratio_t = masked_avg_pool(pred_t, mask_t, eps=self.eps)
        s_t1, cnt_t1, ratio_t1 = masked_avg_pool(pred_t1, mask_t1, eps=self.eps)

        valid = (
            (cnt_t >= self.min_valid_pixels)
            & (cnt_t1 >= self.min_valid_pixels)
            & (ratio_t >= self.min_valid_ratio)
            & (ratio_t1 >= self.min_valid_ratio)
        )
        if valid.sum().item() == 0:
            return pred_t.new_tensor(0.0), 0

        raw = self.smooth_l1(s_t, s_t1)
        return raw[valid].mean(), int(valid.sum().item())

    def _map_consistency_loss(
        self,
        pred_t: torch.Tensor,
        pred_t1: torch.Tensor,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        valid_t = person_mask_is_valid(mask_t, self.min_valid_pixels, self.min_valid_ratio)
        valid_t1 = person_mask_is_valid(mask_t1, self.min_valid_pixels, self.min_valid_ratio)
        valid = valid_t & valid_t1
        if valid.sum().item() == 0:
            return pred_t.new_tensor(0.0), 0

        inter_mask = ((mask_t > 0.5) & (mask_t1 > 0.5)).float()
        cnt = inter_mask.flatten(1).sum(dim=1)
        ratio = cnt / float(mask_t.shape[-2] * mask_t.shape[-1])
        valid = valid & (cnt >= float(self.min_valid_pixels)) & (ratio >= float(self.min_valid_ratio))
        if valid.sum().item() == 0:
            return pred_t.new_tensor(0.0), 0

        raw = self.smooth_l1(pred_t, pred_t1) * inter_mask
        num = raw.flatten(1).sum(dim=1)
        per_sample = num / (cnt + self.eps)
        return per_sample[valid].mean(), int(valid.sum().item())

    def rank_loss(
        self,
        pair_logit: torch.Tensor | None,
        pair_label: torch.Tensor | None,
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if pair_logit is None or pair_label is None or pair_logit.numel() == 0:
            return ref.new_tensor(0.0), 0
        return self.bce(pair_logit, pair_label.float()), int(pair_label.numel())

    def forward(
        self,
        pred_height: torch.Tensor,
        target_height: torch.Tensor,
        valid_mask: torch.Tensor,
        pred_pair: torch.Tensor,
        person_mask: torch.Tensor,
        person_mask_pair: torch.Tensor,
        pair_logit: torch.Tensor | None,
        pair_label: torch.Tensor | None,
    ) -> dict:
        l_rmse = masked_rmse(pred_height, target_height, valid_mask, eps=self.eps)
        l_cons, cons_valid_pairs = self.consistency_loss(pred_height, pred_pair, person_mask, person_mask_pair)
        l_rank, rank_pairs = self.rank_loss(pair_logit, pair_label, ref=pred_height)

        total = self.lambda_rmse * l_rmse + self.lambda_rank * l_rank + self.lambda_cons * l_cons
        return {
            "total": total,
            "rmse": l_rmse,
            "rank": l_rank,
            "consistency": l_cons,
            "cons_valid_pairs": cons_valid_pairs,
            "rank_pairs": rank_pairs,
        }
