from __future__ import annotations

from typing import List

import torch


def _similarity(query_feature: torch.Tensor, gallery_feature: torch.Tensor) -> float:
    q = query_feature.float().view(-1)
    g = gallery_feature.float().view(-1)
    return float(torch.dot(q, g).item())


def split_rank_bands(ranking: List[dict]) -> dict:
    n = len(ranking)
    if n == 0:
        return {"upper": [], "middle": [], "lower": []}

    upper_n = max(1, n // 4)
    lower_n = max(1, n // 4)
    if upper_n + lower_n > n:
        lower_n = max(0, n - upper_n)
    middle_start = upper_n
    middle_end = n - lower_n
    return {
        "upper": ranking[:upper_n],
        "middle": ranking[middle_start:middle_end],
        "lower": ranking[middle_end:],
    }


def build_ranked_candidates(query_records: List[dict], gallery_records: List[dict]) -> List[dict]:
    outputs = []
    for query in query_records:
        candidates = []
        for gallery in gallery_records:
            if gallery.get("camera_id") != query.get("camera_id"):
                continue
            candidates.append(
                {
                    "sequence_id": gallery.get("sequence_id"),
                    "person_id": gallery.get("person_id"),
                    "camera_id": gallery.get("camera_id"),
                    "frame_idx": gallery.get("frame_idx"),
                    "sampled_frames": gallery.get("sampled_frames"),
                    "_similarity": _similarity(query["feature"], gallery["feature"]),
                }
            )
        candidates.sort(key=lambda item: item["_similarity"], reverse=True)
        ranking = []
        for item in candidates:
            clean = dict(item)
            clean.pop("_similarity", None)
            ranking.append(clean)
        outputs.append(
            {
                "query_sequence_id": query.get("sequence_id"),
                "query_person_id": query.get("person_id"),
                "camera_id": query.get("camera_id"),
                "query_frame_idx": query.get("frame_idx"),
                "ranking": ranking,
                "bands": split_rank_bands(ranking),
            }
        )
    return outputs
