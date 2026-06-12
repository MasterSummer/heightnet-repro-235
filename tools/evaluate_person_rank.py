from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.abspath(os.path.dirname(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "..", "src"),
    os.path.join(_HERE, "src"),
]
for _path in _CANDIDATES:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.model import HeightNetTiny
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir


def _torch_load_compat(path: str, map_location: str | torch.device, *, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _require_file(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def _load_rank_labels(rank_dir: str) -> dict[str, list[str]]:
    rank_map: dict[str, list[str]] = {}
    if not rank_dir:
        return rank_map
    for name in os.listdir(rank_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(rank_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            cam = obj.get("camera")
            ranking = obj.get("ranking", [])
            if cam and isinstance(ranking, list):
                rank_map[cam] = [str(x) for x in ranking]
    return rank_map


def _extract_scores(
    loader: DataLoader,
    model: HeightNetTiny,
    segmenter: PersonSegmenter,
    device: torch.device,
) -> dict[str, dict[str, list[float]]]:
    scores_by_camera: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            pred = model(image)["pred_height_map"]
            person_mask = segmenter.infer_batch(batch["image_raw"], device)

            cam = batch.get("camera_id", ["unknown_camera"])[0]
            pid = batch.get("person_id", ["unknown_person"])[0]
            valid = person_mask > 0.5
            if not valid.any():
                continue
            score = (pred * valid.float()).sum() / torch.clamp(valid.float().sum(), min=1.0)
            scores_by_camera[cam][pid].append(float(score.item()))
    return scores_by_camera


def _sample_person_indices(ds: HeightDataset, per_person: int, seed: int) -> List[int]:
    by_person: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(ds.rows):
        by_person[row.person_id].append(idx)
    rng = random.Random(seed)
    chosen: List[int] = []
    for person_id in sorted(by_person):
        indices = list(by_person[person_id])
        if len(indices) <= per_person:
            chosen.extend(indices)
            continue
        chosen.extend(sorted(rng.sample(indices, per_person)))
    return chosen


def _split_rank_bands_simple(person_ids: List[str]) -> dict[str, List[str]]:
    n = len(person_ids)
    if n == 0:
        return {"upper": [], "middle": [], "lower": []}
    upper_n = max(1, n // 4)
    lower_n = max(1, n // 4)
    if upper_n + lower_n > n:
        lower_n = max(0, n - upper_n)
    middle_start = upper_n
    middle_end = n - lower_n
    return {
        "upper": person_ids[:upper_n],
        "middle": person_ids[middle_start:middle_end],
        "lower": person_ids[middle_end:],
    }


def _avg_scores(d: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cam, person_scores in d.items():
        out[cam] = {pid: float(sum(v) / max(len(v), 1)) for pid, v in person_scores.items()}
    return out


def _pairwise_count_and_acc(pred_scores: Dict[str, float], gt_ranking: List[str]) -> tuple[int, int, float]:
    ids = [x for x in gt_ranking if x in pred_scores]
    correct = 0
    total = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            total += 1
            if pred_scores[ids[i]] > pred_scores[ids[j]]:
                correct += 1
    acc = float(correct / total) if total > 0 else 0.0
    return correct, total, acc


def _band_map_from_ranking(person_ids: List[str]) -> dict[str, str]:
    bands = _split_rank_bands_simple(person_ids)
    out: dict[str, str] = {}
    for band_name, ids in bands.items():
        for pid in ids:
            out[str(pid)] = band_name
    return out


def _band_accuracy(pred_ranked_ids: List[str], gt_ranked_ids: List[str]) -> tuple[int, int, float]:
    common = [pid for pid in gt_ranked_ids if pid in set(pred_ranked_ids)]
    pred_filtered = [pid for pid in pred_ranked_ids if pid in set(common)]
    gt_filtered = [pid for pid in gt_ranked_ids if pid in set(common)]
    if not common:
        return 0, 0, 0.0
    pred_bands = _band_map_from_ranking(pred_filtered)
    gt_bands = _band_map_from_ranking(gt_filtered)
    correct = sum(1 for pid in common if pred_bands.get(pid) == gt_bands.get(pid))
    total = len(common)
    return correct, total, float(correct / total) if total > 0 else 0.0


def _spearman(pred_ranked_ids: List[str], gt_ranked_ids: List[str]) -> float | None:
    common = [pid for pid in gt_ranked_ids if pid in set(pred_ranked_ids)]
    n = len(common)
    if n < 2:
        return None
    pred_pos = {pid: idx + 1 for idx, pid in enumerate(pred_ranked_ids) if pid in set(common)}
    gt_pos = {pid: idx + 1 for idx, pid in enumerate(gt_ranked_ids) if pid in set(common)}
    d2 = sum((pred_pos[pid] - gt_pos[pid]) ** 2 for pid in common)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate person-level merged ranking against rank labels.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rank-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _require_file(cfg.paths.train_manifest, "train_manifest")
    _require_file(cfg.paths.test_manifest, "test_manifest")
    _require_file(args.checkpoint, "checkpoint")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_ds = HeightDataset(
        cfg.paths.train_manifest,
        tuple(cfg.data.image_size),
        normalize_rgb=cfg.data.normalize_rgb,
        use_pair_consistency=False,
        train_mode=False,
    )
    test_ds = HeightDataset(
        cfg.paths.test_manifest,
        tuple(cfg.data.image_size),
        normalize_rgb=cfg.data.normalize_rgb,
        use_pair_consistency=False,
        train_mode=False,
    )
    frames_per_person = int(getattr(cfg.eval, "frames_per_person_eval", 10))
    frame_eval_seed = int(getattr(cfg.eval, "frame_eval_seed", 42))
    eval_num_workers = min(int(cfg.train.num_workers), 4)

    train_indices = _sample_person_indices(train_ds, frames_per_person, frame_eval_seed)
    test_indices = _sample_person_indices(test_ds, frames_per_person, frame_eval_seed + 1)
    train_loader = DataLoader(Subset(train_ds, train_indices), batch_size=1, shuffle=False, num_workers=eval_num_workers)
    test_loader = DataLoader(Subset(test_ds, test_indices), batch_size=1, shuffle=False, num_workers=eval_num_workers)

    model = HeightNetTiny(
        base_channels=cfg.model.base_channels,
        comparator_channels=cfg.model.comparator_channels,
        comparator_type=cfg.model.comparator_type,
        comparator_layers=cfg.model.comparator_layers,
        comparator_num_heads=cfg.model.comparator_num_heads,
        comparator_patch_size=cfg.model.comparator_patch_size,
        person_region_mode=cfg.model.person_region_mode,
        bbox_expand_ratio=cfg.model.bbox_expand_ratio,
    ).to(device)
    ckpt = _torch_load_compat(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if not cfg.runtime_seg.enabled:
        raise RuntimeError("runtime_seg.enabled must be true")
    segmenter = PersonSegmenter(
        model_path=cfg.runtime_seg.model_path,
        conf=cfg.runtime_seg.conf,
        iou=cfg.runtime_seg.iou,
        imgsz=cfg.runtime_seg.imgsz,
        strict_native=cfg.runtime_seg.strict_native,
    )

    train_scores = _avg_scores(_extract_scores(train_loader, model, segmenter, device))
    test_scores = _avg_scores(_extract_scores(test_loader, model, segmenter, device))
    rank_map = _load_rank_labels(args.rank_dir)

    result = {"config": {
        "config_path": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "train_manifest": cfg.paths.train_manifest,
        "test_manifest": cfg.paths.test_manifest,
        "rank_dir": os.path.abspath(args.rank_dir),
        "frames_per_person_eval": frames_per_person,
        "frame_eval_seed": frame_eval_seed,
    }, "summary": {}, "cameras": {}}

    pair_accs: List[float] = []
    spearmans: List[float] = []
    band_accs: List[float] = []
    total_pair_correct = 0
    total_pair_count = 0
    total_band_correct = 0
    total_band_count = 0

    cams = sorted(set(train_scores.keys()) | set(test_scores.keys()))
    for cam in cams:
        tr = train_scores.get(cam, {})
        te = test_scores.get(cam, {})
        merged_scores = {}
        merged_scores.update(tr)
        merged_scores.update(te)
        merged_ranking = [{"person_id": pid, "score": float(score), "split": ("train" if pid in tr else "test")}
                          for pid, score in merged_scores.items()]
        merged_ranking.sort(key=lambda x: x["score"], reverse=True)
        pred_ranked_ids = [x["person_id"] for x in merged_ranking]
        gt_ranked_ids = rank_map.get(cam, [])

        pair_correct, pair_total, pair_acc = _pairwise_count_and_acc(merged_scores, gt_ranked_ids)
        band_correct, band_total, band_acc = _band_accuracy(pred_ranked_ids, gt_ranked_ids)
        spearman = _spearman(pred_ranked_ids, gt_ranked_ids)

        result["cameras"][cam] = {
            "n_train_person": len(tr),
            "n_test_person": len(te),
            "n_merged_person": len(merged_scores),
            "merged_ranking": merged_ranking,
            "pairwise_accuracy": pair_acc,
            "pairwise_correct_count": pair_correct,
            "pairwise_total_count": pair_total,
            "band_accuracy": band_acc,
            "band_correct_count": band_correct,
            "band_total_count": band_total,
            "spearman_rho": spearman,
        }
        pair_accs.append(pair_acc)
        band_accs.append(band_acc)
        total_pair_correct += pair_correct
        total_pair_count += pair_total
        total_band_correct += band_correct
        total_band_count += band_total
        if spearman is not None:
            spearmans.append(spearman)

    result["summary"] = {
        "pairwise_accuracy_mean": float(sum(pair_accs) / len(pair_accs)) if pair_accs else None,
        "band_accuracy_mean": float(sum(band_accs) / len(band_accs)) if band_accs else None,
        "spearman_mean": float(sum(spearmans) / len(spearmans)) if spearmans else None,
        "pairwise_correct_count": total_pair_correct,
        "pairwise_total_count": total_pair_count,
        "pairwise_accuracy_global": float(total_pair_correct / total_pair_count) if total_pair_count else None,
        "band_correct_count": total_band_correct,
        "band_total_count": total_band_count,
        "band_accuracy_global": float(total_band_correct / total_band_count) if total_band_count else None,
    }

    ensure_dir(os.path.dirname(args.out))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"[RESULT] written: {args.out}")


if __name__ == "__main__":
    main()
def _split_rank_bands_simple(person_ids: List[str]) -> dict[str, List[str]]:
    n = len(person_ids)
    if n == 0:
        return {"upper": [], "middle": [], "lower": []}
    upper_n = max(1, n // 4)
    lower_n = max(1, n // 4)
    if upper_n + lower_n > n:
        lower_n = max(0, n - upper_n)
    middle_start = upper_n
    middle_end = n - lower_n
    return {
        "upper": person_ids[:upper_n],
        "middle": person_ids[middle_start:middle_end],
        "lower": person_ids[middle_end:],
    }
