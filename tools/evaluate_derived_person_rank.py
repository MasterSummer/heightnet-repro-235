from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.abspath(os.path.dirname(__file__))
for _path in [os.path.join(_HERE, "..", "src"), os.path.join(_HERE, "src"), _HERE]:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)

from heightnet.config import load_config
from heightnet.losses import HeightNetLoss
from heightnet.model import DerivedHeightRanker
from heightnet.runtime_depth import RuntimeDepthEstimator
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir
from train_derived_rank import _build_dataset, collate_fn, derive_height_batch
from heightnet.gallery import sample_frame_row_indices_per_person
from heightnet.losses import person_mask_is_valid


def _torch_load_compat(path: str, map_location: str | torch.device, *, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_rank_labels(rank_dir: str) -> dict[str, list[str]]:
    rank_map: dict[str, list[str]] = {}
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


def _band_map_from_ranking(person_ids: List[str]) -> dict[str, str]:
    bands = _split_rank_bands_simple(person_ids)
    out: dict[str, str] = {}
    for band_name, ids in bands.items():
        for pid in ids:
            out[str(pid)] = band_name
    return out


def _spearman(pred_ranked_ids: List[str], gt_ranked_ids: List[str]) -> float | None:
    common = [pid for pid in gt_ranked_ids if pid in set(pred_ranked_ids)]
    n = len(common)
    if n < 2:
        return None
    pred_pos = {pid: idx + 1 for idx, pid in enumerate(pred_ranked_ids) if pid in set(common)}
    gt_pos = {pid: idx + 1 for idx, pid in enumerate(gt_ranked_ids) if pid in set(common)}
    d2 = sum((pred_pos[pid] - gt_pos[pid]) ** 2 for pid in common)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1.0))


def _band_accuracy(pred_ranked_ids: List[str], gt_ranked_ids: List[str]) -> tuple[int, int, float]:
    common = [pid for pid in gt_ranked_ids if pid in set(pred_ranked_ids)]
    if not common:
        return 0, 0, 0.0
    pred_filtered = [pid for pid in pred_ranked_ids if pid in set(common)]
    gt_filtered = [pid for pid in gt_ranked_ids if pid in set(common)]
    pred_bands = _band_map_from_ranking(pred_filtered)
    gt_bands = _band_map_from_ranking(gt_filtered)
    correct = sum(1 for pid in common if pred_bands.get(pid) == gt_bands.get(pid))
    total = len(common)
    return correct, total, float(correct / total) if total > 0 else 0.0


def _aggregate_person_features(records: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        grouped[str(rec["camera_id"])][str(rec["person_id"])].append(rec)

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for cam, persons in grouped.items():
        for pid, recs in persons.items():
            feats = torch.stack([r["feature"] for r in recs], dim=0)
            mean_feat = feats.mean(dim=0)
            out[cam][pid] = {
                "feature": mean_feat,
                "n_frames": len(recs),
                "frame_indices": [int(r["frame_idx"]) for r in recs],
                "sequence_ids": sorted(set(str(r["sequence_id"]) for r in recs)),
            }
    return out


def _build_person_gallery_fullres(
    model: DerivedHeightRanker,
    dataset,
    device: torch.device,
    runtime_depth: RuntimeDepthEstimator,
    segmenter: PersonSegmenter,
    row_indices: list[int],
    batch_size: int,
    num_workers: int,
    min_valid_pixels: int,
    min_valid_ratio: float,
    eps: float,
    assume_inverse: bool,
    use_ground_anchor: bool = True,
) -> list[dict]:
    records: list[dict] = []
    if not row_indices:
        return records

    loader = DataLoader(
        Subset(dataset, row_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            person_mask, person_bbox = segmenter.infer_batch_regions(batch["image_raw"], device)
            derived = derive_height_batch(
                batch,
                runtime_depth,
                device,
                eps,
                assume_inverse,
                person_mask=person_mask,
                person_bbox=person_bbox,
                crop_and_resize=False,
                use_ground_anchor=use_ground_anchor,
            )
            if tuple(person_mask.shape[-2:]) != tuple(derived.shape[-2:]):
                src_h, src_w = person_mask.shape[-2:]
                dst_h, dst_w = derived.shape[-2:]
                scale_x = float(dst_w) / float(src_w)
                scale_y = float(dst_h) / float(src_h)
                person_mask = torch.nn.functional.interpolate(
                    person_mask.to(device), size=(dst_h, dst_w), mode="nearest"
                )
                person_bbox = person_bbox.to(device=device, dtype=torch.float32).clone()
                person_bbox[:, [0, 2]] *= scale_x
                person_bbox[:, [1, 3]] *= scale_y
            else:
                person_mask = person_mask.to(device)
                person_bbox = person_bbox.to(device=device, dtype=torch.float32)

            keep = person_mask_is_valid(
                person_mask, min_valid_pixels=min_valid_pixels, min_valid_ratio=min_valid_ratio
            )
            if not bool(keep.any().item()):
                continue
            feats = model.encode_person(derived, person_mask, person_bbox, None).detach().cpu()
            keep_cpu = keep.detach().cpu().tolist()
            for idx, is_valid in enumerate(keep_cpu):
                if not is_valid:
                    continue
                records.append(
                    {
                        "sequence_id": batch["sequence_id"][idx],
                        "person_id": batch["person_id"][idx],
                        "camera_id": batch["camera_id"][idx],
                        "frame_idx": int(batch["frame_idx"][idx]),
                        "feature": feats[idx],
                    }
                )
    model.train(was_training)
    return records


def _pairwise_rank_persons(
    model: DerivedHeightRanker,
    person_features: dict[str, dict],
    gt_ranking: list[str],
    device: torch.device,
) -> tuple[list[dict], int, int, float]:
    ids = [pid for pid in gt_ranking if pid in person_features]
    wins = {pid: 0 for pid in ids}
    pair_correct = 0
    pair_total = 0

    model.eval()
    with torch.no_grad():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pid_i = ids[i]
                pid_j = ids[j]
                fi = person_features[pid_i]["feature"].unsqueeze(0).to(device)
                fj = person_features[pid_j]["feature"].unsqueeze(0).to(device)
                prob = torch.sigmoid(model.compare_encoded(fi, fj))[0].item()
                pred_i_taller = prob >= 0.5
                if pred_i_taller:
                    wins[pid_i] += 1
                    pair_correct += 1
                else:
                    wins[pid_j] += 1
                pair_total += 1

    ranking = []
    for pid in ids:
        ranking.append(
            {
                "person_id": pid,
                "wins": int(wins[pid]),
                "n_frames": int(person_features[pid]["n_frames"]),
                "frame_indices": person_features[pid]["frame_indices"],
                "sequence_ids": person_features[pid]["sequence_ids"],
            }
        )
    ranking.sort(key=lambda x: (-x["wins"], x["person_id"]))
    pair_acc = float(pair_correct / pair_total) if pair_total > 0 else 0.0
    return ranking, pair_correct, pair_total, pair_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DerivedHeightRanker with person-level merged ranking.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rank-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    query_ds = _build_dataset(cfg, cfg.paths.test_manifest, cfg.paths.test_video_root, train_mode=False)
    gallery_ds = _build_dataset(cfg, cfg.paths.train_manifest, cfg.paths.train_video_root, train_mode=False)

    model = DerivedHeightRanker(
        comparator_channels=cfg.model.comparator_channels,
        comparator_type=cfg.model.comparator_type,
        comparator_layers=cfg.model.comparator_layers,
        comparator_num_heads=cfg.model.comparator_num_heads,
        comparator_patch_size=cfg.model.comparator_patch_size,
        person_region_mode=cfg.model.person_region_mode,
        bbox_expand_ratio=cfg.model.bbox_expand_ratio,
        histogram_min=getattr(cfg.model, "histogram_min", 0.0),
        histogram_max=getattr(cfg.model, "histogram_max", 3.0),
        compare_type=getattr(cfg.model, "compare_type", "concat"),
        use_geometry_branch=getattr(cfg.model, "use_geometry_branch", False),
        geo_feat_dim=getattr(cfg.model, "geo_feat_dim", 5),
        geo_hidden_dim=getattr(cfg.model, "geo_hidden_dim", 32),
    ).to(device)
    ckpt = _torch_load_compat(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    runtime_depth = RuntimeDepthEstimator(
        depthanything_root=cfg.runtime_depth.depthanything_root,
        encoder=cfg.runtime_depth.encoder,
        checkpoint=cfg.runtime_depth.checkpoint,
        input_size=cfg.runtime_depth.input_size,
    ).to(device)
    segmenter = PersonSegmenter(
        model_path=cfg.runtime_seg.model_path,
        conf=cfg.runtime_seg.conf,
        iou=cfg.runtime_seg.iou,
        imgsz=cfg.runtime_seg.imgsz,
        strict_native=cfg.runtime_seg.strict_native,
    )

    query_indices = sample_frame_row_indices_per_person(
        query_ds,
        max_per_person=cfg.eval.frames_per_person_eval,
        seed=cfg.eval.frame_eval_seed,
    )
    gallery_indices = sample_frame_row_indices_per_person(
        gallery_ds,
        max_per_person=cfg.eval.frames_per_person_eval,
        seed=cfg.eval.frame_eval_seed,
    )

    query_records = _build_person_gallery_fullres(
        model=model,
        dataset=query_ds,
        device=device,
        runtime_depth=runtime_depth,
        segmenter=segmenter,
        row_indices=query_indices,
        batch_size=getattr(cfg.eval, "eval_batch_size", 16),
        num_workers=cfg.train.num_workers,
        min_valid_pixels=cfg.loss.min_valid_pixels,
        min_valid_ratio=cfg.loss.min_valid_ratio,
        eps=cfg.loss.eps,
        assume_inverse=cfg.runtime_depth.assume_inverse,
        use_ground_anchor=getattr(cfg.runtime_depth, "use_ground_anchor", True),
    )
    gallery_records = _build_person_gallery_fullres(
        model=model,
        dataset=gallery_ds,
        device=device,
        runtime_depth=runtime_depth,
        segmenter=segmenter,
        row_indices=gallery_indices,
        batch_size=getattr(cfg.eval, "eval_batch_size", 16),
        num_workers=cfg.train.num_workers,
        min_valid_pixels=cfg.loss.min_valid_pixels,
        min_valid_ratio=cfg.loss.min_valid_ratio,
        eps=cfg.loss.eps,
        assume_inverse=cfg.runtime_depth.assume_inverse,
        use_ground_anchor=getattr(cfg.runtime_depth, "use_ground_anchor", True),
    )

    merged_records = list(gallery_records) + list(query_records)
    per_camera_person = _aggregate_person_features(merged_records)
    rank_map = _load_rank_labels(args.rank_dir)

    result = {
        "config": {
            "config_path": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.checkpoint),
            "train_manifest": cfg.paths.train_manifest,
            "test_manifest": cfg.paths.test_manifest,
            "rank_dir": os.path.abspath(args.rank_dir),
            "frames_per_person_eval": int(cfg.eval.frames_per_person_eval),
            "frame_eval_seed": int(cfg.eval.frame_eval_seed),
        },
        "summary": {},
        "cameras": {},
    }

    pair_accs: List[float] = []
    band_accs: List[float] = []
    spearmans: List[float] = []
    total_pair_correct = 0
    total_pair_count = 0
    total_band_correct = 0
    total_band_count = 0

    for cam in sorted(set(per_camera_person.keys()) & set(rank_map.keys())):
        gt_ranked_ids = rank_map[cam]
        ranking, pair_correct, pair_total, pair_acc = _pairwise_rank_persons(
            model=model,
            person_features=per_camera_person[cam],
            gt_ranking=gt_ranked_ids,
            device=device,
        )
        pred_ranked_ids = [row["person_id"] for row in ranking]
        band_correct, band_total, band_acc = _band_accuracy(pred_ranked_ids, gt_ranked_ids)
        spearman = _spearman(pred_ranked_ids, gt_ranked_ids)

        result["cameras"][cam] = {
            "n_merged_person": len(pred_ranked_ids),
            "merged_ranking": ranking,
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
