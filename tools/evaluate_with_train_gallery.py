from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.metrics import pairwise_accuracy_from_ranked_lists
from heightnet.model import HeightNetTiny
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir


def _torch_load_compat(
    path: str,
    map_location: str | torch.device,
    *,
    weights_only: bool,
):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _require_file(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def _load_rank_labels(rank_dir: str) -> dict:
    rank_map = {}
    if not rank_dir:
        return rank_map
    for name in os.listdir(rank_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(rank_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cam = obj.get("camera")
        ranking = obj.get("ranking", [])
        if cam and isinstance(ranking, list):
            rank_map[cam] = ranking
    return rank_map


def _extract_scores(
    loader: DataLoader,
    model: HeightNetTiny,
    segmenter: PersonSegmenter | None,
    device: torch.device,
) -> dict:
    scores_by_camera = defaultdict(lambda: defaultdict(list))
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            pred = model(image)["pred_height_map"]

            if segmenter is not None:
                if "image_raw" not in batch:
                    continue
                person_mask = segmenter.infer_batch(batch["image_raw"], device)
            else:
                raise RuntimeError("runtime_seg.enabled must be true for evaluate_with_train_gallery.py")

            cam = batch.get("camera_id", ["unknown_camera"])[0]
            pid = batch.get("person_id", ["unknown_person"])[0]
            valid = person_mask > 0.5
            if valid.any():
                score = (pred * valid.float()).sum() / torch.clamp(valid.float().sum(), min=1.0)
                scores_by_camera[cam][pid].append(float(score.item()))
    return scores_by_camera


def _avg_scores(d: dict) -> dict:
    out = {}
    for cam, person_scores in d.items():
        out[cam] = {pid: float(sum(v) / max(len(v), 1)) for pid, v in person_scores.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate checkpoint with train-gallery protocol: "
            "compare each test person against train persons under the same camera."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--rank-dir", type=str, default="")
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
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers)

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

    segmenter = None
    if cfg.runtime_seg.enabled:
        segmenter = PersonSegmenter(
            model_path=cfg.runtime_seg.model_path,
            conf=cfg.runtime_seg.conf,
            iou=cfg.runtime_seg.iou,
            imgsz=cfg.runtime_seg.imgsz,
            strict_native=cfg.runtime_seg.strict_native,
        )

    train_scores = _avg_scores(_extract_scores(train_loader, model, segmenter, device))
    test_scores = _avg_scores(_extract_scores(test_loader, model, segmenter, device))

    result = {
        "config": {
            "config_path": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.checkpoint),
            "train_manifest": cfg.paths.train_manifest,
            "test_manifest": cfg.paths.test_manifest,
        },
        "summary": {},
        "cameras": {},
    }

    cams = sorted(set(train_scores.keys()) | set(test_scores.keys()))
    total_pairs = 0
    total_test_person = 0
    covered_test_person = 0

    for cam in cams:
        tr = train_scores.get(cam, {})
        te = test_scores.get(cam, {})
        camera_obj = {
            "n_train_person": len(tr),
            "n_test_person": len(te),
            "test_vs_train_pairwise": [],
            "merged_ranking": [],
        }

        # Test-vs-train pairwise comparisons in same camera.
        for pid_t, s_t in sorted(te.items()):
            total_test_person += 1
            if tr:
                covered_test_person += 1
            for pid_r, s_r in sorted(tr.items()):
                camera_obj["test_vs_train_pairwise"].append(
                    {
                        "test_person": pid_t,
                        "train_person": pid_r,
                        "score_test": float(s_t),
                        "score_train": float(s_r),
                        "pred_test_taller": 1 if s_t > s_r else 0,
                    }
                )
                total_pairs += 1

        merged = []
        merged.extend([{"person_id": k, "score": float(v), "split": "train"} for k, v in tr.items()])
        merged.extend([{"person_id": k, "score": float(v), "split": "test"} for k, v in te.items()])
        merged.sort(key=lambda x: x["score"], reverse=True)
        camera_obj["merged_ranking"] = merged

        result["cameras"][cam] = camera_obj

    # Optional: accuracy against rank labels using merged scores.
    if args.rank_dir:
        rank_map = _load_rank_labels(args.rank_dir)
        acc_items = []
        for cam in cams:
            if cam not in rank_map:
                continue
            merged_scores = {x["person_id"]: x["score"] for x in result["cameras"][cam]["merged_ranking"]}
            if len(merged_scores) < 2:
                continue
            acc = pairwise_accuracy_from_ranked_lists(merged_scores, rank_map[cam])
            acc_items.append(acc)
            result["cameras"][cam]["pairwise_acc_vs_rank_json"] = float(acc)
        result["summary"]["pairwise_acc_vs_rank_json_mean"] = (
            float(sum(acc_items) / len(acc_items)) if acc_items else None
        )

    result["summary"].update(
        {
            "n_cameras": len(cams),
            "n_test_person_total": total_test_person,
            "n_test_person_with_train_reference": covered_test_person,
            "n_test_vs_train_pairs": total_pairs,
        }
    )

    out_path = args.out or os.path.join(cfg.paths.output_dir, "eval_train_gallery.json")
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"[RESULT] written: {out_path}")


if __name__ == "__main__":
    main()
