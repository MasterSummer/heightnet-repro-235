from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.gallery import build_scalar_prob_fn, build_video_stat_gallery, cross_split_pairwise_metrics
from heightnet.losses import HeightNetLoss
from heightnet.model import HeightNetTiny
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir


def _torch_load_compat(path: str, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _build_dataset(cfg, split: str) -> HeightDataset:
    manifest_path = getattr(cfg.paths, f"{split}_manifest")
    video_root = getattr(cfg.paths, f"{split}_video_root")
    if manifest_path:
        return HeightDataset(
            manifest_path,
            tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=False,
            train_mode=False,
        )
    if video_root:
        return HeightDataset.from_video_root(
            video_root=video_root,
            bg_depth_root=cfg.paths.bg_depth_root,
            image_size=tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=False,
            train_mode=False,
        )
    raise RuntimeError(f"{split} dataset source missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple height-statistics baselines on test-vs-train video pairs.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stats", nargs="+", default=["masked_avg", "p95", "p99", "max"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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
    ckpt = _torch_load_compat(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    segmenter = PersonSegmenter(
        model_path=cfg.runtime_seg.model_path,
        conf=cfg.runtime_seg.conf,
        iou=cfg.runtime_seg.iou,
        imgsz=cfg.runtime_seg.imgsz,
        strict_native=cfg.runtime_seg.strict_native,
    )

    train_ds = _build_dataset(cfg, "train")
    test_ds = _build_dataset(cfg, "test")
    pair_labels = HeightNetLoss.load_pairwise_labels(cfg.loss.pairwise_json)

    train_gallery = build_video_stat_gallery(
        model=model,
        dataset=train_ds,
        device=device,
        segmenter=segmenter,
        num_frames=cfg.eval.video_feature_frames,
        min_valid_pixels=cfg.loss.min_valid_pixels,
        min_valid_ratio=cfg.loss.min_valid_ratio,
    )
    test_gallery = build_video_stat_gallery(
        model=model,
        dataset=test_ds,
        device=device,
        segmenter=segmenter,
        num_frames=cfg.eval.video_feature_frames,
        min_valid_pixels=cfg.loss.min_valid_pixels,
        min_valid_ratio=cfg.loss.min_valid_ratio,
    )

    payload = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "num_train_videos": len(train_gallery),
        "num_test_videos": len(test_gallery),
        "video_feature_frames": cfg.eval.video_feature_frames,
        "stats": {},
    }

    for score_key in args.stats:
        prob_fn = build_scalar_prob_fn(test_gallery, train_gallery, score_key=score_key)
        summary = cross_split_pairwise_metrics(
            query_records=test_gallery,
            gallery_records=train_gallery,
            pairwise_labels=pair_labels,
            prob_fn=prob_fn,
        )
        cameras = {}
        for cam in sorted({rec["camera_id"] for rec in test_gallery} | {rec["camera_id"] for rec in train_gallery}):
            query_cam = [rec for rec in test_gallery if rec["camera_id"] == cam]
            gallery_cam = [rec for rec in train_gallery if rec["camera_id"] == cam]
            if not query_cam or not gallery_cam:
                continue
            cam_prob_fn = build_scalar_prob_fn(query_cam, gallery_cam, score_key=score_key)
            cameras[cam] = {
                **cross_split_pairwise_metrics(
                    query_records=query_cam,
                    gallery_records=gallery_cam,
                    pairwise_labels=pair_labels,
                    prob_fn=cam_prob_fn,
                ),
                "n_test_videos": len(query_cam),
                "n_train_videos": len(gallery_cam),
            }
        payload["stats"][score_key] = {
            "summary": summary,
            "cameras": cameras,
        }
        print(
            f"[BASELINE] stat={score_key} pairwise_acc={summary['pairwise_accuracy']:.4f} "
            f"auc={summary['auc']:.4f} f1={summary['f1']:.4f} n_pairs={summary['n_pairs_eval']}"
        )

    ensure_dir(os.path.dirname(args.out))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OUT] baseline results exported: {args.out}")


if __name__ == "__main__":
    main()
