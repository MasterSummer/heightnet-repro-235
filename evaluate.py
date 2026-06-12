from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.gallery import (
    build_frame_feature_gallery,
    build_video_feature_gallery,
    cross_split_pairwise_metrics,
    load_video_feature_gallery,
    sample_frame_row_indices_per_person,
    save_video_feature_gallery,
)
from heightnet.losses import HeightNetLoss, masked_rmse
from heightnet.model import HeightNetTiny
from heightnet.runtime_depth import RuntimeDepthEstimator, depth_to_height
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


def _build_eval_dataset(cfg) -> HeightDataset:
    if cfg.paths.test_manifest:
        return HeightDataset(
            cfg.paths.test_manifest,
            tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=False,
            train_mode=False,
        )
    if cfg.paths.test_video_root:
        return HeightDataset.from_video_root(
            video_root=cfg.paths.test_video_root,
            bg_depth_root=cfg.paths.bg_depth_root,
            image_size=tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=False,
            train_mode=False,
        )
    raise RuntimeError("test dataset source missing: set either paths.test_manifest or paths.test_video_root")


def _build_train_dataset(cfg) -> HeightDataset:
    if cfg.paths.train_manifest:
        return HeightDataset(
            cfg.paths.train_manifest,
            tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=False,
            train_mode=False,
        )
    if cfg.paths.train_video_root:
        return HeightDataset.from_video_root(
            video_root=cfg.paths.train_video_root,
            bg_depth_root=cfg.paths.bg_depth_root,
            image_size=tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=False,
            train_mode=False,
        )
    raise RuntimeError("train dataset source missing: set either paths.train_manifest or paths.train_video_root")


def collate_fn(batch: list[dict]) -> dict:
    out: dict = {}
    for key in ["image", "image_raw", "height", "mask", "bg_depth", "camera_height_m", "need_online_depth"]:
        if key in batch[0]:
            out[key] = torch.stack([x[key] for x in batch], dim=0)
    for key in ["person_id", "camera_id", "sequence_id"]:
        if key in batch[0]:
            out[key] = [x[key] for x in batch]
    return out


def _resolve_online_supervision(
    batch: dict,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    runtime_depth: RuntimeDepthEstimator | None,
    eps: float,
    assume_inverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "need_online_depth" not in batch:
        return target, valid_mask
    need_cpu = (batch["need_online_depth"] > 0.5).view(-1).cpu()
    if not torch.any(need_cpu):
        return target, valid_mask
    if runtime_depth is None:
        raise RuntimeError("batch requires online depth supervision but runtime_depth is disabled.")

    raw = batch["image_raw"][need_cpu].to(target.device)
    bg = batch["bg_depth"][need_cpu].to(target.device)
    cam_h = batch["camera_height_m"][need_cpu].to(target.device).view(-1, 1, 1, 1)
    depth = runtime_depth.infer_batch(raw)
    h, m = depth_to_height(
        depth=depth,
        bg_depth=bg,
        camera_height_m=cam_h,
        eps=eps,
        assume_inverse=assume_inverse,
    )

    target = target.clone()
    valid_mask = valid_mask.clone()
    need = need_cpu.to(target.device)
    target[need] = h
    valid_mask[need] = m
    return target, valid_mask


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


def _pairs_from_ranking(ranking: List[str]) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            a, b = ranking[i], ranking[j]
            out[(a, b)] = 1
            out[(b, a)] = 0
    return out


def _foreground_height(pred: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = (mask > 0.5).float()
    fg = pred * m
    return fg, m


def _dataset_is_frame_level(ds: HeightDataset) -> bool:
    if not ds.rows:
        return False
    return all(int(row.frame_start) == int(row.frame_end) for row in ds.rows[: min(len(ds.rows), 32)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rank-dir", type=str, default="")
    parser.add_argument("--pairwise-out", type=str, default="")
    parser.add_argument("--gallery-path", type=str, default="")
    parser.add_argument("--gallery-out", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.paths.test_manifest:
        _require_file(cfg.paths.test_manifest, "test_manifest")
    _require_file(args.checkpoint, "checkpoint")
    _require_file(cfg.loss.pairwise_json, "pairwise_json")
    if not cfg.runtime_seg.enabled:
        raise RuntimeError("runtime_seg.enabled must be true for evaluation.")
    _require_file(cfg.runtime_seg.model_path, "runtime segmentation model")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ds = _build_eval_dataset(cfg)
    train_ds = _build_train_dataset(cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers, collate_fn=collate_fn)

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
    model.eval()

    segmenter = PersonSegmenter(
        model_path=cfg.runtime_seg.model_path,
        conf=cfg.runtime_seg.conf,
        iou=cfg.runtime_seg.iou,
        imgsz=cfg.runtime_seg.imgsz,
        strict_native=cfg.runtime_seg.strict_native,
    )
    runtime_depth = None
    if cfg.runtime_depth.enabled:
        runtime_depth = RuntimeDepthEstimator(
            depthanything_root=cfg.runtime_depth.depthanything_root,
            encoder=cfg.runtime_depth.encoder,
            checkpoint=cfg.runtime_depth.checkpoint,
            input_size=cfg.runtime_depth.input_size,
        ).to(device)

    pair_labels = HeightNetLoss.load_pairwise_labels(cfg.loss.pairwise_json)
    rank_labels = _load_rank_labels(args.rank_dir)

    meters = {"rmse": 0.0, "n": 0}
    vis_dir = os.path.join(cfg.paths.output_dir, "vis")
    if cfg.eval.save_visualizations:
        ensure_dir(vis_dir)

    frame_level_eval = _dataset_is_frame_level(ds) and _dataset_is_frame_level(train_ds)
    gallery_path = args.gallery_path or os.path.join(cfg.paths.output_dir, "train_gallery_best.pt")
    if frame_level_eval:
        train_gallery = build_frame_feature_gallery(
            model=model,
            dataset=train_ds,
            device=device,
            segmenter=segmenter,
            row_indices=sample_frame_row_indices_per_person(
                train_ds,
                max_per_person=cfg.eval.frames_per_person_eval,
                seed=cfg.eval.frame_eval_seed,
            ),
            min_valid_pixels=cfg.loss.min_valid_pixels,
            min_valid_ratio=cfg.loss.min_valid_ratio,
        )
    else:
        if os.path.exists(gallery_path):
            train_gallery = load_video_feature_gallery(gallery_path)
        else:
            train_gallery = build_video_feature_gallery(
                model=model,
                dataset=train_ds,
                device=device,
                segmenter=segmenter,
                num_frames=cfg.eval.video_feature_frames,
                min_valid_pixels=cfg.loss.min_valid_pixels,
                min_valid_ratio=cfg.loss.min_valid_ratio,
            )
            gallery_out = args.gallery_out or gallery_path
            ensure_dir(os.path.dirname(gallery_out))
            save_video_feature_gallery(gallery_out, train_gallery, num_frames=cfg.eval.video_feature_frames)

    test_gallery: List[dict] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            image = batch["image"].to(device)
            target = batch["height"].to(device)
            valid_mask = batch["mask"].to(device)
            target, valid_mask = _resolve_online_supervision(
                batch=batch,
                target=target,
                valid_mask=valid_mask,
                runtime_depth=runtime_depth,
                eps=cfg.loss.eps,
                assume_inverse=cfg.runtime_depth.assume_inverse,
            )
            person_mask, person_bbox = segmenter.infer_batch_regions(batch["image_raw"], device)

            pred = model(image)["pred_height_map"]
            meters["rmse"] += masked_rmse(pred, target, valid_mask).item()
            meters["n"] += 1

            cam = batch["camera_id"][0]
            seq = batch["sequence_id"][0]
            pid = batch["person_id"][0]

            if not frame_level_eval:
                feat = model.encode_person(pred[0:1], person_mask[0:1], person_bbox[0:1])[0].detach().cpu()
                test_gallery.append(
                    {
                        "sequence_id": seq,
                        "person_id": pid,
                        "camera_id": cam,
                        "feature": feat,
                    }
                )

            if cfg.eval.save_visualizations and i < cfg.eval.vis_limit:
                rgb = image[0].detach().cpu().numpy().transpose(1, 2, 0)
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
                gt = target[0, 0].detach().cpu().numpy()
                pd = pred[0, 0].detach().cpu().numpy()

                fig, ax = plt.subplots(1, 3, figsize=(14, 4))
                ax[0].imshow(rgb)
                ax[0].set_title("RGB")
                ax[1].imshow(gt, cmap="viridis")
                ax[1].set_title("GT Height")
                ax[2].imshow(pd, cmap="viridis")
                ax[2].set_title("Pred Height")
                for a in ax:
                    a.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(vis_dir, f"{i:05d}.png"), dpi=120)
                plt.close(fig)

    meters["rmse"] /= max(meters["n"], 1)
    print(f"[TEST] rmse={meters['rmse']:.4f}")

    if frame_level_eval:
        test_gallery = build_frame_feature_gallery(
            model=model,
            dataset=ds,
            device=device,
            segmenter=segmenter,
            row_indices=sample_frame_row_indices_per_person(
                ds,
                max_per_person=cfg.eval.frames_per_person_eval,
                seed=cfg.eval.frame_eval_seed,
            ),
            min_valid_pixels=cfg.loss.min_valid_pixels,
            min_valid_ratio=cfg.loss.min_valid_ratio,
        )

    payload = {
        "rmse": meters["rmse"],
        "gallery_path": gallery_path,
        "num_train_gallery_videos": len(train_gallery),
        "num_test_videos": len(test_gallery),
        "cameras": {},
    }

    def _prob_fn(i: int, j: int) -> float:
        q = test_gallery[i]
        g = train_gallery[j]
        with torch.no_grad():
            logit = model.compare_encoded(
                q["feature"].unsqueeze(0).to(device),
                g["feature"].unsqueeze(0).to(device),
            )
        return float(torch.sigmoid(logit)[0].item())

    summary = cross_split_pairwise_metrics(
        query_records=test_gallery,
        gallery_records=train_gallery,
        pairwise_labels=pair_labels,
        prob_fn=_prob_fn,
    )
    payload["summary"] = {
        **summary,
        "rmse": meters["rmse"],
    }

    cameras = sorted({rec["camera_id"] for rec in test_gallery} | {rec["camera_id"] for rec in train_gallery})
    for cam in cameras:
        query_cam = [rec for rec in test_gallery if rec["camera_id"] == cam]
        gallery_cam = [rec for rec in train_gallery if rec["camera_id"] == cam]
        if not query_cam or not gallery_cam:
            continue

        def _prob_cam(i: int, j: int) -> float:
            q = query_cam[i]
            g = gallery_cam[j]
            with torch.no_grad():
                logit = model.compare_encoded(
                    q["feature"].unsqueeze(0).to(device),
                    g["feature"].unsqueeze(0).to(device),
                )
            return float(torch.sigmoid(logit)[0].item())

        cam_metrics = cross_split_pairwise_metrics(
            query_records=query_cam,
            gallery_records=gallery_cam,
            pairwise_labels=pair_labels,
            prob_fn=_prob_cam,
        )
        payload["cameras"][cam] = {
            **cam_metrics,
            "n_test_videos": len(query_cam),
            "n_train_gallery_videos": len(gallery_cam),
        }
        print(
            f"[PAIRWISE] camera={cam} n_test={len(query_cam)} n_train={len(gallery_cam)} "
            f"n_cmp={cam_metrics['n_comparisons']} n_pairs_eval={cam_metrics['n_pairs_eval']} "
            f"pairwise_acc={cam_metrics['pairwise_accuracy']:.4f} auc={cam_metrics['auc']:.4f}"
        )

    print(
        "[SUMMARY] "
        f"pairwise_acc={summary['pairwise_accuracy']:.4f} auc={summary['auc']:.4f} "
        f"f1={summary['f1']:.4f} rmse={meters['rmse']:.4f} n_pairs={summary['n_pairs_eval']}"
    )

    pairwise_out = args.pairwise_out or os.path.join(cfg.paths.output_dir, "pairwise_pred.json")
    ensure_dir(os.path.dirname(pairwise_out))
    with open(pairwise_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[PRED] pairwise exported: {pairwise_out}")


if __name__ == "__main__":
    main()
