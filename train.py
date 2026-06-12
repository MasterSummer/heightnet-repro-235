from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.gallery import (
    build_frame_feature_gallery,
    build_video_feature_gallery,
    cross_split_pairwise_metrics,
    sample_frame_row_indices_per_person,
    save_video_feature_gallery,
)
from heightnet.losses import HeightNetLoss, build_rank_pairs, masked_rmse
from heightnet.model import HeightNetTiny
from heightnet.runtime_depth import RuntimeDepthEstimator, depth_to_height
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir, set_seed


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


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_main_process() -> bool:
    return _get_rank() == 0


def _setup_distributed(cfg_device: str) -> tuple[bool, int, int, torch.device]:
    use_distributed = _is_distributed()
    if not use_distributed:
        device = torch.device(cfg_device if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA devices.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = _get_rank()
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    return True, rank, world_size, device


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _require_file(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def _has_dataset_source(manifest_path: str, video_root: str) -> bool:
    return bool(manifest_path or video_root)


def _build_dataset(
    manifest_path: str,
    video_root: str,
    bg_depth_root: str,
    image_size: tuple[int, int],
    normalize_rgb: bool,
    use_pair_consistency: bool,
    train_mode: bool,
) -> HeightDataset:
    if manifest_path:
        return HeightDataset(
            manifest_path,
            image_size,
            normalize_rgb=normalize_rgb,
            use_pair_consistency=use_pair_consistency,
            train_mode=train_mode,
        )
    return HeightDataset.from_video_root(
        video_root=video_root,
        bg_depth_root=bg_depth_root,
        image_size=image_size,
        normalize_rgb=normalize_rgb,
        use_pair_consistency=use_pair_consistency,
        train_mode=train_mode,
    )


def _gallery_path(output_dir: str, suffix: str) -> str:
    return os.path.join(output_dir, f"train_gallery_{suffix}.pt")


def collate_fn(batch: list[dict]) -> dict:
    out: Dict[str, torch.Tensor | list[str] | list[int]] = {}
    tensor_keys = ["image", "image_raw", "height", "mask", "image_pair", "image_pair_raw", "bg_depth", "camera_height_m", "need_online_depth"]
    for key in tensor_keys:
        if key in batch[0]:
            out[key] = torch.stack([x[key] for x in batch], dim=0)

    list_keys = ["person_id", "camera_id", "sequence_id", "frame_idx", "frame_idx_pair"]
    for key in list_keys:
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


def _denormalize_image_batch(image: torch.Tensor, normalize_rgb: bool) -> torch.Tensor:
    if not normalize_rgb:
        return torch.clamp(image, 0.0, 1.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    return torch.clamp(image * std + mean, 0.0, 1.0)


def _norm_map(x: torch.Tensor) -> torch.Tensor:
    x_min = torch.amin(x, dim=(2, 3), keepdim=True)
    x_max = torch.amax(x, dim=(2, 3), keepdim=True)
    return torch.clamp((x - x_min) / (x_max - x_min + 1e-6), 0.0, 1.0)


def _gather_list_distributed(values: list, device: torch.device, distributed: bool) -> list:
    if not distributed:
        return values
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, values)
    out = []
    for item in gathered:
        out.extend(item)
    return out


@torch.no_grad()
def evaluate(
    model,
    loader,
    query_ds: HeightDataset,
    gallery_ds: HeightDataset,
    device,
    segmenter: PersonSegmenter,
    runtime_depth: RuntimeDepthEstimator | None,
    eps: float,
    pairwise_labels: Dict[str, Dict[Tuple[str, str], int]],
    min_valid_pixels: int,
    min_valid_ratio: float,
    frames_per_person_eval: int,
    frame_eval_seed: int,
    assume_inverse: bool,
    distributed: bool = False,
) -> dict:
    model.eval()
    meters = {"rmse": 0.0, "n": 0}
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["height"].to(device)
        mask = batch["mask"].to(device)
        target, mask = _resolve_online_supervision(
            batch=batch,
            target=target,
            valid_mask=mask,
            runtime_depth=runtime_depth,
            eps=eps,
            assume_inverse=assume_inverse,
        )
        pred = model(image)["pred_height_map"]
        meters["rmse"] += masked_rmse(pred, target, mask).item()
        meters["n"] += 1

    if distributed:
        rmse_sum = torch.tensor([meters["rmse"]], dtype=torch.float64, device=device)
        n_sum = torch.tensor([meters["n"]], dtype=torch.float64, device=device)
        dist.all_reduce(rmse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_sum, op=dist.ReduceOp.SUM)
        meters["rmse"] = (rmse_sum / torch.clamp(n_sum, min=1.0)).item()
        meters["n"] = int(n_sum.item())
    else:
        meters["rmse"] /= max(meters["n"], 1)
    if _is_main_process() or not distributed:
        model_ref = model.module if distributed else model
        query_indices = sample_frame_row_indices_per_person(query_ds, max_per_person=frames_per_person_eval, seed=frame_eval_seed)
        gallery_indices = sample_frame_row_indices_per_person(gallery_ds, max_per_person=frames_per_person_eval, seed=frame_eval_seed)
        query_records = build_frame_feature_gallery(
            model=model_ref,
            dataset=query_ds,
            device=device,
            segmenter=segmenter,
            row_indices=query_indices,
            min_valid_pixels=min_valid_pixels,
            min_valid_ratio=min_valid_ratio,
        )
        gallery_records = build_frame_feature_gallery(
            model=model_ref,
            dataset=gallery_ds,
            device=device,
            segmenter=segmenter,
            row_indices=gallery_indices,
            min_valid_pixels=min_valid_pixels,
            min_valid_ratio=min_valid_ratio,
        )

        def _prob_fn(i: int, j: int) -> float:
            with torch.no_grad():
                logit = model_ref.compare_encoded(
                    query_records[i]["feature"].unsqueeze(0).to(device),
                    gallery_records[j]["feature"].unsqueeze(0).to(device),
                )
            return float(torch.sigmoid(logit)[0].item())

        pair_metrics = cross_split_pairwise_metrics(query_records, gallery_records, pairwise_labels, _prob_fn)
    else:
        pair_metrics = None
    if distributed:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, pair_metrics)
        pair_metrics = next((x for x in gathered if x is not None), None)
        pair_metrics = pair_metrics or {
            "pairwise_accuracy": 0.0,
            "auc": 0.0,
            "f1": 0.0,
            "n_pairs_eval": 0,
            "n_comparisons": 0,
            "rankings": {},
        }
    meters.update(pair_metrics)
    return meters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    distributed = False
    writer = None
    try:
        cfg = load_config(args.config)
        distributed, rank, world_size, device = _setup_distributed(cfg.device)
        set_seed(cfg.seed + rank)

        if not _has_dataset_source(cfg.paths.train_manifest, cfg.paths.train_video_root):
            raise RuntimeError("train dataset source missing: set either paths.train_manifest or paths.train_video_root")
        if not _has_dataset_source(cfg.paths.val_manifest, cfg.paths.val_video_root):
            raise RuntimeError("val dataset source missing: set either paths.val_manifest or paths.val_video_root")
        if cfg.paths.train_manifest:
            _require_file(cfg.paths.train_manifest, "train_manifest")
        if cfg.paths.val_manifest:
            _require_file(cfg.paths.val_manifest, "val_manifest")
        _require_file(cfg.loss.pairwise_json, "pairwise_json")

        if not cfg.runtime_seg.enabled:
            raise RuntimeError("runtime_seg.enabled must be true. person_mask is only from runtime segmentation.")
        _require_file(cfg.runtime_seg.model_path, "runtime segmentation model")

        if _is_main_process():
            ensure_dir(cfg.paths.output_dir)

        train_ds = _build_dataset(
            manifest_path=cfg.paths.train_manifest,
            video_root=cfg.paths.train_video_root,
            bg_depth_root=cfg.paths.bg_depth_root,
            image_size=tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=cfg.data.use_pair_consistency,
            train_mode=True,
        )
        gallery_ds = _build_dataset(
            manifest_path=cfg.paths.train_manifest,
            video_root=cfg.paths.train_video_root,
            bg_depth_root=cfg.paths.bg_depth_root,
            image_size=tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=False,
            train_mode=False,
        )
        val_ds = _build_dataset(
            manifest_path=cfg.paths.val_manifest,
            video_root=cfg.paths.val_video_root,
            bg_depth_root=cfg.paths.bg_depth_root,
            image_size=tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=cfg.data.use_pair_consistency,
            train_mode=False,
        )

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            collate_fn=collate_fn,
        )

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
        if distributed:
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )

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

        criterion = HeightNetLoss(
            lambda_rmse=cfg.loss.lambda_rmse,
            lambda_rank=cfg.loss.lambda_rank,
            lambda_cons=cfg.loss.lambda_cons,
            eps=cfg.loss.eps,
            min_valid_pixels=cfg.loss.min_valid_pixels,
            min_valid_ratio=cfg.loss.min_valid_ratio,
            consistency_mode=cfg.loss.consistency_mode,
        )
        pairwise_labels = HeightNetLoss.load_pairwise_labels(cfg.loss.pairwise_json)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.use_amp and device.type == "cuda")

        if cfg.train.use_tensorboard and _is_main_process() and SummaryWriter is not None:
            tb_dir = os.path.join(cfg.paths.output_dir, "tb")
            ensure_dir(tb_dir)
            writer = SummaryWriter(log_dir=tb_dir)
        elif cfg.train.use_tensorboard and _is_main_process() and SummaryWriter is None:
            print("[warn] tensorboard is unavailable in current environment; continue without it.")

        best_pairwise_accuracy = float("-inf")
        global_step = 0
        start_epoch = 0

        if args.resume:
            _require_file(args.resume, "resume checkpoint")
            ckpt = _torch_load_compat(args.resume, map_location=device, weights_only=False)
            model_state = ckpt.get("model")
            if distributed:
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            best_pairwise_accuracy = float(ckpt.get("best_pairwise_accuracy", float("-inf")))

        for epoch in range(start_epoch, cfg.train.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            model.train()
            running = {"total": 0.0, "rmse": 0.0, "rank": 0.0, "consistency": 0.0, "n": 0}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", disable=not _is_main_process())
            vis_logged = False

            for batch in pbar:
                image = batch["image"].to(device)
                target = batch["height"].to(device)
                mask = batch["mask"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=cfg.train.use_amp and device.type == "cuda"):
                    if "image_pair_raw" not in batch:
                        raise RuntimeError("image_pair_raw is required for consistency branch.")
                    image_pair = batch["image_pair"].to(device)
                    person_mask, person_bbox = segmenter.infer_batch_regions(batch["image_raw"], device)
                    person_mask_pair, _ = segmenter.infer_batch_regions(batch["image_pair_raw"], device)
                    target, mask = _resolve_online_supervision(
                        batch=batch,
                        target=target,
                        valid_mask=mask,
                        runtime_depth=runtime_depth,
                        eps=cfg.loss.eps,
                        assume_inverse=cfg.runtime_depth.assume_inverse,
                    )

                    idx_i, idx_j, pair_label = build_rank_pairs(
                        person_ids=batch["person_id"],
                        camera_ids=batch["camera_id"],
                        person_masks=person_mask,
                        pairwise_labels=pairwise_labels,
                        min_valid_pixels=cfg.loss.min_valid_pixels,
                        min_valid_ratio=cfg.loss.min_valid_ratio,
                        device=device,
                    )

                    batch_size = image.shape[0]
                    forward_out = model(
                        torch.cat([image, image_pair], dim=0),
                        pair_inputs={
                            "idx_i": idx_i,
                            "idx_j": idx_j,
                            "person_mask": person_mask,
                            "person_bbox": person_bbox,
                        }
                        if pair_label.numel() > 0
                        else None,
                    )
                    pred_all = forward_out["pred_height_map"]
                    pred = pred_all[:batch_size]
                    pred_pair = pred_all[batch_size:]
                    pair_logit = forward_out["pair_logit"]

                    losses = criterion(
                        pred_height=pred,
                        target_height=target,
                        valid_mask=mask,
                        pred_pair=pred_pair,
                        person_mask=person_mask,
                        person_mask_pair=person_mask_pair,
                        pair_logit=pair_logit,
                        pair_label=pair_label,
                    )

                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                running["total"] += losses["total"].item()
                running["rmse"] += losses["rmse"].item()
                running["rank"] += losses["rank"].item()
                running["consistency"] += losses["consistency"].item()
                running["n"] += 1
                global_step += 1

                if _is_main_process():
                    pbar.set_postfix(
                        total=running["total"] / running["n"],
                        rmse=running["rmse"] / running["n"],
                        rank=running["rank"] / running["n"],
                        cons=running["consistency"] / running["n"],
                        rank_pairs=losses["rank_pairs"],
                        cons_valid_pairs=losses["cons_valid_pairs"],
                    )

                if writer is not None and global_step % max(cfg.train.log_interval, 1) == 0:
                    writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
                    writer.add_scalar("train/loss_rmse", losses["rmse"].item(), global_step)
                    writer.add_scalar("train/loss_rank", losses["rank"].item(), global_step)
                    writer.add_scalar("train/loss_consistency", losses["consistency"].item(), global_step)
                    writer.add_scalar("train/rank_pairs", losses["rank_pairs"], global_step)
                    writer.add_scalar("train/cons_valid_pairs", losses["cons_valid_pairs"], global_step)

                if writer is not None and not vis_logged:
                    n_vis = min(cfg.train.tb_num_images, image.shape[0])
                    rgb = _denormalize_image_batch(image[:n_vis], cfg.data.normalize_rgb).detach().cpu()
                    gt = _norm_map(target[:n_vis]).repeat(1, 3, 1, 1).detach().cpu()
                    pd = _norm_map(pred[:n_vis]).repeat(1, 3, 1, 1).detach().cpu()
                    writer.add_images("train_vis/rgb", rgb, epoch + 1)
                    writer.add_images("train_vis/gt_height", gt, epoch + 1)
                    writer.add_images("train_vis/pred_height", pd, epoch + 1)
                    vis_logged = True

            metrics = evaluate(
                model,
                val_loader,
                query_ds=val_ds,
                gallery_ds=gallery_ds,
                device=device,
                segmenter=segmenter,
                runtime_depth=runtime_depth,
                eps=cfg.loss.eps,
                pairwise_labels=pairwise_labels,
                min_valid_pixels=cfg.loss.min_valid_pixels,
                min_valid_ratio=cfg.loss.min_valid_ratio,
                frames_per_person_eval=cfg.eval.frames_per_person_eval,
                frame_eval_seed=cfg.eval.frame_eval_seed,
                assume_inverse=cfg.runtime_depth.assume_inverse,
                distributed=distributed,
            )
            if _is_main_process():
                print(
                    f"[VAL] epoch={epoch + 1} "
                    f"pairwise_acc={metrics['pairwise_accuracy']:.4f} "
                    f"auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} "
                    f"rmse={metrics['rmse']:.4f} n_pairs={metrics['n_pairs_eval']} "
                    f"n_cmp={metrics['n_comparisons']}"
                )

            if writer is not None:
                writer.add_scalar("val/rmse", metrics["rmse"], epoch + 1)
                writer.add_scalar("val/pairwise_accuracy", metrics["pairwise_accuracy"], epoch + 1)
                writer.add_scalar("val/auc", metrics["auc"], epoch + 1)
                writer.add_scalar("val/f1", metrics["f1"], epoch + 1)
                writer.add_scalar("val/n_comparisons", metrics["n_comparisons"], epoch + 1)

            if _is_main_process():
                ckpt_last = os.path.join(cfg.paths.output_dir, "checkpoint_last.pt")
                state_dict = model.module.state_dict() if distributed else model.state_dict()
                torch.save(
                    {
                        "model": state_dict,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "best_pairwise_accuracy": best_pairwise_accuracy,
                    },
                    ckpt_last,
                )

                if metrics["pairwise_accuracy"] > best_pairwise_accuracy:
                    best_pairwise_accuracy = metrics["pairwise_accuracy"]
                    ckpt_best = os.path.join(cfg.paths.output_dir, "checkpoint_best.pt")
                    torch.save(
                        {
                            "model": state_dict,
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "best_pairwise_accuracy": best_pairwise_accuracy,
                        },
                        ckpt_best,
                    )
                    if cfg.eval.save_train_gallery:
                        gallery_records = build_video_feature_gallery(
                            model=model,
                            dataset=gallery_ds,
                            device=device,
                            segmenter=segmenter,
                            num_frames=cfg.eval.video_feature_frames,
                            min_valid_pixels=cfg.loss.min_valid_pixels,
                            min_valid_ratio=cfg.loss.min_valid_ratio,
                        )
                        save_video_feature_gallery(
                            _gallery_path(cfg.paths.output_dir, "best"),
                            gallery_records,
                            num_frames=cfg.eval.video_feature_frames,
                        )

            # Keep non-main ranks from entering the next epoch while rank0 is
            # still checkpointing or exporting the best-train gallery.
            if distributed:
                dist.barrier()

    finally:
        if writer is not None:
            writer.close()
        if distributed:
            _cleanup_distributed()


if __name__ == "__main__":
    main()
