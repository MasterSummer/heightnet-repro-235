from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import torch
import numpy as np
import cv2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from heightnet.config import load_config
from heightnet.crop_utils import clamp_bbox_xyxy, expand_bbox_xyxy, resize_and_pad_map
from heightnet.datasets import HeightDataset
from heightnet.gallery import cross_split_pairwise_metrics, sample_frame_row_indices_per_person, save_video_feature_gallery
from heightnet.losses import HeightNetLoss, build_rank_pairs, masked_rmse, person_mask_is_valid
from heightnet.model import DerivedHeightRanker
from heightnet.person_cache import infer_or_load_person_regions
from heightnet.runtime_depth import RuntimeDepthEstimator, depth_to_height
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir, set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


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
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
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


def _build_dataset(cfg, manifest_path: str, video_root: str, train_mode: bool) -> HeightDataset:
    if manifest_path:
        return HeightDataset(
            manifest_path,
            tuple(cfg.data.image_size),
            normalize_rgb=cfg.data.normalize_rgb,
            use_pair_consistency=False,
            train_mode=train_mode,
        )
    return HeightDataset.from_video_root(
        video_root=video_root,
        bg_depth_root=cfg.paths.bg_depth_root,
        image_size=tuple(cfg.data.image_size),
        normalize_rgb=cfg.data.normalize_rgb,
        use_pair_consistency=False,
        train_mode=train_mode,
    )


def collate_fn(batch: list[dict]) -> dict:
    out: dict = {}
    for key in ["image", "image_raw", "bg_depth", "camera_height_m", "height", "mask"]:
        if key in batch[0]:
            out[key] = torch.stack([x[key] for x in batch], dim=0)
    for key in ["person_id", "camera_id", "sequence_id", "frame_idx", "frame_path", "depth_cache_path", "bg_depth_path"]:
        if key in batch[0]:
            out[key] = [x[key] for x in batch]
    return out


def _get_person_regions(batch: dict, segmenter: PersonSegmenter, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frame_paths = batch.get("frame_path", [])
    if frame_paths and all(isinstance(path, str) and path for path in frame_paths):
        return infer_or_load_person_regions(
            images_raw=batch["image_raw"],
            frame_paths=frame_paths,
            segmenter=segmenter,
            device=device,
            allow_inference_fallback=bool(getattr(segmenter, "allow_inference_fallback", True)),
        )
    return segmenter.infer_batch_regions(batch["image_raw"], device)


def _load_ground_mask_cache_batch(
    frame_paths: list[str],
    h: int,
    w: int,
    device: torch.device,
) -> "torch.Tensor | None":
    """Load precomputed SegFormer ground masks from .groundmask.npy cache files.
    Returns [B,1,H,W] bool tensor, or None if any cache is missing."""
    masks = []
    for fp in frame_paths:
        cache = fp + ".groundmask.npy"
        if not os.path.exists(cache):
            return None
        try:
            arr = np.load(cache).astype(bool)
        except (ValueError, OSError):
            return None
        if arr.shape != (h, w):
            import cv2 as _cv2
            arr = _cv2.resize(arr.astype(np.uint8), (w, h), interpolation=_cv2.INTER_NEAREST).astype(bool)
        masks.append(torch.from_numpy(arr).unsqueeze(0))  # [1, H, W]
    return torch.stack(masks, dim=0).to(device)  # [B, 1, H, W]


def _resolve_depth_cache_path(frame_path, depth_cache_path) -> str | None:
    if depth_cache_path and isinstance(depth_cache_path, str) and os.path.exists(depth_cache_path):
        return depth_cache_path
    if frame_path and isinstance(frame_path, str):
        cache = frame_path + ".depth.npy"
        if os.path.exists(cache):
            return cache
    return None


def _load_depth_cache_batch(
    frame_paths: list[str],
    h: int,
    w: int,
    depth_cache_paths: list[str] | None = None,
) -> "torch.Tensor | None":
    depths = []
    for idx, fp in enumerate(frame_paths):
        dcp = depth_cache_paths[idx] if depth_cache_paths else None
        cache = _resolve_depth_cache_path(fp, dcp)
        if cache is None:
            return None
        try:
            arr = np.load(cache).astype(np.float32)
        except (ValueError, OSError):
            return None
        if arr.shape != (h, w):
            import cv2 as _cv2
            arr = _cv2.resize(arr, (w, h), interpolation=_cv2.INTER_LINEAR)
        depths.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.stack(depths, dim=0)


def _build_background_anchor_mask(
    depth: torch.Tensor,
    bg_depth: torch.Tensor,
    person_bbox: "torch.Tensor | None",
    eps: float,
) -> torch.Tensor:
    """Use non-person valid pixels to align per-frame DA2 scale to the background cache."""
    valid = (
        torch.isfinite(depth)
        & torch.isfinite(bg_depth)
        & (depth.abs() > eps)
        & (bg_depth.abs() > eps)
    )
    if person_bbox is None:
        return valid

    mask = valid.clone()
    _, _, h, w = mask.shape
    boxes = person_bbox.detach().cpu()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if x2 <= x1 or y2 <= y1:
            continue
        bw = x2 - x1
        bh = y2 - y1
        x1 = max(0, int(np.floor(x1 - 0.25 * bw)))
        x2 = min(w, int(np.ceil(x2 + 0.25 * bw)))
        y1 = max(0, int(np.floor(y1 - 0.25 * bh)))
        y2 = min(h, int(np.ceil(y2 + 0.25 * bh)))
        if x2 > x1 and y2 > y1:
            mask[idx, :, y1:y2, x1:x2] = False

    too_small = mask.flatten(1).sum(dim=1) < 1024
    if bool(too_small.any().item()):
        mask[too_small] = valid[too_small]
    return mask


def _load_original_frame_batch(frame_paths: list[str], device: torch.device) -> "torch.Tensor | None":
    if not frame_paths:
        return None
    images = []
    for fp in frame_paths:
        if not fp or not os.path.exists(fp):
            return None
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(img).permute(2, 0, 1).contiguous())
    return torch.stack(images, dim=0).to(device=device, dtype=torch.uint8)


def _load_original_bg_depth_batch(bg_depth_paths: list[str], device: torch.device) -> "torch.Tensor | None":
    if not bg_depth_paths:
        return None
    depths = []
    for path in bg_depth_paths:
        if not path or not os.path.exists(path):
            return None
        try:
            arr = np.load(path).astype(np.float32)
        except (ValueError, OSError):
            return None
        if arr.ndim == 3:
            arr = arr[0]
        depths.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.stack(depths, dim=0).to(device=device, dtype=torch.float32)


def _crop_person_geometry_batch(
    height: torch.Tensor,
    person_mask: torch.Tensor,
    person_bbox: torch.Tensor,
    bg_depth: torch.Tensor,
    target_h: int,
    target_w: int,
    crop_expand_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cropped_height = []
    cropped_mask = []
    cropped_bbox = []
    cropped_bg = []

    for idx in range(height.shape[0]):
        m_map = person_mask[idx, 0].detach().cpu().numpy().astype(np.float32)
        bg_map = bg_depth[idx, 0].detach().cpu().numpy().astype(np.float32)
        bbox = person_bbox[idx].detach().cpu().numpy().astype(np.float32)
        src_h, src_w = height.shape[-2:]

        bbox = clamp_bbox_xyxy(expand_bbox_xyxy(bbox, crop_expand_ratio), src_w, src_h)
        x1, y1, x2, y2 = bbox.tolist()
        x1i = max(0, int(np.floor(x1)))
        y1i = max(0, int(np.floor(y1)))
        x2i = min(src_w, int(np.ceil(x2)))
        y2i = min(src_h, int(np.ceil(y2)))
        if x2i <= x1i or y2i <= y1i:
            x1i, y1i, x2i, y2i = 0, 0, src_w, src_h

        m_crop = m_map[y1i:y2i, x1i:x2i]
        bg_crop = bg_map[y1i:y2i, x1i:x2i]

        h_crop = height[idx : idx + 1, :, y1i:y2i, x1i:x2i]
        crop_h, crop_w = h_crop.shape[-2:]
        scale = min(float(target_h) / float(crop_h), float(target_w) / float(crop_w))
        scaled_h = max(1, int(round(crop_h * scale)))
        scaled_w = max(1, int(round(crop_w * scale)))
        h_resized = torch.nn.functional.interpolate(
            h_crop,
            size=(scaled_h, scaled_w),
            mode="bilinear",
            align_corners=False,
        )
        pad_top = (target_h - scaled_h) // 2
        pad_bottom = target_h - scaled_h - pad_top
        pad_left = (target_w - scaled_w) // 2
        pad_right = target_w - scaled_w - pad_left
        h_out = torch.nn.functional.pad(
            h_resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )[0]

        m_out, _ = resize_and_pad_map(m_crop, target_h=target_h, target_w=target_w, pad_value=0.0)
        bg_pad_value = float(np.median(bg_crop)) if bg_crop.size > 0 else 0.0
        bg_out, _ = resize_and_pad_map(bg_crop, target_h=target_h, target_w=target_w, pad_value=bg_pad_value)

        inner_bbox = np.array([
            float(pad_left),
            float(pad_top),
            pad_left + (float(person_bbox[idx, 2].item()) - float(person_bbox[idx, 0].item())) * scale,
            pad_top + (float(person_bbox[idx, 3].item()) - float(person_bbox[idx, 1].item())) * scale,
        ], dtype=np.float32)

        cropped_height.append(h_out)
        cropped_mask.append(torch.from_numpy((m_out > 0.5).astype(np.float32)).unsqueeze(0))
        cropped_bbox.append(torch.from_numpy(inner_bbox))
        cropped_bg.append(torch.from_numpy(bg_out).unsqueeze(0))

    return (
        torch.stack(cropped_height, dim=0).to(height.device, dtype=height.dtype),
        torch.stack(cropped_mask, dim=0).to(person_mask.device, dtype=person_mask.dtype),
        torch.stack(cropped_bbox, dim=0).to(person_bbox.device, dtype=person_bbox.dtype),
        torch.stack(cropped_bg, dim=0).to(bg_depth.device, dtype=bg_depth.dtype),
    )


def derive_height_batch(
    batch: dict,
    runtime_depth: RuntimeDepthEstimator,
    device: torch.device,
    eps: float,
    assume_inverse: bool,
    person_mask: "torch.Tensor | None" = None,
    person_bbox: "torch.Tensor | None" = None,
    crop_and_resize: bool = False,
    crop_expand_ratio: float = 0.0,
    use_ground_anchor: bool = True,
    train_depth: bool = False,
    allow_depth_cache: bool = True,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    raw = batch["image_raw"].to(device)
    bg = batch["bg_depth"].to(device)
    cam_h = batch["camera_height_m"].to(device).view(-1, 1, 1, 1)
    target_h, target_w = bg.shape[-2:]
    original_raw = None

    # Try loading precomputed depth cache (much faster than online DA2 inference)
    depth = None
    if allow_depth_cache and "frame_path" in batch and batch["frame_path"]:
        _, _, h, w = bg.shape
        cached = _load_depth_cache_batch(
            batch["frame_path"],
            h,
            w,
            depth_cache_paths=batch.get("depth_cache_path"),
        )
        if cached is not None:
            depth = cached.to(device)

    if depth is None:
        if "frame_path" in batch and batch["frame_path"]:
            original_raw = _load_original_frame_batch(batch["frame_path"], device)
        if original_raw is not None:
            depth = runtime_depth.forward_batch(original_raw) if train_depth else runtime_depth.infer_batch(original_raw)
        else:
            depth = runtime_depth.forward_batch(raw) if train_depth else runtime_depth.infer_batch(raw)

    bg_for_height = bg
    if original_raw is not None:
        if "bg_depth_path" in batch and batch["bg_depth_path"]:
            original_bg = _load_original_bg_depth_batch(batch["bg_depth_path"], device)
            if original_bg is not None:
                bg_for_height = original_bg

    if tuple(depth.shape[-2:]) != tuple(bg_for_height.shape[-2:]):
        depth = torch.nn.functional.interpolate(depth, size=bg_for_height.shape[-2:], mode="bilinear", align_corners=False)

    # Load precomputed SegFormer ground mask for DA2 scale alignment
    ground_mask = None
    if use_ground_anchor and "frame_path" in batch and batch["frame_path"]:
        _, _, h, w = depth.shape
        ground_mask = _load_ground_mask_cache_batch(batch["frame_path"], h, w, device)
    if use_ground_anchor and ground_mask is None:
        ground_mask = _build_background_anchor_mask(depth, bg_for_height, person_bbox, eps)

    height, _ = depth_to_height(
        depth=depth,
        bg_depth=bg_for_height,
        camera_height_m=cam_h,
        eps=eps,
        assume_inverse=assume_inverse,
        ground_mask=ground_mask,
    )
    if not crop_and_resize:
        if tuple(height.shape[-2:]) != (target_h, target_w):
            height = torch.nn.functional.interpolate(height, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return height

    if person_mask is None or person_bbox is None:
        raise ValueError("person_mask and person_bbox are required when crop_and_resize=True")
    if tuple(person_mask.shape[-2:]) != tuple(height.shape[-2:]):
        person_mask = torch.nn.functional.interpolate(person_mask.to(device), size=height.shape[-2:], mode="nearest")
    else:
        person_mask = person_mask.to(device)
    person_bbox = person_bbox.to(device)

    cropped_height, cropped_mask, cropped_bbox, cropped_bg = _crop_person_geometry_batch(
        height=height,
        person_mask=person_mask,
        person_bbox=person_bbox,
        bg_depth=bg_for_height,
        target_h=target_h,
        target_w=target_w,
        crop_expand_ratio=crop_expand_ratio,
    )
    return cropped_height, cropped_mask, cropped_bbox, cropped_bg


def compute_dense_rank_loss(
    derived: torch.Tensor,
    target_height: torch.Tensor,
    valid_mask: torch.Tensor,
    pair_logit: "torch.Tensor | None",
    pair_label: torch.Tensor,
    lambda_dense: float,
    bce_loss,
    eps: float,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    rmse_loss = masked_rmse(derived, target_height, valid_mask, eps=eps)
    total = rmse_loss * float(lambda_dense) + derived.sum() * 0.0
    rank_pairs = 0
    if pair_logit is not None and pair_label.numel() > 0:
        total = total + bce_loss(pair_logit, pair_label.float())
        rank_pairs = int(pair_label.numel())
    return total, rank_pairs, rmse_loss


def zero_trainable_loss(module: torch.nn.Module) -> torch.Tensor:
    model_ref = module.module if hasattr(module, "module") else module
    total = None
    for param in model_ref.parameters():
        if not param.requires_grad:
            continue
        zero = param.sum() * 0.0
        total = zero if total is None else total + zero
    if total is None:
        raise RuntimeError("No trainable model parameters are available for a zero loss fallback.")
    return total


def build_derived_frame_gallery(
    model: DerivedHeightRanker,
    dataset: HeightDataset,
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
    crop_expand_ratio: float,
    crop_and_resize_person: bool,
    use_ground_anchor: bool = True,
) -> list[dict]:
    records: list[dict] = []
    was_training = model.training
    model.eval()
    if not row_indices:
        model.train(was_training)
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
    for batch in loader:
        person_mask, person_bbox = _get_person_regions(batch, segmenter, device)
        derived_out = derive_height_batch(
            batch,
            runtime_depth,
            device,
            eps,
            assume_inverse,
            person_mask=person_mask,
            person_bbox=person_bbox,
            crop_and_resize=crop_and_resize_person,
            crop_expand_ratio=crop_expand_ratio,
            use_ground_anchor=use_ground_anchor,
        )
        if crop_and_resize_person:
            derived, person_mask, person_bbox, derived_bg = derived_out
        else:
            derived = derived_out
            derived_bg = batch["bg_depth"].to(device)
            person_mask = person_mask.to(device)
            person_bbox = person_bbox.to(device=device, dtype=torch.float32)
        keep = person_mask_is_valid(person_mask, min_valid_pixels=min_valid_pixels, min_valid_ratio=min_valid_ratio)
        if not bool(keep.any().item()):
            continue
        feats = model.encode_person(derived, person_mask, person_bbox, derived_bg).detach().cpu()
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


def _shard_indices(row_indices: list[int], rank: int, world_size: int) -> list[int]:
    if world_size <= 1:
        return row_indices
    return [row_indices[i] for i in range(rank, len(row_indices), world_size)]


def _gather_object_list(local_value, distributed: bool):
    if not distributed:
        return [local_value]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_value)
    return gathered


def _gallery_path(output_dir: str, suffix: str) -> str:
    return os.path.join(output_dir, f"train_gallery_{suffix}.pt")


@torch.no_grad()
def build_derived_video_gallery(
    model: DerivedHeightRanker,
    dataset: HeightDataset,
    device: torch.device,
    runtime_depth: RuntimeDepthEstimator,
    segmenter: PersonSegmenter,
    num_frames: int,
    min_valid_pixels: int,
    min_valid_ratio: float,
    eps: float,
    assume_inverse: bool,
    crop_expand_ratio: float,
    crop_and_resize_person: bool,
    use_ground_anchor: bool = True,
) -> list[dict]:
    model_ref = model.module if hasattr(model, "module") else model
    was_training = model_ref.training
    model_ref.eval()
    records: list[dict] = []

    # Deduplicate by sequence_id: each sequence produces exactly one record.
    seen_sequences: set[str] = set()
    for row in dataset.rows:
        if row.sequence_id in seen_sequences:
            continue
        seen_sequences.add(row.sequence_id)

        frame_features: list[torch.Tensor] = []
        sampled_frames: list[int] = []
        sequence_frame_indices = dataset._sequence_sampled_frames.get(row.sequence_id)
        if sequence_frame_indices:
            frame_indices = sequence_frame_indices[: max(num_frames, 1)]
        else:
            frame_indices = [row.frame_start]

        for frame_idx in frame_indices:
            sample, actual_frame_idx = dataset._build_sample(row, frame_idx)
            batch = {
                "image_raw": sample["image_raw"].unsqueeze(0),
                "bg_depth": sample["bg_depth"].unsqueeze(0),
                "camera_height_m": sample["camera_height_m"].view(1),
                "frame_path": [row.frame_path] if row.frame_path else [],
                "bg_depth_path": [row.bg_depth_path] if row.bg_depth_path else [],
            }
            person_mask, person_bbox = _get_person_regions(batch, segmenter, device)
            derived_out = derive_height_batch(
                batch=batch,
                runtime_depth=runtime_depth,
                device=device,
                eps=eps,
                assume_inverse=assume_inverse,
                person_mask=person_mask,
                person_bbox=person_bbox,
                crop_and_resize=crop_and_resize_person,
                crop_expand_ratio=crop_expand_ratio,
                use_ground_anchor=use_ground_anchor,
            )
            if crop_and_resize_person:
                derived, person_mask, person_bbox, derived_bg = derived_out
            else:
                derived = derived_out
                derived_bg = batch["bg_depth"].to(device)
                person_mask = person_mask.to(device)
                person_bbox = person_bbox.to(device=device, dtype=torch.float32)
            keep = person_mask_is_valid(
                person_mask,
                min_valid_pixels=min_valid_pixels,
                min_valid_ratio=min_valid_ratio,
            )
            # For gallery building, keep the feature even if mask is small:
            # use the actual mask if valid, otherwise fall back to a full-ones mask
            # so that every sampled frame contributes to the sequence feature.
            if not bool(keep[0].item()):
                person_mask = torch.ones_like(person_mask)
            feat = model_ref.encode_person(derived, person_mask, person_bbox, derived_bg)[0].detach().cpu()
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
                "feature": torch.stack(frame_features, dim=0).mean(dim=0),
            }
        )

    model_ref.train(was_training)
    return records


@torch.no_grad()
def evaluate(
    model: DerivedHeightRanker,
    query_ds: HeightDataset,
    gallery_ds: HeightDataset,
    device: torch.device,
    runtime_depth: RuntimeDepthEstimator,
    segmenter: PersonSegmenter,
    pairwise_labels: dict,
    cfg,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    return_records: bool = False,
) -> dict:
    depth_was_training = runtime_depth.model.training if runtime_depth is not None else False
    if runtime_depth is not None:
        runtime_depth.model.eval()
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
    query_indices_local = _shard_indices(query_indices, rank, world_size)
    gallery_indices_local = _shard_indices(gallery_indices, rank, world_size)
    query_records = build_derived_frame_gallery(
        model=model,
        dataset=query_ds,
        device=device,
        runtime_depth=runtime_depth,
        segmenter=segmenter,
        row_indices=query_indices_local,
        batch_size=cfg.eval.eval_batch_size,
        num_workers=cfg.train.num_workers,
        min_valid_pixels=cfg.loss.min_valid_pixels,
        min_valid_ratio=cfg.loss.min_valid_ratio,
        eps=cfg.loss.eps,
        assume_inverse=cfg.runtime_depth.assume_inverse,
        crop_expand_ratio=cfg.model.bbox_expand_ratio,
        crop_and_resize_person=getattr(cfg.model, "crop_and_resize_person", True),
        use_ground_anchor=getattr(cfg.runtime_depth, "use_ground_anchor", True),
    )
    gallery_records = build_derived_frame_gallery(
        model=model,
        dataset=gallery_ds,
        device=device,
        runtime_depth=runtime_depth,
        segmenter=segmenter,
        row_indices=gallery_indices_local,
        batch_size=cfg.eval.eval_batch_size,
        num_workers=cfg.train.num_workers,
        min_valid_pixels=cfg.loss.min_valid_pixels,
        min_valid_ratio=cfg.loss.min_valid_ratio,
        eps=cfg.loss.eps,
        assume_inverse=cfg.runtime_depth.assume_inverse,
        crop_expand_ratio=cfg.model.bbox_expand_ratio,
        crop_and_resize_person=getattr(cfg.model, "crop_and_resize_person", True),
        use_ground_anchor=getattr(cfg.runtime_depth, "use_ground_anchor", True),
    )
    gathered_query = _gather_object_list(query_records, distributed)
    gathered_gallery = _gather_object_list(gallery_records, distributed)
    if distributed:
        query_records = [item for chunk in gathered_query for item in (chunk or [])]
        gallery_records = [item for chunk in gathered_gallery for item in (chunk or [])]

    if distributed and not _is_main_process():
        metrics = None
    else:
        def _prob_fn(i: int, j: int) -> float:
            with torch.no_grad():
                logit = model.compare_encoded(
                    query_records[i]["feature"].unsqueeze(0).to(device),
                    gallery_records[j]["feature"].unsqueeze(0).to(device),
                )
            return float(torch.sigmoid(logit)[0].item())

        metrics = cross_split_pairwise_metrics(query_records, gallery_records, pairwise_labels, _prob_fn)

    if distributed:
        broadcast = [metrics]
        dist.broadcast_object_list(broadcast, src=0)
        metrics = broadcast[0]
    if runtime_depth is not None:
        runtime_depth.model.train(depth_was_training)
    if return_records:
        return {
            "summary": metrics,
            "query_records": query_records,
            "gallery_records": gallery_records,
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    distributed = False
    writer = None
    try:
        cfg = load_config(args.config)
        distributed, rank, world_size, device = _setup_distributed(cfg.device)
        set_seed(cfg.seed + rank)

        if not cfg.runtime_depth.enabled:
            raise RuntimeError("runtime_depth.enabled must be true for derived-height ranking.")
        if not cfg.runtime_seg.enabled:
            raise RuntimeError("runtime_seg.enabled must be true for derived-height ranking.")
        _require_file(cfg.loss.pairwise_json, "pairwise_json")
        allow_seg_fallback = bool(getattr(cfg.runtime_seg, "allow_inference_fallback", True))
        if allow_seg_fallback:
            _require_file(cfg.runtime_seg.model_path, "runtime segmentation model")
        _require_file(cfg.runtime_depth.checkpoint, "runtime depth checkpoint")

        if _is_main_process():
            ensure_dir(cfg.paths.output_dir)

        train_ds = _build_dataset(cfg, cfg.paths.train_manifest, cfg.paths.train_video_root, train_mode=True)
        val_ds = _build_dataset(cfg, cfg.paths.val_manifest, cfg.paths.val_video_root, train_mode=False)
        gallery_ds = _build_dataset(cfg, cfg.paths.train_manifest, cfg.paths.train_video_root, train_mode=False)
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=cfg.train.num_workers > 0,
            collate_fn=collate_fn,
        )

        model = DerivedHeightRanker(
            comparator_channels=cfg.model.comparator_channels,
            comparator_type=cfg.model.comparator_type,
            comparator_layers=cfg.model.comparator_layers,
            comparator_num_heads=cfg.model.comparator_num_heads,
            comparator_patch_size=cfg.model.comparator_patch_size,
            person_region_mode=cfg.model.person_region_mode,
            bbox_expand_ratio=cfg.model.bbox_expand_ratio,
            histogram_min=getattr(cfg.model, 'histogram_min', 0.0),
            histogram_max=getattr(cfg.model, 'histogram_max', 3.0),
            compare_type=getattr(cfg.model, 'compare_type', 'concat'),
            use_geometry_branch=getattr(cfg.model, 'use_geometry_branch', False),
            fusion_type=getattr(cfg.model, 'fusion_type', 'add'),
            geo_feat_dim=getattr(cfg.model, 'geo_feat_dim', 12),
            geo_hidden_dim=getattr(cfg.model, 'geo_hidden_dim', 32),
        ).to(device)
        if distributed:
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )
        model_ref = model.module if distributed else model

        runtime_depth = RuntimeDepthEstimator(
            depthanything_root=cfg.runtime_depth.depthanything_root,
            encoder=cfg.runtime_depth.encoder,
            checkpoint=cfg.runtime_depth.checkpoint,
            input_size=cfg.runtime_depth.input_size,
        ).to(device)
        train_depth = bool(getattr(cfg.runtime_depth, "trainable", False))
        if train_depth:
            runtime_depth.set_trainable(getattr(cfg.runtime_depth, "train_parts", "depth_head"))
            runtime_depth.model.train()
            if distributed:
                runtime_depth.model = DDP(
                    runtime_depth.model,
                    device_ids=[device.index],
                    output_device=device.index,
                    find_unused_parameters=True,
                )
        depth_trainable_params = runtime_depth.trainable_parameters() if train_depth else []
        optimizer_param_groups = [
            {
                "params": model.parameters(),
                "lr": cfg.train.lr,
                "initial_lr": cfg.train.lr,
                "name": "ranker",
            }
        ]
        if depth_trainable_params:
            optimizer_param_groups.append(
                {
                    "params": depth_trainable_params,
                    "lr": float(getattr(cfg.runtime_depth, "depth_lr", 1e-5)),
                    "initial_lr": float(getattr(cfg.runtime_depth, "depth_lr", 1e-5)),
                    "name": "da2_depth_head",
                }
            )
        optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            weight_decay=cfg.train.weight_decay,
        )
        if _is_main_process():
            ranker_params = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
            depth_params = sum(p.numel() for p in depth_trainable_params)
            print(
                f"[E2E] world_size={world_size} per_gpu_batch={cfg.train.batch_size} "
                f"ranker_trainable={ranker_params} da2_trainable={depth_params} "
                f"da2_trainable_enabled={train_depth}",
                flush=True,
            )

        scheduler_type = getattr(cfg.train, 'lr_scheduler', 'none').lower()
        warmup_epochs = getattr(cfg.train, 'warmup_epochs', 0)
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-7)
            if _is_main_process():
                print(f"[SCHEDULER] CosineAnnealingLR T_max={cfg.train.epochs}")
        elif scheduler_type == 'step':
            step_size = getattr(cfg.train, 'lr_step_size', 20)
            step_gamma = getattr(cfg.train, 'lr_step_gamma', 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma)
            if _is_main_process():
                print(f"[SCHEDULER] StepLR step={step_size} gamma={step_gamma}")
        warmup_lr_scale = 1.0

        scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.use_amp and device.type == "cuda")
        bce = torch.nn.BCEWithLogitsLoss()
        listwise_lambda = float(getattr(cfg.loss, "listwise_lambda", 0.0))
        if listwise_lambda > 0:
            raise RuntimeError("listwise_lambda > 0 requires ListNetLoss support, which is unavailable in this branch")
        listnet_fn = None
        pairwise_labels = HeightNetLoss.load_pairwise_labels(cfg.loss.pairwise_json)
        lambda_dense = float(getattr(cfg.loss, "lambda_rmse", 1.0))

        segmenter = None
        if allow_seg_fallback:
            segmenter = PersonSegmenter(
                model_path=cfg.runtime_seg.model_path,
                conf=cfg.runtime_seg.conf,
                iou=cfg.runtime_seg.iou,
                imgsz=cfg.runtime_seg.imgsz,
                strict_native=cfg.runtime_seg.strict_native,
            )
            segmenter.allow_inference_fallback = True

        if cfg.train.use_tensorboard and SummaryWriter is not None and _is_main_process():
            tb_dir = os.path.join(cfg.paths.output_dir, "tb")
            ensure_dir(tb_dir)
            writer = SummaryWriter(log_dir=tb_dir)

        best_acc = float("-inf")
        global_step = 0
        start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                print(f"[RESUME] Loading checkpoint from {args.resume}")
                ckpt = torch.load(args.resume, map_location=device, weights_only=False)
                model_ref.load_state_dict(ckpt["model"])
                if "runtime_depth" in ckpt:
                    runtime_depth.load_state_dict(ckpt["runtime_depth"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scaler.load_state_dict(ckpt["scaler"])
                start_epoch = ckpt.get("epoch", 0)
                global_step = ckpt.get("global_step", 0)
                best_acc = ckpt.get("best_pairwise_accuracy", float("-inf"))
                if scheduler is not None and "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                print(f"[RESUME] Resuming from epoch {start_epoch}, global_step={global_step}, best_acc={best_acc}")
            else:
                print(f"[RESUME] WARNING: checkpoint not found at {args.resume}, starting fresh")
        for epoch in range(start_epoch, cfg.train.epochs):
            model.train()
            if train_depth:
                runtime_depth.model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if _is_main_process():
                print(f"[TRAIN] epoch={epoch + 1} start", flush=True)
            iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}") if _is_main_process() else train_loader
            for batch in iterator:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=cfg.train.use_amp and device.type == "cuda"):
                    person_mask, person_bbox = _get_person_regions(batch, segmenter, device)
                    derived_out = derive_height_batch(
                        batch=batch,
                        runtime_depth=runtime_depth,
                        device=device,
                        eps=cfg.loss.eps,
                        assume_inverse=cfg.runtime_depth.assume_inverse,
                        person_mask=person_mask,
                        person_bbox=person_bbox,
                        crop_and_resize=getattr(cfg.model, "crop_and_resize_person", True),
                        crop_expand_ratio=cfg.model.bbox_expand_ratio,
                        use_ground_anchor=getattr(cfg.runtime_depth, "use_ground_anchor", True),
                        train_depth=train_depth,
                        allow_depth_cache=bool(getattr(cfg.runtime_depth, "use_depth_cache_during_train", True)),
                    )
                    if getattr(cfg.model, "crop_and_resize_person", True):
                        derived, person_mask, person_bbox, derived_bg = derived_out
                    else:
                        derived = derived_out
                        derived_bg = batch["bg_depth"].to(device)
                        person_mask = person_mask.to(device)
                        person_bbox = person_bbox.to(device=device, dtype=torch.float32)
                    idx_i, idx_j, pair_label = build_rank_pairs(
                        person_ids=batch["person_id"],
                        camera_ids=batch["camera_id"],
                        person_masks=person_mask,
                        pairwise_labels=pairwise_labels,
                        min_valid_pixels=cfg.loss.min_valid_pixels,
                        min_valid_ratio=cfg.loss.min_valid_ratio,
                        device=device,
                    )
                    pair_logit = None
                    dummy_pair_logit = None
                    if pair_label.numel() > 0:
                        model_idx_i = idx_i
                        model_idx_j = idx_j
                    else:
                        batch_size = int(derived.shape[0])
                        model_idx_i = torch.zeros(1, dtype=torch.long, device=device)
                        model_idx_j = torch.full(
                            (1,),
                            1 if batch_size > 1 else 0,
                            dtype=torch.long,
                            device=device,
                        )
                    out = model(
                        pair_inputs={
                            "height_map": derived,
                            "idx_i": model_idx_i,
                            "idx_j": model_idx_j,
                            "person_mask": person_mask,
                            "person_bbox": person_bbox,
                            "bg_depth": derived_bg,
                        }
                    )
                    if pair_label.numel() > 0:
                        pair_logit = out["pair_logit"]
                    else:
                        dummy_pair_logit = out["pair_logit"]

                    loss, rank_pairs, rmse_loss = compute_dense_rank_loss(
                        derived=derived,
                        target_height=batch["height"].to(device),
                        valid_mask=batch["mask"].to(device),
                        pair_logit=pair_logit,
                        pair_label=pair_label,
                        lambda_dense=lambda_dense,
                        bce_loss=bce,
                        eps=cfg.loss.eps,
                    )

                    if listwise_lambda > 0 and pair_label.numel() > 0:
                        feats_i = model_ref.encode_person(
                            derived, person_mask, person_bbox,
                            derived_bg,
                        ).index_select(0, idx_i)
                        feats_j = model_ref.encode_person(
                            derived, person_mask, person_bbox,
                            derived_bg,
                        ).index_select(0, idx_j)
                        pair_score = model_ref.compare_encoded(feats_i, feats_j)
                        person_scores_dict = {}
                        for k_idx in range(len(idx_i)):
                            cam = batch["camera_id"][idx_i[k_idx].item()]
                            pid_i = batch["person_id"][idx_i[k_idx].item()]
                            pid_j = batch["person_id"][idx_j[k_idx].item()]
                            person_scores_dict.setdefault(cam, {})
                            if pid_i not in person_scores_dict[cam]:
                                person_scores_dict[cam][pid_i] = []
                            if pid_j not in person_scores_dict[cam]:
                                person_scores_dict[cam][pid_j] = []
                            person_scores_dict[cam][pid_i].append(float(pair_score[k_idx].item()))
                            person_scores_dict[cam][pid_j].append(float(-pair_score[k_idx].item()))
                        ps_for_loss = {c: [(p, sum(v)/len(v)) for p, v in pv.items()] for c, pv in person_scores_dict.items()}
                        height_labels_path = getattr(cfg.loss, "height_labels_json", "")
                        if height_labels_path and os.path.exists(height_labels_path):
                            import json as _json
                            with open(height_labels_path) as hf:
                                hl = _json.load(hf)
                            l_loss = listnet_fn(ps_for_loss, hl)
                            loss = loss + listwise_lambda * l_loss

                if not loss.requires_grad:
                    loss = loss + zero_trainable_loss(model)
                if dummy_pair_logit is not None:
                    loss = loss + dummy_pair_logit.sum() * 0.0
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_params = list(model.parameters()) + list(depth_trainable_params)
                torch.nn.utils.clip_grad_norm_(clip_params, cfg.train.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

                if _is_main_process():
                    iterator.set_postfix(rank=float(loss.item()), rank_pairs=rank_pairs)
                    if writer is not None:
                        writer.add_scalar("loss/rmse_dense", float(rmse_loss.item()), global_step)
                        writer.add_scalar("loss/rank", float(loss.item()), global_step)
                        writer.add_scalar("train/rank_pairs", rank_pairs, global_step)

            should_validate = ((epoch + 1) % max(1, cfg.eval.validate_every_epochs) == 0) or (epoch + 1 == cfg.train.epochs)
            metrics = None
            if should_validate:
                metrics = evaluate(
                    model=model_ref,
                    query_ds=val_ds,
                    gallery_ds=gallery_ds,
                    device=device,
                    runtime_depth=runtime_depth,
                    segmenter=segmenter,
                    pairwise_labels=pairwise_labels,
                    cfg=cfg,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                )
            if _is_main_process():
                ckpt = {
                    "model": model_ref.state_dict(),
                    "runtime_depth": runtime_depth.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_pairwise_accuracy": best_acc,
                }
                if scheduler is not None:
                    ckpt["scheduler"] = scheduler.state_dict()
                torch.save(ckpt, os.path.join(cfg.paths.output_dir, "checkpoint_last.pt"))
                if should_validate and metrics is not None:
                    print(
                        f"[VAL] epoch={epoch + 1} pairwise_acc={metrics['pairwise_accuracy']:.4f} "
                        f"auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} "
                        f"n_pairs={metrics['n_pairs_eval']} n_cmp={metrics['n_comparisons']}"
                    )
                    if writer is not None:
                        writer.add_scalar("val/pairwise_accuracy", metrics["pairwise_accuracy"], epoch + 1)
                        writer.add_scalar("val/auc", metrics["auc"], epoch + 1)
                        writer.add_scalar("val/f1", metrics["f1"], epoch + 1)
                    if metrics["pairwise_accuracy"] > best_acc:
                        best_acc = float(metrics["pairwise_accuracy"])
                        ckpt["best_pairwise_accuracy"] = best_acc
                        torch.save(ckpt, os.path.join(cfg.paths.output_dir, "checkpoint_best.pt"))
                        if cfg.eval.save_train_gallery:
                            gallery_records = build_derived_video_gallery(
                                model=model_ref,
                                dataset=gallery_ds,
                                device=device,
                                runtime_depth=runtime_depth,
                                segmenter=segmenter,
                                num_frames=cfg.eval.video_feature_frames,
                                min_valid_pixels=cfg.loss.min_valid_pixels,
                                min_valid_ratio=cfg.loss.min_valid_ratio,
                                eps=cfg.loss.eps,
                                assume_inverse=cfg.runtime_depth.assume_inverse,
                                crop_expand_ratio=cfg.model.bbox_expand_ratio,
                                crop_and_resize_person=getattr(cfg.model, "crop_and_resize_person", True),
                                use_ground_anchor=getattr(cfg.runtime_depth, "use_ground_anchor", True),
                            )
                            save_video_feature_gallery(
                                _gallery_path(cfg.paths.output_dir, "best"),
                                gallery_records,
                                num_frames=cfg.eval.video_feature_frames,
                            )
                else:
                    print(f"[VAL] epoch={epoch + 1} skipped")
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_lr_scale = (epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = pg.get("initial_lr", pg["lr"]) * warmup_lr_scale
                if _is_main_process():
                    print(f"[WARMUP] epoch={epoch+1}/{warmup_epochs} lr_scale={warmup_lr_scale:.3f}")
            elif warmup_epochs > 0 and epoch == warmup_epochs:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg.get("initial_lr", pg["lr"])
                if _is_main_process():
                    print(f"[WARMUP] done, lr restored to {cfg.train.lr}")

            if scheduler is not None and epoch >= warmup_epochs:
                scheduler.step()
                if _is_main_process():
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"[SCHEDULER] epoch={epoch+1} lr={current_lr:.2e}")

        if writer is not None:
            writer.close()
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
