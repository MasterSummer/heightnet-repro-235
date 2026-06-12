from __future__ import annotations

import os

import cv2
import numpy as np
import torch


def person_mask_cache_path(frame_path: str) -> str:
    return frame_path + ".person_mask.npy"


def person_bbox_cache_path(frame_path: str) -> str:
    return frame_path + ".person_bbox.npy"


def load_person_region_cache(
    frame_path: str,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not frame_path:
        return None

    mask_path = person_mask_cache_path(frame_path)
    bbox_path = person_bbox_cache_path(frame_path)
    if not os.path.exists(mask_path) or not os.path.exists(bbox_path):
        return None

    try:
        mask = np.load(mask_path).astype(np.float32)
        bbox = np.load(bbox_path).astype(np.float32).reshape(-1)
    except (ValueError, OSError):
        return None

    if mask.ndim != 2 or bbox.shape[0] < 4:
        return None

    src_h, src_w = mask.shape
    if src_h <= 0 or src_w <= 0:
        return None

    if (src_h, src_w) != (target_h, target_w):
        scale_x = float(target_w) / float(src_w)
        scale_y = float(target_h) / float(src_h)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        bbox = bbox[:4].copy()
        bbox[[0, 2]] *= scale_x
        bbox[[1, 3]] *= scale_y
    else:
        bbox = bbox[:4].copy()

    bbox[0] = float(np.clip(bbox[0], 0.0, target_w))
    bbox[2] = float(np.clip(bbox[2], 0.0, target_w))
    bbox[1] = float(np.clip(bbox[1], 0.0, target_h))
    bbox[3] = float(np.clip(bbox[3], 0.0, target_h))
    return mask, bbox.astype(np.float32)


@torch.no_grad()
def infer_or_load_person_regions(
    images_raw: torch.Tensor,
    frame_paths: list[str],
    segmenter,
    device: torch.device,
    allow_inference_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images_raw.ndim != 4:
        raise ValueError(f"images_raw should be [B,3,H,W], got {tuple(images_raw.shape)}")

    batch_size, _, image_h, image_w = images_raw.shape
    if len(frame_paths) != batch_size:
        raise ValueError(f"frame_paths length {len(frame_paths)} does not match batch size {batch_size}")

    masks = torch.zeros((batch_size, 1, image_h, image_w), dtype=torch.float32, device=device)
    boxes = torch.full((batch_size, 4), -1.0, dtype=torch.float32, device=device)

    missing_indices: list[int] = []
    for idx, frame_path in enumerate(frame_paths):
        cached = load_person_region_cache(frame_path, target_h=image_h, target_w=image_w)
        if cached is None:
            missing_indices.append(idx)
            continue
        mask, bbox = cached
        masks[idx, 0] = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
        boxes[idx] = torch.from_numpy(bbox).to(device=device, dtype=torch.float32)

    if missing_indices and not allow_inference_fallback:
        examples = ", ".join(frame_paths[idx] for idx in missing_indices[:3])
        raise RuntimeError(
            f"missing person region cache for {len(missing_indices)} frame(s), "
            f"and inference fallback is disabled: {examples}"
        )

    if missing_indices:
        infer_masks, infer_boxes = segmenter.infer_batch_regions(images_raw[missing_indices], device)
        for offset, batch_idx in enumerate(missing_indices):
            masks[batch_idx] = infer_masks[offset]
            boxes[batch_idx] = infer_boxes[offset]

    return masks, boxes
