from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.gallery import uniform_sample_indices
from heightnet.losses import person_mask_is_valid
from heightnet.model import HeightNetTiny
from heightnet.runtime_depth import RuntimeDepthEstimator, depth_to_height
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir


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


def _torch_load_compat(path: str, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _middle_frame_idx(frame_start: int, frame_end: int) -> int:
    return frame_start + (frame_end - frame_start) // 2


def _iter_person_samples(datasets: List[Tuple[str, HeightDataset]], max_per_person: int) -> Iterable[tuple[str, str, object, HeightDataset]]:
    rows_by_person: dict[str, list[tuple[str, object, HeightDataset]]] = defaultdict(list)
    for split, dataset in datasets:
        for row in dataset.rows:
            rows_by_person[row.person_id].append((split, row, dataset))

    for person_id, rows in sorted(rows_by_person.items()):
        rows = sorted(rows, key=lambda item: (item[0], item[1].sequence_id))
        for idx in uniform_sample_indices(len(rows), max_per_person):
            split, row, dataset = rows[idx]
            yield person_id, split, row, dataset


def _save_panel(
    out_path: str,
    rgb_chw: torch.Tensor,
    bg_depth: torch.Tensor,
    runtime_depth: torch.Tensor,
    derived_height: torch.Tensor,
    valid_mask: torch.Tensor,
    person_mask: torch.Tensor,
    pred_height: torch.Tensor,
    title: str,
) -> None:
    rgb = rgb_chw.detach().cpu().numpy().transpose(1, 2, 0).astype("uint8")
    bg_np = bg_depth[0, 0].detach().cpu().numpy()
    depth_np = runtime_depth[0, 0].detach().cpu().numpy()
    derived_np = derived_height[0, 0].detach().cpu().numpy()
    valid_np = valid_mask[0, 0].detach().cpu().numpy()
    mask_np = person_mask[0, 0].detach().cpu().numpy()
    pred_np = pred_height[0, 0].detach().cpu().numpy()
    fg_pred_np = pred_np * (mask_np > 0.5)

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    panels = [
        (rgb, "RGB", None),
        (bg_np, "BG Depth", "viridis"),
        (depth_np, "Runtime Depth", "viridis"),
        (derived_np, "Derived Height", "viridis"),
        (valid_np, "Valid Mask", "gray"),
        (mask_np, "Person Mask", "gray"),
        (pred_np, "Pred Height", "viridis"),
        (fg_pred_np, "Foreground Pred Height", "viridis"),
    ]
    for axis, (image, name, cmap) in zip(ax.flatten(), panels):
        if cmap is None:
            axis.imshow(image)
        else:
            axis.imshow(image, cmap=cmap)
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export bg-depth/runtime-depth/height diagnostics for manual inspection.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--samples-per-person", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--index-out", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.runtime_depth.enabled:
        raise RuntimeError("runtime_depth.enabled must be true for supervision diagnostics")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

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
    runtime_depth = RuntimeDepthEstimator(
        depthanything_root=cfg.runtime_depth.depthanything_root,
        encoder=cfg.runtime_depth.encoder,
        checkpoint=cfg.runtime_depth.checkpoint,
        input_size=cfg.runtime_depth.input_size,
    ).to(device)

    datasets = [(split, _build_dataset(cfg, split)) for split in args.splits]
    exported_rows = []

    with torch.no_grad():
        for person_id, split, row, dataset in _iter_person_samples(datasets, max_per_person=max(args.samples_per_person, 1)):
            request_frame = _middle_frame_idx(row.frame_start, row.frame_end)
            sample, actual_frame = dataset._build_sample(row, request_frame)
            image = sample["image"].unsqueeze(0).to(device)
            image_raw = sample["image_raw"].unsqueeze(0)
            bg_depth = sample["bg_depth"].unsqueeze(0).to(device)
            cam_h = sample["camera_height_m"].view(1, 1, 1, 1).to(device)

            pred_height = model(image)["pred_height_map"]
            person_mask = segmenter.infer_batch(image_raw, device)
            runtime_depth_map = runtime_depth.infer_batch(image_raw.to(device))
            derived_height, valid_mask = depth_to_height(
                depth=runtime_depth_map,
                bg_depth=bg_depth,
                camera_height_m=cam_h,
                eps=cfg.loss.eps,
                assume_inverse=cfg.runtime_depth.assume_inverse,
            )
            keep = person_mask_is_valid(
                person_mask,
                min_valid_pixels=cfg.loss.min_valid_pixels,
                min_valid_ratio=cfg.loss.min_valid_ratio,
            )

            person_dir = os.path.join(args.out_dir, person_id)
            ensure_dir(person_dir)
            base_name = f"{split}__{row.sequence_id}__frame{actual_frame:06d}.png"
            out_path = os.path.join(person_dir, base_name)
            title = (
                f"{person_id} | {split} | {row.camera_id} | {row.sequence_id} | frame={actual_frame} | "
                f"mask_valid={bool(keep[0].item())}"
            )
            _save_panel(
                out_path=out_path,
                rgb_chw=sample["image_raw"],
                bg_depth=bg_depth,
                runtime_depth=runtime_depth_map,
                derived_height=derived_height,
                valid_mask=valid_mask,
                person_mask=person_mask,
                pred_height=pred_height,
                title=title,
            )
            exported_rows.append(
                {
                    "person_id": person_id,
                    "split": split,
                    "camera_id": row.camera_id,
                    "sequence_id": row.sequence_id,
                    "bg_depth_path": row.bg_depth_path,
                    "requested_frame": request_frame,
                    "actual_frame": actual_frame,
                    "mask_valid": bool(keep[0].item()),
                    "out_path": os.path.abspath(out_path),
                }
            )

    index_out = args.index_out or os.path.join(args.out_dir, "index.json")
    with open(index_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "checkpoint": os.path.abspath(args.checkpoint),
                "samples_per_person": int(args.samples_per_person),
                "num_exported": len(exported_rows),
                "rows": exported_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[EXPORT] saved {len(exported_rows)} diagnostic panels to {args.out_dir}")
    print(f"[EXPORT] index: {index_out}")


if __name__ == "__main__":
    main()
