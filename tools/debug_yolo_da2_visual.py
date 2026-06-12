from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from heightnet.datasets import HeightDataset
from heightnet.image_ops import letterbox_gray
from heightnet.person_cache import person_bbox_cache_path, person_mask_cache_path
from heightnet.runtime_depth import depth_to_height
from heightnet.runtime_seg import PersonSegmenter


def put_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.42) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def fit_cell(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(cell_w / w, (cell_h - 18) / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell_h, cell_w, 3), 255, np.uint8)
    y0 = 18 + (cell_h - 18 - nh) // 2
    x0 = (cell_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def heatmap(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = arr.astype(np.float32)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask.astype(bool)
    if valid.any():
        lo, hi = np.percentile(values[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((values - lo) / (hi - lo), 0, 1)
    else:
        norm = np.zeros_like(values, np.float32)
    norm[~np.isfinite(norm)] = 0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    active = mask.astype(bool)
    if active.any():
        color_img = np.zeros_like(out)
        color_img[:] = color
        out[active] = (0.55 * out[active] + 0.45 * color_img[active]).astype(np.uint8)
    return out


def pick_rows(cfg: dict, n_per_split: int) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for split in ["train", "test"]:
        manifest = pd.read_csv(cfg["paths"][f"{split}_manifest"])
        for pid in manifest["person_id"].drop_duplicates().head(n_per_split):
            sub = manifest[manifest["person_id"] == pid]
            one = sub.iloc[len(sub) // 2]
            rows.append((split, int(one.name)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-per-split", type=int, default=3)
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    image_h, image_w = map(int, cfg["data"]["image_size"])
    rows = pick_rows(cfg, args.n_per_split)

    seg_cfg = cfg["runtime_seg"]
    segmenter = PersonSegmenter(
        model_path=seg_cfg["model_path"],
        conf=float(seg_cfg.get("conf", 0.25)),
        iou=float(seg_cfg.get("iou", 0.7)),
        imgsz=int(seg_cfg.get("imgsz", 640)),
        strict_native=bool(seg_cfg.get("strict_native", True)),
    )

    cell_w, cell_h = 260, 150
    cols = [
        "frame",
        "YOLO mask+bbox",
        "parsing mask+bbox",
        "DA2 depth",
        "bg depth",
        "height inv=True",
        "height inv=False",
    ]
    grid = np.full((cell_h * (len(rows) + 1), cell_w * len(cols), 3), 255, np.uint8)
    for ci, name in enumerate(cols):
        put_text(grid, name, ci * cell_w + 8, 14, scale=0.45)

    stats = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets: dict[str, HeightDataset] = {}
    for ri, (split, midx) in enumerate(rows, start=1):
        if split not in datasets:
            datasets[split] = HeightDataset(
                cfg["paths"][f"{split}_manifest"],
                tuple(cfg["data"]["image_size"]),
                normalize_rgb=bool(cfg["data"].get("normalize_rgb", True)),
                use_pair_consistency=False,
                train_mode=False,
            )
        sample = datasets[split][midx]

        frame_rgb = sample["image_raw"].permute(1, 2, 0).numpy().astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        y_masks, y_boxes = segmenter.infer_batch_regions(sample["image_raw"].unsqueeze(0).contiguous(), device)
        ymask = y_masks[0, 0].detach().cpu().numpy() > 0.5
        ybbox = y_boxes[0].detach().cpu().numpy()

        mask_path = person_mask_cache_path(sample["frame_path"])
        bbox_path = person_bbox_cache_path(sample["frame_path"])
        parsing_mask = np.load(mask_path).astype(bool)
        parsing_bbox = np.load(bbox_path).astype(np.float32).astype(int).tolist()
        if parsing_mask.shape != (image_h, image_w):
            parsing_mask = cv2.resize(
                parsing_mask.astype(np.uint8), (image_w, image_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        depth = np.load(sample["depth_cache_path"]).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[0]
        if depth.shape != (image_h, image_w):
            depth = letterbox_gray(depth, image_h, image_w, interpolation=cv2.INTER_LINEAR, fill_value=0.0)
        bg_depth = sample["bg_depth"][0].numpy().astype(np.float32)

        depth_t = torch.from_numpy(depth)[None, None]
        bg_t = torch.from_numpy(bg_depth)[None, None]
        cam_t = sample["camera_height_m"].reshape(1, 1, 1, 1)
        height_inv, _ = depth_to_height(depth_t, bg_t, cam_t, assume_inverse=True)
        height_raw, _ = depth_to_height(depth_t, bg_t, cam_t, assume_inverse=False)
        height_inv_np = height_inv[0, 0].numpy()
        height_raw_np = height_raw[0, 0].numpy()

        title = f"{split} {sample['person_id']} f{sample['frame_idx']}"
        visuals = []
        frame_cell = frame_bgr.copy()
        put_text(frame_cell, title, 4, 14, scale=0.38)
        visuals.append(frame_cell)

        yolo_cell = overlay_mask(frame_bgr, ymask, (0, 255, 0))
        if ybbox[0] >= 0:
            x1, y1, x2, y2 = map(int, ybbox)
            cv2.rectangle(yolo_cell, (x1, y1), (x2, y2), (0, 255, 255), 2)
        visuals.append(yolo_cell)

        parsing_cell = overlay_mask(frame_bgr, parsing_mask, (255, 0, 255))
        if parsing_bbox[0] >= 0:
            x1, y1, x2, y2 = map(int, parsing_bbox)
            cv2.rectangle(parsing_cell, (x1, y1), (x2, y2), (0, 255, 255), 2)
        visuals.extend(
            [
                parsing_cell,
                heatmap(depth),
                heatmap(bg_depth),
                heatmap(height_inv_np, parsing_mask),
                heatmap(height_raw_np, parsing_mask),
            ]
        )

        for ci, visual in enumerate(visuals):
            cell = fit_cell(visual, cell_w, cell_h)
            if ci > 0:
                put_text(cell, title, 4, 14, scale=0.35)
            grid[ri * cell_h : (ri + 1) * cell_h, ci * cell_w : (ci + 1) * cell_w] = cell

        stats.append(
            {
                "title": title,
                "frame_path": sample["frame_path"],
                "depth_cache_path": sample["depth_cache_path"],
                "yolo_bbox": [round(float(x), 1) for x in ybbox.tolist()],
                "yolo_mask_px": int(ymask.sum()),
                "parsing_bbox": parsing_bbox,
                "parsing_mask_px": int(parsing_mask.sum()),
                "da2_min_p50_max": [
                    float(np.nanmin(depth)),
                    float(np.nanmedian(depth)),
                    float(np.nanmax(depth)),
                ],
                "bg_min_p50_max": [
                    float(np.nanmin(bg_depth)),
                    float(np.nanmedian(bg_depth)),
                    float(np.nanmax(bg_depth)),
                ],
                "height_inv_max_in_person": float(np.nanmax(height_inv_np[parsing_mask])) if parsing_mask.any() else 0.0,
                "height_raw_max_in_person": float(np.nanmax(height_raw_np[parsing_mask])) if parsing_mask.any() else 0.0,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), grid)
    print(f"wrote {output}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
