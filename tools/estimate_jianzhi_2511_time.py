from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.config import load_config
from heightnet.jianzhi_parsing import build_jianzhi_parsing_index, generate_person_region_cache_from_index
from heightnet.model import DerivedHeightRanker
from heightnet.runtime_depth import RuntimeDepthEstimator


def _sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in indices]


def _decode_one(row: dict):
    frame_path = str(row.get("frame_path", "")).strip()
    if frame_path and os.path.exists(frame_path):
        return cv2.imread(frame_path, cv2.IMREAD_COLOR)
    video_path = str(row.get("video_path", "")).strip()
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(row.get("frame_idx", row.get("frame_start", 0))))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _time_decode(rows: list[dict]) -> dict:
    t0 = time.perf_counter()
    ok = 0
    for row in rows:
        frame = _decode_one(row)
        ok += int(frame is not None)
    elapsed = time.perf_counter() - t0
    return _stage_result(elapsed, len(rows), ok)


def _time_cache(rows: list[dict], parsing_root: str, overwrite_existing: bool) -> dict:
    t0 = time.perf_counter()
    index = build_jianzhi_parsing_index(parsing_root)
    build_elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    report = generate_person_region_cache_from_index(rows, index, overwrite_existing=overwrite_existing)
    cache_elapsed = time.perf_counter() - t1
    result = _stage_result(cache_elapsed, len(rows), int(report["generated"]) + int(report["skipped_existing"]))
    result["index_seconds"] = build_elapsed
    result["num_index_records"] = len(index.records)
    result["cache_report"] = report
    return result


def _time_depth_cache(rows: list[dict], cfg_path: str, batch_size: int, write_cache: bool) -> dict:
    cfg = load_config(cfg_path)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    runtime_depth = RuntimeDepthEstimator(
        depthanything_root=cfg.runtime_depth.depthanything_root,
        encoder=cfg.runtime_depth.encoder,
        checkpoint=cfg.runtime_depth.checkpoint,
        input_size=cfg.runtime_depth.input_size,
    ).to(device)
    target_h, target_w = [int(x) for x in cfg.data.image_size]
    ok = 0
    t0 = time.perf_counter()
    for start in range(0, len(rows), max(1, int(batch_size))):
        images = []
        paths = []
        for row in rows[start : start + max(1, int(batch_size))]:
            frame = _decode_one(row)
            if frame is None:
                continue
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous())
            paths.append(str(row.get("frame_path", "")).strip())
        if not images:
            continue
        batch = torch.stack(images, dim=0).to(device=device, dtype=torch.uint8)
        depth = runtime_depth.infer_batch(batch).detach().cpu().numpy()
        ok += int(depth.shape[0])
        if write_cache:
            for idx, path in enumerate(paths):
                if path:
                    npy = depth[idx, 0].astype("float32")
                    np.save(path + ".depth.npy", npy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return _stage_result(elapsed, len(rows), ok)


def _time_synthetic_train_step(cfg_path: str) -> dict:
    cfg = load_config(cfg_path)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    b = max(2, int(cfg.train.batch_size))
    h, w = [int(x) for x in cfg.data.image_size]
    model = DerivedHeightRanker(
        comparator_channels=cfg.model.comparator_channels,
        comparator_type=cfg.model.comparator_type,
        comparator_layers=cfg.model.comparator_layers,
        comparator_num_heads=cfg.model.comparator_num_heads,
        comparator_patch_size=cfg.model.comparator_patch_size,
        person_region_mode=cfg.model.person_region_mode,
        bbox_expand_ratio=cfg.model.bbox_expand_ratio,
        histogram_min=cfg.model.histogram_min,
        histogram_max=cfg.model.histogram_max,
        compare_type=cfg.model.compare_type,
        use_geometry_branch=cfg.model.use_geometry_branch,
        geo_feat_dim=cfg.model.geo_feat_dim,
        geo_hidden_dim=cfg.model.geo_hidden_dim,
    ).to(device)
    model.train()
    height = torch.rand((b, 1, h, w), dtype=torch.float32, device=device)
    mask = torch.ones((b, 1, h, w), dtype=torch.float32, device=device)
    bg = torch.rand((b, 1, h, w), dtype=torch.float32, device=device) + 1.0
    bbox = torch.tensor([[0.0, 0.0, float(w), float(h)] for _ in range(b)], device=device)
    idx_i = torch.arange(0, b - 1, device=device)
    idx_j = torch.arange(1, b, device=device)
    pair_label = torch.ones((b - 1,), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    out = model(
        {
            "height_map": height,
            "person_mask": mask,
            "person_bbox": bbox,
            "bg_depth": bg,
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["pair_logit"], pair_label)
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "seconds": elapsed,
        "batch_size": b,
        "seconds_per_sample": elapsed / float(b),
        "device": str(device),
        "loss": float(loss.detach().cpu().item()),
    }


def _stage_result(elapsed: float, n: int, ok: int) -> dict:
    return {
        "seconds": elapsed,
        "items": int(n),
        "ok": int(ok),
        "seconds_per_item": elapsed / max(float(n), 1.0),
        "items_per_second": float(n) / elapsed if elapsed > 0 else 0.0,
    }


def _project(stage: dict, total: int) -> dict:
    return {
        "estimated_total_seconds": float(stage.get("seconds_per_item", 0.0)) * float(total),
        "estimated_total_hours": float(stage.get("seconds_per_item", 0.0)) * float(total) / 3600.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate jianzhi_2511 preprocessing and training step timing.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--parsing-json-root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sample-sizes", type=str, default="100,500,2000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache-sample", action="store_true")
    parser.add_argument("--include-depth-cache-time", action="store_true")
    parser.add_argument("--write-depth-cache-sample", action="store_true")
    parser.add_argument("--depth-batch-size", type=int, default=4)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    rows = frame.to_dict(orient="records")
    sample_sizes = [int(x) for x in str(args.sample_sizes).split(",") if x.strip()]
    results = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "total_manifest_rows": len(rows),
        "sample_sizes": {},
    }

    for n in sample_sizes:
        sample = _sample_rows(rows, n, int(args.seed) + n)
        decode = _time_decode(sample)
        cache = _time_cache(sample, args.parsing_json_root, overwrite_existing=bool(args.overwrite_cache_sample))
        stage = {
            "decode": decode,
            "person_region_cache": cache,
            "projected_decode": _project(decode, len(rows)),
            "projected_person_region_cache": _project(cache, len(rows)),
        }
        if args.include_depth_cache_time:
            depth_cache = _time_depth_cache(
                sample,
                args.config,
                batch_size=int(args.depth_batch_size),
                write_cache=bool(args.write_depth_cache_sample),
            )
            stage["depth_cache"] = depth_cache
            stage["projected_depth_cache"] = _project(depth_cache, len(rows))
        results["sample_sizes"][str(n)] = stage
        print(
            f"[TIME] n={n} decode={decode['seconds_per_item']:.4f}s/item "
            f"cache={cache['seconds_per_item']:.4f}s/item ok_cache={cache['ok']}/{cache['items']}"
        )

    results["synthetic_train_step"] = _time_synthetic_train_step(args.config)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[TIME_REPORT] {out}")


if __name__ == "__main__":
    main()
