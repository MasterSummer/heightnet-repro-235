#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute DA2 depth npy files for videos.")
    p.add_argument("--video-root", required=True, type=str)
    p.add_argument("--depthanything-root", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"], type=str)
    p.add_argument("--input-size", default=518, type=int)
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--shard-index", default=0, type=int)
    p.add_argument("--num-shards", default=1, type=int)
    p.add_argument("--frame-step", default=1, type=int, help="Compute every Nth frame. Default computes all frames.")
    p.add_argument(
        "--output-root",
        default="",
        type=str,
        help="Optional mirrored output root. Default writes beside each video under .depth_cache.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-videos", default=0, type=int)
    p.add_argument("--summary-name", default="", type=str)
    return p.parse_args()


def torch_load_weights(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_da2_model(depthanything_root: str, checkpoint: str, encoder: str, device: torch.device):
    if depthanything_root not in sys.path:
        sys.path.insert(0, depthanything_root)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch_load_weights(checkpoint))
    return model.to(device).eval()


def discover_videos(video_root: Path) -> list[Path]:
    return sorted(p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def depth_path_for(video_root: Path, video_path: Path, frame_idx: int, output_root: Path | None) -> Path:
    stem = f"{video_path.stem}.frame_{frame_idx:06d}.depth.npy"
    if output_root is None:
        return video_path.parent / ".depth_cache" / stem
    rel_parent = video_path.parent.relative_to(video_root)
    return output_root / rel_parent / ".depth_cache" / stem


@torch.no_grad()
def compute_video(
    video_root: Path,
    video_path: Path,
    output_root: Path | None,
    model,
    input_size: int,
    frame_step: int,
    overwrite: bool,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "status": "open_failed", "saved": 0, "skipped": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    skipped = 0
    failed = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step != 0:
            idx += 1
            continue

        out_path = depth_path_for(video_root, video_path, idx, output_root)
        if out_path.exists() and not overwrite:
            skipped += 1
            idx += 1
            continue

        try:
            depth = model.infer_image(frame, input_size).astype(np.float32)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_name(out_path.name + ".tmp")
            with open(tmp_path, "wb") as f:
                np.save(f, depth)
            os.replace(str(tmp_path), str(out_path))
            saved += 1
        except Exception as exc:
            failed += 1
            print(f"[warn] failed frame video={video_path} frame={idx}: {exc}", flush=True)
        idx += 1

    cap.release()
    return {
        "video": str(video_path),
        "status": "ok" if failed == 0 else "partial",
        "total_frames": total_frames,
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    video_root = Path(args.video_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else None
    if not video_root.exists():
        raise FileNotFoundError(f"video root not found: {video_root}")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} shard={args.shard_index}/{args.num_shards}", flush=True)
    print(f"[init] video_root={video_root}", flush=True)
    print(f"[init] output_root={output_root or 'beside videos'}", flush=True)

    videos = discover_videos(video_root)
    shard_videos = videos[args.shard_index :: args.num_shards]
    if args.limit_videos > 0:
        shard_videos = shard_videos[: args.limit_videos]
    print(f"[scan] total_videos={len(videos)} shard_videos={len(shard_videos)}", flush=True)

    model = build_da2_model(args.depthanything_root, args.checkpoint, args.encoder, device)
    results = []
    totals = {"saved": 0, "skipped": 0, "failed": 0}
    for i, video_path in enumerate(shard_videos, 1):
        print(f"[video {i}/{len(shard_videos)}] {video_path}", flush=True)
        result = compute_video(
            video_root=video_root,
            video_path=video_path,
            output_root=output_root,
            model=model,
            input_size=args.input_size,
            frame_step=args.frame_step,
            overwrite=args.overwrite,
        )
        results.append(result)
        for key in totals:
            totals[key] += int(result.get(key, 0))
        print(
            f"[done] saved={result.get('saved', 0)} skipped={result.get('skipped', 0)} failed={result.get('failed', 0)}",
            flush=True,
        )

    summary = {
        "video_root": str(video_root),
        "output_root": str(output_root) if output_root else "",
        "encoder": args.encoder,
        "checkpoint": args.checkpoint,
        "input_size": args.input_size,
        "frame_step": args.frame_step,
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_total_videos": len(videos),
        "n_shard_videos": len(shard_videos),
        **totals,
        "results": results,
    }
    summary_name = args.summary_name or f"da2_depth_summary.shard{args.shard_index}.json"
    summary_dir = output_root if output_root is not None else video_root
    summary_path = summary_dir / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
