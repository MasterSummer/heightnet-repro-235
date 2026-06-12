from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


CACHE_RE = re.compile(r"(?P<stem>.+)\.frame_(?P<frame>\d{6})\.depth\.npy$")
CAM_RE = re.compile(r"_(?P<h>\d+d\d+)_(?P<a>\d+)_(?P<p>\d+w)_", re.I)


def base_camera(stem: str) -> str | None:
    m = CAM_RE.search(stem)
    if not m:
        return None
    return f"{m.group('h').lower()}_{m.group('a')}"


def decode_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok and frame is not None else None


def yolo_mask(model: YOLO, frame_bgr: np.ndarray, conf: float, device: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    results = model.predict(frame_bgr, conf=conf, verbose=False, device=device)
    if not results:
        return out
    res = results[0]
    if res.masks is not None and res.masks.data is not None:
        for seg in res.masks.data:
            seg = torch.nn.functional.interpolate(
                seg[None, None].float(), size=(h, w), mode="nearest"
            )[0, 0].detach().cpu().numpy()
            out |= seg > 0.5
    elif res.boxes is not None:
        for box in res.boxes.xyxy.detach().cpu().numpy().astype(int):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                out[y1:y2, x1:x2] = True
    return out


def colorize(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    canvas = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not valid.any():
        return canvas
    vals = depth[valid]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.zeros(depth.shape, dtype=np.uint8)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    canvas[valid] = heat[valid]
    return canvas


def preview(avg: np.ndarray, count: np.ndarray, out_png: Path) -> None:
    heat = colorize(avg)
    cov = np.zeros(avg.shape, dtype=np.uint8)
    if count.max() > 0:
        cov = np.clip(count / count.max() * 255, 0, 255).astype(np.uint8)
    cov = cv2.cvtColor(cov, cv2.COLOR_GRAY2BGR)
    unseen = np.zeros_like(heat)
    unseen[count <= 0] = (0, 0, 255)
    panel = cv2.hconcat([heat, cov, unseen])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), panel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/data2/dataset/jianzhi_2511/video2/1124_man1/.depth_cache")
    ap.add_argument("--video-root", default="/data2/dataset/jianzhi_2511/video2")
    ap.add_argument("--preferred-person", default="1124_man1")
    ap.add_argument("--out-root", default="/data1/zyding/jianzhi_2511_base_camera_bg_depth")
    ap.add_argument("--yolo-model", default="/home/zyding/height/Depth-Anything-V2/yolov8n-seg.pt")
    ap.add_argument("--max-frames-per-camera", type=int, default=120)
    ap.add_argument("--target-h", type=int, default=1440)
    ap.add_argument("--target-w", type=int, default=2560)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    grouped = defaultdict(list)
    for p in sorted(Path(args.cache_dir).glob("*.depth.npy")):
        m = CACHE_RE.match(p.name)
        if not m:
            continue
        stem = m.group("stem")
        cam = base_camera(stem)
        if not cam:
            continue
        grouped[cam].append((stem, int(m.group("frame")), p))

    model = YOLO(args.yolo_model)
    out_root = Path(args.out_root)
    summary = {}
    for cam, items in sorted(grouped.items()):
        if len(items) > args.max_frames_per_camera:
            idx = np.linspace(0, len(items) - 1, args.max_frames_per_camera).round().astype(int).tolist()
            items = [items[i] for i in idx]
        acc = np.zeros((args.target_h, args.target_w), dtype=np.float64)
        cnt = np.zeros((args.target_h, args.target_w), dtype=np.float64)
        used = 0
        misses = 0
        for stem, frame_idx, depth_path in items:
            video = Path(args.video_root) / args.preferred_person / f"{stem}.mp4"
            if not video.exists():
                matches = sorted(Path(args.video_root).glob(f"*/{stem}.mp4"))
                video = matches[0] if matches else video
            frame = decode_frame(video, frame_idx) if video.exists() else None
            if frame is None:
                misses += 1
                continue
            depth = np.load(depth_path, allow_pickle=True).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[0]
            if depth.shape[:2] != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = yolo_mask(model, frame, args.conf, args.device)
            depth = cv2.resize(depth, (args.target_w, args.target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), (args.target_w, args.target_h), interpolation=cv2.INTER_NEAREST) > 0
            valid = np.isfinite(depth) & (depth > 0) & (~mask)
            acc[valid] += depth[valid]
            cnt[valid] += 1
            used += 1
        avg = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(np.float32)
        cam_dir = out_root / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        np.save(cam_dir / f"{cam}_avg_depth.npy", avg)
        preview(avg, cnt, cam_dir / f"{cam}_preview.png")
        summary[cam] = {
            "cache_frames_available": len(grouped[cam]),
            "frames_selected": len(items),
            "frames_used": used,
            "decode_misses": misses,
            "coverage_ratio": float((cnt > 0).sum() / cnt.size),
            "npy": str(cam_dir / f"{cam}_avg_depth.npy"),
            "preview": str(cam_dir / f"{cam}_preview.png"),
        }
        print(cam, summary[cam], flush=True)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
