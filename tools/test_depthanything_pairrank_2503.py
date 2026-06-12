from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Zero-training test: Depth Anything V2 + h=C_h*(D_b-D_f)/D_b + pairwise rank."
    )
    parser.add_argument("--video-root", type=str, default=str(repo_root / "2503_test_videos"))
    parser.add_argument("--bg-depth-root", type=str, default=str(repo_root / "height" / "2503_test_bg_depthmap"))
    parser.add_argument("--pairwise-root", type=str, default=str(repo_root / "height" / "2503_test_pairwise_rank"))
    parser.add_argument("--out-json", type=str, default=str(repo_root / "height" / "analysis_runs" / "depthanything_v2_pairrank_2503.json"))
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--camera-height", type=float, default=6.4)
    parser.add_argument(
        "--camera-height-from-name",
        action="store_true",
        help="Parse C_h from camera id like 300cm_inside -> 3.0m (scaled by --camera-height-scale).",
    )
    parser.add_argument(
        "--camera-height-scale",
        type=float,
        default=0.01,
        help="Scale parsed numeric prefix in camera id (default 0.01: cm -> m).",
    )
    parser.add_argument("--frame-step", type=int, default=12)
    parser.add_argument("--max-frames-per-video", type=int, default=24)
    parser.add_argument("--min-positive-pixels", type=int, default=64)
    return parser.parse_args()


def setup_depthanything_import(repo_root: Path) -> None:
    da_root = repo_root / "Depth-Anything-V2"
    if not da_root.exists():
        raise FileNotFoundError(f"Depth-Anything-V2 not found: {da_root}")
    if str(da_root) not in sys.path:
        sys.path.insert(0, str(da_root))


def build_model(encoder: str, checkpoint: Path):
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    state_dict = _torch_load_compat(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval(), device


def infer_camera_id(text: str) -> str | None:
    m = re.search(r"(\d+cm_(?:inside|outside|slantside|side|front|back))", text, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def infer_camera_height(camera: str, default_height: float, use_from_name: bool, scale: float) -> float:
    if not use_from_name:
        return float(default_height)
    m = re.match(r"(\d+(?:\.\d+)?)cm_", camera, flags=re.IGNORECASE)
    if not m:
        return float(default_height)
    return float(m.group(1)) * float(scale)


def load_bg_depth_map(bg_depth_root: Path, camera: str, target_hw: Tuple[int, int]) -> np.ndarray | None:
    p = bg_depth_root / camera / f"{camera}_avg_depth.npy"
    if not p.exists():
        return None
    arr = np.load(str(p)).astype(np.float32)
    h, w = target_hw
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return arr


def sample_video_depths(model, video_path: Path, input_size: int, frame_step: int, max_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    depths: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            d = model.infer_image(frame, input_size).astype(np.float32)
            depths.append(d)
            if len(depths) >= max_frames:
                break
        idx += 1
    cap.release()
    return depths


def height_map_from_depth(d_f: np.ndarray, d_b: np.ndarray, camera_height: float) -> np.ndarray:
    valid = np.isfinite(d_f) & np.isfinite(d_b) & (np.abs(d_b) > 1e-6)
    h = np.zeros_like(d_f, dtype=np.float32)
    h[valid] = camera_height * (d_b[valid] - d_f[valid]) / d_b[valid]
    return np.clip(h, 0.0, camera_height * 3.0)


def frame_height_score(hmap: np.ndarray, min_positive_pixels: int) -> float:
    pos = hmap[hmap > 0]
    if pos.size < min_positive_pixels:
        return 0.0
    k = max(32, int(pos.size * 0.002))
    if k >= pos.size:
        top = pos
    else:
        top = np.partition(pos, pos.size - k)[-k:]
    return float(np.median(top))


def evaluate_pairwise(
    pairwise_root: Path, scores_by_camera: Dict[str, Dict[str, float]]
) -> Tuple[Dict[str, dict], dict]:
    per_camera = {}
    all_correct = 0
    all_total = 0

    for p in sorted(pairwise_root.glob("camera_*_pairs.json")):
        camera = p.name.replace("camera_", "").replace("_pairs.json", "")
        with open(p, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        cam_scores = scores_by_camera.get(camera, {})
        correct = 0
        total = 0
        covered_ids = set()
        for item in pairs:
            i = item.get("id_i")
            j = item.get("id_j")
            y = int(item.get("y", 1))
            if i not in cam_scores or j not in cam_scores:
                continue
            pred = 1 if cam_scores[i] > cam_scores[j] else 0
            if pred == y:
                correct += 1
            total += 1
            covered_ids.add(i)
            covered_ids.add(j)
        acc = (correct / total) if total > 0 else 0.0
        per_camera[camera] = {
            "acc": acc,
            "correct": correct,
            "total": total,
            "n_ids_covered": len(covered_ids),
            "n_ids_with_scores": len(cam_scores),
        }
        all_correct += correct
        all_total += total

    acc_values = [x["acc"] for x in per_camera.values() if x["total"] > 0]
    summary = {
        "pairwise_acc_weighted": (all_correct / all_total) if all_total > 0 else 0.0,
        "pairwise_acc_macro": (sum(acc_values) / len(acc_values)) if acc_values else 0.0,
        "pairwise_correct": all_correct,
        "pairwise_total": all_total,
        "cameras_evaluated": len([1 for x in per_camera.values() if x["total"] > 0]),
        "cameras_with_scores": len([1 for x in per_camera.values() if x["n_ids_with_scores"] > 0]),
    }
    return per_camera, summary


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    setup_depthanything_import(repo_root)

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (repo_root / "Depth-Anything-V2" / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth")
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    video_root = Path(args.video_root).expanduser().resolve()
    bg_depth_root = Path(args.bg_depth_root).expanduser().resolve()
    pairwise_root = Path(args.pairwise_root).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    model, device = build_model(args.encoder, checkpoint)
    print(f"[INFO] device={device}")
    print(f"[INFO] video_root={video_root}")
    print(f"[INFO] bg_depth_root={bg_depth_root}")

    scores_by_camera: Dict[str, Dict[str, float]] = {}
    debug_by_video = {}

    video_files = sorted(video_root.glob("*/*.mp4"))
    if not video_files:
        raise RuntimeError(f"No mp4 files found in {video_root}")

    for idx, video_path in enumerate(video_files, start=1):
        person_id = video_path.parent.name
        camera = infer_camera_id(video_path.name)
        if not camera:
            continue

        depths = sample_video_depths(
            model=model,
            video_path=video_path,
            input_size=args.input_size,
            frame_step=max(1, args.frame_step),
            max_frames=max(1, args.max_frames_per_video),
        )
        if not depths:
            continue

        h, w = depths[0].shape
        bg_map = load_bg_depth_map(bg_depth_root, camera, (h, w))
        if bg_map is None:
            continue
        camera_h = infer_camera_height(
            camera=camera,
            default_height=args.camera_height,
            use_from_name=args.camera_height_from_name,
            scale=args.camera_height_scale,
        )

        frame_scores = []
        for d_f in depths:
            hmap = height_map_from_depth(d_f, bg_map, camera_h)
            s = frame_height_score(hmap, min_positive_pixels=args.min_positive_pixels)
            frame_scores.append(s)
        if not frame_scores:
            continue

        video_score = float(np.median(np.array(frame_scores, dtype=np.float32)))
        scores_by_camera.setdefault(camera, {}).setdefault(person_id, [])
        scores_by_camera[camera][person_id].append(video_score)
        debug_by_video[str(video_path)] = {
            "person_id": person_id,
            "camera": camera,
            "camera_height": camera_h,
            "n_sampled_frames": len(depths),
            "video_score": video_score,
        }
        if idx % 10 == 0:
            print(f"[INFO] processed {idx}/{len(video_files)} videos")

    # Aggregate multi-video same (camera, person) by mean.
    for camera, m in list(scores_by_camera.items()):
        for pid, vals in list(m.items()):
            m[pid] = float(np.mean(np.array(vals, dtype=np.float32)))

    per_camera_eval, summary = evaluate_pairwise(pairwise_root, scores_by_camera)

    result = {
        "config": {
            "video_root": str(video_root),
            "bg_depth_root": str(bg_depth_root),
            "pairwise_root": str(pairwise_root),
            "encoder": args.encoder,
            "checkpoint": str(checkpoint),
            "input_size": args.input_size,
            "camera_height": args.camera_height,
            "camera_height_from_name": args.camera_height_from_name,
            "camera_height_scale": args.camera_height_scale,
            "frame_step": args.frame_step,
            "max_frames_per_video": args.max_frames_per_video,
            "min_positive_pixels": args.min_positive_pixels,
        },
        "summary": summary,
        "per_camera_eval": per_camera_eval,
        "scores_by_camera": scores_by_camera,
        "debug_by_video": debug_by_video,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[RESULT] summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[RESULT] written: {out_json}")


if __name__ == "__main__":
    main()
