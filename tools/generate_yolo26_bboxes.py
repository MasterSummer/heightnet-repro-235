#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VideoItem:
    person_id: str
    video_filename: str
    video_path: str
    split: str
    action: str

    @property
    def video_stem(self) -> str:
        return Path(self.video_filename).stem

    @property
    def json_stem(self) -> str:
        return f"{self.person_id}_{self.video_stem}"


def get_rank_world() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    if not 0 <= rank < world_size:
        rank = 0
    return rank, world_size


def load_manifest_rows(paths: list[Path], action_filter: str = "") -> list[VideoItem]:
    seen: set[str] = set()
    rows: list[VideoItem] = []
    action_filter = action_filter.strip().lower()
    for path in paths:
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            video_path = str(row.get("video_path", "")).strip()
            if not video_path or video_path in seen:
                continue
            action = str(row.get("action", "")).strip()
            if action_filter and action.lower() != action_filter:
                continue
            seen.add(video_path)
            rows.append(
                VideoItem(
                    person_id=str(row.get("person_id", "")).strip(),
                    video_filename=str(row.get("video_filename", Path(video_path).name)).strip(),
                    video_path=video_path,
                    split=str(row.get("split", "")).strip(),
                    action=action,
                )
            )
    return sorted(rows, key=lambda x: (x.person_id, x.video_filename, x.video_path))


def sample_frame_indices(num_frames: int, frames_per_video: int) -> list[int]:
    if num_frames <= 0:
        return []
    n = max(1, min(int(frames_per_video), int(num_frames)))
    if n == 1:
        return [min(num_frames // 2, num_frames - 1)]
    xs = np.linspace(0, num_frames - 1, num=n)
    out = sorted({int(round(float(x))) for x in xs})
    if len(out) < n:
        for idx in range(num_frames):
            if idx not in out:
                out.append(idx)
            if len(out) >= n:
                break
    return sorted(out[:n])


def read_video_frames(video_path: str, frame_indices: list[int]) -> tuple[list[np.ndarray], list[int], dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], {"reason": "open_failed"}
    frames: list[np.ndarray] = []
    decoded: list[int] = []
    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            decoded.append(int(frame_idx))
    finally:
        cap.release()
    return frames, decoded, {}


def choose_person_box(result: Any, image_w: int, image_h: int, min_box_height: int) -> dict[str, float] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy),), dtype=np.float32)
    best: dict[str, float] | None = None
    best_score = -1.0
    center_x = image_w * 0.5
    for idx, box in enumerate(xyxy):
        if int(round(float(cls[idx]))) != 0:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = float(np.clip(x1, 0.0, image_w))
        x2 = float(np.clip(x2, 0.0, image_w))
        y1 = float(np.clip(y1, 0.0, image_h))
        y2 = float(np.clip(y2, 0.0, image_h))
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if h < float(min_box_height) or w <= 0.0:
            continue
        area = w * h
        cx = x1 + 0.5 * w
        center_penalty = abs(cx - center_x) / max(float(image_w), 1.0)
        score = area * max(float(conf[idx]), 0.01) * (1.0 - 0.25 * center_penalty)
        if score > best_score:
            best_score = score
            best = {
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "score": float(conf[idx]),
            }
    return best


def build_scanner_json(item: VideoItem, detections: list[dict[str, Any]]) -> dict[str, Any]:
    payload_items = [
        {
            "frame_id": int(det["frame_id"]),
            "rect": [float(det["x"]), float(det["y"]), float(det["w"]), float(det["h"])],
            "score": float(det["score"]),
            "source": "yolo26",
        }
        for det in detections
    ]
    return {
        "0": {
            "video_path": item.video_path,
            "person_id": item.person_id,
            "video_stem": item.video_stem,
            "split": item.split,
            "action": item.action,
            "sub_track": [
                {
                    "track_id": 0,
                    "data": {
                        "0_0": payload_items,
                    },
                }
            ],
        }
    }


def process_video(item: VideoItem, model: Any, args: argparse.Namespace, device_id: str) -> tuple[str, dict[str, Any]]:
    out_path = Path(args.out_dir) / f"{item.json_stem}.json"
    if args.skip_existing and out_path.exists():
        return "skipped", {"video": item.video_path, "output": str(out_path)}

    cap = cv2.VideoCapture(item.video_path)
    if not cap.isOpened():
        return "failed", {"video": item.video_path, "reason": "open_failed"}
    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if num_frames <= 0 or image_w <= 0 or image_h <= 0:
        return "failed", {"video": item.video_path, "reason": "bad_video_meta"}

    frame_indices = sample_frame_indices(num_frames, int(args.frames_per_video))
    frames, decoded_indices, err = read_video_frames(item.video_path, frame_indices)
    if err:
        return "failed", {"video": item.video_path, **err}
    if not frames:
        return "failed", {"video": item.video_path, "reason": "no_decoded_frames"}

    detections: list[dict[str, Any]] = []
    for start in range(0, len(frames), int(args.batch_size)):
        batch = frames[start : start + int(args.batch_size)]
        batch_indices = decoded_indices[start : start + int(args.batch_size)]
        results = model.predict(
            source=batch,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            device=device_id,
            classes=[0],
            verbose=False,
        )
        for result, frame_idx in zip(results, batch_indices):
            box = choose_person_box(result, image_w=image_w, image_h=image_h, min_box_height=int(args.min_box_height))
            if box is None:
                continue
            detections.append({"frame_id": int(frame_idx), **box})

    if len(detections) < int(args.min_detections):
        status = "low_coverage"
    else:
        status = "ok"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_scanner_json(item, detections), ensure_ascii=False), encoding="utf-8")
    return status, {
        "video": item.video_path,
        "output": str(out_path),
        "frames_requested": len(frame_indices),
        "frames_decoded": len(frames),
        "detections": len(detections),
        "num_frames": num_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate main_card_out-compatible bbox JSONs with YOLO26 person detection.")
    parser.add_argument("--manifest", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--frames-per-video", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--min-box-height", type=int, default=20)
    parser.add_argument("--min-detections", type=int, default=5)
    parser.add_argument("--action-filter", default="")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=Path("runs/yolo26_bbox_all/reports"))
    args = parser.parse_args()

    from ultralytics import YOLO

    rank, world_size = get_rank_world()
    device_id = str(int(os.environ.get("LOCAL_RANK", str(rank))))
    print(f"[init] rank={rank} world_size={world_size} device={device_id} model={args.model}", flush=True)

    rows = load_manifest_rows(args.manifest, action_filter=str(args.action_filter))
    if args.limit_videos > 0:
        rows = rows[: int(args.limit_videos)]
    selected = rows[rank::world_size]
    print(f"[init] total_videos={len(rows)} selected={len(selected)} out_dir={args.out_dir}", flush=True)

    model = YOLO(str(args.model))
    stats: dict[str, int] = {"selected": len(selected), "ok": 0, "low_coverage": 0, "skipped": 0, "failed": 0}
    results: list[dict[str, Any]] = []
    started = time.time()
    for idx, item in enumerate(selected, 1):
        status, payload = process_video(item, model=model, args=args, device_id=device_id)
        stats[status] = stats.get(status, 0) + 1
        payload.update({"status": status, "person_id": item.person_id, "split": item.split, "action": item.action})
        results.append(payload)
        if idx % 25 == 0 or idx == len(selected):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[progress] rank={rank} processed={idx}/{len(selected)} "
                f"ok={stats.get('ok', 0)} low={stats.get('low_coverage', 0)} "
                f"failed={stats.get('failed', 0)} rate={idx / elapsed:.3f}/s",
                flush=True,
            )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"yolo26_bbox_report.rank{rank}.json"
    report_path.write_text(json.dumps({"stats": stats, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[summary] " + json.dumps(stats, ensure_ascii=False), flush=True)
    print(f"[report] {report_path}", flush=True)


if __name__ == "__main__":
    main()
