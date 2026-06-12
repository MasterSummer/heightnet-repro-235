from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.config import load_config
from heightnet.runtime_seg import PersonSegmenter


def person_frame_is_valid(
    person_mask: np.ndarray,
    min_person_pixels: int,
    min_person_area_ratio: float,
    center_margin_ratio: float,
) -> bool:
    mask = person_mask > 0.5
    h, w = mask.shape
    area = int(mask.sum())
    if area < int(min_person_pixels):
        return False
    if area / float(max(h * w, 1)) < float(min_person_area_ratio):
        return False

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return False
    cx = float(xs.min() + xs.max()) / 2.0 / float(max(w - 1, 1))
    cy = float(ys.min() + ys.max()) / 2.0 / float(max(h - 1, 1))
    m = float(center_margin_ratio)
    return m <= cx <= 1.0 - m and m <= cy <= 1.0 - m


def scan_valid_frames(
    video_path: str,
    segmenter: PersonSegmenter,
    device: torch.device,
    image_size: tuple[int, int],
    frame_start: int,
    frame_end: int,
    min_person_pixels: int,
    min_person_area_ratio: float,
    center_margin_ratio: float,
) -> list[int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    target_h, target_w = image_size
    valid_frames: list[int] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx < frame_start:
                idx += 1
                continue
            if idx > frame_end:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            image_raw = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
            person_mask = segmenter.infer_batch(image_raw, device)[0, 0].detach().cpu().numpy()
            if person_frame_is_valid(
                person_mask=person_mask,
                min_person_pixels=min_person_pixels,
                min_person_area_ratio=min_person_area_ratio,
                center_margin_ratio=center_margin_ratio,
            ):
                valid_frames.append(idx)
            idx += 1
    finally:
        cap.release()
    return valid_frames


def process_manifest(
    manifest_path: Path,
    out_manifest_path: Path,
    valid_frames_root: Path,
    segmenter: PersonSegmenter,
    device: torch.device,
    image_size: tuple[int, int],
    min_person_pixels: int,
    min_person_area_ratio: float,
    center_margin_ratio: float,
) -> dict:
    frame = pd.read_csv(manifest_path)
    out_rows = []
    dropped = []
    valid_frames_root.mkdir(parents=True, exist_ok=True)

    for row in frame.to_dict(orient="records"):
        valid_frames = scan_valid_frames(
            video_path=str(row["video_path"]),
            segmenter=segmenter,
            device=device,
            image_size=image_size,
            frame_start=int(row["frame_start"]),
            frame_end=int(row["frame_end"]),
            min_person_pixels=min_person_pixels,
            min_person_area_ratio=min_person_area_ratio,
            center_margin_ratio=center_margin_ratio,
        )
        if not valid_frames:
            dropped.append(
                {
                    "video_path": str(row["video_path"]),
                    "sequence_id": str(row["sequence_id"]),
                    "reason": "no_valid_person_frames",
                }
            )
            continue
        valid_path = valid_frames_root / f"{row['sequence_id']}.npy"
        np.save(valid_path, np.asarray(valid_frames, dtype=np.int32))
        row["valid_frames_path"] = str(valid_path.resolve())
        row["frame_start"] = int(valid_frames[0])
        row["frame_end"] = int(valid_frames[-1])
        out_rows.append(row)

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_manifest_path, index=False)
    return {
        "input_manifest": str(manifest_path.resolve()),
        "output_manifest": str(out_manifest_path.resolve()),
        "num_rows_in": int(len(frame)),
        "num_rows_out": int(len(out_rows)),
        "num_dropped": int(len(dropped)),
        "dropped": dropped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter manifests down to valid person frames and save valid frame indices per video.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", nargs="+", required=True, help="One or more manifest CSV files to filter.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory that will receive filtered manifests and valid frame .npy files.")
    parser.add_argument("--min-person-pixels", type=int, default=4096)
    parser.add_argument("--min-person-area-ratio", type=float, default=0.03)
    parser.add_argument("--center-margin-ratio", type=float, default=0.15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    segmenter = PersonSegmenter(
        model_path=cfg.runtime_seg.model_path,
        conf=cfg.runtime_seg.conf,
        iou=cfg.runtime_seg.iou,
        imgsz=cfg.runtime_seg.imgsz,
        strict_native=cfg.runtime_seg.strict_native,
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for manifest in args.manifest:
        manifest_path = Path(manifest).resolve()
        split_name = manifest_path.stem.replace("_manifest", "")
        out_manifest = out_dir / manifest_path.name
        valid_frames_root = out_dir / "valid_frames" / split_name
        report = process_manifest(
            manifest_path=manifest_path,
            out_manifest_path=out_manifest,
            valid_frames_root=valid_frames_root,
            segmenter=segmenter,
            device=device,
            image_size=tuple(cfg.data.image_size),
            min_person_pixels=args.min_person_pixels,
            min_person_area_ratio=args.min_person_area_ratio,
            center_margin_ratio=args.center_margin_ratio,
        )
        reports.append(report)
        print(
            f"[FILTER] {manifest_path.name}: kept={report['num_rows_out']}/{report['num_rows_in']} "
            f"dropped={report['num_dropped']}"
        )

    report_path = out_dir / "filter_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
