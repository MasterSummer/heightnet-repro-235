from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.config import load_config
from heightnet.runtime_seg import PersonSegmenter


def person_box_is_valid(box: np.ndarray) -> bool:
    if box.shape[0] < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return x2 > x1 and y2 > y1


def load_rgb_tensor(frame_path: str) -> torch.Tensor:
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"cannot open frame image: {frame_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    chw = np.transpose(frame, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def filter_frame_rows_person_present(
    rows: Iterable[dict],
    segmenter: PersonSegmenter,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    dropped: list[dict] = []
    for row in rows:
        frame_path = str(row.get("frame_path", "")).strip()
        if not frame_path:
            dropped.append({**row, "drop_reason": "missing_frame_path"})
            continue
        try:
            image_raw = load_rgb_tensor(frame_path)
            _, person_bbox = segmenter.infer_batch_regions(image_raw, device)
            box = person_bbox[0].detach().cpu().numpy()
            if person_box_is_valid(box):
                kept.append(dict(row))
            else:
                dropped.append({**row, "drop_reason": "no_person_detected"})
        except Exception as exc:
            dropped.append({**row, "drop_reason": f"error:{exc.__class__.__name__}"})
    return kept, dropped


def process_manifest(
    manifest_path: Path,
    out_manifest_path: Path,
    segmenter: PersonSegmenter,
    device: torch.device,
) -> dict:
    frame = pd.read_csv(manifest_path)
    kept_rows, dropped_rows = filter_frame_rows_person_present(
        rows=frame.to_dict(orient="records"),
        segmenter=segmenter,
        device=device,
    )
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kept_rows).to_csv(out_manifest_path, index=False)
    return {
        "input_manifest": str(manifest_path.resolve()),
        "output_manifest": str(out_manifest_path.resolve()),
        "num_rows_in": int(len(frame)),
        "num_rows_out": int(len(kept_rows)),
        "num_dropped": int(len(dropped_rows)),
        "dropped": dropped_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter an extracted frame-level manifest to frames where YOLO detects at least one person."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
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
        out_manifest = out_dir / manifest_path.name
        report = process_manifest(
            manifest_path=manifest_path,
            out_manifest_path=out_manifest,
            segmenter=segmenter,
            device=device,
        )
        reports.append(report)
        print(
            f"[FILTER_FRAME_PERSON] {manifest_path.name}: kept={report['num_rows_out']}/{report['num_rows_in']} "
            f"dropped={report['num_dropped']}"
        )

    report_path = out_dir / "filter_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
