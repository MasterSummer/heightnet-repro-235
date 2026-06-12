from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_frame_rows(frame: pd.DataFrame, sample_fps: float) -> list[dict]:
    rows: list[dict] = []
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    for row in frame.to_dict(orient="records"):
        frame_start = int(row["frame_start"])
        frame_end = int(row["frame_end"])
        fps = float(row["fps"])
        if frame_end < frame_start:
            continue

        if fps <= 0:
            frame_indices = [frame_start]
        else:
            stride = max(int(round(fps / sample_fps)), 1)
            frame_indices = list(range(frame_start, frame_end + 1, stride))
            if not frame_indices:
                frame_indices = [frame_start]

        for frame_idx in frame_indices:
            item = dict(row)
            item["frame_path"] = ""
            item["source_frame_start"] = frame_start
            item["source_frame_end"] = frame_end
            item["frame_idx"] = int(frame_idx)
            item["frame_start"] = int(frame_idx)
            item["frame_end"] = int(frame_idx)
            rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a video-level manifest into a sampled frame-level manifest.")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for manifest in args.manifest:
        manifest_path = Path(manifest).resolve()
        frame = pd.read_csv(manifest_path)
        rows = build_frame_rows(frame, sample_fps=float(args.sample_fps))
        out_path = out_dir / manifest_path.name
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(
            f"[FRAME_MANIFEST] {manifest_path.name}: videos={len(frame)} frames={len(rows)} sample_fps={args.sample_fps}"
        )


if __name__ == "__main__":
    main()
