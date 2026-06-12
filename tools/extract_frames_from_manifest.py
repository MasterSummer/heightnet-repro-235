from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def _candidate_frame_indices(frame_idx: int, max_frame_idx: int) -> list[int]:
    candidates: list[int] = []
    seen = set()
    deltas = [0]
    for offset in range(1, 6):
        deltas.extend([offset, -offset])
    for delta in deltas:
        idx = frame_idx + delta
        if 0 <= idx <= max_frame_idx and idx not in seen:
            candidates.append(idx)
            seen.add(idx)
    return candidates


def extract_frame(video_path: str, frame_idx: int, out_path: Path) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame_idx = max(n_frames - 1, 0)
        frame = None
        actual_idx = None
        for candidate in _candidate_frame_indices(frame_idx, max_frame_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(candidate))
            ok, maybe = cap.read()
            if ok and maybe is not None:
                frame = maybe
                actual_idx = candidate
                break
        if frame is None or actual_idx is None:
            raise RuntimeError(f"failed to decode frame={frame_idx} from {video_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"failed to write frame image: {out_path}")
        return int(actual_idx)
    finally:
        cap.release()


def process_manifest(manifest_path: Path, out_manifest: Path, frames_root: Path) -> dict:
    frame = pd.read_csv(manifest_path)
    rows = []
    for row in frame.to_dict(orient="records"):
        person_id = str(row["person_id"])
        sequence_id = str(row["sequence_id"])
        frame_idx = int(row["frame_idx"])
        filename = f"{sequence_id}__frame{frame_idx:06d}.jpg"
        frame_path = frames_root / person_id / filename
        actual_idx = extract_frame(str(row["video_path"]), frame_idx, frame_path)
        row["frame_path"] = str(frame_path.resolve())
        row["decoded_frame_idx"] = int(actual_idx)
        rows.append(row)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    return {
        "input_manifest": str(manifest_path.resolve()),
        "output_manifest": str(out_manifest.resolve()),
        "num_frames": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RGB frame images from a frame-level manifest and write frame_path back into a new manifest.")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    frames_root = out_dir / "frames"
    for manifest in args.manifest:
        manifest_path = Path(manifest).resolve()
        out_manifest = out_dir / manifest_path.name
        summary = process_manifest(manifest_path, out_manifest, frames_root)
        print(f"[EXTRACT] {manifest_path.name}: frames={summary['num_frames']}")


if __name__ == "__main__":
    main()
