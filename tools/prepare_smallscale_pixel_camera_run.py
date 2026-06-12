from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from subsample_manifest_per_person import subsample_rows


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _candidate_frame_indices(frame_idx: int, max_frame_idx: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for delta in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
        idx = frame_idx + delta
        if 0 <= idx <= max_frame_idx and idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _depth_cache_path(video_path: Path, frame_idx: int) -> Path:
    return video_path.parent / ".depth_cache" / f"{video_path.stem}.frame_{frame_idx:06d}.depth.npy"


def _link_depth_cache(video_path: Path, frame_idx: int, frame_path: Path) -> bool:
    depth_src = _depth_cache_path(video_path, frame_idx)
    if not depth_src.exists():
        return False
    depth_dst = Path(str(frame_path) + ".depth.npy")
    if depth_dst.exists() or depth_dst.is_symlink():
        return True
    os.symlink(str(depth_src), str(depth_dst))
    return True


def _extract_frame(video_path: Path, frame_idx: int, frame_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame_idx = max(n_frames - 1, 0)
        for candidate in _candidate_frame_indices(frame_idx, max_frame_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(candidate))
            ok, frame = cap.read()
            if ok and frame is not None:
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"failed to write frame image: {frame_path}")
                return int(candidate)
    finally:
        cap.release()
    raise RuntimeError(f"failed to decode frame={frame_idx} from {video_path}")


def _materialize_frames(frame: pd.DataFrame, frames_root: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    extracted = 0
    reused = 0
    linked_depth = 0
    missing_depth = 0
    for row in frame.to_dict(orient="records"):
        video_path = Path(str(row["video_path"]))
        person_id = str(row["person_id"])
        sequence_id = str(row["sequence_id"])
        frame_idx = int(row.get("frame_idx", row.get("frame_start", 0)))
        filename = f"{_safe_name(sequence_id)}__frame{frame_idx:06d}.jpg"
        frame_path = frames_root / _safe_name(person_id) / filename

        if frame_path.exists():
            actual_idx = int(row.get("decoded_frame_idx", frame_idx))
            reused += 1
        else:
            actual_idx = _extract_frame(video_path, frame_idx, frame_path)
            extracted += 1

        row["frame_path"] = str(frame_path.resolve())
        row["decoded_frame_idx"] = int(actual_idx)
        if _link_depth_cache(video_path, int(actual_idx), frame_path):
            linked_depth += 1
        else:
            missing_depth += 1
        rows.append(row)

    return pd.DataFrame(rows), {
        "extracted_frames": int(extracted),
        "reused_frames": int(reused),
        "linked_depth": int(linked_depth),
        "missing_depth": int(missing_depth),
    }


def _write_manifest(
    src_path: Path,
    out_path: Path,
    target_rows: int,
    seed: int,
    per_person_cap: int,
    materialize_frames: bool,
) -> dict:
    frame = pd.read_csv(src_path)
    sampled = subsample_rows(
        frame=frame,
        target_rows=min(int(target_rows), int(len(frame))),
        seed=seed,
        per_person_cap=per_person_cap,
    )
    materialize_summary = {}
    if materialize_frames:
        sampled, materialize_summary = _materialize_frames(sampled, frames_root=out_path.parent / "frames")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)
    summary = {
        "input_rows": int(len(frame)),
        "output_rows": int(len(sampled)),
        "num_people": int(sampled["person_id"].nunique()),
    }
    summary.update(materialize_summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a smaller per-camera run with balanced frame subsampling.")
    parser.add_argument("--camera-dir", type=str, required=True)
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config-out", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--camera-id", type=str, required=True)
    parser.add_argument("--train-rows", type=int, default=4800)
    parser.add_argument("--val-rows", type=int, default=960)
    parser.add_argument("--test-rows", type=int, default=1920)
    parser.add_argument("--per-person-cap", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validate-every-epochs", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--materialize-frames", action="store_true")
    args = parser.parse_args()

    camera_dir = Path(args.camera_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    config_out = Path(args.config_out).resolve()

    split_targets = {
        "train": int(args.train_rows),
        "val": int(args.val_rows),
        "test": int(args.test_rows),
    }
    split_summaries = {}
    for split, target_rows in split_targets.items():
        split_summaries[split] = _write_manifest(
            src_path=camera_dir / f"{split}_manifest.csv",
            out_path=out_dir / f"{split}_manifest.csv",
            target_rows=target_rows,
            seed=int(args.seed),
            per_person_cap=int(args.per_person_cap),
            materialize_frames=bool(args.materialize_frames),
        )

    for name in ["pairs.json", "rank.json", "summary.json"]:
        src = camera_dir / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())

    with Path(args.base_config).resolve().open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["train_manifest"] = str((out_dir / "train_manifest.csv").resolve())
    cfg["paths"]["val_manifest"] = str((out_dir / "val_manifest.csv").resolve())
    cfg["paths"]["test_manifest"] = str((out_dir / "test_manifest.csv").resolve())
    cfg["paths"]["output_dir"] = str(Path(args.output_dir).resolve())
    cfg["loss"]["pairwise_json"] = str((out_dir / "pairs.json").resolve())
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["eval"]["validate_every_epochs"] = int(args.validate_every_epochs)
    cfg["eval"]["eval_batch_size"] = int(args.eval_batch_size)

    config_out.parent.mkdir(parents=True, exist_ok=True)
    with config_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    summary = {
        "camera_id": args.camera_id,
        "camera_dir": str(camera_dir),
        "out_dir": str(out_dir),
        "config_out": str(config_out),
        "output_dir": str(Path(args.output_dir).resolve()),
        "splits": split_summaries,
        "estimated_steps_per_epoch": int((split_summaries["train"]["output_rows"] + max(int(cfg["train"]["batch_size"]) - 1, 0)) // int(cfg["train"]["batch_size"])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
