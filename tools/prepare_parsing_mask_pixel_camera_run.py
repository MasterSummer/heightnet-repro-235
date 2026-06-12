from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
import cv2

_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, "..", "src"))

from heightnet.parsing_mask import parse_parsing_filelist_path, parsing_dir_from_sequence_id
from prepare_smallscale_pixel_camera_run import (
    _candidate_frame_indices,
    _depth_cache_path,
    _link_depth_cache,
    _safe_name,
)
from subsample_manifest_per_person import subsample_rows


def _rows_with_parsing_and_depth(frame: pd.DataFrame, parsing_index: dict[str, dict[int, str]]) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict(orient="records"):
        sequence_id = str(row["sequence_id"])
        frame_idx = int(row.get("frame_idx", row.get("frame_start", 0)))
        parsing_dir = parsing_dir_from_sequence_id(sequence_id)
        member = parsing_index.get(parsing_dir, {}).get(frame_idx)
        if not member:
            continue
        video_path = Path(str(row["video_path"]))
        if not _depth_cache_path(video_path, frame_idx).exists():
            continue
        row["parsing_member"] = member
        rows.append(row)
    return pd.DataFrame(rows)


def _decode_frame_from_open_capture(
    cap: cv2.VideoCapture,
    video_path: Path,
    frame_idx: int,
    frame_path: Path,
) -> int:
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
    raise RuntimeError(f"failed to decode frame={frame_idx} from {video_path}")


def _materialize_frames_grouped(frame: pd.DataFrame, frames_root: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    extracted = 0
    reused = 0
    linked_depth = 0
    missing_depth = 0

    planned = []
    for position, row in enumerate(frame.to_dict(orient="records")):
        video_path = Path(str(row["video_path"]))
        person_id = str(row["person_id"])
        sequence_id = str(row["sequence_id"])
        frame_idx = int(row.get("frame_idx", row.get("frame_start", 0)))
        filename = f"{_safe_name(sequence_id)}__frame{frame_idx:06d}.jpg"
        frame_path = frames_root / _safe_name(person_id) / filename
        planned.append((position, row, video_path, frame_idx, frame_path))

    by_video: dict[str, list[tuple[int, dict, Path, int, Path]]] = {}
    for item in planned:
        by_video.setdefault(str(item[2]), []).append(item)

    materialized_by_position: dict[int, dict] = {}
    for video_path_text, items in by_video.items():
        cap = None
        try:
            missing_items = []
            for position, row, video_path, frame_idx, frame_path in items:
                if frame_path.exists():
                    actual_idx = int(row.get("decoded_frame_idx", frame_idx))
                    reused += 1
                    row["frame_path"] = str(frame_path.resolve())
                    row["decoded_frame_idx"] = int(actual_idx)
                    if _link_depth_cache(video_path, int(actual_idx), frame_path):
                        linked_depth += 1
                    else:
                        missing_depth += 1
                    materialized_by_position[position] = row
                else:
                    missing_items.append((position, row, video_path, frame_idx, frame_path))

            if missing_items:
                cap = cv2.VideoCapture(video_path_text)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open video: {video_path_text}")
                by_frame_idx = {int(item[3]): item for item in missing_items}
                min_idx = min(by_frame_idx)
                max_idx = max(by_frame_idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(min_idx))
                current = min_idx
                while current <= max_idx and by_frame_idx:
                    ok, image = cap.read()
                    if not ok or image is None:
                        break
                    item = by_frame_idx.pop(current, None)
                    if item is not None:
                        position, row, video_path, frame_idx, frame_path = item
                        frame_path.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(frame_path), image):
                            raise RuntimeError(f"failed to write frame image: {frame_path}")
                        extracted += 1
                        row["frame_path"] = str(frame_path.resolve())
                        row["decoded_frame_idx"] = int(frame_idx)
                        if _link_depth_cache(video_path, int(frame_idx), frame_path):
                            linked_depth += 1
                        else:
                            missing_depth += 1
                        materialized_by_position[position] = row
                    current += 1

                for position, row, video_path, frame_idx, frame_path in by_frame_idx.values():
                    actual_idx = _decode_frame_from_open_capture(cap, video_path, frame_idx, frame_path)
                    extracted += 1
                    row["frame_path"] = str(frame_path.resolve())
                    row["decoded_frame_idx"] = int(actual_idx)
                    if _link_depth_cache(video_path, int(actual_idx), frame_path):
                        linked_depth += 1
                    else:
                        missing_depth += 1
                    materialized_by_position[position] = row
        finally:
            if cap is not None:
                cap.release()

    for position in range(len(planned)):
        rows.append(materialized_by_position[position])
    return pd.DataFrame(rows), {
        "extracted_frames": int(extracted),
        "reused_frames": int(reused),
        "linked_depth": int(linked_depth),
        "missing_depth": int(missing_depth),
    }


def _write_manifest(
    src_path: Path,
    out_path: Path,
    parsing_index: dict[str, dict[int, str]],
    target_rows: int,
    seed: int,
    per_person_cap: int,
    materialize_frames: bool,
) -> tuple[dict, list[str]]:
    frame = pd.read_csv(src_path)
    eligible = _rows_with_parsing_and_depth(frame, parsing_index)
    if eligible.empty:
        if src_path.name == "test_manifest.csv":
            empty = frame.head(0).copy()
            empty["parsing_member"] = []
            out_path.parent.mkdir(parents=True, exist_ok=True)
            empty.to_csv(out_path, index=False)
            return {
                "input_rows": int(len(frame)),
                "eligible_rows": 0,
                "output_rows": 0,
                "num_people": 0,
                "note": "empty test split because no rows have both parsing and depth cache",
            }, []
        raise RuntimeError(f"no rows with both parsing and depth cache: {src_path}")
    sampled = subsample_rows(
        frame=eligible,
        target_rows=min(int(target_rows), int(len(eligible))),
        seed=seed,
        per_person_cap=per_person_cap,
    )
    materialize_summary = {}
    if materialize_frames:
        sampled, materialize_summary = _materialize_frames_grouped(sampled, frames_root=out_path.parent / "frames")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)
    summary = {
        "input_rows": int(len(frame)),
        "eligible_rows": int(len(eligible)),
        "output_rows": int(len(sampled)),
        "num_people": int(sampled["person_id"].nunique()),
    }
    summary.update(materialize_summary)
    return summary, sorted(set(sampled["parsing_member"].astype(str).tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a per-camera small run whose sampled frames have parsing masks.")
    parser.add_argument("--camera-dir", type=str, required=True)
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--parsing-filelist", type=str, required=True)
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
    parsing_index = parse_parsing_filelist_path(args.parsing_filelist)

    split_targets = {
        "train": int(args.train_rows),
        "val": int(args.val_rows),
        "test": int(args.test_rows),
    }
    split_summaries = {}
    parsing_members: set[str] = set()
    for split, target_rows in split_targets.items():
        split_summary, split_members = _write_manifest(
            src_path=camera_dir / f"{split}_manifest.csv",
            out_path=out_dir / f"{split}_manifest.csv",
            parsing_index=parsing_index,
            target_rows=target_rows,
            seed=int(args.seed),
            per_person_cap=int(args.per_person_cap),
            materialize_frames=bool(args.materialize_frames),
        )
        split_summaries[split] = split_summary
        parsing_members.update(split_members)

    for name in ["pairs.json", "rank.json", "summary.json"]:
        src = camera_dir / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())

    members_path = out_dir / "parsing_members.txt"
    members_path.write_text("\n".join(sorted(parsing_members)) + "\n", encoding="utf-8")

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
        "parsing_members": int(len(parsing_members)),
        "parsing_members_file": str(members_path.resolve()),
        "splits": split_summaries,
        "estimated_steps_per_epoch": int(
            (split_summaries["train"]["output_rows"] + max(int(cfg["train"]["batch_size"]) - 1, 0))
            // int(cfg["train"]["batch_size"])
        ),
    }
    (out_dir / "parsing_prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
