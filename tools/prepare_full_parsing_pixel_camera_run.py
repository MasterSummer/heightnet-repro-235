from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, "..", "src"))

from heightnet.parsing_mask import parsing_dir_from_sequence_id, resolve_extracted_parsing_path
from prepare_parsing_mask_pixel_camera_run import _materialize_frames_grouped
from prepare_smallscale_pixel_camera_run import _depth_cache_path, _safe_name


def _parsing_has_person(parsing_path: Path, min_pixels: int) -> tuple[bool, int]:
    mask = cv2.imread(str(parsing_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, 0
    pixels = int((mask > 0).sum())
    return pixels >= int(min_pixels), pixels


def _parse_parsing_filelist_multi(path: str | Path) -> dict[str, dict[int, list[str]]]:
    index: dict[str, dict[int, list[str]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            member = line.strip()
            if not member:
                continue
            parts = Path(member).parts
            if len(parts) < 3:
                continue
            parsing_dir = parts[-3]
            try:
                frame_idx = int(Path(parts[-1]).stem)
            except ValueError:
                continue
            index.setdefault(parsing_dir, {}).setdefault(frame_idx, []).append(member)
    return index


def _select_best_parsing_member(
    members: list[str],
    parsing_root: Path,
    min_parsing_pixels: int,
) -> tuple[str, Path, bool, int]:
    best_member = ""
    best_path = Path()
    best_pixels = -1
    for member in members:
        parsing_path = resolve_extracted_parsing_path(parsing_root, member)
        if not parsing_path.exists():
            continue
        _has_person, pixels = _parsing_has_person(parsing_path, min_pixels=min_parsing_pixels)
        if pixels > best_pixels:
            best_member = member
            best_path = parsing_path
            best_pixels = pixels
    if best_pixels < 0:
        return "", Path(), False, 0
    return best_member, best_path, best_pixels >= int(min_parsing_pixels), int(best_pixels)


def _classify_rows(
    frame: pd.DataFrame,
    parsing_index: dict[str, dict[int, list[str]]],
    parsing_root: Path,
    min_parsing_pixels: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept: list[dict] = []
    dropped: list[dict] = []
    for row in frame.to_dict(orient="records"):
        sequence_id = str(row["sequence_id"])
        frame_idx = int(row.get("frame_idx", row.get("frame_start", 0)))
        video_path = Path(str(row["video_path"]))
        parsing_dir = parsing_dir_from_sequence_id(sequence_id)
        member = parsing_index.get(parsing_dir, {}).get(frame_idx)

        out_row = dict(row)
        out_row["frame_idx"] = frame_idx
        if not _depth_cache_path(video_path, frame_idx).exists():
            out_row["drop_reason"] = "missing_depth"
            dropped.append(out_row)
            continue
        members = parsing_index.get(parsing_dir, {}).get(frame_idx, [])
        if not members:
            out_row["drop_reason"] = "missing_parsing"
            dropped.append(out_row)
            continue

        member, parsing_path, has_person, pixels = _select_best_parsing_member(
            members=members,
            parsing_root=parsing_root,
            min_parsing_pixels=min_parsing_pixels,
        )
        out_row["parsing_member"] = member
        out_row["parsing_path"] = str(parsing_path)
        out_row["parsing_candidates"] = len(members)
        if not member or not parsing_path.exists():
            out_row["drop_reason"] = "missing_extracted_parsing"
            dropped.append(out_row)
            continue

        out_row["parsing_person_pixels"] = pixels
        if not has_person:
            out_row["drop_reason"] = "no_person_by_parsing"
            dropped.append(out_row)
            continue
        kept.append(out_row)

    return pd.DataFrame(kept), pd.DataFrame(dropped)


def _materialize_dropped_frames(dropped: pd.DataFrame, no_person_root: Path) -> pd.DataFrame:
    if dropped.empty:
        return dropped
    frame = dropped.copy()
    # Reuse the grouped decoder by materializing to a temporary frames tree first,
    # then move paths into the explicit no-person tree.
    materialized, _summary = _materialize_frames_grouped(frame, frames_root=no_person_root)
    rows = []
    for row in materialized.to_dict(orient="records"):
        src = Path(str(row["frame_path"]))
        person_id = _safe_name(str(row["person_id"]))
        sequence_id = _safe_name(str(row["sequence_id"]))
        frame_idx = int(row.get("frame_idx", row.get("frame_start", 0)))
        dst = no_person_root / _safe_name(str(row.get("drop_reason", "dropped"))) / person_id / f"{sequence_id}__frame{frame_idx:06d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and src.resolve() != dst.resolve():
            if dst.exists():
                src.unlink(missing_ok=True)
            else:
                shutil.move(str(src), str(dst))
        row["no_person_frame_path"] = str(dst.resolve())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_split(
    src_path: Path,
    out_path: Path,
    parsing_index: dict[str, dict[int, list[str]]],
    parsing_root: Path,
    no_person_root: Path,
    min_parsing_pixels: int,
    materialize_no_person: bool,
) -> tuple[dict, set[str], pd.DataFrame]:
    frame = pd.read_csv(src_path)
    kept, dropped = _classify_rows(
        frame=frame,
        parsing_index=parsing_index,
        parsing_root=parsing_root,
        min_parsing_pixels=min_parsing_pixels,
    )
    if kept.empty:
        raise RuntimeError(f"no person-present rows for split: {src_path}")

    kept, materialize_summary = _materialize_frames_grouped(kept, frames_root=out_path.parent / "frames")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    if not dropped.empty:
        dropped.insert(0, "split", src_path.stem.replace("_manifest", ""))
        if materialize_no_person:
            dropped = _materialize_dropped_frames(dropped, no_person_root=no_person_root)

    reason_counts = dropped["drop_reason"].value_counts().sort_index().to_dict() if not dropped.empty else {}
    summary = {
        "input_rows": int(len(frame)),
        "kept_rows": int(len(kept)),
        "dropped_rows": int(len(dropped)),
        "num_people": int(kept["person_id"].nunique()),
        "num_sequences": int(kept["sequence_id"].nunique()),
        "drop_reasons": {str(k): int(v) for k, v in reason_counts.items()},
    }
    summary.update(materialize_summary)
    return summary, set(kept["parsing_member"].astype(str).tolist()), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full per-camera manifests, filtering no-person frames by parsing.")
    parser.add_argument("--camera-dir", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--parsing-filelist", required=True)
    parser.add_argument("--parsing-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validate-every-epochs", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--min-parsing-pixels", type=int, default=8)
    parser.add_argument("--materialize-no-person", action="store_true")
    args = parser.parse_args()

    camera_dir = Path(args.camera_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    parsing_root = Path(args.parsing_root).resolve()
    no_person_root = out_dir / "no_person_frames"
    parsing_index = _parse_parsing_filelist_multi(args.parsing_filelist)

    split_summaries = {}
    parsing_members: set[str] = set()
    dropped_all = []
    for split in ["train", "val", "test"]:
        summary, members, dropped = _write_split(
            src_path=camera_dir / f"{split}_manifest.csv",
            out_path=out_dir / f"{split}_manifest.csv",
            parsing_index=parsing_index,
            parsing_root=parsing_root,
            no_person_root=no_person_root,
            min_parsing_pixels=int(args.min_parsing_pixels),
            materialize_no_person=bool(args.materialize_no_person),
        )
        split_summaries[split] = summary
        parsing_members.update(members)
        if not dropped.empty:
            dropped_all.append(dropped)

    if dropped_all:
        dropped_frame = pd.concat(dropped_all, ignore_index=True)
    else:
        dropped_frame = pd.DataFrame()
    dropped_frame.to_csv(out_dir / "no_person_manifest.csv", index=False)
    (out_dir / "parsing_members.txt").write_text("\n".join(sorted(parsing_members)) + "\n", encoding="utf-8")

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

    config_out = Path(args.config_out).resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)
    with config_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    train_rows = split_summaries["train"]["kept_rows"]
    batch = int(cfg["train"]["batch_size"])
    summary = {
        "camera_id": args.camera_id,
        "camera_dir": str(camera_dir),
        "out_dir": str(out_dir),
        "config_out": str(config_out),
        "output_dir": str(Path(args.output_dir).resolve()),
        "parsing_root": str(parsing_root),
        "parsing_members": int(len(parsing_members)),
        "no_person_manifest": str((out_dir / "no_person_manifest.csv").resolve()),
        "no_person_frames": str(no_person_root.resolve()),
        "splits": split_summaries,
        "estimated_steps_per_epoch": int((train_rows + max(batch - 1, 0)) // batch),
    }
    (out_dir / "full_prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
