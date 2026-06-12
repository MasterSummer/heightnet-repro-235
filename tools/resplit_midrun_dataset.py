from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SPLITS = ("train", "val", "test")
MODALITIES = ("rgb", "depth", "height", "valid_mask", "person_mask")


@dataclass
class SequenceInfo:
    sequence_id: str
    person_id: str
    camera_id: str
    num_frames: int
    sources: Dict[str, Path]  # modality -> source sequence dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resplit heightnet_data_midrun into two datasets: "
            "(1) grouped by person_id, (2) grouped by person_id+camera_id."
        )
    )
    parser.add_argument("--data-root", type=str, required=True, help="Existing midrun dataroot")
    parser.add_argument("--out-person-root", type=str, required=True, help="Output root for person-level split")
    parser.add_argument(
        "--out-person-camera-root",
        type=str,
        required=True,
        help="Output root for person+camera-level split",
    )
    parser.add_argument(
        "--repro-data-root",
        type=str,
        required=True,
        help="heightnet_repro/data root to rewrite with split metadata/manifests",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="Use symlink (fast, space-saving) or copy.",
    )
    return parser.parse_args()


def infer_person_camera(sequence_id: str) -> Tuple[str, str]:
    person_pat = re.compile(r"\d{4}_(?:man|woman)\d+", re.IGNORECASE)
    camera_pat = re.compile(r"\d+cm_(?:inside|outside|slantside)", re.IGNORECASE)
    m_person = person_pat.search(sequence_id)
    m_camera = camera_pat.search(sequence_id)
    person_id = m_person.group(0) if m_person else "unknown_person"
    camera_id = m_camera.group(0).lower() if m_camera else "unknown_camera"
    return person_id, camera_id


def collect_sequences(data_root: Path) -> List[SequenceInfo]:
    rgb_map: Dict[str, Path] = {}
    all_sources: Dict[str, Dict[str, Path]] = {m: {} for m in MODALITIES}

    for split in SPLITS:
        for mod in MODALITIES:
            base = data_root / split / mod
            if not base.exists():
                continue
            for seq_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
                all_sources[mod][seq_dir.name] = seq_dir
                if mod == "rgb":
                    rgb_map[seq_dir.name] = seq_dir

    infos: List[SequenceInfo] = []
    for seq_id, rgb_dir in sorted(rgb_map.items()):
        person_id, camera_id = infer_person_camera(seq_id)
        num_frames = len(list(rgb_dir.glob("*.png")))
        sources = {}
        for mod in MODALITIES:
            if seq_id in all_sources[mod]:
                sources[mod] = all_sources[mod][seq_id]
        infos.append(
            SequenceInfo(
                sequence_id=seq_id,
                person_id=person_id,
                camera_id=camera_id,
                num_frames=num_frames,
                sources=sources,
            )
        )
    return infos


def assign_groups(
    groups: Dict[str, List[SequenceInfo]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[SequenceInfo]]:
    total = sum(len(v) for v in groups.values())
    target = {
        "train": int(round(total * train_ratio)),
        "val": int(round(total * val_ratio)),
    }
    target["test"] = max(0, total - target["train"] - target["val"])

    cur = {"train": 0, "val": 0, "test": 0}
    out = {"train": [], "val": [], "test": []}

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    for k in keys:
        items = groups[k]
        n = len(items)
        # Choose split with largest remaining capacity.
        remain = {s: target[s] - cur[s] for s in SPLITS}
        best = max(SPLITS, key=lambda s: (remain[s], -cur[s]))
        if remain[best] < n:
            # fallback: split with fewest assigned samples.
            best = min(SPLITS, key=lambda s: cur[s])
        out[best].extend(items)
        cur[best] += n

    for s in SPLITS:
        out[s] = sorted(out[s], key=lambda x: x.sequence_id)
    return out


def materialize_split(
    split_map: Dict[str, List[SequenceInfo]],
    out_root: Path,
    link_mode: str,
) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for info in split_map[split]:
            for mod, src_dir in info.sources.items():
                dst_dir = out_root / split / mod / info.sequence_id
                dst_dir.parent.mkdir(parents=True, exist_ok=True)
                if link_mode == "symlink":
                    dst_dir.symlink_to(src_dir)
                else:
                    shutil.copytree(src_dir, dst_dir)


def write_scheme_outputs(
    scheme_name: str,
    split_map: Dict[str, List[SequenceInfo]],
    out_root: Path,
    repro_data_root: Path,
) -> None:
    total = sum(len(v) for v in split_map.values())
    summary = {
        "scheme": scheme_name,
        "out_root": str(out_root),
        "counts": {s: len(split_map[s]) for s in SPLITS},
        "total_sequences": total,
    }
    with (out_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    scheme_dir = repro_data_root / scheme_name
    if scheme_dir.exists():
        shutil.rmtree(scheme_dir)
    scheme_dir.mkdir(parents=True, exist_ok=True)

    # Write split sequence lists.
    for split in SPLITS:
        with (scheme_dir / f"{split}_sequences.txt").open("w", encoding="utf-8") as f:
            for info in split_map[split]:
                f.write(info.sequence_id + "\n")

    # Write RGB-only manifests so downstream steps can inspect split composition now.
    for split in SPLITS:
        rows = []
        for info in split_map[split]:
            rgb_dir = out_root / split / "rgb" / info.sequence_id
            rows.append(
                {
                    "sequence_id": info.sequence_id,
                    "person_id": info.person_id,
                    "camera_id": info.camera_id,
                    "num_frames": info.num_frames,
                    "rgb_dir": str(rgb_dir.resolve()),
                }
            )
        csv_path = scheme_dir / f"{split}_manifest_rgb.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sequence_id", "person_id", "camera_id", "num_frames", "rgb_dir"],
            )
            writer.writeheader()
            writer.writerows(rows)

    with (scheme_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_person_root = Path(args.out_person_root).expanduser().resolve()
    out_pc_root = Path(args.out_person_camera_root).expanduser().resolve()
    repro_data_root = Path(args.repro_data_root).expanduser().resolve()
    repro_data_root.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0,1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1")

    infos = collect_sequences(data_root)
    if not infos:
        raise RuntimeError(f"No sequences found under {data_root}/{{train,val,test}}/rgb")

    # Scheme A: group by person_id.
    groups_person: Dict[str, List[SequenceInfo]] = {}
    for x in infos:
        groups_person.setdefault(x.person_id, []).append(x)
    split_person = assign_groups(groups_person, args.train_ratio, args.val_ratio, args.seed)
    materialize_split(split_person, out_person_root, args.link_mode)
    write_scheme_outputs("person_split", split_person, out_person_root, repro_data_root)

    # Scheme B: group by person_id + camera_id.
    groups_pc: Dict[str, List[SequenceInfo]] = {}
    for x in infos:
        groups_pc.setdefault(f"{x.person_id}__{x.camera_id}", []).append(x)
    split_pc = assign_groups(groups_pc, args.train_ratio, args.val_ratio, args.seed)
    materialize_split(split_pc, out_pc_root, args.link_mode)
    write_scheme_outputs("person_camera_split", split_pc, out_pc_root, repro_data_root)

    print(
        json.dumps(
            {
                "person_split": {s: len(split_person[s]) for s in SPLITS},
                "person_camera_split": {s: len(split_pc[s]) for s in SPLITS},
                "repro_data_root": str(repro_data_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
