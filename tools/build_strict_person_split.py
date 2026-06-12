#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cross_camera_heightmap_fusion_core import load_height_labels


def build_person_split(people: list[str], seed: int = 42) -> dict[str, list[str]]:
    ordered = sorted(set(str(person) for person in people))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if len(ordered) < 3:
        raise ValueError("strict person split requires at least three people")
    if len(ordered) == 31:
        train_count, val_count = 21, 5
    else:
        train_count = max(1, int(round(len(ordered) * 0.68)))
        val_count = max(1, int(round(len(ordered) * 0.16)))
        if train_count + val_count >= len(ordered):
            val_count = 1
            train_count = len(ordered) - 2
    return {
        "train": sorted(ordered[:train_count]),
        "val": sorted(ordered[train_count:train_count + val_count]),
        "test": sorted(ordered[train_count + val_count:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-path", type=Path, default=Path("/home/zyding/data/label/rank.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = load_height_labels(args.label_path)
    splits = build_person_split(list(labels), seed=args.seed)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "label_path": str(args.label_path),
        **splits,
        "counts": {key: len(value) for key, value in splits.items()},
        "overlap_counts": {
            "train_val": len(set(splits["train"]) & set(splits["val"])),
            "train_test": len(set(splits["train"]) & set(splits["test"])),
            "val_test": len(set(splits["val"]) & set(splits["test"])),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("[OUT]", args.out)


if __name__ == "__main__":
    main()
