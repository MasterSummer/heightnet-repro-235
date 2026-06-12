#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID_CLOTHING = {"Coat", "Wb", "Long"}
UNSEEN_TEST_PERSONS = {
    "1128_man1",
    "1201_woman1",
    "1202_woman1",
    "1203_woman2",
    "1205_man1",
    "1209_man1",
}
VAL_PERSONS = {
    "1127_woman1",
    "1201_woman2",
    "1201_woman3",
}
EXCLUDED_ABSENT_PERSONS = {
    "1124_man1",
    "1124_woman1",
    "1124_woman2",
    "1124_woman3",
    "1125_man1",
    "1125_man2",
    "1126_man2",
    "1126_man3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build clothing-based train/val/test split for the MLP ranking experiment."
    )
    parser.add_argument(
        "--features",
        default="/data1/zyding/jianzhi_2511_all_camera_rect_sequence/sequence_features.json",
        help="Path to sequence_features.json",
    )
    parser.add_argument(
        "--rank-csv",
        default="/data1/zyding/jianzhi_2511_clothing_split_mlp/15_server_reference/rank.json",
        help="Path to ranking CSV file stored with a .json suffix",
    )
    parser.add_argument(
        "--out-dir",
        default="/data1/zyding/jianzhi_2511_clothing_split_mlp",
        help="Directory for clothing_split.json and split_audit.json",
    )
    return parser.parse_args()


def load_features(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "meta" not in data or "features" not in data:
        raise ValueError(f"Unexpected features JSON structure in {path}")
    return data


def load_rank_persons(path):
    persons = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = ["penson_id", "height(cm)", "weight(kg)"]
        if reader.fieldnames != expected:
            raise ValueError(
                f"Unexpected CSV header in {path}: {reader.fieldnames}; expected {expected}"
            )
        for row in reader:
            person_id = (row.get("penson_id") or "").strip()
            if person_id:
                persons.append(person_id)
    return persons


def parse_clothing_type(seq_id):
    if "__" not in seq_id:
        return None
    try:
        suffix = seq_id.split("__", 1)[1]
        clothing = suffix.split("_pants", 1)[0]
    except Exception:
        return None
    if clothing not in VALID_CLOTHING:
        return None
    return clothing


def base_camera_group(camera_id):
    parts = str(camera_id).split("_")
    if len(parts) < 2:
        return str(camera_id)
    return "_".join(parts[:2])


def finalize_bucket(bucket):
    bucket["persons"] = sorted(bucket["persons"])
    bucket["seq_ids"] = sorted(bucket["seq_ids"])
    bucket["count_persons"] = len(bucket["persons"])
    bucket["count_sequences"] = len(bucket["seq_ids"])
    return bucket


def main():
    args = parse_args()
    features_data = load_features(args.features)
    rank_persons = load_rank_persons(args.rank_csv)
    meta = features_data["meta"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_person_set = set(rank_persons)
    train_persons = rank_person_set - UNSEEN_TEST_PERSONS - VAL_PERSONS
    excluded_persons = UNSEEN_TEST_PERSONS | EXCLUDED_ABSENT_PERSONS

    split_buckets = {
        "train": {"persons": set(), "seq_ids": []},
        "val": {"persons": set(), "seq_ids": []},
        "test": {"persons": set(), "seq_ids": []},
        "excluded": {"persons": set(), "seq_ids": []},
    }
    seq_split_map = {}
    warnings = []
    per_camera = defaultdict(lambda: {split: 0 for split in ["train", "val", "test", "excluded"]})
    per_camera_by_clothing = defaultdict(
        lambda: defaultdict(lambda: {split: 0 for split in ["train", "val", "test", "excluded"]})
    )
    clothing_counts = {split: Counter() for split in ["train", "val", "test", "excluded"]}
    present_persons = set()
    long_persons = set()

    for seq_id, info in meta.items():
        person_id = info.get("person_id") or seq_id.split("__", 1)[0]
        present_persons.add(person_id)
        clothing = parse_clothing_type(seq_id)
        if clothing is None:
            warning = f"Warning: failed to parse valid clothing type from sequence_id={seq_id}; skipping"
            warnings.append(warning)
            print(warning, file=sys.stderr)
            continue
        if clothing == "Long":
            long_persons.add(person_id)

        if clothing == "Long":
            split = "test"
        elif clothing in {"Coat", "Wb"}:
            if person_id in EXCLUDED_ABSENT_PERSONS:
                split = "excluded"
            elif person_id in UNSEEN_TEST_PERSONS:
                split = "excluded"
            elif person_id in VAL_PERSONS:
                split = "val"
            elif person_id in train_persons:
                split = "train"
            else:
                warning = (
                    f"Warning: person_id={person_id} from sequence_id={seq_id} not covered by split rules; skipping"
                )
                warnings.append(warning)
                print(warning, file=sys.stderr)
                continue
        else:
            warning = f"Warning: unsupported clothing type={clothing} for sequence_id={seq_id}; skipping"
            warnings.append(warning)
            print(warning, file=sys.stderr)
            continue

        seq_split_map[seq_id] = split
        split_buckets[split]["persons"].add(person_id)
        split_buckets[split]["seq_ids"].append(seq_id)

        camera_group = base_camera_group(info.get("camera_id", ""))
        per_camera[camera_group][split] += 1
        per_camera_by_clothing[camera_group][clothing][split] += 1
        clothing_counts[split][clothing] += 1

    features_persons_not_in_rank = sorted(present_persons - rank_person_set)
    rank_persons_missing_from_features = sorted(rank_person_set - present_persons)

    output = {
        "split_method": {
            "name": "clothing_based_mlp_rank_split",
            "rules": {
                "test": "All Long sequences for every person present in features.",
                "val": "Coat and Wb sequences for 1127_woman1, 1201_woman2, 1201_woman3.",
                "train": "Remaining Coat and Wb sequences from ranked persons, excluding val persons and unseen persons.",
                "excluded": "Coat and Wb sequences from six unseen persons; eight absent persons are tracked as excluded but have no sequences in these features.",
            },
            "valid_clothing": sorted(VALID_CLOTHING),
            "unseen_test_persons": sorted(UNSEEN_TEST_PERSONS),
            "val_persons": sorted(VAL_PERSONS),
            "excluded_absent_persons": sorted(EXCLUDED_ABSENT_PERSONS),
            "train_persons": sorted(train_persons),
            "rank_person_count": len(rank_persons),
            "feature_person_count": len(present_persons),
            "rank_persons_missing_from_features": rank_persons_missing_from_features,
            "features_persons_not_in_rank": features_persons_not_in_rank,
            "present_long_persons": sorted(long_persons),
            "warnings": warnings,
        },
        "train": finalize_bucket(split_buckets["train"]),
        "val": finalize_bucket(split_buckets["val"]),
        "test": finalize_bucket(split_buckets["test"]),
        "excluded": {
            **finalize_bucket(split_buckets["excluded"]),
            "absent_persons": sorted(EXCLUDED_ABSENT_PERSONS),
        },
        "per_camera": {
            camera: {
                "counts": dict(counts),
                "by_clothing": {clothing: dict(split_counts) for clothing, split_counts in sorted(by_clothing.items())},
            }
            for camera, counts, by_clothing in sorted(
                (camera, counts, per_camera_by_clothing[camera]) for camera, counts in per_camera.items()
            )
        },
        "seq_split_map": {seq_id: seq_split_map[seq_id] for seq_id in sorted(seq_split_map)},
    }

    violations = {
        "long_in_train": [seq_id for seq_id, split in seq_split_map.items() if split == "train" and parse_clothing_type(seq_id) == "Long"],
        "long_in_val": [seq_id for seq_id, split in seq_split_map.items() if split == "val" and parse_clothing_type(seq_id) == "Long"],
        "coat_or_wb_in_test": [
            seq_id for seq_id, split in seq_split_map.items() if split == "test" and parse_clothing_type(seq_id) in {"Coat", "Wb"}
        ],
        "unseen_in_train_or_val": [
            seq_id
            for seq_id, split in seq_split_map.items()
            if split in {"train", "val"} and (meta[seq_id].get("person_id") or seq_id.split("__", 1)[0]) in UNSEEN_TEST_PERSONS
        ],
        "excluded_persons_in_any_split": [
            seq_id
            for seq_id, split in seq_split_map.items()
            if split in {"train", "val", "test"}
            and (meta[seq_id].get("person_id") or seq_id.split("__", 1)[0]) in excluded_persons
            and parse_clothing_type(seq_id) in {"Coat", "Wb"}
        ],
    }

    camera_threshold_violations = {}
    for camera, counts in sorted(per_camera.items()):
        bad = {split: count for split, count in counts.items() if split in {"train", "val", "test"} and count < 2}
        if bad:
            camera_threshold_violations[camera] = bad

    leakage_count = sum(len(v) for v in violations.values()) + sum(len(v) for v in camera_threshold_violations.values())

    audit = {
        "split_method": output["split_method"]["name"],
        "counts": {
            split: {
                "num_sequences": output[split]["count_sequences"],
                "num_persons": output[split]["count_persons"],
                "clothing": dict(sorted(clothing_counts[split].items())),
            }
            for split in ["train", "val", "test", "excluded"]
        },
        "checks": {
            "long_in_train": {
                "count": len(violations["long_in_train"]),
                "ok": len(violations["long_in_train"]) == 0,
                "examples": sorted(violations["long_in_train"])[:10],
            },
            "long_in_val": {
                "count": len(violations["long_in_val"]),
                "ok": len(violations["long_in_val"]) == 0,
                "examples": sorted(violations["long_in_val"])[:10],
            },
            "coat_wb_in_test": {
                "count": len(violations["coat_or_wb_in_test"]),
                "ok": len(violations["coat_or_wb_in_test"]) == 0,
                "examples": sorted(violations["coat_or_wb_in_test"])[:10],
            },
            "unseen_in_train_val": {
                "count": len(violations["unseen_in_train_or_val"]),
                "ok": len(violations["unseen_in_train_or_val"]) == 0,
                "examples": sorted(violations["unseen_in_train_or_val"])[:10],
            },
            "excluded_persons_in_any_split": {
                "count": len(violations["excluded_persons_in_any_split"]),
                "ok": len(violations["excluded_persons_in_any_split"]) == 0,
                "examples": sorted(violations["excluded_persons_in_any_split"])[:10],
            },
            "camera_min_sequences": {
                "ok": len(camera_threshold_violations) == 0,
                "required_min_per_camera": {"train": 2, "val": 2, "test": 2},
                "violations": camera_threshold_violations,
            },
        },
        "leakage_count": leakage_count,
        "warning_count": len(warnings),
        "warnings": warnings,
    }

    split_path = out_dir / "clothing_split.json"
    audit_path = out_dir / "split_audit.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"Wrote split file to {split_path}")
    print(f"Wrote audit file to {audit_path}")
    print(json.dumps({
        "train_sequences": output["train"]["count_sequences"],
        "val_sequences": output["val"]["count_sequences"],
        "test_sequences": output["test"]["count_sequences"],
        "excluded_sequences": output["excluded"]["count_sequences"],
        "leakage_count": audit["leakage_count"],
        "warning_count": audit["warning_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
