from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


def _require_dir(path: str, name: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_pair_labels(rank_dir: str) -> dict[str, list[dict]]:
    if not rank_dir:
        return {}

    pair_map: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(rank_dir, "*.json"))):
        name = os.path.basename(path)
        if name == "all_pairs.json":
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list) or not rows:
            continue
        camera = rows[0].get("camera", "")
        if not camera:
            camera = name.replace("camera_", "").replace("_pairs.json", "")
        pair_map[camera.lower()] = rows
    return pair_map


def _pairwise_acc_from_labels(
    pred_scores: dict[str, float],
    pair_labels: list[dict],
    allowed_left_ids: set[str] | None = None,
    allowed_right_ids: set[str] | None = None,
) -> tuple[float | None, int]:
    correct = 0
    total = 0
    for row in pair_labels:
        id_i = row.get("id_i")
        id_j = row.get("id_j")
        y = row.get("y")
        if id_i not in pred_scores or id_j not in pred_scores:
            continue
        if allowed_left_ids is not None and id_i not in allowed_left_ids and id_j not in allowed_left_ids:
            continue
        if allowed_right_ids is not None:
            forward_ok = id_i in allowed_left_ids and id_j in allowed_right_ids if allowed_left_ids is not None else False
            backward_ok = id_j in allowed_left_ids and id_i in allowed_right_ids if allowed_left_ids is not None else False
            if not (forward_ok or backward_ok):
                continue

        pred_y = 1 if pred_scores[id_i] > pred_scores[id_j] else 0
        if int(pred_y) == int(y):
            correct += 1
        total += 1
    return (float(correct / total), total) if total > 0 else (None, 0)


def _infer_person_camera(sequence_id: str, rel_height_path: str) -> tuple[str, str]:
    person_pat = re.compile(r"\d{4}_(?:man|woman)\d+", re.IGNORECASE)
    camera_pat = re.compile(r"\d+cm_(?:inside|outside|slantside)", re.IGNORECASE)

    full_text = f"{sequence_id} {rel_height_path}"
    person_match = person_pat.search(full_text)
    camera_match = camera_pat.search(full_text)
    person_id = person_match.group(0) if person_match else "unknown_person"
    camera_id = camera_match.group(0).lower() if camera_match else "unknown_camera"
    return person_id, camera_id


def _load_mask(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    if path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(path)
        arr = arr.astype(np.float32) / 255.0
    arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr > 0.5


def _collect_split_rows(data_root: str, split: str) -> list[dict]:
    split_root = os.path.join(data_root, split)
    _require_dir(split_root, f"{split} split")
    height_files = sorted(glob.glob(os.path.join(split_root, "height", "*", "*.npy")))

    rows = []
    for height_path in height_files:
        rel = os.path.relpath(height_path, os.path.join(split_root, "height"))
        sequence_id = rel.split(os.sep)[0]
        stem = os.path.splitext(os.path.basename(height_path))[0]
        person_id, camera_id = _infer_person_camera(sequence_id, rel)
        valid_mask_path = os.path.join(split_root, "valid_mask", sequence_id, f"{stem}.npy")
        person_mask_npy = os.path.join(split_root, "person_mask", sequence_id, f"{stem}.npy")
        person_mask_png = os.path.join(split_root, "person_mask", sequence_id, f"{stem}.png")
        person_mask_path = ""
        if os.path.exists(person_mask_npy):
            person_mask_path = person_mask_npy
        elif os.path.exists(person_mask_png):
            person_mask_path = person_mask_png

        if not os.path.exists(valid_mask_path):
            continue

        rows.append(
            {
                "split": split,
                "sequence_id": sequence_id,
                "frame_idx": int(stem),
                "person_id": person_id,
                "camera_id": camera_id,
                "height_path": os.path.abspath(height_path),
                "valid_mask_path": os.path.abspath(valid_mask_path),
                "person_mask_path": os.path.abspath(person_mask_path) if person_mask_path else "",
            }
        )
    return rows


def _frame_score(height_path: str, valid_mask_path: str, person_mask_path: str, score_stat: str) -> float | None:
    height = np.load(height_path).astype(np.float32)
    mask = _load_mask(valid_mask_path, height.shape)
    mask_source = "valid_mask"
    if person_mask_path and os.path.exists(person_mask_path):
        person_mask = _load_mask(person_mask_path, height.shape)
        joint_mask = mask & person_mask
        if joint_mask.any():
            mask = joint_mask
            mask_source = "person_mask"

    values = height[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    if score_stat == "max":
        score = float(np.max(values))
    elif score_stat == "p99":
        score = float(np.percentile(values, 99))
    else:
        raise ValueError(f"Unsupported score_stat: {score_stat}")

    return score, mask_source


def _aggregate_person_scores(rows: list[dict], score_stat: str) -> tuple[dict, list[dict]]:
    by_camera_person = defaultdict(list)
    debug_rows = []

    for row in rows:
        out = _frame_score(
            row["height_path"],
            row["valid_mask_path"],
            row["person_mask_path"],
            score_stat,
        )
        if out is None:
            continue
        score, mask_source = out
        by_camera_person[(row["camera_id"], row["person_id"])].append(score)
        debug_rows.append(
            {
                "split": row["split"],
                "camera_id": row["camera_id"],
                "person_id": row["person_id"],
                "sequence_id": row["sequence_id"],
                "frame_idx": row["frame_idx"],
                "score": score,
                "mask_source": mask_source,
            }
        )

    aggregated = defaultdict(dict)
    for (camera_id, person_id), scores in by_camera_person.items():
        aggregated[camera_id][person_id] = float(sum(scores) / len(scores))
    return aggregated, debug_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare test persons against all train persons directly from height maps in data_root."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--score-stat", type=str, default="p99", choices=["p99", "max"])
    parser.add_argument("--rank-dir", type=str, default="")
    args = parser.parse_args()

    _require_dir(args.data_root, "data_root")
    if args.rank_dir:
        _require_dir(args.rank_dir, "rank_dir")
    train_rows = _collect_split_rows(args.data_root, "train")
    test_rows = _collect_split_rows(args.data_root, "test")
    pair_label_map = _load_pair_labels(args.rank_dir)

    train_scores, train_debug = _aggregate_person_scores(train_rows, args.score_stat)
    test_scores, test_debug = _aggregate_person_scores(test_rows, args.score_stat)

    result = {
        "config": {
            "data_root": os.path.abspath(args.data_root),
            "score_stat": args.score_stat,
        },
        "summary": {},
        "cameras": {},
    }

    cameras = sorted(set(train_scores.keys()) | set(test_scores.keys()))
    total_pairs = 0
    total_test_person = 0
    covered_test_person = 0
    summary_rows = []

    for camera in cameras:
        tr = train_scores.get(camera, {})
        te = test_scores.get(camera, {})
        total_test_person += len(te)
        if tr:
            covered_test_person += len(te)

        pairwise_rows = []
        merged_ranking = []
        for test_person, score_test in sorted(te.items()):
            wins = 0
            losses = 0
            for train_person, score_train in sorted(tr.items()):
                pred_test_taller = 1 if score_test > score_train else 0
                pairwise_rows.append(
                    {
                        "test_person": test_person,
                        "train_person": train_person,
                        "score_test": float(score_test),
                        "score_train": float(score_train),
                        "pred_test_taller": pred_test_taller,
                    }
                )
                if pred_test_taller == 1:
                    wins += 1
                else:
                    losses += 1
                total_pairs += 1
            summary_rows.append(
                {
                    "camera": camera,
                    "test_person": test_person,
                    "score": float(score_test),
                    "wins_vs_train": wins,
                    "losses_vs_train": losses,
                    "win_rate_vs_train": float(wins / (wins + losses)) if (wins + losses) > 0 else None,
                }
            )

        merged_ranking.extend([{"person_id": pid, "score": float(score), "split": "train"} for pid, score in tr.items()])
        merged_ranking.extend([{"person_id": pid, "score": float(score), "split": "test"} for pid, score in te.items()])
        merged_ranking.sort(key=lambda item: item["score"], reverse=True)

        rank_map = {item["person_id"]: idx + 1 for idx, item in enumerate(merged_ranking)}
        for row in summary_rows:
            if row["camera"] == camera and row["test_person"] in rank_map:
                row["rank_in_merged"] = rank_map[row["test_person"]]
                row["train_count_same_camera"] = len(tr)
                row["test_count_same_camera"] = len(te)

        result["cameras"][camera] = {
            "n_train_person": len(tr),
            "n_test_person": len(te),
            "test_vs_train_pairwise": pairwise_rows,
            "merged_ranking": merged_ranking,
        }

        if args.rank_dir:
            merged_scores = {item["person_id"]: float(item["score"]) for item in merged_ranking}
            merged_acc, merged_total = _pairwise_acc_from_labels(
                merged_scores,
                pair_label_map.get(camera, []),
            )
            test_train_acc, test_train_total = _pairwise_acc_from_labels(
                merged_scores,
                pair_label_map.get(camera, []),
                allowed_left_ids=set(te.keys()),
                allowed_right_ids=set(tr.keys()),
            )
            result["cameras"][camera]["pairwise_acc_vs_rank_labels"] = merged_acc
            result["cameras"][camera]["pairwise_acc_vs_rank_labels_n_pairs"] = merged_total
            result["cameras"][camera]["test_vs_train_acc_vs_rank_labels"] = test_train_acc
            result["cameras"][camera]["test_vs_train_acc_vs_rank_labels_n_pairs"] = test_train_total

    result["summary"] = {
        "n_cameras": len(cameras),
        "n_test_person_total": total_test_person,
        "n_test_person_with_train_reference": covered_test_person,
        "n_test_vs_train_pairs": total_pairs,
        "score_stat": args.score_stat,
    }

    if args.rank_dir:
        merged_accs = [
            camera_obj["pairwise_acc_vs_rank_labels"]
            for camera_obj in result["cameras"].values()
            if camera_obj.get("pairwise_acc_vs_rank_labels") is not None
        ]
        test_train_accs = [
            camera_obj["test_vs_train_acc_vs_rank_labels"]
            for camera_obj in result["cameras"].values()
            if camera_obj.get("test_vs_train_acc_vs_rank_labels") is not None
        ]
        result["summary"]["pairwise_acc_vs_rank_labels_mean"] = (
            float(sum(merged_accs) / len(merged_accs)) if merged_accs else None
        )
        result["summary"]["test_vs_train_acc_vs_rank_labels_mean"] = (
            float(sum(test_train_accs) / len(test_train_accs)) if test_train_accs else None
        )

    out_path = args.out or os.path.join(args.data_root, f"heightmap_train_gallery_{args.score_stat}.json")
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.splitext(out_path)[0] + "_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    debug_csv = os.path.splitext(out_path)[0] + "_frame_scores.csv"
    pd.DataFrame(train_debug + test_debug).to_csv(debug_csv, index=False)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"[RESULT] written: {out_path}")
    print(f"[RESULT] summary_csv: {summary_csv}")
    print(f"[RESULT] frame_scores_csv: {debug_csv}")


if __name__ == "__main__":
    main()
