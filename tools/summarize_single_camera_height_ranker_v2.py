#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_single_camera_height_ranker_v2 import (
    DEFAULT_TEST_CAMERAS,
    load_items,
    pairwise_metrics,
    split_items,
)


GRID = (0.0, 0.25, 0.5, 0.75)


def _load_results(path: Path) -> dict:
    return json.loads((path / "results.json").read_text(encoding="utf-8"))


def _candidate_seed_scores(camera_result: dict, split: str) -> list[dict[str, float]]:
    out = []
    for seed_result in camera_result.get("seeds", []):
        ckpt_path = Path(seed_result["checkpoint_path"])
        import torch

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        out.append({str(k): float(v) for k, v in ckpt.get("scores", {}).get(split, {}).items()})
    return out


def _average_scores(score_dicts: list[dict[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for scores in score_dicts:
        for key, value in scores.items():
            grouped.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _aux_scores(items) -> tuple[dict[str, float], dict[str, float]]:
    geom = {item.sequence_id: float(item.geom_score) for item in items}
    robust = {item.sequence_id: float(item.robust_height_score) for item in items}
    return geom, robust


def _combine(model_scores: dict[str, float], geom_scores: dict[str, float], robust_scores: dict[str, float], alpha: float, beta: float) -> dict[str, float]:
    keys = set(model_scores)
    return {
        key: float(model_scores[key] + alpha * geom_scores.get(key, 0.0) + beta * robust_scores.get(key, 0.0))
        for key in keys
    }


def summarize(args) -> dict:
    result_root = Path(args.result_dir)
    payload = _load_results(result_root)
    result_cameras = {
        camera_id
        for camera_id, camera_result in payload.get("camera_results", {}).items()
        if camera_result and not camera_result.get("skipped")
    }
    if args.target_cameras:
        result_cameras &= set(args.target_cameras)
    items = load_items(Path(args.data_root), Path(args.feature_root), allowed_cameras=result_cameras or None)
    summary = {"cameras": {}, "macro": {}}
    test_values = []
    for camera_id in DEFAULT_TEST_CAMERAS:
        if result_cameras and camera_id not in result_cameras:
            continue
        camera_result = payload.get("camera_results", {}).get(camera_id)
        if not camera_result or camera_result.get("skipped"):
            continue
        by_split = split_items(items, camera_id)
        val_items = by_split["val"] or by_split["train"]
        test_items = by_split["test"]
        val_seed_scores = _candidate_seed_scores(camera_result, "val")
        test_seed_scores = _candidate_seed_scores(camera_result, "test")
        if not val_seed_scores or not test_seed_scores:
            continue
        candidates = []
        for idx, scores in enumerate(val_seed_scores):
            candidates.append((f"seed{idx + 1}", scores, test_seed_scores[idx]))
        candidates.append(("seed_average", _average_scores(val_seed_scores), _average_scores(test_seed_scores)))

        geom_val, robust_val = _aux_scores(val_items)
        geom_test, robust_test = _aux_scores(test_items)
        best = None
        for label, val_model, test_model in candidates:
            for alpha in GRID:
                for beta in GRID:
                    val_scores = _combine(val_model, geom_val, robust_val, alpha, beta)
                    val_metrics = pairwise_metrics(val_items, val_scores)
                    key = val_metrics["pairwise_accuracy"] or -1.0
                    if best is None or key > best["key"]:
                        test_scores = _combine(test_model, geom_test, robust_test, alpha, beta)
                        best = {
                            "key": key,
                            "selection": label,
                            "alpha": alpha,
                            "beta": beta,
                            "val": val_metrics,
                            "test": pairwise_metrics(test_items, test_scores),
                        }
        summary["cameras"][camera_id] = {
            "selection": best["selection"],
            "alpha": best["alpha"],
            "beta": best["beta"],
            "val": best["val"],
            "test": best["test"],
        }
        if best["test"]["pairwise_accuracy"] is not None:
            test_values.append(float(best["test"]["pairwise_accuracy"]))
    summary["macro"] = {
        "camera_count": len(test_values),
        "macro_test_pairwise_accuracy": float(np.mean(test_values)) if test_values else None,
        "median_test_pairwise_accuracy": float(np.median(test_values)) if test_values else None,
        "min_test_pairwise_accuracy": float(np.min(test_values)) if test_values else None,
        "max_test_pairwise_accuracy": float(np.max(test_values)) if test_values else None,
    }
    out_path = Path(args.out) if args.out else result_root / "ensemble_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["macro"], ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="runs/single_camera_height_ranker_v2/default")
    parser.add_argument("--data-root", default="/home/zyding/data")
    parser.add_argument("--feature-root", default="/home/zyding/height/jianzhi_2511_sequence/features")
    parser.add_argument("--out", default="")
    parser.add_argument("--target-cameras", nargs="*", default=None)
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
