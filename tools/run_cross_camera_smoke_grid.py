#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(root: Path, out_root: Path, cross: float, track: float, gpu: int, args) -> dict:
    name = f"cross_{cross:g}_track_{track:g}"
    out_dir = out_root / name
    log_path = out_root / "logs" / f"{name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "tools/train_cross_camera_heightmap_fusion.py",
        "--out-dir", str(out_dir),
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--consistency-batch-size", str(args.consistency_batch_size),
        "--track-batch-size", str(args.track_batch_size),
        "--limit-per-split-camera", str(args.limit_per_split_camera),
        "--lambda-cross", str(cross),
        "--lambda-track", str(track),
        "--device", f"cuda:{gpu}",
        "--seed", str(args.seed),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True)
    result_path = out_dir / "results.json"
    return {
        "name": name,
        "lambda_cross": cross,
        "lambda_track": track,
        "gpu": gpu,
        "returncode": completed.returncode,
        "log": str(log_path),
        "result": str(result_path),
        "payload": json.loads(result_path.read_text(encoding="utf-8")) if completed.returncode == 0 else None,
    }


def _selection_key(item: dict) -> tuple:
    metrics = item["payload"]["best_val_metrics"]
    return (
        metrics.get("cross_camera_pairwise_accuracy") or -1.0,
        metrics.get("all_pairwise_accuracy") or -1.0,
        metrics.get("kendall_tau") or -1.0,
        -float(item["lambda_cross"]),
        -float(item["lambda_track"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("runs/cross_camera_heightmap_fusion/smoke_grid"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--consistency-batch-size", type=int, default=4)
    parser.add_argument("--track-batch-size", type=int, default=4)
    parser.add_argument("--limit-per-split-camera", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out_root.mkdir(parents=True, exist_ok=True)
    combinations = [(cross, track) for cross in (0.1, 0.2, 0.5) for track in (0.05, 0.1, 0.2)]
    jobs = [(cross, track, args.gpus[index % len(args.gpus)]) for index, (cross, track) in enumerate(combinations)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(_run, root, args.out_root, cross, track, gpu, args) for cross, track, gpu in jobs]
        results = []
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            results.append(item)
            print("[GRID]", item["name"], "returncode=", item["returncode"], flush=True)
    failures = [item for item in results if item["returncode"] != 0]
    if failures:
        raise RuntimeError(f"smoke grid failures: {[item['name'] for item in failures]}")
    best = max(results, key=_selection_key)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "selection_order": ["cross_camera_pairwise_accuracy", "all_pairwise_accuracy", "kendall_tau", "smaller_lambda_cross", "smaller_lambda_track"],
        "best": {key: best[key] for key in ("name", "lambda_cross", "lambda_track", "result", "log")},
        "runs": [
            {
                "name": item["name"],
                "lambda_cross": item["lambda_cross"],
                "lambda_track": item["lambda_track"],
                "gpu": item["gpu"],
                "result": item["result"],
                "log": item["log"],
                "best_val_metrics": item["payload"]["best_val_metrics"],
            }
            for item in sorted(results, key=lambda row: row["name"])
        ],
    }
    (args.out_root / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2))
    print("[OUT]", args.out_root / "selection.json")


if __name__ == "__main__":
    main()
