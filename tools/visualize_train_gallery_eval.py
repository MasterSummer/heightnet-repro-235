from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def _require_file(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _plot_merged_ranking(camera: str, merged_ranking: list[dict], out_path: str) -> None:
    if not merged_ranking:
        return

    labels = [item["person_id"] for item in merged_ranking]
    scores = [float(item["score"]) for item in merged_ranking]
    colors = ["#4C78A8" if item.get("split") == "train" else "#F58518" for item in merged_ranking]

    width = max(10, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.bar(range(len(labels)), scores, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Predicted Height Score")
    ax.set_title(f"{camera}: merged ranking (train+test)")
    ax.grid(axis="y", alpha=0.25)

    for idx, item in enumerate(merged_ranking):
        ax.text(idx, scores[idx], item.get("split", ""), rotation=90, ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_pairwise_heatmap(camera: str, pairwise_rows: list[dict], out_path: str) -> None:
    if not pairwise_rows:
        return

    test_people = sorted({row["test_person"] for row in pairwise_rows})
    train_people = sorted({row["train_person"] for row in pairwise_rows})
    if not test_people or not train_people:
        return

    value_map = {
        (row["test_person"], row["train_person"]): int(row["pred_test_taller"]) for row in pairwise_rows
    }
    matrix = [
        [value_map.get((test_person, train_person), 0) for train_person in train_people]
        for test_person in test_people
    ]

    fig_w = max(8, 0.7 * len(train_people))
    fig_h = max(5, 0.6 * len(test_people))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(train_people)))
    ax.set_yticks(range(len(test_people)))
    ax.set_xticklabels(train_people, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(test_people, fontsize=9)
    ax.set_xlabel("Train Person")
    ax.set_ylabel("Test Person")
    ax.set_title(f"{camera}: test vs train taller-prediction")

    for r, test_person in enumerate(test_people):
        for c, train_person in enumerate(train_people):
            value = matrix[r][c]
            ax.text(c, r, str(value), ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("pred_test_taller")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_summary_rows(camera: str, merged_ranking: list[dict], pairwise_rows: list[dict]) -> list[dict]:
    rank_map = {item["person_id"]: idx + 1 for idx, item in enumerate(merged_ranking)}
    train_count = sum(1 for item in merged_ranking if item.get("split") == "train")
    test_count = sum(1 for item in merged_ranking if item.get("split") == "test")

    by_test_person: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})
    score_map: dict[str, float] = {}
    for item in merged_ranking:
        score_map[item["person_id"]] = float(item["score"])

    for row in pairwise_rows:
        test_person = row["test_person"]
        if int(row["pred_test_taller"]) == 1:
            by_test_person[test_person]["wins"] += 1
        else:
            by_test_person[test_person]["losses"] += 1

    rows = []
    for person_id, stats in sorted(by_test_person.items()):
        rows.append(
            {
                "camera": camera,
                "test_person": person_id,
                "score": score_map.get(person_id),
                "rank_in_merged": rank_map.get(person_id),
                "train_count_same_camera": train_count,
                "test_count_same_camera": test_count,
                "wins_vs_train": stats["wins"],
                "losses_vs_train": stats["losses"],
                "win_rate_vs_train": (
                    stats["wins"] / (stats["wins"] + stats["losses"])
                    if (stats["wins"] + stats["losses"]) > 0
                    else None
                ),
            }
        )
    return rows


def _plot_accuracy_bar(
    items: list[tuple[str, float, int]],
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    if not items:
        return

    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    counts = [item[2] for item in items]

    width = max(8, 0.7 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(range(len(labels)), values, color="#54A24B")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    for idx, (value, count) in enumerate(zip(values, counts)):
        ax.text(idx, value, f"{value:.3f}\nn={count}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize train-gallery evaluation JSON.")
    parser.add_argument("--input", type=str, required=True, help="Path to eval_train_gallery.json")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory for plots and CSV")
    args = parser.parse_args()

    _require_file(args.input, "input json")
    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.input), "eval_train_gallery_vis")
    ranking_dir = os.path.join(out_dir, "merged_ranking")
    heatmap_dir = os.path.join(out_dir, "pairwise_heatmap")
    metrics_dir = os.path.join(out_dir, "metrics")
    _ensure_dir(ranking_dir)
    _ensure_dir(heatmap_dir)
    _ensure_dir(metrics_dir)

    summary_rows: list[dict] = []
    merged_acc_items: list[tuple[str, float, int]] = []
    test_train_acc_items: list[tuple[str, float, int]] = []
    cameras = payload.get("cameras", {})
    for camera, camera_obj in sorted(cameras.items()):
        merged_ranking = camera_obj.get("merged_ranking", [])
        pairwise_rows = camera_obj.get("test_vs_train_pairwise", [])

        safe_camera = _safe_name(camera)
        _plot_merged_ranking(
            camera,
            merged_ranking,
            os.path.join(ranking_dir, f"{safe_camera}_merged_ranking.png"),
        )
        _plot_pairwise_heatmap(
            camera,
            pairwise_rows,
            os.path.join(heatmap_dir, f"{safe_camera}_pairwise_heatmap.png"),
        )
        rows = _build_summary_rows(camera, merged_ranking, pairwise_rows)

        merged_acc = camera_obj.get("pairwise_acc_vs_rank_labels")
        merged_acc_n = camera_obj.get("pairwise_acc_vs_rank_labels_n_pairs")
        test_train_acc = camera_obj.get("test_vs_train_acc_vs_rank_labels")
        test_train_acc_n = camera_obj.get("test_vs_train_acc_vs_rank_labels_n_pairs")
        if merged_acc is not None:
            merged_acc_items.append((camera, float(merged_acc), int(merged_acc_n or 0)))
        if test_train_acc is not None:
            test_train_acc_items.append((camera, float(test_train_acc), int(test_train_acc_n or 0)))

        for row in rows:
            row["pairwise_acc_vs_rank_labels"] = merged_acc
            row["pairwise_acc_vs_rank_labels_n_pairs"] = merged_acc_n
            row["test_vs_train_acc_vs_rank_labels"] = test_train_acc
            row["test_vs_train_acc_vs_rank_labels_n_pairs"] = test_train_acc_n
        summary_rows.extend(rows)

    summary_csv = os.path.join(out_dir, "test_vs_train_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    _plot_accuracy_bar(
        merged_acc_items,
        title="Merged Pairwise Accuracy vs Rank Labels",
        ylabel="Accuracy",
        out_path=os.path.join(metrics_dir, "pairwise_acc_vs_rank_labels.png"),
    )
    _plot_accuracy_bar(
        test_train_acc_items,
        title="Test-vs-Train Accuracy vs Rank Labels",
        ylabel="Accuracy",
        out_path=os.path.join(metrics_dir, "test_vs_train_acc_vs_rank_labels.png"),
    )

    print(f"[VIS] ranking_dir={ranking_dir}")
    print(f"[VIS] heatmap_dir={heatmap_dir}")
    print(f"[VIS] metrics_dir={metrics_dir}")
    print(f"[VIS] summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
