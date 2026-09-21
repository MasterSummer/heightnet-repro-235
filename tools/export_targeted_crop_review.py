#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _record_from_npz(path: Path) -> dict | None:
    try:
        data = np.load(path)
        bbox = np.asarray(data["bbox_feats"], dtype=np.float32)
        crops = np.asarray(data["heightmap_crops"], dtype=np.float32)
        valid_count = int(np.asarray(data["valid_count"]).reshape(-1)[0])
    except Exception:
        return None
    if valid_count <= 0 or bbox.shape[0] < valid_count or crops.shape[0] < valid_count:
        return None
    h = bbox[:valid_count, 0]
    w = bbox[:valid_count, 1]
    score = bbox[:valid_count, 6]
    crop_sum = np.abs(crops[:valid_count]).reshape(valid_count, -1).sum(axis=1)
    return {
        "path": path,
        "person_id": path.parent.name,
        "stem": path.stem,
        "valid_count": valid_count,
        "median_h": float(np.median(h)),
        "min_h": float(np.min(h)),
        "median_w": float(np.median(w)),
        "median_score": float(np.median(score)),
        "min_score": float(np.min(score)),
        "crop_std": float(np.std(crops[:valid_count])),
        "min_crop_sum": float(np.min(crop_sum)),
        "crops": crops[:valid_count],
    }


def _sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def _save_panel(record: dict, category: str, idx: int, out_dir: Path) -> Path | None:
    crops = record["crops"]
    n = min(8, int(crops.shape[0]))
    if n <= 0:
        return None
    chosen = np.linspace(0, crops.shape[0] - 1, num=n).round().astype(int).tolist()
    fig, axes = plt.subplots(1, n + 1, figsize=(2.3 * (n + 1), 3.2))
    vals = crops[chosen, 0]
    vmin = float(np.percentile(vals, 2))
    vmax = float(np.percentile(vals, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    for ax, frame_idx, crop in zip(axes[:-1], chosen, vals):
        ax.imshow(crop, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"f{frame_idx}", fontsize=9)
        ax.axis("off")
    axes[-1].imshow(crops[:, 0].mean(axis=0), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[-1].set_title("mean", fontsize=9)
    axes[-1].axis("off")
    title = (
        f"{category} | {record['person_id']} | {record['stem']}\n"
        f"vc={record['valid_count']} med_h={record['median_h']:.3f} "
        f"med_w={record['median_w']:.3f} med_score={record['median_score']:.3f} "
        f"crop_std={record['crop_std']:.4f}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    rel_path = Path(category) / f"{idx:02d}__{record['person_id']}__{record['stem']}.png"
    out_path = out_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return rel_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export targeted heightmap crop review panels.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    records = [r for p in sorted(args.feature_root.glob("**/*.npz")) if (r := _record_from_npz(p)) is not None]
    categories = {
        "low_valid_lt5": [r for r in records if r["valid_count"] < 5],
        "low_valid_5to7": [r for r in records if 5 <= r["valid_count"] < 8],
        "small_person_median_h_lt006": [r for r in records if r["median_h"] < 0.06],
        "low_score_median_lt04": [r for r in records if r["median_score"] < 0.40],
        "good_reference_vc_ge12_h_ge010_score_ge07": [
            r for r in records if r["valid_count"] >= 12 and r["median_h"] >= 0.10 and r["median_score"] >= 0.70
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for category, rows in categories.items():
        for idx, record in enumerate(_sample(rows, int(args.samples_per_category), rng)):
            rel = _save_panel(record, category, idx, args.out_dir)
            if rel is None:
                continue
            item = {
                key: record[key]
                for key in (
                    "person_id",
                    "stem",
                    "valid_count",
                    "median_h",
                    "min_h",
                    "median_w",
                    "median_score",
                    "min_score",
                    "crop_std",
                    "min_crop_sum",
                )
            }
            item["category"] = category
            item["npz_path"] = str(record["path"])
            item["image"] = str(rel)
            exports.append(item)

    summary = {key: len(value) for key, value in categories.items()}
    payload = {"summary": summary, "exported": len(exports), "exports": exports}
    (args.out_dir / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    parts = ["<html><head><meta charset=\"utf-8\"><title>Targeted Crop Review</title></head><body>"]
    parts.append("<h1>Targeted Crop Review</h1>")
    parts.append("<pre>" + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + "</pre>")
    for category in categories:
        parts.append(f"<h2>{html.escape(category)}</h2>")
        for item in [x for x in exports if x["category"] == category]:
            parts.append("<div style=\"margin:18px 0;border-top:1px solid #ccc;padding-top:12px\">")
            parts.append(
                "<p><b>{}</b><br>vc={} med_h={:.3f} med_score={:.3f}<br><code>{}</code></p>".format(
                    html.escape(item["stem"]),
                    item["valid_count"],
                    item["median_h"],
                    item["median_score"],
                    html.escape(item["npz_path"]),
                )
            )
            parts.append(f"<img src=\"{html.escape(item['image'])}\" style=\"max-width:100%;height:auto\">")
            parts.append("</div>")
    parts.append("</body></html>")
    (args.out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "summary": summary, "exported": len(exports)}, indent=2))


if __name__ == "__main__":
    main()
