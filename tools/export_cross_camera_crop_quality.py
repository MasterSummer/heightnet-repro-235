#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cross_camera_heightmap_fusion_core import load_height_labels, load_npz_frames, load_split_rows


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _apply_person_split(rows: list[dict], person_split: dict[str, list[str]]) -> list[dict]:
    person_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for person_id in person_split.get(split, []):
            person_to_split[str(person_id)] = split

    output: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        split = person_to_split.get(str(row.get("person_id")))
        if split is None:
            continue
        key = (str(row.get("person_id")), str(row.get("video_stem") or Path(str(row.get("video_filename", ""))).stem))
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["split"] = split
        output.append(item)
    return output


def _load_rows(data_root: Path, feature_root: Path, person_split_json: Path | None, splits: list[str]) -> list[dict]:
    rows_by_split = {
        split: load_split_rows(data_root / f"{split}.csv", feature_root, split)
        for split in ("train", "val", "test")
    }
    rows = sum(rows_by_split.values(), [])
    if person_split_json:
        split_payload = json.loads(person_split_json.read_text(encoding="utf-8"))
        rows = _apply_person_split(rows, split_payload)
    return [row for row in rows if row["split"] in set(splits)]


def _balanced_sample(rows: list[dict], *, samples_per_split: int, samples_per_camera: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_split_camera: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_split_camera[(str(row["split"]), str(row["camera_id"]))].append(row)

    sampled: list[dict] = []
    split_counts: dict[str, int] = defaultdict(int)
    for key in sorted(by_split_camera):
        group = by_split_camera[key]
        order = rng.permutation(len(group)).tolist()
        limit = len(group) if samples_per_camera <= 0 else min(samples_per_camera, len(group))
        for idx in order[:limit]:
            row = group[idx]
            split = str(row["split"])
            if samples_per_split > 0 and split_counts[split] >= samples_per_split:
                continue
            sampled.append(row)
            split_counts[split] += 1
    sampled.sort(key=lambda item: (str(item["split"]), str(item["camera_id"]), str(item["person_id"]), str(item["video_stem"])))
    return sampled


def _pick_frame_indices(count: int, wanted: int) -> list[int]:
    if count <= 0:
        return []
    if wanted <= 1:
        return [count // 2]
    positions = np.linspace(0, count - 1, num=min(wanted, count))
    return sorted({int(round(float(pos))) for pos in positions})


def _normalize_for_display(images: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([img[np.isfinite(img)].reshape(-1) for img in images if np.isfinite(img).any()])
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or float(hi - lo) < 1e-6:
        lo, hi = float(values.min()), float(values.max())
    if float(hi - lo) < 1e-6:
        hi = lo + 1.0
    return float(lo), float(hi)


def _crop_stats(crops: np.ndarray, tabular: np.ndarray) -> dict:
    crop_values = crops.astype(np.float32, copy=False)
    finite = crop_values[np.isfinite(crop_values)]
    if finite.size:
        p01, p50, p99 = np.percentile(finite, [1.0, 50.0, 99.0])
        crop_min = float(finite.min())
        crop_max = float(finite.max())
        crop_mean = float(finite.mean())
        crop_std = float(finite.std())
        nonzero_ratio = float(np.mean(np.abs(finite) > 1e-6))
    else:
        p01 = p50 = p99 = crop_min = crop_max = crop_mean = crop_std = nonzero_ratio = 0.0

    bbox_h = tabular[:, 0] if tabular.size else np.asarray([], dtype=np.float32)
    bbox_w = tabular[:, 1] if tabular.size else np.asarray([], dtype=np.float32)
    rect_score = tabular[:, 6] if tabular.shape[1] > 6 else np.asarray([], dtype=np.float32)
    flags: list[str] = []
    if crops.shape[0] < 2:
        flags.append("very_few_valid_frames")
    if nonzero_ratio < 0.05:
        flags.append("mostly_empty_crop")
    if crop_std < 1e-4:
        flags.append("low_crop_contrast")
    if bbox_h.size and float(np.nanmedian(bbox_h)) < 0.05:
        flags.append("tiny_bbox_height")

    return {
        "valid_count": int(crops.shape[0]),
        "crop_min": crop_min,
        "crop_p01": float(p01),
        "crop_p50": float(p50),
        "crop_p99": float(p99),
        "crop_max": crop_max,
        "crop_mean": crop_mean,
        "crop_std": crop_std,
        "nonzero_ratio": nonzero_ratio,
        "bbox_h_mean": float(np.nanmean(bbox_h)) if bbox_h.size else None,
        "bbox_w_mean": float(np.nanmean(bbox_w)) if bbox_w.size else None,
        "rect_score_mean": float(np.nanmean(rect_score)) if rect_score.size else None,
        "flags": flags,
    }


def _save_crop_panel(
    out_path: Path,
    row: dict,
    crops: np.ndarray,
    tabular: np.ndarray,
    frame_indices: list[int],
    stats: dict,
    height_cm: float | None,
) -> None:
    selected = [crops[idx, 0] for idx in frame_indices]
    mean_crop = crops[:, 0].mean(axis=0)
    display_images = selected + [mean_crop]
    vmin, vmax = _normalize_for_display(display_images)

    fig, axes = plt.subplots(1, len(display_images), figsize=(3.0 * len(display_images), 4.8), squeeze=False)
    axes_flat = axes[0]
    for ax, image, label in zip(axes_flat, display_images, [f"f{idx}" for idx in frame_indices] + ["video mean"]):
        ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    height_text = f", h={height_cm:.1f}cm" if height_cm is not None else ""
    flags = ",".join(stats["flags"]) if stats["flags"] else "ok"
    title = (
        f"{row['split']} | {row['person_id']}{height_text} | {row['camera_id']}\n"
        f"{row['video_stem']} | frames={stats['valid_count']} | "
        f"crop_std={stats['crop_std']:.4g} | nonzero={stats['nonzero_ratio']:.2f} | {flags}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_html_gallery(out_dir: Path, exported: list[dict]) -> Path:
    def esc(value: object) -> str:
        text = str(value)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    cards = []
    for item in sorted(exported, key=lambda x: (not bool(x["flags"]), x["split"], x["camera_id"], x["person_id"], x["video_stem"])):
        rel = Path(item["out_path"]).resolve().relative_to(out_dir.resolve())
        flags = ", ".join(item["flags"]) if item["flags"] else "ok"
        cards.append(
            "<article class='card'>"
            f"<a href='{esc(rel)}'><img src='{esc(rel)}' loading='lazy'></a>"
            f"<div><b>{esc(item['split'])} / {esc(item['camera_id'])} / {esc(item['person_id'])}</b></div>"
            f"<div>{esc(item['video_stem'])}</div>"
            f"<div>frames={esc(item['valid_count'])} std={float(item['crop_std']):.4g} nonzero={float(item['nonzero_ratio']):.2f}</div>"
            f"<div class='flags'>{esc(flags)}</div>"
            "</article>"
        )
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HeightNet Crop Quality</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
.card img { width: 100%; display: block; border-radius: 4px; }
.flags { color: #a33; font-weight: 600; }
</style>
</head>
<body>
<h1>HeightNet Crop Quality</h1>
<p>Sorted with flagged samples first. Click any image to open the full panel.</p>
<section class="grid">
""" + "\n".join(cards) + """
</section>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export video-level heightmap crop panels for manual crop-quality inspection.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/zyding/data"))
    parser.add_argument("--feature-root", type=Path, default=Path("/home/zyding/height/jianzhi_2511_sequence/features"))
    parser.add_argument("--person-split-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--samples-per-split", type=int, default=60, help="0 means no per-split cap.")
    parser.add_argument("--samples-per-camera", type=int, default=8, help="0 means no per-camera cap.")
    parser.add_argument("--frames-per-video", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label-path", type=Path, default=Path("/home/zyding/data/label/rank.json"))
    args = parser.parse_args()

    out_dir = _as_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_height_labels(args.label_path) if args.label_path.exists() else {}

    rows = _load_rows(args.data_root, args.feature_root, args.person_split_json, list(args.splits))
    existing = [row for row in rows if Path(row["npz_path"]).exists()]
    sampled = _balanced_sample(
        existing,
        samples_per_split=int(args.samples_per_split),
        samples_per_camera=int(args.samples_per_camera),
        seed=int(args.seed),
    )

    exported: list[dict] = []
    errors: list[dict] = []
    for row in sampled:
        try:
            frames = load_npz_frames(row["npz_path"])
            frame_indices = _pick_frame_indices(frames.count, int(args.frames_per_video))
            stats = _crop_stats(frames.crops, frames.tabular)
            height_cm = labels.get(str(row["person_id"]))
            filename = f"{row['person_id']}__{row['camera_id']}__{row['video_stem']}.png"
            out_path = out_dir / str(row["split"]) / str(row["camera_id"]) / filename
            _save_crop_panel(out_path, row, frames.crops, frames.tabular, frame_indices, stats, height_cm)
            exported.append({
                **{key: row.get(key) for key in ("split", "person_id", "video_filename", "video_stem", "camera_id", "npz_path")},
                "height_cm": height_cm,
                "frame_indices": frame_indices,
                "out_path": str(out_path),
                **stats,
            })
        except Exception as exc:  # noqa: BLE001 - diagnostics should keep scanning after bad samples.
            errors.append({**row, "error": str(exc)})

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "feature_root": str(args.feature_root),
        "person_split_json": str(args.person_split_json) if args.person_split_json else "",
        "splits": list(args.splits),
        "source_rows": len(rows),
        "existing_npz_rows": len(existing),
        "exported_count": len(exported),
        "error_count": len(errors),
        "flag_counts": dict(sorted({
            flag: sum(flag in item["flags"] for item in exported)
            for flag in sorted({flag for item in exported for flag in item["flags"]})
        }.items())),
        "exports": exported,
        "errors": errors,
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = _write_html_gallery(out_dir, exported)
    print(f"[CROP_QUALITY] source_rows={len(rows)} existing_npz_rows={len(existing)} exported={len(exported)} errors={len(errors)}")
    print(f"[CROP_QUALITY] index={index_path}")
    print(f"[CROP_QUALITY] html={html_path}")


if __name__ == "__main__":
    main()
