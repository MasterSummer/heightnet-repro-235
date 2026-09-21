#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.convert_all_to_npz import ParsingCache, sample_frame_indices_from_bbox


def _manifest_index(paths: list[Path]) -> dict[tuple[str, str], tuple[str, str]]:
    out = {}
    for csv_path in paths:
        df = pd.read_csv(csv_path)
        for row in df.to_dict(orient="records"):
            person = str(row.get("person_id", "")).strip()
            stem = Path(str(row.get("video_filename", ""))).stem
            out[(person, stem)] = (str(row.get("video_path", "")).strip(), csv_path.stem)
    return out


def _record(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            bbox = np.asarray(data["bbox_feats"], dtype=np.float32)
            crops = np.asarray(data["heightmap_crops"], dtype=np.float32)
            valid_count = int(np.asarray(data["valid_count"]).reshape(-1)[0])
    except Exception:
        return None
    n = min(valid_count, len(bbox), len(crops))
    if n <= 0:
        return None
    return {
        "path": path,
        "person": path.parent.name,
        "stem": path.stem,
        "valid_count": n,
        "median_h": float(np.median(bbox[:n, 0])),
        "min_h": float(np.min(bbox[:n, 0])),
        "median_w": float(np.median(bbox[:n, 1])),
        "median_score": float(np.median(bbox[:n, 6])),
        "min_score": float(np.min(bbox[:n, 6])),
        "crop_std": float(np.std(crops[:n])),
    }


def _sample_categories(records: list[dict], samples_per_category: int, seed: int) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    categories = {
        "random": records[:],
        "small_kept_h008_012": [r for r in records if 0.08 <= r["median_h"] < 0.12],
        "low_score_kept_lt04": [r for r in records if r["median_score"] < 0.4],
        "wide_or_close_h_ge025": [r for r in records if r["median_h"] >= 0.25],
        "low_crop_std": sorted(records, key=lambda r: r["crop_std"])[: max(2000, samples_per_category)],
    }
    selected = []
    seen = set()
    for category, rows in categories.items():
        rows = list(rows)
        rng.shuffle(rows)
        for record in rows[:samples_per_category]:
            key = str(record["path"])
            if key in seen:
                continue
            seen.add(key)
            selected.append((category, record))
    return selected


def _map_filtered_rows_to_source(
    filtered_bbox: np.ndarray,
    filtered_crops: np.ndarray,
    source_bbox: np.ndarray,
    source_crops: np.ndarray,
) -> list[int]:
    source_indices: list[int] = []
    ptr = 0
    for row_idx in range(len(filtered_bbox)):
        found = False
        while ptr < len(source_bbox):
            bbox_match = np.allclose(source_bbox[ptr, :7], filtered_bbox[row_idx, :7], rtol=1e-5, atol=1e-6)
            crop_match = (
                source_crops[ptr].shape == filtered_crops[row_idx].shape
                and np.allclose(source_crops[ptr], filtered_crops[row_idx], rtol=1e-5, atol=1e-6)
            )
            if bbox_match and crop_match:
                source_indices.append(ptr)
                ptr += 1
                found = True
                break
            ptr += 1
        if not found:
            raise ValueError(f"could not map filtered row {row_idx} back to source row")
    return source_indices


def _export_one(
    category: str,
    record: dict,
    index: int,
    manifest: dict[tuple[str, str], tuple[str, str]],
    json_root: Path,
    source_root: Path | None,
    out_dir: Path,
    frames_per_panel: int,
    parsing_cache: ParsingCache,
    min_bbox_h_norm: float,
    min_bbox_w_norm: float,
    min_bbox_score: float,
) -> tuple[dict | None, dict | None]:
    key = (record["person"], record["stem"])
    if key not in manifest:
        return None, {"record": str(record["path"]), "reason": "missing_manifest"}
    video_path, split = manifest[key]
    video = Path(video_path)
    json_path = json_root / f"{record['person']}_{record['stem']}.json"
    if not video.exists() or not json_path.exists():
        return None, {
            "record": str(record["path"]),
            "reason": "missing_video_or_json",
            "video": str(video),
            "json": str(json_path),
        }

    try:
        frame_bbox_map = parsing_cache.frame_bbox_map(json_path)
        capture = cv2.VideoCapture(str(video))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = sample_frame_indices_from_bbox(frame_bbox_map, total_frames, 20)
        with np.load(record["path"], allow_pickle=False) as data:
            crops = np.asarray(data["heightmap_crops"], dtype=np.float32)
            bbox_feats = np.asarray(data["bbox_feats"], dtype=np.float32)
            valid_count = int(np.asarray(data["valid_count"]).reshape(-1)[0])

        source_indices = list(range(len(crops)))
        if source_root is not None:
            source_path = source_root / record["person"] / f"{record['stem']}.npz"
            with np.load(source_path, allow_pickle=False) as source_data:
                source_bbox = np.asarray(source_data["bbox_feats"], dtype=np.float32)
                source_crops = np.asarray(source_data["heightmap_crops"], dtype=np.float32)
                source_valid_count = int(np.asarray(source_data["valid_count"]).reshape(-1)[0])
            source_n = min(source_valid_count, len(source_bbox), len(source_crops), len(sample_indices))
            filtered_n = min(valid_count, len(crops), len(bbox_feats))
            source_indices = _map_filtered_rows_to_source(
                bbox_feats[:filtered_n],
                crops[:filtered_n],
                source_bbox[:source_n],
                source_crops[:source_n],
            )

        n = min(valid_count, len(crops), len(bbox_feats), len(source_indices))
        if n <= 0:
            capture.release()
            return None, {"record": str(record["path"]), "reason": "no_rows"}

        picks = np.linspace(0, n - 1, num=min(frames_per_panel, n)).round().astype(int).tolist()
        fig, axes = plt.subplots(len(picks), 2, figsize=(9, 3.2 * len(picks)))
        if len(picks) == 1:
            axes = np.asarray([axes])

        suspicious = []
        for row_idx, pick in enumerate(picks):
            source_pick = source_indices[pick]
            frame_idx = sample_indices[source_pick]
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = capture.read()
            if ok:
                entry = frame_bbox_map.get(int(frame_idx))
                if entry:
                    x1, y1, x2, y2 = [int(round(v)) for v in entry["bbox_xyxy"]]
                    frame_h, frame_w = frame.shape[:2]
                    bw = max(0, x2 - x1)
                    bh = max(0, y2 - y1)
                    score = float(entry.get("score", 0.0))
                    if score < 0.25 or bh / max(frame_h, 1) < 0.08 or bw / max(frame_w, 1) < 0.02:
                        suspicious.append(pick)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    label = f"row={pick} src={source_pick} frame={frame_idx} s={score:.2f} h={bh/frame_h:.3f} w={bw/frame_w:.3f}"
                    cv2.putText(frame, label, (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                axes[row_idx, 0].imshow(frame)
            axes[row_idx, 0].axis("off")
            axes[row_idx, 0].set_title("original + bbox")
            axes[row_idx, 1].imshow(crops[pick, 0], cmap="viridis", aspect="auto")
            axes[row_idx, 1].axis("off")
            axes[row_idx, 1].set_title(f"height crop row={pick} src={source_pick} h={bbox_feats[pick,0]:.3f} w={bbox_feats[pick,1]:.3f} s={bbox_feats[pick,6]:.2f}")
        capture.release()

        fig.suptitle(
            f"{category} | {split} | {record['person']} | {record['stem']} | "
            f"vc={record['valid_count']} med_h={record['median_h']:.3f} med_s={record['median_score']:.2f}",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        rel = Path(category) / f"{index:03d}__{record['person']}__{record['stem']}.jpg"
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        return None, {"record": str(record["path"]), "reason": repr(exc)}

    item = {k: v for k, v in record.items() if k != "path"}
    item.update(
        {
            "category": category,
            "split": split,
            "npz_path": str(record["path"]),
            "video_path": str(video),
            "json_path": str(json_path),
            "image": str(rel),
            "sampled_rows": picks,
            "source_rows": [source_indices[pick] for pick in picks],
            "suspicious_sample_rows": suspicious,
        }
    )
    return item, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export strict source-frame bbox overlay panels aligned with height crops.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=None, help="Original unfiltered NPZ root used to reconstruct filtered row to source frame mapping.")
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", nargs="+", type=Path, default=[Path("/home/zyding/data/train.csv"), Path("/home/zyding/data/val.csv"), Path("/home/zyding/data/test.csv")])
    parser.add_argument("--samples-per-category", type=int, default=8)
    parser.add_argument("--frames-per-panel", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-bbox-h-norm", type=float, default=0.08)
    parser.add_argument("--min-bbox-w-norm", type=float, default=0.02)
    parser.add_argument("--min-bbox-score", type=float, default=0.25)
    args = parser.parse_args()

    records = [record for path in sorted(args.feature_root.glob("**/*.npz")) if (record := _record(path)) is not None]
    selected = _sample_categories(records, int(args.samples_per_category), int(args.seed))
    manifest = _manifest_index(args.manifest)
    parsing_cache = ParsingCache()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    exports = []
    failures = []
    for index, (category, record) in enumerate(selected):
        item, failure = _export_one(
            category,
            record,
            index,
            manifest,
            args.json_root,
            args.source_root,
            args.out_dir,
            int(args.frames_per_panel),
            parsing_cache,
            float(args.min_bbox_h_norm),
            float(args.min_bbox_w_norm),
            float(args.min_bbox_score),
        )
        if item is not None:
            exports.append(item)
        if failure is not None:
            failures.append(failure)

    payload = {
        "seed": int(args.seed),
        "feature_root": str(args.feature_root),
        "json_root": str(args.json_root),
        "record_count": len(records),
        "selected": len(selected),
        "exported": len(exports),
        "failures": failures,
        "exports": exports,
    }
    (args.out_dir / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    parts = ["<html><head><meta charset='utf-8'><title>Strict Crop Overlay Audit</title></head><body>"]
    parts.append("<h1>Strict Crop Overlay Audit</h1>")
    parts.append("<pre>" + html.escape(json.dumps({k: payload[k] for k in ("record_count", "selected", "exported")}, ensure_ascii=False, indent=2)) + "</pre>")
    for item in exports:
        parts.append("<div style='margin:20px 0;border-top:1px solid #ccc;padding-top:12px'>")
        parts.append(
            f"<h2>{html.escape(item['category'])} | {html.escape(item['split'])} | "
            f"{html.escape(item['person'])} | med_h={item['median_h']:.3f} med_score={item['median_score']:.2f}</h2>"
        )
        parts.append(f"<p><code>{html.escape(item['stem'])}</code><br><code>{html.escape(item['npz_path'])}</code></p>")
        parts.append(f"<img src='{html.escape(item['image'])}' style='max-width:100%;height:auto'>")
        parts.append("</div>")
    parts.append("</body></html>")
    (args.out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "selected": len(selected), "exported": len(exports), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
