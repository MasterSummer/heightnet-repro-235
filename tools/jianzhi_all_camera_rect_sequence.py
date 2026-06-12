from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np

VIDEO_RE = re.compile(
    r"(?P<upper>[^_]+)_(?P<lower>[^_]+)_(?P<shoe>[^_]+)_(?P<scene>[^_]+)_"
    r"(?P<height>\d+d\d+)_(?P<angle>\d+)_(?P<pixel>\d+w)_(?P<gender>[^_]+)_(?P<brightness>[^_]+)_(?P<dataset>[^_.]+)$",
    re.I,
)
PERSON_RE = re.compile(r"^(\d{4}_(?:man|woman)\d+)_(.+)$", re.I)

TRAIN_PEOPLE = [
    "1126_man1", "1126_woman1", "1127_man1", "1127_man2", "1127_man3", "1128_man2",
    "1128_woman1", "1201_man1", "1202_man1", "1202_woman2", "1202_woman3", "1203_man1",
    "1203_woman1", "1204_man1", "1204_woman1", "1204_woman2", "1204_woman3", "1205_woman1",
    "1205_woman2", "1209_woman1", "1209_woman2", "1209_woman3",
]
VAL_PEOPLE = ["1127_woman1", "1201_woman2", "1201_woman3"]
TEST_PEOPLE = ["1128_man1", "1201_woman1", "1202_woman1", "1203_woman2", "1205_man1", "1209_man1"]
SPLIT_BY_PERSON = {}
SPLIT_BY_PERSON.update({p: "train" for p in TRAIN_PEOPLE})
SPLIT_BY_PERSON.update({p: "val" for p in VAL_PEOPLE})
SPLIT_BY_PERSON.update({p: "test" for p in TEST_PEOPLE})


def parse_camera(stem: str) -> str | None:
    m = VIDEO_RE.match(stem)
    if not m:
        return None
    return f"{m.group('height').lower()}_{m.group('angle')}_{m.group('pixel').lower()}"


def camera_height_m(camera_id: str) -> float:
    m = re.match(r"(\d+)d(\d+)_", camera_id)
    return float(f"{m.group(1)}.{m.group(2)}") if m else 0.0


def load_heights(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        out = {}
        for row in csv.DictReader(f):
            pid = row.get("penson_id") or row.get("person_id")
            if pid:
                out[pid] = float(row["height(cm)"])
    return out


def read_rect_json(path: Path) -> dict[int, tuple[float, float, float, float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    best: dict[int, tuple[float, float, float, float, float]] = {}
    def visit(obj):
        if isinstance(obj, dict):
            if "frame_id" in obj and "rect" in obj:
                try:
                    fid = int(obj["frame_id"])
                    rect = obj["rect"]
                    x, y, w, h = map(float, rect[:4])
                    score = float(obj.get("score", 0.0))
                    if w > 0 and h > 0 and (fid not in best or score > best[fid][4]):
                        best[fid] = (x, y, w, h, score)
                except Exception:
                    pass
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)
    visit(raw)
    return best


def video_size(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0, 0, 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h, n, fps


def aggregate(values: list[float]) -> float:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return float("nan")
    k = int(len(vals) * 0.1)
    if len(vals) - 2 * k <= 0:
        return float(median(vals))
    return float(sum(vals[k:len(vals)-k]) / (len(vals) - 2 * k))


def rankdata(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j + 2.0) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    if len(a) < 2:
        return None
    aa = np.asarray(a, dtype=np.float64); bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean(); bb = bb - bb.mean()
    den = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    return float((aa * bb).sum() / den) if den > 0 else None


def spearman(pred, gt):
    return pearson(rankdata(pred), rankdata(gt))


def pair_acc(ranking: list[str], heights: dict[str, float]) -> float:
    correct = total = 0
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            a = ranking[i]; b = ranking[j]
            if abs(heights[a] - heights[b]) <= 0:
                continue
            total += 1
            correct += int(heights[a] > heights[b])
    return correct / total if total else 0.0


def bisect_eval(ranking: list[str], heights: dict[str, float], overlap_ratio: float = 0.2):
    n = len(ranking)
    vals = sorted(heights[x] for x in ranking)
    threshold = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    overlap_n = int(round(n * overlap_ratio))
    mid = n // 2
    start = max(0, mid - overlap_n // 2)
    end = min(n, start + overlap_n)
    overlap = set(ranking[start:end])
    high = {sid for idx, sid in enumerate(ranking) if idx < mid or sid in overlap}
    low = {sid for idx, sid in enumerate(ranking) if idx >= mid or sid in overlap}
    hit = 0; sizes = []
    for sid in ranking:
        gt = "high" if heights[sid] >= threshold else "low"
        if sid in high and sid in low:
            ok = True; size = len(high | low)
        elif sid in high:
            ok = gt == "high"; size = len(high)
        elif sid in low:
            ok = gt == "low"; size = len(low)
        else:
            ok = False; size = 0
        hit += int(ok); sizes.append(size)
    avg_size = sum(sizes) / max(1, len(sizes))
    return {"hit_rate": hit / max(1, n), "hit_count": hit, "evaluated_count": n, "avg_candidate_size": avg_size, "search_space_reduction_ratio": 1.0 - avg_size / max(1, n), "overlap_n": len(overlap)}


def evaluate_scores(scores: dict[str, float], seq_meta: dict[str, dict], heights: dict[str, float]):
    ids = [sid for sid, v in scores.items() if math.isfinite(float(v)) and seq_meta[sid]["person_id"] in heights]
    ranking = sorted(ids, key=lambda sid: (-scores[sid], sid))
    h_seq = {sid: heights[seq_meta[sid]["person_id"]] for sid in ids}
    return {
        "n": len(ids),
        "pairwise_accuracy": pair_acc(ranking, h_seq),
        "spearman": spearman([scores[sid] for sid in ids], [h_seq[sid] for sid in ids]),
        "bisect": bisect_eval(ranking, h_seq),
    }


def write_manifest(path: Path, rows: list[dict]):
    fields = ["video_path","sequence_id","person_id","camera_id","frame_start","frame_end","fps","height_cache_path","valid_mask_cache_path","depth_cache_path","bg_depth_path","camera_height_m","frame_path","source_frame_start","source_frame_end","frame_idx","decoded_frame_idx"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", default="/data2/dataset/jianzhi_2511/video2")
    ap.add_argument("--rect-root", default="/data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/scanner/main_card_out")
    ap.add_argument("--rank-csv", default="/home/zyding/height/jianzhi_15_meta/rank.json")
    ap.add_argument("--out-dir", default="/data1/zyding/jianzhi_2511_all_camera_rect_sequence")
    ap.add_argument("--max-frames-per-seq", type=int, default=0, help="0 keeps all rect frames in manifests")
    args = ap.parse_args()

    video_root = Path(args.video_root)
    rect_root = Path(args.rect_root)
    out_dir = Path(args.out_dir)
    heights = load_heights(Path(args.rank_csv))
    allowed = set(SPLIT_BY_PERSON) & set(heights)
    rows_by_split = {"train": [], "val": [], "test": []}
    seq_meta = {}
    metrics_by_seq = defaultdict(lambda: defaultdict(list))
    missing_rect = []
    bad_video = []

    videos = sorted(video_root.glob("*/*.mp4"))
    for vp in videos:
        pid = vp.parent.name
        if pid not in allowed:
            continue
        stem = vp.stem
        cam = parse_camera(stem)
        if not cam:
            continue
        rect_path = rect_root / f"{pid}_{stem}.json"
        if not rect_path.exists():
            missing_rect.append(f"{pid}_{stem}")
            continue
        rects = read_rect_json(rect_path)
        if not rects:
            continue
        vw, vh, nframes, fps = video_size(vp)
        if vw <= 0 or vh <= 0:
            bad_video.append(str(vp)); continue
        sid = f"{pid}__{stem}"
        split = SPLIT_BY_PERSON[pid]
        seq_meta[sid] = {"sequence_id": sid, "person_id": pid, "camera_id": cam, "video_path": str(vp), "n_rect_frames": len(rects), "video_w": vw, "video_h": vh}
        frame_ids = sorted(rects)
        if args.max_frames_per_seq and len(frame_ids) > args.max_frames_per_seq:
            idx = np.linspace(0, len(frame_ids) - 1, args.max_frames_per_seq).round().astype(int).tolist()
            frame_ids = [frame_ids[i] for i in idx]
        for fid in frame_ids:
            x, y, w, h, score = rects[fid]
            x1 = max(0.0, min(float(vw), x)); y1 = max(0.0, min(float(vh), y))
            x2 = max(0.0, min(float(vw), x + w)); y2 = max(0.0, min(float(vh), y + h))
            bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            metrics_by_seq[sid]["bbox_h_norm"].append(bh / vh)
            metrics_by_seq[sid]["bbox_w_norm"].append(bw / vw)
            metrics_by_seq[sid]["bbox_y1_norm_neg"].append(-y1 / vh)
            metrics_by_seq[sid]["bbox_y2_norm"].append(y2 / vh)
            metrics_by_seq[sid]["bbox_cy_norm"].append((y1 + y2) * 0.5 / vh)
            metrics_by_seq[sid]["area_norm"].append((bw * bh) / (vw * vh))
            metrics_by_seq[sid]["rect_score"].append(score)
            rows_by_split[split].append({
                "video_path": str(vp), "sequence_id": sid, "person_id": pid, "camera_id": cam,
                "frame_start": fid, "frame_end": fid, "fps": fps, "camera_height_m": camera_height_m(cam),
                "source_frame_start": 0, "source_frame_end": nframes - 1, "frame_idx": fid, "decoded_frame_idx": fid,
            })

    for split, rows in rows_by_split.items():
        write_manifest(out_dir / f"{split}_manifest.csv", rows)

    seq_features = {}
    for sid, payload in metrics_by_seq.items():
        seq_features[sid] = {k: aggregate(v) for k, v in payload.items()}

    cameras = sorted({m["camera_id"] for m in seq_meta.values()})
    camera_results = {}
    for cam in cameras:
        cam_ids = [sid for sid, m in seq_meta.items() if m["camera_id"] == cam]
        metric_scores = {metric: {sid: seq_features[sid].get(metric, float("nan")) for sid in cam_ids} for metric in ["bbox_h_norm","bbox_w_norm","bbox_y1_norm_neg","bbox_y2_norm","bbox_cy_norm","area_norm","rect_score"]}
        # Simple z-score combos
        def z(metric):
            vals = np.array([v for v in metric_scores[metric].values() if math.isfinite(float(v))], dtype=np.float64)
            mu = float(vals.mean()) if vals.size else 0.0; sd = float(vals.std()) if vals.size else 1.0
            return {sid: ((metric_scores[metric][sid] - mu) / sd if sd > 1e-9 else 0.0) for sid in cam_ids if math.isfinite(float(metric_scores[metric][sid]))}
        zh, zy2, zy1, za = z("bbox_h_norm"), z("bbox_y2_norm"), z("bbox_y1_norm_neg"), z("area_norm")
        combo = {sid: zh.get(sid,0.0) + 0.5*zy2.get(sid,0.0) + 0.5*zy1.get(sid,0.0) + 0.25*za.get(sid,0.0) for sid in cam_ids}
        metric_scores["combo_bbox_position"] = combo
        camera_results[cam] = {metric: evaluate_scores(scores, seq_meta, heights) for metric, scores in metric_scores.items()}

    summary = {
        "out_dir": str(out_dir),
        "rank_csv": str(args.rank_csv),
        "split_people": {"train": TRAIN_PEOPLE, "val": VAL_PEOPLE, "test": TEST_PEOPLE},
        "split_rows": {k: len(v) for k, v in rows_by_split.items()},
        "split_sequences": {k: len({r["sequence_id"] for r in v}) for k, v in rows_by_split.items()},
        "split_people_counts": {k: len({r["person_id"] for r in v}) for k, v in rows_by_split.items()},
        "n_sequences": len(seq_meta),
        "n_cameras": len(cameras),
        "missing_rect_count": len(missing_rect),
        "missing_rect_sample": missing_rect[:20],
        "bad_video_count": len(bad_video),
        "camera_sequence_counts": Counter(m["camera_id"] for m in seq_meta.values()),
        "camera_results": camera_results,
    }
    (out_dir / "sequence_features.json").write_text(json.dumps({"meta": seq_meta, "features": seq_features}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "all_camera_bbox_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["out_dir","split_rows","split_sequences","split_people_counts","n_sequences","n_cameras","missing_rect_count","bad_video_count"]}, ensure_ascii=False, indent=2))
    top = []
    for cam, res in camera_results.items():
        best_metric, best = max(res.items(), key=lambda kv: kv[1].get("pairwise_accuracy") or 0)
        top.append((best.get("pairwise_accuracy") or 0, cam, best_metric, best.get("spearman"), best.get("bisect",{}).get("hit_rate"), best.get("n")))
    print("TOP_CAMERAS")
    for item in sorted(top, reverse=True)[:20]:
        print(item)

if __name__ == "__main__":
    main()
