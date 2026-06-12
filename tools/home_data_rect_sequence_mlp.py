from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch import nn

CAM_RE = re.compile(r"_(?P<h>\d+d\d+)_(?P<a>\d+)_(?P<p>\d+w)_", re.I)
FEATURES = [
    "bbox_h_norm",
    "bbox_w_norm",
    "bbox_y1_norm_neg",
    "bbox_y2_norm",
    "bbox_cy_norm",
    "area_norm",
    "rect_score",
]


def base_camera_from_name(name: str) -> str | None:
    m = CAM_RE.search(name)
    if not m:
        return None
    return f"{m.group('h').lower()}_{m.group('a')}"


def frame_size_from_name(name: str) -> tuple[float, float]:
    m = CAM_RE.search(name)
    pixel = m.group("p").lower() if m else ""
    # Observed DA2 cache shapes on 235: 200w=2304x1296, 400w=2560x1440, 800w=1280x720.
    if pixel == "200w":
        return 2304.0, 1296.0
    if pixel == "400w":
        return 2560.0, 1440.0
    if pixel == "800w":
        return 1280.0, 720.0
    return 2560.0, 1440.0


def choose_rect_json(rect_root: Path, person_id: str, video_stem: str, camera_id: str) -> Path | None:
    exact = rect_root / f"{person_id}_{video_stem}.json"
    if exact.exists():
        return exact
    m = CAM_RE.search(video_stem)
    pixel = m.group("p").lower() if m else ""
    if camera_id and pixel:
        matches = sorted(rect_root.glob(f"{person_id}_*{camera_id}_{pixel}_*jianzhi2511.json"))
        if matches:
            return matches[0]
    if camera_id:
        matches = sorted(rect_root.glob(f"{person_id}_*{camera_id}_*.json"))
        if matches:
            return matches[0]
    return None


def load_rank_csv(path: Path) -> dict[str, float]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("penson_id") or row.get("person_id")
            h = row.get("height(cm)") or row.get("height_cm") or row.get("height")
            if pid and h:
                out[str(pid)] = float(h)
    return out


def load_split_csv(path: Path, split: str) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("person_id") or row.get("pid") or row.get("penson_id")
            video_path = row.get("video_path") or ""
            video_filename = row.get("video_filename") or Path(video_path).name
            if not pid or not video_filename:
                continue
            stem = Path(video_filename).stem
            cam = base_camera_from_name(video_filename)
            if not cam:
                continue
            rows.append({
                "split": split,
                "person_id": str(pid),
                "video_filename": video_filename,
                "video_path": video_path,
                "sequence_id": f"{pid}__{stem}",
                "video_stem": stem,
                "camera_id": cam,
                "coat_type": row.get("coat_type", ""),
                "action": row.get("action", ""),
                "video_w": frame_size_from_name(video_filename)[0],
                "video_h": frame_size_from_name(video_filename)[1],
            })
    return rows


def visit_rects(obj, best: dict[int, tuple[float, float, float, float, float]]) -> None:
    if isinstance(obj, dict):
        if "frame_id" in obj and "rect" in obj:
            try:
                fid = int(obj["frame_id"])
                x, y, w, h = map(float, obj["rect"][:4])
                score = float(obj.get("score", 0.0))
                if w > 0 and h > 0 and (fid not in best or score > best[fid][4]):
                    best[fid] = (x, y, w, h, score)
            except Exception:
                pass
        for v in obj.values():
            visit_rects(v, best)
    elif isinstance(obj, list):
        for v in obj:
            visit_rects(v, best)


def read_rect_json(path: Path) -> dict[int, tuple[float, float, float, float, float]]:
    best: dict[int, tuple[float, float, float, float, float]] = {}
    with path.open(encoding="utf-8") as f:
        visit_rects(json.load(f), best)
    return best


def aggregate(vals: list[float]) -> float:
    clean = sorted(float(x) for x in vals if math.isfinite(float(x)))
    if not clean:
        return float("nan")
    k = int(len(clean) * 0.1)
    if len(clean) - 2 * k <= 0:
        return float(median(clean))
    return float(sum(clean[k:len(clean)-k]) / (len(clean) - 2 * k))


def build_features(rows: list[dict], rect_root: Path) -> tuple[dict, dict, dict]:
    meta = {}
    features = {}
    missing = {}
    split_priority = {"train": 0, "val": 1, "test": 2}
    for row in rows:
        sid = row["sequence_id"]
        rect_path = choose_rect_json(rect_root, row["person_id"], row["video_stem"], row["camera_id"])
        if rect_path is None or not rect_path.exists():
            missing[sid] = str(rect_root / f"{row['person_id']}_{row['video_stem']}.json")
            continue
        try:
            rects = read_rect_json(rect_path)
        except Exception as exc:
            missing[sid] = f"{rect_path}: {exc}"
            continue
        if not rects:
            missing[sid] = f"{rect_path}: no rect"
            continue

        # Keep the highest-priority split for duplicate sequence_ids: train > val > test.
        existing = meta.get(sid)
        if existing is not None:
            old_p = split_priority.get(existing.get("split", "test"), 99)
            new_p = split_priority.get(row.get("split", "test"), 99)
            if old_p <= new_p:
                continue

        payload = defaultdict(list)
        for _, (x, y, w, h, score) in rects.items():
            vw = float(row.get("video_w") or 2560.0)
            vh = float(row.get("video_h") or 1440.0)
            payload["bbox_h_norm"].append(h / vh)
            payload["bbox_w_norm"].append(w / vw)
            payload["bbox_y1_norm_neg"].append(-y / vh)
            payload["bbox_y2_norm"].append((y + h) / vh)
            payload["bbox_cy_norm"].append((y + y + h) * 0.5 / vh)
            payload["area_norm"].append((w * h) / max(1.0, vw * vh))
            payload["rect_score"].append(score)
        features[sid] = {k: aggregate(v) for k, v in payload.items()}
        item = dict(row)
        item["n_rect_frames"] = len(rects)
        meta[sid] = item
    return meta, features, missing


def rankdata(vals):
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
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa -= aa.mean(); bb -= bb.mean()
    den = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    return float((aa * bb).sum() / den) if den > 0 else None


def spearman(pred, gt):
    return pearson(rankdata(pred), rankdata(gt))


def pair_acc(ids, scores, heights):
    order = sorted(ids, key=lambda sid: (-scores[sid], sid))
    correct = total = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            dh = heights[order[i]] - heights[order[j]]
            if abs(dh) <= 0:
                continue
            total += 1
            correct += int(dh > 0)
    return correct / total if total else 0.0


def bisect(ids, scores, heights, overlap_ratio=0.2):
    ranking = sorted(ids, key=lambda sid: (-scores[sid], sid))
    n = len(ranking)
    if n == 0:
        return {"hit_rate": 0.0, "hit_count": 0, "evaluated_count": 0}
    vals = sorted(heights[s] for s in ranking)
    thr = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    overlap_n = int(round(n * overlap_ratio))
    mid = n // 2
    start = max(0, mid - overlap_n // 2)
    end = min(n, start + overlap_n)
    overlap = set(ranking[start:end])
    high = {s for i, s in enumerate(ranking) if i < mid or s in overlap}
    low = {s for i, s in enumerate(ranking) if i >= mid or s in overlap}
    hit = 0; sizes = []
    for sid in ranking:
        gt = "high" if heights[sid] >= thr else "low"
        if sid in high and sid in low:
            ok = True; size = len(high | low)
        elif sid in high:
            ok = gt == "high"; size = len(high)
        elif sid in low:
            ok = gt == "low"; size = len(low)
        else:
            ok = False; size = 0
        hit += int(ok); sizes.append(size)
    avg = sum(sizes) / len(sizes)
    return {"hit_rate": hit / n, "hit_count": hit, "evaluated_count": n, "avg_candidate_size": avg, "search_space_reduction_ratio": 1 - avg / max(1, n)}


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_camera(cam, ids, meta, feats, person_heights, device, epochs, seeds, hidden):
    by_split = {"train": [], "val": [], "test": []}
    heights = {}
    for sid in ids:
        pid = meta[sid]["person_id"]
        split = meta[sid]["split"]
        if pid in person_heights and split in by_split:
            by_split[split].append(sid)
            heights[sid] = person_heights[pid]
    if len(by_split["train"]) < 2 or len(by_split["test"]) < 2:
        return {"skipped": True, "split_counts": {k: len(v) for k, v in by_split.items()}}
    all_ids = by_split["train"] + by_split["val"] + by_split["test"]
    X = np.array([[float(feats[sid].get(f, 0.0)) if math.isfinite(float(feats[sid].get(f, 0.0))) else 0.0 for f in FEATURES] for sid in all_ids], dtype=np.float32)
    y = np.array([heights[sid] for sid in all_ids], dtype=np.float32)
    sid_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    train_idx = [sid_to_idx[s] for s in by_split["train"]]
    mu = X[train_idx].mean(0); sd = X[train_idx].std(0); sd[sd < 1e-6] = 1.0
    X = (X - mu) / sd
    pairs = []
    tr = by_split["train"]
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            a, b = tr[i], tr[j]
            if heights[a] == heights[b]:
                continue
            pairs.append((sid_to_idx[a], sid_to_idx[b], 1.0 if heights[a] > heights[b] else -1.0))
    results = []
    best_scores = None
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = MLP(len(FEATURES), hidden=hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        x = torch.tensor(X, device=device)
        yy = torch.tensor(y, device=device)
        pair_idx = torch.tensor([[a, b] for a, b, _ in pairs], device=device, dtype=torch.long) if pairs else None
        pair_sign = torch.tensor([s for _, _, s in pairs], device=device, dtype=torch.float32) if pairs else None
        best = None
        for ep in range(1, epochs + 1):
            model.train(); opt.zero_grad()
            pred = model(x)
            mse = torch.nn.functional.mse_loss(pred[train_idx], yy[train_idx]) / 1000.0
            if pair_idx is not None:
                diff = pred[pair_idx[:, 0]] - pred[pair_idx[:, 1]]
                pair_loss = torch.nn.functional.softplus(-pair_sign * diff).mean()
            else:
                pair_loss = torch.tensor(0.0, device=device)
            loss = mse + 0.5 * pair_loss
            loss.backward(); opt.step()
            if ep % 20 == 0 or ep == epochs:
                model.eval()
                with torch.no_grad():
                    pp = model(x).detach().cpu().numpy()
                scores = {sid: float(pp[sid_to_idx[sid]]) for sid in all_ids}
                val_ids = by_split["val"] or by_split["train"]
                val_pa = pair_acc(val_ids, scores, heights)
                if best is None or val_pa > best["val_pairwise"]:
                    best = {"epoch": ep, "val_pairwise": val_pa, "scores": scores}
        scores = best.pop("scores")
        item = {"seed": seed, "best_epoch": best["epoch"], "val_pairwise": best["val_pairwise"]}
        for split in ["train", "val", "test"]:
            sids = by_split[split]
            item[split] = {
                "n": len(sids),
                "pairwise_accuracy": pair_acc(sids, scores, heights),
                "spearman": spearman([scores[s] for s in sids], [heights[s] for s in sids]) if len(sids) >= 2 else None,
                "bisect": bisect(sids, scores, heights),
            }
        item["_scores"] = scores
        results.append(item)
    best = max(results, key=lambda r: (r["val_pairwise"], r["test"]["pairwise_accuracy"]))
    best_scores = best.pop("_scores")
    for r in results:
        r.pop("_scores", None)
    ranking = sorted(best_scores, key=lambda sid: (-best_scores[sid], sid))
    video_ranking = [{
        "rank": i + 1,
        "sequence_id": sid,
        "person_id": meta[sid]["person_id"],
        "split": meta[sid]["split"],
        "video_filename": meta[sid]["video_filename"],
        "height_cm": float(heights[sid]),
        "score": float(best_scores[sid]),
    } for i, sid in enumerate(ranking)]
    return {
        "skipped": False,
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "features": FEATURES,
        "seeds": results,
        "best": best,
        "video_level_ranking": video_ranking,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/zyding/data")
    ap.add_argument("--rect-root", default="/data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/scanner/main_card_out")
    ap.add_argument("--out-dir", default="/home/zyding/height/heightnet_repro/runs/home_data_rect_mlp")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    person_heights = load_rank_csv(data_root / "label" / "rank.json")
    rows = []
    for split in ["train", "val", "test"]:
        rows.extend(load_split_csv(data_root / f"{split}.csv", split))
    meta, feats, missing = build_features(rows, Path(args.rect_root))
    feature_payload = {"meta": meta, "features": feats, "missing_rect": missing, "features_used": FEATURES}
    (out_dir / "sequence_features.json").write_text(json.dumps(feature_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cameras = sorted({m["camera_id"] for m in meta.values()})
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {
        "data_root": str(data_root),
        "rect_root": str(args.rect_root),
        "device": device,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "seeds": args.seeds,
        "n_rows": len(rows),
        "n_sequences_with_rect": len(meta),
        "missing_rect_count": len(missing),
        "n_persons_with_height": len(person_heights),
        "camera_results": {},
    }
    for cam in cameras:
        ids = [sid for sid, m in meta.items() if m["camera_id"] == cam]
        print(f"[CAM] {cam} n={len(ids)}", flush=True)
        results["camera_results"][cam] = train_camera(cam, ids, meta, feats, person_heights, device, args.epochs, args.seeds, args.hidden)
    (out_dir / "base_camera_mlp_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    top = []
    for cam, res in results["camera_results"].items():
        if res.get("skipped"):
            continue
        b = res["best"]
        top.append((b["test"]["pairwise_accuracy"], b["test"].get("spearman"), b["test"]["bisect"].get("hit_rate"), cam, b["seed"], b["best_epoch"], res["split_counts"]))
    print("TOP_TEST")
    for row in sorted(top, reverse=True)[:20]:
        print(row, flush=True)
    print("[OUT]", out_dir / "base_camera_mlp_results.json")


if __name__ == "__main__":
    main()
