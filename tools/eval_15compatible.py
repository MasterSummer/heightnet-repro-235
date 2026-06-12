from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np


def rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals); i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j+1]] == vals[order[i]]: j += 1
        avg = (i + j + 2.0) / 2.0
        for k in range(i, j+1): ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    if len(a) < 2: return None
    aa = np.asarray(a, dtype=np.float64); bb = np.asarray(b, dtype=np.float64)
    aa -= aa.mean(); bb -= bb.mean()
    den = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    return float((aa * bb).sum() / den) if den > 0 else None


def spearman(pred, gt):
    return pearson(rankdata(pred), rankdata(gt))


def pairwise_accuracy(pred_ranking, gt_ranking, heights):
    """Compute pairwise accuracy: for all i<j in GT ranking, is pred order correct?"""
    pred_pos = {pid: i for i, pid in enumerate(pred_ranking)}
    common = [pid for pid in gt_ranking if pid in pred_pos]
    correct = total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            dh = heights.get(common[i], 0) - heights.get(common[j], 0)
            if abs(dh) <= 1e-9: continue
            total += 1
            if dh > 0 and pred_pos[common[i]] < pred_pos[common[j]]:
                correct += 1
            elif dh < 0 and pred_pos[common[i]] > pred_pos[common[j]]:
                correct += 1
    return correct / total if total > 0 else 0.0, correct, total


def split_rank_bands(ranking, n_bands=3):
    """Split ranking into upper/middle/lower bands (quartile-based, matching 15-server)."""
    n = len(ranking)
    if n < 3:
        return {"upper": ranking, "middle": [], "lower": []}
    q = max(1, n // 4)
    upper = set(ranking[:q])
    lower = set(ranking[-q:])
    middle = set(ranking[q:-q]) if n > 2 * q else set()
    return {"upper": upper, "middle": middle, "lower": lower}


def band_accuracy(pred_ranking, gt_ranking, heights):
    """Compute band accuracy: for each person, is predicted band same as GT band?"""
    gt_bands = split_rank_bands(gt_ranking)
    pred_pos = {pid: i for i, pid in enumerate(pred_ranking)}
    pred_bands = split_rank_bands(pred_ranking)

    common = [pid for pid in gt_ranking if pid in pred_pos]
    if not common:
        return 0.0, 0, 0

    correct = total = 0
    for pid in common:
        gt_band = None
        if pid in gt_bands["upper"]: gt_band = "upper"
        elif pid in gt_bands["lower"]: gt_band = "lower"
        else: gt_band = "middle"

        pred_band = None
        if pid in pred_bands["upper"]: pred_band = "upper"
        elif pid in pred_bands["lower"]: pred_band = "lower"
        else: pred_band = "middle"

        total += 1
        if gt_band == pred_band:
            correct += 1

    return correct / total if total > 0 else 0.0, correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ranking', default='/data1/zyding/jianzhi_2511_clothing_split_mlp/sequence_ranking_per_camera.json')
    ap.add_argument('--rank-by-camera-dir', default='/data1/zyding/jianzhi_2511_clothing_split_mlp/15_server_reference/rank_by_camera')
    ap.add_argument('--all-pairs', default='/data1/zyding/jianzhi_2511_clothing_split_mlp/15_server_reference/all_pairs.json')
    ap.add_argument('--rank-csv', default='/data1/zyding/jianzhi_2511_clothing_split_mlp/15_server_reference/rank.json')
    ap.add_argument('--out', default='/data1/zyding/jianzhi_2511_clothing_split_mlp/metrics_15compatible.json')
    args = ap.parse_args()

    # Load heights
    import csv
    heights = {}
    with open(args.rank_csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pid = r.get('penson_id') or r.get('person_id')
            if pid: heights[pid] = float(r['height(cm)'])

    # Load sequence ranking
    ranking_data = json.load(open(args.ranking))

    # Load GT rank_by_camera
    gt_rankings = {}
    for fname in sorted(os.listdir(args.rank_by_camera_dir)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(args.rank_by_camera_dir, fname)
        d = json.load(open(fpath))
        cam = d['camera']
        gt_rankings[cam] = d['ranking']  # list of person_ids sorted by height desc

    # Load all_pairs (optional, for reference)
    all_pairs = json.load(open(args.all_pairs)) if os.path.exists(args.all_pairs) else []

    # 8 persons from 15 not in 235 features - exclude from evaluation
    excluded_persons = {"1124_man1", "1124_woman1", "1124_woman2", "1124_woman3",
                        "1125_man1", "1125_man2", "1126_man2", "1126_man3"}

    cam_metrics = {}
    for cam in sorted(ranking_data['cameras'].keys()):
        cam_ranking = ranking_data['cameras'][cam]['ranking']

        # Aggregate sequence scores to person scores (mean)
        person_scores = {}
        for item in cam_ranking:
            pid = item['person_id']
            score = item['score']
            person_scores.setdefault(pid, []).append(score)

        person_avg = {pid: sum(scores) / len(scores) for pid, scores in person_scores.items()}

        # Build predicted ranking (sorted by avg score desc)
        pred_ranking = sorted(person_avg.keys(), key=lambda p: (-person_avg[p], p))

        # Get GT ranking, filter out excluded persons
        gt_full = gt_rankings.get(cam, [])
        gt_ranking = [pid for pid in gt_full if pid not in excluded_persons and pid in heights]

        # Filter pred to only include persons in GT
        gt_set = set(gt_ranking)
        pred_filtered = [pid for pid in pred_ranking if pid in gt_set]

        if len(gt_ranking) < 2:
            cam_metrics[cam] = {"skipped": True, "reason": "insufficient GT persons"}
            continue

        # Compute metrics
        pa, pa_correct, pa_total = pairwise_accuracy(pred_filtered, gt_ranking, heights)
        ba, ba_correct, ba_total = band_accuracy(pred_filtered, gt_ranking, heights)

        pred_scores_for_spearman = [person_avg.get(pid, 0) for pid in gt_ranking]
        gt_heights_for_spearman = [heights.get(pid, 0) for pid in gt_ranking]
        sp = spearman(pred_scores_for_spearman, gt_heights_for_spearman)

        cam_metrics[cam] = {
            "pairwise_accuracy": pa,
            "pairwise_correct": pa_correct,
            "pairwise_total": pa_total,
            "band_accuracy": ba,
            "band_correct": ba_correct,
            "band_total": ba_total,
            "spearman_rho": sp,
            "n_persons_gt": len(gt_ranking),
            "n_persons_pred": len(pred_filtered),
            "n_persons_excluded": len(gt_full) - len(gt_ranking),
        }

    # Summary (macro average across cameras)
    valid_cams = {k: v for k, v in cam_metrics.items() if not v.get('skipped')}
    if valid_cams:
        summary = {
            "pairwise_accuracy_mean": sum(v['pairwise_accuracy'] for v in valid_cams.values()) / len(valid_cams),
            "band_accuracy_mean": sum(v['band_accuracy'] for v in valid_cams.values()) / len(valid_cams),
            "spearman_mean": sum(v['spearman_rho'] for v in valid_cams.values() if v['spearman_rho'] is not None) / max(1, sum(1 for v in valid_cams.values() if v['spearman_rho'] is not None)),
            "n_cameras": len(valid_cams),
            "pairwise_correct_total": sum(v['pairwise_correct'] for v in valid_cams.values()),
            "pairwise_total_total": sum(v['pairwise_total'] for v in valid_cams.values()),
        }
    else:
        summary = {"n_cameras": 0}

    output = {"summary": summary, "cameras": cam_metrics}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"[OUT] {args.out}")
    print(f"Valid cameras: {len(valid_cams)}/{len(cam_metrics)}")
    if valid_cams:
        print(f"Pairwise accuracy (mean): {summary['pairwise_accuracy_mean']:.4f}")
        print(f"Band accuracy (mean): {summary['band_accuracy_mean']:.4f}")
        print(f"Spearman rho (mean): {summary['spearman_mean']:.4f}")


if __name__ == '__main__':
    main()
