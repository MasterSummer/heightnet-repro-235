from __future__ import annotations
import argparse, json, os, glob
from pathlib import Path


def parse_clothing(seq_id):
    parts = seq_id.split("__")
    if len(parts) < 2:
        return None
    after = parts[1]
    if "_pants" in after:
        return after.split("_pants")[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='/data1/zyding/jianzhi_2511_clothing_split_mlp/base_camera_mlp_results.json')
    ap.add_argument('--features', default='/data1/zyding/jianzhi_2511_all_camera_rect_sequence/sequence_features.json')
    ap.add_argument('--split-json', default='/data1/zyding/jianzhi_2511_clothing_split_mlp/clothing_split.json')
    ap.add_argument('--rank-csv', default='/home/zyding/height/jianzhi_15_meta/rank.json')
    ap.add_argument('--out', default='/data1/zyding/jianzhi_2511_clothing_split_mlp/sequence_ranking_per_camera.json')
    args = ap.parse_args()

    import csv
    heights = {}
    with open(args.rank_csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pid = r.get('penson_id') or r.get('person_id')
            if pid: heights[pid] = float(r['height(cm)'])

    meta = json.load(open(args.features))['meta']
    split_data = json.load(open(args.split_json))
    seq_split = split_data['seq_split_map']

    results = json.load(open(args.results))

    output = {
        "description": "Sequence-level ranking per base camera from clothing-split MLP model",
        "model": "MLP (7 features), MSE + 0.5*Pairwise loss, AdamW lr=2e-3 wd=1e-3, 300 epochs",
        "split_method": split_data.get("split_method", ""),
        "split_people": {
            "train": sorted(split_data["train"]["persons"]),
            "val": sorted(split_data["val"]["persons"]),
            "test": sorted(split_data["test"]["persons"]),
        },
        "cameras": {}
    }

    for cam in sorted(results['camera_results'].keys()):
        r = results['camera_results'][cam]
        if r.get('skipped'):
            continue
        best = r['best']
        ranking_list = r.get('video_level_ranking', [])

        cam_info = {
            "n_sequences": {
                "train": best['train']['n'],
                "val": best['val']['n'],
                "test": best['test']['n']
            },
            "best_seed": best['seed'],
            "best_epoch": best['best_epoch'],
            "metrics": {
                "val_pairwise_accuracy": best['val']['pairwise_accuracy'],
                "test_pairwise_accuracy": best['test']['pairwise_accuracy'],
                "test_spearman": best['test']['spearman'],
                "test_bisect_hit_rate": best['test']['bisect']['hit_rate'],
                "test_bisect_search_space_reduction": best['test']['bisect']['search_space_reduction_ratio']
            },
            "ranking": []
        }

        for item in ranking_list:
            seq_id = item['sequence_id']
            cam_info['ranking'].append({
                'rank': item['rank'],
                'sequence_id': seq_id,
                'person_id': item['person_id'],
                'camera_id': item['pixel_camera_id'],
                'clothing': item.get('clothing', parse_clothing(seq_id)),
                'split': item.get('split', seq_split.get(seq_id, '')),
                'height_cm': item['height_cm'],
                'score': round(item['score'], 4),
                'video_path': meta.get(seq_id, {}).get('video_path', '')
            })

        output['cameras'][cam] = cam_info

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[OUT] {args.out}")
    print(f"Cameras: {sorted(output['cameras'].keys())}")
    for cam in sorted(output['cameras'].keys()):
        c = output['cameras'][cam]
        print(f"  {cam}: {len(c['ranking'])} sequences, test_PA={c['metrics']['test_pairwise_accuracy']:.3f}")


if __name__ == '__main__':
    main()
