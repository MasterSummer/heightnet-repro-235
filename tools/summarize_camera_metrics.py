import json
from pathlib import Path

run = Path('/home/zyding/height/heightnet_repro/runs/home_base_camera_rect_sequence')
inp = run / 'base_camera_mlp_results.json'
out_json = run / 'camera_metrics_summary.json'
out_md = run / 'camera_metrics_summary.md'
r = json.load(open(inp, encoding='utf-8'))
summary = {
    'source': str(inp),
    'device': r.get('device'),
    'epochs': r.get('epochs'),
    'features': r.get('features'),
    'n_cameras': len(r.get('camera_results', {})),
    'cameras': {},
}
rows = []
for cam, res in sorted(r.get('camera_results', {}).items()):
    if res.get('skipped'):
        item = {'skipped': True, 'split_counts': res.get('split_counts', {})}
    else:
        best = res.get('best', {})
        test = best.get('test', {})
        val = best.get('val', {})
        train = best.get('train', {})
        bis = test.get('bisect', {})
        item = {
            'skipped': False,
            'split_counts': res.get('split_counts', {}),
            'best_seed': best.get('seed'),
            'best_epoch': best.get('best_epoch'),
            'val_pairwise_for_selection': best.get('val_pairwise'),
            'train': {
                'pairwise_accuracy': train.get('pairwise_accuracy'),
                'spearman': train.get('spearman'),
                'bisect_hit_rate': train.get('bisect', {}).get('hit_rate'),
            },
            'val': {
                'pairwise_accuracy': val.get('pairwise_accuracy'),
                'spearman': val.get('spearman'),
                'bisect_hit_rate': val.get('bisect', {}).get('hit_rate'),
            },
            'test': {
                'n': test.get('n'),
                'pairwise_accuracy': test.get('pairwise_accuracy'),
                'spearman': test.get('spearman'),
                'bisect': {
                    'hit_rate': bis.get('hit_rate'),
                    'hit_count': bis.get('hit_count'),
                    'evaluated_count': bis.get('evaluated_count'),
                    'avg_candidate_size': bis.get('avg_candidate_size'),
                    'search_space_reduction_ratio': bis.get('search_space_reduction_ratio'),
                },
            },
            'video_level_ranking_count': len(res.get('video_level_ranking', [])),
        }
        rows.append((cam, item))
    summary['cameras'][cam] = item

rows_sorted = sorted(
    rows,
    key=lambda kv: (
        kv[1]['test']['bisect'].get('hit_rate') or 0,
        kv[1]['test'].get('pairwise_accuracy') or 0,
    ),
    reverse=True,
)
summary['cameras_sorted_by_test_bisect_hit_rate'] = [cam for cam, _ in rows_sorted]
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

lines = []
lines.append('# Camera Metrics Summary')
lines.append('')
lines.append(f'Source: `{inp}`')
lines.append(f'Device: `{r.get("device")}`')
lines.append(f'Epochs: `{r.get("epochs")}`')
lines.append(f'Features: `{", ".join(r.get("features", []))}`')
lines.append('')
lines.append('## Sorted By Test Bisect Hit Rate')
lines.append('')
lines.append('| Camera | Test bisect.hit_rate | Test pairacc | Test Spearman | Search reduction | Test n | Best seed | Best epoch | Split train/val/test |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for cam, item in rows_sorted:
    t = item['test']
    b = t['bisect']
    sc = item['split_counts']
    lines.append(
        '| {cam} | {hit:.4f} | {pa:.4f} | {sp:.4f} | {red:.4f} | {n} | {seed} | {ep} | {tr}/{va}/{te} |'.format(
            cam=cam,
            hit=b.get('hit_rate') or 0,
            pa=t.get('pairwise_accuracy') or 0,
            sp=t.get('spearman') or 0,
            red=b.get('search_space_reduction_ratio') or 0,
            n=t.get('n'),
            seed=item.get('best_seed'),
            ep=item.get('best_epoch'),
            tr=sc.get('train'),
            va=sc.get('val'),
            te=sc.get('test'),
        )
    )
lines.append('')
lines.append('## Full Metrics')
lines.append('')
for cam, item in rows_sorted:
    t = item['test']
    v = item['val']
    tr = item['train']
    b = t['bisect']
    lines.append(f'### {cam}')
    lines.append('')
    lines.append(f'- Best seed/epoch: `{item.get("best_seed")}` / `{item.get("best_epoch")}`')
    lines.append(f'- Split counts: `{item.get("split_counts")}`')
    lines.append(f'- Train: pairacc `{tr.get("pairwise_accuracy"):.4f}`, spearman `{tr.get("spearman"):.4f}`, bisect.hit_rate `{tr.get("bisect_hit_rate"):.4f}`')
    lines.append(f'- Val: pairacc `{v.get("pairwise_accuracy"):.4f}`, spearman `{v.get("spearman"):.4f}`, bisect.hit_rate `{v.get("bisect_hit_rate"):.4f}`')
    lines.append(f'- Test: pairacc `{t.get("pairwise_accuracy"):.4f}`, spearman `{t.get("spearman"):.4f}`, bisect.hit_rate `{b.get("hit_rate"):.4f}`')
    lines.append(f'- Test bisect: hit `{b.get("hit_count")}/{b.get("evaluated_count")}`, avg candidate size `{b.get("avg_candidate_size"):.4f}`, search reduction `{b.get("search_space_reduction_ratio"):.4f}`')
    lines.append(f'- Video ranking entries: `{item.get("video_level_ranking_count")}`')
    lines.append('')
out_md.write_text('\n'.join(lines), encoding='utf-8')
print(out_json)
print(out_md)
print('top_by_hit_rate')
for cam, item in rows_sorted[:10]:
    print(cam, item['test']['bisect'].get('hit_rate'), item['test'].get('pairwise_accuracy'), item['test'].get('spearman'))
