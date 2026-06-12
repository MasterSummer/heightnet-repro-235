from __future__ import annotations
import json, csv, math, random, re
from pathlib import Path
from collections import defaultdict
from statistics import median
import numpy as np
import torch
from torch import nn

CAM_RE = re.compile(r"_(?P<h>\d+d\d+)_(?P<a>\d+)_(?P<p>\d+w)_", re.I)
FEATURES = ["bbox_h_norm", "bbox_w_norm", "bbox_y1_norm_neg", "bbox_y2_norm", "bbox_cy_norm", "area_norm", "rect_score"]

def base_camera_from_name(name: str):
    m = CAM_RE.search(name)
    return f"{m.group('h').lower()}_{m.group('a')}" if m else None

def frame_size_from_name(name: str):
    m = CAM_RE.search(name); pixel = m.group('p').lower() if m else ''
    if pixel == '200w': return 2304.0, 1296.0
    if pixel == '400w': return 2560.0, 1440.0
    if pixel == '800w': return 1280.0, 720.0
    return 2560.0, 1440.0

def load_rank_csv(path: Path):
    out = {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pid = row.get('penson_id') or row.get('person_id')
            h = row.get('height(cm)') or row.get('height_cm') or row.get('height')
            if pid and h: out[str(pid)] = float(h)
    return out

def visit_rects(obj, best):
    if isinstance(obj, dict):
        if 'frame_id' in obj and 'rect' in obj:
            try:
                fid = int(obj['frame_id']); x,y,w,h = map(float, obj['rect'][:4]); score = float(obj.get('score', 0.0))
                if w > 0 and h > 0 and (fid not in best or score > best[fid][4]):
                    best[fid] = (x,y,w,h,score)
            except Exception:
                pass
        for v in obj.values(): visit_rects(v, best)
    elif isinstance(obj, list):
        for v in obj: visit_rects(v, best)

def read_rect_json(path: Path):
    best = {}
    with path.open(encoding='utf-8') as f: visit_rects(json.load(f), best)
    return best

def aggregate(vals):
    clean = sorted(float(x) for x in vals if math.isfinite(float(x)))
    if not clean: return float('nan')
    k = int(len(clean) * 0.1)
    if len(clean) - 2*k <= 0: return float(median(clean))
    return float(sum(clean[k:len(clean)-k]) / (len(clean)-2*k))

def choose_rect_json(rect_root: Path, person_id: str, video_stem: str, camera_id: str):
    exact = rect_root / f"{person_id}_{video_stem}.json"
    if exact.exists(): return exact
    m = CAM_RE.search(video_stem); pixel = m.group('p').lower() if m else ''
    if camera_id and pixel:
        matches = sorted(rect_root.glob(f"{person_id}_*{camera_id}_{pixel}_*jianzhi2511.json"))
        if matches: return matches[0]
    if camera_id:
        matches = sorted(rect_root.glob(f"{person_id}_*{camera_id}_*.json"))
        if matches: return matches[0]
    return None

def feature_from_rects(rects, video_filename):
    vw, vh = frame_size_from_name(video_filename)
    payload = defaultdict(list)
    for _, (x,y,w,h,score) in rects.items():
        payload['bbox_h_norm'].append(h / vh)
        payload['bbox_w_norm'].append(w / vw)
        payload['bbox_y1_norm_neg'].append(-y / vh)
        payload['bbox_y2_norm'].append((y+h) / vh)
        payload['bbox_cy_norm'].append((y+y+h)*0.5 / vh)
        payload['area_norm'].append((w*h) / max(1.0, vw*vh))
        payload['rect_score'].append(score)
    return {k: aggregate(v) for k,v in payload.items()}

class MLP(nn.Module):
    def __init__(self, d, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_model(cam, meta, feats, heights, best_seed, best_epoch, hidden=128, device='cuda:0'):
    ids = [sid for sid,m in meta.items() if m['camera_id'] == cam]
    by_split = {'train': [], 'val': [], 'test': []}
    usable_heights = {}
    for sid in ids:
        pid = meta[sid]['person_id']; sp = meta[sid]['split']
        if pid in heights and sp in by_split:
            by_split[sp].append(sid); usable_heights[sid] = heights[pid]
    all_ids = by_split['train'] + by_split['val'] + by_split['test']
    X = np.array([[float(feats[sid].get(f,0.0)) if math.isfinite(float(feats[sid].get(f,0.0))) else 0.0 for f in FEATURES] for sid in all_ids], dtype=np.float32)
    y = np.array([usable_heights[sid] for sid in all_ids], dtype=np.float32)
    sid_to_idx = {sid:i for i,sid in enumerate(all_ids)}
    train_idx = [sid_to_idx[s] for s in by_split['train']]
    mu = X[train_idx].mean(0); sd = X[train_idx].std(0); sd[sd < 1e-6] = 1.0
    Xn = (X - mu) / sd
    pairs = []
    tr = by_split['train']
    for i in range(len(tr)):
        for j in range(i+1, len(tr)):
            a,b = tr[i], tr[j]
            if usable_heights[a] == usable_heights[b]: continue
            pairs.append((sid_to_idx[a], sid_to_idx[b], 1.0 if usable_heights[a] > usable_heights[b] else -1.0))
    random.seed(best_seed); np.random.seed(best_seed); torch.manual_seed(best_seed); torch.cuda.manual_seed_all(best_seed)
    model = MLP(len(FEATURES), hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    x = torch.tensor(Xn, device=device); yy = torch.tensor(y, device=device)
    pair_idx = torch.tensor([[a,b] for a,b,_ in pairs], device=device, dtype=torch.long) if pairs else None
    pair_sign = torch.tensor([s for _,_,s in pairs], device=device, dtype=torch.float32) if pairs else None
    for _ in range(1, int(best_epoch)+1):
        model.train(); opt.zero_grad(); pred = model(x)
        mse = torch.nn.functional.mse_loss(pred[train_idx], yy[train_idx]) / 1000.0
        if pair_idx is not None:
            diff = pred[pair_idx[:,0]] - pred[pair_idx[:,1]]
            pair_loss = torch.nn.functional.softplus(-pair_sign * diff).mean()
        else:
            pair_loss = torch.tensor(0.0, device=device)
        loss = mse + 0.5 * pair_loss; loss.backward(); opt.step()
    model.eval()
    return model, mu, sd

def discover_softlink_records(root: Path, rect_root: Path):
    meta, feats, missing = {}, {}, {}
    for video in sorted(root.glob('*/*/*.mp4')):
        person = video.parts[-3]; group = video.parts[-2]; stem = video.stem; cam = base_camera_from_name(video.name)
        if not cam: continue
        sid = f"{person}__{group}__{stem}"
        rect_path = choose_rect_json(rect_root, person, stem, cam)
        if rect_path is None or not rect_path.exists():
            missing[sid] = str(rect_root / f"{person}_{stem}.json"); continue
        try: rects = read_rect_json(rect_path)
        except Exception as e: missing[sid] = f"{rect_path}: {e}"; continue
        if not rects: missing[sid] = f"{rect_path}: no rect"; continue
        feats[sid] = feature_from_rects(rects, video.name)
        meta[sid] = {'sequence_id': sid, 'person_id': person, 'group': group, 'video_filename': video.name, 'video_path': str(video), 'camera_id': cam, 'n_rect_frames': len(rects)}
    return meta, feats, missing

def main():
    result_path = Path('/home/zyding/height/heightnet_repro/runs/home_data_rect_mlp_20260521_fix2/base_camera_mlp_results.json')
    feature_path = Path('/home/zyding/height/heightnet_repro/runs/home_data_rect_mlp_20260521_fix2/sequence_features.json')
    soft_root = Path('/data2/dataset/jianzhi_2511/jianzhi_2511_spilt_coat_softlink')
    rect_root = Path('/data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/scanner/main_card_out')
    out_path = Path('/home/zyding/height/heightnet_repro/runs/home_data_rect_mlp_20260521_fix2/softlink_full_camera_rankings.json')
    heights = load_rank_csv(Path('/home/zyding/data/label/rank.json'))
    train_payload = json.load(feature_path.open()); train_meta = train_payload['meta']; train_feats = train_payload['features']
    results = json.load(result_path.open())
    soft_meta, soft_feats, missing = discover_softlink_records(soft_root, rect_root)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output = {'softlink_root': str(soft_root), 'n_softlink_videos_with_features': len(soft_meta), 'missing_rect_count': len(missing), 'cameras': {}}
    for cam in sorted(results['camera_results']):
        res = results['camera_results'][cam]
        if res.get('skipped'): continue
        best = res['best']; model, mu, sd = train_model(cam, train_meta, train_feats, heights, best['seed'], best['best_epoch'], results.get('hidden',128), device)
        ids = [sid for sid,m in soft_meta.items() if m['camera_id'] == cam]
        if not ids: continue
        X = np.array([[float(soft_feats[sid].get(f,0.0)) if math.isfinite(float(soft_feats[sid].get(f,0.0))) else 0.0 for f in FEATURES] for sid in ids], dtype=np.float32)
        X = (X - mu) / sd
        with torch.no_grad(): scores = model(torch.tensor(X, device=device)).detach().cpu().numpy()
        ranking = []
        for sid, score in sorted(zip(ids, scores), key=lambda x: (-float(x[1]), x[0])):
            m = soft_meta[sid]; pid = m['person_id']
            ranking.append({'rank': len(ranking)+1, 'sequence_id': sid, 'person_id': pid, 'group': m['group'], 'video_filename': m['video_filename'], 'video_path': m['video_path'], 'height_cm': heights.get(pid), 'has_height_label': pid in heights, 'score': float(score), 'n_rect_frames': m['n_rect_frames']})
        output['cameras'][cam] = {'best_seed': best['seed'], 'best_epoch': best['best_epoch'], 'training_test_metrics_labeled': best['test'], 'n_ranked': len(ranking), 'n_with_height_label': sum(1 for r in ranking if r['has_height_label']), 'ranking': ranking}
        print('[CAM]', cam, 'ranked', len(ranking), 'label', output['cameras'][cam]['n_with_height_label'], flush=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[OUT]', out_path)

if __name__ == '__main__': main()
