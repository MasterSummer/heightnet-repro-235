from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

TRAIN_PEOPLE = [
    "1126_man1", "1126_woman1", "1127_man1", "1127_man2", "1127_man3", "1128_man2",
    "1128_woman1", "1201_man1", "1202_man1", "1202_woman2", "1202_woman3", "1203_man1",
    "1203_woman1", "1204_man1", "1204_woman1", "1204_woman2", "1204_woman3", "1205_woman1",
    "1205_woman2", "1209_woman1", "1209_woman2", "1209_woman3",
]
VAL_PEOPLE = ["1127_woman1", "1201_woman2", "1201_woman3"]
TEST_PEOPLE = ["1128_man1", "1201_woman1", "1202_woman1", "1203_woman2", "1205_man1", "1209_man1"]
SPLIT = {p:"train" for p in TRAIN_PEOPLE}
SPLIT.update({p:"val" for p in VAL_PEOPLE})
SPLIT.update({p:"test" for p in TEST_PEOPLE})

FEATURES = ["bbox_h_norm", "bbox_w_norm", "bbox_y1_norm_neg", "bbox_y2_norm", "bbox_cy_norm", "area_norm", "rect_score"]


def load_heights_csv(path):
    import csv
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pid = r.get('penson_id') or r.get('person_id')
            if pid: out[pid] = float(r['height(cm)'])
    return out


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


def pearson(a,b):
    if len(a) < 2: return None
    aa=np.asarray(a,dtype=np.float64); bb=np.asarray(b,dtype=np.float64)
    aa-=aa.mean(); bb-=bb.mean(); den=float(np.sqrt((aa*aa).sum()*(bb*bb).sum()))
    return float((aa*bb).sum()/den) if den>0 else None


def spearman(pred, gt): return pearson(rankdata(pred), rankdata(gt))


def pair_acc(ids, scores, heights):
    order = sorted(ids, key=lambda sid: (-scores[sid], sid))
    correct = total = 0
    for i in range(len(order)):
        for j in range(i+1, len(order)):
            dh = heights[order[i]] - heights[order[j]]
            if abs(dh) <= 0: continue
            total += 1; correct += int(dh > 0)
    return correct / total if total else 0.0


def bisect(ids, scores, heights, overlap_ratio=0.2):
    ranking = sorted(ids, key=lambda sid: (-scores[sid], sid))
    n=len(ranking)
    if n == 0: return {"hit_rate": 0, "hit_count": 0, "evaluated_count": 0}
    vals=sorted(heights[s] for s in ranking)
    thr=vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2
    overlap_n=int(round(n*overlap_ratio)); mid=n//2
    start=max(0, mid-overlap_n//2); end=min(n, start+overlap_n)
    ov=set(ranking[start:end])
    high={s for i,s in enumerate(ranking) if i < mid or s in ov}
    low={s for i,s in enumerate(ranking) if i >= mid or s in ov}
    hit=0; sizes=[]
    for s in ranking:
        gt='high' if heights[s] >= thr else 'low'
        if s in high and s in low: ok=True; size=len(high|low)
        elif s in high: ok=(gt=='high'); size=len(high)
        elif s in low: ok=(gt=='low'); size=len(low)
        else: ok=False; size=0
        hit += int(ok); sizes.append(size)
    avg=sum(sizes)/len(sizes)
    return {"hit_rate": hit/n, "hit_count": hit, "evaluated_count": n, "avg_candidate_size": avg, "search_space_reduction_ratio": 1-avg/max(1,n)}


class MLP(nn.Module):
    def __init__(self, d, hidden=128, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,hidden//2), nn.ReLU(), nn.Linear(hidden//2,1))
    def forward(self,x): return self.net(x).squeeze(-1)


def train_one(cam, ids, meta, feats, person_heights, device, epochs, seeds):
    by_split={k:[] for k in ['train','val','test']}
    heights={}
    for sid in ids:
        pid=meta[sid]['person_id']
        sp=SPLIT.get(pid)
        if sp and pid in person_heights:
            by_split[sp].append(sid); heights[sid]=person_heights[pid]
    if min(len(by_split['train']), len(by_split['val']), len(by_split['test'])) < 2:
        return {"skipped": True, "split_counts": {k:len(v) for k,v in by_split.items()}}
    X=np.array([[float(feats[sid].get(f,0.0)) if math.isfinite(float(feats[sid].get(f,0.0))) else 0.0 for f in FEATURES] for sid in ids], dtype=np.float32)
    sid_to_idx={sid:i for i,sid in enumerate(ids)}
    train_idx=[sid_to_idx[s] for s in by_split['train']]
    mu=X[train_idx].mean(0); sd=X[train_idx].std(0); sd[sd<1e-6]=1.0
    X=(X-mu)/sd
    y=np.array([heights.get(sid,0.0) for sid in ids], dtype=np.float32)
    # Pair labels train only
    pairs=[]
    tr=by_split['train']
    for i in range(len(tr)):
        for j in range(i+1,len(tr)):
            a,b=tr[i],tr[j]
            if heights[a] == heights[b]: continue
            pairs.append((sid_to_idx[a], sid_to_idx[b], 1.0 if heights[a] > heights[b] else -1.0))
    results=[]
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model=MLP(len(FEATURES)).to(device)
        opt=torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        x=torch.tensor(X, device=device); yy=torch.tensor(y, device=device)
        pair_t=torch.tensor(pairs, device=device, dtype=torch.long) if pairs else None
        best=None
        for ep in range(1, epochs+1):
            model.train(); opt.zero_grad(); pred=model(x)
            mse=torch.nn.functional.mse_loss(pred[train_idx], yy[train_idx]) / 1000.0
            if pair_t is not None and len(pair_t)>0:
                ia=pair_t[:,0]; ib=pair_t[:,1]; sign=pair_t[:,2].float()
                pair_loss=torch.nn.functional.softplus(-sign*(pred[ia]-pred[ib])).mean()
            else:
                pair_loss=torch.tensor(0.0,device=device)
            loss=mse+0.5*pair_loss; loss.backward(); opt.step()
            if ep % 20 == 0 or ep == epochs:
                model.eval()
                with torch.no_grad(): pp=model(x).detach().cpu().numpy()
                score={sid:float(pp[sid_to_idx[sid]]) for sid in ids}
                val_pa=pair_acc(by_split['val'], score, heights)
                if best is None or val_pa > best['val_pairwise']:
                    best={"epoch": ep, "val_pairwise": val_pa, "score": score}
        score=best.pop('score')
        item={"seed": seed, "best_epoch": best['epoch'], "val_pairwise": best['val_pairwise']}
        for split in ['train','val','test']:
            sids=by_split[split]
            item[split]={"n": len(sids), "pairwise_accuracy": pair_acc(sids, score, heights), "spearman": spearman([score[s] for s in sids], [heights[s] for s in sids]), "bisect": bisect(sids, score, heights)}
        results.append(item)
    best=max(results, key=lambda r: (r['val_pairwise'], r['test']['pairwise_accuracy']))
    return {"skipped": False, "split_counts": {k:len(v) for k,v in by_split.items()}, "features": FEATURES, "seeds": results, "best": best}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--feature-json', default='/data1/zyding/jianzhi_2511_all_camera_rect_sequence/sequence_features.json')
    ap.add_argument('--rank-csv', default='/home/zyding/height/jianzhi_15_meta/rank.json')
    ap.add_argument('--out', default='/data1/zyding/jianzhi_2511_all_camera_rect_sequence/all_camera_mlp_results.json')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1,2,3])
    args=ap.parse_args()
    payload=json.load(open(args.feature_json))
    meta=payload['meta']; feats=payload['features']; heights=load_heights_csv(args.rank_csv)
    cams=sorted(set(m['camera_id'] for m in meta.values()))
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    out={"device": device, "epochs": args.epochs, "features": FEATURES, "camera_results": {}}
    for cam in cams:
        ids=[sid for sid,m in meta.items() if m['camera_id']==cam]
        print('[CAM]', cam, 'n=', len(ids), flush=True)
        out['camera_results'][cam]=train_one(cam, ids, meta, feats, heights, device, args.epochs, args.seeds)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    top=[]
    for cam,res in out['camera_results'].items():
        if res.get('skipped'): continue
        b=res['best']; top.append((b['test']['pairwise_accuracy'], b['test']['spearman'], b['test']['bisect']['hit_rate'], cam, b['seed'], b['best_epoch'], res['split_counts']))
    print('TOP_TEST')
    for row in sorted(top, reverse=True)[:20]: print(row, flush=True)
    print('[OUT]', args.out)

if __name__ == '__main__': main()
