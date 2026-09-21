# HeightNet Repro 235 Handoff

This repository is the active experiment code for video-level height ranking on server `235`.

## Ownership Model

- Edit code locally first.
- Sync code to server `235`.
- Run training/evaluation experiments on server `235`.
- Keep generated data, checkpoints, and HTML audits out of Git.

Server details:

```text
host: gait-server-235
user: zyding
remote repo: /home/zyding/height/heightnet_repro
default data root: /home/zyding/data
```

Local mirror used for Git handoff:

```text
/Users/yiding/code/gait/heightnet-repro-235
origin: https://github.com/MasterSummer/heightnet-repro-235.git
```

The broader local workspace also has `/Users/yiding/code/gait/height/heightnet_repro`. Treat that as a development checkout with drift from this 235 mirror.

## Metric Policy

Use video-level evaluation by default: each video is one independent sample.

Do not use person-camera aggregation or person-level aggregation as the primary metric. If an older aggregation-based result is reported for comparison, label it as non-primary.

Use person-disjoint train/validation/test splits when judging model quality.

## Environment Setup

On server `235`:

```bash
ssh gait-server-235
cd /home/zyding/height/heightnet_repro

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch CUDA build that matches the server driver if the generic `pip install -r requirements.txt` pulls a CPU build. Check with:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Depth-Anything-V2 is expected outside this repo:

```text
/home/zyding/height/Depth-Anything-V2
/home/zyding/height/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
```

YOLO weights may be referenced as `yolo26n.pt` or another local `.pt` file. Put weights in a stable server path and pass `--model` explicitly.

## Local To Server Sync

From the local workspace root:

```bash
/Users/yiding/code/gait/scripts/sync_heightnet_to_235.sh --dry-run
/Users/yiding/code/gait/scripts/sync_heightnet_to_235.sh
```

The sync script excludes generated outputs such as `runs/`, `remote_runs/`, `data/`, caches, `.npz`, `.npy`, checkpoints, and logs. It does not delete server-only files.

After syncing:

```bash
ssh gait-server-235
cd /home/zyding/height/heightnet_repro
git status --short
```

## Current Recommended Pipeline

The current handoff flow is for yolo26 bbox generation, NPZ conversion with exact bbox matching/filtering, crop quality audit, and cross-camera heightmap fusion training.

### 1. Generate YOLO bbox JSONs

```bash
cd /home/zyding/height/heightnet_repro
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  tools/generate_yolo26_bboxes.py \
  --manifest /home/zyding/data/train.csv /home/zyding/data/val.csv /home/zyding/data/test.csv \
  --out-dir runs/yolo26_bbox_all/main_card_out \
  --model /path/to/yolo26n.pt \
  --frames-per-video 20 \
  --imgsz 960 \
  --conf 0.25 \
  --iou 0.7 \
  --min-detections 5
```

The output JSON name must match:

```text
<person_id>_<video_stem>.json
```

`tools/convert_all_to_npz.py` defaults to exact matching and will reject missing exact JSONs unless `--allow-parsing-fallback` is passed.

### 2. Convert videos to NPZ features

Recommended filtered feature root:

```text
/home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8
```

Example:

```bash
cd /home/zyding/height/heightnet_repro
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python tools/convert_all_to_npz.py \
  --manifest /home/zyding/data/train.csv /home/zyding/data/val.csv /home/zyding/data/test.csv \
  --parsing-json-root runs/yolo26_bbox_all/main_card_out \
  --output-root /home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8 \
  --bg-depth-root /home/zyding/height/jianzhi_2511_spilt_coat/bg_depthmap \
  --depthanything-root /home/zyding/height/Depth-Anything-V2 \
  --checkpoint /home/zyding/height/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --frames-per-video 20 \
  --compact-valid-frames \
  --min-bbox-h-norm 0.08 \
  --min-bbox-w-norm 0.02 \
  --min-bbox-score 0.25 \
  --overwrite-summary
```

Important behavior:

- Default `--exact-parsing-only` prevents accidental fallback to another video.
- `--compact-valid-frames` stores only bbox-matched rows, so `valid_count == bbox_feats.shape[0] == heightmap_crops.shape[0]`.
- Without `--compact-valid-frames`, the converter keeps uniform sampled frames and uses zero placeholders for missing bbox rows.

### 3. Validate NPZ schema

Quick audit:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path('/home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8')
bad = []
counts = []
for path in sorted(root.glob('**/*.npz')):
    try:
        data = np.load(path, allow_pickle=False)
        bbox = data['bbox_feats']
        stats = data['height_stats']
        crops = data['heightmap_crops']
        valid_count = int(data['valid_count'].reshape(-1)[0])
        ok = (
            bbox.ndim == 2 and bbox.shape[1] == 7 and
            stats.ndim == 2 and stats.shape[0] == bbox.shape[0] and
            crops.ndim == 4 and tuple(crops.shape[1:]) == (1, 128, 64) and
            valid_count == bbox.shape[0] == crops.shape[0] and
            valid_count > 0
        )
        counts.append(valid_count)
        if not ok:
            bad.append(str(path))
    except Exception as exc:
        bad.append(f'{path}: {exc!r}')
print({'files': len(counts) + len(bad), 'bad': len(bad), 'min_count': min(counts) if counts else None})
print('\n'.join(bad[:20]))
raise SystemExit(1 if bad else 0)
PY
```

### 4. Export crop quality review

```bash
python tools/export_cross_camera_crop_quality.py \
  --feature-root /home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8 \
  --person-split-json runs/cross_camera_heightmap_fusion/strict_person_split_seed1/person_split.json \
  --out-dir runs/cross_camera_heightmap_fusion/crop_quality_yolo26_filtered_h008_w002_s025_min8 \
  --splits train val test \
  --samples-per-split 80 \
  --samples-per-camera 8 \
  --frames-per-video 4 \
  --seed 1
```

Open `index.html` from the output directory to inspect crop panels.

For targeted reviews:

```bash
python tools/export_targeted_crop_review.py \
  --feature-root /home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8 \
  --out-dir runs/cross_camera_heightmap_fusion/crop_quality_targeted_seed1 \
  --samples-per-category 16 \
  --seed 1
```

### 5. Train cross-camera heightmap fusion

Single GPU smoke:

```bash
python tools/train_cross_camera_heightmap_fusion.py \
  --feature-root /home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8 \
  --person-split-json runs/cross_camera_heightmap_fusion/strict_person_split_seed1/person_split.json \
  --out-dir runs/cross_camera_heightmap_fusion/smoke_yolo26_seed1 \
  --epochs 2 \
  --steps-per-epoch 2 \
  --batch-size 4 \
  --consistency-batch-size 2 \
  --track-batch-size 2 \
  --seed 1
```

4-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  tools/train_cross_camera_heightmap_fusion.py \
  --feature-root /home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8 \
  --person-split-json runs/cross_camera_heightmap_fusion/strict_person_split_seed1/person_split.json \
  --out-dir runs/cross_camera_heightmap_fusion/georankh_margin_yolo26_filtered_h008_w002_s025_min8_ddp_seed1 \
  --epochs 300 \
  --steps-per-epoch 100 \
  --batch-size 32 \
  --consistency-batch-size 8 \
  --track-batch-size 8 \
  --lambda-cross 0.5 \
  --lambda-track 0.2 \
  --lambda-id-score 0.2 \
  --lambda-id-same-camera 0.2 \
  --lambda-margin 1.0 \
  --lambda-rank-bce 0.0 \
  --min-hard-gap-cm 3.0 \
  --margin-tau-cm 6.0 \
  --margin-max 2.0 \
  --seed 1
```

Result file:

```text
runs/cross_camera_heightmap_fusion/<run_name>/results.json
```

The script prints and writes both primary video-level metrics and explicitly named non-primary legacy aggregation metrics.

### 6. One-shot helper

After NPZ conversion has been launched, this helper waits for conversion to finish, audits NPZs, exports crop review HTML, then starts the DDP experiment:

```bash
bash tools/run_yolo26_fixed_experiment_after_npz.sh
```

Read and edit the path variables at the top of the script before reuse.

## Common Files

- `tools/generate_yolo26_bboxes.py`: sample frames and write detector bbox JSONs.
- `tools/convert_all_to_npz.py`: convert videos to sequence NPZ features.
- `tools/train_cross_camera_heightmap_fusion.py`: current cross-camera fusion trainer.
- `tools/cross_camera_heightmap_fusion_core.py`: NPZ loader, geometry helpers, shared model code.
- `tools/export_cross_camera_crop_quality.py`: balanced crop quality HTML review.
- `tools/export_strict_crop_overlay_audit.py`: overlay audit for filtered NPZ rows against source video/JSON.
- `tools/repack_filtered_npz_by_crop_quality.py`: post-filter and repack existing NPZ roots.
- `tests/test_convert_all_to_npz.py`: regression tests for exact bbox filtering and compact/non-compact NPZ behavior.

## Git Hygiene

Commit code, configs, tests, and docs. Do not commit:

- `runs/`
- `remote_runs/`
- `.npz`, `.npy`
- checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- logs and local virtualenvs

Before pushing:

```bash
python -m pytest tests/test_convert_all_to_npz.py tests/test_cross_camera_heightmap_fusion.py tests/test_sequence_npz_feature_extraction.py -q
python -m py_compile \
  tools/convert_all_to_npz.py \
  tools/train_cross_camera_heightmap_fusion.py \
  tools/export_cross_camera_crop_quality.py \
  tools/export_strict_crop_overlay_audit.py \
  tools/export_targeted_crop_review.py \
  tools/generate_yolo26_bboxes.py \
  tools/repack_exact_normal_npz.py \
  tools/repack_filtered_npz_by_crop_quality.py
```

## Troubleshooting

SSH timeout to 235:

```bash
nc -vz -G 5 192.168.100.235 22
ping -c 2 192.168.100.235
```

If both fail, connect to the required internal network/VPN or check whether the server is up.

Missing exact parsing JSON:

- Check that `runs/yolo26_bbox_all/main_card_out/<person_id>_<video_stem>.json` exists.
- Do not use `--allow-parsing-fallback` for full conversion unless you are intentionally reproducing legacy behavior.

CUDA unavailable:

- Verify server driver and PyTorch CUDA build.
- Reinstall PyTorch from the official CUDA wheel index matching the server.

Bad NPZ schema:

- Ensure `--compact-valid-frames` was used for the filtered yolo26 feature root.
- Re-run the schema audit in this document.
