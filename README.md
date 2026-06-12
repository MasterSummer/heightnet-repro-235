# HeightNet Reproduction Skeleton

This folder provides a runnable HeightNet-style baseline centered on video-level pairwise ranking:

1. Build train/val/test manifests directly from raw videos with person-level split.
2. Decode video frames online during training/evaluation.
3. Predict dense height maps and generate runtime person masks on the fly.
4. Optimize dense height regression as an auxiliary task and video-level pairwise ranking as the primary task.

## Project Structure

- `configs/default.yaml`: default training config.
- `src/heightnet/`: dataset, model, losses, metrics, utility code.
- `tools/precompute_height_labels.py`: create `height/*.npy` and `valid_mask/*.npy`.
- `tools/build_manifest.py`: scan raw videos and build train/val/test manifests with person-level split.
- `tools/generate_pairwise_rank_from_2503.py`: generate pairwise labels from `2503_test_rank`.
- `train.py`: training entrypoint.
- `evaluate.py`: testing + visualization.

## Expected Data Layout

```text
data_root/
  <person_id>/
    *.mp4
  height_cache/<person_id>__<video_stem>.npz  # optional
  valid_mask_cache/<person_id>__<video_stem>.npz  # optional
  depth_cache/<person_id>__<video_stem>.npz|npy  # optional
```

Default label-generation behavior:

- Corrected formula: `h = C_h * (D_b - D_f) / D_b`.
- `D_b` must come from camera background depth map (`--bg-depth-root`).
- `person_mask` can be generated at runtime by segmentation model (recommended).
- `pairwise_rank` is used in training via `loss.pairwise_json` and in evaluation via `--rank-dir`.

## Quick Start

```bash
cd /Users/yiding/code/gait/height/heightnet_repro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Build manifests directly from raw videos with person-level split
python tools/build_manifest.py \
  --video-root /path/to/video_root \
  --out-dir /Users/yiding/code/gait/height/heightnet_repro/data \
  --bg-depth-root /path/to/2503_test_bg_depthmap \
  --allow-online-depth-supervision

# 2) Train
# set runtime_seg.model_path in configs/default.yaml first (e.g. yolov8n-seg.pt)
# pairwise supervision defaults to data/pairwise_rank/all_pairs.json
python train.py --config configs/default.yaml

# 3) Evaluate
python evaluate.py --config configs/default.yaml --checkpoint runs/default/checkpoint_best.pt --rank-dir /Users/yiding/code/gait/2503_test_rank

# 4) Optional: export explicit pairwise labels from 2503_test_rank
python tools/generate_pairwise_rank_from_2503.py --rank-dir /Users/yiding/code/gait/2503_test_rank --out-dir data/pairwise_rank
```

### 4-GPU Training (A100 x4, 最小可跑)

```bash
cd /Users/yiding/code/gait/height/heightnet_repro
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# 1) 生成视频级 manifest（只需一次）
python tools/build_manifest.py \
  --video-root /path/to/video_root \
  --out-dir /Users/yiding/code/gait/height/heightnet_repro/data/run_4gpu \
  --bg-depth-root /path/to/bg_depth_root \
  --allow-online-depth-supervision

# 2) 训练（配置见 HEIGHTNET_REPRO_IO_FLOW.md 第 8 节）
torchrun --standalone --nproc_per_node=4 train.py --config configs/a100_4gpu_min.yaml

# 3) 评估
python evaluate.py \
  --config configs/a100_4gpu_min.yaml \
  --checkpoint /Users/yiding/code/gait/height/heightnet_repro/runs/a100_4gpu_min/checkpoint_best.pt \
  --rank-dir /Users/yiding/code/gait/2503_test_rank
```

## Notes

- This is a pragmatic scaffold, not the exact original HeightNet implementation.
- Current default model selection is based on validation `pairwise accuracy`, not RMSE.
- Runtime segmentation keeps only the largest detected person mask per frame.
- To align closer to paper-reported numbers, you should replace `HeightNetTiny` with the LapDepth-style encoder-decoder and tune data generation details.
