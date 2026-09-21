# HeightNet 复现实验骨架

当前服务器 235 的交接流程请先阅读 [HANDOFF.md](HANDOFF.md)。
下面的内容介绍较早的 HeightNet 风格实验骨架，保留用于参考。

本目录提供了一个可以运行的 HeightNet 风格基线，核心是视频级 pairwise ranking：

1. 直接从原始视频构建 train/val/test manifest，并按人员划分数据集。
2. 在训练和评估过程中在线解码视频帧。
3. 预测 dense height map，并在运行时生成人员 mask。
4. 将 dense height regression 作为辅助任务，将视频级 pairwise ranking 作为主任务。

## 项目结构

- `configs/default.yaml`：默认训练配置。
- `src/heightnet/`：数据集、模型、损失、指标和工具代码。
- `tools/precompute_height_labels.py`：生成 `height/*.npy` 和 `valid_mask/*.npy`。
- `tools/build_manifest.py`：扫描原始视频，并按人员划分生成 train/val/test manifest。
- `tools/generate_pairwise_rank_from_2503.py`：从 `2503_test_rank` 生成 pairwise 标签。
- `train.py`：训练入口。
- `evaluate.py`：测试和可视化入口。

## 预期数据结构

```text
data_root/
  <person_id>/
    *.mp4
  height_cache/<person_id>__<video_stem>.npz  # 可选
  valid_mask_cache/<person_id>__<video_stem>.npz  # 可选
  depth_cache/<person_id>__<video_stem>.npz|npy  # 可选
```

默认标签生成行为：

- 修正后的公式：`h = C_h * (D_b - D_f) / D_b`。
- `D_b` 必须来自摄像头背景深度图（`--bg-depth-root`）。
- `person_mask` 可以由分割模型在运行时生成（推荐方式）。
- 训练时通过 `loss.pairwise_json` 使用 `pairwise_rank`，评估时通过 `--rank-dir` 使用。

## 快速开始

```bash
cd /Users/yiding/code/gait/height/heightnet_repro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) 从原始视频构建 manifest，并按人员划分
python tools/build_manifest.py \
  --video-root /path/to/video_root \
  --out-dir /Users/yiding/code/gait/height/heightnet_repro/data \
  --bg-depth-root /path/to/2503_test_bg_depthmap \
  --allow-online-depth-supervision

# 2) 训练
# 先在 configs/default.yaml 中设置 runtime_seg.model_path，例如 yolov8n-seg.pt
# pairwise supervision 默认使用 data/pairwise_rank/all_pairs.json
python train.py --config configs/default.yaml

# 3) 评估
python evaluate.py --config configs/default.yaml --checkpoint runs/default/checkpoint_best.pt --rank-dir /Users/yiding/code/gait/2503_test_rank

# 4) 可选：从 2503_test_rank 导出显式 pairwise 标签
python tools/generate_pairwise_rank_from_2503.py --rank-dir /Users/yiding/code/gait/2503_test_rank --out-dir data/pairwise_rank
```

### 4 卡训练（A100 x4，最小可运行配置）

```bash
cd /Users/yiding/code/gait/height/heightnet_repro
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# 1) 生成视频级 manifest（只需执行一次）
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

## 说明

- 这是一个面向实验的复现骨架，不是原始 HeightNet 的完整实现。
- 当前默认根据验证集 `pairwise accuracy` 选择模型，而不是根据 RMSE。
- 运行时分割每帧只保留检测到的最大人员 mask。
- 如果需要更接近论文中的结果，应将 `HeightNetTiny` 替换为 LapDepth 风格的 encoder-decoder，并进一步调整数据生成细节。
