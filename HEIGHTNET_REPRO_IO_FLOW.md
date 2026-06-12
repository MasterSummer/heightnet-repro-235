# HeightNet Repro: 输入输出与数据流说明（视频直入 + 高度图排序）

## 1. 项目目标

本项目采用 HeightNet 思路，但任务定义调整为“相对高低排序”而非“绝对身高回归”：
- 输入为视频流（不预先切分为图片）。
- 主干输出每像素高度图（相对地面高度）。
- 不再通过“人体区域最大值 -> 身高数值”做比较。
- 改为在去背景后高度图上接排序 head，直接做高低二分类/排序学习。

---

## 2. 总体输入与输出

### 2.1 训练输入

- 视频文件：`{split}/videos/<sequence_id>.mp4`
- 可选预计算深度缓存：`{split}/depth_cache/<sequence_id>.npy|npz`（无缓存时在线推理）
- 摄像头背景平均深度图（必需）：  
  `<bg_depth_root>/<camera_id>/<camera_id>_avg_depth.npy`
- 排序监督（pair）文件：`data/pairwise_rank/all_pairs.json`

说明：
- DataLoader 在训练时从视频中按索引解码帧，不再依赖 `rgb/*.png` 作为主输入。
- 人体掩码默认在线分割获得（可选离线缓存）。

### 2.2 核心输出

- 主输出：`pred_height_map`（每帧高度图）
- 排序输出：`rank_logit`（二分类，高/低）
- 模型权重：`runs/<exp>/checkpoint_last.pt`、`checkpoint_best.pt`
- 可视化：`runs/<exp>/vis/*.png`（含去背景高度图与排序结果）
- 评估日志：`pairwise ranking accuracy`、`AUC/F1`、`RMSE`

---

## 3. 目录结构（期望）

```text
data_root/
  train/
    videos/*.mp4
    person_mask_cache/*.npy              # 可选
    depth_cache/*.npz                    # 可选
  val/
    videos/*.mp4
  test/
    videos/*.mp4
```

`heightnet_repro/data/` 下生成：
- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`

---

## 4. 关键脚本的输入输出

## 4.1 `tools/prepare_midrun_dataset.py`

输入：
- 视频目录：`<video_root>/<person_id>/*.mp4`

输出：
- 视频级数据集拆分（train/val/test）
- 序列清单：`midrun_train.txt`、`midrun_val.txt`、`midrun_test.txt`
- 不再输出“抽帧后的 rgb 数据集”

## 4.2 `tools/build_manifest.py`

输入：
- `data_root`（以 `videos` 为主）

输出：
- `*_manifest.csv`（视频级索引表）

每行主要字段（建议）：
- `video_path`
- `sequence_id`
- `person_id`
- `camera_id`
- `frame_start`
- `frame_end`
- `fps`


## 4.4 `train.py`

输入：
- `train_manifest.csv`、`val_manifest.csv`
- `configs/default.yaml`

输出：
- `checkpoint_last.pt`、`checkpoint_best.pt`

损失（更新后）：
- 排序二分类损失：`BCEWithLogits(rank_logit, y_pair)`（主损失）
- 排序边际损失（可选）：`MarginRankingLoss(s_i, s_j, y)`
- 相邻帧一致性损失（必须保留，已改可运行版本）：
  - 用相邻帧 `t,t+1` 的同一 track mask 提取去背景高度图区域；
  - 对区域内特征做 `masked average pooling` 得到 `s_t,s_t+1`；
  - `L_cons = SmoothL1(s_t, s_t+1)`；
  - 若任一帧有效像素数 `< min_valid_pixels`，该对样本跳过并返回 0，不参与反传；
  - 批次级采用 `valid_pair_count>0` 才归一化，避免 NaN/除零导致训练中断。


## 4.5 `evaluate.py`

输入：
- `test_manifest.csv`
- 训练好的 checkpoint
- `--rank-dir`（如 `2503_test_rank`）

输出：
- 每摄像头及总体 `pairwise ranking accuracy`
- `AUC/F1`（高低二分类）
- 可视化图表

---

## 5. 端到端数据流（新）

1. 视频清单构建  
`raw videos -> train/val/test manifest`

2. 训练时在线解码  
`video -> sampled adjacent frames`

3. 深度/高度图估计与去背景  
`frame + bg_depth_map -> pred_height_map`  
`pred_height_map * person_mask -> foreground_height_map`

4. 排序 head  
`foreground_height_map -> rank_head -> rank_logit`

5. 模型训练  
`L_total = L_rank_cls + lambda_rank * L_margin + lambda_cons * L_cons`

6. 评估  
`rank_logit + pair labels -> pairwise ranking accuracy / AUC / F1`

---

## 6. 一致性损失无法运行的修复约束（落地要求）

- 不再使用“最大值池化 + 空 mask”路径，改为 `masked average pooling`。
- 对每个相邻帧对增加有效性检查：
  - mask 非空；
  - 去背景后有效像素比例 `>= min_valid_ratio`；
  - 对应 track/person id 一致。
- 无有效相邻对时：
  - `L_cons` 直接置 0；
  - 日志记录 `cons_valid_pairs=0`；
  - 不报错、不终止训练。
- 所有分母都加 `eps`（如 `1e-6`），杜绝除零。

---

## 7. 常见问题

1. `manifest` 全是 0 行：  
通常是 `videos/` 路径或 split 规则不匹配。

2. 在线分割报错找不到模型：  
检查 `runtime_seg.model_path` 是否为有效 `*-seg.pt`。

3. `pairwise` 没结果：  
检查 `manifest` 里的 `person_id/camera_id` 是否能匹配 `rank-dir` 的 JSON。

4. 一致性损失出现 NaN 或崩溃：  
先检查 `cons_valid_pairs` 是否长期为 0，再检查 `min_valid_pixels/min_valid_ratio/eps` 配置。

---

## 8. 服务器最小可跑（4xA100 40G）

### 8.1 生成视频级 manifest（只需一次）

```bash
cd /Users/yiding/code/gait/height/heightnet_repro

python tools/build_manifest.py \
  --data-root /path/to/data_root \
  --split all \
  --out-dir /Users/yiding/code/gait/height/heightnet_repro/data/run_4gpu \
  --bg-depth-root /path/to/bg_depth_root
```

### 8.2 4 卡训练配置（示例）

推荐直接使用仓库内模板：`configs/a100_4gpu_min.yaml`（只改路径字段）。

保存为 `configs/a100_4gpu_min.yaml`：

```yaml
seed: 42
device: cuda

paths:
  train_manifest: /Users/yiding/code/gait/height/heightnet_repro/data/run_4gpu/train_manifest.csv
  val_manifest: /Users/yiding/code/gait/height/heightnet_repro/data/run_4gpu/val_manifest.csv
  test_manifest: /Users/yiding/code/gait/height/heightnet_repro/data/run_4gpu/test_manifest.csv
  output_dir: /Users/yiding/code/gait/height/heightnet_repro/runs/a100_4gpu_min

train:
  epochs: 20
  batch_size: 16
  num_workers: 8
  lr: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip_norm: 1.0
  use_amp: true
  use_tensorboard: true
  log_interval: 20
  tb_num_images: 4

loss:
  lambda_rmse: 1.0
  lambda_rank: 1.0
  lambda_cons: 0.1
  eps: 1.0e-6
  min_valid_pixels: 64
  min_valid_ratio: 0.002
  pairwise_json: /Users/yiding/code/gait/height/heightnet_repro/data/pairwise_rank/all_pairs.json

data:
  image_size: [352, 704]
  normalize_rgb: true
  use_pair_consistency: true

model:
  name: heightnet_tiny
  base_channels: 32
  comparator_channels: 16

eval:
  save_visualizations: true
  vis_limit: 20

runtime_seg:
  enabled: true
  model_path: /path/to/yolov8n-seg.pt
  conf: 0.25
  iou: 0.7
  imgsz: 640
  strict_native: true
```

### 8.3 启动 4 卡 DDP 训练

```bash
cd /Users/yiding/code/gait/height/heightnet_repro
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node=4 train.py --config configs/a100_4gpu_min.yaml
```

### 8.4 评估

```bash
python evaluate.py \
  --config configs/a100_4gpu_min.yaml \
  --checkpoint /Users/yiding/code/gait/height/heightnet_repro/runs/a100_4gpu_min/checkpoint_best.pt \
  --rank-dir /Users/yiding/code/gait/2503_test_rank
```
