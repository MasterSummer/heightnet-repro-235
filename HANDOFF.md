# HeightNet 235 服务器交接说明

本仓库是服务器 `235` 上用于视频级身高排序实验的当前代码仓库。

## 工作方式

- 先在本地修改代码。
- 将代码同步到服务器 `235`。
- 在服务器 `235` 上运行训练和评估实验。
- 生成数据、模型权重和 HTML 审查结果不提交到 Git。

服务器信息：

```text
主机别名：gait-server-235
用户：zyding
远程仓库目录：/home/zyding/height/heightnet_repro
默认数据目录：/home/zyding/data
```

用于 Git 交接的本地镜像：

```text
/Users/yiding/code/gait/heightnet-repro-235
origin: https://github.com/MasterSummer/heightnet-repro-235.git
```

本地工作区中另有 `/Users/yiding/code/gait/height/heightnet_repro`。该目录属于开发副本，可能与本仓库的 235 镜像存在差异。需要交接或推送服务器实验代码时，以本仓库为准。

## 评估口径

默认使用视频级评估：每个视频作为一个独立样本。

主指标不得将视频聚合为 person-camera 样本，也不得将视频聚合为 person 样本。若为了和历史结果对比而报告聚合指标，必须明确标注为非主指标（non-primary）。

评估模型质量时，训练集、验证集和测试集应使用按人员划分的 person-disjoint split，保证同一人员不会跨集合出现。

## 环境配置

在服务器 `235` 上执行：

```bash
ssh gait-server-235
cd /home/zyding/height/heightnet_repro

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果通用的 `pip install -r requirements.txt` 安装成了 CPU 版本，进行 GPU 训练前需要根据服务器驱动安装匹配的 PyTorch CUDA 版本。检查环境：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Depth-Anything-V2 默认位于仓库外部：

```text
/home/zyding/height/Depth-Anything-V2
/home/zyding/height/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
```

YOLO 权重可以使用 `yolo26n.pt` 或其他本地 `.pt` 文件。建议将权重放在固定的服务器路径，并通过 `--model` 明确传入。

## 本地同步到服务器

在本地工作区根目录执行：

```bash
/Users/yiding/code/gait/scripts/sync_heightnet_to_235.sh --dry-run
/Users/yiding/code/gait/scripts/sync_heightnet_to_235.sh
```

同步脚本会排除 `runs/`、`remote_runs/`、`data/`、缓存、`.npz`、`.npy`、模型权重和日志等生成内容，不会删除服务器端专有文件。

同步后检查服务器工作区：

```bash
ssh gait-server-235
cd /home/zyding/height/heightnet_repro
git status --short
```

## 当前推荐流程

当前交接流程包括：生成 yolo26 检测框 JSON、使用精确匹配和过滤条件转换 NPZ、审查裁剪质量，以及训练跨摄像头 heightmap fusion 模型。

### 1. 生成 YOLO 检测框 JSON

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

输出 JSON 文件名必须符合：

```text
<person_id>_<video_stem>.json
```

`tools/convert_all_to_npz.py` 默认使用精确匹配。如果找不到对应的精确 JSON，会直接拒绝处理；只有明确需要兼容旧行为时，才传入 `--allow-parsing-fallback`。

### 2. 将视频转换为 NPZ 特征

推荐的过滤后特征目录：

```text
/home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8
```

示例：

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

重要行为：

- 默认启用 `--exact-parsing-only`，防止误用其他视频的检测结果。
- `--compact-valid-frames` 只保存与检测框匹配的帧，因此 `valid_count == bbox_feats.shape[0] == heightmap_crops.shape[0]`。
- 不使用 `--compact-valid-frames` 时，转换器保留均匀采样的帧；缺少检测框的行用全零占位。

### 3. 检查 NPZ 数据结构

快速审查：

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

### 4. 导出裁剪质量审查结果

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

打开输出目录中的 `index.html`，检查各个裁剪面板。

针对特定类别审查：

```bash
python tools/export_targeted_crop_review.py \
  --feature-root /home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8 \
  --out-dir runs/cross_camera_heightmap_fusion/crop_quality_targeted_seed1 \
  --samples-per-category 16 \
  --seed 1
```

### 5. 训练跨摄像头 heightmap fusion

单卡冒烟测试：

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

4 卡训练：

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

结果文件：

```text
runs/cross_camera_heightmap_fusion/<run_name>/results.json
```

脚本会同时输出并保存视频级主指标，以及明确标注为非主指标的历史聚合指标。

### 6. 一键辅助脚本

启动 NPZ 转换后，可以使用以下脚本等待转换结束、检查 NPZ、导出裁剪审查 HTML，然后启动 DDP 实验：

```bash
bash tools/run_yolo26_fixed_experiment_after_npz.sh
```

重复使用前，先阅读并修改脚本顶部的路径变量。

## 常用文件

- `tools/generate_yolo26_bboxes.py`：抽帧并写入检测框 JSON。
- `tools/convert_all_to_npz.py`：将视频转换为序列 NPZ 特征。
- `tools/train_cross_camera_heightmap_fusion.py`：当前跨摄像头 fusion 训练脚本。
- `tools/cross_camera_heightmap_fusion_core.py`：NPZ 加载、几何处理和共享模型代码。
- `tools/export_cross_camera_crop_quality.py`：导出平衡采样的裁剪质量 HTML 审查页。
- `tools/export_strict_crop_overlay_audit.py`：将过滤后的 NPZ 行与原视频和 JSON 做叠加审查。
- `tools/repack_filtered_npz_by_crop_quality.py`：对已有 NPZ 目录做后处理过滤和重新打包。
- `tests/test_convert_all_to_npz.py`：精确检测框过滤、compact/non-compact NPZ 行为的回归测试。

## Git 使用规范

应提交代码、配置、测试和文档。以下内容不要提交：

- `runs/`
- `remote_runs/`
- `.npz`、`.npy`
- 模型权重（`*.pt`、`*.pth`、`*.ckpt`）
- 日志和本地虚拟环境

推送前执行：

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

## 常见问题

连接 235 超时：

```bash
nc -vz -G 5 192.168.100.235 22
ping -c 2 192.168.100.235
```

如果两项都失败，请检查是否已连接必要的内网或 VPN，以及服务器是否在线。

找不到精确匹配的 parsing JSON：

- 检查 `runs/yolo26_bbox_all/main_card_out/<person_id>_<video_stem>.json` 是否存在。
- 除非有意复现旧流程，否则完整转换时不要使用 `--allow-parsing-fallback`。

CUDA 不可用：

- 检查服务器驱动和 PyTorch CUDA 构建版本。
- 根据服务器环境，从 PyTorch 官方 CUDA wheel 源重新安装匹配版本。

NPZ 数据结构错误：

- 确认过滤后的 yolo26 特征目录使用了 `--compact-valid-frames`。
- 重新执行本说明中的 NPZ 数据结构检查命令。
