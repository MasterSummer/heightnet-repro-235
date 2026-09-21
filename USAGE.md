# HeightNet 使用说明

本项目的主流程是：原始视频和标注 -> YOLO 人体框 -> NPZ 特征 -> 裁剪质量审查 -> 跨摄像头身高排序训练 -> results.json。

所有命令都在仓库根目录执行。先安装依赖：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
~~~

## 一、程序入口

| 阶段 | 入口脚本 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| 生成检测框 | tools/generate_yolo26_bboxes.py | manifest、视频、YOLO 权重 | 每个视频一个 bbox JSON、检测报告 |
| 生成特征 | tools/convert_all_to_npz.py | manifest、视频、bbox JSON、背景深度、Depth-Anything | 每个视频一个 NPZ、转换汇总 |
| 生成人员划分 | tools/build_strict_person_split.py | 身高标签文件 | person_split.json |
| 检查输入 | tools/audit_cross_camera_fusion_inputs.py | manifest、标签、NPZ、coverage manifest | input_audit.json |
| 审查裁剪 | tools/export_cross_camera_crop_quality.py | NPZ、manifest、人员划分 | index.html、PNG、index.json |
| 定向审查 | tools/export_targeted_crop_review.py | NPZ | 分类审查 HTML、PNG、index.json |
| 训练模型 | tools/train_cross_camera_heightmap_fusion.py | NPZ、标签、人员划分 | checkpoint_best.pt、results.json |

已有完整实验脚本时，也可以使用 tools/run_yolo26_fixed_experiment_after_npz.sh。该脚本会等待 NPZ 转换结束，执行数据检查和裁剪审查，然后启动训练；重复使用前需要先修改脚本顶部的路径变量。

## 二、路径变量

下面的变量只是示例，请替换成实际路径：

~~~bash
REPO=/path/to/heightnet-repro
DATA_ROOT=/path/to/data
VIDEO_ROOT=/path/to/videos
BBOX_ROOT=$REPO/runs/yolo26_bbox/main_card_out
FEATURE_ROOT=/path/to/features_yolo26_bbox_filtered
BG_DEPTH_ROOT=/path/to/bg_depthmap
DA_ROOT=/path/to/Depth-Anything-V2
DA_CHECKPOINT=$DA_ROOT/checkpoints/depth_anything_v2_vits.pth
LABEL_PATH=$DATA_ROOT/label/rank.json
SPLIT_JSON=$REPO/runs/person_split.json
~~~

manifest 中的 video_path 必须是可以直接打开的完整路径。当前转换脚本不会自动把 --video-root 拼接到 video_path 前面。

## 三、输入文件格式

### 1. 视频 manifest CSV

检测脚本可以接收一个或多个 CSV；特征转换和训练默认读取：

~~~text
DATA_ROOT/train.csv
DATA_ROOT/val.csv
DATA_ROOT/test.csv
~~~

推荐字段如下：

~~~csv
person_id,video_filename,video_path,split,action,coat_type
p001,p001_walk_3d0_1_720w.mp4,/path/to/videos/p001_walk_3d0_1_720w.mp4,train,walk,normal
p002,p002_walk_3d0_1_720w.mp4,/path/to/videos/p002_walk_3d0_1_720w.mp4,val,walk,normal
~~~

字段说明：

- person_id：人员 ID，必须稳定且与身高标签、bbox 文件名一致。
- video_filename：视频文件名，推荐填写；缺失时脚本会从 video_path 取文件名。
- video_path：视频实际路径，必须存在。
- split：train、val 或 test。
- action、coat_type：可选元数据；检测筛选和实验分析会使用它们。

摄像头 ID 从视频文件名中解析。文件名需要包含类似 _3d0_1_ 的片段，对应摄像头 ID 3d0_1，摄像头高度按 3d0 解析为 3.0 米。例如：

~~~text
p001_walk_3d0_1_720w.mp4  ->  camera_id = 3d0_1
~~~

### 2. 身高标签文件

默认路径是 DATA_ROOT/label/rank.json。虽然扩展名是 .json，当前代码按 CSV 文本读取：

~~~csv
person_id,height(cm)
p001,172.5
p002,168.0
~~~

身高列也可以命名为 height_cm 或 height；人员列也兼容 penson_id 或 pid。

### 3. 人员划分 JSON

训练时建议显式传入按人员划分的 JSON，保证同一人员不会出现在多个集合：

~~~json
{
  "train": ["p001", "p004"],
  "val": ["p002"],
  "test": ["p003"]
}
~~~

也可以使用脚本生成：

~~~bash
python tools/build_strict_person_split.py \
  --label-path "$LABEL_PATH" \
  --seed 1 \
  --out "$SPLIT_JSON"
~~~

### 4. 背景深度文件

转换特征时，背景深度目录需要按摄像头组织：

~~~text
BG_DEPTH_ROOT/
  3d0_1/3d0_1_avg_depth.npy
  3d0_2/3d0_2_avg_depth.npy
~~~

每个 .npy 应是二维深度数组。代码会在尺寸不同时调整到视频帧尺寸；也兼容包含 depth 字段的 NPZ 文件。

### 5. bbox JSON

文件名必须是：

~~~text
<person_id>_<video_stem>.json
~~~

例如 manifest 中 person_id=p001、video_filename=p001_walk_3d0_1_720w.mp4，对应：

~~~text
BBOX_ROOT/p001_p001_walk_3d0_1_720w.json
~~~

文件内容采用 main_card_out 兼容结构：

~~~json
{
  "0": {
    "video_path": "/path/to/videos/p001_walk_3d0_1_720w.mp4",
    "person_id": "p001",
    "video_stem": "p001_walk_3d0_1_720w",
    "split": "train",
    "action": "walk",
    "sub_track": [
      {
        "track_id": 0,
        "data": {
          "0_0": [
            {
              "frame_id": 120,
              "rect": [120.0, 40.0, 180.0, 460.0],
              "score": 0.95,
              "source": "yolo26"
            }
          ]
        }
      }
    ]
  }
}
~~~

其中 frame_id 从 0 开始，rect 为像素坐标 [x, y, width, height]，score 为检测置信度。

### 6. NPZ 特征文件

输出路径为：

~~~text
FEATURE_ROOT/<person_id>/<video_stem>.npz
~~~

过滤模式下推荐使用 --compact-valid-frames。一个合法 NPZ 至少包含：

| 字段 | 形状或类型 | 含义 |
| --- | --- | --- |
| bbox_feats | (N, 7) float32 | 归一化 bbox 特征和检测分数 |
| height_stats | (N, 8) float32 | bbox 内高度图统计量 |
| heightmap_crops | (N, 1, 128, 64) float32 | bbox 周围的高度图裁剪 |
| valid_count | 一个整数 | 有效帧数量 |
| camera_id | 一个字符串 | 摄像头 ID |
| video_stem | 一个字符串 | 视频 stem |

使用 --compact-valid-frames 时应满足：

~~~text
valid_count == bbox_feats.shape[0] == height_stats.shape[0] == heightmap_crops.shape[0]
~~~

不使用该参数时，转换器保留均匀采样的帧，缺失 bbox 的行用零填充；这种模式不适合作为过滤后的正式训练输入。

## 四、运行流程

### 1. 生成 YOLO bbox

~~~bash
python tools/generate_yolo26_bboxes.py \
  --manifest "$DATA_ROOT/train.csv" "$DATA_ROOT/val.csv" "$DATA_ROOT/test.csv" \
  --out-dir "$BBOX_ROOT" \
  --model /path/to/yolo26n.pt \
  --frames-per-video 20 \
  --imgsz 960 \
  --conf 0.25 \
  --iou 0.7 \
  --min-detections 5 \
  --report-dir "$REPO/runs/yolo26_bbox/reports"
~~~

输出：

~~~text
BBOX_ROOT/<person_id>_<video_stem>.json
runs/yolo26_bbox/reports/yolo26_bbox_report.rank0.json
~~~

报告包含 stats 和 results。重点查看 ok、low_coverage、failed 数量，以及每个视频的 detections 和 frames_requested。

### 2. 转换 NPZ 特征

~~~bash
python tools/convert_all_to_npz.py \
  --video-root "$VIDEO_ROOT" \
  --manifest "$DATA_ROOT/train.csv" "$DATA_ROOT/val.csv" "$DATA_ROOT/test.csv" \
  --parsing-json-root "$BBOX_ROOT" \
  --output-root "$FEATURE_ROOT" \
  --bg-depth-root "$BG_DEPTH_ROOT" \
  --depthanything-root "$DA_ROOT" \
  --checkpoint "$DA_CHECKPOINT" \
  --encoder vits \
  --frames-per-video 20 \
  --compact-valid-frames \
  --min-bbox-h-norm 0.08 \
  --min-bbox-w-norm 0.02 \
  --min-bbox-score 0.25 \
  --overwrite-summary
~~~

默认只接受精确的 <person_id>_<video_stem>.json。只有需要复现旧行为时才使用 --allow-parsing-fallback。

输出：

~~~text
FEATURE_ROOT/<person_id>/<video_stem>.npz
FEATURE_ROOT/convert_all_to_npz_summary.rank0.json
~~~

汇总文件包含：

- stats.selected：选中的视频数。
- stats.ok：成功生成 NPZ 的视频数。
- stats.missing_parsing：找不到精确 bbox JSON 的视频数。
- stats.missing_bg：找不到背景深度的数量。
- results[*].valid_count：当前视频保留的有效帧数。
- results[*].missing_bbox_frames：均匀参考采样中缺少 bbox 的帧数。
- results[*].output：输出 NPZ 路径。

### 3. 检查 NPZ

先检查关键字段和形状：

~~~bash
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path("/path/to/features")
bad = []
counts = []
for path in sorted(root.glob("**/*.npz")):
    try:
        data = np.load(path, allow_pickle=False)
        bbox = data["bbox_feats"]
        stats = data["height_stats"]
        crops = data["heightmap_crops"]
        valid_count = int(data["valid_count"].reshape(-1)[0])
        ok = (
            bbox.shape[1] == 7
            and stats.shape[0] == bbox.shape[0]
            and crops.shape == (bbox.shape[0], 1, 128, 64)
            and valid_count == bbox.shape[0] == crops.shape[0]
            and valid_count > 0
        )
        counts.append(valid_count)
        if not ok:
            bad.append(str(path))
    except Exception as exc:
        bad.append(f"{path}: {exc!r}")
print({"files": len(counts) + len(bad), "bad": len(bad), "min_valid_count": min(counts) if counts else None})
print("\n".join(bad[:20]))
raise SystemExit(1 if bad else 0)
PY
~~~

也可以使用输入审查脚本。它还会检查标签、集合划分和跨摄像头样本：

~~~bash
python tools/audit_cross_camera_fusion_inputs.py \
  --data-root "$DATA_ROOT" \
  --feature-root "$FEATURE_ROOT" \
  --coverage-manifest /path/to/coverage_manifest.json \
  --out "$REPO/runs/input_audit.json"
~~~

### 4. 查看裁剪质量

~~~bash
python tools/export_cross_camera_crop_quality.py \
  --data-root "$DATA_ROOT" \
  --feature-root "$FEATURE_ROOT" \
  --person-split-json "$SPLIT_JSON" \
  --out-dir "$REPO/runs/crop_quality" \
  --splits train val test \
  --samples-per-split 80 \
  --samples-per-camera 8 \
  --frames-per-video 4 \
  --label-path "$LABEL_PATH" \
  --seed 1
~~~

查看：

~~~text
runs/crop_quality/index.html
runs/crop_quality/index.json
runs/crop_quality/<split>/<camera_id>/*.png
~~~

浏览器打开 index.html，优先查看带有以下标记的样本：very_few_valid_frames、mostly_empty_crop、low_crop_contrast、tiny_bbox_height。

定向查看异常和参考样本：

~~~bash
python tools/export_targeted_crop_review.py \
  --feature-root "$FEATURE_ROOT" \
  --out-dir "$REPO/runs/crop_quality_targeted" \
  --samples-per-category 16 \
  --seed 1
~~~

输出目录中的 index.html 按 valid_count、bbox 高度、检测分数和裁剪统计值分类展示。

### 5. 训练跨摄像头模型

先用单卡做冒烟测试：

~~~bash
python tools/train_cross_camera_heightmap_fusion.py \
  --data-root "$DATA_ROOT" \
  --feature-root "$FEATURE_ROOT" \
  --person-split-json "$SPLIT_JSON" \
  --out-dir "$REPO/runs/fusion_smoke" \
  --epochs 2 \
  --steps-per-epoch 2 \
  --batch-size 4 \
  --consistency-batch-size 2 \
  --track-batch-size 2 \
  --seed 1
~~~

确认输入和输出正常后，再运行完整训练：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  tools/train_cross_camera_heightmap_fusion.py \
  --data-root "$DATA_ROOT" \
  --feature-root "$FEATURE_ROOT" \
  --person-split-json "$SPLIT_JSON" \
  --out-dir "$REPO/runs/fusion_yolo26" \
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
~~~

## 五、训练结果怎么看

训练目录至少包含：

~~~text
runs/fusion_yolo26/
  checkpoint_best.pt
  results.json
~~~

快速查看核心指标：

~~~bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/path/to/runs/fusion_yolo26/results.json")
data = json.loads(path.read_text(encoding="utf-8"))
print("best_epoch:", data["best_epoch"])
print("best_val_metrics:", json.dumps(data["best_val_metrics"], ensure_ascii=False, indent=2))
print("primary_test_video_level:", json.dumps(data["primary_test_video_level"], ensure_ascii=False, indent=2))
PY
~~~

results.json 的重要字段：

- best_epoch：验证集选择出的最佳 epoch。
- best_val_metrics：最佳 epoch 的验证集指标。
- primary_test_video_level：测试集主结果。每个可用视频是一个独立样本。
- primary_test_video_level.all_pairwise_accuracy：全部视频对的排序准确率。
- primary_test_video_level.same_camera_pairwise_accuracy：同摄像头视频对准确率。
- primary_test_video_level.cross_camera_pairwise_accuracy：跨摄像头视频对准确率，通常是最需要关注的主指标。
- primary_test_video_level.spearman：预测排序与真实身高排序的 Spearman 相关系数。
- primary_test_video_level.kendall_tau：Kendall tau 排序相关系数。
- primary_test_video_level.bisect_hit_rate：按预测排序二分高低身高后的命中率。
- primary_test_video_level.height_gap_bucket_pairwise_accuracy：按真实身高差分桶后的准确率。
- history：每个 epoch 的训练损失和验证指标，可用于画训练曲线。

结果解释时，优先使用 primary_test_video_level。以下字段仅用于和历史结果对比，不能作为主指标：

- non_primary_legacy_person_camera_aggregated_split
- non_primary_strict_no_train_overlap_person_camera_aggregated

可以直接查看完整 JSON：

~~~bash
python -m json.tool /path/to/runs/fusion_yolo26/results.json | less
~~~

## 六、常见状态和处理方式

### bbox 报告中 low_coverage 很多

检查视频是否能打开、YOLO 权重是否正确、conf 是否过高，以及 min-box-height 是否过大。low_coverage 仍会生成 JSON，但正式转换前应查看对应视频的检测数量。

### NPZ 中 missing_parsing 很多

检查 bbox 文件名是否严格满足：

~~~text
<person_id>_<video_stem>.json
~~~

确认 video_stem 来自 video_filename 去掉扩展名后的结果。不要直接使用 --allow-parsing-fallback 掩盖命名错误。

### NPZ 中 missing_bg 很多

检查视频文件名中的摄像头 ID，以及背景深度文件是否存在：

~~~text
<bg_depth_root>/<camera_id>/<camera_id>_avg_depth.npy
~~~

### 训练结果为空或指标为 null

通常表示有效 NPZ 太少、标签缺失、某个集合只有一个人员，或跨摄像头配对不足。先检查 input_audit.json、转换汇总和 primary_test_video_level.pair_counts。

### CUDA 不可用

检查：

~~~bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
~~~

GPU 训练需要安装与驱动匹配的 PyTorch CUDA 构建版本。
