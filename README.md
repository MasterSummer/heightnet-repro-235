# HeightNet

HeightNet 是一个面向视频级身高排序的实验项目。主流程为：

~~~text
manifest + 视频 + 标签
        |
        v
YOLO 人体框 JSON
        |
        v
视频级 NPZ 特征
        |
        +--> 裁剪质量 HTML 审查
        |
        v
跨摄像头 heightmap fusion 训练
        |
        v
results.json + checkpoint_best.pt
~~~

完整的输入格式、运行命令、输出结构和结果查看方法请阅读 [USAGE.md](USAGE.md)。

## 快速入口

~~~bash
# 安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 生成 YOLO bbox
python tools/generate_yolo26_bboxes.py --help

# 转换 NPZ
python tools/convert_all_to_npz.py --help

# 查看裁剪质量
python tools/export_cross_camera_crop_quality.py --help

# 训练和查看结果
python tools/train_cross_camera_heightmap_fusion.py --help
python -m json.tool runs/<run_name>/results.json | less
~~~

## 目录说明

- tools/generate_yolo26_bboxes.py：从视频生成 bbox JSON。
- tools/convert_all_to_npz.py：从视频和深度数据生成 NPZ。
- tools/export_cross_camera_crop_quality.py：生成裁剪质量审查页面。
- tools/export_targeted_crop_review.py：按异常类别生成定向审查页面。
- tools/train_cross_camera_heightmap_fusion.py：训练跨摄像头身高排序模型。
- tests/：回归测试。
- requirements.txt：Python 依赖。
