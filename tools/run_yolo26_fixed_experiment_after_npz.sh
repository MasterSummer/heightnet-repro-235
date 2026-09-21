#!/usr/bin/env bash
set -euo pipefail

REPO=/home/zyding/height/heightnet_repro
FEATURE_ROOT=/home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8
BBOX_ROOT="$REPO/runs/yolo26_bbox_all/main_card_out"
SPLIT_JSON="$REPO/runs/cross_camera_heightmap_fusion/strict_person_split_seed1/person_split.json"
AUDIT_DIR="$REPO/runs/yolo26_npz_all/audit"
CROP_DIR="$REPO/runs/cross_camera_heightmap_fusion/crop_quality_yolo26_filtered_h008_w002_s025_min8"
TRAIN_DIR="$REPO/runs/cross_camera_heightmap_fusion/georankh_margin_yolo26_filtered_h008_w002_s025_min8_ddp_seed1"
CONVERT_PATTERN="tools/convert_all_to_npz.py --manifest /home/zyding/data/train.csv /home/zyding/data/val.csv /home/zyding/data/test.csv --parsing-json-root $BBOX_ROOT --output-root $FEATURE_ROOT"

cd "$REPO"
mkdir -p "$AUDIT_DIR" "$TRAIN_DIR"

echo "[WAIT] waiting for yolo26 NPZ conversion to finish"
while pgrep -af "$CONVERT_PATTERN" >/dev/null; do
  date
  find "$FEATURE_ROOT" -type f -name '*.npz' 2>/dev/null | wc -l
  sleep 300
done

echo "[AUDIT] validating generated NPZ bbox/crop tensors"
python3 - <<'PY'
import json
from pathlib import Path

import numpy as np

feature_root = Path("/home/zyding/height/jianzhi_2511_sequence/features_yolo26_bbox_filtered_h008_w002_s025_min8")
out_path = Path("/home/zyding/height/heightnet_repro/runs/yolo26_npz_all/audit/npz_bbox_crop_audit.json")
paths = sorted(feature_root.glob("**/*.npz"))
bad = []
row_counts = []
for path in paths:
    try:
        data = np.load(path)
        bbox = data["bbox_feats"]
        crops = data["heightmap_crops"]
        valid_count = int(data["valid_count"].reshape(-1)[0]) if "valid_count" in data.files else int(bbox.shape[0])
        bbox_sum = float(np.abs(bbox).sum())
        crop_sum = float(np.abs(crops).sum())
        ok = (
            bbox.ndim == 2
            and bbox.shape[1] == 7
            and crops.ndim == 4
            and tuple(crops.shape[1:]) == (1, 128, 64)
            and valid_count == bbox.shape[0] == crops.shape[0]
            and valid_count > 0
            and bbox_sum > 0.0
            and crop_sum > 0.0
        )
        row_counts.append(valid_count)
        if not ok:
            bad.append({
                "path": str(path),
                "bbox_shape": list(bbox.shape),
                "crop_shape": list(crops.shape),
                "valid_count": valid_count,
                "bbox_sum": bbox_sum,
                "crop_sum": crop_sum,
            })
    except Exception as exc:
        bad.append({"path": str(path), "error": repr(exc)})

summary = {
    "feature_root": str(feature_root),
    "npz_files": len(paths),
    "bad_count": len(bad),
    "min_valid_count": min(row_counts) if row_counts else None,
    "median_valid_count": float(np.median(row_counts)) if row_counts else None,
    "max_valid_count": max(row_counts) if row_counts else None,
    "bad_examples": bad[:100],
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if bad:
    raise SystemExit("NPZ bbox/crop audit failed")
PY

echo "[CROP] exporting crop quality HTML"
python3 tools/export_cross_camera_crop_quality.py \
  --feature-root "$FEATURE_ROOT" \
  --person-split-json "$SPLIT_JSON" \
  --out-dir "$CROP_DIR" \
  --splits train val test \
  --samples-per-split 80 \
  --samples-per-camera 8 \
  --frames-per-video 4 \
  --seed 1

echo "[TRAIN] starting yolo26 bbox fixed DDP experiment"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  tools/train_cross_camera_heightmap_fusion.py \
  --feature-root "$FEATURE_ROOT" \
  --person-split-json "$SPLIT_JSON" \
  --out-dir "$TRAIN_DIR" \
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
  --seed 1 \
  > "$TRAIN_DIR/train.log" 2>&1

echo "[DONE] train result: $TRAIN_DIR/results.json"
