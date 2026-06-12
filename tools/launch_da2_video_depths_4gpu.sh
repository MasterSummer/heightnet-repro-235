#!/usr/bin/env bash
set -euo pipefail

VIDEO_ROOT="${VIDEO_ROOT:-/data2/dataset/jianzhi_2511/video2}"
DEPTHANYTHING_ROOT="${DEPTHANYTHING_ROOT:-/home/zyding/height/Depth-Anything-V2}"
CHECKPOINT="${CHECKPOINT:-/home/zyding/height/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/home/zyding/height/logs/da2_video_depths_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
FRAME_STEP="${FRAME_STEP:-1}"

mkdir -p "${LOG_DIR}"

for GPU in 0 1 2 3; do
  LOG_FILE="${LOG_DIR}/gpu${GPU}.log"
  CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_da2_video_depths.py"
    --video-root "${VIDEO_ROOT}"
    --depthanything-root "${DEPTHANYTHING_ROOT}"
    --checkpoint "${CHECKPOINT}"
    --encoder vitl
    --input-size 518
    --gpu 0
    --shard-index "${GPU}"
    --num-shards 4
    --frame-step "${FRAME_STEP}"
  )
  if [[ -n "${OUTPUT_ROOT}" ]]; then
    CMD+=(--output-root "${OUTPUT_ROOT}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
  echo "[launch] gpu=${GPU} pid=$! log=${LOG_FILE}"
done

echo "[logs] ${LOG_DIR}"
