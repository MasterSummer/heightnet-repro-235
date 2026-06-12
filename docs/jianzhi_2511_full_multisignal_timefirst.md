# jianzhi_2511 full multisignal time-first runbook

This run uses parsing JSON bbox records instead of YOLO for person localization.
YOLO remains configured only as a fallback if cache is missing, so the cache
precompute report must be checked before full training.

## 1. Build manifests

```bash
cd /data1/zhaoyd/heightnet_repro

python tools/build_jianzhi_2511_manifests.py \
  --video-root /data2/dataset/jianzhi_2511/video2 \
  --bg-depth-root /data1/dataset/jianzhi_2511_partial_5_bg_depthmap/2d5_0_200w \
  --out-dir /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests \
  --sample-fps 1.0
```

## 2. Extract sampled frames

```bash
python tools/extract_frames_from_manifest.py \
  --manifest \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/train_manifest.csv \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/val_manifest.csv \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/test_manifest.csv \
  --out-dir /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests
```

## 3. Estimate time before full preprocessing

```bash
python tools/estimate_jianzhi_2511_time.py \
  --manifest /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/train_manifest.csv \
  --parsing-json-root /data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/scanner/main_card_out \
  --config configs/jianzhi_2511_full_multisignal_timefirst.yaml \
  --out /data1/zhaoyd/jianzhi_2511_full_multisignal/time_estimate.json \
  --sample-sizes 100,500,2000
```

## 4. Precompute bbox/mask cache from parsing JSON

```bash
python tools/precompute_jianzhi_parsing_person_regions.py \
  --manifest \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/train_manifest.csv \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/val_manifest.csv \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/test_manifest.csv \
  --parsing-json-root /data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/scanner/main_card_out \
  --out-report /data1/zhaoyd/jianzhi_2511_full_multisignal/person_region_cache_report.json
```

Check `missing_record`, `invalid_bbox`, and `empty_mask` before training.

## 5. Visual QA

```bash
python tools/visualize_jianzhi_person_cache.py \
  --manifest /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/train_manifest.csv \
  --out-dir /data1/zhaoyd/jianzhi_2511_full_multisignal/vis_person_cache \
  --limit 50
```

## 6. Optional DA2 depth cache

This avoids online DA2 inference during training.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/precompute_depth_from_frame_manifest.py \
  --config configs/jianzhi_2511_full_multisignal_timefirst.yaml \
  --manifest \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/train_manifest.csv \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/val_manifest.csv \
    /data1/zhaoyd/jianzhi_2511_full_multisignal/manifests/test_manifest.csv \
  --batch-size 4
```

## 7. Smoke then full training

Prepare smoke `pairwise.json` and `height_labels.json`, then:

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/height/bin/python train_derived_rank.py \
  --config configs/jianzhi_2511_smoke_multisignal_timefirst.yaml
```

After smoke passes and full labels exist:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/height/bin/torchrun \
  --standalone --nproc_per_node=4 \
  train_derived_rank.py --config configs/jianzhi_2511_full_multisignal_timefirst.yaml
```
