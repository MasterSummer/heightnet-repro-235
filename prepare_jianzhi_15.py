#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from heightnet.runtime_seg import PersonSegmenter
from heightnet.frame_filter import is_valid_person_frame


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--bg_depth_path", required=True)
    p.add_argument("--camera_height", type=float, default=2.5)
    p.add_argument("--camera_id", default="normal_2d5_0_200w_jianzhi2511")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--video_pattern", default="2d5_0_200w")
    p.add_argument("--split_seed", type=int, default=42)
    return p.parse_args()


def extract_frames(video_path, output_dir, fps=1.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Path(video_path).stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARN: cannot open {video_path}")
        return []
    vfps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if vfps <= 0 or total_frames <= 0:
        cap.release()
        return []
    interval = max(1, round(vfps / fps))
    saved = []
    idx = 0
    while True:
        frame_idx = idx * interval
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        out_file = output_dir / f"{base}__frame{frame_idx:06d}.jpg"
        if not out_file.exists():
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append((str(out_file), frame_idx))
        idx += 1
    cap.release()
    return saved


def build_pairwise_json(person_ids, height_map, camera_id):
    pairs = []
    for i, pi in enumerate(person_ids):
        for pj in person_ids[i + 1:]:
            hi = height_map.get(pi, 0)
            hj = height_map.get(pj, 0)
            if hi <= 0 or hj <= 0:
                continue
            if hi > hj:
                pairs.append({"id_i": pi, "id_j": pj, "y": 1, "camera": camera_id})
            elif hi < hj:
                pairs.append({"id_i": pj, "id_j": pi, "y": 1, "camera": camera_id})
            else:
                pairs.append({"id_i": pi, "id_j": pj, "y": 0, "camera": camera_id})
    return pairs


def precompute_depth(frame_paths, da2_root, checkpoint, encoder, input_size, gpu, overwrite=False):
    import contextlib
    import io
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if da2_root not in sys.path:
        sys.path.insert(0, da2_root)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    skipped = 0
    done = 0
    for fpath in frame_paths:
        cache_path = fpath + ".depth.npy"
        if not overwrite and os.path.exists(cache_path):
            skipped += 1
            continue
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  WARN: cannot read {fpath}")
            continue
        with torch.no_grad():
            depth = model.infer_image(img, input_size).astype(np.float32)
        np.save(cache_path, depth)
        done += 1
        if done % 50 == 0:
            print(f"  depth: {done} done, {skipped} skipped")
    print(f"  DA2 depth done: {done} computed, {skipped} skipped")


def precompute_groundmask(frame_paths, model_name, gpu, overwrite=False):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    GROUND_CLASS_IDS = {3, 6, 9, 13, 29, 52, 91, 117}

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model = model.to(device).eval()

    skipped = 0
    done = 0
    for fpath in frame_paths:
        cache_path = fpath + ".groundmask.npy"
        if not overwrite and os.path.exists(cache_path):
            skipped += 1
            continue
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            inputs = processor(images=img_rgb, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            logits_up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = logits_up.argmax(dim=1)[0].cpu().numpy()
            ground_mask = np.isin(pred, list(GROUND_CLASS_IDS)).astype(bool)
        np.save(cache_path, ground_mask)
        done += 1
        if done % 50 == 0:
            print(f"  groundmask: {done} done, {skipped} skipped")
    print(f"  SegFormer groundmask done: {done} computed, {skipped} skipped")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    frames_root = output_root / "frames"
    manifests_root = output_root / "manifests"

    frames_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(data_root / "Personnel_File.csv")
    labels.columns = [c.strip() for c in labels.columns]
    pid_col = [c for c in labels.columns if "id" in c.lower()][0]
    h_col = [c for c in labels.columns if "height" in c.lower()][0]
    labels = labels.rename(columns={pid_col: "person_id", h_col: "height_cm"})
    labels["person_id"] = labels["person_id"].astype(str).str.strip()
    height_map = dict(zip(labels["person_id"], labels["height_cm"]))

    infer_device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    segmenter = PersonSegmenter(
        model_path="/data1/zhaoyd/Depth-Anything-V2/yolov8n-seg.pt",
        conf=0.25,
        iou=0.7,
        imgsz=640,
        strict_native=True,
    )

    eligible_persons = set()
    for person_dir in sorted(data_root.iterdir()):
        if not person_dir.is_dir():
            continue
        pid = person_dir.name
        has_video = any(args.video_pattern in vf.name for vf in person_dir.glob("*.mp4")) if args.video_pattern else any(person_dir.glob("*.mp4"))
        if has_video:
            eligible_persons.add(pid)
    eligible_persons = sorted(eligible_persons & set(height_map.keys()))
    print(f"Eligible persons (have {args.video_pattern} videos): {eligible_persons}")

    np.random.seed(args.split_seed)
    np.random.shuffle(eligible_persons)
    n = len(eligible_persons)
    train_n = max(1, int(n * 0.7))
    val_n = max(1, int(n * 0.15))
    if train_n + val_n >= n:
        val_n = max(1, n - train_n - 1)
    train_persons = set(eligible_persons[:train_n])
    val_persons = set(eligible_persons[train_n:train_n + val_n])
    test_persons = set(eligible_persons[train_n + val_n:])

    print(f"Split: {len(train_persons)} train, {len(val_persons)} val, {len(test_persons)} test")
    print(f"Train: {sorted(train_persons)}")
    print(f"Val: {sorted(val_persons)}")
    print(f"Test: {sorted(test_persons)}")

    manifest_rows = []
    all_frame_paths = []

    for person_dir in sorted(data_root.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        if person_id not in height_map:
            continue

        for video_file in sorted(person_dir.glob("*.mp4")):
            if args.video_pattern and args.video_pattern not in video_file.name:
                continue

            sequence_id = f"{person_id}__{video_file.stem}"
            frame_dir = frames_root / person_id
            print(f"Extracting: {person_id} / {video_file.name}")
            frames = extract_frames(str(video_file), frame_dir, fps=args.fps)
            print(f"  -> {len(frames)} raw frames")

            retained = 0
            for frame_path, frame_idx in frames:
                img_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
                person_mask, person_bbox = segmenter.infer_batch_regions(img_tensor, infer_device)
                mask_np = (person_mask[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                bbox_np = person_bbox[0].detach().cpu().numpy()
                if not is_valid_person_frame(
                    bbox=bbox_np,
                    mask=mask_np,
                    min_bbox_w=8,
                    min_bbox_h=16,
                    min_mask_pixels=64,
                    min_mask_ratio=0.002,
                ):
                    continue
                manifest_rows.append({
                    "video_path": str(video_file),
                    "sequence_id": sequence_id,
                    "person_id": person_id,
                    "camera_id": args.camera_id,
                    "frame_start": frame_idx,
                    "frame_end": frame_idx,
                    "fps": args.fps,
                    "bg_depth_path": args.bg_depth_path,
                    "camera_height_m": args.camera_height,
                    "frame_path": frame_path,
                })
                all_frame_paths.append(frame_path)
                retained += 1
            print(f"  -> {retained} frames retained after filtering")

    if not manifest_rows:
        print("ERROR: no frames extracted and retained!")
        return

    df = pd.DataFrame(manifest_rows)
    print(f"\nTotal retained frames: {len(df)}")

    for split_name, split_persons in [("train", train_persons), ("val", val_persons), ("test", test_persons)]:
        split_df = df[df["person_id"].isin(split_persons)]
        out_path = manifests_root / f"{split_name}_manifest.csv"
        split_df.to_csv(out_path, index=False)
        print(f"  {split_name}: {len(split_df)} frames, {split_df['person_id'].nunique()} persons -> {out_path}")

    all_pids = sorted(set(r["person_id"] for r in manifest_rows))
    pairwise = build_pairwise_json(all_pids, height_map, args.camera_id)
    pairwise_path = manifests_root / "pairwise.json"
    with open(pairwise_path, "w") as f:
        json.dump(pairwise, f, indent=2)
    print(f"\nPairwise pairs: {len(pairwise)} -> {pairwise_path}")

    height_labels_path = manifests_root / "height_labels.json"
    hl = {pid: int(h) for pid, h in height_map.items() if pid in all_pids}
    with open(height_labels_path, "w") as f:
        json.dump(hl, f, indent=2)
    print(f"Height labels: {len(hl)} persons -> {height_labels_path}")

    dataset_info = {
        "camera_id": args.camera_id,
        "camera_height_m": args.camera_height,
        "split": {"train": sorted(train_persons), "val": sorted(val_persons), "test": sorted(test_persons)},
        "height_map": hl,
        "fps": args.fps,
        "bg_depth_path": args.bg_depth_path,
    }
    with open(manifests_root / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\n=== Step 1 done: frames + manifests ===")
    print(f"Frames: {frames_root}")
    print(f"Manifests: {manifests_root}")

    print("\n=== Step 2: Precomputing DA2 depth ===")
    da2_root = "/data1/zhaoyd/Depth-Anything-V2"
    da2_ckpt = "/data1/zhaoyd/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"
    if os.path.exists(da2_ckpt):
        precompute_depth(sorted(set(all_frame_paths)), da2_root=da2_root, checkpoint=da2_ckpt, encoder="vitl", input_size=518, gpu=args.gpu)
    else:
        print(f"  DA2 checkpoint not found at {da2_ckpt}, skipping depth precompute")

    print("\n=== Step 3: Precomputing SegFormer ground masks ===")
    segformer_model = "nvidia/segformer-b5-finetuned-ade-640-640"
    try:
        precompute_groundmask(sorted(set(all_frame_paths)), model_name=segformer_model, gpu=args.gpu)
    except Exception as e:
        print(f"  SegFormer precompute failed: {e}")

    print("\n=== ALL DONE ===")
    print(f"Next: create config yaml pointing to {manifests_root}")


if __name__ == "__main__":
    main()
