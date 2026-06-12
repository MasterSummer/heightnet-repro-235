from __future__ import annotations

import argparse
import glob
import hashlib
import os

import cv2
import numpy as np


def process_split(
    data_root: str,
    split: str,
    model_path: str,
    conf: float,
    iou: float,
    imgsz: int,
    strict_native: bool,
    overwrite_existing: bool,
    num_shards: int,
    shard_id: int,
    device: str,
) -> int:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required for segmentation person mask generation") from e

    split_root = os.path.join(data_root, split)
    rgb_files = sorted(glob.glob(os.path.join(split_root, "rgb", "*", "*.png")))
    if not rgb_files:
        raise RuntimeError(f"no rgb frames found in split={split}, root={os.path.join(split_root, 'rgb')}")

    model = YOLO(model_path)
    count = 0
    skipped_existing = 0
    skipped_shard = 0
    for rgb_path in rgb_files:
        rel = os.path.relpath(rgb_path, os.path.join(split_root, "rgb"))
        sequence_id = rel.split(os.sep)[0]
        if num_shards > 1:
            hval = int(hashlib.md5(sequence_id.encode("utf-8")).hexdigest(), 16)
            if (hval % num_shards) != shard_id:
                skipped_shard += 1
                continue
        frame_stem = os.path.splitext(os.path.basename(rgb_path))[0]
        out_dir = os.path.join(split_root, "person_mask", sequence_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{frame_stem}.npy")
        if os.path.exists(out_path) and not overwrite_existing:
            skipped_existing += 1
            continue

        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        res = model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=[0],  # person class in COCO
            retina_masks=True,  # return masks aligned to original image size
            verbose=False,
            device=device,
        )
        person_mask = np.zeros((h, w), dtype=np.uint8)
        if res and len(res) > 0:
            r = res[0]
            if getattr(r, "masks", None) is not None and r.masks.data is not None:
                masks = r.masks.data.detach().cpu().numpy()
                for m in masks:
                    m_u8 = (m > 0.5).astype(np.uint8)
                    if m_u8.shape != (h, w):
                        if strict_native:
                            raise RuntimeError(
                                f"mask shape mismatch for {rgb_path}: pred={m_u8.shape}, img={(h, w)}"
                            )
                        m_u8 = cv2.resize(m_u8, (w, h), interpolation=cv2.INTER_NEAREST)
                    person_mask = np.maximum(person_mask, m_u8)

        np.save(out_path, person_mask.astype(np.float32))
        count += 1
    print(
        f"[{split}] generated={count}, skipped_existing={skipped_existing}, skipped_by_shard={skipped_shard}"
    )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build person masks from segmentation model (YOLO-seg).")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--model-path", type=str, required=True, help="Path to YOLO segmentation model.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="", help='GPU device id, e.g. "0","1","2","3". Empty=auto.')
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing person_mask npy.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total parallel shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Current shard id in [0, num_shards).")
    parser.add_argument(
        "--allow-empty-split",
        action="store_true",
        help="Skip split when no RGB found instead of raising error.",
    )
    parser.add_argument(
        "--strict-native",
        action="store_true",
        help="Do not resize predicted masks in post-process. Raise error on shape mismatch.",
    )
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    total = 0
    for s in splits:
        try:
            n = process_split(
                data_root=args.data_root,
                split=s,
                model_path=args.model_path,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                strict_native=args.strict_native,
                overwrite_existing=args.overwrite_existing,
                num_shards=args.num_shards,
                shard_id=args.shard_id,
                device=args.device,
            )
        except RuntimeError as e:
            if args.allow_empty_split and "no rgb frames found" in str(e):
                print(f"[skip] {e}")
                continue
            raise
        total += n
        print(f"[{s}] person_mask(seg) generated: {n}")
    print(f"Done. total={total}")


if __name__ == "__main__":
    main()
