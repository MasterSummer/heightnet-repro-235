from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--max-points", type=int, default=1000)
    args = parser.parse_args()

    split_rgb = os.path.join(args.data_root, args.split, "rgb")
    sequences = sorted(glob.glob(os.path.join(split_rgb, "*")))

    orb = cv2.ORB_create(nfeatures=args.max_points)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for seq_dir in sequences:
        seq = os.path.basename(seq_dir)
        frames = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
        if len(frames) < 2:
            continue

        out_seq = os.path.join(args.out_root, seq)
        os.makedirs(out_seq, exist_ok=True)

        for i in range(len(frames) - 1):
            p0 = frames[i]
            p1 = frames[i + 1]
            img0 = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
            if img0 is None or img1 is None:
                continue

            kp0, des0 = orb.detectAndCompute(img0, None)
            kp1, des1 = orb.detectAndCompute(img1, None)
            if des0 is None or des1 is None:
                continue

            raw = matcher.match(des0, des1)
            raw = sorted(raw, key=lambda m: m.distance)[: args.max_points]
            if len(raw) < 8:
                continue

            pts0 = np.float32([kp0[m.queryIdx].pt for m in raw])
            pts1 = np.float32([kp1[m.trainIdx].pt for m in raw])
            if pts0.ndim != 2 or pts1.ndim != 2 or pts0.shape[1] != 2 or pts1.shape[1] != 2:
                continue
            if len(pts0) < 8 or len(pts1) < 8:
                continue

            finite = (
                np.isfinite(pts0).all(axis=1)
                & np.isfinite(pts1).all(axis=1)
            )
            pts0 = pts0[finite]
            pts1 = pts1[finite]
            if len(pts0) < 8:
                continue

            # Geometric filtering for robust consistency supervision.
            try:
                _, inlier = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 1.5, 0.99)
            except cv2.error:
                inlier = None

            if inlier is not None:
                inlier = inlier.ravel().astype(bool)
                if inlier.shape[0] == pts0.shape[0]:
                    pts0 = pts0[inlier]
                    pts1 = pts1[inlier]

            # Fallback: if geometric filter fails or becomes empty, keep best raw matches.
            if len(pts0) == 0:
                continue
            if len(pts0) > args.max_points:
                pts0 = pts0[: args.max_points]
                pts1 = pts1[: args.max_points]

            stem0 = os.path.splitext(os.path.basename(p0))[0]
            stem1 = os.path.splitext(os.path.basename(p1))[0]
            out_path = os.path.join(out_seq, f"{int(stem0):06d}_{int(stem1):06d}.npz")
            np.savez(out_path, pts0=pts0, pts1=pts1, orig_h=img0.shape[0], orig_w=img0.shape[1])

    print("Pair matches generated.")


if __name__ == "__main__":
    main()
