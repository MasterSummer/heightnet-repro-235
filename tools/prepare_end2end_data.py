from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end data preparation for HeightNet repro. "
            "Auto-generates height/valid_mask/person_mask and manifests."
        )
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--bg-depth-root", type=str, required=True)
    parser.add_argument("--camera-height-m", type=float, default=-1.0)
    parser.add_argument(
        "--camera-height-mode",
        type=str,
        default="fixed",
        choices=["fixed", "from-sequence-cm"],
        help="Use fixed camera height or infer from sequence name (e.g. 350cm_* -> 3.5m).",
    )
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--matches-root", type=str, default="data/matches")
    parser.add_argument("--fg-rel-thresh", type=float, default=0.06)
    parser.add_argument("--min-area", type=int, default=128)
    parser.add_argument(
        "--person-mask-source",
        type=str,
        default="seg",
        choices=["seg", "depth"],
        help="How to generate person_mask: segmentation model or depth-background difference.",
    )
    parser.add_argument(
        "--person-seg-model",
        type=str,
        default="",
        help="Path to YOLO segmentation model (required when --person-mask-source seg).",
    )
    parser.add_argument("--person-seg-conf", type=float, default=0.25)
    parser.add_argument("--person-seg-iou", type=float, default=0.7)
    parser.add_argument("--person-seg-imgsz", type=int, default=640)
    parser.add_argument(
        "--person-seg-strict-native",
        action="store_true",
        help="Do not resize predicted masks in post-process.",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    precompute = os.path.join(project_root, "tools", "precompute_height_labels.py")
    build_manifest = os.path.join(project_root, "tools", "build_manifest.py")
    build_person_mask_seg = os.path.join(project_root, "tools", "build_person_mask_seg.py")

    for split in ["train", "val", "test"]:
        cmd = [
            sys.executable,
            precompute,
            "--data-root",
            args.data_root,
            "--split",
            split,
            "--camera-height-mode",
            args.camera_height_mode,
            "--bg-depth-root",
            args.bg_depth_root,
            "--fg-rel-thresh",
            str(args.fg_rel_thresh),
            "--min-area",
            str(args.min_area),
            "--allow-empty-split",
        ]
        if args.camera_height_mode == "fixed":
            if args.camera_height_m <= 0:
                raise ValueError("--camera-height-m must be > 0 when --camera-height-mode=fixed")
            cmd.extend(["--camera-height-m", str(args.camera_height_m)])
        if args.person_mask_source == "seg":
            cmd.append("--skip-person-mask")
        run(
            cmd
        )

    if args.person_mask_source == "seg":
        if not args.person_seg_model:
            raise ValueError("--person-seg-model is required when --person-mask-source seg")
        run(
            [
                sys.executable,
                build_person_mask_seg,
                "--data-root",
                args.data_root,
                "--split",
                "all",
                "--model-path",
                args.person_seg_model,
                "--conf",
                str(args.person_seg_conf),
                "--iou",
                str(args.person_seg_iou),
                "--imgsz",
                str(args.person_seg_imgsz),
                *(["--strict-native"] if args.person_seg_strict_native else []),
            ]
        )

    os.makedirs(args.matches_root, exist_ok=True)
    run(
        [
            sys.executable,
            build_manifest,
            "--data-root",
            args.data_root,
            "--split",
            "all",
            "--out-dir",
            args.out_dir,
            "--bg-depth-root",
            args.bg_depth_root,
            "--allow-empty-split",
        ]
    )
    print("End-to-end data preparation done.")


if __name__ == "__main__":
    main()
