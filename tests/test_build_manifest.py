from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.build_manifest import build_camera_splits, split_person_ids, write_camera_configs


class BuildManifestTests(unittest.TestCase):
    def test_split_person_ids_has_no_leakage(self) -> None:
        people = ["p1", "p2", "p3", "p4", "p5", "p6"]

        splits = split_person_ids(people, train_ratio=0.5, val_ratio=0.25, seed=7)

        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])

        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(test)
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(val.isdisjoint(test))
        self.assertEqual(train | val | test, set(people))

    def test_build_camera_splits_keeps_person_split_per_camera(self) -> None:
        rows = [
            {"person_id": "p1", "camera_id": "cam_a", "sequence_id": "p1_a_1"},
            {"person_id": "p1", "camera_id": "cam_a", "sequence_id": "p1_a_2"},
            {"person_id": "p2", "camera_id": "cam_a", "sequence_id": "p2_a_1"},
            {"person_id": "p3", "camera_id": "cam_a", "sequence_id": "p3_a_1"},
            {"person_id": "p4", "camera_id": "cam_a", "sequence_id": "p4_a_1"},
            {"person_id": "p1", "camera_id": "cam_b", "sequence_id": "p1_b_1"},
            {"person_id": "p2", "camera_id": "cam_b", "sequence_id": "p2_b_1"},
            {"person_id": "p3", "camera_id": "cam_b", "sequence_id": "p3_b_1"},
            {"person_id": "p4", "camera_id": "cam_b", "sequence_id": "p4_b_1"},
        ]

        out = build_camera_splits(rows, train_ratio=0.5, val_ratio=0.25, seed=3)

        self.assertEqual(set(out.keys()), {"cam_a", "cam_b"})
        for camera_id, split_rows in out.items():
            self.assertEqual(set(split_rows.keys()), {"train", "val", "test"})
            train_people = {row["person_id"] for row in split_rows["train"]}
            val_people = {row["person_id"] for row in split_rows["val"]}
            test_people = {row["person_id"] for row in split_rows["test"]}
            self.assertTrue(train_people)
            self.assertTrue(val_people)
            self.assertTrue(test_people)
            self.assertTrue(train_people.isdisjoint(val_people), camera_id)
            self.assertTrue(train_people.isdisjoint(test_people), camera_id)
            self.assertTrue(val_people.isdisjoint(test_people), camera_id)

    def test_write_camera_configs_points_to_camera_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.yaml"
            out_dir = root / "camera_manifests"
            out_dir.mkdir(parents=True)
            (out_dir / "cam_a").mkdir()
            for split in ("train", "val", "test"):
                pd.DataFrame([{"video_path": "/tmp/x.mp4"}]).to_csv(out_dir / "cam_a" / f"{split}_manifest.csv", index=False)
            template.write_text(
                "\n".join(
                    [
                        "seed: 42",
                        "device: cuda",
                        "paths:",
                        "  train_manifest: \"\"",
                        "  val_manifest: \"\"",
                        "  test_manifest: \"\"",
                        "  train_video_root: /old/train",
                        "  val_video_root: /old/val",
                        "  test_video_root: /old/test",
                        "  bg_depth_root: /bg",
                        "  output_dir: /runs/base",
                        "train:",
                        "  epochs: 1",
                        "  batch_size: 1",
                        "  num_workers: 0",
                        "  lr: 1.0e-4",
                        "  weight_decay: 1.0e-4",
                        "  grad_clip_norm: 1.0",
                        "  use_amp: false",
                        "data:",
                        "  image_size: [32, 32]",
                        "  normalize_rgb: true",
                        "  use_pair_consistency: true",
                        "loss:",
                        "  lambda_rmse: 1.0",
                        "  lambda_rank: 1.0",
                        "  lambda_cons: 0.1",
                        "  eps: 1.0e-6",
                        "  min_valid_pixels: 1",
                        "  min_valid_ratio: 0.0",
                        "  pairwise_json: /pairs.json",
                        "model:",
                        "  name: heightnet_tiny",
                        "  base_channels: 32",
                        "eval:",
                        "  save_visualizations: false",
                        "  vis_limit: 1",
                        "runtime_seg:",
                        "  enabled: true",
                        "  model_path: /seg.pt",
                        "  conf: 0.25",
                        "  iou: 0.7",
                        "  imgsz: 640",
                        "  strict_native: true",
                        "runtime_depth:",
                        "  enabled: false",
                        "  depthanything_root: \"\"",
                        "  encoder: vits",
                        "  checkpoint: \"\"",
                        "  input_size: 518",
                    ]
                ),
                encoding="utf-8",
            )

            config_paths = write_camera_configs(
                camera_ids=["cam_a"],
                out_root=out_dir,
                template_config=template,
                config_out_dir=root / "configs_by_camera",
                runs_root="/runs/by_camera",
            )

            cfg_text = Path(config_paths["cam_a"]).read_text(encoding="utf-8")
            self.assertIn("train_manifest:", cfg_text)
            self.assertIn("camera_manifests/cam_a/train_manifest.csv", cfg_text)
            self.assertIn("train_video_root: ''", cfg_text)
            self.assertIn("output_dir: /runs/by_camera/cam_a", cfg_text)


if __name__ == "__main__":
    unittest.main()
