import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.home_data_heightmap_fusion import FEATURE_SCHEMA, FusionRegressor
from tools.score_softlink_heightmap_fusion import load_fusion_checkpoint, score_manifest_candidates


class ScoreSoftlinkHeightmapFusionTest(unittest.TestCase):
    def _write_npz(self, feature_root: Path, person_id: str, video_stem: str) -> None:
        out_dir = feature_root / person_id
        out_dir.mkdir(parents=True, exist_ok=True)
        bbox = np.array([
            [0.5, 0.2, -0.1, 0.6, 0.25, 0.10, 0.9],
            [0.6, 0.2, -0.2, 0.7, 0.30, 0.12, 0.8],
        ], dtype=np.float32)
        stats = np.zeros((2, 8), dtype=np.float32)
        stats[:, 2] = [1.7, 1.8]
        stats[:, 4] = [1.9, 2.0]
        crops = np.ones((2, 1, 128, 64), dtype=np.float32)
        np.savez(out_dir / f"{video_stem}.npz", bbox_feats=bbox, height_stats=stats, heightmap_crops=crops)

    def _write_checkpoint(self, ckpt_dir: Path, camera_id: str, variant: str = "fused_concat") -> Path:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model = FusionRegressor(variant, tabular_dim=len(FEATURE_SCHEMA), hidden=128)
        ckpt_path = ckpt_dir / f"{camera_id}_{variant}_best.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "mu": np.zeros(len(FEATURE_SCHEMA), dtype=np.float32),
            "sd": np.ones(len(FEATURE_SCHEMA), dtype=np.float32),
            "feature_schema": list(FEATURE_SCHEMA),
            "camera_id": camera_id,
            "variant": variant,
        }, ckpt_path)
        return ckpt_path

    def test_checkpoint_schema_validation_rejects_missing_schema(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "2d5_0_fused_concat_best.pt"
            torch.save({"model_state_dict": {}, "mu": np.zeros(9), "sd": np.ones(9), "camera_id": "2d5_0", "variant": "fused_concat"}, ckpt_path)
            with self.assertRaisesRegex(ValueError, "feature_schema"):
                load_fusion_checkpoint(ckpt_path, camera_id="2d5_0", variant="fused_concat", device="cpu")

    def test_checkpoint_mismatch_errors_report_actual_and_expected_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_path = self._write_checkpoint(root, "2d5_0", variant="fused_concat")

            with self.assertRaisesRegex(ValueError, "camera_id '2d5_0' does not match expected '2d5_1'"):
                load_fusion_checkpoint(ckpt_path, camera_id="2d5_1", variant="fused_concat", device="cpu")

            with self.assertRaisesRegex(ValueError, "variant 'fused_concat' does not match expected 'crop_only'"):
                load_fusion_checkpoint(ckpt_path, camera_id="2d5_0", variant="crop_only", device="cpu")

    def test_checkpoint_feature_schema_mismatch_reports_checkpoint_schema(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "2d5_0_fused_concat_best.pt"
            model = FusionRegressor("fused_concat", tabular_dim=len(FEATURE_SCHEMA), hidden=128)
            bad_schema = list(FEATURE_SCHEMA)[:-1] + ["bad_height_max"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "mu": np.zeros(len(FEATURE_SCHEMA), dtype=np.float32),
                "sd": np.ones(len(FEATURE_SCHEMA), dtype=np.float32),
                "feature_schema": bad_schema,
                "camera_id": "2d5_0",
                "variant": "fused_concat",
            }, ckpt_path)

            with self.assertRaisesRegex(ValueError, "feature_schema mismatch; expected .*height_max.*got .*bad_height_max"):
                load_fusion_checkpoint(ckpt_path, camera_id="2d5_0", variant="fused_concat", device="cpu")

    def test_scores_labeled_and_unlabeled_manifest_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feature_root = root / "features"
            checkpoint_dir = root / "checkpoints"
            manifest_path = root / "coverage_manifest.json"
            camera_id = "2d5_0"
            self._write_npz(feature_root, "labeled_person", "labeled_video_2d5_0")
            self._write_npz(feature_root, "unlabeled_person", "unlabeled_video_2d5_0")
            self._write_checkpoint(checkpoint_dir, camera_id)
            manifest = {
                "summary": {"total_softlink_candidates": 2},
                "candidates": [
                    {
                        "candidate_id": "labeled-1",
                        "source": "softlink",
                        "split": "softlink",
                        "person_id": "labeled_person",
                        "video_stem": "labeled_video_2d5_0",
                        "video_filename": "labeled_video_2d5_0.mp4",
                        "video_path": "/softlinks/labeled_person/labeled_video_2d5_0.mp4",
                        "group": camera_id,
                        "camera_id": camera_id,
                        "expected_npz_path": str(feature_root / "labeled_person" / "labeled_video_2d5_0.npz"),
                        "height_cm": 175.0,
                        "has_height_label": True,
                        "status": "scored",
                        "reasons": [],
                    },
                    {
                        "candidate_id": "unlabeled-1",
                        "source": "softlink",
                        "split": "softlink",
                        "person_id": "unlabeled_person",
                        "video_stem": "unlabeled_video_2d5_0",
                        "video_filename": "unlabeled_video_2d5_0.mp4",
                        "video_path": "/softlinks/unlabeled_person/unlabeled_video_2d5_0.mp4",
                        "group": camera_id,
                        "camera_id": camera_id,
                        "expected_npz_path": str(feature_root / "unlabeled_person" / "unlabeled_video_2d5_0.npz"),
                        "height_cm": None,
                        "has_height_label": False,
                        "status": "scored",
                        "reasons": [],
                    },
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            payload = score_manifest_candidates(
                checkpoint_dir=checkpoint_dir,
                coverage_manifest=manifest_path,
                feature_root=feature_root,
                variant="fused_concat",
                limit_videos=20,
                device="cpu",
            )

            entries = payload["camera_rankings"][camera_id]
            self.assertEqual(2, len(entries))
            unlabeled = next(e for e in entries if e["candidate_id"] == "unlabeled-1")
            self.assertIsNone(unlabeled["height_cm"])
            self.assertFalse(unlabeled["has_height_label"])
            self.assertEqual("scored", unlabeled["status"])
            self.assertEqual([], unlabeled["reasons"])
            self.assertEqual(camera_id, unlabeled["group"])
            self.assertEqual("/softlinks/unlabeled_person/unlabeled_video_2d5_0.mp4", unlabeled["video_path"])
            for field in ["score", "rank", "person_id", "group", "video_stem", "video_path", "camera_id", "checkpoint_path"]:
                self.assertIn(field, unlabeled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
