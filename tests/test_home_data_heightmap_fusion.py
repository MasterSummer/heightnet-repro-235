from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.home_data_heightmap_fusion import FusionRegressor, VARIANTS, aggregate_sequence_npz, variant_result_label


class HomeDataHeightmapFusionTest(unittest.TestCase):
    def test_aggregate_sequence_npz_returns_9_dim_tabular_and_mean_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.npz"
            bbox_feats = np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ],
                dtype=np.float32,
            )
            height_stats = np.array(
                [
                    [10.0, 11.0, 20.0, 13.0, 30.0, 15.0, 16.0, 17.0],
                    [110.0, 111.0, 40.0, 113.0, 50.0, 115.0, 116.0, 117.0],
                ],
                dtype=np.float32,
            )
            crops = np.stack(
                [np.ones((1, 128, 64), dtype=np.float32), np.full((1, 128, 64), 3.0, dtype=np.float32)],
                axis=0,
            )
            np.savez(path, bbox_feats=bbox_feats, height_stats=height_stats, heightmap_crops=crops, valid_count=np.asarray([2], dtype=np.int32))

            tabular, crop = aggregate_sequence_npz(path)

        self.assertEqual(tabular.shape, (9,))
        np.testing.assert_allclose(tabular, np.array([2, 3, 4, 5, 6, 7, 8, 30, 40], dtype=np.float32))
        self.assertEqual(crop.shape, (1, 128, 64))
        np.testing.assert_allclose(crop, np.full((1, 128, 64), 2.0, dtype=np.float32))

    def test_aggregate_sequence_npz_rejects_nonfinite_tabular_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq_nonfinite_tabular.npz"
            bbox_feats = np.ones((2, 7), dtype=np.float32)
            height_stats = np.zeros((2, 8), dtype=np.float32)
            height_stats[:, 2] = np.nan
            height_stats[:, 4] = [1.0, 2.0]
            crops = np.ones((2, 1, 128, 64), dtype=np.float32)
            np.savez(path, bbox_feats=bbox_feats, height_stats=height_stats, heightmap_crops=crops)

            with self.assertRaisesRegex(ValueError, "non-finite tabular aggregate"):
                aggregate_sequence_npz(path)

    def test_aggregate_sequence_npz_rejects_nonfinite_crop_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq_nonfinite_crop.npz"
            bbox_feats = np.ones((2, 7), dtype=np.float32)
            height_stats = np.zeros((2, 8), dtype=np.float32)
            height_stats[:, 2] = [1.0, 2.0]
            height_stats[:, 4] = [3.0, 4.0]
            crops = np.ones((2, 1, 128, 64), dtype=np.float32)
            crops[:, :, 0, 0] = np.inf
            np.savez(path, bbox_feats=bbox_feats, height_stats=height_stats, heightmap_crops=crops)

            with self.assertRaisesRegex(ValueError, "non-finite crop aggregate"):
                aggregate_sequence_npz(path)

    def test_fused_concat_forward_supports_batch_one_and_two(self) -> None:
        model = FusionRegressor("fused_concat", tabular_dim=9, hidden=16)
        model.eval()
        one = model(torch.zeros((1, 9), dtype=torch.float32), torch.zeros((1, 1, 128, 64), dtype=torch.float32))
        two = model(torch.zeros((2, 9), dtype=torch.float32), torch.zeros((2, 1, 128, 64), dtype=torch.float32))
        self.assertEqual(tuple(one.shape), (1,))
        self.assertEqual(tuple(two.shape), (2,))

    def test_all_requested_variants_forward(self) -> None:
        tab = torch.zeros((2, 9), dtype=torch.float32)
        crop = torch.zeros((2, 1, 128, 64), dtype=torch.float32)
        for variant in ["bbox_only", "bbox_heightstats", "crop_only", "fused_concat"]:
            with self.subTest(variant=variant):
                model = FusionRegressor(variant, tabular_dim=9, hidden=16)
                model.eval()
                out = model(tab, crop)
                self.assertEqual(tuple(out.shape), (2,))

    def test_negative_control_variants_are_supported_and_labeled(self) -> None:
        self.assertEqual(
            tuple(VARIANTS),
            (
                "bbox_only",
                "bbox_heightstats",
                "crop_only",
                "fused_concat",
                "shuffled_heightstats",
                "shuffled_crops",
            ),
        )
        for variant in ["shuffled_heightstats", "shuffled_crops"]:
            with self.subTest(variant=variant):
                model = FusionRegressor(variant, tabular_dim=9, hidden=16)
                model.eval()
                out = model(
                    torch.zeros((2, 9), dtype=torch.float32),
                    torch.zeros((2, 1, 128, 64), dtype=torch.float32),
                )
                self.assertEqual(tuple(out.shape), (2,))
                self.assertTrue(variant_result_label(variant).startswith("negative_control:"))
                self.assertIn(variant, variant_result_label(variant))



if __name__ == "__main__":
    unittest.main()

