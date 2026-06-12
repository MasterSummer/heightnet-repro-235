from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sequence_npz_features import load_sequence_npz_features


class SequenceNpzFeatureExtractionTest(unittest.TestCase):
    def test_tabular_features_are_bbox_plus_p90_and_max_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "known.npz"
            bbox_feats = np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                ],
                dtype=np.float32,
            )
            height_stats = np.array(
                [
                    [100.0, 101.0, 190.0, 103.0, 250.0, 105.0, 106.0, 107.0],
                    [200.0, 201.0, 290.0, 203.0, 350.0, 205.0, 206.0, 207.0],
                ],
                dtype=np.float32,
            )
            np.savez(
                path,
                bbox_feats=bbox_feats,
                height_stats=height_stats,
                heightmap_crops=np.zeros((2, 1, 128, 64), dtype=np.float32),
                valid_count=np.asarray([2], dtype=np.int32),
                camera_id=np.asarray(["E01"]),
                video_stem=np.asarray(["synthetic"]),
            )

            features, crops = load_sequence_npz_features(path)

        expected = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 190.0, 250.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 290.0, 350.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(features.shape, (2, 9))
        np.testing.assert_allclose(features, expected)
        self.assertEqual(crops.shape, (2, 1, 128, 64))

    def test_wrong_stat_indices_are_detectable_by_synthetic_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "known.npz"
            np.savez(
                path,
                bbox_feats=np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], dtype=np.float32),
                height_stats=np.array([[100.0, 101.0, 190.0, 999.0, 250.0, 888.0, 777.0, 666.0]], dtype=np.float32),
                heightmap_crops=np.zeros((1, 1, 128, 64), dtype=np.float32),
                valid_count=np.asarray([1], dtype=np.int32),
            )

            features, _ = load_sequence_npz_features(path)

        self.assertEqual(features.tolist(), [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 190.0, 250.0]])
        self.assertNotIn(999.0, features[0].tolist())
        self.assertNotIn(888.0, features[0].tolist())

    def test_crop_frame_count_matches_valid_feature_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "misaligned.npz"
            np.savez(
                path,
                bbox_feats=np.zeros((2, 7), dtype=np.float32),
                height_stats=np.zeros((2, 8), dtype=np.float32),
                heightmap_crops=np.zeros((1, 1, 128, 64), dtype=np.float32),
                valid_count=np.asarray([2], dtype=np.int32),
            )

            with self.assertRaisesRegex(ValueError, "heightmap_crops frame count"):
                load_sequence_npz_features(path)


if __name__ == "__main__":
    unittest.main()
