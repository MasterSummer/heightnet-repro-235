from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_cross_camera_heightmap_fusion import (
    HeightRegressionHead,
    _height_regression_loss,
    _pair_targets_and_weights,
    _parse_bucket_weights,
    _weighted_bce_with_logits,
)


class NearNeighborObjectiveTest(unittest.TestCase):
    def test_soft_pair_targets_reflect_height_gap_and_bucket_weights(self) -> None:
        rows_left = [
            {"height_cm": 171.0},
            {"height_cm": 160.0},
            {"height_cm": 180.0},
        ]
        rows_right = [
            {"height_cm": 170.0},
            {"height_cm": 164.0},
            {"height_cm": 170.0},
        ]
        weights_by_bucket = _parse_bucket_weights("lt3=2.0,3to5=1.5,5to8=1.2,ge8=1.0")

        targets, weights = _pair_targets_and_weights(
            rows_left,
            rows_right,
            label_mode="soft",
            soft_temperature_cm=2.0,
            bucket_weights=weights_by_bucket,
        )

        expected = torch.sigmoid(torch.tensor([0.5, -2.0, 5.0]))
        self.assertTrue(torch.allclose(targets, expected))
        self.assertEqual(weights.tolist(), [2.0, 1.5, 1.0])

    def test_weighted_bce_normalizes_by_weight_sum(self) -> None:
        logits = torch.tensor([0.0, 0.0])
        targets = torch.tensor([1.0, 0.0])
        weights = torch.tensor([3.0, 1.0])
        loss = _weighted_bce_with_logits(logits, targets, weights)
        self.assertAlmostEqual(float(loss), 0.69314718, places=6)

    def test_height_regression_head_uses_normalized_centimeter_targets(self) -> None:
        head = HeightRegressionHead(embedding_dim=2)
        with torch.no_grad():
            head.linear.weight.zero_()
            head.linear.bias.zero_()
        embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rows = [{"height_cm": 170.0}, {"height_cm": 174.0}]

        loss = _height_regression_loss(head, embeddings, rows, height_mu=170.0, height_sd=2.0, device=torch.device("cpu"))

        self.assertAlmostEqual(float(loss), torch.nn.functional.smooth_l1_loss(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 2.0])).item())


if __name__ == "__main__":
    unittest.main()
