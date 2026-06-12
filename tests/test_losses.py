from __future__ import annotations

import unittest

import torch

from src.heightnet.losses import HeightNetLoss, build_rank_pairs
from src.heightnet.metrics import same_camera_quicksort_metrics


class LossTests(unittest.TestCase):
    def test_build_rank_pairs_skips_empty_or_tiny_person_masks(self) -> None:
        person_ids = ["p1", "p2", "p3"]
        camera_ids = ["cam", "cam", "cam"]
        masks = torch.zeros((3, 1, 8, 8), dtype=torch.float32)
        masks[0, 0, :4, :4] = 1.0
        masks[1, 0, 0, 0] = 1.0
        masks[2, 0, :4, :4] = 1.0
        pairwise_labels = {"cam": {("p1", "p2"): 1, ("p2", "p1"): 0, ("p1", "p3"): 1, ("p3", "p1"): 0}}

        idx_i, idx_j, y = build_rank_pairs(
            person_ids=person_ids,
            camera_ids=camera_ids,
            person_masks=masks,
            pairwise_labels=pairwise_labels,
            min_valid_pixels=4,
            min_valid_ratio=0.05,
            device=torch.device("cpu"),
        )

        self.assertEqual(idx_i.tolist(), [0])
        self.assertEqual(idx_j.tolist(), [2])
        self.assertEqual(y.tolist(), [1.0])

    def test_same_camera_quicksort_metrics_reduces_comparisons(self) -> None:
        records = [
            {"camera_id": "camA", "person_id": "p1"},
            {"camera_id": "camA", "person_id": "p2"},
            {"camera_id": "camA", "person_id": "p3"},
            {"camera_id": "camA", "person_id": "p4"},
            {"camera_id": "camB", "person_id": "p1"},
        ]
        pairwise_labels = {
            "camA": {
                ("p1", "p2"): 1,
                ("p2", "p1"): 0,
                ("p1", "p3"): 1,
                ("p3", "p1"): 0,
                ("p1", "p4"): 1,
                ("p4", "p1"): 0,
                ("p2", "p3"): 0,
                ("p3", "p2"): 1,
                ("p2", "p4"): 1,
                ("p4", "p2"): 0,
                ("p3", "p4"): 1,
                ("p4", "p3"): 0,
            }
        }
        probs = {
            (0, 1): 0.9,
            (0, 2): 0.8,
            (0, 3): 0.95,
            (1, 2): 0.2,
            (1, 3): 0.7,
            (2, 1): 0.8,
            (2, 3): 0.6,
        }
        calls = []

        def _prob(i: int, j: int) -> float:
            calls.append((i, j))
            if (i, j) in probs:
                return probs[(i, j)]
            return 1.0 - probs[(j, i)]

        out = same_camera_quicksort_metrics(
            records=records,
            pairwise_labels=pairwise_labels,
            prob_fn=_prob,
        )

        self.assertLess(out["n_comparisons"], 6)
        self.assertEqual(out["n_pairs_eval"], out["n_comparisons"])
        self.assertEqual(out["pairwise_accuracy"], 1.0)
        self.assertGreater(out["auc"], 0.99)
        self.assertGreater(out["f1"], 0.99)
        self.assertEqual(len(calls), out["n_comparisons"])

    def test_map_consistency_loss_is_nonzero_for_foreground_change(self) -> None:
        criterion = HeightNetLoss(
            lambda_rmse=1.0,
            lambda_rank=1.0,
            lambda_cons=0.1,
            eps=1e-6,
            min_valid_pixels=4,
            min_valid_ratio=0.05,
            consistency_mode="map",
        )
        pred_t = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        pred_t1 = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        pred_t[:, :, :4, :4] = 1.0
        pred_t1[:, :, :4, :4] = 2.0
        mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        mask[:, :, :4, :4] = 1.0

        loss, n_valid = criterion.consistency_loss(pred_t, pred_t1, mask, mask)

        self.assertEqual(n_valid, 1)
        self.assertGreater(float(loss.item()), 0.1)


if __name__ == "__main__":
    unittest.main()
