import unittest

import torch

from src.heightnet.losses import masked_rmse
from train_derived_rank import compute_dense_rank_loss


class DenseSupervisionTests(unittest.TestCase):
    def test_masked_rmse_is_nonzero_for_mismatched_dense_targets(self):
        pred = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        target = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        valid = torch.ones((1, 1, 4, 4), dtype=torch.float32)

        loss = masked_rmse(pred, target, valid, eps=1e-6)

        self.assertGreater(float(loss.item()), 0.0)

    def test_compute_dense_rank_loss_stays_nonzero_without_pair_labels(self):
        derived = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        target = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        valid = torch.ones((1, 1, 4, 4), dtype=torch.float32)

        loss, rank_pairs, rmse_loss = compute_dense_rank_loss(
            derived=derived,
            target_height=target,
            valid_mask=valid,
            pair_logit=None,
            pair_label=torch.empty((0,), dtype=torch.float32),
            lambda_dense=1.0,
            bce_loss=torch.nn.BCEWithLogitsLoss(),
            eps=1e-6,
        )

        self.assertEqual(rank_pairs, 0)
        self.assertGreater(float(rmse_loss.item()), 0.0)
        self.assertGreater(float(loss.item()), 0.0)

    def test_compute_dense_rank_loss_keeps_zero_loss_connected_for_empty_ddp_batch(self):
        derived = torch.zeros((1, 1, 4, 4), dtype=torch.float32, requires_grad=True)
        target = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        valid = torch.zeros((1, 1, 4, 4), dtype=torch.float32)

        loss, rank_pairs, _ = compute_dense_rank_loss(
            derived=derived,
            target_height=target,
            valid_mask=valid,
            pair_logit=None,
            pair_label=torch.empty((0,), dtype=torch.float32),
            lambda_dense=0.0,
            bce_loss=torch.nn.BCEWithLogitsLoss(),
            eps=1e-6,
        )
        loss.backward()

        self.assertEqual(rank_pairs, 0)
        self.assertIsNotNone(derived.grad)


if __name__ == "__main__":
    unittest.main()
