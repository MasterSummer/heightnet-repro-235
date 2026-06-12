from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "home_data_heightmap_fusion_v2.py"
SPEC = importlib.util.spec_from_file_location("home_data_heightmap_fusion_v2", MODULE_PATH)
home_data_heightmap_fusion_v2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(home_data_heightmap_fusion_v2)


class HomeDataHeightmapFusionV2Test(unittest.TestCase):
    def test_load_sequence_npz_returns_all_frames(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seq.npz"
            bbox = np.arange(4 * 7, dtype=np.float32).reshape(4, 7)
            stats = np.zeros((4, 5), dtype=np.float32)
            stats[:, 2] = [1, 2, 3, 4]
            stats[:, 4] = [5, 6, 7, 8]
            crops = np.ones((4, 1, 128, 64), dtype=np.float32)
            np.savez(path, bbox_feats=bbox, height_stats=stats, heightmap_crops=crops)

            tabular, loaded_crops = home_data_heightmap_fusion_v2.load_sequence_npz_all_frames(path)

        self.assertEqual(tabular.shape, (4, 9))
        self.assertEqual(loaded_crops.shape, (4, 1, 128, 64))
        np.testing.assert_array_equal(tabular[:, :7], bbox)
        np.testing.assert_array_equal(tabular[:, 7], [1, 2, 3, 4])
        np.testing.assert_array_equal(tabular[:, 8], [5, 6, 7, 8])

    def test_pad_sequences_builds_mask_and_zero_padding(self):
        tabular = {
            "a": np.ones((2, 9), dtype=np.float32),
            "b": np.full((4, 9), 2, dtype=np.float32),
        }
        crops = {
            "a": np.ones((2, 1, 128, 64), dtype=np.float32),
            "b": np.full((4, 1, 128, 64), 2, dtype=np.float32),
        }

        x, c, mask = home_data_heightmap_fusion_v2.pad_sequence_batch(["a", "b"], tabular, crops)

        self.assertEqual(x.shape, (2, 4, 9))
        self.assertEqual(c.shape, (2, 4, 1, 128, 64))
        self.assertEqual(mask.tolist(), [[True, True, False, False], [True, True, True, True]])
        self.assertTrue(np.all(x[0, 2:] == 0))
        self.assertTrue(np.all(c[0, 2:] == 0))

    def test_masked_attention_ignores_padded_frames(self):
        pool = home_data_heightmap_fusion_v2.MaskedAttentionPool(dim=2)
        with torch.no_grad():
            pool.score.weight.zero_()
            pool.score.bias.zero_()
        feats = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])

        pooled, weights = pool(feats, mask)

        torch.testing.assert_close(pooled, torch.tensor([[3.0, 5.0]]))
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5, 0.0]]))

    def test_film_and_gated_fusion_shapes_without_camera_input(self):
        film = home_data_heightmap_fusion_v2.FiLMConditioner(condition_dim=4, feature_dim=6)
        features = torch.ones(2, 3, 6)
        condition = torch.ones(2, 3, 4)
        out = film(features, condition)
        self.assertEqual(out.shape, features.shape)

        fusion = home_data_heightmap_fusion_v2.GatedFusion(dim=6, n_inputs=3)
        fused, weights = fusion([out, out + 1, out + 2])
        self.assertEqual(fused.shape, out.shape)
        self.assertEqual(weights.shape, (2, 3, 3))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))

    def test_model_variants_forward_shapes_and_no_camera_argument(self):
        x = torch.randn(3, 5, 9)
        crops = torch.randn(3, 5, 1, 128, 64)
        mask = torch.ones(3, 5, dtype=torch.bool)

        for variant in home_data_heightmap_fusion_v2.VARIANTS:
            model = home_data_heightmap_fusion_v2.TemporalFusionRanker(variant=variant, hidden=16)
            out = model(x, crops, mask)
            self.assertEqual(set(out), {"score", "height", "ordinal_logits", "attention_weights"})
            self.assertEqual(out["score"].shape, (3,))
            self.assertEqual(out["height"].shape, (3,))
            self.assertEqual(out["ordinal_logits"].shape[0], 3)

    def test_checkpoint_selection_never_uses_test_metrics(self):
        results = [
            {"seed": 2, "best_epoch": 20, "val_pairwise": 0.8, "val": {"spearman": 0.3, "bisect": {"hit_rate": 0.6}}, "test": {"pairwise_accuracy": 0.99}},
            {"seed": 1, "best_epoch": 10, "val_pairwise": 0.8, "val": {"spearman": 0.5, "bisect": {"hit_rate": 0.6}}, "test": {"pairwise_accuracy": 0.10}},
        ]

        best = home_data_heightmap_fusion_v2.select_best_seed_result(results)

        self.assertEqual(best["seed"], 1)

    def test_loss_history_writer_outputs_expected_columns(self):
        rows = [
            {"camera_id": "cam", "variant": "film_attn_fusion", "seed": 1, "epoch": 10, "split": "train", "total_loss": 1.0, "mse_loss": 0.1, "pair_loss": 0.2, "ordinal_loss": 0.3},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "loss.csv"
            home_data_heightmap_fusion_v2.write_loss_history(path, rows)
            with path.open(newline="", encoding="utf-8") as f:
                loaded = list(csv.DictReader(f))

        self.assertEqual(loaded[0]["camera_id"], "cam")
        self.assertEqual(loaded[0]["epoch"], "10")
        self.assertIn("ordinal_loss", loaded[0])


if __name__ == "__main__":
    unittest.main()
