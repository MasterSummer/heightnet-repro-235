#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "home_data_heightmap_fusion_es",
    ROOT / "tools" / "home_data_heightmap_fusion_es.py",
)
es = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(es)


class HeightmapFusionESTest(unittest.TestCase):
    def test_early_stopper_uses_min_delta_and_patience(self):
        stopper = es.EarlyStopper(patience=2, min_delta=0.01)
        self.assertTrue(stopper.update({"val_pairwise": 0.50, "spearman": 0.1, "bisect": 0.4, "epoch": 20, "seed": 1}))
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.update({"val_pairwise": 0.505, "spearman": 0.2, "bisect": 0.4, "epoch": 40, "seed": 1}))
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.update({"val_pairwise": 0.509, "spearman": 0.3, "bisect": 0.4, "epoch": 60, "seed": 1}))
        self.assertTrue(stopper.should_stop)
        self.assertTrue(stopper.update({"val_pairwise": 0.515, "spearman": 0.1, "bisect": 0.4, "epoch": 80, "seed": 1}))
        self.assertFalse(stopper.should_stop)

    def test_selection_key_never_uses_test_metrics(self):
        low_test = {"seed": 1, "best_epoch": 40, "val_pairwise": 0.7, "val": {"spearman": 0.4, "bisect": {"hit_rate": 0.5}}, "test": {"pairwise_accuracy": 0.1}}
        high_test = {"seed": 2, "best_epoch": 40, "val_pairwise": 0.69, "val": {"spearman": 0.9, "bisect": {"hit_rate": 0.9}}, "test": {"pairwise_accuracy": 0.99}}
        self.assertIs(es.select_best_seed_result([low_test, high_test]), low_test)

    def test_new_fusion_variants_forward_shapes(self):
        x = torch.randn(4, 9)
        c = torch.randn(4, 1, 128, 64)
        for variant in ("bbox_film_crop", "bbox_quality_gated_fusion", "bbox_crop_cross_attn"):
            model = es.FusionRegressor(variant, tabular_dim=9, hidden=32)
            y = model(x, c)
            self.assertEqual(tuple(y.shape), (4,))

    def test_quality_gate_is_in_unit_interval(self):
        model = es.FusionRegressor("bbox_quality_gated_fusion", tabular_dim=9, hidden=32)
        x = torch.randn(5, 9)
        c = torch.randn(5, 1, 128, 64)
        _ = model(x, c)
        gate = model.last_gate
        self.assertIsNotNone(gate)
        self.assertTrue(torch.all(gate >= 0))
        self.assertTrue(torch.all(gate <= 1))

    def test_write_loss_history_outputs_expected_columns(self):
        rows = [{
            "camera_id": "2d5_0",
            "variant": "crop_only",
            "seed": 1,
            "epoch": 20,
            "split": "train",
            "total_loss": 1.0,
            "mse_loss": 0.8,
            "pair_loss": 0.2,
        }]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "loss.csv"
            es.write_loss_history(path, rows)
            with path.open(newline="") as f:
                loaded = list(csv.DictReader(f))
        self.assertEqual(list(loaded[0].keys()), ["camera_id", "variant", "seed", "epoch", "split", "total_loss", "mse_loss", "pair_loss"])
        self.assertEqual(loaded[0]["camera_id"], "2d5_0")


if __name__ == "__main__":
    unittest.main()
