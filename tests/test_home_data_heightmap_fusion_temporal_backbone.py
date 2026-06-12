#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "home_data_heightmap_fusion_temporal_backbone",
    ROOT / "tools" / "home_data_heightmap_fusion_temporal_backbone.py",
)
tb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tb)


class TemporalBackboneFusionTest(unittest.TestCase):
    def test_parse_candidate(self):
        cfg = tb.parse_candidate("resnet18_pretrained_gated")
        self.assertEqual(cfg.backbone, "resnet18")
        self.assertTrue(cfg.pretrained)
        self.assertEqual(cfg.fusion, "gated")
        cfg = tb.parse_candidate("vit_b16_scratch_gated")
        self.assertEqual(cfg.backbone, "vit_b16")
        self.assertFalse(cfg.pretrained)

    def test_masked_frame_consistency_ignores_padding(self):
        scores = torch.tensor([[1.0, 1.0, 99.0], [2.0, 4.0, 100.0]])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        loss = tb.masked_frame_consistency_loss(scores, mask)
        self.assertAlmostEqual(float(loss), 0.5, places=5)

    def test_temporal_backbone_ranker_forward_shapes_and_gate(self):
        model = tb.TemporalBackboneRanker(tb.CandidateConfig("resnet18", False, "gated"), hidden=32)
        tabular = torch.randn(2, 3, 9)
        crops = torch.randn(2, 3, 1, 128, 64)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        out = model(tabular, crops, mask)
        self.assertEqual(tuple(out["video_score"].shape), (2,))
        self.assertEqual(tuple(out["frame_scores"].shape), (2, 3))
        self.assertEqual(tuple(out["gate"].shape), (2, 1))
        self.assertTrue(torch.all(out["gate"] >= 0))
        self.assertTrue(torch.all(out["gate"] <= 1))

    def test_model_forward_has_no_camera_argument(self):
        model = tb.TemporalBackboneRanker(tb.CandidateConfig("resnet18", False, "gated"), hidden=32)
        self.assertNotIn("camera", model.forward.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
