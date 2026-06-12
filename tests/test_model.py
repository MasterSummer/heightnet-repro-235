from __future__ import annotations

import unittest

import torch

from src.heightnet.model import DerivedHeightRanker, HeightNetTiny


class ModelTests(unittest.TestCase):
    def test_pair_heads_support_all_configured_types(self) -> None:
        image = torch.rand((2, 3, 64, 64), dtype=torch.float32)
        mask = torch.ones((2, 1, 64, 64), dtype=torch.float32)
        for comparator_type in ("conv", "resnet", "vit"):
            model = HeightNetTiny(
                base_channels=8,
                comparator_channels=8,
                comparator_type=comparator_type,
                comparator_layers=2,
                comparator_num_heads=2,
                comparator_patch_size=8,
            )
            pred = model(image)["pred_height_map"]
            feat = model.encode_person(pred, mask)
            logit = model.compare_encoded(feat[:1], feat[1:])

            self.assertEqual(tuple(pred.shape), (2, 1, 64, 64))
            self.assertEqual(tuple(feat.shape), (2, 8))
            self.assertEqual(tuple(logit.shape), (1,))

    def test_bbox_person_region_keeps_only_bbox_area(self) -> None:
        model = HeightNetTiny(
            base_channels=8,
            comparator_channels=8,
            person_region_mode="bbox",
            bbox_expand_ratio=0.0,
        )
        height = torch.arange(36, dtype=torch.float32).view(1, 1, 6, 6)
        mask = torch.ones((1, 1, 6, 6), dtype=torch.float32)
        bbox = torch.tensor([[1.0, 2.0, 4.0, 5.0]], dtype=torch.float32)

        fg = model.build_person_region(height, mask, bbox)

        expected = torch.zeros_like(height)
        expected[:, :, 2:5, 1:4] = height[:, :, 2:5, 1:4]
        self.assertTrue(torch.equal(fg, expected))

    def test_bbox_person_region_expand_ratio_expands_box(self) -> None:
        model = HeightNetTiny(
            base_channels=8,
            comparator_channels=8,
            person_region_mode="bbox",
            bbox_expand_ratio=0.5,
        )
        height = torch.ones((1, 1, 6, 6), dtype=torch.float32)
        mask = torch.ones((1, 1, 6, 6), dtype=torch.float32)
        bbox = torch.tensor([[2.0, 2.0, 4.0, 4.0]], dtype=torch.float32)

        fg = model.build_person_region(height, mask, bbox)

        expected = torch.zeros_like(height)
        expected[:, :, 1:5, 1:5] = 1.0
        self.assertTrue(torch.equal(fg, expected))

    def test_derived_height_ranker_outputs_only_pair_logits(self) -> None:
        model = DerivedHeightRanker(
            comparator_channels=8,
            comparator_type="resnet",
            comparator_layers=2,
            comparator_num_heads=2,
            comparator_patch_size=8,
        )
        height = torch.rand((2, 1, 64, 64), dtype=torch.float32)
        mask = torch.ones((2, 1, 64, 64), dtype=torch.float32)

        feat = model.encode_person(height, mask)
        out = model(
            pair_inputs=(
                height[:1],
                mask[:1],
                height[1:],
                mask[1:],
            )
        )

        self.assertEqual(tuple(feat.shape), (2, 8))
        self.assertIsNone(out["pred_height_map"])
        self.assertEqual(tuple(out["pair_logit"].shape), (1,))


if __name__ == "__main__":
    unittest.main()
