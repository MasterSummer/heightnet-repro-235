import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.heightnet.person_cache import (
    infer_or_load_person_regions,
    load_person_region_cache,
    person_bbox_cache_path,
    person_mask_cache_path,
)


class _DummySegmenter:
    def infer_batch_regions(self, images_raw, device):
        b, _, h, w = images_raw.shape
        masks = torch.ones((b, 1, h, w), dtype=torch.float32, device=device)
        boxes = torch.tensor([[1.0, 2.0, float(w - 1), float(h - 2)] for _ in range(b)], device=device)
        return masks, boxes


class PersonCacheTests(unittest.TestCase):
    def test_load_person_region_cache_resizes_mask_and_bbox(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = str(Path(tmpdir) / "frame.jpg")
            np.save(person_mask_cache_path(frame_path), np.ones((10, 20), dtype=np.float32))
            np.save(person_bbox_cache_path(frame_path), np.array([2.0, 3.0, 10.0, 8.0], dtype=np.float32))

            mask, bbox = load_person_region_cache(frame_path, target_h=20, target_w=40)

            self.assertEqual(mask.shape, (20, 40))
            np.testing.assert_allclose(bbox, np.array([4.0, 6.0, 20.0, 16.0], dtype=np.float32))

    def test_infer_or_load_person_regions_uses_cache_then_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cached_path = str(Path(tmpdir) / "cached.jpg")
            missing_path = str(Path(tmpdir) / "missing.jpg")
            np.save(person_mask_cache_path(cached_path), np.zeros((6, 8), dtype=np.float32))
            np.save(person_bbox_cache_path(cached_path), np.array([0.0, 0.0, 4.0, 5.0], dtype=np.float32))

            images_raw = torch.zeros((2, 3, 6, 8), dtype=torch.uint8)
            masks, boxes = infer_or_load_person_regions(
                images_raw=images_raw,
                frame_paths=[cached_path, missing_path],
                segmenter=_DummySegmenter(),
                device=torch.device("cpu"),
            )

            self.assertEqual(tuple(masks.shape), (2, 1, 6, 8))
            self.assertEqual(tuple(boxes.shape), (2, 4))
            self.assertAlmostEqual(float(masks[0].sum().item()), 0.0, places=6)
            self.assertAlmostEqual(float(masks[1].sum().item()), 48.0, places=6)


if __name__ == "__main__":
    unittest.main()
