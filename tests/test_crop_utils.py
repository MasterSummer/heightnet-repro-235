import unittest
import numpy as np
from heightnet.crop_utils import expand_bbox_xyxy, clamp_bbox_xyxy, resize_and_pad_map


class CropUtilsTest(unittest.TestCase):
    def test_expand_bbox_xyxy_uses_symmetric_ratio(self):
        bbox = np.array([100.0, 50.0, 140.0, 130.0], dtype=np.float32)
        expanded = expand_bbox_xyxy(bbox, expand_ratio=0.25)
        self.assertTrue(np.allclose(expanded, np.array([90.0, 30.0, 150.0, 150.0], dtype=np.float32)))

    def test_clamp_bbox_xyxy_clamps_to_image_bounds(self):
        bbox = np.array([-5.0, 10.0, 210.0, 120.0], dtype=np.float32)
        clamped = clamp_bbox_xyxy(bbox, image_w=200, image_h=100)
        self.assertTrue(np.allclose(clamped, np.array([0.0, 10.0, 200.0, 100.0], dtype=np.float32)))

    def test_resize_and_pad_map_preserves_aspect_ratio(self):
        arr = np.ones((100, 200), dtype=np.float32)
        out, meta = resize_and_pad_map(arr, target_h=200, target_w=200, pad_value=0.0)
        self.assertEqual(out.shape, (200, 200))
        self.assertEqual(meta["scaled_h"], 100)
        self.assertEqual(meta["scaled_w"], 200)
        self.assertEqual(meta["pad_top"], 50)
        self.assertEqual(meta["pad_bottom"], 50)
        self.assertEqual(meta["pad_left"], 0)
        self.assertEqual(meta["pad_right"], 0)


if __name__ == "__main__":
    unittest.main()
