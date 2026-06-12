import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from heightnet.config import load_config
from train_derived_rank import _crop_person_geometry_batch, derive_height_batch


class _DummyDepth:
    def __init__(self):
        self.seen_shape = None
        self.forward_seen_shape = None

    def infer_batch(self, images_raw: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(images_raw.shape)
        b, _, h, w = images_raw.shape
        return torch.full((b, 1, h, w), 2.0, dtype=torch.float32, device=images_raw.device)

    def forward_batch(self, images_raw: torch.Tensor) -> torch.Tensor:
        self.forward_seen_shape = tuple(images_raw.shape)
        b, _, h, w = images_raw.shape
        return torch.full((b, 1, h, w), 2.0, dtype=torch.float32, device=images_raw.device)


class PersonGeometryCropTest(unittest.TestCase):
    def test_default_config_uses_025_bbox_expand_ratio(self):
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        cfg = load_config(str(cfg_path))
        self.assertAlmostEqual(float(cfg.model.bbox_expand_ratio), 0.25, places=6)

    def test_derive_height_batch_crops_original_resolution_geometry_back_to_target_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = str(Path(tmpdir) / "frame.png")
            bg_depth_path = str(Path(tmpdir) / "bg.npy")

            frame = np.zeros((12, 20, 3), dtype=np.uint8)
            frame[:, :, 1] = 127
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            bg = np.full((12, 20), 4.0, dtype=np.float32)
            np.save(bg_depth_path, bg)

            batch = {
                "image_raw": torch.zeros((1, 3, 6, 10), dtype=torch.uint8),
                "bg_depth": torch.zeros((1, 1, 6, 10), dtype=torch.float32),
                "camera_height_m": torch.tensor([2.0], dtype=torch.float32),
                "frame_path": [frame_path],
                "bg_depth_path": [bg_depth_path],
            }
            person_mask = torch.zeros((1, 1, 12, 20), dtype=torch.float32)
            person_mask[:, :, 3:9, 8:12] = 1.0
            person_bbox = torch.tensor([[8.0, 3.0, 12.0, 9.0]], dtype=torch.float32)
            runtime_depth = _DummyDepth()

            height, cropped_mask, cropped_bbox, cropped_bg = derive_height_batch(
                batch=batch,
                runtime_depth=runtime_depth,
                device=torch.device("cpu"),
                eps=1e-6,
                assume_inverse=False,
                person_mask=person_mask,
                person_bbox=person_bbox,
                crop_and_resize=True,
                crop_expand_ratio=0.25,
                use_ground_anchor=False,
            )

            self.assertEqual(runtime_depth.seen_shape, (1, 3, 12, 20))
            self.assertEqual(tuple(height.shape), (1, 1, 6, 10))
            self.assertEqual(tuple(cropped_mask.shape), (1, 1, 6, 10))
            self.assertEqual(tuple(cropped_bg.shape), (1, 1, 6, 10))
            self.assertTrue(torch.count_nonzero(cropped_mask).item() > 0)

            x1, y1, x2, y2 = cropped_bbox[0].tolist()
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, 10.0)
            self.assertLessEqual(y2, 6.0)
            self.assertGreater(x2 - x1, 0.0)
            self.assertGreater(y2 - y1, 0.0)

    def test_derive_height_batch_uses_original_bg_depth_path_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = str(Path(tmpdir) / "frame.png")
            bg_depth_path = str(Path(tmpdir) / "bg.npy")

            frame = np.zeros((12, 20, 3), dtype=np.uint8)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            bg = np.full((12, 20), 5.0, dtype=np.float32)
            np.save(bg_depth_path, bg)

            batch = {
                "image_raw": torch.zeros((1, 3, 6, 10), dtype=torch.uint8),
                "bg_depth": torch.zeros((1, 1, 6, 10), dtype=torch.float32),
                "camera_height_m": torch.tensor([2.0], dtype=torch.float32),
                "frame_path": [frame_path],
                "bg_depth_path": [bg_depth_path],
            }
            person_mask = torch.zeros((1, 1, 12, 20), dtype=torch.float32)
            person_mask[:, :, 3:9, 8:12] = 1.0
            person_bbox = torch.tensor([[8.0, 3.0, 12.0, 9.0]], dtype=torch.float32)
            runtime_depth = _DummyDepth()

            _, _, _, cropped_bg = derive_height_batch(
                batch=batch,
                runtime_depth=runtime_depth,
                device=torch.device("cpu"),
                eps=1e-6,
                assume_inverse=False,
                person_mask=person_mask,
                person_bbox=person_bbox,
                crop_and_resize=True,
                crop_expand_ratio=0.25,
                use_ground_anchor=False,
            )

            self.assertGreater(float(cropped_bg.mean().item()), 4.0)

    def test_derive_height_batch_with_depth_cache_does_not_require_original_raw_binding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = str(Path(tmpdir) / "frame.png")
            bg_depth_path = str(Path(tmpdir) / "bg.npy")
            depth_cache_path = frame_path + ".depth.npy"

            frame = np.zeros((12, 20, 3), dtype=np.uint8)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            bg = np.full((12, 20), 5.0, dtype=np.float32)
            np.save(bg_depth_path, bg)
            np.save(depth_cache_path, np.full((12, 20), 2.0, dtype=np.float32))

            batch = {
                "image_raw": torch.zeros((1, 3, 6, 10), dtype=torch.uint8),
                "bg_depth": torch.zeros((1, 1, 6, 10), dtype=torch.float32),
                "camera_height_m": torch.tensor([2.0], dtype=torch.float32),
                "frame_path": [frame_path],
                "bg_depth_path": [bg_depth_path],
            }
            person_mask = torch.zeros((1, 1, 12, 20), dtype=torch.float32)
            person_mask[:, :, 3:9, 8:12] = 1.0
            person_bbox = torch.tensor([[8.0, 3.0, 12.0, 9.0]], dtype=torch.float32)
            runtime_depth = _DummyDepth()

            height, cropped_mask, cropped_bbox, cropped_bg = derive_height_batch(
                batch=batch,
                runtime_depth=runtime_depth,
                device=torch.device("cpu"),
                eps=1e-6,
                assume_inverse=False,
                person_mask=person_mask,
                person_bbox=person_bbox,
                crop_and_resize=True,
                crop_expand_ratio=0.25,
                use_ground_anchor=False,
            )

            self.assertIsNone(runtime_depth.seen_shape)
            self.assertEqual(tuple(height.shape), (1, 1, 6, 10))
            self.assertEqual(tuple(cropped_mask.shape), (1, 1, 6, 10))
            self.assertEqual(tuple(cropped_bbox.shape), (1, 4))
            self.assertEqual(tuple(cropped_bg.shape), (1, 1, 6, 10))

    def test_derive_height_batch_train_depth_skips_depth_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = str(Path(tmpdir) / "frame.png")
            bg_depth_path = str(Path(tmpdir) / "bg.npy")
            depth_cache_path = frame_path + ".depth.npy"

            frame = np.zeros((12, 20, 3), dtype=np.uint8)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            np.save(bg_depth_path, np.full((12, 20), 5.0, dtype=np.float32))
            np.save(depth_cache_path, np.full((12, 20), 9.0, dtype=np.float32))

            batch = {
                "image_raw": torch.zeros((1, 3, 6, 10), dtype=torch.uint8),
                "bg_depth": torch.zeros((1, 1, 6, 10), dtype=torch.float32),
                "camera_height_m": torch.tensor([2.0], dtype=torch.float32),
                "frame_path": [frame_path],
                "bg_depth_path": [bg_depth_path],
            }
            runtime_depth = _DummyDepth()

            height = derive_height_batch(
                batch=batch,
                runtime_depth=runtime_depth,
                device=torch.device("cpu"),
                eps=1e-6,
                assume_inverse=False,
                crop_and_resize=False,
                use_ground_anchor=False,
                train_depth=True,
                allow_depth_cache=False,
            )

            self.assertIsNone(runtime_depth.seen_shape)
            self.assertEqual(runtime_depth.forward_seen_shape, (1, 3, 12, 20))
            self.assertGreater(float(height.mean().item()), 0.0)

    def test_crop_person_geometry_preserves_height_gradient(self):
        height = torch.arange(36, dtype=torch.float32).view(1, 1, 6, 6)
        height.requires_grad_(True)
        person_mask = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
        person_mask[:, :, 1:5, 2:5] = 1.0
        person_bbox = torch.tensor([[2.0, 1.0, 5.0, 5.0]], dtype=torch.float32)
        bg_depth = torch.ones((1, 1, 6, 6), dtype=torch.float32)

        cropped_height, _, _, _ = _crop_person_geometry_batch(
            height=height,
            person_mask=person_mask,
            person_bbox=person_bbox,
            bg_depth=bg_depth,
            target_h=6,
            target_w=6,
            crop_expand_ratio=0.0,
        )
        cropped_height.sum().backward()

        self.assertTrue(cropped_height.requires_grad)
        self.assertIsNotNone(height.grad)
        self.assertGreater(float(height.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
