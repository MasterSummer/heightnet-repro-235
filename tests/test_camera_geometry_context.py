from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cross_camera_heightmap_fusion_core import (
    CrossCameraFusionRanker,
    camera_geometry_features,
    image_point_to_world_ground,
    pixels_per_meter_at_image_point,
)


class CameraGeometryContextTest(unittest.TestCase):
    def test_ground_projection_and_pixels_per_meter_are_finite(self) -> None:
        world = image_point_to_world_ground(400.0, 500.0, image_width=800, image_height=600, cam_height=2.5, pitch_deg=18.0)
        ppm = pixels_per_meter_at_image_point(400.0, 500.0, image_width=800, image_height=600, cam_height=2.5, pitch_deg=18.0)

        self.assertEqual(world.shape, (2,))
        self.assertTrue(np.all(np.isfinite(world)))
        self.assertTrue(np.isfinite(ppm))
        self.assertGreater(ppm, 1.0)

    def test_camera_geometry_features_change_with_foot_position(self) -> None:
        upper = np.array([0.25, 0.1, -0.2, 0.45, 0.325, 0.025, 0.8, 1.0, 2.0], dtype=np.float32)
        lower = np.array([0.25, 0.1, -0.45, 0.70, 0.575, 0.025, 0.8, 1.0, 2.0], dtype=np.float32)

        upper_features = camera_geometry_features(upper, "2d5_0", image_width=800, image_height=600, pitch_deg=18.0)
        lower_features = camera_geometry_features(lower, "2d5_0", image_width=800, image_height=600, pitch_deg=18.0)

        self.assertEqual(upper_features.shape, (7,))
        self.assertTrue(np.all(np.isfinite(upper_features)))
        self.assertFalse(np.allclose(upper_features, lower_features))

    def test_ranker_accepts_optional_camera_geometry_context(self) -> None:
        model = CrossCameraFusionRanker(camera_count=2, camera_geometry_dim=7)
        tabular = torch.randn(3, 9)
        crops = torch.randn(3, 1, 128, 64)
        camera_index = torch.tensor([0, 1, 0])
        camera_height = torch.tensor([2.5, 3.5, 2.5])
        geometry = torch.randn(3, 7)

        encoded = model.encode(tabular, crops, camera_index, camera_height, geometry)

        self.assertEqual(encoded.shape, (3, model.embedding_dim))


if __name__ == "__main__":
    unittest.main()
