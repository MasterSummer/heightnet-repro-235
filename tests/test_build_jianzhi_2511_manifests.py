from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("pandas", types.SimpleNamespace())

from tools.build_jianzhi_2511_manifests import _infer_camera_id


class BuildJianzhi2511ManifestsTests(unittest.TestCase):
    def test_infer_camera_id_normalizes_pixel_camera_resolution_suffix(self) -> None:
        self.assertEqual(
            _infer_camera_id(Path("/tmp/1209_woman1/foo_2d5_0_400w_female.mp4")),
            "2d5_0",
        )
        self.assertEqual(
            _infer_camera_id(Path("/tmp/1209_woman1/foo_2d5_0_200w_female.mp4")),
            "2d5_0",
        )
        self.assertEqual(
            _infer_camera_id(Path("/tmp/1209_woman1/foo_4d5_150_800w_female.mp4")),
            "4d5_150",
        )

    def test_infer_camera_id_keeps_legacy_camera_naming(self) -> None:
        self.assertEqual(
            _infer_camera_id(Path("/tmp/person/bar_300cm_inside.mp4")),
            "300cm_inside",
        )


if __name__ == "__main__":
    unittest.main()
