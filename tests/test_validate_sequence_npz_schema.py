from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_sequence_npz_schema import validate_npz_file


class ValidateSequenceNpzSchemaTest(unittest.TestCase):
    def test_invalid_npz_reports_schema_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(
                path,
                bbox_feats=np.zeros((2, 6), dtype=np.float32),
                height_stats=np.array(
                    [
                        [0.0, 0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, np.nan, 0.0, 3.0],
                    ],
                    dtype=np.float32,
                ),
                heightmap_crops=np.zeros((1, 1, 128, 64), dtype=np.float32),
            )

            result = validate_npz_file(path)

        self.assertFalse(result.valid)
        self.assertTrue(any("bbox_feats" in reason and "7" in reason for reason in result.reasons))
        self.assertTrue(any("frame count" in reason for reason in result.reasons))
        self.assertTrue(any("p90" in reason and "finite" in reason for reason in result.reasons))

    def test_valid_npz_passes_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.npz"
            np.savez(
                path,
                bbox_feats=np.zeros((2, 7), dtype=np.float32),
                height_stats=np.array(
                    [
                        [0.0, 0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 3.0, 0.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
                heightmap_crops=np.zeros((2, 1, 128, 64), dtype=np.float32),
            )

            result = validate_npz_file(path)

        self.assertTrue(result.valid)
        self.assertEqual(result.reasons, [])


if __name__ == "__main__":
    unittest.main()
