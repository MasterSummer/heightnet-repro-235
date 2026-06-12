from __future__ import annotations

import unittest

import numpy as np

from src.heightnet.runtime_seg import select_largest_person_mask


class RuntimeSegTests(unittest.TestCase):
    def test_select_largest_person_mask_picks_largest_instance(self) -> None:
        small = np.zeros((8, 8), dtype=np.float32)
        small[0:2, 0:2] = 1.0

        large = np.zeros((8, 8), dtype=np.float32)
        large[2:6, 2:7] = 1.0

        out = select_largest_person_mask(np.stack([small, large], axis=0), 8, 8)

        self.assertEqual(out.shape, (8, 8))
        self.assertTrue(np.array_equal(out, (large > 0.5).astype(np.uint8)))

    def test_select_largest_person_mask_returns_zeros_when_empty(self) -> None:
        out = select_largest_person_mask(np.zeros((0, 8, 8), dtype=np.float32), 8, 8)

        self.assertEqual(out.shape, (8, 8))
        self.assertEqual(int(out.sum()), 0)


if __name__ == "__main__":
    unittest.main()
