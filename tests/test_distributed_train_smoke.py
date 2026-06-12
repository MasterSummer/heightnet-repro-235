from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_cross_camera_heightmap_fusion import (
    _distributed_context_from_env,
    _sample_identity_consistency_rows,
)


class DistributedTrainSmokeTest(unittest.TestCase):
    def test_world_size_one_uses_non_distributed_context(self) -> None:
        ctx = _distributed_context_from_env("cpu", environ={"WORLD_SIZE": "1"})

        self.assertFalse(ctx.distributed)
        self.assertEqual(ctx.rank, 0)
        self.assertEqual(ctx.local_rank, 0)
        self.assertEqual(ctx.world_size, 1)
        self.assertEqual(ctx.device, torch.device("cpu"))
        self.assertTrue(ctx.is_main)

    def test_identity_consistency_sampler_returns_repeated_people(self) -> None:
        rows = [
            {"person_id": "p1", "camera_id": "2d5_0", "npz_path": "a"},
            {"person_id": "p1", "camera_id": "2d5_30", "npz_path": "b"},
            {"person_id": "p2", "camera_id": "2d5_0", "npz_path": "c"},
            {"person_id": "p2", "camera_id": "3d5_0", "npz_path": "d"},
            {"person_id": "p3", "camera_id": "2d5_0", "npz_path": "e"},
        ]

        sampled = _sample_identity_consistency_rows(rows, group_count=2, rng=np.random.default_rng(7))
        people = [row["person_id"] for row in sampled]

        self.assertEqual(len(sampled), 4)
        self.assertTrue(all(people.count(person_id) == 2 for person_id in set(people)))
        self.assertNotIn("p3", people)


if __name__ == "__main__":
    unittest.main()
