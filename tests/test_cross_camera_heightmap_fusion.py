from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cross_camera_heightmap_fusion_core import (
    CrossCameraFusionRanker,
    camera_height_m,
    filter_labeled_rows,
    filter_strict_no_train_overlap,
    load_npz_frames,
    sample_cross_camera_triplet,
    sample_track_frame_pair,
    score_identity_consistency_loss,
    soft_copeland_scores,
)


def _write_npz(path: Path) -> None:
    bbox = np.arange(4 * 7, dtype=np.float32).reshape(4, 7)
    stats = np.zeros((4, 8), dtype=np.float32)
    stats[:, 2] = [10, 20, 99, 99]
    stats[:, 4] = [30, 40, 99, 99]
    crops = np.zeros((4, 1, 128, 64), dtype=np.float32)
    crops[0] = 1
    crops[1] = 2
    crops[2:] = 99
    np.savez(
        path,
        bbox_feats=bbox,
        height_stats=stats,
        heightmap_crops=crops,
        valid_count=np.array([2], dtype=np.int32),
        camera_id=np.array(["2d5_0"]),
        video_stem=np.array(["demo"]),
    )


class CrossCameraFusionTest(unittest.TestCase):
    def test_npz_loader_uses_p90_max_and_valid_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.npz"
            _write_npz(path)
            frames = load_npz_frames(path)
        self.assertEqual(frames.tabular.shape, (2, 9))
        self.assertEqual(frames.tabular[:, 7:].tolist(), [[10.0, 30.0], [20.0, 40.0]])
        self.assertEqual(frames.crops.shape, (2, 1, 128, 64))

    def test_comparator_is_antisymmetric(self) -> None:
        model = CrossCameraFusionRanker(camera_count=2)
        a = torch.randn(3, model.embedding_dim)
        b = torch.randn(3, model.embedding_dim)
        self.assertTrue(torch.allclose(model.compare_encoded(a, b), -model.compare_encoded(b, a)))

    def test_comparator_is_latent_score_difference(self) -> None:
        model = CrossCameraFusionRanker(camera_count=2)
        a = torch.randn(4, model.embedding_dim)
        b = torch.randn(4, model.embedding_dim)

        expected = (model.score(a) - model.score(b)).squeeze(1)

        self.assertTrue(torch.allclose(model.compare_encoded(a, b), expected))

    def test_forward_score_and_compare_ops_match_explicit_methods(self) -> None:
        model = CrossCameraFusionRanker(camera_count=2)
        a = torch.randn(4, model.embedding_dim)
        b = torch.randn(4, model.embedding_dim)

        self.assertTrue(torch.allclose(model(op="score", embeddings=a), model.score(a).squeeze(1)))
        self.assertTrue(
            torch.allclose(
                model(op="compare_encoded", encoded_a=a, encoded_b=b),
                model.compare_encoded(a, b),
            )
        )

    def test_score_identity_consistency_loss_backpropagates(self) -> None:
        model = CrossCameraFusionRanker(camera_count=2)
        embeddings = torch.randn(5, model.embedding_dim, requires_grad=True)
        person_ids = ["p1", "p1", "p2", "p3", "p3"]

        loss, group_count = score_identity_consistency_loss(model, embeddings, person_ids)
        loss.backward()

        self.assertEqual(group_count, 2)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(embeddings.grad)
        self.assertGreater(float(embeddings.grad.abs().sum().item()), 0.0)

    def test_unknown_camera_height_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot parse camera height"):
            camera_height_m("unknown")

    def test_triplet_and_track_pair_sampling(self) -> None:
        rows = [
            {"person_id": "p1", "camera_id": "2d5_0", "npz_path": "a"},
            {"person_id": "p1", "camera_id": "3d5_0", "npz_path": "b"},
            {"person_id": "p2", "camera_id": "2d5_0", "npz_path": "c"},
        ]
        a, ap, b = sample_cross_camera_triplet(rows, np.random.default_rng(1))
        self.assertEqual(a["person_id"], ap["person_id"])
        self.assertNotEqual(a["person_id"], b["person_id"])
        self.assertNotEqual(a["camera_id"], ap["camera_id"])
        i, j = sample_track_frame_pair(4, np.random.default_rng(1))
        self.assertNotEqual(i, j)

    def test_strict_subset_removes_train_people(self) -> None:
        test_rows = [{"person_id": "p1"}, {"person_id": "p2"}]
        self.assertEqual(filter_strict_no_train_overlap(test_rows, {"p1"}), [{"person_id": "p2"}])

    def test_unlabeled_rows_do_not_enter_training(self) -> None:
        rows = [{"person_id": "p1"}, {"person_id": "p2"}]
        self.assertEqual(filter_labeled_rows(rows, {"p1": 170.0}), [{"person_id": "p1", "height_cm": 170.0}])

    def test_soft_copeland_chunking_matches_direct_computation(self) -> None:
        embeddings = torch.randn(7, 5)
        compare = lambda a, b: (a[:, 0] - b[:, 0])
        chunked = soft_copeland_scores(embeddings, compare, chunk_size=3)
        direct = torch.stack([
            torch.sigmoid(embeddings[i, 0] - embeddings[:, 0]).sum() - 0.5
            for i in range(len(embeddings))
        ])
        self.assertTrue(torch.allclose(chunked, direct))


if __name__ == "__main__":
    unittest.main()
