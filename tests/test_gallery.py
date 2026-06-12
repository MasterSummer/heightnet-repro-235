from __future__ import annotations

import unittest

import torch

from src.heightnet.gallery import (
    aggregate_video_feature,
    aggregate_video_scalar,
    build_scalar_prob_fn,
    cross_split_pairwise_metrics,
    foreground_height_scores,
    uniform_frame_indices,
    uniform_sample_indices,
)


class GalleryTests(unittest.TestCase):
    def test_uniform_sample_indices_downsamples_evenly(self) -> None:
        self.assertEqual(uniform_sample_indices(0, 10), [])
        self.assertEqual(uniform_sample_indices(3, 10), [0, 1, 2])
        self.assertEqual(uniform_sample_indices(10, 4), [0, 3, 6, 9])

    def test_uniform_frame_indices_spans_video_range(self) -> None:
        self.assertEqual(uniform_frame_indices(0, 0, 4), [0])
        self.assertEqual(uniform_frame_indices(0, 7, 4), [0, 2, 5, 7])
        self.assertEqual(uniform_frame_indices(3, 5, 8), [3, 4, 5])

    def test_aggregate_video_feature_means_frame_vectors(self) -> None:
        frames = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0]),
        ]

        out = aggregate_video_feature(frames)

        self.assertTrue(torch.allclose(out, torch.tensor([3.0, 4.0])))

    def test_aggregate_video_scalar_means_frame_scores(self) -> None:
        self.assertAlmostEqual(aggregate_video_scalar([1.0, 3.0, 5.0]), 3.0)

    def test_foreground_height_scores_extracts_expected_statistics(self) -> None:
        pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        mask = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])

        out = foreground_height_scores(pred, mask)

        self.assertAlmostEqual(out["masked_avg"], (1.0 + 3.0 + 4.0) / 3.0, places=5)
        self.assertAlmostEqual(out["max"], 4.0, places=5)
        self.assertGreaterEqual(out["p99"], out["p95"])
        self.assertLessEqual(out["p99"], out["max"] + 1e-6)

    def test_cross_split_pairwise_metrics_uses_all_query_gallery_pairs(self) -> None:
        query_records = [
            {"camera_id": "cam", "person_id": "q1", "sequence_id": "q1_v1"},
            {"camera_id": "cam", "person_id": "q2", "sequence_id": "q2_v1"},
        ]
        gallery_records = [
            {"camera_id": "cam", "person_id": "g1", "sequence_id": "g1_v1"},
            {"camera_id": "cam", "person_id": "g2", "sequence_id": "g2_v1"},
            {"camera_id": "other", "person_id": "g3", "sequence_id": "g3_v1"},
        ]
        pairwise_labels = {
            "cam": {
                ("q1", "g1"): 1,
                ("g1", "q1"): 0,
                ("q1", "g2"): 0,
                ("g2", "q1"): 1,
                ("q2", "g1"): 1,
                ("g1", "q2"): 0,
                ("q2", "g2"): 1,
                ("g2", "q2"): 0,
            }
        }
        probs = {
            (0, 0): 0.9,
            (0, 1): 0.2,
            (1, 0): 0.7,
            (1, 1): 0.8,
        }

        def _prob_fn(i: int, j: int) -> float:
            return probs[(i, j)]

        out = cross_split_pairwise_metrics(
            query_records=query_records,
            gallery_records=gallery_records,
            pairwise_labels=pairwise_labels,
            prob_fn=_prob_fn,
        )

        self.assertEqual(out["n_pairs_eval"], 4)
        self.assertEqual(out["n_comparisons"], 4)
        self.assertEqual(out["pairwise_accuracy"], 1.0)
        self.assertGreater(out["auc"], 0.99)
        self.assertGreater(out["f1"], 0.99)

    def test_build_scalar_prob_fn_maps_larger_score_to_higher_probability(self) -> None:
        query_records = [{"masked_avg": 2.0}]
        gallery_records = [{"masked_avg": 1.0}, {"masked_avg": 3.0}]

        prob_fn = build_scalar_prob_fn(query_records, gallery_records, score_key="masked_avg")

        self.assertGreater(prob_fn(0, 0), 0.5)
        self.assertLess(prob_fn(0, 1), 0.5)


if __name__ == "__main__":
    unittest.main()
