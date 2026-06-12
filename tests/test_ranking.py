import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.heightnet.ranking import build_ranked_candidates, split_rank_bands
from train_derived_rank import evaluate


class RankingTests(unittest.TestCase):
    def test_build_ranked_candidates_sorts_same_camera_gallery_without_scores(self):
        query_records = [
            {
                "sequence_id": "q_seq",
                "person_id": "q_person",
                "camera_id": "cam_a",
                "frame_idx": 10,
                "feature": torch.tensor([1.0, 0.0], dtype=torch.float32),
            }
        ]
        gallery_records = [
            {
                "sequence_id": "g2_seq",
                "person_id": "g2",
                "camera_id": "cam_a",
                "frame_idx": 2,
                "feature": torch.tensor([0.2, 0.9], dtype=torch.float32),
            },
            {
                "sequence_id": "g1_seq",
                "person_id": "g1",
                "camera_id": "cam_a",
                "frame_idx": 1,
                "feature": torch.tensor([0.9, 0.0], dtype=torch.float32),
            },
            {
                "sequence_id": "g3_seq",
                "person_id": "g3",
                "camera_id": "cam_b",
                "frame_idx": 3,
                "feature": torch.tensor([1.0, 1.0], dtype=torch.float32),
            },
        ]

        ranked = build_ranked_candidates(query_records, gallery_records)

        self.assertEqual(len(ranked), 1)
        item = ranked[0]
        self.assertEqual(item["query_person_id"], "q_person")
        self.assertEqual(item["camera_id"], "cam_a")
        self.assertEqual([x["person_id"] for x in item["ranking"]], ["g1", "g2"])
        self.assertNotIn("score", item["ranking"][0])

    def test_split_rank_bands_uses_top25_middle50_bottom25(self):
        ranking = [{"person_id": f"p{i}"} for i in range(8)]

        bands = split_rank_bands(ranking)

        self.assertEqual([x["person_id"] for x in bands["upper"]], ["p0", "p1"])
        self.assertEqual([x["person_id"] for x in bands["middle"]], ["p2", "p3", "p4", "p5"])
        self.assertEqual([x["person_id"] for x in bands["lower"]], ["p6", "p7"])

    def test_build_ranked_candidates_embeds_bands_per_query(self):
        query_records = [
            {
                "sequence_id": "q_seq",
                "person_id": "q_person",
                "camera_id": "cam_a",
                "frame_idx": 10,
                "feature": torch.tensor([1.0, 0.0], dtype=torch.float32),
            }
        ]
        gallery_records = [
            {
                "sequence_id": f"g{i}_seq",
                "person_id": f"g{i}",
                "camera_id": "cam_a",
                "frame_idx": i,
                "feature": torch.tensor([1.0 - 0.1 * i, 0.0], dtype=torch.float32),
            }
            for i in range(4)
        ]

        ranked = build_ranked_candidates(query_records, gallery_records)

        self.assertIn("bands", ranked[0])
        self.assertEqual(list(ranked[0]["bands"].keys()), ["upper", "middle", "lower"])
        for band_items in ranked[0]["bands"].values():
            for item in band_items:
                self.assertNotIn("score", item)

    def test_evaluate_can_return_records_for_ranking_export(self):
        cfg = SimpleNamespace(
            eval=SimpleNamespace(frames_per_person_eval=1, frame_eval_seed=0, eval_batch_size=1),
            train=SimpleNamespace(num_workers=0),
            loss=SimpleNamespace(min_valid_pixels=1, min_valid_ratio=0.0, eps=1e-6),
            runtime_depth=SimpleNamespace(assume_inverse=False, use_ground_anchor=False),
            model=SimpleNamespace(bbox_expand_ratio=0.25),
        )
        query_records = [{"sequence_id": "q", "person_id": "qp", "camera_id": "cam", "feature": torch.tensor([1.0])}]
        gallery_records = [{"sequence_id": "g", "person_id": "gp", "camera_id": "cam", "feature": torch.tensor([1.0])}]

        with patch("train_derived_rank.sample_frame_row_indices_per_person", side_effect=[[0], [0]]), \
             patch("train_derived_rank.build_derived_frame_gallery", side_effect=[query_records, gallery_records]), \
             patch("train_derived_rank.cross_split_pairwise_metrics", return_value={"pairwise_accuracy": 1.0, "auc": 1.0, "f1": 1.0, "n_pairs_eval": 1, "n_comparisons": 1}):
            out = evaluate(
                model=SimpleNamespace(compare_encoded=lambda a, b: torch.tensor([1.0])),
                query_ds=SimpleNamespace(rows=[object()]),
                gallery_ds=SimpleNamespace(rows=[object()]),
                device=torch.device("cpu"),
                runtime_depth=None,
                segmenter=None,
                pairwise_labels={},
                cfg=cfg,
                return_records=True,
            )

        self.assertIn("summary", out)
        self.assertIn("query_records", out)
        self.assertIn("gallery_records", out)


if __name__ == "__main__":
    unittest.main()
