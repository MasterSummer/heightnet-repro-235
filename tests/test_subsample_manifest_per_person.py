import unittest

import pandas as pd

from tools.subsample_manifest_per_person import subsample_rows


class SubsampleManifestPerPersonTests(unittest.TestCase):
    def test_subsample_rows_keeps_all_people_when_target_allows(self):
        frame = pd.DataFrame(
            [
                {"person_id": "p1", "frame_idx": 1},
                {"person_id": "p1", "frame_idx": 2},
                {"person_id": "p2", "frame_idx": 1},
                {"person_id": "p2", "frame_idx": 2},
                {"person_id": "p3", "frame_idx": 1},
            ]
        )

        sampled = subsample_rows(frame, target_rows=4, seed=0)

        self.assertEqual(len(sampled), 4)
        self.assertEqual(set(sampled["person_id"].tolist()), {"p1", "p2", "p3"})

    def test_subsample_rows_respects_per_person_cap(self):
        frame = pd.DataFrame(
            [{"person_id": "p1", "frame_idx": i} for i in range(5)]
            + [{"person_id": "p2", "frame_idx": i} for i in range(5)]
        )

        sampled = subsample_rows(frame, target_rows=10, seed=0, per_person_cap=2)

        counts = sampled["person_id"].value_counts().to_dict()
        self.assertEqual(counts, {"p1": 2, "p2": 2})


if __name__ == "__main__":
    unittest.main()
