from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_strict_person_split import build_person_split
from tools.train_cross_camera_heightmap_fusion import _apply_person_split


class StrictPersonSplitTest(unittest.TestCase):
    def test_default_split_is_deterministic_and_disjoint(self) -> None:
        people = [f"p{i:02d}" for i in range(31)]
        first = build_person_split(people, seed=42)
        second = build_person_split(people, seed=42)
        self.assertEqual(first, second)
        self.assertEqual({key: len(first[key]) for key in ("train", "val", "test")}, {"train": 21, "val": 5, "test": 5})
        self.assertFalse(set(first["train"]) & set(first["val"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["val"]) & set(first["test"]))

    def test_training_rows_are_filtered_by_strict_person_split(self) -> None:
        source = {
            "train": [{"person_id": "p1", "video_filename": "a"}, {"person_id": "p2", "video_filename": "a"}],
            "val": [{"person_id": "p1", "video_filename": "b"}, {"person_id": "p2", "video_filename": "b"}],
            "test": [{"person_id": "p1", "video_filename": "a"}, {"person_id": "p3", "video_filename": "a"}],
        }
        split = {"train": ["p1"], "val": ["p2"], "test": ["p3"]}
        filtered = _apply_person_split(source, split)
        self.assertEqual(filtered, {
            "train": [{"person_id": "p1", "video_filename": "a"}, {"person_id": "p1", "video_filename": "b"}],
            "val": [{"person_id": "p2", "video_filename": "a"}, {"person_id": "p2", "video_filename": "b"}],
            "test": [{"person_id": "p3", "video_filename": "a"}],
        })


if __name__ == "__main__":
    unittest.main()
