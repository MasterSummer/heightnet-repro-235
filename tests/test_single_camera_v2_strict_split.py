from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_single_camera_height_ranker_v2.py"
SPEC = importlib.util.spec_from_file_location("train_single_camera_height_ranker_v2", MODULE_PATH)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


class SingleCameraV2StrictSplitTest(unittest.TestCase):
    def test_apply_person_split_reassigns_all_rows_and_deduplicates_sequences(self):
        rows = [
            {"split": "train", "person_id": "p1", "sequence_id": "p1__a"},
            {"split": "test", "person_id": "p1", "sequence_id": "p1__b"},
            {"split": "val", "person_id": "p2", "sequence_id": "p2__c"},
            {"split": "train", "person_id": "p3", "sequence_id": "p3__d"},
            {"split": "test", "person_id": "p3", "sequence_id": "p3__d"},
            {"split": "train", "person_id": "ignored", "sequence_id": "ignored__x"},
        ]
        split = {"train": ["p1"], "val": ["p2"], "test": ["p3"]}

        reassigned = mod.apply_person_split(rows, split)

        self.assertEqual([row["sequence_id"] for row in reassigned], ["p1__a", "p1__b", "p2__c", "p3__d"])
        self.assertEqual(
            {row["sequence_id"]: row["split"] for row in reassigned},
            {"p1__a": "train", "p1__b": "train", "p2__c": "val", "p3__d": "test"},
        )

    def test_person_overlap_counts_reports_zero_for_strict_split(self):
        split = {"train": ["p1", "p2"], "val": ["p3"], "test": ["p4"]}

        self.assertEqual(
            mod.person_overlap_counts(split),
            {"train_val": 0, "train_test": 0, "val_test": 0},
        )


if __name__ == "__main__":
    unittest.main()
