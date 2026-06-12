from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from tools.filter_frame_manifest_person_present import (
    filter_frame_rows_person_present,
    person_box_is_valid,
    process_manifest,
)


class _FakeSegmenter:
    def __init__(self, boxes_by_path: dict[str, list[float]]) -> None:
        self.boxes_by_path = boxes_by_path
        self.current_frame_path = ""

    def infer_batch_regions(self, images_raw: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        _ = images_raw, device
        box = self.boxes_by_path.get(self.current_frame_path, [-1.0, -1.0, -1.0, -1.0])
        masks = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        boxes = torch.tensor([box], dtype=torch.float32)
        return masks, boxes


class FilterFrameManifestPersonPresentTests(unittest.TestCase):
    def test_person_box_is_valid(self) -> None:
        self.assertTrue(person_box_is_valid(np.asarray([1.0, 2.0, 5.0, 6.0], dtype=np.float32)))
        self.assertFalse(person_box_is_valid(np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)))
        self.assertFalse(person_box_is_valid(np.asarray([5.0, 2.0, 5.0, 6.0], dtype=np.float32)))

    def test_filter_frame_rows_person_present_keeps_only_detected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_path = root / "keep.jpg"
            drop_path = root / "drop.jpg"
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(keep_path), img)
            cv2.imwrite(str(drop_path), img)

            segmenter = _FakeSegmenter(
                {
                    str(keep_path): [2.0, 3.0, 20.0, 25.0],
                    str(drop_path): [-1.0, -1.0, -1.0, -1.0],
                }
            )
            rows = [
                {"frame_path": str(keep_path), "sequence_id": "keep"},
                {"frame_path": str(drop_path), "sequence_id": "drop"},
            ]

            original_loader = __import__("tools.filter_frame_manifest_person_present", fromlist=["load_rgb_tensor"])
            load_rgb_tensor = original_loader.load_rgb_tensor

            def _patched_loader(frame_path: str) -> torch.Tensor:
                segmenter.current_frame_path = frame_path
                return load_rgb_tensor(frame_path)

            original_loader.load_rgb_tensor = _patched_loader
            try:
                kept, dropped = filter_frame_rows_person_present(rows, segmenter, torch.device("cpu"))
            finally:
                original_loader.load_rgb_tensor = load_rgb_tensor

            self.assertEqual([row["sequence_id"] for row in kept], ["keep"])
            self.assertEqual([row["sequence_id"] for row in dropped], ["drop"])

    def test_process_manifest_writes_filtered_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_path = root / "frame.jpg"
            cv2.imwrite(str(frame_path), np.zeros((16, 16, 3), dtype=np.uint8))
            manifest = root / "train_manifest.csv"
            pd.DataFrame(
                [
                    {
                        "frame_path": str(frame_path),
                        "sequence_id": "seq1",
                        "person_id": "p1",
                        "camera_id": "cam",
                        "frame_idx": 10,
                    }
                ]
            ).to_csv(manifest, index=False)
            out_manifest = root / "out" / "train_manifest.csv"
            segmenter = _FakeSegmenter({str(frame_path): [1.0, 1.0, 8.0, 8.0]})

            original_loader = __import__("tools.filter_frame_manifest_person_present", fromlist=["load_rgb_tensor"])
            load_rgb_tensor = original_loader.load_rgb_tensor

            def _patched_loader(path: str) -> torch.Tensor:
                segmenter.current_frame_path = path
                return load_rgb_tensor(path)

            original_loader.load_rgb_tensor = _patched_loader
            try:
                report = process_manifest(manifest, out_manifest, segmenter, torch.device("cpu"))
            finally:
                original_loader.load_rgb_tensor = load_rgb_tensor

            out = pd.read_csv(out_manifest)
            self.assertEqual(len(out), 1)
            self.assertEqual(report["num_rows_out"], 1)


if __name__ == "__main__":
    unittest.main()
