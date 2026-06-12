import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from heightnet.jianzhi_parsing import (
    build_jianzhi_parsing_index,
    generate_person_region_cache_from_index,
    normalize_bbox_xyxy,
)


class JianzhiParsingCacheTest(unittest.TestCase):
    def test_normalize_bbox_accepts_xywh_and_clips_to_frame(self):
        box = normalize_bbox_xyxy(
            {"x": -5, "y": 10, "w": 35, "h": 40},
            image_width=24,
            image_height=36,
        )

        np.testing.assert_allclose(box, np.array([0.0, 10.0, 24.0, 36.0], dtype=np.float32))

    def test_build_index_reads_nested_bbox_and_parsing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "main_card_out" / "personA" / "seq1.json"
            json_path.parent.mkdir(parents=True)
            parsing_path = root / "main_card_out" / "personA" / "seq1" / "000001.png"
            parsing_path.parent.mkdir(parents=True)
            cv2.imwrite(str(parsing_path), np.ones((2, 3), dtype=np.uint8) * 255)
            json_path.write_text(
                json.dumps(
                    {
                        "video_name": "walk_400cm_inside.mp4",
                        "frames": [
                            {
                                "frame_idx": 1,
                                "pasing": {
                                    "bbox": [4, 5, 14, 25],
                                    "parsing_path": str(parsing_path),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            index = build_jianzhi_parsing_index(str(root / "main_card_out"))

            rec = index.lookup("personA__walk_400cm_inside", 1)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.person_id, "personA")
            self.assertEqual(rec.frame_idx, 1)
            self.assertEqual(rec.parsing_path, parsing_path)
            np.testing.assert_allclose(rec.bbox_xyxy, np.array([4, 5, 14, 25], dtype=np.float32))

    def test_generate_cache_pastes_parsing_crop_into_bbox_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_path = root / "frame.png"
            frame = np.zeros((8, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_path), frame)

            parsing_path = root / "parsing.png"
            parsing = np.zeros((2, 2), dtype=np.uint8)
            parsing[:, 0] = 255
            cv2.imwrite(str(parsing_path), parsing)

            index = build_jianzhi_parsing_index.from_records(
                [
                    {
                        "sequence_id": "personA__seq1",
                        "person_id": "personA",
                        "frame_idx": 3,
                        "bbox": [2, 1, 6, 5],
                        "parsing_path": str(parsing_path),
                    }
                ]
            )
            rows = [
                {"sequence_id": "personA__seq1", "person_id": "personA", "frame_idx": 3, "frame_path": str(frame_path)},
                {"sequence_id": "personA__seq1", "person_id": "personA", "frame_idx": 4, "frame_path": str(frame_path)},
            ]

            report = generate_person_region_cache_from_index(rows, index)

            self.assertEqual(report["generated"], 1)
            self.assertEqual(report["missing_record"], 1)
            bbox = np.load(str(frame_path) + ".person_bbox.npy")
            mask = np.load(str(frame_path) + ".person_mask.npy")
            np.testing.assert_allclose(bbox, np.array([2, 1, 6, 5], dtype=np.float32))
            self.assertEqual(mask.shape, (8, 10))
            self.assertGreater(int(mask[1:5, 2:6].sum()), 0)
            self.assertEqual(int(mask[:, :2].sum()), 0)

    def test_cache_skips_nested_rect_without_parsing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_path = root / "frame.png"
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_path), frame)
            json_path = root / "main_card_out" / "personA_seq1.json"
            json_path.parent.mkdir(parents=True)
            json_path.write_text(
                json.dumps(
                    {
                        "2": {
                            "sub_track": [
                                {
                                    "data": {
                                        "2_0": [
                                            {
                                                "frame_id": 188,
                                                "rect": [10, 20, 30, 40],
                                                "score": 0.7,
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            index = build_jianzhi_parsing_index(str(root / "main_card_out"))

            rec = index.lookup("personA__seq1", 188)
            self.assertIsNotNone(rec)
            self.assertIsNone(rec.parsing_path)
            np.testing.assert_allclose(rec.bbox_xyxy, np.array([10, 20, 40, 60], dtype=np.float32))

            rows = [
                {
                    "sequence_id": "personA__seq1",
                    "person_id": "personA",
                    "frame_idx": 188,
                    "frame_path": str(frame_path),
                }
            ]
            report = generate_person_region_cache_from_index(rows, index)
            self.assertEqual(report["generated"], 0)
            self.assertEqual(report["missing_parsing"], 1)
            self.assertFalse(Path(str(frame_path) + ".person_bbox.npy").exists())
            self.assertFalse(Path(str(frame_path) + ".person_mask.npy").exists())

    def test_nested_rect_resolves_sidecar_parsing_bitmap_by_track_and_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_path = root / "frame.png"
            cv2.imwrite(str(frame_path), np.zeros((80, 100, 3), dtype=np.uint8))
            json_root = root / "main_card_out"
            json_path = json_root / "personA_seq1.json"
            json_path.parent.mkdir(parents=True)
            bitmap_path = root / "parsing" / "personA_seq1" / "2" / "188.bmp"
            bitmap_path.parent.mkdir(parents=True)
            cv2.imwrite(str(bitmap_path), np.ones((4, 3), dtype=np.uint8) * 255)
            json_path.write_text(
                json.dumps(
                    {
                        "2": {
                            "sub_track": [
                                {
                                    "data": {
                                        "2_0": [
                                            {
                                                "frame_id": 188,
                                                "rect": [10, 20, 30, 40],
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            index = build_jianzhi_parsing_index(str(json_root), parsing_bitmap_root=str(root / "parsing"))
            rec = index.lookup("personA__seq1", 188)

            self.assertIsNotNone(rec)
            self.assertEqual(rec.parsing_path, bitmap_path.resolve())
            report = generate_person_region_cache_from_index(
                [{"sequence_id": "personA__seq1", "person_id": "personA", "frame_idx": 188, "frame_path": str(frame_path)}],
                index,
            )
            self.assertEqual(report["generated"], 1)
            self.assertTrue(Path(str(frame_path) + ".person_bbox.npy").exists())
            self.assertTrue(Path(str(frame_path) + ".person_mask.npy").exists())


if __name__ == "__main__":
    unittest.main()
