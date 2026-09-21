from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.convert_all_to_npz import ParsingCache, VideoRecord, filter_frame_bbox_map, process_video


class _FakeDepthModel:
    def infer_image(self, frame, input_size):
        return np.ones(frame.shape[:2], dtype=np.float32)


class ConvertAllToNpzTest(unittest.TestCase):
    def test_filter_frame_bbox_map_drops_low_score_small_and_edge_boxes(self) -> None:
        frame_map = {
            0: {"bbox_xyxy": (10.0, 10.0, 40.0, 90.0), "score": 0.8},
            1: {"bbox_xyxy": (10.0, 10.0, 40.0, 90.0), "score": 0.2},
            2: {"bbox_xyxy": (10.0, 10.0, 15.0, 30.0), "score": 0.8},
            3: {"bbox_xyxy": (0.0, 10.0, 35.0, 90.0), "score": 0.8},
        }
        args = argparse.Namespace(
            min_bbox_score=0.5,
            min_bbox_h_norm=0.3,
            min_bbox_w_norm=0.1,
            max_bbox_h_norm=0.95,
            max_bbox_w_norm=0.95,
            bbox_edge_margin_norm=0.02,
        )

        kept, reasons = filter_frame_bbox_map(frame_map, frame_w=100, frame_h=100, args=args)

        self.assertEqual(sorted(kept), [0])
        self.assertEqual(reasons["low_bbox_score"], 1)
        self.assertEqual(reasons["small_bbox_height"], 1)
        self.assertEqual(reasons["edge_clipped_bbox"], 1)

    def test_compact_valid_frames_stores_only_bbox_matched_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "videos" / "p1"
            video_root.mkdir(parents=True)
            video_path = video_root / "Coat_pants_sneakers_normal_2d5_0_200w_male_brightweak_jianzhi2511.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (16, 12),
            )
            for idx in range(4):
                writer.write(np.full((12, 16, 3), idx * 40, dtype=np.uint8))
            writer.release()

            parsing_root = root / "parsing"
            parsing_root.mkdir()
            parsing_path = parsing_root / "p1_Coat_pants_sneakers_normal_2d5_0_200w_male_brightweak_jianzhi2511.json"
            parsing_path.write_text(
                json.dumps(
                    {
                        "1": {
                            "sub_track": [
                                {
                                    "data": {
                                        "1_0": [
                                            {"frame_id": 1, "rect": [3, 2, 5, 8], "score": 0.7},
                                            {"frame_id": 3, "rect": [4, 2, 5, 8], "score": 0.8},
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            bg_root = root / "bg"
            (bg_root / "2d5_0").mkdir(parents=True)
            np.save(bg_root / "2d5_0" / "2d5_0_avg_depth.npy", np.full((12, 16), 2.0, dtype=np.float32))

            args = argparse.Namespace(
                output_root=str(root / "out"),
                skip_existing=False,
                bg_depth_root=str(bg_root),
                parsing_json_root=str(parsing_root),
                cache_persons=[],
                frames_per_video=4,
                compact_valid_frames=True,
                exact_parsing_only=True,
                input_size=16,
                dry_run=False,
            )
            record = VideoRecord(
                person_id="p1",
                video_filename=video_path.name,
                video_path=str(video_path),
                split="test",
                coat_type="Coat",
                action="normal",
            )

            status, payload = process_video(
                record,
                args,
                parsing_index={},
                parsing_cache=ParsingCache(),
                da2_model=_FakeDepthModel(),
                da2_device="cpu",
            )

            self.assertEqual(status, "ok")
            self.assertEqual(payload["valid_count"], 2)
            self.assertEqual(payload["missing_bbox_frames"], 2)
            with np.load(root / "out" / "p1" / f"{record.video_stem}.npz", allow_pickle=False) as data:
                self.assertEqual(int(data["valid_count"][0]), 2)
                self.assertEqual(data["bbox_feats"].shape, (2, 7))
                self.assertEqual(data["height_stats"].shape, (2, 8))
                self.assertEqual(data["heightmap_crops"].shape, (2, 1, 128, 64))
                self.assertGreater(float(np.abs(data["bbox_feats"]).sum()), 0.0)
                self.assertGreater(float(np.abs(data["heightmap_crops"]).sum()), 0.0)

    def test_non_compact_mode_keeps_uniform_samples_with_zero_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "videos" / "p1"
            video_root.mkdir(parents=True)
            video_path = video_root / "Coat_pants_sneakers_normal_2d5_0_200w_male_brightweak_jianzhi2511.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (16, 12),
            )
            for idx in range(4):
                writer.write(np.full((12, 16, 3), idx * 40, dtype=np.uint8))
            writer.release()

            parsing_root = root / "parsing"
            parsing_root.mkdir()
            parsing_path = parsing_root / "p1_Coat_pants_sneakers_normal_2d5_0_200w_male_brightweak_jianzhi2511.json"
            parsing_path.write_text(
                json.dumps(
                    {
                        "1": {
                            "sub_track": [
                                {
                                    "data": {
                                        "1_0": [
                                            {"frame_id": 1, "rect": [3, 2, 5, 8], "score": 0.7},
                                            {"frame_id": 3, "rect": [4, 2, 5, 8], "score": 0.8},
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            bg_root = root / "bg"
            (bg_root / "2d5_0").mkdir(parents=True)
            np.save(bg_root / "2d5_0" / "2d5_0_avg_depth.npy", np.full((12, 16), 2.0, dtype=np.float32))

            args = argparse.Namespace(
                output_root=str(root / "out"),
                skip_existing=False,
                bg_depth_root=str(bg_root),
                parsing_json_root=str(parsing_root),
                cache_persons=[],
                frames_per_video=4,
                compact_valid_frames=False,
                exact_parsing_only=True,
                input_size=16,
                dry_run=False,
            )
            record = VideoRecord(
                person_id="p1",
                video_filename=video_path.name,
                video_path=str(video_path),
                split="test",
                coat_type="Coat",
                action="normal",
            )

            status, payload = process_video(
                record,
                args,
                parsing_index={},
                parsing_cache=ParsingCache(),
                da2_model=_FakeDepthModel(),
                da2_device="cpu",
            )

            self.assertEqual(status, "ok")
            self.assertEqual(payload["valid_count"], 2)
            self.assertEqual(payload["missing_bbox_frames"], 2)
            with np.load(root / "out" / "p1" / f"{record.video_stem}.npz", allow_pickle=False) as data:
                self.assertEqual(int(data["valid_count"][0]), 2)
                self.assertEqual(data["bbox_feats"].shape, (4, 7))
                self.assertEqual(data["height_stats"].shape, (4, 8))
                self.assertEqual(data["heightmap_crops"].shape, (4, 1, 128, 64))
                self.assertTrue(np.allclose(data["bbox_feats"][0], 0.0))
                self.assertGreater(float(np.abs(data["bbox_feats"][1]).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
