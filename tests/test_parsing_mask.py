import unittest

import numpy as np

from src.heightnet.parsing_mask import (
    build_full_frame_mask_from_parsing,
    parsing_dir_from_sequence_id,
    parse_parsing_filelist,
)


class ParsingMaskTests(unittest.TestCase):
    def test_parsing_dir_from_sequence_id_joins_pid_and_video_stem(self):
        sequence_id = "1203__woman1_Coat_pants_sneakers_normal_2d5_0_400w_female_brightweak_jianzhi2511"

        self.assertEqual(
            parsing_dir_from_sequence_id(sequence_id),
            "1203_woman1_Coat_pants_sneakers_normal_2d5_0_400w_female_brightweak_jianzhi2511",
        )

    def test_parse_parsing_filelist_indexes_any_track_id_by_frame(self):
        lines = [
            "parsing/1203_woman1_Coat_pants_sneakers_normal_2d5_0_400w_female_brightweak_jianzhi2511/2/123.bmp\n",
            "parsing/1203_woman1_Coat_pants_sneakers_normal_2d5_0_400w_female_brightweak_jianzhi2511/7/124.bmp\n",
            "not/a/parsing/file.txt\n",
        ]

        index = parse_parsing_filelist(lines)
        parsing_dir = "1203_woman1_Coat_pants_sneakers_normal_2d5_0_400w_female_brightweak_jianzhi2511"

        self.assertEqual(index[parsing_dir][123], lines[0].strip())
        self.assertEqual(index[parsing_dir][124], lines[1].strip())

    def test_build_full_frame_mask_from_parsing_pastes_resized_crop_in_bbox(self):
        parsing = np.array(
            [
                [0, 10],
                [20, 0],
            ],
            dtype=np.uint8,
        )

        mask = build_full_frame_mask_from_parsing(
            parsing_mask=parsing,
            bbox_xyxy=np.array([1, 1, 5, 3], dtype=np.float32),
            image_height=5,
            image_width=7,
        )

        self.assertEqual(mask.shape, (5, 7))
        self.assertEqual(float(mask[:1].sum()), 0.0)
        self.assertEqual(float(mask[3:].sum()), 0.0)
        self.assertEqual(float(mask[:, :1].sum()), 0.0)
        self.assertEqual(float(mask[:, 5:].sum()), 0.0)
        self.assertGreater(float(mask[1:3, 1:5].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
