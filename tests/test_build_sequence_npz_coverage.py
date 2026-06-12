import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools import build_sequence_npz_coverage as cov


class BuildSequenceNpzCoverageTest(unittest.TestCase):
    def write_csv(self, path, rows):
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['person_id', 'video_filename', 'video_path', 'split', 'coat_type', 'action'])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_includes_labeled_csv_and_unlabeled_softlink_with_status_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            soft = root / 'softlink'
            features = root / 'features'
            rects = root / 'rects'
            data.mkdir(); soft.mkdir(); features.mkdir(); rects.mkdir()

            train_video = data / 'labeled_person' / 'Coat_normal_2d5_0_200w_jianzhi2511.mp4'
            train_video.parent.mkdir(parents=True)
            train_video.write_text('video', encoding='utf-8')
            self.write_csv(data / 'train.csv', [{
                'person_id': 'labeled_person',
                'video_filename': train_video.name,
                'video_path': str(train_video),
                'split': 'train',
                'coat_type': 'Coat',
                'action': 'normal',
            }])
            self.write_csv(data / 'val.csv', [])
            self.write_csv(data / 'test.csv', [])

            rank_path = data / 'rank.json'
            rank_path.write_text('penson_id,height(cm)\nlabeled_person,175\n', encoding='utf-8')

            (features / 'labeled_person').mkdir()
            (features / 'labeled_person' / (train_video.stem + '.npz')).write_bytes(b'npz')
            (rects / f'labeled_person_{train_video.stem}.json').write_text('{}', encoding='utf-8')

            soft_video = soft / 'unlabeled_person' / 'bag' / 'Coat_bag_3d5_90_400w_jianzhi2511.mp4'
            soft_video.parent.mkdir(parents=True)
            soft_video.write_text('video', encoding='utf-8')
            (features / 'unlabeled_person').mkdir()
            (features / 'unlabeled_person' / (soft_video.stem + '.npz')).write_bytes(b'npz')

            skipped_video = soft / 'missing_person' / 'normal' / 'Coat_normal_4d5_30_200w_jianzhi2511.mp4'
            skipped_video.parent.mkdir(parents=True)
            skipped_video.write_text('video', encoding='utf-8')

            manifest = cov.build_manifest(
                train_csv=data / 'train.csv',
                val_csv=data / 'val.csv',
                test_csv=data / 'test.csv',
                rank_json=rank_path,
                softlink_root=soft,
                feature_root=features,
                rect_root=rects,
                out_path=root / 'coverage.json',
            )

            records = {r['candidate_id']: r for r in manifest['candidates']}
            labeled_key = f'csv:train:labeled_person:{train_video.stem}'
            soft_key = f'softlink:unlabeled_person:bag:{soft_video.stem}'
            skipped_key = f'softlink:missing_person:normal:{skipped_video.stem}'

            self.assertEqual(records[labeled_key]['expected_npz_path'], str(features / 'labeled_person' / (train_video.stem + '.npz')))
            self.assertEqual(records[labeled_key]['camera_id'], '2d5_0')
            self.assertTrue(records[labeled_key]['has_height_label'])
            self.assertEqual(records[labeled_key]['height_cm'], 175.0)
            self.assertEqual(records[labeled_key]['status'], 'scored')

            self.assertFalse(records[soft_key]['has_height_label'])
            self.assertEqual(records[soft_key]['height_cm'], None)
            self.assertEqual(records[soft_key]['status'], 'partial')
            self.assertIn('missing_rect_json', records[soft_key]['reasons'])

            self.assertEqual(records[skipped_key]['status'], 'skipped')
            self.assertIn('missing_expected_npz', records[skipped_key]['reasons'])
            self.assertEqual(manifest['summary']['total_labeled_candidates'], 1)
            self.assertEqual(manifest['summary']['total_softlink_candidates'], 2)
            self.assertEqual(manifest['summary']['unlabeled_count'], 2)
            self.assertEqual(manifest['summary']['status_counts']['scored'], 1)
            self.assertEqual(manifest['summary']['status_counts']['partial'], 1)
            self.assertEqual(manifest['summary']['status_counts']['skipped'], 1)


if __name__ == '__main__':
    unittest.main()
