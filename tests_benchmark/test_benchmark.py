import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tinyocr.benchmark import (check_journal, make_scorer, peak_memory, score_regions,
                               summarize_source, validate_polygons)
from tinyocr import benchmark
from tinyocr import prepare_benchmark
from tinyocr.common import write_json
from tinyocr.metrics import counts
from tinyocr.prepare_benchmark import (PRIOR, box_polygon, image_record, prior_index,
                                       region, safe_tar_member)


class OfficialScoringTests(unittest.TestCase):
    def setUp(self):
        self.scorer = make_scorer()
        self.a = region('Hello', box_polygon([0, 0, 10, 10]))
        self.b = region('World', box_polygon([20, 0, 30, 10]))

    def test_perfect_and_case_sensitive_matching(self):
        perfect = score_regions(self.scorer, [self.a, self.b], [self.a, self.b])
        self.assertEqual(perfect['tp_det_cnt'], 2)
        self.assertEqual(perfect['tp_e2e_cnt'], 2)
        self.assertEqual(self.scorer.evaluate(perfect)['E2E-Fscore'], 1)
        wrong_case = dict(self.b, text='world')
        result = score_regions(self.scorer, [self.a, self.b], [self.a, wrong_case])
        self.assertEqual(result['tp_det_cnt'], 2)
        self.assertEqual(result['tp_e2e_cnt'], 1)
        self.assertEqual(self.scorer.evaluate(result)['E2E-Fscore'], .5)

    def test_missing_extra_and_duplicate_predictions(self):
        outside = region('extra', box_polygon([100, 0, 110, 10]))
        result = score_regions(self.scorer, [self.a, self.b], [self.a, outside])
        self.assertEqual(result['num_prediction'], 2)
        self.assertEqual(result['num_groundtruth'], 2)
        self.assertEqual(result['tp_det_cnt'], 1)
        self.assertEqual(self.scorer.evaluate(result)['Det-Fscore'], .5)
        duplicated = score_regions(self.scorer, [self.a], [self.a, self.a])
        self.assertEqual(duplicated['tp_e2e_cnt'], 1)
        self.assertAlmostEqual(self.scorer.evaluate(duplicated)['E2E-Precision'], .5)
        missing = score_regions(self.scorer, [self.a], [])
        self.assertEqual(missing['tp_det_cnt'], 0)
        self.assertEqual(self.scorer.evaluate(missing)['Det-Recall'], 0)

    def test_ignored_region_does_not_reward_or_penalize_text(self):
        ignored = dict(self.b, text='', legible=False)
        result = score_regions(self.scorer, [self.a, ignored], [self.a, dict(self.b, text='anything')])
        self.assertEqual(result['num_groundtruth'], 1)
        self.assertEqual(result['num_prediction'], 1)
        self.assertEqual(result['tp_e2e_cnt'], 1)

    def test_source_granularity_cannot_be_substituted(self):
        line = region('Hello World', box_polygon([0, 0, 30, 10]))
        result = score_regions(self.scorer, [self.a, self.b], [line])
        self.assertEqual(result['tp_det_cnt'], 0)
        self.assertEqual(score_regions(self.scorer, [line], [line])['tp_e2e_cnt'], 1)

    def test_source_zero_area_region_stays_in_denominator(self):
        zero_area = region('source error', box_polygon([20, 0, 20, 10]))
        result = score_regions(self.scorer, [self.a, zero_area], [self.a])
        self.assertEqual(result['num_groundtruth'], 2)
        self.assertEqual(result['tp_det_cnt'], 1)
        self.assertEqual(self.scorer.evaluate(result)['Det-Recall'], .5)

    def test_zero_area_prediction_is_not_silently_removed(self):
        zero_area = region('native box', [[2, 0], [8, 10], [8, 10], [2, 0]])
        with self.assertLogs(level='ERROR'):
            result = score_regions(self.scorer, [self.a], [self.a, zero_area])
        self.assertEqual(result['num_prediction'], 2)
        self.assertEqual(result['tp_e2e_cnt'], 1)
        self.assertEqual(self.scorer.evaluate(result)['E2E-Precision'], .5)

    def test_incomplete_pages_excluded_from_cer_and_weighted_counts(self):
        official = score_regions(self.scorer, [self.a], [self.a])
        rows = [dict(official_counts=official, latency_ms=10, complete_reference=True,
                     normalized=counts('abcd', 'abce'), case_sensitive=counts('abcd', 'abce', False)),
                dict(official_counts=official, latency_ms=30, complete_reference=True,
                     normalized=counts('ab', 'ab'), case_sensitive=counts('ab', 'ab', False)),
                dict(official_counts=official, latency_ms=100, complete_reference=False)]
        result = summarize_source(self.scorer, rows)
        self.assertEqual(result['images'], 3)
        self.assertEqual(result['row_order_diagnostic']['complete_reference_pages'], 2)
        self.assertAlmostEqual(result['row_order_diagnostic']['normalized']['cer'], 1/6)
        self.assertEqual(result['row_order_diagnostic']['pages_excluded_for_illegible_regions'], 1)
        self.assertEqual(result['latency_ms'], {'p50': 30, 'p95': 100})


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.refs = {'one': {'id': 'one', 'source': 'textocr'}}
        self.row = {'id': 'one', 'source': 'textocr', 'model': 'v6-tiny',
                    'latency_ms': 1, 'lines': [], 'words': [], 'text': ''}

    def test_journal_complete_coverage_and_invalid_rows(self):
        path = self.root / 'rows.jsonl'
        path.write_text(json.dumps(self.row)+'\n', encoding='utf-8')
        self.assertEqual(check_journal(path, self.refs, 'v6-tiny', True), {'one'})
        for key, value in [('id', 'other'), ('model', 'v6-w8-rec'), ('latency_ms', float('nan')),
                           ('latency_ms', 0), ('text', 'not returned by any line')]:
            row = dict(self.row, **{key: value})
            path.write_text(json.dumps(row)+'\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                check_journal(path, self.refs, 'v6-tiny', True)
        path.write_text('', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            check_journal(path, self.refs, 'v6-tiny', True)

    def test_torn_and_duplicate_journals_are_not_silently_repaired(self):
        path = self.root / 'rows.jsonl'
        for value in [json.dumps(self.row), (json.dumps(self.row)+'\n')*2, '\n']:
            path.write_text(value, encoding='utf-8')
            with self.assertRaises(ValueError):
                check_journal(path, self.refs, 'v6-tiny')
            self.assertEqual(path.read_text(encoding='utf-8'), value)

    def test_invalid_word_geometry_is_rejected(self):
        for poly in [[[0, 0]], [[0, 0], [1, 1], [float('nan'), 2]]]:
            with self.assertRaises(ValueError):
                validate_polygons([{'text': 'a', 'polygon': poly}])

    def test_archive_paths_links_and_image_dimensions(self):
        for name in ('../escape.jpg', '/escape.jpg', 'C:/escape.jpg', 'a\\escape.jpg'):
            with self.assertRaises(ValueError):
                safe_tar_member(tarfile.TarInfo(name))
        link = tarfile.TarInfo('a.jpg')
        link.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError):
            safe_tar_member(link)
        oversized = tarfile.TarInfo('a.jpg')
        oversized.size = (16 << 20) + 1
        with self.assertRaises(ValueError):
            safe_tar_member(oversized)
        candidate = {'source': 'textocr', 'source_id': 'abc', 'width': 4, 'height': 2}
        stream = io.BytesIO()
        Image.new('RGB', (4, 2), 'red').save(stream, format='PNG')
        data = image_record(self.root, candidate, stream.getvalue(), True)
        self.assertEqual(data, image_record(self.root, candidate, stream.getvalue(), False))
        with self.assertRaises(ValueError):
            image_record(self.root, dict(candidate, width=3), stream.getvalue(), True)
        with patch.object(prepare_benchmark, 'PIXEL_CAP', 7):
            with self.assertRaises(ValueError):
                image_record(self.root, candidate, stream.getvalue(), True)
        (self.root / data['image']).write_bytes(b'corrupt')
        with self.assertRaises(OSError):
            image_record(self.root, candidate, stream.getvalue(), True)

    def test_prior_final_and_ambiguous_legacy_parents_are_excluded(self):
        image = self.root / 'example.png'
        Image.new('RGB', (4, 2), 'red').save(image)
        receipt = {'id': 'receipt', 'source_id': 'receipt-parent', 'domain': 'receipt-page',
                   'width': 4, 'height': 2, 'image': 'example.png'}
        scene = dict(receipt, id='scene', source_id='final-parent', domain='full-scene', role='final')
        docs = [{'items': [receipt]}, {'items': [scene]},
                {'items': [{'candidate_parent_ids': ['possible-one', 'possible-two']}]}, {'items': []}]
        for name, value in zip(PRIOR, docs):
            write_json(self.root / name, value)
        receipts, scenes, pixels, hashes = prior_index(self.root)
        self.assertEqual(receipts, {'receipt-parent'})
        self.assertEqual(scenes, {'final-parent', 'possible-one', 'possible-two'})
        self.assertEqual(len(pixels), 1)
        self.assertEqual(len(next(iter(pixels.values()))), 2)
        self.assertEqual(set(hashes), set(PRIOR))

    def test_actual_process_memory_measurement(self):
        self.assertGreater(peak_memory(), 0)

    def test_completed_model_run_does_not_infer(self):
        finish = self.root / benchmark.RUNS / 'v6-tiny-finish.json'
        finish.parent.mkdir(parents=True)
        finish.write_text('{}')
        with patch.object(benchmark, 'ROOT', self.root), \
                patch.object(benchmark, 'verify_data', return_value={'items': []}), \
                patch.object(benchmark, 'fingerprint', return_value={}), \
                patch.object(benchmark, 'verify_model') as verifier, \
                patch('tinyocr.baseline.Baseline') as model:
            benchmark.run_model('v6-tiny')
            verifier.assert_called_once()
            model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
