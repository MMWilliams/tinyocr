"""Record source annotation diagnostics without changing labels or running OCR."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from shapely.geometry import Polygon

from .common import ROOT, write_json
from .intake import sha256
from .prepare_benchmark import MANIFEST

REPORT = 'results/benchmark-source-audit.json'


def audit():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding='utf-8'))
    result = {
        'schema': 1, 'manifest_sha256': sha256(ROOT / MANIFEST),
        'audit_code_sha256_lf': hashlib.sha256(
            Path(__file__).read_text(encoding='utf-8').encode()).hexdigest(),
        'upstream_image_counts': manifest['upstream_image_counts'],
        'selected_counts': manifest['selected_counts'],
        'excluded_by_source_reason': dict(Counter(
            i['id'].split('-')[0] + ':' + i['reason'] for i in manifest['excluded'])),
        'shared_source_ids': manifest['shared_source_ids'],
        'shared_decoded_pixels': manifest['shared_decoded_pixels'],
        'sources': {},
    }
    for source in manifest['selected_counts']:
        items = [i for i in manifest['items'] if i['source'] == source]
        totals = Counter({key: 0 for key in (
            'regions', 'legible_regions', 'complete_reference_pages',
            'invalid_geometry', 'zero_area', 'out_of_frame')})
        geometry_issues = []
        for item in items:
            path = ROOT / item['annotations']
            if sha256(path) != item['annotation_sha256']:
                raise ValueError('Benchmark annotation changed')
            annotation = json.loads(path.read_text(encoding='utf-8'))
            totals['regions'] += len(annotation['regions'])
            totals['legible_regions'] += sum(r['legible'] for r in annotation['regions'])
            totals['complete_reference_pages'] += annotation['complete_reference']
            for index, region in enumerate(annotation['regions']):
                poly = Polygon(region['polygon'])
                totals['invalid_geometry'] += not poly.is_valid
                totals['zero_area'] += poly.area <= 0
                totals['out_of_frame'] += any(
                    x < 0 or y < 0 or x > item['width'] or y > item['height']
                    for x, y in region['polygon'])
                if not poly.is_valid or poly.area <= 0:
                    geometry_issues.append({'id': item['id'], 'region_index': index,
                                            'text': region['text'], 'polygon': region['polygon'],
                                            'legible': region['legible'], 'area': poly.area})
        result['sources'][source] = {
            'images': len(items), **dict(totals),
            'max_pixels': max(i['width'] * i['height'] for i in items),
            'orientations': dict(Counter(str(i['source_exif_orientation']) for i in items)),
            'geometry_issues': geometry_issues,
            'representative_id': items[0]['id'],
        }
    return result


def main():
    result = audit()
    path = ROOT / REPORT
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != result:
            raise ValueError('Benchmark source audit changed')
        print('Source annotation audit verified; no inference or label changes')
    else:
        write_json(path, result)
        print('Source annotation audit saved; no inference or label changes')
    for source, values in result['sources'].items():
        print(source, {k: v for k, v in values.items() if k != 'geometry_issues'})


if __name__ == '__main__':
    main()
