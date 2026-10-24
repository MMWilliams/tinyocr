"""Prepare overlap-filtered SROIE/TextOCR/HierText comparison data, never train."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

from PIL import Image

from .common import ROOT, reading_order, write_json
from .intake import download, safe_member, sha256

DATA = Path('data/benchmarks-v1')
MANIFEST = Path('eval/benchmarks-v1.json')
SOURCES = Path('results/benchmark-sources.json')
HIER_REV = '70b6620b2b112597d8219e11eee9773a1403827c'
SROIE_REV = 'bffe40c26759f3376ec2b3ae9031dbba54cd587c'
PIXEL_CAP = 40_000_000
FILES = {
    'sroie.parquet': (f'https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE/resolve/{SROIE_REV}/data/test-00000-of-00001.parquet', 256 << 20),
    'sroie-card.md': (f'https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE/raw/{SROIE_REV}/README.md', 2 << 20),
    'textocr-val.json': ('https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json', 128 << 20),
    'textocr-images.zip': ('https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip', 8 << 30),
    'textocr-source.html': ('https://textvqa.org/textocr/dataset/', 2 << 20),
    'hiertext-val.json.gz': (f'https://raw.githubusercontent.com/google-research-datasets/hiertext/{HIER_REV}/gt/validation.jsonl.gz', 128 << 20),
    'hiertext-images.tgz': ('https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz', 1 << 30),
}
VENDOR = ('__init__.py', 'evaluator.py', 'np_box_list.py', 'np_box_list_ops.py',
          'np_box_mask_list.py', 'np_box_mask_list_ops.py', 'np_box_ops.py',
          'np_mask_ops.py', 'polygon_list.py', 'polygon_ops.py')
PRIOR = ('eval/manifest.json', 'eval/expansion-region-v2.json', 'eval/legacy_scene_lineage.json',
         'eval/srd-candidates.json')


def acquire(root=ROOT, offline=False):
    records = {}
    for name, (url, cap) in FILES.items():
        if not offline:
            print('Acquiring/verifying ' + name, flush=True)
        records[name] = download(url, root / DATA / name, cap, offline=offline)
    for name in (*('evaluator/' + n for n in VENDOR), 'LICENSE', 'README.md'):
        path = Path('third_party/hiertext') / name
        records[path.as_posix()] = download(
            f'https://raw.githubusercontent.com/google-research-datasets/hiertext/{HIER_REV}/{name}',
            root / path, 2 << 20, offline=offline)
    value = {'schema': 1, 'files': records, 'hiertext_git_revision': HIER_REV,
             'sroie_git_revision': SROIE_REV,
             'rights': {'sroie': 'Mirror card: CC BY 4.0; mirror corrections retained.',
                        'textocr': 'Publisher: CC BY 4.0; underlying Open Images rights also apply.',
                        'hiertext': 'Official dataset/repository: CC BY-SA 4.0; preserve upstream attribution.'}}
    path = root / SOURCES
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Benchmark source evidence changed')
    elif offline:
        raise ValueError('Missing benchmark source evidence')
    else:
        write_json(path, value)
    return value


def prior_index(root=ROOT):
    receipts, scenes, pixels = set(), set(), defaultdict(list)
    hashes = {}
    for name in PRIOR:
        path = root / name
        hashes[name] = sha256(path)
        doc = json.loads(path.read_text(encoding='utf-8'))
        for item in doc['items']:
            if name.endswith('legacy_scene_lineage.json'):
                scenes.update(item['candidate_parent_ids'])
                continue
            if item.get('domain') == 'receipt-page':
                receipts.add(item['source_id'])
            if item.get('domain') == 'full-scene':
                scenes.add(item['source_id'])
            pixel = item.get('pixel_sha256')
            if pixel is None:
                with Image.open(root / item['image']) as im:
                    pixel = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            pixels[(item['width'], item['height'], pixel)].append(name + ':' + item['id'])
    return receipts, scenes, pixels, hashes


def polygon(points):
    if points and isinstance(points[0], (int, float)):
        if len(points) % 2:
            raise ValueError('Odd number of flattened polygon coordinates')
        points = list(zip(points[::2], points[1::2]))
    if len(points) < 3:
        raise ValueError('Annotation polygon needs at least three points')
    result = [[float(x), float(y)] for x, y in points]
    if any(not (-1e7 < v < 1e7) for p in result for v in p):
        raise ValueError('Invalid annotation coordinate')
    return result


def box_polygon(box):
    x1, y1, x2, y2 = box
    return polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def region(text, points, legible=True):
    points = polygon(points)
    xs, ys = zip(*points)
    return {'text': text, 'polygon': points, 'legible': bool(legible),
            'box': [min(xs), min(ys), max(xs), max(ys)]}


def references(root=ROOT):
    import pyarrow.parquet as pq

    candidates = []
    source_counts = {}
    receipt_file = pq.ParquetFile(root / DATA / 'sroie.parquet')
    for batch in receipt_file.iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        if len(row['words']) != len(row['bboxes']):
            raise ValueError('Mismatched SROIE boxes/text')
        candidates.append({'source': 'sroie', 'source_id': row['key'], 'source_split': 'test',
                           'width': row['image_size']['width'], 'height': row['image_size']['height'],
                           'regions': [region(t, box_polygon(b)) for t, b in zip(row['words'], row['bboxes'])],
                           '_image_bytes': row['image']['bytes']})
    source_counts['sroie'] = len(candidates)
    text = json.loads((root / DATA / 'textocr-val.json').read_text(encoding='utf-8'))
    source_counts['textocr'] = len(text['imgs'])
    for identifier, item in sorted(text['imgs'].items()):
        annotations = [text['anns'][str(a)] for a in text['imgToAnns'][identifier]]
        candidates.append({'source': 'textocr', 'source_id': identifier, 'source_split': 'validation',
                           'width': item['width'], 'height': item['height'],
                           'regions': [region(a['utf8_string'], a['points'], a['utf8_string'].strip() not in ('', '.'))
                                       for a in annotations]})
    del text
    with gzip.open(root / DATA / 'hiertext-val.json.gz', 'rb') as stream:
        raw = stream.read((512 << 20) + 1)
    if len(raw) > 512 << 20:
        raise ValueError('Oversized decompressed HierText annotations')
    hier = json.loads(raw)
    del raw
    source_counts['hiertext'] = len(hier['annotations'])
    for item in sorted(hier['annotations'], key=lambda i: i['image_id']):
        words = [w for p in item['paragraphs'] for line in p['lines'] for w in line['words']]
        candidates.append({'source': 'hiertext', 'source_id': item['image_id'], 'source_split': 'validation',
                           'width': item['image_width'], 'height': item['image_height'],
                           'regions': [region(w['text'], w['vertices'], w['legible']) for w in words]})
    if source_counts != {'sroie': 361, 'textocr': 3124, 'hiertext': 1724}:
        raise ValueError('Unexpected upstream split counts: ' + repr(source_counts))
    return sorted(candidates, key=lambda i: (i['source'], i['source_id'])), source_counts


def safe_tar_member(member):
    parts = PurePosixPath(member.name).parts
    if (not parts or member.name.startswith('/') or '\\' in member.name or ':' in member.name or
            any(p == '..' for p in parts) or not (member.isfile() or member.isdir()) or
            not 0 <= member.size <= 16 << 20):
        raise ValueError('Unsafe/oversized benchmark TAR member')


def image_record(root, candidate, body, restore):
    if not 0 < len(body) <= 16 << 20:
        raise ValueError('Oversized/empty benchmark image')
    source_hash = hashlib.sha256(body).hexdigest()
    with Image.open(io.BytesIO(body)) as opened:
        if opened.width * opened.height > PIXEL_CAP or opened.size != (candidate['width'], candidate['height']):
            raise ValueError('Source image dimensions mismatch or oversized')
        rgb = opened.convert('RGB')
        pixel = hashlib.sha256(rgb.tobytes()).hexdigest()
        orientation = opened.getexif().get(274)
    identifier = candidate['source'] + '-' + candidate['source_id']
    if not re.fullmatch(r'[a-z0-9-]+', identifier, re.I):
        raise ValueError('Unsafe source ID')
    relative = DATA / 'images' / (identifier + '.png')
    path = root / relative
    if not path.exists() and restore:
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(path, format='PNG')
    if not path.is_file():
        raise ValueError('Missing canonical benchmark image')
    with Image.open(path) as saved:
        if saved.size != rgb.size or saved.convert('RGB').tobytes() != rgb.tobytes():
            raise ValueError('Canonical benchmark pixels changed')
    return {'image': relative.as_posix(), 'sha256': sha256(path), 'pixel_sha256': pixel,
            'source_image_sha256': source_hash, 'source_exif_orientation': orientation}


def build(root=ROOT, restore=True):
    acquire(root, offline=not restore)
    candidates, source_counts = references(root)
    receipts, scenes, prior_pixels, prior_hashes = prior_index(root)
    items, excluded = [], []
    zip_path, tar_path = root / DATA / 'textocr-images.zip', root / DATA / 'hiertext-images.tgz'
    with zipfile.ZipFile(zip_path) as z, tarfile.open(tar_path, 'r:gz') as t:
        zip_members, tar_members = {}, {}
        zip_entries = z.infolist()
        if len(zip_entries) > 50000 or sum(i.file_size for i in zip_entries) > 16 << 30:
            raise ValueError('Oversized TextOCR archive expansion')
        for info in zip_entries:
            safe_member(info)
            if info.is_dir():
                continue
            key = PurePosixPath(info.filename).stem
            if key in zip_members:
                raise ValueError('Duplicate TextOCR archive ID')
            zip_members[key] = info
        tar_entries = t.getmembers()
        if len(tar_entries) > 10000 or sum(m.size for m in tar_entries) > 2 << 30:
            raise ValueError('Oversized HierText archive expansion')
        for member in tar_entries:
            safe_tar_member(member)
            if not member.isfile():
                continue
            key = PurePosixPath(member.name).stem
            if key in tar_members:
                raise ValueError('Duplicate HierText archive ID')
            tar_members[key] = member
        # Reading a compressed TAR in source-ID order would repeatedly decompress it.
        # Cache selected original bytes once in archive order, with per-file hashes.
        wanted_hier = {c['source_id'] for c in candidates if c['source'] == 'hiertext'}
        cache = root / DATA / 'hiertext-originals'
        for key, member in sorted(tar_members.items(), key=lambda pair: pair[1].offset):
            if key not in wanted_hier:
                continue
            if not re.fullmatch(r'[0-9a-f]+', key):
                raise ValueError('Invalid HierText ID')
            target = cache / (key + '.jpg')
            digest_path = target.with_suffix('.jpg.sha256')
            if target.exists() and digest_path.exists():
                if sha256(target) != digest_path.read_text().strip():
                    raise ValueError('HierText image checkpoint corruption')
                continue
            if not restore:
                raise ValueError('Missing HierText original checkpoint')
            with t.extractfile(member) as stream:
                body = stream.read((16 << 20) + 1)
            if len(body) != member.size:
                raise ValueError('Truncated TAR image')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            digest_path.write_text(sha256(target), encoding='ascii')
        for index, candidate in enumerate(candidates):
            source, source_id = candidate['source'], candidate['source_id']
            identifier = source + '-' + source_id
            if source_id in (receipts if source == 'sroie' else scenes):
                excluded.append({'id': identifier, 'reason': 'prior-source-parent-or-legacy-candidate'})
                continue
            if source == 'sroie':
                body = candidate['_image_bytes']
            elif source == 'textocr':
                body = z.read(zip_members[source_id])
            else:
                body = (cache / (source_id + '.jpg')).read_bytes()
            image = image_record(root, candidate, body, restore)
            overlap = prior_pixels.get((candidate['width'], candidate['height'], image['pixel_sha256']))
            if overlap:
                excluded.append({'id': identifier, 'reason': 'prior-decoded-pixel-overlap', 'matches': overlap})
                continue
            regs = candidate['regions']
            if not regs:
                raise ValueError('Missing benchmark annotations')
            annotation = {'schema': 1, 'id': identifier, 'regions': regs,
                          'complete_reference': all(r['legible'] for r in regs),
                          'row_order_text': '\n'.join(r['text'] for r in reading_order(regs))}
            annotation_path = DATA / 'annotations' / (identifier + '.json')
            if (root / annotation_path).exists():
                if json.loads((root / annotation_path).read_text(encoding='utf-8')) != annotation:
                    raise ValueError('Saved annotation differs from upstream')
            elif restore:
                write_json(root / annotation_path, annotation)
            else:
                raise ValueError('Missing benchmark annotation')
            items.append({'id': identifier, 'source': source, 'source_id': source_id,
                          'source_split': candidate['source_split'], 'role': 'comparative-development',
                          'width': candidate['width'], 'height': candidate['height'], **image,
                          'annotations': annotation_path.as_posix(), 'annotation_sha256': sha256(root / annotation_path),
                          'reference_regions': len(regs), 'legible_regions': sum(r['legible'] for r in regs),
                          'complete_reference': annotation['complete_reference']})
            if restore and (index + 1) % 100 == 0:
                print(f'Prepared {index + 1}/{len(candidates)} source images', flush=True)
    pixels, identities = defaultdict(list), defaultdict(list)
    for item in items:
        pixels[(item['width'], item['height'], item['pixel_sha256'])].append(item['id'])
        if item['source'] != 'sroie':
            identities[item['source_id']].append(item['id'])
    return {'schema': 1, 'purpose': 'broader-public-comparative-development',
            'source_evidence_sha256': sha256(root / SOURCES), 'prior_manifest_sha256': prior_hashes,
            'upstream_image_counts': source_counts, 'selected_counts': dict(Counter(i['source'] for i in items)),
            'excluded': excluded, 'shared_source_ids': [v for v in identities.values() if len(v) > 1],
            'shared_decoded_pixels': [v for v in pixels.values() if len(v) > 1], 'items': items}


def verify(root=ROOT, deep=False):
    saved = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    if deep:
        if build(root, restore=False) != saved:
            raise ValueError('Benchmark reconstruction mismatch')
    else:
        if saved['source_evidence_sha256'] != sha256(root / SOURCES):
            raise ValueError('Source evidence changed')
        for name, digest in saved['prior_manifest_sha256'].items():
            if sha256(root / name) != digest:
                raise ValueError('Prior evaluation manifest changed')
        seen = set()
        for item in saved['items']:
            if item['id'] in seen or item['role'] != 'comparative-development':
                raise ValueError('Duplicate or invalid benchmark item')
            seen.add(item['id'])
            for path_key, hash_key in (('image', 'sha256'), ('annotations', 'annotation_sha256')):
                if sha256(root / item[path_key]) != item[hash_key]:
                    raise ValueError('Benchmark file corruption')
        if dict(Counter(i['source'] for i in saved['items'])) != saved['selected_counts']:
            raise ValueError('Benchmark source counts changed')
    return saved


def prepare(root=ROOT):
    if (root / MANIFEST).exists():
        return verify(root, deep=True)
    value = build(root)
    write_json(root / MANIFEST, value)
    return verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--acquire-only', action='store_true')
    args = parser.parse_args()
    if args.acquire_only:
        acquire()
        return
    value = verify(deep=True) if args.verify_only else prepare()
    print(json.dumps({'selected': value['selected_counts'], 'excluded': len(value['excluded']),
                      'cross_source_shared_ids': len(value['shared_source_ids'])}, indent=2))


if __name__ == '__main__':
    main()
