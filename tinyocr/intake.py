"""Bounded, offline-verifiable SRD candidate intake. Never imports an OCR model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import stat
from urllib.parse import urljoin, urlsplit
import zipfile

from PIL import Image
import requests

from .common import ROOT, write_json


def sha256(path):
    """Hash bounded downloads without allocating the entire archive in RAM."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

SOURCE = 'https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/'
ARCHIVE_NAME = 'large-receipt-image-dataset-SRD.zip'
LOCAL = Path('data/independent-intake/srd')
CATALOG = Path('eval/srd-candidates.json')
REPORT = Path('results/srd-intake.json')
REVIEW = Path('eval/srd-review.json')
DOWNLOAD_CAP = 32 * 1024 * 1024
PAGE_CAP = 2 * 1024 * 1024
MEMBER_CAP = 16 * 1024 * 1024
EXPANSION_CAP = 256 * 1024 * 1024
MEMBER_COUNT_CAP = 2048
PIXEL_CAP = 25_000_000
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def https_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Expected public HTTPS URL')
    return url


def download(url, path, cap, session=None, offline=False):
    """Checkpoint complete transfers only; never append to an unverified partial body.

    Read/write no more than cap body bytes, even without Content-Length. An unknown
    length body reaching the cap is rejected because EOF cannot be proven in-budget.
    """
    https_url(url)
    path = Path(path)
    checkpoint = path.with_suffix(path.suffix + '.download.json')
    partial = path.with_suffix(path.suffix + '.part')
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text(encoding='utf-8'))
        if (record['requested_url'] != url or record['cap_bytes'] != cap or
                not path.is_file() or not 0 < record['bytes'] <= cap or
                path.stat().st_size != record['bytes'] or sha256(path) != record['sha256'] or
                record['content_length'] not in (None, record['bytes'])):
            raise ValueError('Download checkpoint mismatch')
        https_url(record['resolved_url'])
        return record
    if path.exists():
        raise ValueError('Uncheckpointed completed file; inspect before retrying')
    if offline:
        raise ValueError('Missing download checkpoint')
    path.parent.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    try:
        with client.get(url, stream=True, timeout=(15, 60),
                        headers={'Accept-Encoding': 'identity'}) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError('Expected complete HTTP 200 response')
            https_url(response.url)
            if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise ValueError('Encoded transfer not supported')
            value = response.headers.get('Content-Length')
            if value is not None and response.headers.get('Transfer-Encoding'):
                raise ValueError('Ambiguous transfer length')
            length = int(value) if value is not None else None
            if length is not None and not 0 < length <= cap:
                raise ValueError('Oversized or empty declared download')
            count = 0
            with partial.open('wb') as output:
                while count < cap:
                    chunk = response.raw.read(min(65536, cap - count), decode_content=False)
                    if not chunk:
                        break
                    if len(chunk) > cap - count:
                        raise ValueError('Oversized streamed download')
                    output.write(chunk)
                    count += len(chunk)
                if count == 0 or (length is not None and count != length):
                    raise ValueError('Truncated or mismatched download')
                if count == cap and length is None:
                    raise ValueError('Streamed download reached cap without proven EOF')
            record = {'requested_url': url, 'resolved_url': response.url,
                      'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                      'bytes': count, 'cap_bytes': cap, 'content_length': length,
                      'sha256': sha256(partial),
                      'etag': response.headers.get('ETag'),
                      'last_modified': response.headers.get('Last-Modified')}
        partial.replace(path)
        write_json(checkpoint, record)
        return record
    finally:
        partial.unlink(missing_ok=True)
        if session is None:
            client.close()


class PublisherPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.words = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href' and v)

    def handle_data(self, data):
        self.words.append(data)


def publisher_evidence(page):
    parser = PublisherPage()
    parser.feed(page.decode('utf-8'))
    links = {urljoin(SOURCE, link) for link in parser.links
             if urlsplit(link).path.endswith('/' + ARCHIVE_NAME)}
    if len(links) != 1:
        raise ValueError('Expected one publisher-linked SRD archive')
    url = https_url(links.pop())
    if urlsplit(url).hostname != urlsplit(SOURCE).hostname:
        raise ValueError('Unexpected archive host; requires source review')
    text = ' '.join(' '.join(parser.words).split())
    if 'The MIT License (MIT)' not in text or 'ExpressExpense.com' not in text:
        raise ValueError('Publisher rights evidence changed; review required')
    match = re.search(re.escape(ARCHIVE_NAME) + r'\s+\([^)]*\)\s+([0-9a-fA-F]+)', text)
    checksum = match.group(1) if match else None
    return {'archive_url': url, 'publisher_license_claim': 'MIT',
            'attribution': 'ExpressExpense.com', 'source_page': SOURCE,
            'publisher_md5_text': checksum,
            'publisher_md5_well_formed': bool(checksum and re.fullmatch(r'[0-9a-fA-F]{32}', checksum)),
            'rights_status': 'publisher-claim-recorded; collection-review-pending'}


def safe_member(info):
    name = info.filename
    parts = name.rstrip('/').split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                *(f'LPT{i}' for i in range(1, 10))}
    if (not name or info.orig_filename != name or '\\' in name or name.startswith('/') or
            any(not p or p in ('.', '..') or p.endswith((' ', '.')) or
                any(ord(c) < 32 or c in ':<>"|?*' for c in p) or
                p.split('.')[0].upper() in reserved for p in parts)):
        raise ValueError('Unsafe archive member: ' + repr(name))
    kind = stat.S_IFMT(info.external_attr >> 16)
    if (kind not in (0, stat.S_IFREG, stat.S_IFDIR) or
            (kind != 0 and (kind == stat.S_IFDIR) != info.is_dir())):
        raise ValueError('Unsupported archive member type: ' + name)
    if info.flag_bits & 1:
        raise ValueError('Encrypted archive member')
    if not 0 <= info.file_size <= MEMBER_CAP or info.compress_size > DOWNLOAD_CAP:
        raise ValueError('Oversized archive member')


def inventory(path, root, restore=False):
    members, images, rights = [], [], []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MEMBER_COUNT_CAP:
            raise ValueError('Empty or oversized member count')
        seen = set()
        total = 0
        for info in infos:
            safe_member(info)
            key = info.filename.rstrip('/').casefold()
            if key in seen:
                raise ValueError('Duplicate or case-alias archive member')
            seen.add(key)
            total += info.file_size
            if total > EXPANSION_CAP:
                raise ValueError('Oversized archive expansion')
        for info in sorted(infos, key=lambda i: i.filename):
            if info.is_dir():
                members.append({'name': info.filename, 'kind': 'directory', 'bytes': 0})
                continue
            with archive.open(info) as stream:
                body = stream.read(MEMBER_CAP + 1)  # zipfile validates CRC on EOF.
            if len(body) != info.file_size or len(body) > MEMBER_CAP:
                raise ValueError('Member size mismatch')
            digest = hashlib.sha256(body).hexdigest()
            name = Path(info.filename)
            metadata = '__MACOSX' in name.parts or name.name.startswith('._') or name.name == '.DS_Store'
            kind = 'metadata' if metadata else 'other'
            if not metadata and name.suffix.lower() in IMAGE_EXTENSIONS:
                kind = 'image'
                with Image.open(io.BytesIO(body)) as opened:
                    if opened.width * opened.height > PIXEL_CAP or getattr(opened, 'n_frames', 1) != 1:
                        raise ValueError('Oversized or multiframe image')
                    rgb = opened.convert('RGB')
                    pixels = hashlib.sha256(rgb.tobytes()).hexdigest()
                    size = rgb.size
                    fmt = opened.format
                    orientation = opened.getexif().get(274)
                # No member names are ever used as extraction paths.
                relative = LOCAL / 'images' / (digest + name.suffix.lower())
                target = root / relative
                if not target.exists() and restore:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(body)
                if not target.is_file() or sha256(target) != digest:
                    raise ValueError('Missing or corrupt intake image: ' + str(relative))
                images.append({'source_id': info.filename, 'image': relative.as_posix(),
                               'sha256': digest, 'pixel_sha256': pixels,
                               'width': size[0], 'height': size[1], 'format': fmt,
                               'exif_orientation': orientation})
            elif not metadata and re.search(r'license|licence|copyright|readme|attribution', name.name, re.I):
                kind = 'rights-or-readme'
                rights.append({'name': info.filename, 'sha256': digest,
                               'text_utf8': body.decode('utf-8', errors='replace')})
            members.append({'name': info.filename, 'kind': kind, 'bytes': len(body),
                            'compressed_bytes': info.compress_size, 'crc32': f'{info.CRC:08x}',
                            'sha256': digest})
    if not images:
        raise ValueError('No decoded images in archive')
    return {'members': members, 'images': images, 'rights_files': rights,
            'expanded_bytes': total}


def load_review(root, archive_hash, ids):
    path = root / REVIEW
    if not path.exists():
        return {}, None
    review = json.loads(path.read_text(encoding='utf-8'))
    if (set(review) != {'schema', 'archive_sha256', 'method', 'coverage', 'items'} or
            review['schema'] != 1 or review['archive_sha256'] != archive_hash or
            review['method'] != 'assistant-visual-triage' or
            set(review['items']) - set(ids)):
        raise ValueError('Invalid candidate review provenance')
    fields = {'capture_type', 'template_group', 'merchant_group', 'language', 'parent_id',
              'difficulty_tags', 'notes', 'reviewed_views'}
    for item in review['items'].values():
        if set(item) != fields or item['capture_type'] not in {
                'unknown', 'photo-like-unconfirmed', 'scan-like-unconfirmed',
                'screenshot-of-photo-unconfirmed', 'render-like-unconfirmed'}:
            raise ValueError('Invalid candidate review fields')
        if item['parent_id'] not in ids:
            raise ValueError('Invalid candidate parent group')
        for key in ('difficulty_tags', 'reviewed_views'):
            if not isinstance(item[key], list) or not all(isinstance(v, str) for v in item[key]):
                raise ValueError('Invalid candidate review list')
        for key in fields - {'difficulty_tags', 'reviewed_views'}:
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError('Invalid candidate review string')
    return review['items'], {'path': REVIEW.as_posix(), 'sha256': sha256(path),
                            'method': review['method'], 'coverage': review['coverage']}


def build(root, offline=False):
    local = root / LOCAL
    page = download(SOURCE, local / 'source.html', PAGE_CAP, offline=offline)
    rights = publisher_evidence((local / 'source.html').read_bytes())
    archive = download(rights['archive_url'], local / 'srd.zip', DOWNLOAD_CAP, offline=offline)
    actual_md5 = hashlib.md5((local / 'srd.zip').read_bytes(), usedforsecurity=False).hexdigest()
    if rights['publisher_md5_well_formed'] and rights['publisher_md5_text'].lower() != actual_md5:
        raise ValueError('Publisher MD5 mismatch')
    data = inventory(local / 'srd.zip', root, restore=not offline)
    ids = [f'srd-{index:03d}' for index in range(1, len(data['images']) + 1)]
    reviews, review_record = load_review(root, archive['sha256'], ids)
    items = []
    exact_parents = {}
    for identifier, image in zip(ids, data['images']):
        exact_parent = exact_parents.setdefault((image['width'], image['height'], image['pixel_sha256']), identifier)
        triage = reviews.get(identifier, {
            'capture_type': 'unknown', 'template_group': 'pending', 'merchant_group': 'pending',
            'language': 'pending', 'difficulty_tags': [], 'notes': 'Awaiting visual triage.',
            'reviewed_views': [], 'parent_id': exact_parent})
        items.append({'id': identifier, 'source': 'ExpressExpense-SRD',
                      'source_revision': archive['sha256'], 'original_url': rights['archive_url'],
                      'parent_identity_status': 'provisional-exact-hash-and-visual-group',
                      'capture_session_id': 'unknown', **image, **triage,
                      'group_status': 'provisional-assistant-triage' if identifier in reviews else 'pending',
                      'rights_evidence': {'source_page_sha256': page['sha256'],
                                          'archive_rights_files': [r['name'] for r in data['rights_files']]},
                      'role': 'candidate-only', 'split': 'unassigned',
                      'transcription_status': 'pending', 'human_review_status': 'pending',
                      'cross_corpus_duplicate_review': 'pending', 'inference_status': 'not-run'})
    by_id = {item['id']: item for item in items}
    for item in items:
        parent = by_id[item['parent_id']]
        exact_parent = by_id[exact_parents[(item['width'], item['height'], item['pixel_sha256'])]]
        if parent['parent_id'] != parent['id'] or exact_parent['parent_id'] != item['parent_id']:
            raise ValueError('Inconsistent parent groups or separated exact duplicates')
    catalog = {'schema': 1, 'purpose': 'unlabeled-candidate-intake-only',
               'pixel_hash_basis': 'decoded RGB stored pixels, without EXIF transpose; dimensions recorded separately',
               'source': rights, 'review': review_record, 'items': items}
    report = {'schema': 1, 'source_page': page, 'archive': archive,
              'archive_md5': actual_md5, 'rights': rights, 'rights_files': data['rights_files'],
              'limits': {'download_bytes': DOWNLOAD_CAP, 'member_bytes': MEMBER_CAP,
                         'expansion_bytes': EXPANSION_CAP, 'members': MEMBER_COUNT_CAP,
                         'pixels_per_image': PIXEL_CAP},
              'expanded_bytes': data['expanded_bytes'], 'members': data['members'],
              'image_count': len(items), 'unique_file_hashes': len({i['sha256'] for i in items}),
              'unique_pixel_hashes': len({(i['width'], i['height'], i['pixel_sha256']) for i in items}),
              'capture_triage': dict(Counter(i['capture_type'] for i in items)),
              'review': review_record, 'benchmark_approved': False}
    return catalog, report


def verify(root=ROOT):
    catalog, report = build(root, offline=True)
    if json.loads((root / CATALOG).read_text(encoding='utf-8')) != catalog:
        raise ValueError('Candidate catalog mismatch')
    report['catalog_sha256'] = sha256(root / CATALOG)
    if json.loads((root / REPORT).read_text(encoding='utf-8')) != report:
        raise ValueError('Intake report mismatch')
    return report


def prepare(root=ROOT, apply_review=False):
    if (root / REPORT).exists() and not apply_review:
        return verify(root)
    catalog, report = build(root, offline=apply_review)
    if apply_review:
        # Updating review must not rewrite acquisition evidence or member inventory.
        previous = json.loads((root / REPORT).read_text(encoding='utf-8'))
        mutable = {'review', 'catalog_sha256', 'capture_triage'}
        if ({k: v for k, v in previous.items() if k not in mutable} !=
                {k: v for k, v in report.items() if k not in mutable}):
            raise ValueError('Acquisition evidence changed during review update')
    write_json(root / CATALOG, catalog)
    report['catalog_sha256'] = sha256(root / CATALOG)
    write_json(root / REPORT, report)
    return verify(root)


def contact_sheets(root=ROOT):
    from PIL import ImageDraw

    verify(root)
    items = json.loads((root / CATALOG).read_text(encoding='utf-8'))['items']
    directory = root / LOCAL / 'contact-sheets'
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(items), 20):
        sheet = Image.new('RGB', (1600, 1800), '#d4d4d4')
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(items[start:start + 20]):
            x, y = (index % 5) * 320, (index // 5) * 450
            with Image.open(root / item['image']) as opened:
                thumb = opened.convert('RGB')
                thumb.thumbnail((310, 420))
                sheet.paste(thumb, (x + (320 - thumb.width) // 2, y + 25))
            draw.text((x + 5, y + 5), item['id'] + ' ' + Path(item['source_id']).name, fill='black')
        path = directory / f'{start // 20 + 1:02d}.jpg'
        sheet.save(path, quality=92)
        paths.append(path.relative_to(root).as_posix())
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--apply-review', action='store_true')
    mode.add_argument('--contact-sheets', action='store_true')
    args = parser.parse_args()
    if args.contact_sheets:
        print(json.dumps(contact_sheets(), indent=2))
        return
    report = verify() if args.verify_only else prepare(apply_review=args.apply_review)
    print(json.dumps({k: report[k] for k in ('image_count', 'unique_file_hashes',
                                           'unique_pixel_hashes', 'capture_triage', 'benchmark_approved')}, indent=2))


if __name__ == '__main__':
    main()
