import copy
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from tinyocr import intake
from tinyocr.common import sha256, write_json


class RawBody(io.BytesIO):
    def read(self, size=-1, decode_content=False):
        return super().read(size)


class Response:
    status_code = 200
    url = 'https://example.org/archive.zip'

    def __init__(self, body, length=None):
        self.raw = RawBody(body)
        self.headers = {} if length is None else {'Content-Length': str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self.response


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def zip(self, entries):
        path = self.root / 'fixture.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name, body in entries:
                archive.writestr(name, body)
        # Windows ZipInfo normalizes backslashes when writing. Inject the original
        # spelling into both headers to exercise an actual untrusted ZIP name.
        for name, _ in entries:
            if isinstance(name, str) and '\\' in name:
                path.write_bytes(path.read_bytes().replace(name.replace('\\', '/').encode(), name.encode()))
        return path

    def png(self, color='red'):
        output = io.BytesIO()
        Image.new('RGB', (3, 2), color).save(output, format='PNG')
        return output.getvalue()

    def test_download_checkpoint_offline_and_corruption(self):
        path = self.root / 'download.zip'
        client = Session(Response(b'1234', 4))
        saved = intake.download(Response.url, path, 5, client)
        self.assertEqual(saved['bytes'], 4)
        self.assertEqual(saved['sha256'], sha256(path))
        with patch.object(intake.requests, 'Session', side_effect=AssertionError('network')):
            self.assertEqual(intake.download(Response.url, path, 5, offline=True), saved)
        self.assertEqual(client.calls, 1)
        path.write_bytes(b'4321')
        with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
            intake.download(Response.url, path, 5, client)
        self.assertEqual(client.calls, 1)

    def test_download_caps_and_truncation_leave_no_checkpoint(self):
        cases = [(b'123456', 6), (b'123456', None), (b'12', 4), (b'', 0), (b'1234', 2)]
        for body, length in cases:
            with self.subTest(length=length, body=body):
                path = self.root / 'bad.zip'
                response = Response(body, length)
                with self.assertRaises(ValueError):
                    intake.download(Response.url, path, 5, Session(response))
                self.assertLessEqual(response.raw.tell(), 5)
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix('.zip.part').exists())
                self.assertFalse(path.with_suffix('.zip.download.json').exists())

    def test_unknown_length_and_stale_partial(self):
        path = self.root / 'ok.zip'
        path.with_suffix('.zip.part').write_bytes(b'untrusted previous response')
        record = intake.download(Response.url, path, 5, Session(Response(b'1234')))
        self.assertEqual(path.read_bytes(), b'1234')
        self.assertIsNone(record['content_length'])
        second = self.root / 'exact.zip'
        self.assertEqual(intake.download(Response.url, second, 4, Session(Response(b'1234', 4)))['bytes'], 4)

    def test_encoded_or_ambiguous_transfer_is_rejected(self):
        for extra in ({'Content-Encoding': 'gzip'}, {'Transfer-Encoding': 'chunked'}):
            response = Response(b'12345', 4)
            response.headers.update(extra)
            with self.assertRaises(ValueError):
                intake.download(Response.url, self.root / 'bad.zip', 4, Session(response))
            self.assertEqual(response.raw.tell(), 0)
            self.assertFalse((self.root / 'bad.zip').exists())

    def test_unsafe_member_paths_and_types(self):
        names = ['../escape.png', '/escape.png', 'C:/escape.png', 'folder\\escape.png',
                 'a//b.png', './image.png', 'a/../b.png', 'file.png:stream',
                 'CON.png', 'a./b.png', 'trailing /b.png']
        for name in names:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                    intake.inventory(self.zip([(name, b'bad')]), self.root, True)
        link = zipfile.ZipInfo('link.png')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(ValueError, 'member type'):
            intake.inventory(self.zip([(link, b'target')]), self.root, True)
        self.assertFalse((self.root / intake.LOCAL / 'images').exists())

    def test_duplicate_case_alias_and_expansion_caps(self):
        with self.assertRaisesRegex(ValueError, 'case-alias'):
            intake.inventory(self.zip([('A.png', b'1'), ('a.png', b'2')]), self.root, True)
        path = self.zip([('a.txt', b'1234'), ('b.txt', b'5678')])
        with patch.object(intake, 'EXPANSION_CAP', 7):
            with self.assertRaisesRegex(ValueError, 'expansion'):
                intake.inventory(path, self.root, True)
        with patch.object(intake, 'MEMBER_CAP', 3):
            with self.assertRaisesRegex(ValueError, 'archive member'):
                intake.inventory(path, self.root, True)
        with patch.object(intake, 'MEMBER_COUNT_CAP', 1):
            with self.assertRaisesRegex(ValueError, 'member count'):
                intake.inventory(path, self.root, True)

    def test_inventory_pixels_rights_and_safe_materialization(self):
        path = self.zip([('folder/picture.png', self.png()), ('LICENSE.txt', b'Example fixture rights'),
                         ('__MACOSX/._picture.png', b'not an image'), ('run.exe', b'never execute')])
        data = intake.inventory(path, self.root, True)
        self.assertEqual(len(data['images']), 1)
        self.assertEqual(data['rights_files'][0]['text_utf8'], 'Example fixture rights')
        self.assertEqual([m['kind'] for m in data['members']], ['rights-or-readme', 'metadata', 'image', 'other'])
        image = data['images'][0]
        with Image.open(self.root / image['image']) as opened:
            self.assertEqual(opened.size, (3, 2))
            self.assertEqual(opened.tobytes(), bytes([255, 0, 0]) * 6)
        self.assertFalse((self.root / 'folder').exists())
        self.assertFalse((self.root / 'run.exe').exists())
        self.assertEqual(intake.inventory(path, self.root), data)
        (self.root / image['image']).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'corrupt intake image'):
            intake.inventory(path, self.root, True)

    def test_corrupt_crc_invalid_image_and_pixel_limit(self):
        path = self.zip([('note.txt', b'abcdef')])
        path.write_bytes(path.read_bytes().replace(b'abcdef', b'abcdeg'))
        with self.assertRaises(zipfile.BadZipFile):
            intake.inventory(path, self.root, True)
        with self.assertRaises(OSError):
            intake.inventory(self.zip([('bad.png', b'not an image')]), self.root, True)
        with patch.object(intake, 'PIXEL_CAP', 5):
            with self.assertRaisesRegex(ValueError, 'Oversized or multiframe'):
                intake.inventory(self.zip([('a.png', self.png())]), self.root, True)

    def test_exif_orientation_preserves_stored_pixels(self):
        output = io.BytesIO()
        im = Image.new('RGB', (3, 2), 'red')
        exif = Image.Exif()
        exif[274] = 6
        im.save(output, format='JPEG', exif=exif)
        path = self.zip([('rotated.jpg', output.getvalue())])
        item = intake.inventory(path, self.root, True)['images'][0]
        self.assertEqual((item['width'], item['height'], item['exif_orientation']), (3, 2, 6))
        self.assertEqual((self.root / item['image']).read_bytes(), output.getvalue())

    def test_exact_duplicates_share_a_provisional_parent(self):
        self.fixture()
        path = self.zip([('first.png', self.png()), ('second.png', self.png())])
        local = self.root / intake.LOCAL / 'srd.zip'
        local.write_bytes(path.read_bytes())
        checkpoint = local.with_suffix('.zip.download.json')
        record = json.loads(checkpoint.read_text())
        record.update(sha256=sha256(local), bytes=local.stat().st_size,
                      content_length=local.stat().st_size)
        write_json(checkpoint, record)
        report = intake.prepare(self.root)
        catalog = json.loads((self.root / intake.CATALOG).read_text())
        self.assertEqual(report['image_count'], 2)
        self.assertEqual(report['unique_pixel_hashes'], 1)
        self.assertEqual([i['parent_id'] for i in catalog['items']], ['srd-001', 'srd-001'])

    def fixture(self):
        page = ('<p>The MIT License (MIT) ExpressExpense.com</p><a href="/' +
                intake.ARCHIVE_NAME + '">' + intake.ARCHIVE_NAME + '</a> (19.2MB) abc123').encode()
        archive = self.zip([('receipt.png', self.png())]).read_bytes()
        client = Session(Response(page, len(page)))
        intake.download(intake.SOURCE, self.root / intake.LOCAL / 'source.html', intake.PAGE_CAP, client)
        url = intake.publisher_evidence(page)['archive_url']
        intake.download(url, self.root / intake.LOCAL / 'srd.zip', intake.DOWNLOAD_CAP,
                        Session(Response(archive, len(archive))))

    def test_end_to_end_offline_verifier_and_candidate_lock(self):
        self.fixture()
        with patch.object(intake.requests, 'Session', side_effect=AssertionError('network')):
            report = intake.prepare(self.root)
            self.assertEqual(report['image_count'], 1)
            self.assertFalse(report['benchmark_approved'])
            self.assertFalse(report['rights']['publisher_md5_well_formed'])
            self.assertEqual(intake.verify(self.root), report)
            before = sha256(self.root / intake.CATALOG), sha256(self.root / intake.REPORT)
            self.assertEqual(intake.prepare(self.root), report)
            self.assertEqual(before, (sha256(self.root / intake.CATALOG), sha256(self.root / intake.REPORT)))
            catalog = json.loads((self.root / intake.CATALOG).read_text())
            for field, value in [('split', 'final'), ('transcription_status', 'complete'),
                                 ('human_review_status', 'approved'), ('inference_status', 'run')]:
                bad = copy.deepcopy(catalog)
                bad['items'][0][field] = value
                write_json(self.root / intake.CATALOG, bad)
                with self.assertRaisesRegex(ValueError, 'catalog mismatch'):
                    intake.verify(self.root)
            write_json(self.root / intake.CATALOG, catalog)
            (self.root / intake.LOCAL / 'srd.zip').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                intake.verify(self.root)

    def test_review_update_preserves_acquisition_and_rejects_split_override(self):
        self.fixture()
        report = intake.prepare(self.root)
        review = {'schema': 1, 'archive_sha256': report['archive']['sha256'],
                  'method': 'assistant-visual-triage', 'coverage': 'test fixture',
                  'items': {'srd-001': {'capture_type': 'unknown', 'template_group': 'pending',
                            'parent_id': 'srd-001',
                            'merchant_group': 'pending', 'language': 'pending', 'difficulty_tags': [],
                            'notes': 'Fixture review', 'reviewed_views': ['fixture']}}}
        write_json(self.root / intake.REVIEW, review)
        with self.assertRaises(ValueError):
            intake.verify(self.root)
        updated = intake.prepare(self.root, apply_review=True)
        self.assertEqual(updated['archive'], report['archive'])
        self.assertEqual(updated['members'], report['members'])
        review['items']['srd-001']['split'] = 'final'
        write_json(self.root / intake.REVIEW, review)
        with self.assertRaisesRegex(ValueError, 'review fields'):
            intake.prepare(self.root, apply_review=True)


if __name__ == '__main__':
    unittest.main()
