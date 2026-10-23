import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tinyocr.common import sha256
from tinyocr.validate import validate


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (10, 10), "white").save(self.root / "image.png")
        self.item = {"id": "a", "domain": "synthetic-screen", "task": "page",
                     "text": "A", "source": "local-ui-smoke", "image": "image.png",
                     "sha256": sha256(self.root / "image.png"), "width": 10, "height": 10}

    def run_manifest(self, items):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"schema": 1, "purpose": "evaluation-only-pilot", "items": items}))
        return validate(path, self.root)

    def test_valid_fixture(self):
        self.assertEqual(len(self.run_manifest([self.item])["items"]), 1)

    def test_corruption_is_rejected(self):
        (self.root / "image.png").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_manifest([self.item])

    def test_duplicate_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            self.run_manifest([self.item, self.item])

    def test_bad_box_rejected(self):
        self.item["regions"] = [{"text": "A", "box": [0, 0, 11, 5]}]
        with self.assertRaisesRegex(ValueError, "Out-of-bounds"):
            self.run_manifest([self.item])

    def test_empty_set_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty manifest"):
            self.run_manifest([])

    def test_missing_reference_rejected(self):
        self.item["text"] = " "
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            self.run_manifest([self.item])


if __name__ == "__main__":
    unittest.main()
