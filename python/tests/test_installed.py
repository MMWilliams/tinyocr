"""Public contracts exercised with actual bundled models and authored images."""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import tinyocr
from tinyocr import TinyOCR

FIXTURES = Path(__file__).parent / "fixtures"


class InstalledOCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = patch("socket.socket.connect", side_effect=AssertionError("OCR attempted network access"))
        cls.network.start()
        cls.addClassCleanup(cls.network.stop)
        cls.ocr = TinyOCR()

    def test_installed_version_and_assets(self):
        self.assertEqual(importlib.metadata.version("tinyocr-onnx"), "0.1.0")
        self.assertEqual(tinyocr.__version__, "0.1.0")
        assets = Path(tinyocr.__file__).parent / "assets"
        expected = {
            "PP-OCRv6_det_tiny.onnx": "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5",
            "PP-OCRv6_rec_tiny.onnx": "38cd559cb4b24c5056592d981b13e086f9472db9b3c8968296f0d0c2b4a65531",
        }
        for filename, digest in expected.items():
            self.assertEqual(hashlib.sha256((assets / filename).read_bytes()).hexdigest(), digest)
        for module in (self.ocr._engine.text_det, self.ocr._engine.text_rec):
            session = module.session.session
            self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
            self.assertEqual(session.get_session_options().intra_op_num_threads, 1)
            self.assertEqual(session.get_session_options().inter_op_num_threads, 1)

    def test_real_offline_ocr(self):
        result = self.ocr(FIXTURES / "receipt.png")
        self.assertEqual(result.text, "HELLO WORLD\nTOTAL 123.45")
        self.assertEqual([r["text"] for r in result.lines], ["HELLO WORLD", "TOTAL 123.45"])
        self.assertEqual([r["text"] for r in result.words], ["HELLO", "WORLD", "TOTAL", "123.45"])
        for region in result.lines + result.words:
            self.assertEqual(len(region["polygon"]), 4)
            for x, y in region["polygon"]:
                self.assertTrue(math.isfinite(x) and math.isfinite(y))
                self.assertTrue(0 <= x <= 700 and 0 <= y <= 210)
        self.assertEqual(json.loads(json.dumps(result.to_dict()))["text"], result.text)

    def test_word_toggle_keeps_lines(self):
        with_words = self.ocr(FIXTURES / "receipt.png")
        without_words = self.ocr(FIXTURES / "receipt.png", words=False)
        self.assertEqual(with_words.text, without_words.text)
        self.assertEqual(with_words.lines, without_words.lines)
        self.assertEqual(without_words.words, [])
        self.assertEqual(self.ocr(FIXTURES / "receipt.png").words, with_words.words)

    def test_blank(self):
        self.assertEqual(self.ocr(FIXTURES / "blank.png").to_dict(), {"text": "", "lines": [], "words": []})

    def test_missing_and_invalid_image(self):
        with self.assertRaises(FileNotFoundError):
            self.ocr(FIXTURES / "missing.png")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.png"
            path.write_text("not an image", encoding="utf-8")
            with self.assertRaises(Exception):
                self.ocr(path)

    def test_cli_text_json_version_and_errors(self):
        # The audit hook also forbids network access inside the CLI subprocess.
        bootstrap = (
            "import sys,runpy; "
            "sys.addaudithook(lambda event,args: "
            "(_ for _ in ()).throw(RuntimeError('network forbidden')) "
            "if event == 'socket.connect' else None); "
            "runpy.run_module('tinyocr',run_name='__main__')"
        )
        def cli(*args):
            return subprocess.run([sys.executable, "-c", bootstrap, *map(str, args)],
                                  capture_output=True, text=True, encoding="utf-8", timeout=60)
        result = cli(FIXTURES / "receipt.png")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "HELLO WORLD\nTOTAL 123.45")
        result = cli(FIXTURES / "receipt.png", "--json", "--no-words")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["text"], "HELLO WORLD\nTOTAL 123.45")
        self.assertEqual(data["words"], [])
        self.assertEqual(cli("--version").stdout.strip(), "tinyocr-onnx 0.1.0")
        result = cli(FIXTURES / "missing.png")
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not exist", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.png"
            path.write_text("not an image", encoding="utf-8")
            result = cli(path)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)

    def test_console_entry_point(self):
        executable = shutil.which("tinyocr")
        self.assertIsNotNone(executable)
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "tinyocr-onnx 0.1.0")


if __name__ == "__main__":
    unittest.main()
