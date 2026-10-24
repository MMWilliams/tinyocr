"""Command line interface for local image OCR."""

import argparse
import json
from pathlib import Path
import sys

from . import TinyOCR, __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline CPU OCR with bundled TinyOCR models")
    parser.add_argument("image", type=Path, help="local image path")
    parser.add_argument("--json", action="store_true", help="emit text, lines and words as JSON")
    parser.add_argument("--no-words", action="store_true", help="omit word geometry")
    parser.add_argument("--version", action="version", version=f"tinyocr-onnx {__version__}")
    args = parser.parse_args(argv)
    if not args.image.expanduser().is_file():
        parser.error(f"Image does not exist or is not a file: {args.image}")
    try:
        result = TinyOCR()(args.image, words=not args.no_words)
    except Exception as exc:
        print(f"tinyocr: {exc}", file=sys.stderr)
        return 1
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result.to_dict(), ensure_ascii=False) if args.json else result.text)
    return 0
