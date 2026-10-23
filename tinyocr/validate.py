import json
from collections import Counter
from pathlib import Path

from PIL import Image

from .common import MANIFEST, ROOT, sha256


def validate(manifest_path=MANIFEST, root=ROOT):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("purpose") != "evaluation-only-pilot":
        raise ValueError("Not an evaluation-only manifest")
    items = manifest["items"]
    if not items:
        raise ValueError("Empty manifest")
    ids, hashes = set(), set()
    for item in items:
        if item["id"] in ids:
            raise ValueError(f"Duplicate id: {item['id']}")
        ids.add(item["id"])
        if item["task"] not in ("page", "crop"):
            raise ValueError("Unknown OCR task")
        if not isinstance(item["text"], str) or (not item["text"].strip() and item["domain"] != "blank-control"):
            raise ValueError("Missing reference")
        if item["source"] != "local-ui-smoke":
            source = manifest["sources"][item["source"]]
            if not source.get("revision") or item["split"] != "test" or item["reference_kind"] != "human":
                raise ValueError("Unpinned/non-test source or non-human reference")
        path = (Path(root) / item["image"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Image escapes project root")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Image checksum mismatch: {item['id']}")
        with Image.open(path) as image:
            image.load()
            if image.size != (item["width"], item["height"]) or min(image.size) < 1:
                raise ValueError("Invalid image dimensions")
        # GT crops can legitimately repeat price fields; disallow duplicate full samples.
        if "parent_id" not in item:
            if item["sha256"] in hashes:
                raise ValueError("Duplicate full-sample image")
            hashes.add(item["sha256"])
        for region in item.get("regions", []):
            x0, y0, x1, y1 = region["box"]
            if not (0 <= x0 < x1 <= item["width"] and 0 <= y0 < y1 <= item["height"]):
                raise ValueError(f"Out-of-bounds region: {item['id']}")
    for item in items:
        if "parent_id" in item and item["parent_id"] not in ids:
            raise ValueError("Orphan crop")
    print("Validated", len(items), "items:", dict(Counter(i["domain"] for i in items)))
    return manifest


if __name__ == "__main__":
    validate()
