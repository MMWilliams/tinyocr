import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from .common import MANIFEST, ROOT, sha256
from .expand import EXPANSION


def select_dev(manifest):
    items = [i for i in manifest["items"] if i["role"] == "dev"]
    if not items:
        raise ValueError("No development samples")
    return items


def check_training_overlap(training_items, evaluation_items):
    identities = {(i["source"], i["source_id"]) for i in evaluation_items}
    pixels = {i["pixel_sha256"] for i in evaluation_items}
    for item in training_items:
        if not all(item.get(k) for k in ("source", "source_id", "source_split", "pixel_sha256")):
            raise ValueError("Training item lacks provenance required for leakage check")
        if (item["source"],item["source_id"]) in identities or item["pixel_sha256"] in pixels:
            raise ValueError("Training/evaluation overlap")
        if ((item["source"] == "textocr" and item["source_split"] == "val") or
            (item["source"] == "screens" and item["source_split"] in ("valid","test")) or
            (item["source"] in ("receipts", "scene") and item["source_split"] == "test") or
            item["source"] == "local-ui-smoke"):
            raise ValueError("Reserved source split cannot be used for training")


def check(manifest_path=EXPANSION, root=ROOT):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != 2 or manifest.get("purpose") != "evaluation-only-expansion":
        raise ValueError("Wrong manifest type")
    if manifest["pilot_manifest_sha256"] != sha256(Path(root)/"eval/manifest.json"):
        raise ValueError("Original pilot changed")
    lineage_path = Path(root)/"eval/legacy_scene_lineage.json"
    if manifest["lineage_sha256"] != sha256(lineage_path):
        raise ValueError("Lineage audit changed")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    excluded = {parent for entry in lineage["items"] for parent in entry["candidate_parent_ids"]}
    ids, groups, pixels, source_roles = {}, {}, {}, {}
    for item in manifest["items"]:
        if item["id"] in ids:
            raise ValueError("Duplicate item ID")
        ids[item["id"]] = item
        if item["role"] not in ("dev", "final"):
            raise ValueError("Invalid role")
        if item["source"] not in manifest["sources"]:
            raise ValueError("Missing source")
        if item["source"] == "textocr" and item["source_id"] in excluded:
            raise ValueError("Legacy crop candidate parent entered expansion")
        for table, key in ((groups,item["group_id"]),(pixels,item["pixel_sha256"]),
                           (source_roles,(item["source"],item["source_id"]))):
            if key in table and table[key] != item["role"]:
                raise ValueError("Development/final overlap")
            table[key] = item["role"]
        path = (Path(root)/item["image"]).resolve()
        if not path.is_relative_to(Path(root).resolve()) or sha256(path) != item["sha256"]:
            raise ValueError("Image path/hash mismatch")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (item["width"],item["height"]):
                raise ValueError("Image dimensions changed")
            if hashlib.sha256(image.tobytes()).hexdigest() != item["pixel_sha256"]:
                raise ValueError("Decoded-pixel checksum mismatch")
        if item["score_mode"] == "annotated-recall":
            if item["domain"] != "real-screen" or not item.get("regions") or "text" in item:
                raise ValueError("Partial-screen labels must not become full-page truth")
        elif item["score_mode"] != "transcription" or not item.get("text", "").strip():
            raise ValueError("Invalid transcription reference")
        for region in item.get("regions", []):
            x0,y0,x1,y1 = region["box"]
            if not (0 <= x0 < x1 <= item["width"] and 0 <= y0 < y1 <= item["height"]):
                raise ValueError("Region out of image bounds")
    for item in manifest["items"]:
        if "parent_id" in item:
            parent = ids.get(item["parent_id"])
            if parent is None or (parent["role"],parent["group_id"]) != (item["role"],item["group_id"]):
                raise ValueError("Crop split differs from parent")
    for role in ("dev","final"):
        if not any(i["role"]==role for i in manifest["items"]):
            raise ValueError("Empty partition")
    print("Expansion verified:", dict(Counter((i["role"], i["domain"]) for i in manifest["items"])))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path)
    args = parser.parse_args()
    manifest = check()
    if args.training_manifest:
        training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
        check_training_overlap(training["items"], manifest["items"])
        print("Training manifest passes recorded source/pixel overlap checks")
