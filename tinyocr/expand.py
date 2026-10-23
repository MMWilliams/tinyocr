"""Prepare real screens/full scenes, preserving the original pilot unchanged."""
import csv
import hashlib
import io
import json
import time
import zipfile

from PIL import Image

from .common import MANIFEST, ROOT, reading_order, sha256, write_json
from .prepare import get
from .remotezip import RangeReader, read_member
from .screen_labels import parse_screen

EXPANSION = ROOT / "eval/expansion-region-v2.json"
DATA = ROOT / "data/expansion"
TEXT_REV = "0d0a421b9f09f2d00358af52091de57bfaee1684"
SCREEN_REV = "3b1700164de1f6fb870cd486e2ac14ee1c375aa4"
GOOGLE_REV = "053a4a1c4737969def99fd36084d7adf3715798f"
TEXT_BASE = f"https://huggingface.co/datasets/redactable-llm/TextOCR/resolve/{TEXT_REV}"


def ranked(values):
    return sorted(values, key=lambda x: hashlib.sha256(("tinyocr-expansion-v1:"+str(x)).encode()).hexdigest())


def cache_file(name, url):
    path = DATA / name
    digest_path = path.with_suffix(path.suffix+".sha256")
    if not path.exists():
        payload = get(url).content
        if len(payload) > 64*1024*1024:
            raise ValueError("Metadata download exceeds 64MB")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if digest_path.exists() and digest_path.read_text().strip() != sha256(path):
        raise ValueError("Cached source metadata checksum changed")
    digest_path.write_text(sha256(path), encoding="ascii")
    return path


def image_fields(image, name):
    path = DATA / "images" / (name+".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return {"image": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
            "pixel_sha256": hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(),
            "width": image.width, "height": image.height}


def screen_row(split, offset):
    checkpoint = DATA / f"screen-{split}-{offset}.json"
    image_path = DATA / f"screen-{split}-{offset}.jpg"
    if checkpoint.exists() and image_path.exists():
        row = json.loads(checkpoint.read_text(encoding="utf-8"))
        if row["revision"] != SCREEN_REV or row["image_sha256"] != sha256(image_path):
            raise ValueError("Bad screen checkpoint")
        return row["row"], Image.open(image_path).convert("RGB")
    response = get("https://datasets-server.huggingface.co/rows", {
        "dataset": "bevaya/RICO-ScreenAnnotation", "config": "default", "split": split,
        "offset": offset, "length": 1}).json()
    entry = response["rows"][0]
    if response.get("partial") or entry.get("truncated_cells"):
        raise ValueError("Truncated source row")
    row = entry["row"]
    if SCREEN_REV not in row["image"]["src"]:
        raise ValueError("Viewer source revision changed")
    image_path.write_bytes(get(row["image"]["src"]).content)
    row.pop("image")
    write_json(checkpoint, {"row": row, "revision": SCREEN_REV, "image_sha256": sha256(image_path)})
    time.sleep(1)
    return row, Image.open(image_path).convert("RGB")


def prepare():
    existing = None
    if EXPANSION.exists():
        existing = json.loads(EXPANSION.read_text(encoding="utf-8"))
        if all((ROOT/item["image"]).exists() for item in existing["items"]):
            from .check_expansion import check
            check()
            print("Frozen expansion reused without downloads")
            return
        print("Restoring missing expansion images from pinned sources", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    text_path = cache_file("textocr-val.json", TEXT_BASE+"/TextOCR_0.1_val.json")
    text = json.loads(text_path.read_text(encoding="utf-8"))
    pilot = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit = []
    excluded_images = set()
    for item in pilot["items"]:
        if item["domain"] == "scene-crop":
            matches = [a for a in text["anns"].values() if a["utf8_string"] == item["text"]]
            candidates = sorted({a["image_id"] for a in matches})
            if not candidates:
                raise ValueError("Unmapped legacy crop; cannot select a disjoint TextOCR holdout")
            excluded_images.update(candidates)
            audit.append({"pilot_id": item["id"], "original_text": item["text"],
                          "candidate_parent_ids": candidates,
                          "candidate_annotation_ids": [a["id"] for a in matches],
                          "status": "candidate-only; exact text match does not prove image identity"})
    write_json(ROOT/"eval/legacy_scene_lineage.json", {"source_revision": TEXT_REV,
                "source_metadata_sha256": sha256(text_path), "items": audit,
                "policy": "Exclude ALL candidate parent images from expanded dev/final; never train on legacy crops or TextOCR val"})
    items, sources = [], {
        "textocr": {"revision": TEXT_REV, "url": TEXT_BASE, "metadata_sha256": sha256(text_path),
                    "license": "Original TextOCR CC-BY-4.0; image rights follow original dataset terms",
                    "reference_kind": "human", "source_split": "val"},
        "screens": {"revision": SCREEN_REV, "google_revision": GOOGLE_REV,
                    "url": "https://github.com/google-research-datasets/screen_annotation",
                    "license": "Screen Annotation CC-BY-4.0; underlying RICO screenshots have their own terms",
                    "reference_kind": "human-corrected automated annotations"}}
    for role, split, total, count in (("dev", "valid", 2311, 12), ("final", "test", 4217, 8)):
        original_path = cache_file(f"google-{split}.csv",
            f"https://raw.githubusercontent.com/google-research-datasets/screen_annotation/{GOOGLE_REV}/{split}.csv")
        with original_path.open(encoding="utf-8", newline="") as f:
            original = {r["screen_id"]: r["screen_annotation"] for r in csv.DictReader(f)}
        sources["screens"][f"{split}_labels_sha256"] = sha256(original_path)
        for offset in ranked(range(total))[:count]:
            row, image = screen_row(split, offset)
            if original[row["screen_id"]] != row["screen_annotation"]:
                raise ValueError("Mirror label differs from original Google release")
            identifier = "screen-"+row["screen_id"]
            regions = reading_order(parse_screen(row["screen_annotation"], image.width, image.height))
            base = {"source": "screens", "source_id": row["screen_id"], "source_split": split,
                    "role": role, "reference_kind": "human-corrected", "group_id": identifier}
            items.append({**base, "id": identifier, "domain": "real-screen", "task": "page",
                          "score_mode": "annotated-recall", "regions": regions,
                          **image_fields(image, identifier)})
            for ri in range(min(5, len(regions))):
                index = ri*len(regions)//min(5,len(regions))
                region = regions[index]
                box = [round(v) for v in region["box"]]
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("Degenerate screen crop")
                crop_id = f"{identifier}-crop-{index}"
                items.append({**base, "id": crop_id, "domain": "real-screen-region", "task": "page",
                              "parent_id": identifier, "text": region["text"], "crop_box": box,
                              "score_mode": "transcription", **image_fields(image.crop(box), crop_id)})
            print(f"Prepared {role} screen source {row['screen_id']}", flush=True)
    # Eligibility is fixed from source annotations, never baseline predictions.
    eligible = []
    for identifier in text["imgs"]:
        annotations = [text["anns"][aid] for aid in text["imgToAnns"][identifier]]
        if identifier in excluded_images:
            continue
        if 3 <= len(annotations) <= 30 and all(a["utf8_string"].strip() not in ("", ".") for a in annotations):
            eligible.append(identifier)
    selected = ranked(eligible)[:20]
    remote, archive = None, None
    for n, identifier in enumerate(selected):
        role = "dev" if n < 12 else "final"
        path = DATA / f"textocr-{identifier}.jpg"
        if not path.exists():
            if remote is None:
                remote = RangeReader(TEXT_BASE+"/train_val_images.zip")
                archive = zipfile.ZipFile(remote)
            path.write_bytes(read_member(archive, f"train_images/{identifier}.jpg"))
        image = Image.open(path).convert("RGB")
        source_image = text["imgs"][identifier]
        if image.size != (source_image["width"],source_image["height"]):
            raise ValueError("Source dimensions mismatch")
        regions = []
        for aid in text["imgToAnns"][identifier]:
            a = text["anns"][aid]
            x,y,w,h = a["bbox"]
            regions.append({"annotation_id": aid, "text": a["utf8_string"],
                            "box": [max(0,x),max(0,y),min(image.width,x+w),min(image.height,y+h)],
                            "polygon": a["points"]})
        regions = reading_order(regions)
        name = "textocr-"+identifier
        items.append({"id": name, "source": "textocr", "source_id": identifier,
                      "source_split": "val", "role": role, "group_id": name,
                      "reference_kind": "human", "task": "page", "domain": "full-scene",
                      "score_mode": "transcription", "text": "\n".join(r["text"] for r in regions),
                      "regions": regions, **image_fields(image, name)})
        print(f"Prepared {role} full scene {n+1}/20", flush=True)
    manifest = {"schema": 2, "purpose": "evaluation-only-expansion", "sources": sources, "items": items,
                "pilot_manifest_sha256": sha256(MANIFEST),
                "lineage_sha256": sha256(ROOT/"eval/legacy_scene_lineage.json"),
                "selection": "SHA256-ranked seed tinyocr-expansion-v1; screens 12 valid dev/8 test final; scenes 12 dev/8 final",
                "scene_eligibility": "3-30 annotated regions; no empty or illegible-dot labels; exclude every candidate legacy parent",
                "screen_region_protocol": "All UI-region crops use detection plus recognition: source labels can span multiple lines",
                "archive_bytes_downloaded": remote.bytes_downloaded if remote else 0,
                "limitations": ["Small pilot; screenshot labels cover annotated UI elements, not all text",
                                "Source-image disjointness is checked; app/template-level separation is not established",
                                "Legacy TextOCR parents are candidates from exact annotation text, not proven crop matches",
                                "Full-scene CER includes reading-order differences; no official spotting F-score",
                                "Final is reserved for future use; no baseline predictions generated here"]}
    if existing is not None:
        if {k:v for k,v in manifest.items() if k != "archive_bytes_downloaded"} != {k:v for k,v in existing.items() if k != "archive_bytes_downloaded"}:
            raise ValueError("Restored expansion differs from frozen manifest")
    else:
        write_json(EXPANSION, manifest)
    from .check_expansion import check
    check()


if __name__ == "__main__":
    prepare()
