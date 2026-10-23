"""Cache small eval-only samples; no training data and no paid APIs."""
import argparse
import io
import json
import time
from datetime import datetime, timezone

import requests
from PIL import Image, ImageDraw, ImageFont

from .common import MANIFEST, ROOT, reading_order, sha256, write_json

SOURCES = {
    "receipts": {"repo": "jsdnrs/ICDAR2019-SROIE", "split": "test", "license": "CC-BY-4.0 (dataset card)",
                 "revision": "bffe40c26759f3376ec2b3ae9031dbba54cd587c"},
    "scene": {"repo": "MiXaiLL76/TextOCR_OCR", "split": "test",
              "license": "Original TextOCR: CC-BY-4.0; mirror metadata says MIT (conflict; not cleared for redistribution)",
              "revision": "26a2f865d6d131fd296ee9984d3d29cfdf887d54"},
}


def get(url, params=None):
    for attempt in range(8):
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 429 or r.status_code >= 500:
                r.raise_for_status()
            else:
                r.raise_for_status()
                return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            if isinstance(exc, requests.HTTPError) and r.status_code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 7:
                raise
            delay = min(5 * 2 ** attempt, 45)
            print(f"Transient download error; retrying in {delay}s", flush=True)
            time.sleep(delay)


def fetch_row(source, offset):
    result = get("https://datasets-server.huggingface.co/rows", {
        "dataset": source["repo"], "config": "default", "split": source["split"],
        "offset": offset, "length": 1,
    }).json()
    if result.get("partial") or result["rows"][0].get("truncated_cells"):
        raise ValueError("Incomplete upstream annotation")
    row = result["rows"][0]["row"]
    asset = row["image"]
    if isinstance(asset, list):
        asset = asset[0]
    # Dataset-server URLs identify the exact source revision. Never mix revisions.
    if source["revision"] not in asset["src"]:
        raise ValueError("Dataset viewer revision differs from pinned source; retry later")
    return row, asset["src"], result["num_rows_total"]


def save_image(image, filename):
    path = ROOT / "data/eval/images" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")
    return {"image": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
            "width": image.width, "height": image.height}


def synthetic_screens():
    """Controlled raster UI smoke fixtures; explicitly NOT real screenshots."""
    texts = ["Project settings", "Export selected files", "Invoice #A-1042", "Total: $127.50",
             "Search files and folders", "Save changes", "Upload complete: 12 files", "Cancel"]
    items = []
    font_path = ROOT.parent / "AppData/Local/Microsoft/Windows/Fonts/Inter-Regular.ttf"
    if not font_path.exists():
        font_path = __import__("pathlib").Path("C:/Windows/Fonts/segoeui.ttf")
    for i, (size, dark) in enumerate([(12, False), (16, False), (24, False), (12, True), (16, True), (24, True)]):
        font = ImageFont.truetype(str(font_path), size)
        im = Image.new("RGB", (720, 450), "#171a21" if dark else "#f7f8fa")
        draw = ImageDraw.Draw(im)
        regions = []
        for j, text in enumerate(texts):
            xy = (30, 20 + 50 * j)
            box = list(draw.textbbox(xy, text, font=font))
            draw.text(xy, text, fill="#e8edf5" if dark else "#202631", font=font)
            regions.append({"text": text, "box": box})
        items.append({"id": f"screen-smoke-{i}", "domain": "synthetic-screen", "task": "page",
                      "source": "local-ui-smoke", "source_id": str(i), "split": "smoke",
                      "reference_kind": "authored", "text": "\n".join(texts), "regions": regions,
                      "font_sha256": sha256(font_path), **save_image(im, f"screen-{i}.png")})
    im = Image.new("RGB", (720, 450), "white")
    items.append({"id": "blank-0", "domain": "blank-control", "task": "page", "source": "local-ui-smoke",
                  "source_id": "blank", "split": "smoke", "reference_kind": "authored", "text": "",
                  **save_image(im, "blank.png")})
    return items


def prepare(n):
    if n < 1:
        raise ValueError("n must be positive")
    existing = None
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing["n_per_source"] != n:
            raise ValueError("Existing frozen pilot has a different n; use a separate manifest for a new evaluation")
        if all((ROOT / item["image"]).exists() for item in existing["items"]):
            from .validate import validate
            validate()
            print("Using validated frozen cache; no downloads")
            return
        print("Restoring missing images for the frozen manifest", flush=True)
    sources, items = {}, []
    for key, source_config in SOURCES.items():
        source = dict(source_config)
        source["url"] = f"https://huggingface.co/datasets/{source['repo']}/tree/{source['revision']}"
        card = get(f"https://huggingface.co/datasets/{source['repo']}/raw/{source['revision']}/README.md").text
        card_path = ROOT / f"data/eval/sources/{key}.md"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(card, encoding="utf-8")
        _, _, total = fetch_row(source, 0)
        source["total_rows"] = total
        if n > total:
            raise ValueError("Requested sample exceeds split")
        sources[key] = source
        for k in range(n):
            offset = k * total // n
            raw_path = ROOT / f"data/eval/sources/{key}-{offset}.json"
            image_path = ROOT / f"data/eval/images/{key}-{offset}.png"
            cached = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
            if cached.get("_source_revision") == source["revision"] and image_path.exists():
                image = Image.open(image_path).convert("RGB")
                if sha256(image_path) != cached.get("_image_sha256"):
                    raise ValueError("Partial download cache checksum mismatch")
                row = cached
            else:
                row, url, _ = fetch_row(source, offset)
                image = Image.open(io.BytesIO(get(url).content)).convert("RGB")
                saved = save_image(image, f"{key}-{offset}.png")
                # Checkpoint without expiring URLs; each resume verifies revision/hash.
                write_json(raw_path, {**{field: value for field, value in row.items() if field != "image"},
                                      "_source_revision": source["revision"], "_image_sha256": saved["sha256"]})
                time.sleep(1)
            item = {"id": f"{key}-{offset}", "source": key, "source_id": str(row.get("key", offset)),
                    "row": offset, "split": source["split"], "reference_kind": "human",
                    "domain": "receipt-page" if key == "receipts" else "scene-crop",
                    "task": "page" if key == "receipts" else "crop",
                    **save_image(image, f"{key}-{offset}.png")}
            if key == "receipts":
                if len(row["words"]) != len(row["bboxes"]):
                    raise ValueError("Mismatched receipt labels and boxes")
                regions = [{"text": t, "box": b} for t, b in zip(row["words"], row["bboxes"])]
                item["regions"] = reading_order(regions)
                item["text"] = "\n".join(r["text"] for r in item["regions"])
                # Separate recognition on GT regions from end-to-end OCR on the page.
                for j in range(min(10, len(regions))):
                    ri = j * len(regions) // min(10, len(regions))
                    region = regions[ri]
                    crop = image.crop(tuple(region["box"]))
                    items.append({"id": f"receipt-crop-{offset}-{ri}", "source": key,
                                  "source_id": item["source_id"], "parent_id": item["id"],
                                  "region_index": ri, "row": offset, "split": source["split"],
                                  "reference_kind": "human", "domain": "receipt-crop", "task": "crop",
                                  "text": region["text"],
                                  **save_image(crop, f"receipt-crop-{offset}-{ri}.png")})
            else:
                item["text"] = row["text"]
            items.append(item)
            print(f"{key} {k+1}/{n}", flush=True)
    items.extend(synthetic_screens())
    manifest = {"schema": 1, "purpose": "evaluation-only-pilot", "n_per_source": n,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "sampling": "evenly spaced row offsets, no accuracy-based filtering",
                "sources": sources, "items": items}
    if existing is not None:
        if existing["sources"] != sources or existing["items"] != items:
            raise ValueError("Restored data differs from frozen manifest; refusing to replace references")
    else:
        write_json(MANIFEST, manifest)
    from .validate import validate
    validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    prepare(parser.parse_args().n)
