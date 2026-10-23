import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval/manifest.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize(text, casefold=True):
    # Retain digits and punctuation. Only canonical Unicode, whitespace and case.
    text = " ".join(unicodedata.normalize("NFC", text).split())
    return text.casefold() if casefold else text


def reading_order(regions):
    """Row-major order using half the smaller box height as row tolerance.

    Deterministic convenience for receipts, not a general document-layout metric.
    Regions contain text and pixel xyxy box. Applied to both references and output.
    """
    rows = []
    for region in sorted(regions, key=lambda r: ((r["box"][1] + r["box"][3]) / 2, r["box"][0])):
        x0, y0, x1, y1 = region["box"]
        cy, h = (y0 + y1) / 2, y1 - y0
        if rows and abs(cy - rows[-1][0]) <= 0.5 * min(h, rows[-1][1]):
            rows[-1][2].append(region)
        else:
            rows.append([cy, h, [region]])
    return [r for _, _, row in rows for r in sorted(row, key=lambda r: r["box"][0])]
