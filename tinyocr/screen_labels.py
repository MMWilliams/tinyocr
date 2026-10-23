"""Extract OCR labels, not generated icon/image descriptions, from ScreenAI."""
import re

from .common import normalize, reading_order

ELEMENT = re.compile(r"(?:^|[,()]\s*)([A-Z_]+)\s+(.*?)\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)(?=\s*[,()]|$)")
TEXT_TYPES = {"TEXT", "BUTTON", "LABEL", "TEXT_INPUT", "CHECKBOX", "RADIO_BUTTON", "SWITCH"}


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def area(box):
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def parse_screen(label, width, height):
    regions = []
    for match in ELEMENT.finditer(label):
        kind, text, x0, x1, y0, y1 = match.groups()
        text = text.strip()
        if kind not in TEXT_TYPES or not text:
            continue
        x0, x1, y0, y1 = map(int, (x0, x1, y0, y1))
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("Invalid normalized screen coordinates")
        box = [x0*width/1000, y0*height/1000, x1*width/1000, y1*height/1000]
        # Hierarchical annotations can repeat the same label on a parent and child.
        if any(normalize(r["text"]) == normalize(text) and
               intersection(r["box"], box) / min(area(r["box"]), area(box)) >= 0.8 for r in regions):
            continue
        regions.append({"text": text, "box": box, "element_type": kind})
    if not regions:
        raise ValueError("No OCR text labels parsed")
    return regions


def annotated_recall(reference_regions, predicted_regions):
    """Exact normalized label recovery at the labeled location; no precision claim.

    A prediction must overlap at least half the smaller box area. A labeled phrase
    may be contained in a longer predicted line, but must match token boundaries.
    This deliberately does not claim to be a standard detection mAP metric.
    """
    recovered = 0
    for ref in reference_regions:
        wanted = normalize(ref["text"])
        candidates = [p for p in predicted_regions if area(p["box"]) > 0 and
                      intersection(ref["box"], p["box"]) / min(area(ref["box"]), area(p["box"])) >= 0.5]
        pattern = r"(?<!\w)" + re.escape(wanted) + r"(?!\w)"
        joined = normalize(" ".join(p["text"] for p in reading_order(candidates)))
        recovered += int(bool(re.search(pattern, joined)))
    return {"recovered": recovered, "annotated": len(reference_regions)}
