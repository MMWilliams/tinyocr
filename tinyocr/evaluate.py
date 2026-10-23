"""Measure warm single-thread CPU OCR on the frozen pilot, without training."""
import argparse
import importlib.metadata
import os
import platform
import statistics
import time
from datetime import datetime, timezone

from .baseline import Baseline
from .common import MANIFEST, ROOT, sha256, write_json
from .metrics import aggregate, counts
from .validate import validate


def evaluate(name, repeats):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    manifest = validate()
    started = time.perf_counter()
    model = Baseline(name)
    load_seconds = time.perf_counter() - started
    for task in ("crop", "page"):
        item = next(x for x in manifest["items"] if x["task"] == task)
        for _ in range(2):
            model.predict(ROOT / item["image"], task)
    rows = []
    for i, item in enumerate(manifest["items"]):
        samples = []
        prediction, regions = None, None
        for _ in range(repeats):
            begin = time.perf_counter()
            text, detected = model.predict(ROOT / item["image"], item["task"])
            samples.append(1000 * (time.perf_counter() - begin))
            if prediction is not None and prediction != text:
                raise RuntimeError(f"Nondeterministic prediction: {item['id']}")
            prediction, regions = text, detected
        rows.append({"id": item["id"], "domain": item["domain"], "task": item["task"],
                     "reference": item["text"], "prediction": prediction,
                     "regions": regions, "latency_ms": samples,
                     "normalized": counts(item["text"], prediction),
                     "case_sensitive": counts(item["text"], prediction, casefold=False)})
        if (i+1) % 25 == 0:
            print(f"{name}: {i+1}/{len(manifest['items'])}", flush=True)
    domains = {}
    for domain in sorted({r["domain"] for r in rows}):
        selected = [r for r in rows if r["domain"] == domain]
        latencies = sorted(statistics.median(r["latency_ms"]) for r in selected)
        domains[domain] = {"normalized": aggregate([r["normalized"] for r in selected]),
                           "case_sensitive": aggregate([r["case_sensitive"] for r in selected]),
                           "p50_ms": statistics.median(latencies),
                           "p95_ms": latencies[__import__("math").ceil(len(latencies)*0.95)-1],
                           "mean_ms": statistics.mean(latencies)}
    result = {"model": name, "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": sha256(MANIFEST), "lock_sha256": sha256(ROOT / "uv.lock"),
              "environment": {"platform": platform.platform(), "cpu": platform.processor(),
                              "cpu_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
                              "python": platform.python_version(),
                              "packages": {p: importlib.metadata.version(p) for p in
                                           ("rapidocr", "onnxruntime", "opencv-python-headless", "numpy", "pillow")}},
              "load_seconds_including_possible_download": load_seconds,
              "timing": {"repeats": repeats, "warmups_per_task": 2,
                         "includes": "file read/decode, preprocessing, detection for pages, recognition, postprocessing, reading order",
                         "excludes": "model loading/downloads, scoring, report serialization",
                         "percentiles": "across per-item medians; nearest-rank p95"},
              "model_details": model.metadata(), "domains": domains, "predictions": rows,
              "limitations": ["Small diagnostic pilot, not a final test set or release gate",
                              "Screens are synthetic raster fixtures, not real application screenshots",
                              "Scene set contains word crops only; no scene detection evaluation",
                              "No guarantee that public baseline training excluded these public benchmarks",
                              "Page CER includes row-order disagreements; no detection F-score yet",
                              "Case-sensitive receipt scores are diagnostic only: source labels are uppercased"]}
    path = ROOT / f"runs/{name}.json"
    write_json(path, result)
    print(f"Saved {path}")
    for domain, score in domains.items():
        metric = score["normalized"]
        print(domain, "n=", metric["n"], "CER=", metric["cer"], "WER=", metric["wer"], "p50_ms=", round(score["p50_ms"], 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["v6-tiny", "v5-english"], default="v6-tiny")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    evaluate(args.model, args.repeats)
