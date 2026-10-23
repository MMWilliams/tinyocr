# Tiny OCR compression experiment — 2026-10-23

The selected development candidate is **v6-w8-rec**: original float32 detector plus an int8-weight recognizer. Its two ONNX files total **3,099,213 bytes (3.10 MB)**, down from 6,319,431 bytes (51.0% smaller). It meets the <=3.5 MB artifact target.

These are compressed pretrained PaddleOCR weights, not a newly trained model. No training, calibration, paid API call or GPU allocation was used. Original model files, dependency lock, pilot references and baseline results remain unchanged. All 48 reserved final items were untouched during selection; the subsequent [frozen final comparison](FINAL_RESULTS.md) passed with unchanged accuracy aggregates. That partition is now consumed.

## Accuracy on development data

| Metric | Original v6 tiny | v6-w8-rec |
| --- | ---: | ---: |
| Receipt-page CER (20 pages) | 11.743% | 11.718% |
| Receipt-region CER (200 crops) | 7.167% | 6.914% |
| Legacy scene-crop CER (20 crops) | 26.984% | 26.190% |
| Full-scene CER (12 images) | 32.028% | 31.748% |
| Screenshot-region CER (55 regions) | 3.985% | 3.856% |
| Screen labels recovered (12 screens) | 99/113 | 100/113 |
| Previously recovered screen labels lost | — | 0 |
| Synthetic screen CER / blank false positives | 0% / 0 | 0% / 0 |

Every domain meets the predefined character-error gate. This does not mean every prediction improved: receipt-page **WER increases from 31.508% to 31.658%** (three additional word edits), even though character edits fall by three. Word error was not the acceptance metric; the regression is retained in the reports. Small gains from rounding should not be interpreted as a trained quality improvement.

The final benchmark has not been used to select among candidates, so these remain development results. Existing dataset limitations and annotation caveats in EXPANSION.md still apply.

## Candidates tested

| Candidate | Artifact size | Accuracy outcome |
| --- | ---: | --- |
| v6-w8: both modules int8 weights | 1.87 MB | Rejected: full-scene character edits 229 → 239 |
| v6-w16: both modules float16 weights | 3.30 MB | Rejected: receipt-page edits 1366 → 1369; screenshot-region edits 31 → 32 |
| v6-w8-rec: float32 detector + int8 recognizer | 3.10 MB | Passes all character-error, screen-label and size gates |

All attempts and original measurements remain available. Detector precision was retained to avoid shifting text boxes before recognition. Selection used the fixed development sets; no references, thresholds, crop rules or sample membership were changed to improve acceptance.

## Storage and runtime behavior

Two nine-pass comparisons (normal and reversed model-loading order) ran all 326 development inputs: **5,868 paired observations / 11,736 OCR calls**. Per-domain median p50 ratios ranged from 0.984–1.017 (candidate/original). Neither run flagged a domain slower on every pass under the predefined screen. Tail latency was noisier: full-scene median p95 ratios were 1.059 and 1.025 respectively. Speed is broadly similar on this desktop, not proven identical; small regressions and different-device behavior remain possible. Both raw runs and per-pass ratios are retained.

Eligible Conv/MatMul/Gemm weights are quantized per output channel with symmetric int8 values. Constant Cast/Mul nodes recover float32 weights. Biases, normalization values, other consumers and small weights remain unchanged. This follows ONNX's [linear dequantization arithmetic](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html), expressed with Cast/Mul so the original opset 11 graph does not need conversion. Unsupported/shared consumer patterns are left alone.

Actual ONNX Runtime 1.31.0 optimized-graph inspection confirmed that the recognizer's added weight-decoding nodes were folded at initialization. Execution remains float32. This is a model-file size reduction; **no RAM reduction or integer-compute speedup is claimed**. Runtime libraries are not included in the 3.10 MB figure. CPU timing evidence is recorded separately in `results/paired-v6-w8-rec.json`.

The original PaddleOCR/RapidOCR Apache-2.0 model terms continue to apply. Export provenance, source/output SHA-256 hashes, tensor shapes and maximum weight error are in `results/export-v6-w8-rec.json`.

## Run locally

The selected files are under `models/v6-w8-rec/`.
Their hashes and pipeline configuration remain frozen in `results/candidate-lock.json`; the completed final evaluation checked that exact lock.

```powershell
uv run --locked python -m tinyocr.ocr path\to\image.png
uv run --locked python -m tinyocr.ocr path\to\image.png --output runs\prediction.json
```

Rebuild and verify from the cached original baseline:

```powershell
uv run --locked --with onnx==1.23.2 python -m tinyocr.compress --mode w8-rec
uv run --locked python -m unittest discover -s tests -v
uv run --locked --with onnx==1.23.2 python -m unittest discover -s tests_export -v
uv run --locked python -m tinyocr.evaluate --model v6-w8-rec
uv run --locked python -m tinyocr.evaluate_expansion --model v6-w8-rec
uv run --locked python -m tinyocr.benchmark_pair --model v6-w8-rec
uv run --locked python -m tinyocr.benchmark_pair --model v6-w8-rec --reverse-load
uv run --locked python -m tinyocr.compare_compression --model v6-w8-rec
```

ONNX 1.23.2 is a pinned temporary export/test dependency supplied by `uv --with`; it does not alter the locked inference environment. The comparison command requires accuracy/size acceptance and the nine-pass paired speed screen. The speed screen flags any domain whose p50 or p95 is slower in all nine passes, retaining all ratios; passing is evidence against a consistent regression on this machine, not proof of equal speed everywhere.
