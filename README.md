# tinyocr

English-first OCR research for small, fast CPU deployment. Started 2026-10-23 after tinyasr: measure real-world behavior before training a student.

The working implementation is an evaluation harness and two pretrained PaddleOCR baselines via RapidOCR/ONNX Runtime. A custom TinyOCR model has not been trained.

## Run

From this directory, with uv and Python 3.12:

```powershell
uv sync --locked
uv run --locked python -m tinyocr.prepare --n 20
uv run --locked python -m tinyocr.validate
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m tinyocr.evaluate --model v6-tiny
uv run --locked python -m tinyocr.evaluate --model v5-english
uv run --locked python -m tinyocr.verify_run --model v6-tiny
uv run --locked python -m tinyocr.verify_run --model v5-english
```

Preparation downloads a small pinned evaluation sample, not entire datasets. It resumes interrupted image downloads, checks source revisions and image hashes, and reuses a complete cache without network access. A frozen manifest with missing images can be restored; mismatched pixels/labels fail rather than silently replacing the evaluation. HF viewer availability for the pinned revisions is required on first download. Synthetic fixtures currently use Windows Segoe UI (or the local Inter font); the manifest pins the font hash, so rebuilding on another machine may require the same font.

`eval/manifest.json` freezes sample selection, references, provenance, dimensions and pixel-file hashes. Downloaded images/source cards are in `data/eval/`; weights are in `models/`; complete prediction and timing reports are in `runs/`. Tracked report summaries are in `results/`. Dependencies are pinned by `uv.lock`.

## What is measured

| Domain | Samples | Task |
| --- | ---: | --- |
| SROIE receipt pages | 20 | Detection + recognition + row ordering |
| SROIE annotated regions | 200 | Recognition on ground-truth crops from those same 20 pages |
| TextOCR scene words | 20 | Recognition on public human-labeled crops |
| Synthetic screen fixtures | 6 | Authored light/dark text at 12/16/24 px; smoke checks only |
| Blank control | 1 | Empty output / false positive characters |

Receipt crops are correlated with their parent pages; they are not 200 independent documents. The scene mirror is a filtered crop collection with no original parent-image IDs, so leakage against future training data cannot yet be ruled out. The pilot must never enter training. Since we inspect its errors, it is development evidence; reserve a separate untouched final test set later.

CER/WER use summed edit counts divided by summed reference lengths, reported per domain. Whitespace is collapsed, Unicode is NFC-normalized, and case is folded. Digits and punctuation are retained. Missing predictions count as deletions, extra text as insertions, and error rates may exceed 100%. Case-sensitive scores are also saved, but SROIE's uppercased labels make those diagnostic only. Blank references have undefined CER/WER and an explicit false-positive count. No combined score mixes crops, pages and synthetic checks.

Page text is ordered geometrically by rows in both reference and output. Page CER therefore includes ordering errors; it is not a localization F-score. No rotation classifier is enabled. Detector inputs have a maximum side of 960 px, recognition batch size is one, and both ORT thread counts and OpenCV threads are explicitly checked/set to one. Timing includes local file decode, preprocessing and postprocessing; it excludes initialization/downloads and metric computation. Each image is run three times after warmup; p50/p95 are over per-image median durations. This is desktop CPU evidence, not phone performance.

## Real-screen and full-scene expansion

[Expanded results and commands](EXPANSION.md) cover actual mobile-app screenshots and full scenes: 79 development items and 48 reserved final items. The expanded evaluator runs development inputs only. [Label audit](eval/LABEL_AUDIT.md) records review findings without changing original pilot references.

## Next experiment

See [baseline results](BASELINES.md), [expanded results](EXPANSION.md) and [.claude/TODO.md](.claude/TODO.md). Next: try compressing the v6 baseline against the fixed development sets. Before training: finish independent human label adjudication, independent receipt-photo/rotation coverage, source lineage and training-data review.

Existing constraints remain: no paid API runs; GPU 1 free; any future training uses at most half of GPU 0 and about 20 GB RAM. Startup runs use CPU only.
