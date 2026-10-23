# Real-screen and full-scene evaluation — 2026-10-23

The expanded development comparison is complete. The final set is reserved and has no model predictions. The original 247-item pilot and its results remain unchanged.

| Development measurement | PP-OCRv6 tiny, 6.32 MB | PP-OCRv5 English, 12.69 MB |
| --- | ---: | ---: |
| Real-screen labeled-location recovery, 12 screens / 113 labels | 99/113 (87.6%) | 102/113 (90.3%) |
| Real-screen latency p50 / p95 | 120.3 / 161.7 ms | 311.7 / 463.5 ms |
| Screenshot-region CER, 55 regions | 3.98% | 11.70% |
| Screenshot-region latency p50 / p95 | 5.24 / 24.08 ms | 13.70 / 47.37 ms |
| Full-scene diagnostic CER, 12 images | 32.03% | 32.87% |
| Full-scene labeled-location recovery, 129 annotations | 60/129 (46.5%) | 68/129 (52.7%) |
| Full-scene latency p50 / p95 | 114.2 / 171.0 ms | 211.9 / 330.2 ms |

CPU/provider/thread controls and three-repetition timing match the initial harness. Numbers come from `results/expanded-*.json`; all 79 dev predictions per model are in `runs/expanded-*.json`. Background load was not controlled, so timing variation between runs is expected. The set is small and should not be treated as a population estimate or a release benchmark.

Screens are actual mobile-app screenshots from [RICO-ScreenAnnotation](https://huggingface.co/datasets/bevaya/RICO-ScreenAnnotation). Every selected annotation was compared exactly with the pinned [original Google release](https://github.com/google-research-datasets/screen_annotation/tree/053a4a1c4737969def99fd36084d7adf3715798f). Google describes the labels as automatically produced, then verified and corrected by human raters. These labels cover UI elements rather than every visible character. We therefore report annotated text recovery at its location, without claiming full-page CER or precision. Extracted text types are TEXT, BUTTON, LABEL, TEXT_INPUT, CHECKBOX, RADIO_BUTTON and SWITCH; overlapping duplicate parent/child labels are collapsed.

Location-aware recovery is a custom diagnostic: predicted boxes must intersect at least half the smaller reference/prediction box area. Candidate predictions are ordered by rows, and the normalized complete reference phrase must occur at word boundaries. This supports multiline labels and labels inside longer detected lines. It is not an official detection or spotting metric, and a spurious large box could overstate coverage; keep raw predictions for review.

Full scenes use the [TextOCR validation annotation mirror](https://huggingface.co/datasets/redactable-llm/TextOCR/tree/0d0a421b9f09f2d00358af52091de57bfaee1684) and original image IDs. Only images with 3–30 regions and no explicitly illegible-dot labels are eligible. The selected examples include physical signs, books, clothing, packaging and small instrument text. Row-order disagreements and label noise contribute to CER. The 7.07 GB source archive was accessed through validated HTTP ranges: only **8.39 MB** of ZIP bytes were transferred for the selected images and directory. Source JSON metadata is cached separately.

## Partition and provenance

| Partition | Screens | Derived screenshot regions | Full scenes | Total evaluation items |
| --- | ---: | ---: | ---: | ---: |
| Development | 12 | 55 | 12 | 79 |
| Reserved final | 8 | 32 | 8 | 48 |

Selection uses a fixed SHA-256 ordering, never model performance. Screens use upstream valid/test splits; full-scene parent IDs are split before derived evaluation. Validation checks parent/source IDs and decoded-pixel hashes across partitions, and each region inherits its parent's split. Exact text lookup identified candidate original parent IDs for all 20 legacy crops; **all** candidates are excluded from the new scene set. This is a conservative exclusion, not proof of original crop identity. App/template-level near-duplicate separation and public-baseline training contamination are not established.

`eval/expansion-region-v2.json` is the active frozen manifest. `eval/expansion.json` preserves the initial diagnostic protocol. Screenshot regions can contain several lines: passing them all directly to a single-line recognizer was a measurement mismatch. Every screenshot region now uses detection plus recognition; image bytes, labels and split assignments are identical. The original recognizer-only results are retained as `expanded-*-recognizer-diagnostic.json`. They are not single-line recognition quality claims. See [label audit](eval/LABEL_AUDIT.md) for review of all legacy crops and development images, including noisy punctuation, neighboring text and a non-English screen label.

## Reproduce

```powershell
uv run --locked python -m tinyocr.expand
uv run --locked python -m tinyocr.check_expansion
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m tinyocr.evaluate_expansion --model v6-tiny
uv run --locked python -m tinyocr.evaluate_expansion --model v5-english
uv run --locked python -m tinyocr.evaluate_expansion --model v6-tiny --verify-only
uv run --locked python -m tinyocr.evaluate_expansion --model v5-english --verify-only
```

Complete caches require no network. Missing PNGs are rebuilt from pinned sources and must exactly match frozen records. Downloaded source images/cards remain local under `data/expansion`; the public source terms still apply. No images or models were uploaded.

Future training inputs must record canonical source, original source ID, source split and decoded-pixel hash. `check_expansion --training-manifest <path>` rejects known source/pixel overlap and reserved TextOCR val, Screen Annotation valid/test, and original receipt/scene test splits. This checks declared provenance; renamed sources and unknown upstream training overlap cannot be ruled out mechanically. No training corpus has yet been approved or consumed.

## Next experiment

Use v6 tiny as the compression reference. First try a weight-only int8 artifact, reusing the tinyasr export approach if compatible, before spending GPU time on a new student. A provisional engineering target is **at most 3.5 MB** for detector plus recognizer, no extra missed screenshot labels versus 99/113, no increased per-domain CER on the current dev sets, and no repeatable latency regression. These are dev acceptance criteria, not claims of generalization. Lock the candidate before evaluating reserved final inputs.

Training readiness remains partial: independent human adjudication of flagged labels, proven legacy crop lineage, broader languages/rotations, independent receipt photos and app/template separation are still backlog work. This turn verified the expanded evaluator, cached data, split guards and measured baselines.
