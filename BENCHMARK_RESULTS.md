# Broader public OCR benchmark comparison

Completed 2026-10-23 (local time; run timestamps are 2026-10-24 UTC). **Verified comparison:** each frozen model completed 5,089 images across SROIE, TextOCR and HierText. The 3.10 MB compressed candidate is 50.957% smaller, with E2E F1 lower by 0.079-0.257 percentage points. Latency is mixed: candidate p50 is 21.23% higher on receipts, 16.16% higher on HierText and 6.91% lower on TextOCR in this single pass. Peak process RAM is effectively unchanged at about 500 MiB. This evaluates frozen pretrained `v6-tiny` and compressed `v6-w8-rec`; no TinyOCR student was trained.

## Coverage and provenance

The comparison uses every entry in three pinned public splits after exclusions defined before inference. These are comparative development results, not a new untouched final test. Original model weights, thresholds, dependency lock and consumed final-test evidence remain unchanged.

| Source / split | Published images | Prior-parent exclusions | Evaluated images | Legible regions | Ignored regions | Pages eligible for CER/WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SROIE mirror / test | 361 | 20 | 341 | 18,418 | 0 | 341 |
| TextOCR / validation | 3,124 | 97 | 3,027 | 103,502 | 41,040 | 748 |
| HierText / validation | 1,724 | 3 | 1,721 | 177,889 | 31,936 | 461 |

Total: 5,089 images per model, with 120 prior-parent exclusions. The exclusion set covers all 20 prior SROIE pages, all prior full-scene parents and every possible legacy TextOCR crop parent. Decoded-pixel checks found no additional overlap with the prior evaluation/SRD catalogs. The selected SROIE split contains two exact duplicate image pairs, leaving 339 distinct decoded receipt images; these published entries remain included. HierText and TextOCR share 23 source scene IDs at different decoded resolutions. Scores are reported separately, never pooled.

The source audit retains 495 HierText and 713 TextOCR polygons extending outside the image frame, nine invalid TextOCR polygons, and one zero-area SROIE region. These were found before inference. Coordinates and text were not edited: the official evaluator applies its existing geometry repair, and the zero-area reference remains an unmatched item in the denominator. Source-declared and decoded image dimensions match. Four TextOCR images declare EXIF orientation 1; the rest have no orientation tag. Of 55 source SROIE scans above 25 MP, 53 remain after prior-parent exclusions; the amended 40 MP cap accommodates this stratum without filtering images for size.

Reproducible records: [manifest](eval/benchmarks-v1.json), [source downloads and hashes](results/benchmark-sources.json), [source annotation audit](results/benchmark-source-audit.json), and [TextOCR publisher snapshot](results/benchmark-rights.json). Downloaded images remain local under `data/benchmarks-v1/`.

| Source | Frozen source and publisher terms |
| --- | --- |
| SROIE | [jsdnrs mirror at `bffe40c`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE/tree/bffe40c26759f3376ec2b3ae9031dbba54cd587c); its card declares CC BY 4.0. Mirror corrections and region boxes are retained. |
| TextOCR | [Official validation release](https://textvqa.org/textocr/dataset/); publisher bundle declares CC BY 4.0. Underlying image rights also apply; image-specific reuse/training review remains separate. |
| HierText | [Official repository at `70b6620`](https://github.com/google-research-datasets/hiertext/tree/70b6620b2b112597d8219e11eee9773a1403827c); dataset/repository declares CC BY-SA 4.0. Upstream evaluator, license and attribution are preserved in `third_party/hiertext/`. |

## Accuracy

All scores below are percentages; higher is better. PQ multiplies F1 by mean matched-box IoU, incorporating localization tightness. Full precision, recall, counts and scores are in [the machine-readable comparison](results/benchmark-comparison.json).

| Source | Model | Detection F1 | Detection PQ | E2E F1 | E2E PQ |
| --- | --- | ---: | ---: | ---: | ---: |
| SROIE regions | `v6-tiny` | 88.584 | 67.118 | 41.036 | 30.104 |
| SROIE regions | `v6-w8-rec` | 88.598 | 67.123 | 40.779 | 29.911 |
| TextOCR words | `v6-tiny` | 58.515 | 40.087 | 43.412 | 30.067 |
| TextOCR words | `v6-w8-rec` | 58.364 | 40.000 | 43.334 | 30.028 |
| HierText words | `v6-tiny` | 48.392 | 30.827 | 40.330 | 25.745 |
| HierText words | `v6-w8-rec` | 48.252 | 30.745 | 40.204 | 25.667 |

The candidate loses 0.257 percentage points of E2E F1 on SROIE, 0.079 on TextOCR and 0.127 on HierText. These are small observed losses, not exact accuracy preservation or a statistical equivalence claim. Absolute scene-text performance remains limited.

The unchanged HierText polygon evaluator uses mutual-best polygon matches at IoU >= 0.5. End-to-end (E2E) matches also require exact, case-sensitive text. Predictions overlapping an ignored region over at least half their area are ignored under the evaluator's default rules. SROIE uses native predicted lines against text regions; TextOCR and HierText use RapidOCR's native word boxes. Word boundaries never use reference labels. Six development-fixture/model checks established that requesting words leaves line text and polygons unchanged.

Only HierText uses its native word evaluator here, on an overlap-filtered validation subset. SROIE and TextOCR scores use a shared HierText protocol and must not be presented as native leaderboard results. SROIE's uppercased reference strings make case differences especially consequential for exact-text E2E scoring. Word-box detection can also change with recognition/alignment even though the detector weights are identical.

Post-run geometry inspection found four baseline and seven candidate zero-area native word boxes, with no invalid line polygons. One zero-area word per model belongs to SROIE, where only lines are scored; three baseline and six candidate word boxes therefore reach word scoring. The official evaluator logs these degeneracies and leaves them as unmatched predictions. None were deleted or regenerated. A regression test added after observing the warnings confirms their precision penalty without changing scorer code or predictions. Empty page outputs were also retained: baseline/candidate counts are 0/0 SROIE, 160/172 TextOCR and 17/18 HierText.

## Supplementary page-order diagnostics

All values are percentages; lower is better. Coverage is restricted to the complete-reference page counts in the coverage table.

| Source | Model | Normalized CER | Normalized WER | Case-sensitive CER | Case-sensitive WER |
| --- | --- | ---: | ---: | ---: | ---: |
| SROIE regions | `v6-tiny` | 12.452 | 33.115 | 33.944 | 56.407 |
| SROIE regions | `v6-w8-rec` | 12.549 | 33.660 | 34.008 | 56.684 |
| TextOCR words | `v6-tiny` | 30.628 | 52.930 | 31.181 | 53.634 |
| TextOCR words | `v6-w8-rec` | 30.678 | 52.981 | 31.218 | 53.672 |
| HierText words | `v6-tiny` | 20.833 | 34.396 | 21.149 | 34.757 |
| HierText words | `v6-w8-rec` | 20.907 | 34.416 | 21.210 | 34.751 |

Normalized CER increases in all three sources. The largest normalized WER change is on receipts: 33.115% to 33.660%, an increase of 0.545 percentage points.

CER/WER are summed edit counts divided by summed reference character/word counts. Normalized scores use NFC, collapsed whitespace and case folding; punctuation and digits remain. Case-sensitive scores are also retained. Both reference and prediction text use the existing geometric row-order heuristic. These diagnostics include reading-order errors and are not canonical scene-text recognition metrics.

Pages containing any ignored region are excluded from these diagnostics: 0 SROIE, 2,279 TextOCR and 1,260 HierText pages. Their detection/E2E scores remain included. The diagnostic subset therefore differs substantially from the full scene benchmark and may favor easier pages. Non-English references remain included; the recognizer is English-first.

## CPU latency and process memory

| Source | Baseline p50 / p95 (ms) | Candidate p50 / p95 (ms) | p50 change | p95 change |
| --- | ---: | ---: | ---: | ---: |
| SROIE regions | 217.35 / 346.84 | 263.50 / 436.40 | +21.23% | +25.82% |
| TextOCR words | 109.45 / 303.65 | 101.89 / 254.12 | -6.91% | -16.31% |
| HierText words | 166.23 / 383.70 | 193.09 / 495.64 | +16.16% | +29.17% |

The candidate is slower on receipts and HierText in this pass, and faster on TextOCR. This run supplies no general speedup claim. The large per-source timing differences warrant a repeated paired benchmark with the same word-output path before drawing a deployment latency conclusion; no extra runs were used to select more favorable timings.

| Resource | Baseline | Candidate |
| --- | ---: | ---: |
| Model artifact bytes | 6,319,431 | 3,099,213 |
| Model size (decimal MB) | 6.32 | 3.10 |
| Initialization (ms) | 119.08 | 206.90 |
| Peak process working set (MiB) | 498.84 | 500.10 |
| Complete uninterrupted memory coverage | Yes | Yes |

Model artifacts shrink by 50.957%; process RAM does not decrease. Both models complete in distinct, sequential processes: baseline PID 62236, 2026-10-24 03:02:11-03:16:14 UTC; candidate PID 47168, 03:16:35-03:31:22 UTC. Neither run resumed from an earlier segment.

Hardware: AMD Ryzen 9 9900X, Windows, Python 3.12.9. Both ORT sessions and OpenCV use one thread, CPUExecutionProvider, detector maximum side 960, recognition batch size one and no orientation classifier. Each model runs in its own sequential process with two warmups on an existing development receipt and one measured call per new image. Timing includes file decode, OCR, geometric row ordering and native word-box postprocessing; model initialization, download, scoring and journal writes are outside the timed call. p95 is the nearest-rank 95th percentile across images.

Both processes recorded identical package versions: RapidOCR 3.10.0, ONNX Runtime 1.31.0, NumPy 2.5.4, Pillow 12.3.0, Shapely 2.2.0 and absl-py 2.3.1. The original lock and separate benchmark dependency overlay are hash-pinned in both start records.

Peak working set covers the entire model process, including imports, manifest, runtime, decoded images, models and output adaptation. It is not model-only RAM. This is a single desktop pass, not repeated latency certification or phone evidence. The candidate stores int8 recognizer weights but executes FP32 math; reduced file size does not imply integer-compute acceleration or proportional RAM savings.

## Verification evidence

The immutable [pre-inference protocol](BENCHMARK_VERIFICATION.md) retains its declaration-time pending cells; outcomes belong here so its run-pinned hash stays unchanged. Commands run from `C:\Users\reese\tinyocr`. Below, `BENCH` means `uv run --locked --with-requirements benchmark-requirements.txt`; optional benchmark dependencies are isolated from `uv.lock`.

| Obligation | Executed command | Outcome |
| --- | --- | --- |
| Full source reconstruction, hashes, exclusions and pixels | `BENCH python -m tinyocr.prepare_benchmark --verify-only` | Passed, exit 0; 5,089 selected / 120 excluded / 23 shared scene IDs |
| Source geometry and annotation audit | `BENCH python -m tinyocr.audit_benchmark` | Passed, exit 0; saved diagnostics reproduced without label changes or OCR |
| Scoring, ignored regions, geometry, journals and archive safety | `BENCH python -m unittest discover -s tests_benchmark -v` | Passed, 14 tests, zero skipped; includes observed degenerate-prediction handling |
| Word adapter preserves line output | `BENCH python -m tinyocr.benchmark --check-adapter` | Passed, six fixture/model pairs; no new benchmark images used |
| Baseline predictions | `BENCH python -m tinyocr.benchmark --model v6-tiny` | Passed, 5,089 first predictions in one process |
| Candidate predictions | `BENCH python -m tinyocr.benchmark --model v6-w8-rec` | Passed, 5,089 first predictions in one process |
| Completed runs do not infer again | Repeat both completed `--model` commands above | Passed, both report completed journals verified with no inference |
| Official scoring | `BENCH python -m tinyocr.benchmark --score` | Passed, exit 0; all 10,178 predictions scored |
| Fresh-process full recomputation and integrity checks | `BENCH python -m tinyocr.benchmark --verify-only` | Passed, exit 0; exact stored comparison reproduced without inference |
| Existing unit suite | `uv run --locked python -m unittest discover -s tests -v` | Passed, 53 tests, zero skipped |
| Existing SRD intake | `uv run --locked python -m tinyocr.intake --verify-only` | Passed, 200 candidate entries / 198 unique images; benchmark approval remains false |
| Existing robustness evidence | `uv run --locked python -m tinyocr.evaluate_robustness --verify-only` | Passed, 80 items per model; original failures retained |
| Consumed final evidence | `uv run --locked python -m tinyocr.evaluate_final --verify-only` | Passed, 48 items per model; frozen transfer check unchanged |
| Patch whitespace | `git diff --check` | Passed, exit 0; only existing Windows line-ending notices |

Documentation links, completion markers, new-code syntax and whitespace checks passed. A path-scoped `git diff --exit-code` confirmed that `uv.lock`, `pyproject.toml`, baseline/common/metrics code, the original final protocol/results and candidate lock remain unchanged.

Manual inspection covered publisher snapshots and evaluator matching/ignore/geometry code, the first stored source image and reference regions in each dataset, the six adapter records, and representative saved line/word coordinates. It does not replace independent human label adjudication. Empty detections remain in prediction journals and scores.

Supplementary assertions cross-checked every scored ground-truth total and complete-reference page count against the independently saved source audit, confirmed F1 = 2TP/(ground-truth + prediction count), and checked that process IDs differ, run time windows do not overlap, and both completion records have zero resumed items. All passed. Evaluator geometry warnings described above are retained and covered by explicit scoring tests; they are not skipped examples or failed commands.

First predictions, start fingerprints and process completion records are in `runs/benchmarks-v1/`. A completed `--model` invocation verifies saved evidence without running inference. Do not delete journals or casually alter fingerprinted code/protocol to rerun this experiment. Recompute scores with `--verify-only`; the machine-readable comparison is [results/benchmark-comparison.json](results/benchmark-comparison.json).

Public-pretraining overlap, near-duplicate source lineage beyond the recorded checks, independent real capture coverage and fully human-adjudicated fresh labels remain open. These measurements do not establish readiness for release or create a fresh final partition. The separate collection backlog remains in [.claude/TODO.md](.claude/TODO.md).
