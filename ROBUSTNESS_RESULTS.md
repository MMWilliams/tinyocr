# Rotation and negative diagnostics — 2026-10-23

**Verified diagnostic implementation and evidence; substantial model limitations found.** Both frozen models struggle with rotated receipts and scenes. Both emit a false-positive `C` on the procedural circle pattern. The compressed model adds errors in some rotated subsets. No model, preprocessing, old reference or acceptance threshold was changed in response.

The suite has **80 development cases per model**: six orientations of four receipt pages, four full scenes and four UI-region crops, plus eight procedural negatives. The 12 selected inputs come from 11 original source parents because two UI crops share one screenshot. These are correlated derivatives of previously exposed data, not a fresh holdout or 80 independent captures. All selection and transformations were declared before inference in [ROBUSTNESS_VERIFICATION.md](ROBUSTNESS_VERIFICATION.md).

## Character error rates

Angles are counterclockwise. Each cell shows original v6-tiny / compressed v6-w8-rec normalized CER. Reference denominators per angle are 2,152 receipt characters, 156 scene characters and 43 UI characters.

| Angle | Receipt pages | Full scenes | UI regions |
| --- | ---: | ---: | ---: |
| 0° | 9.80% / 9.90% | 20.51% / 21.15% | 0% / 0% |
| -15° | 48.61% / 48.75% | 23.08% / 24.36% | 0% / 0% |
| +15° | 42.43% / 42.33% | 21.79% / 20.51% | 0% / 0% |
| 90° | 83.64% / 83.46% | 89.10% / 87.82% | 83.72% / 97.67% |
| 180° | 81.92% / 81.27% | 85.90% / 87.82% | 79.07% / 97.67% |
| 270° | 85.46% / 86.34% | 66.03% / 65.38% | 0% / 0% |

The unchanged pipeline has no explicit orientation classifier. These measurements include the detector's crop handling and geometric row ordering. In particular, rotated multi-line receipt CER can rise when output order no longer matches the original reading order, even if some text is recognized. It is not an isolated recognizer measurement. The strong difference between 90° and 270° UI results is preserved rather than averaged away.

The compressed model's 180° UI subset has 42 edits versus 34 originally; its 90° subset has 42 versus 36. The 270° receipt subset adds 19 edits. Some other subsets improve. Even unrotated subsets have small regressions despite the previously improved full development aggregates; compression did not guarantee every input improves. There was no new quality pass threshold for this diagnostic, and these results do not supersede or broaden the earlier final-set claim.

## Negative controls and timing

Both models return empty text for white, black, gray, gradient, checker, horizontal-line and seeded-noise images. Both predict `C` for the circle-pattern image: **1/8 images with text, one false-positive character**. Blank CER/WER remain undefined rather than zero. These controls are authored graphics; they do not measure false positives on real text-free photographs.

All predictions, normalized/case-sensitive counters, three-repetition timings, model metadata and per-domain p50/p95 values are retained in [results/robustness-v1.json](results/robustness-v1.json). Timing is one desktop comparison with alternating model order and two warmups/model, including decode, preprocessing, inference and postprocessing. It is descriptive, not a speed acceptance result. No claim of phone performance or reduced RAM follows.

## Verification outcomes

| Obligation from pre-inference plan | Actual evidence |
| --- | --- |
| Development-only derivatives | Frozen manifest has 80 dev items, 12 selected inputs / 11 original parents; selection tests reject final and partial-label inputs |
| Deterministic images/provenance | Repeated preparation regenerates and compares every image's pixels; manifest bytes unchanged; corruption unit test rejects altered pixels |
| Correct and complete scoring | Verifier recomputes all 160 model/item records, counters and domain summaries; records eight negatives/model including the false positive |
| Frozen models and old evidence | Preflight checked exact existing weights/configs and one-thread CPU settings; prior final verifier passed unchanged |
| Real execution | Both models completed all 80 cases, three runs/case. Completed comparison verifies without new inference or overwriting report bytes |
| Collection design | [COLLECTION_PLAN.md](COLLECTION_PLAN.md) specifies source evidence, group separation, human adjudication, acquisition and fresh holdout criteria; new collection remains pending |

Commands from the project root:

```powershell
uv run --locked python -m tinyocr.robustness prepare
uv run --locked python -m tinyocr.evaluate_robustness
uv run --locked python -m tinyocr.evaluate_robustness --verify-only
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m tinyocr.evaluate_final --verify-only
```

The final suite passed **41 tests, none skipped**, including six new robustness tests. Separate diagnostic verification and prior-final verification exited 0. The initial PowerShell command wrapper returned 1 when redirected native stderr warnings became `NativeCommandError`; the Python evaluation had reached its completion verifier and saved every result. The saved evidence was subsequently verified with exit 0, without repeating inference. Empty-detection warnings are scored empty outputs, not discarded samples. Repeated preparation plus completed evaluation preserved the SHA-256 of the manifest and report.

The manifest is [eval/robustness-v1.json](eval/robustness-v1.json); generated images are local under `data/robustness/images/`. Before inference, its count field was clarified to distinguish 12 selected inputs from 11 original parents; sample selection and pixels were unchanged. The protocol's pending matrix is preserved as declared; this document records the outcomes.

## Remaining collection work

The source review and collection design are complete, but no new independent receipt photos, difficult-sign photos, complete screen transcripts or real negative images have been collected in this step. No fresh final partition is frozen. The next action is the bounded SRD archive intake and capture/template review described in COLLECTION_PLAN.md; its scanned-image description means it cannot automatically satisfy the receipt-photo requirement. Independent human transcription review, source lineage and rights evidence remain required before training or a new release claim.
