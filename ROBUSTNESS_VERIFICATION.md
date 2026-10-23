# Development robustness verification

Scope: implement the next backlog design step and measure rotation/blank failure modes without changing models or consuming another final partition. Source: .claude/TODO.md and the user's instruction to proceed after the frozen final comparison. This is development diagnostics, not new independent evidence or a release gate.

Before inference: choose four parents per receipt-page, full-scene and UI-region domain by SHA-256 ordering with seed `tinyocr-robustness-v1`. Original pilot receipts are development; expanded inputs must explicitly have role=dev. Derive 0, -15, +15, 90, 180 and 270 degree versions using Pillow RGB rotation, bicubic sampling, expanded canvas, white fill. Add eight procedural blank/background controls. Parent text remains unchanged: page CER includes failure to restore reading order. No rotated bounding-box recall is claimed. Evaluate the same frozen baseline and candidate, three repetitions, alternating model order; use separate files and preserve first completed results.

| Requirement and source | Command / check | Expected result before execution | Result |
| --- | --- | --- | --- |
| Development-only derivatives (consumed final must remain unchanged) | `uv run --locked python -m tinyocr.robustness prepare`; unit selection tests | 12 eligible dev input images, 72 transformed items, 8 authored negatives; no final-derived images; UI regions may share original screen parents | Pending |
| Deterministic images and provenance (engineering integrity) | Repeat prepare; compare manifest SHA; corruption tests | Source hashes match; regenerated pixels/manifest match; corrupted image fails verification | Pending |
| Metrics retain blank false positives and all failures (existing metrics contract) | Unit tests and `uv run --locked python -m tinyocr.evaluate_robustness --verify-only` | Exactly 80 items/model; counters recomputed, blanks have undefined CER and explicit false-positive counts | Pending |
| Frozen model and prior evidence (candidate lock) | Evaluator preflight and existing final verifier | Same hashes, configs and CPU threads; no edits to prior evidence | Pending |
| Real diagnostic behavior (backlog rotation/blank scope) | `uv run --locked python -m tinyocr.evaluate_robustness` | Execute both models, retain every result including failures; no quality threshold inferred after results | Pending |
| New independent collection (backlog design scope) | Review COLLECTION_PLAN.md against primary sources | Concrete sampling, label, rights, grouping and holdout criteria; no synthetic/derived sample called independent | Pending |

The outcome matrix will be recorded in ROBUSTNESS_RESULTS.md. New data collection and independent human adjudication are separate work; this diagnostic does not complete those requirements.
