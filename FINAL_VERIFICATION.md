# Reserved-final evaluation protocol (declared before inference)

Source: user approved the next step in .claude/TODO.md: evaluate the frozen 3.10 MB candidate against the original baseline once, with no final-driven candidate changes.

Models: v6-tiny and the exact v6-w8-rec artifacts/configuration in results/candidate-lock.json. Inputs: all 48 role=final items in eval/expansion-region-v2.json (8 screens, 32 correlated UI regions, 8 full scenes). Development data may be used for warmup only.

One evaluation event, three predeclared repetitions per model/item for timing, alternating baseline/candidate order. Preserve all predictions, references, boxes, timings, provenance and start/completion records. A completed rerun command verifies saved evidence and performs no new inference. Interrupted runs require explicit resume with the same protocol and frozen artifacts.

Before seeing results: compare per-domain normalized character-edit counts and previously recovered screen labels. A transfer check passes only if no transcription domain has more character errors and no baseline-recovered screen label is lost. WER, aggregate label recovery and latency are reported alongside; none substitutes for a failed check. Passing on this small set does not establish broad production readiness. No final-data tuning or label corrections in this task.

| Requirement / risk | Runnable check, project root | Pass criterion | Evidence |
| --- | --- | --- | --- |
| Frozen models, configuration, dependencies and data | `uv run --locked python -m tinyocr.evaluate_final` preflight | Exact hashes/configs match the prior lock and baseline; all source/image checks pass before final inference | Pending |
| Final-only coverage and correct scoring | `uv run --locked python -m unittest discover -s tests -v` and final verifier | Exactly 48 IDs/model; no dev IDs in results; same normalization and partial-screen recall; counters and summaries recompute | Pending |
| Preserve first-run evidence | Repeat `evaluate_final` after completion | Refuses changed protocol; completed run verifies rather than overwrites or re-infers; journal completeness checked | Pending |
| Actual held-out comparison | `uv run --locked python -m tinyocr.evaluate_final --verify-only` | Both models executed on every reserved item; all comparisons reported, including regressions | Pending |
| Preserve previous artifacts | Existing candidate/dev verifiers | No changes to model files, candidate lock, prior manifests or prior reports | Pending |

Measurement integrity and model transfer are separate conclusions. A fully verified evaluation can demonstrate a failed model-quality transfer check.
