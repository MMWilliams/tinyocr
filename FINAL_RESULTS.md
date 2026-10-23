# Frozen final comparison — 2026-10-23

**Verified:** the frozen 3,099,213-byte v6-w8-rec candidate passes the predeclared transfer checks against the 6,319,431-byte original v6-tiny baseline on all 48 reserved items. No candidate, configuration, reference or sample changes followed final inference. This establishes compression transfer on this sample; absolute full-scene accuracy remains poor.

| Normalized accuracy | Original v6-tiny | Frozen v6-w8-rec |
| --- | ---: | ---: |
| Full scenes: 8 images, 447 reference characters | 214 errors / 47.875% CER | 214 errors / 47.875% CER |
| Full-scene WER: 78 reference words | 50 errors / 64.103% | 50 errors / 64.103% |
| UI regions: 32 regions, 317 reference characters | 16 errors / 5.047% CER | 16 errors / 5.047% CER |
| UI-region WER: 54 reference words | 7 errors / 12.963% | 7 errors / 12.963% |
| Screens: 8 images, partial annotation recovery | 46/51 labels / 90.196% | 46/51 labels / 90.196% |
| Previously recovered screen labels lost | — | 0 |

All measured accuracy aggregates match. The gate required no increase in per-domain character edits and no loss of baseline-recovered screen labels. Full-scene annotated-word recovery is also unchanged at 35/78. Full-screen CER is intentionally undefined because the screen annotations are incomplete.

| Desktop CPU latency, p50 / p95 in ms | Original v6-tiny | Frozen v6-w8-rec |
| --- | ---: | ---: |
| Full scenes | 117.61 / 173.03 | 117.74 / 170.93 |
| Real screens | 96.40 / 149.00 | 95.44 / 147.63 |
| UI regions | 4.50 / 14.86 | 4.52 / 17.25 |

Timing uses three predeclared repetitions per model/item, per-image medians, and alternating model order. Development images alone supplied warmup. The UI-region p95 is approximately 16.1% higher (2.39 ms); this single small event is descriptive, not evidence of identical speed. Earlier repeated development timing remains in COMPRESSION.md. Empty-detection warnings were retained as empty predictions and scored, not excluded.

## Verification evidence

The immutable pre-inference plan is [FINAL_VERIFICATION.md](FINAL_VERIFICATION.md); its pending cells describe the state at declaration. Outcomes are recorded here to preserve the protocol hash. All commands below run from the project root under the locked Python 3.12 environment.

| Planned obligation | Executed check | Result |
| --- | --- | --- |
| Frozen artifacts, pipeline and data | `uv run --locked python -m tinyocr.evaluate_final` | Exit 0; exact model/config/dependency/data checks passed before inference; 48 items per model completed |
| Scoring, partition, corruption and repeat-run protection | `uv run --locked python -m unittest discover -s tests -v` | Exit 0; 35 tests passed, none skipped, including 7 final-evaluation tests |
| Complete held-out evidence | `evaluate_final.verify()` under `uv run --locked python` | Exit 0; recomputed all counters/aggregates from 96 saved model/item records; transfer_pass=True |
| First-run evidence preserved | `evaluate_final.evaluate()` after completion, with SHA-256 comparison before/after | Exit 0; verification only; start, journal and report bytes unchanged |
| Prior evidence preserved | `verify_run.verify(name)` and `evaluate_expansion.verify(name)` for both models | Exit 0; 247 pilot and 79 expanded-dev predictions/model verified, including model hashes; prior tracked reports/manifests/lock unchanged |

Start/configuration/code fingerprints are in [results/final-start.json](results/final-start.json), complete predictions and timings in [results/final-predictions.jsonl](results/final-predictions.jsonl), and machine-readable aggregates in [results/final-comparison.json](results/final-comparison.json). Code/protocol text hashes normalize line endings to LF; model/evidence hashes cover raw bytes. To verify again without inference:

```powershell
uv run --locked python -m tinyocr.evaluate_final --verify-only
```

The evidence verifier uses locally cached source images and models. Missing caches must be restored from the pinned preparation/export procedures; saved results alone do not establish artifact integrity.

## Scope and next work

There are only eight independent screens and eight full scenes; the 32 UI regions share screen parents. Partial-screen recovery measures recall, not false-positive text. Full-scene CER includes reading order and upstream annotation noise. Public-pretraining contamination and app/template overlap are unknown. This final set contains no new receipt-photo or rotation coverage, and these desktop measurements do not establish phone performance. The earlier receipt-page development WER regression remains a limitation.

The reserved partition is now consumed. Future model selection or tuning needs a newly reserved, independent test set. Next work is broader real-world coverage and independent label/source review before training or a release claim. The candidate remains a compressed pretrained model, with float32 computation and no demonstrated RAM reduction.
