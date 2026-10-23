# Startup acceptance

Commands run from `C:\Users\reese\tinyocr` using `uv run --locked python`.
Requirements come from the user-confirmed OCR pivot and `C:\Users\reese\.claude\HANDOFF.md`.

| Requirement or risk | Source | Runnable check | Pass criterion | Evidence |
| --- | --- | --- | --- | --- |
| Real-world evaluation before training | Handoff goal and ASR generalization failure | `uv run --locked python -m tinyocr.prepare --n 20` and `uv run --locked python -m tinyocr.validate` | Downloaded human-labeled receipts and scene text; source IDs, checksums, eval-only designation; no empty or duplicate full samples | PASS, 247 items: 20 receipt pages, 200 related crops, 20 scene crops, 6 synthetic screens, 1 blank; exit 0 |
| Honest error measurement, including missing/extra text | Inferred scoring integrity risk | `uv run --locked python -m unittest discover -s tests -v` | Known edit-distance examples, micro aggregation, blank output, punctuation and case behavior pass | PASS, 14 tests, 0 skips, exit 0 |
| Reproducible one-thread CPU baseline | Handoff baseline survey | `uv run --locked python -m tinyocr.evaluate --model v6-tiny` and same with `--model v5-english` | Actual inference on all samples; per-domain CER/WER/exact match and latency; model hashes and byte sizes; CPU provider and thread settings verified | PASS, 247 predictions each, 3 repetitions each, both exit 0; see results/*.json and BASELINES.md |
| Offline repeatability and corrupt-data detection | Inferred cache integrity risk | `uv run --locked python -m tinyocr.validate` plus validation tests and repeated `prepare --n 20` | Local manifest and image checksums verified without a network call; bad manifests fail | PASS, complete cache reused without downloads; corrupt file, missing reference, empty manifest, duplicate ID and bad box cases rejected |
| Saved results cover every sample and match frozen artifacts | Inferred dropped-sample/report integrity risk | `uv run --locked python -m tinyocr.verify_run --model v6-tiny` and same with `--model v5-english` | Exact ID coverage, rechecked counts/aggregates, positive timings, matching model/image/lock hashes, CPU settings | PASS, both exit 0 |
| Missing image restoration preserves frozen evidence | Inferred download/cache recovery risk | Move `data/eval/images/blank.png` to `runs/blank-restore-test.png`, run `prepare --n 20`, then `verify_run --model v6-tiny` | Regenerate missing fixture without changing manifest or image hash; prior report still verifies | PASS, both commands exit 0, frozen sources/items unchanged |

Screen coverage must be labeled accurately: any rendered fixtures are synthetic smoke checks, not evidence of real application generalization. Cropped recognition and full-image OCR are distinct tasks. No phone-speed claims from desktop measurements.

Manual review: inspected a contact sheet of all 20 scene crops (`runs/scene-review.png`); real scene appearance confirmed, but noisy annotations, neighboring text and rotated/partial crops were found. This limits interpretation, as documented in BASELINES.md. The initial harness is verified; broad OCR readiness and training-data leakage checks remain unverified backlog work.

The benchmark ran before relocating the unchanged manifest from `data/eval/manifest.json` to tracked `eval/manifest.json`. Its SHA-256 is identical and both saved reports pass verification against the tracked file. Inference/scoring code and dependencies were unchanged; timings were not needlessly rerun.
