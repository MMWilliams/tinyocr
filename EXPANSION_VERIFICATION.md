# Evaluation expansion acceptance

Source: user approved the next step in BASELINES.md and .claude/TODO.md: real screenshots, full scenes, label audit, and disjoint dev/final sets before training. Existing pilot files/results must remain unchanged.

| Requirement / risk | Runnable check (project root) | Pass criterion | Evidence |
| --- | --- | --- | --- |
| Real screens and full scene images with source IDs | `uv run --locked python -m tinyocr.expand` | Pinned upstream labels/images cached; screen labels verified against original Google release; full-scene annotations retain original image IDs | PASS: 20 real screens, 87 derived regions, 20 full scenes; original Google label equality checked; cached rerun exit 0 |
| Disjoint development and reserved final inputs | `uv run --locked python -m tinyocr.check_expansion` | IDs and decoded-pixel hashes disjoint; derived crops inherit parent split; final inputs not evaluated by default | PASS: 79 dev / 48 final; source/pixel and cross-split crop mutation tests pass |
| Don't score partial screenshot labels as full-page truth | `uv run --locked python -m unittest discover -s tests -v` | Screen parser distinguishes OCR text from icon descriptions; missing annotated text hurts recall; extra text outside labeled scope does not claim a precision/CER score | PASS: 28 tests, no skips; location, token boundary, multiline, duplicate hierarchy and partial-truth checks pass |
| Bounded downloads and corrupt-data rejection | Same unit suite plus actual preparation | HTTP range reads validate status/offset/length; no accidental multi-GB ZIP download; cached content hashes verified | PASS: real 7.07 GB archive accessed with 8,392,863 transferred bytes; wrong-range/full-response tests pass; missing PNG regenerated with identical frozen metadata/hash |
| Preserve pilot evidence | Both existing `tinyocr.verify_run` commands | All original references, images, weights and metrics still match | PASS: both exit 0, 247 original predictions each |
| Measure expanded dev set; reserve final | `uv run --locked python -m tinyocr.evaluate_expansion --model v6-tiny` and v5-english | Every dev input evaluated, no final IDs in outputs; reports retain manifests/weights/thread configuration | PASS: both exit 0, 79 dev predictions each, zero final predictions; both `--verify-only` checks pass |

Manual obligation: visually inspect a deterministic development sample and all 20 legacy scene crops, record findings without silently changing upstream labels. An assistant visual audit is not an independent human reannotation. Reserve final samples without inspecting baseline predictions.

Completed: all 24 development full images and 20 legacy crops visually inspected; findings in eval/LABEL_AUDIT.md. Error-region review exposed multiline labels passed to a single-line recognizer. Corrected the protocol for ALL screen regions, retained diagnostic runs and original manifest, and reran both models. Images, transcripts and splits are unchanged; the final partition was neither visually reviewed nor inferred on.

Result: expanded harness/data/partition checks verified. Broader training readiness remains partial (human adjudication, definitive legacy crop identity, app/template separation and data licensing review). The later restore/checker edits do not change inference or metric definitions; final saved reports pass current verification without another timing run.
