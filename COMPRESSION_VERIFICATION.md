# Compression acceptance (before implementation)

Source: user approved the next experiment in EXPANSION.md / .claude/TODO.md.
Commands use the locked runtime; graph editing/tests additionally use `uv run --locked --with onnx==1.23.2 python ...` so the inference lock and historical evidence stay unchanged.

| Requirement / risk | Runnable check | Pass criterion | Evidence |
| --- | --- | --- | --- |
| Small exported artifacts | `python -m tinyocr.compress` and comparison report | Detector + recognizer <= 3,500,000 bytes; source models unchanged; hashes and transformation recorded | PASS: v6-w8-rec = 3,099,213 bytes; original detector copied byte-for-byte, original files untouched |
| Correct weight conversion | ONNX checker and `python -m unittest discover -s tests_export -v` in export environment | Known weights reconstruct within quantization bounds; zeros finite; shape, dtype and metadata preserved; actual ORT inference agrees with reconstructed-weight reference | PASS: 6 export tests + 28 existing tests, no skips; full-check ONNX validation; actual ORT graph audit shows weight decode folded at initialization |
| No dev character-error regression | Evaluate candidate on original pilot and expanded dev, then `python -m tinyocr.compare_compression --model <candidate>` | No per-domain CER increase; >=99/113 screen labels recovered, and no previously recovered screen label lost; blank control unchanged | PASS for v6-w8-rec across 247 pilot + 79 expanded dev items; 100/113 screen labels, zero previously recovered labels lost. Receipt WER is slightly worse, disclosed separately |
| CPU speed | `python -m tinyocr.benchmark_pair --model v6-w8-rec` and repeat with `--reverse-load` | Nine interleaved passes per run, alternating order, all 326 dev inputs; same provider/threads. Flag any domain with higher p50 or p95 on all nine passes; retain ratios even when this screen passes | PASS for the predefined screen in both runs, 5,868 paired observations verified. Median p50 ratios 0.984–1.017. Full-scene p95 +5.9% then +2.5%; not a proof of identical speed |
| Preserve original evidence and final isolation | Existing verifiers plus comparison checks | Original weights/references/results unchanged; expanded runs contain all 79 dev IDs, no final IDs; dependency lock unchanged | PASS: comparison invokes original and candidate verifiers; every timing pass checks all 326 dev IDs and excludes final inputs |
| Reproducible export and usable local image command | Re-export `--mode w8-rec`, rerun comparison, then `tinyocr.ocr data/expansion/images/screen-1810.png --output runs/ocr-smoke.json` | Rebuilt hashes match recorded artifacts; CLI prediction equals saved dev prediction and supports Unicode on Windows | PASS: byte-identical rebuild; CLI text exactly matches; fixed default cp1252 stdout by explicitly using UTF-8 |

Weight storage compression is not an integer-compute or RAM reduction claim: decoded weights may be folded back to float32 at session load. First try per-output-channel int8; if rounding harms accuracy, test float16 weight storage as a distinct candidate rather than weakening the accuracy gate.

Rejected experiments remain recorded: v6-w8 (both modules int8) fails full-scene CER; v6-w16 fails receipt-page and screenshot-region CER. The accepted precision split follows those results: retain the original detector and compress only recognizer weights. No sample, reference or accuracy threshold was changed.

One newly added Add-op fixture initially retained MatMul's output shape. ONNX's independent shape check correctly rejected it. The fixture was corrected to Add's broadcast result [16,16], then all six export tests passed; conversion logic and assertion coverage were unchanged.

Verified scope: compressed artifact construction, recorded development character-error/label/size gates, and the stated desktop speed screen. The rejected variants fail their accuracy gates and remain rejected. Overall model generalization, no-loss WER, smaller RAM use and final-set performance are not established.

Candidate source hashes and pipeline configuration are frozen in results/candidate-lock.json before any final evaluation. That file pins the acceptance report and expanded manifest too.
