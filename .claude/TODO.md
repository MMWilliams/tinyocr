# Tiny OCR

Confirmed 2026-10-23: OCR follows the shelved tinyasr project.

Initial scope: English-first, very small, fast on one CPU thread; eventual consumer laptops and phones. Start with a measured open baseline and real-world evaluation before selecting a student architecture or training.

- [x] Establish cached, reproducible human-labeled receipt and scene-text evaluation: 20 SROIE pages, 200 GT receipt crops, 20 TextOCR word crops; pinned manifest and source revisions.
- [x] Add initial screen smoke coverage: six authored light/dark raster fixtures and a blank control, kept separate from real inputs.
- [x] Measure current small open baselines on one CPU thread: v6 tiny 6.32 MB, receipt-page CER 11.74%, p50 199.6 ms; v5 English 12.69 MB, CER 9.84%, p50 675.6 ms. See BASELINES.md and results/*.json. Full predictions in runs/.
- [x] Define an initial compression target from results: <=3.5 MB detector+recognizer, no additional missed screen labels vs 99/113, no per-domain dev CER increase or repeatable latency regression. Provisional, not a release claim.
- [ ] Expand to real screenshot transcriptions, blank images, difficult signs, receipt photos and rotated text before training.
- [ ] Independently audit TextOCR crop labels and recover original image IDs: some source labels/crops are noisy, and parent IDs are missing. Retain original pilot scores; version any reviewed reference changes.
- [x] Separate expanded development (79 items) and reserved final (48 items) by original image ID and decoded-pixel hash; no final inference. Training-overlap guard implemented for declared provenance/reserved splits.
- [x] Add 20 real mobile-app screens with labels matched to Google's human-corrected release and 20 full TextOCR scenes; measure both baselines on dev only. See EXPANSION.md.
- [x] Audit all 20 legacy scene crops and dev images; retain original labels. Candidate parent IDs recorded; exclude every candidate from new scenes. Definitive pixel-based lineage and human adjudication remain open.
- [ ] Build a mixed-domain training pilot only after baseline error analysis and data/license review.

Inherited resource limits: training <= half of GPU 0, GPU 1 free, <= one-third of RAM (~20 GB); no paid API runs. Check free commit before training. CPU-only startup requires no GPU allocation. Preserve tinyasr artifacts.

Next concrete work: prototype weight-only int8 compression of v6 tiny, measure artifact bytes and CPU behavior on both fixed development evaluations, and compare against the acceptance criteria in EXPANSION.md. Preserve final-set isolation until a candidate is locked. No custom model or training run is in flight.

Startup lessons: HF rows API returned 429 after ~35 images; preparation now checkpoints revision/hash-verified downloads and backs off up to 45 seconds. RapidOCR shared detector normalization defaults differ from official Paddle inference configs; the adapter sets the official mean/std explicitly. Scene crop CER is not scene detection accuracy. Public benchmark overlap with pretrained weights is unknown.
