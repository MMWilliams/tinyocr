# Tiny OCR

Confirmed 2026-10-23: OCR follows the shelved tinyasr project.

Initial scope: English-first, very small, fast on one CPU thread; eventual consumer laptops and phones. Start with a measured open baseline and real-world evaluation before selecting a student architecture or training.

- [x] Establish cached, reproducible human-labeled receipt and scene-text evaluation: 20 SROIE pages, 200 GT receipt crops, 20 TextOCR word crops; pinned manifest and source revisions.
- [x] Add initial screen smoke coverage: six authored light/dark raster fixtures and a blank control, kept separate from real inputs.
- [x] Measure current small open baselines on one CPU thread: v6 tiny 6.32 MB, receipt-page CER 11.74%, p50 199.6 ms; v5 English 12.69 MB, CER 9.84%, p50 675.6 ms. See BASELINES.md and results/*.json. Full predictions in runs/.
- [ ] Define the student size/accuracy/latency target from results; no arbitrary inherited ASR throughput target.
- [ ] Expand to real screenshot transcriptions, blank images, difficult signs, receipt photos and rotated text before training.
- [ ] Independently audit TextOCR crop labels and recover original image IDs: some source labels/crops are noisy, and parent IDs are missing. Retain original pilot scores; version any reviewed reference changes.
- [ ] Use separate development and untouched final test sets; prevent image/source overlap with training.
- [ ] Build a mixed-domain training pilot only after baseline error analysis and data/license review.

Inherited resource limits: training <= half of GPU 0, GPU 1 free, <= one-third of RAM (~20 GB); no paid API runs. Check free commit before training. CPU-only startup requires no GPU allocation. Preserve tinyasr artifacts.

Next concrete work: collect a small human-transcribed set of real application screenshots and full scene images with original image IDs. Add a disjoint final-test split before selecting compression/training targets. The initial setup is complete; no custom model or training run is in flight.

Startup lessons: HF rows API returned 429 after ~35 images; preparation now checkpoints revision/hash-verified downloads and backs off up to 45 seconds. RapidOCR shared detector normalization defaults differ from official Paddle inference configs; the adapter sets the official mean/std explicitly. Scene crop CER is not scene detection accuracy. Public benchmark overlap with pretrained weights is unknown.
