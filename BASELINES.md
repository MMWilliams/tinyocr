# OCR startup baseline — 2026-10-23

Measured on this machine: AMD Ryzen 9 9900X, Windows, Python 3.12.9. ONNX Runtime 1.31.0, RapidOCR 3.10.0; CPU provider only, intra/inter-op threads = 1 and OpenCV threads = 1. No GPU or paid API use. See `results/` for exact hashes, versions, configs, counts and timing summaries; `runs/` contains all predictions.

| Baseline | Detector + recognizer files | Receipt page CER / p50 | Receipt crop CER / p50 | Scene crop CER / p50 | Synthetic screen CER / p50 |
| --- | ---: | --- | --- | --- | --- |
| PP-OCRv6 tiny | 6.32 MB | 11.74% / 199.6 ms | 7.17% / 2.75 ms | 26.98% / 2.67 ms | 0% / 49.1 ms |
| PP-OCRv5 mobile, English recognizer | 12.69 MB | 9.84% / 675.6 ms | 5.14% / 10.87 ms | 24.60% / 10.67 ms | 0% / 137.1 ms |

MB means decimal artifact bytes, excluding runtime libraries and RAM. Sample counts: 20 receipt pages, 200 correlated receipt crops, 20 scene crops, six authored screen fixtures. Both baselines returned no text for the blank control. Neither has an orientation classifier enabled. Models are evaluated sequentially; desktop background load was not controlled.

PP-OCRv6 tiny is the initial size/speed reference: about half the artifact size and 3.4x faster receipt-page median than the English v5 pipeline, with worse accuracy in this pilot. V5 English is the accuracy reference. These are existing pretrained models, not newly trained TinyOCR models. The six easy synthetic screens do not establish real screenshot accuracy.

The first engineering decision is to expand and audit evaluation before choosing a student architecture. Public model training overlap is unknown. Twenty scene crops provide only a rough diagnostic, and inspection shows tilted/upside-down text, neighboring text inside crops, and questionable annotations (for example the printed `3/$5` is labeled with an extra backslash). Keep original labels and raw outputs; do not silently correct labels based on model predictions. An independent annotation review must be versioned before results are used to set training targets.

## Source survey

- [PaddleOCR's official model table](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html) lists tiny v6 detector/recognizer storage at 1.9/4.4 MB. The [v6 architecture documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md) describes its lightweight backbone and recognition neck. These made it the first baseline to measure.
- [Official recognition model documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md) lists the English v5 mobile recognizer at 7.5 MB. Our exported ONNX artifact sizes differ from the documentation's model packaging, so the table above uses actual files.
- [RapidOCR](https://github.com/RapidAI/RapidOCR) supplies the CPU ONNX pipeline and converted models. Its [license statement](https://github.com/RapidAI/RapidOCR/blob/main/README.md) identifies Apache-2.0 for its code and PaddleOCR-derived model artifacts. Package version and downloaded weight hashes are recorded; this project does not relicense third-party weights.
- [Tesseract tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) is an additional integer-LSTM candidate under its [Apache-2.0 license](https://github.com/tesseract-ocr/tessdata_fast/blob/main/LICENSE). It was surveyed but not installed or timed; no local speed/accuracy claim is made for it. Two small neural pipelines were sufficient for the startup comparison.

Published speed columns use other runtimes, thread counts and hardware; they were not treated as one-thread measurements for this project. Detector normalization follows the official inference configs for [v6 tiny](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/blob/main/inference.yml) and [v5 mobile](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det/blob/main/inference.yml), rather than RapidOCR's shared default normalization. Both use a fixed maximum detector side of 960 pixels here; this is a local comparison configuration, not a reproduction of published benchmark settings.

## Evaluation sources

- [SROIE mirror](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE): human region transcriptions and pixel boxes, uppercased reference text; card declares CC-BY-4.0. Original challenge attribution and mirror modifications are retained in the locally cached source card. Twenty evenly spaced rows from the 361-row test split, with ten deterministic regions per page.
- [TextOCR crop mirror](https://huggingface.co/datasets/MiXaiLL76/TextOCR_OCR): twenty evenly spaced rows from its 14,669-row test split. Its card describes a size-filtered subset; this is not representative full-scene detection. The card's MIT metadata conflicts with its own reference to original TextOCR CC-BY-4.0. Record both; do not infer training or redistribution clearance from the mirror metadata.
- Local screen fixtures: authored text rendered at three font sizes on light/dark backgrounds, plus a white blank. Their references are generated from authored text, not human transcriptions of real screenshots. Font hash and image hashes are pinned.

These public datasets are used locally for evaluation only. No training corpus was downloaded or license-cleared during startup.
