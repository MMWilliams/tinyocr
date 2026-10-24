---
license: apache-2.0
language:
  - en
pipeline_tag: image-to-text
tags:
  - ocr
  - onnx
  - paddleocr
  - rapidocr
  - cpu
  - quantized
base_model:
  - PaddlePaddle/PP-OCRv6_tiny_det_onnx
  - PaddlePaddle/PP-OCRv6_tiny_rec_onnx
base_model_relation: quantized
datasets:
  - jsdnrs/ICDAR2019-SROIE
model-index:
  - name: TinyOCR v6-w8-rec
    results: []
---

# TinyOCR 0.1.0

English-first, offline CPU OCR with **3.10 MB of ONNX model files**. This is a
compressed pretrained PaddleOCR PP-OCRv6 model pair, not a newly trained TinyOCR
student. The detector is unchanged; the recognizer stores selected weights as
per-channel int8 values with constant decoding to float32. Computation remains
float32. No training or calibration was performed. The dataset tag describes
evaluation, not TinyOCR training.

## Use

Python 3.12:

```sh
pip install tinyocr-onnx==0.1.0
tinyocr receipt.png
tinyocr receipt.png --json
```

```python
from tinyocr import TinyOCR

ocr = TinyOCR()
result = ocr("receipt.png")
print(result.text)
print(result.lines)
print(result.words)
```

The wheel bundles these exact model files and verifies their SHA-256 hashes.
It needs no model download or HF account after installation. Dependencies and
runtime memory are larger than the model files. Inputs are local image paths;
outputs contain text, line polygons/xyxy bounds and native word polygons in
original-image pixels. See the [runtime documentation](https://github.com/MMWilliams/tinyocr/tree/master/python).

## Artifacts and provenance

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `PP-OCRv6_det_tiny.onnx` | 1,829,618 | `f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5` |
| `PP-OCRv6_rec_tiny.onnx` | 1,269,595 | `38cd559cb4b24c5056592d981b13e086f9472db9b3c8968296f0d0c2b4a65531` |

Total: 3,099,213 bytes, versus 6,319,431 original bytes (50.957% smaller).
Original recognizer SHA-256: `e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b`.
Source ONNX files came from RapidAI/RapidOCR v3.10.0 on ModelScope. NOTICE gives
the exact URLs and upstream attribution. `export-provenance.json` records
modified tensors; the export code is in the
[source repository](https://github.com/MMWilliams/tinyocr).

## Measured results

5,089 images per model with 120 prior-parent exclusions. These are comparative
development results; public pretraining overlap is unknown.

| Dataset | Images | Original E2E F1 (%) | TinyOCR E2E F1 (%) | Candidate p50 change |
| --- | ---: | ---: | ---: | ---: |
| SROIE | 341 | 41.036 | 40.779 | +21.23% |
| TextOCR | 3,027 | 43.412 | 43.334 | -6.91% |
| HierText | 1,721 | 40.330 | 40.204 | +16.16% |

Small observed E2E F1 losses of 0.079–0.257 percentage points, not exact accuracy
preservation. SROIE/TextOCR use the shared HierText polygon evaluator, not their
native leaderboard protocols. HierText uses its native word evaluator on the
overlap-filtered subset. Latency is one pass per model on the same Windows CPU
environment, not a repeated speed study. Peak process working set was
498.84 MiB original / 500.10 MiB candidate. **No general speedup or RAM saving is
claimed.** See the [full benchmark](https://github.com/MMWilliams/tinyocr/blob/master/BENCHMARK_RESULTS.md)
for counts, source rights, protocols, timing and caveats. Images/labels are not
included in this model repository.

## Configuration and limitations

RapidOCR 3.10.0; ONNX Runtime 1.31.0; one-thread CPU sessions and OpenCV; detector
ImageNet normalization, maximum side 960, threshold 0.2, box threshold 0.4,
unclip 1.4, no dilation; recognizer batch one; no orientation classifier.
The installed API matches the frozen research text/line/word adapter on three
development fixtures. English-first evaluation does not establish multilingual
performance. Rotations, difficult scenes and complex layouts can fail.
Reading order is geometric, not layout understanding. Native word polygons can
have zero area; they are retained for benchmark parity. No mobile performance
claim is made. A consumed final comparison is preserved in the research repo;
future tuning needs a fresh, independently labeled final partition.

## License and links

Apache-2.0 for the runtime and PaddleOCR-derived models. See LICENSE and NOTICE
for PaddlePaddle / RapidAI attribution and the recognizer modification notice.
No affiliation with or endorsement by PaddlePaddle or RapidAI is implied.

- [GitHub source and full evaluation reports](https://github.com/MMWilliams/tinyocr)
- [PyPI tinyocr-onnx](https://pypi.org/project/tinyocr-onnx/)
- [Original detector card](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)
- [Original recognizer card](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)
