# TinyOCR

Offline, English-first CPU OCR with **3.10 MB of bundled ONNX model files**.
TinyOCR packages a compressed pretrained PaddleOCR PP-OCRv6 recognizer and the
original tiny detector, using RapidOCR and ONNX Runtime. It is an alpha release,
not a newly trained model. The full installation and runtime memory are larger
than the model files.

## Install and use

Python **3.12** is supported. Windows, Linux and macOS are tested in CI.

```sh
python -m pip install tinyocr-onnx==0.1.0
tinyocr receipt.png
tinyocr receipt.png --json
python -m tinyocr receipt.png --no-words
```

```python
from tinyocr import TinyOCR

ocr = TinyOCR()  # Load once and reuse. No model downloads.
result = ocr("receipt.png")
print(result.text)
print(result.lines)     # text, polygon, and [x0, y0, x1, y1] bounds
print(result.words)     # text and native word polygons
data = result.to_dict() # JSON-serializable
```

Inputs are local image paths (PNG, JPEG, and other Pillow-supported images).
Coordinates are original-image pixels. Text uses a simple geometric row order,
not document-layout analysis. Blank images produce empty text and region lists.
`words=False` omits word geometry. Native word polygons can have zero area;
these are preserved to match the published benchmark.

The package name is **tinyocr-onnx**; the import and CLI are **tinyocr**. Do not
install the unrelated `tinyocr` distribution into the same environment. When
using this research checkout, run the installed package outside the repository
root so the research `tinyocr/` directory does not shadow it.

## Measured tradeoffs

The frozen pair is 3,099,213 bytes versus the original 6,319,431 bytes (50.96%
smaller). The recognizer stores int8 weights but executes float32 math. This is
a model-storage reduction; it is not a general speed or RAM reduction.

| Dataset | Images | Original E2E F1 (%) | TinyOCR E2E F1 (%) |
| --- | ---: | ---: | ---: |
| SROIE | 341 | 41.036 | 40.779 |
| TextOCR | 3,027 | 43.412 | 43.334 |
| HierText | 1,721 | 40.330 | 40.204 |

In one timed pass, candidate p50 latency was 21.23% higher on SROIE, 16.16%
higher on HierText, and 6.91% lower on TextOCR. Peak process working set was
about 500 MiB for both models on the measured Windows machine. SROIE/TextOCR
use the shared HierText polygon protocol, not their native leaderboard scores.
Public pretraining overlap is unknown; these are comparative development results.
See the [full benchmark and limitations](https://github.com/MMWilliams/tinyocr/blob/master/BENCHMARK_RESULTS.md).

Inference uses one-thread CPU ONNX sessions, detector maximum side 960 px,
recognition batch size one, and no orientation classifier. Initialization also
sets OpenCV and common BLAS thread controls to one for the current process.
Rotated text, difficult scenes and complex layouts can fail. English-first
evaluation does not establish multilingual performance. No mobile performance
claim is made.

## Source and license

[GitHub](https://github.com/MMWilliams/tinyocr) ·
[Hugging Face models](https://huggingface.co/maureesewilliams/tinyocr) ·
[PyPI](https://pypi.org/project/tinyocr-onnx/)

Apache-2.0. See LICENSE and NOTICE for PaddlePaddle and RapidAI attribution and
the recognizer modification notice. Bundled artifacts are verified by SHA-256
on initialization. No evaluation datasets are included.
