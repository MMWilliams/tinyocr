"""TinyOCR: offline CPU OCR using bundled, compressed PaddleOCR weights."""

from .runtime import OCRResult, TinyOCR

__version__ = "0.1.0"
__all__ = ["OCRResult", "TinyOCR", "__version__"]
