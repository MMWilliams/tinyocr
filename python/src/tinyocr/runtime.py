"""Frozen v6-w8-rec pipeline, distributed independently of the research harness."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import threading


@dataclass
class OCRResult:
    """Text in row order, line polygons/xyxy bounds, and native word polygons.

    Coordinates are in pixels of the original image. Native word boxes can be
    degenerate; these are retained to match the published benchmark adapter.
    """

    text: str
    lines: list[dict]
    words: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _reading_order(regions):
    # Same half-smaller-height row tolerance as the frozen research adapter.
    rows = []
    for region in sorted(regions, key=lambda r: ((r["box"][1] + r["box"][3]) / 2, r["box"][0])):
        x0, y0, x1, y1 = region["box"]
        cy, h = (y0 + y1) / 2, y1 - y0
        if rows and abs(cy - rows[-1][0]) <= 0.5 * min(h, rows[-1][1]):
            rows[-1][2].append(region)
        else:
            rows.append([cy, h, [region]])
    return [r for _, _, row in rows for r in sorted(row, key=lambda r: r["box"][0])]


class TinyOCR:
    """Reusable, one-thread CPU OCR engine for local image paths.

    Initialization verifies and loads both bundled models. No model download
    or account is required. Sets OpenCV and common BLAS thread controls to one.
    Calls on one instance are serialized because RapidOCR updates call options.
    """

    def __init__(self):
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"

        import cv2
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangDet, LangRec, ModelType, OCRVersion

        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        assets = Path(__file__).resolve().parent / "assets"
        manifest = json.loads((assets / "models.json").read_text(encoding="utf-8"))
        for item in manifest["artifacts"]:
            path = assets / item["filename"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Bundled model missing or damaged: {path.name}. Reinstall tinyocr-onnx.")
        params = {
            "Global.model_root_dir": str(assets),
            "Global.use_cls": False,
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": 1,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.use_dml": False,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.TINY,
            "Rec.model_type": ModelType.TINY,
            "Det.lang_type": LangDet.CH,
            "Rec.lang_type": LangRec.EN,
            "Det.mean": [0.485, 0.456, 0.406],
            "Det.std": [0.229, 0.224, 0.225],
            "Det.limit_side_len": 960,
            "Det.limit_type": "max",
            "Det.thresh": 0.2,
            "Det.box_thresh": 0.4,
            "Det.unclip_ratio": 1.4,
            "Det.max_candidates": 3000,
            "Det.use_dilation": False,
            "Rec.rec_batch_num": 1,
            "Rec.rec_img_shape": [3, 48, 320],
            "Det.model_path": str(assets / "PP-OCRv6_det_tiny.onnx"),
            "Rec.model_path": str(assets / "PP-OCRv6_rec_tiny.onnx"),
        }
        self._engine = RapidOCR(params=params)
        self._engine._load_det_model()
        self._engine._load_rec_model()
        for module in (self._engine.text_det, self._engine.text_rec):
            session = module.session.session
            opts = session.get_session_options()
            if (session.get_providers() != ["CPUExecutionProvider"] or
                    opts.intra_op_num_threads != 1 or opts.inter_op_num_threads != 1):
                raise RuntimeError("TinyOCR requires one-thread CPU sessions")
        self._lock = threading.Lock()

    def __call__(self, image: str | os.PathLike, *, words: bool = True) -> OCRResult:
        """Read a local image. Set words=False to omit native word geometry."""
        from rapidocr.main import flatten_word_results

        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist or is not a file: {path}")
        with self._lock:
            result = self._engine(str(path), use_det=True, use_cls=False, use_rec=True,
                                  return_word_box=words, return_single_char_box=False)
            lines = []
            if result.boxes is not None:
                for text, points in zip(result.txts, result.boxes):
                    polygon = [[float(x), float(y)] for x, y in points]
                    xs, ys = zip(*polygon)
                    lines.append({"text": text, "polygon": polygon,
                                  "box": [min(xs), min(ys), max(xs), max(ys)]})
            lines = _reading_order(lines)
            word_regions = [
                {"text": text, "polygon": [[float(x), float(y)] for x, y in points]}
                for text, _, points in flatten_word_results(getattr(result, "word_results", None))
            ] if words else []
        return OCRResult("\n".join(r["text"] for r in lines), lines, word_regions)
