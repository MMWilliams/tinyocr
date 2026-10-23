import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import cv2
from omegaconf import OmegaConf
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangDet, LangRec, ModelType, OCRVersion

from .common import ROOT, reading_order, sha256

MODEL_NAMES = ("v6-tiny", "v5-english", "v6-w8", "v6-w16", "v6-w8-rec")


class Baseline:
    def __init__(self, name):
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        v6 = name.startswith("v6-")
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown baseline: {name}")
        self.directory = ROOT / "models" / name
        version = OCRVersion.PPOCRV6 if v6 else OCRVersion.PPOCRV5
        model_type = ModelType.TINY if v6 else ModelType.MOBILE
        params = {
            "Global.model_root_dir": str(self.directory),
            "Global.use_cls": False,
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": 1,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.use_dml": False,
            "Det.ocr_version": version, "Rec.ocr_version": version,
            "Det.model_type": model_type, "Rec.model_type": model_type,
            "Det.lang_type": LangDet.CH, "Rec.lang_type": LangRec.EN,
            # Official Paddle inference.yml uses ImageNet normalization for both.
            "Det.mean": [0.485, 0.456, 0.406], "Det.std": [0.229, 0.224, 0.225],
            # Same bounded detector resolution for local comparisons.
            "Det.limit_side_len": 960, "Det.limit_type": "max",
            "Det.thresh": 0.2 if v6 else 0.3,
            "Det.box_thresh": 0.4 if v6 else 0.6,
            "Det.unclip_ratio": 1.4 if v6 else 1.5,
            "Det.max_candidates": 3000 if v6 else 1000,
            "Det.use_dilation": False,
            "Rec.rec_batch_num": 1,
            "Rec.rec_img_shape": [3, 48, 320],
        }
        if name in ("v6-w8", "v6-w16", "v6-w8-rec"):
            for section,filename in (("Det","PP-OCRv6_det_tiny.onnx"),("Rec","PP-OCRv6_rec_tiny.onnx")):
                path=self.directory/filename
                if not path.exists():
                    raise FileNotFoundError(f"Export candidate first: {path}")
                params[f"{section}.model_path"] = str(path)
        self.engine = RapidOCR(params=params)
        # Pinned RapidOCR 3.10 lazily loads weights; force load outside timed calls.
        self.engine._load_det_model()
        self.engine._load_rec_model()
        self.sessions = []
        for label, module in (("detector", self.engine.text_det), ("recognizer", self.engine.text_rec)):
            session = module.session.session
            opts = session.get_session_options()
            providers = session.get_providers()
            if providers != ["CPUExecutionProvider"] or opts.intra_op_num_threads != 1 or opts.inter_op_num_threads != 1:
                raise RuntimeError("Baseline must use one-thread CPU sessions")
            self.sessions.append({"module": label, "providers": providers,
                                  "intra_threads": opts.intra_op_num_threads,
                                  "inter_threads": opts.inter_op_num_threads})

    def predict(self, path, task):
        result = self.engine(str(path), use_det=task == "page", use_cls=False, use_rec=True)
        texts = list(result.txts) if result.txts is not None else []
        boxes = getattr(result, "boxes", None)
        regions = []
        if boxes is not None:
            for text, polygon in zip(texts, boxes):
                xs, ys = polygon[:, 0], polygon[:, 1]
                regions.append({"text": text, "box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]})
            regions = reading_order(regions)
            texts = [r["text"] for r in regions]
        return "\n".join(texts), regions

    def metadata(self):
        artifacts = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                      "sha256": sha256(p)} for p in sorted(self.directory.glob("*")) if p.is_file()]
        return {"artifacts": artifacts, "artifact_bytes": sum(p["bytes"] for p in artifacts),
                "sessions": self.sessions, "opencv_threads": cv2.getNumThreads(),
                "config": OmegaConf.to_container(self.engine.cfg, resolve=True, enum_to_str=True)}
