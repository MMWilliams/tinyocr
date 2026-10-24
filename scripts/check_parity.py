"""Compare release runtime with frozen research inference on three dev fixtures.

Run from the repository root: uv run --locked python scripts/check_parity.py
This does not use or retune against the consumed final partition.
"""

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyocr.baseline import Baseline
from tinyocr.benchmark import predict
from tinyocr.common import sha256
from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location("tinyocr_release", ROOT / "python/src/tinyocr/__init__.py")
release = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = release
spec.loader.exec_module(release)

baseline, packaged = Baseline("v6-w8-rec"), release.TinyOCR()
original_config = OmegaConf.to_container(baseline.engine.cfg, resolve=True, enum_to_str=True)
packaged_config = OmegaConf.to_container(packaged._engine.cfg, resolve=True, enum_to_str=True)
def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in ("model_root_dir", "model_path")}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value
assert normalize(original_config) == normalize(packaged_config), "Pipeline configuration changed"
pilot = json.loads((ROOT / "eval/manifest.json").read_text(encoding="utf-8"))
expansion = json.loads((ROOT / "eval/expansion-region-v2.json").read_text(encoding="utf-8"))
fixtures = [next(i for i in pilot["items"] if i["domain"] == "receipt-page"),
            next(i for i in pilot["items"] if i["domain"] == "synthetic-screen"),
            next(i for i in expansion["items"] if i["domain"] == "full-scene" and i["role"] == "dev")]
rows = []
for item in fixtures:
    expected = predict(baseline, ROOT / item["image"])
    actual = packaged(ROOT / item["image"]).to_dict()
    assert expected == actual, f"Prediction mismatch: {item['id']}"
    rows.append({"id": item["id"], "image_sha256": sha256(ROOT / item["image"]),
                 "exact_text_lines_words": True, "lines": len(actual["lines"]), "words": len(actual["words"])})
result = {"version": release.__version__, "config_equal_except_model_paths": True,
          "development_fixtures": rows, "final_images_used": 0,
          "packages": {p: importlib.metadata.version(p) for p in ("rapidocr", "onnxruntime", "numpy", "pillow")}}
(ROOT / "results/publish-parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
