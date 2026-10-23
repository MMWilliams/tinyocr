"""Check that saved baseline evidence covers the frozen evaluation exactly."""
import argparse
import json
import math

from .common import MANIFEST, ROOT, sha256
from .metrics import aggregate, counts
from .validate import validate
from .baseline import MODEL_NAMES


def verify(name):
    manifest = validate()
    path = ROOT / f"runs/{name}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["manifest_sha256"] == sha256(MANIFEST), "Changed manifest"
    assert report["lock_sha256"] == sha256(ROOT / "uv.lock"), "Changed dependencies"
    predictions = report["predictions"]
    references = {item["id"]: item for item in manifest["items"]}
    assert len(predictions) == len(references), "Missing/extra predictions"
    assert {p["id"] for p in predictions} == set(references), "Missing/duplicate sample IDs"
    for prediction in predictions:
        item = references[prediction["id"]]
        assert prediction["reference"] == item["text"], "Reference changed"
        assert prediction["domain"] == item["domain"] and prediction["task"] == item["task"]
        for key, casefold in (("normalized", True), ("case_sensitive", False)):
            assert prediction[key] == counts(item["text"], prediction["prediction"], casefold)
        assert len(prediction["latency_ms"]) == report["timing"]["repeats"]
        assert all(math.isfinite(t) and t > 0 for t in prediction["latency_ms"])
    for domain, score in report["domains"].items():
        selected = [p for p in predictions if p["domain"] == domain]
        for key in ("normalized", "case_sensitive"):
            assert score[key] == aggregate([p[key] for p in selected]), "Incorrect aggregation"
    assert set(report["domains"]) == {p["domain"] for p in predictions}
    details = report["model_details"]
    assert details["opencv_threads"] == 1
    assert len(details["sessions"]) == 2
    for session in details["sessions"]:
        assert session["providers"] == ["CPUExecutionProvider"]
        assert session["intra_threads"] == session["inter_threads"] == 1
    for artifact in details["artifacts"]:
        artifact_path = ROOT / artifact["path"]
        assert sha256(artifact_path) == artifact["sha256"], "Changed model weights"
        assert artifact_path.stat().st_size == artifact["bytes"]
    assert details["artifact_bytes"] == sum(a["bytes"] for a in details["artifacts"])
    print(f"Verified {name}: {len(predictions)} predictions, all domains, model hashes and CPU/thread evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    verify(parser.parse_args().model)
