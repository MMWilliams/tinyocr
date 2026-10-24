"""Verify TinyOCR 0.1.0 public bytes against its successful publishing run.

Download the run's distributions artifact ZIP first, then supply it unchanged.
This uses unauthenticated public APIs and never uploads or changes a release.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "MMWilliams/tinyocr"
VERSION = "0.1.0"
FILENAMES = {
    "tinyocr_onnx-0.1.0-py3-none-any.whl",
    "tinyocr_onnx-0.1.0.tar.gz",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "TinyOCR-release-verification"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--artifact-zip", type=Path, required=True)
    args = parser.parse_args()
    run_api = f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{args.run_id}"
    run = json.loads(fetch(run_api))
    require(run["status"] == "completed" and run["conclusion"] == "success", "Publishing run did not succeed")
    require(run["event"] == "workflow_dispatch" and run["head_branch"] == "master", "Unexpected publishing trigger")
    require(run["path"] == ".github/workflows/publish.yml", "Unexpected workflow")
    # Retried publish jobs reuse the successful build/tests from earlier attempts.
    # Take each job's latest result across every attempt of this same source run.
    latest_jobs = {}
    for attempt in range(1, run["run_attempt"] + 1):
        response = json.loads(fetch(run_api + f"/attempts/{attempt}/jobs?per_page=100"))
        require(response["total_count"] == len(response["jobs"]), "Job results require pagination")
        latest_jobs.update({job["name"]: job for job in response["jobs"]})
    jobs = list(latest_jobs.values())
    required_jobs = {"build", "test (ubuntu-latest)", "test (windows-latest)", "test (macos-latest)", "publish"}
    require(required_jobs == {job["name"] for job in jobs}, "Unexpected job inventory")
    require(all(job["conclusion"] == "success" for job in jobs), "A required job did not pass")
    artifacts = json.loads(fetch(run_api + "/artifacts"))["artifacts"]
    distributions = [artifact for artifact in artifacts if artifact["name"] == "distributions"]
    require(len(distributions) == 1, "Expected exactly one distributions artifact")
    artifact = distributions[0]
    archive_data = args.artifact_zip.read_bytes()
    archive_hash = sha256(archive_data)
    require(artifact["digest"] == "sha256:" + archive_hash, "CI archive digest mismatch")
    require(artifact["size_in_bytes"] == len(archive_data), "CI archive size mismatch")
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        require(len(archive.namelist()) == 2 and set(archive.namelist()) == FILENAMES, "Unexpected CI distribution inventory")
        ci_files = {name: archive.read(name) for name in FILENAMES}
    metadata = json.loads(fetch(f"https://pypi.org/pypi/tinyocr-onnx/{VERSION}/json"))
    require(metadata["info"]["name"] == "tinyocr-onnx" and metadata["info"]["version"] == VERSION, "Unexpected PyPI project/version")
    require(len(metadata["urls"]) == 2 and {item["filename"] for item in metadata["urls"]} == FILENAMES, "Unexpected PyPI file inventory")
    models = json.loads((ROOT / "python/src/tinyocr/assets/models.json").read_text(encoding="utf-8"))["artifacts"]
    files = []
    destination = args.artifact_zip.parent / "pypi"
    destination.mkdir(exist_ok=True)
    for item in sorted(metadata["urls"], key=lambda entry: entry["filename"]):
        name = item["filename"]
        require(not item["yanked"], f"Yanked distribution: {name}")
        data = fetch(item["url"])
        digest = sha256(data)
        require(data == ci_files[name], f"PyPI bytes differ from tested CI bytes: {name}")
        require(digest == item["digests"]["sha256"] and len(data) == item["size"], f"PyPI metadata mismatch: {name}")
        verified_models = []
        if name.endswith(".whl"):
            with zipfile.ZipFile(io.BytesIO(data)) as wheel:
                payloads = {model["filename"]: wheel.read("tinyocr/assets/" + model["filename"]) for model in models}
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as sdist:
                payloads = {model["filename"]: sdist.extractfile(f"tinyocr_onnx-{VERSION}/src/tinyocr/assets/" + model["filename"]).read() for model in models}
        for model in models:
            payload = payloads[model["filename"]]
            require(sha256(payload) == model["sha256"] and len(payload) == model["bytes"], f"Frozen model mismatch in {name}: {model['filename']}")
            verified_models.append(model)
        (destination / name).write_bytes(data)
        files.append({"filename": name, "sha256": digest, "bytes": len(data), "url": item["url"], "upload_time": item["upload_time_iso_8601"], "models": verified_models})
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "tinyocr-onnx", "version": VERSION,
        "url": f"https://pypi.org/project/tinyocr-onnx/{VERSION}/",
        "run_id": args.run_id, "run_attempt": run["run_attempt"], "run_url": run["html_url"], "source_commit": run["head_sha"],
        "artifact_id": artifact["id"], "archive_sha256": archive_hash,
        "public_distributions_match_tested_ci_bytes": True,
        "frozen_model_payloads_match": True,
        "jobs": [{key: job[key] for key in ("id", "name", "status", "conclusion")} for job in jobs],
        "files": files,
    }
    output = ROOT / "results/publish-pypi.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: both public distributions match tested CI bytes and frozen model hashes; {output}")


if __name__ == "__main__":
    main()
