"""Review, publish and read back an explicit inventory of frozen model artifacts.

uv run --no-project --with huggingface-hub python scripts/publish_hf.py
Add --publish after reviewing the inventory. Uses the normal HF login/token.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "maureesewilliams/tinyocr"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    assets = ROOT / "python/src/tinyocr/assets"
    files = {
        "README.md": ROOT / "MODEL_CARD.md",
        "LICENSE": ROOT / "LICENSE",
        "NOTICE": ROOT / "NOTICE",
        "models.json": assets / "models.json",
        "PP-OCRv6_det_tiny.onnx": assets / "PP-OCRv6_det_tiny.onnx",
        "PP-OCRv6_rec_tiny.onnx": assets / "PP-OCRv6_rec_tiny.onnx",
        "export-provenance.json": ROOT / "results/export-v6-w8-rec.json",
        "BENCHMARK_RESULTS.md": ROOT / "BENCHMARK_RESULTS.md",
    }
    inventory = [{"path": name, "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in files.items()]
    print(json.dumps(inventory, indent=2), flush=True)
    api = HfApi()
    user = api.whoami()["name"]
    if user != REPO.split("/")[0]:
        raise RuntimeError(f"Unexpected Hugging Face account: {user}")
    if not args.publish:
        print("Inventory only. No upload performed.")
        return
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    info = api.model_info(REPO)
    if info.private:
        raise RuntimeError("Existing repo is private; refusing to change its visibility implicitly")
    existing = set(api.list_repo_files(REPO)) - {".gitattributes"}
    if existing - files.keys():
        raise RuntimeError(f"Unexpected existing remote files: {sorted(existing - files.keys())}")
    commit = api.create_commit(REPO, repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()],
        commit_message="Release TinyOCR 0.1.0 frozen 3.10 MB OCR models and evidence",
        parent_commit=info.sha)
    with tempfile.TemporaryDirectory(prefix="tinyocr-hf-readback-") as cache:
        for item in inventory:
            downloaded = hf_hub_download(REPO, item["path"], revision=commit.oid, cache_dir=cache, token=False)
            assert hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() == item["sha256"], item["path"]
    evidence = {"repository": REPO, "commit": commit.oid, "url": f"https://huggingface.co/{REPO}",
                "public_readback_all_hashes_match": True, "files": inventory}
    (ROOT / "results/publish-huggingface.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
