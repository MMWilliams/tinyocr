# TinyOCR 0.1.0 publication

Status on 2026-10-24: **verified**. GitHub, Hugging Face and PyPI are public.
PyPI's wheel and source archive match the tested CI distributions byte for byte,
including both frozen model payloads. A fresh installation from production PyPI
outside the checkout passed pip check and all 7 API/CLI/offline OCR tests with
no skips. Windows/Linux/macOS installed-wheel CI also passed (7 tests each).

## Published artifacts

- GitHub: https://github.com/MMWilliams/tinyocr
- Hugging Face: https://huggingface.co/maureesewilliams/tinyocr
- PyPI: https://pypi.org/project/tinyocr-onnx/0.1.0/

Published package source revision: `faf7c43d2e3cdc490bd404420c72476f6aa03f91`.
Initial GitHub source/workflow read-back revision: `e09f5381ddc9c4a8ea054f1de3a3bca48e4186d3`.
Hugging Face revision: `46f0f44e65383248cbeeced5dca6397c595314fa`.
Unauthenticated read-back matched both frozen ONNX hashes from GitHub and all
eight uploaded files from Hugging Face. Evidence: `results/publish-github.json`
and `results/publish-huggingface.json`.
All three public project pages returned HTTP 200 without authentication after
publication; see `results/publish-public-links.json`.

The runtime is a separate setuptools project under `python/`, with its own
lockfile. Root research code, dependency lock, model bytes and first-prediction
journals were preserved. The wheel includes only runtime code, the two model
files, model manifest and licenses. Evaluation data and the separately licensed
HierText evaluator are excluded from the distribution. The detector is
unchanged; the recognizer modification and upstream attribution are documented
in NOTICE and MODEL_CARD.md.

## Executed verification

All commands ran from the repository root unless otherwise noted.

| Check | Actual result |
| --- | --- |
| `uv build --project python` | Wheel and sdist built successfully; wheel is approximately 2.74 MB compressed |
| `uvx twine check --strict python/dist/*` | Both distributions passed |
| `uv run --project python --directory python python -m unittest discover -s tests -v` | 7 tests passed, none skipped |
| `.venv/Scripts/python.exe scripts/check_release.py python/dist/tinyocr_onnx-0.1.0-py3-none-any.whl` | Fresh temporary venv outside checkout, installed import verified, pip check passed, 7 tests passed |
| `.venv/Scripts/python.exe scripts/check_pypi.py --run-id 40074348038 --artifact-zip runs/publish-ci-40074348038/distributions.zip` | Exit 0; successful publishing run, CI ZIP digest verified, both public distributions match tested bytes/PyPI metadata, both model payloads match in wheel and sdist |
| `.venv/Scripts/python.exe scripts/check_release.py tinyocr-onnx==0.1.0` | Exit 0 on Windows/Python 3.12.9; fresh venv outside checkout, production PyPI with pip cache disabled, pip check and 7 tests passed, no skips |
| `uv run --locked python scripts/check_parity.py` | Exact config/text/line/word parity on receipt, synthetic screen and full-scene development fixtures; zero final images used |
| `uv run --locked python -m unittest discover -s tests -v` | 53 tests passed, none skipped |
| `uv run --locked --with-requirements benchmark-requirements.txt python -m unittest discover -s tests_benchmark -v` | 14 tests passed, none skipped |
| `uv run --locked --with-requirements benchmark-requirements.txt python -m tinyocr.benchmark --verify-only` | Full 5,089-image/model fingerprints, journals and scores verified without inference |
| `uv run --locked python -m tinyocr.intake --verify-only` | 200 candidates / 198 unique images verified |
| `uv run --locked python -m tinyocr.evaluate_robustness --verify-only` | 80 cases/model verified; original failures retained |
| `uv run --locked python -m tinyocr.evaluate_final --verify-only` | Frozen final evidence and transfer checks verified without inference |
| GitHub Actions installed-wheel matrix | Windows, Linux and macOS, Python 3.12: 7 tests per platform passed, none skipped |
| Distribution and upload inventory | Two exact model hashes, licenses present, no local datasets/run journals/caches; credential-pattern scan passed |

The OCR checks load the real bundled models, prohibit network connections,
recognize an authored two-line fixture, handle a blank, preserve line output
when toggling words, verify CPU thread configuration, and exercise text/JSON
CLI output, the console command, version, missing files and invalid images.

CI: https://github.com/MMWilliams/tinyocr/actions/runs/40060878536

`results/publish-ci.json` records jobs; `results/publish-ci-artifacts.json`
records the downloaded CI archive and individual distribution digests.
`results/publish-local-artifacts.json` records the separate local build.
Archive hashes differ between local Windows and CI builds due to serialization
and archive metadata; both model payloads match the frozen hashes. PyPI received
the exact distributions built and tested within its publishing run, recorded in
`results/publish-release-artifacts.json` and `results/publish-pypi.json`.

The initial staged whitespace check found trailing whitespace and an extra EOF
blank line in two unchanged, hash-pinned upstream HierText files. Their source
bytes were deliberately retained. The check excluding `third_party/hiertext/**`
passed. Known zero-area word-box warnings in benchmark verification and empty
detection warnings on the blank fixture are expected and tested.

## PyPI publication verification

Publishing run: https://github.com/MMWilliams/tinyocr/actions/runs/40074348038
at source revision `faf7c43d2e3cdc490bd404420c72476f6aa03f91`, attempt 2.
The upload succeeded at 2026-10-24 17:35 UTC using Trusted Publishing. Only the
failed publish job was retried; it reused the artifacts that passed build,
strict metadata validation and all three installed-wheel jobs (7 tests per
platform, no skips).

The downloaded distributions ZIP matches GitHub's artifact digest, and both
distributions contain the exact frozen model payloads. Public downloads match
those distributions byte for byte. Full URLs, upload times, job conclusions,
archive digest and model hashes are in `results/publish-pypi.json`.

| Public distribution | Bytes | SHA-256 |
| --- | ---: | --- |
| `tinyocr_onnx-0.1.0-py3-none-any.whl` | 2,736,997 | `05a3752de71a04b5bfe363618c3c5c9aaab1e76ff820d2c87a882b1a0d1e7928` |
| `tinyocr_onnx-0.1.0.tar.gz` | 2,747,122 | `f0a89722272ebfbd7286c857eeff1f85f6ea0eb95b5c0fb1956e2f0432afcc5f` |

The independent installation downloaded `tinyocr-onnx==0.1.0` from production
PyPI with pip's cache disabled. The temporary venv and test working directory
were outside the checkout. pip check passed; all 7 tests passed, no skips, in
4.157 seconds. OCR tests prohibited network connections and used the real
bundled models. Evidence: `results/publish-pypi-install.json`; full local log:
`runs/publish-ci-40074348038/pypi-clean-install.log`.

Attempt 1 failed before upload with `invalid-publisher`. The account page then
showed no pending publisher. The user registered `tinyocr-onnx`, GitHub owner
`MMWilliams`, repository `tinyocr`, workflow `publish.yml`, environment `pypi`,
and attempt 2 succeeded. Attempt-1 evidence is retained in
`results/publish-attempt-1.json`. No tokens were requested in chat.

Repeat the release acceptance checks while the CI artifact remains available:

```sh
python scripts/check_pypi.py --run-id 40074348038 --artifact-zip runs/publish-ci-40074348038/distributions.zip
python scripts/check_release.py tinyocr-onnx==0.1.0
```

The saved ZIP is GitHub artifact `12552043076`. Public verification initially
encountered transient HTTP 502 responses from GitHub's attempts API; the
unchanged verifier subsequently completed successfully. The `tinyocr-onnx`
package should not be confused with the unrelated PyPI distribution `tinyocr`.
