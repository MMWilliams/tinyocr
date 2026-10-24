# TinyOCR 0.1.0 publication

Status on 2026-10-24: **failed publication verification**. GitHub and Hugging Face
are public; local and Windows/Linux/macOS installed-wheel checks passed. After
the user reported adding the publisher, PyPI rejected the publishing workflow
with `invalid-publisher`. No PyPI upload occurred; public read-back and a clean
install from PyPI remain blocked on correcting the publisher registration.

## Published artifacts

- GitHub: https://github.com/MMWilliams/tinyocr
- Hugging Face: https://huggingface.co/maureesewilliams/tinyocr
- Intended PyPI project: https://pypi.org/project/tinyocr-onnx/ (not yet published)

GitHub source/workflow revision: `e09f5381ddc9c4a8ea054f1de3a3bca48e4186d3`.
Hugging Face revision: `46f0f44e65383248cbeeced5dca6397c595314fa`.
Unauthenticated read-back matched both frozen ONNX hashes from GitHub and all
eight uploaded files from Hugging Face. Evidence: `results/publish-github.json`
and `results/publish-huggingface.json`.

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
and archive metadata; both model payloads match the frozen hashes. PyPI will
receive the exact distributions built and tested within its publishing run.

The initial staged whitespace check found trailing whitespace and an extra EOF
blank line in two unchanged, hash-pinned upstream HierText files. Their source
bytes were deliberately retained. The check excluding `third_party/hiertext/**`
passed. Known zero-area word-box warnings in benchmark verification and empty
detection warnings on the blank fixture are expected and tested.

## Remaining PyPI step

Publishing run: https://github.com/MMWilliams/tinyocr/actions/runs/40074348038
at source revision `faf7c43d2e3cdc490bd404420c72476f6aa03f91`.
Build, strict metadata validation and all three installed-wheel jobs passed
(7 tests per platform, no skips). The publish job failed during OIDC exchange:
`invalid-publisher`: valid token, but no corresponding publisher was found.
The public PyPI version API returned HTTP 404. Evidence, including the actual
workflow claims and test log excerpts: `results/publish-attempt-1.json`.

The downloaded distributions ZIP matches GitHub's artifact digest, and both
distributions contain the exact frozen model payloads. File digests are recorded
in `results/publish-release-artifacts.json`. The public verifier
`scripts/check_pypi.py` correctly rejected the unsuccessful workflow before
attempting to claim release success.

Confirm/correct the pending publisher on **production PyPI** at
https://pypi.org/manage/account/publishing/:

| Field | Value |
| --- | --- |
| PyPI project | `tinyocr-onnx` |
| GitHub owner | `MMWilliams` |
| Repository | `tinyocr` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

The actual GitHub token claims match the owner, repository, workflow and
environment above. Account-side field confirmation was requested from the user;
no tokens were requested in chat. PyPI documents this error as a mismatch between
the registered publisher and token claims. See
[PyPI's troubleshooting documentation](https://docs.pypi.org/trusted-publishers/troubleshooting/).

After correcting the registration, retry only the failed publish job so it
uploads the already-tested distributions:

```sh
gh run rerun 40074348038 --repo MMWilliams/tinyocr --failed
```

The workflow builds once, tests that wheel on all three platforms, then uploads
the same wheel and sdist using OIDC. Inspect the publishing run, read back PyPI
files and compare their hashes with that run's artifacts, then execute:

```sh
python scripts/check_release.py tinyocr-onnx==0.1.0
```

The public-byte verification command is:

```sh
python scripts/check_pypi.py --run-id 40074348038 --artifact-zip runs/publish-ci-40074348038/distributions.zip
```

It requires a successful publishing run, compares both public downloads to the
digest-verified CI archive, and checks frozen model payloads in the wheel and
sdist. Record its results and the clean-install result here before marking the
publication task complete. The `tinyocr-onnx` package should not be confused
with the unrelated existing PyPI distribution named `tinyocr`.
