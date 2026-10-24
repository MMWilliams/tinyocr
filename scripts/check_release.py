"""Install a wheel or exact PyPI requirement in a fresh venv outside the checkout.

Usage: python scripts/check_release.py python/dist/tinyocr_onnx-0.1.0-py3-none-any.whl
       python scripts/check_release.py tinyocr-onnx==0.1.0
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


def main():
    root = Path(__file__).resolve().parents[1]
    target = sys.argv[1]
    if Path(target).is_file():
        target = str(Path(target).resolve())
    with tempfile.TemporaryDirectory(prefix="tinyocr-release-") as directory:
        work = Path(directory)
        venv.create(work / "venv", with_pip=True)
        binaries = work / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PATH"] = str(binaries) + os.pathsep + env.get("PATH", "")
        def run(*args):
            subprocess.run([str(python), *args], cwd=work, env=env, check=True)
        run("-m", "pip", "install", "--disable-pip-version-check", target)
        run("-m", "pip", "check")
        run("-c", "import tinyocr,sys; from pathlib import Path; "
            "p=Path(tinyocr.__file__).resolve(); "
            "assert p.is_relative_to(Path(sys.prefix).resolve()), p; print('Installed import:',p)")
        shutil.copytree(root / "python/tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        run("-m", "unittest", "discover", "-s", "tests", "-v")
        print("PASS: clean installed API/CLI, bundled hashes, real offline OCR, blank and error cases", flush=True)


if __name__ == "__main__":
    main()
