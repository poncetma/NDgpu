"""Rebuild dist/ndgpu-src.zip, the archive the Colab notebooks pip-install.

The notebooks upload this zip rather than cloning, so it is the *only* thing
that carries source changes to the GPU. It is easy to forget, and a stale zip
does not fail -- it silently runs old code, which is worse. Run this after any
change under ndgpu/ and before opening a notebook on Colab:

    python tools/build_src_zip.py

Ships the package sources plus the vendored cross-section data (see the
package-data entry in pyproject.toml, without which pip drops the .npz files).
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "ndgpu-src.zip"
TOP_FILES = ["pyproject.toml", "README.md"]
KEEP_SUFFIXES = {".py", ".npz", ".csv", ".xml", ".txt", ".json", ".dat"}


def build(out: Path = OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in TOP_FILES:
            z.write(ROOT / name, name)
        for root, dirs, files in os.walk(ROOT / "ndgpu"):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            rel = Path(root).relative_to(ROOT)
            z.write(root, f"{rel}/")
            for f in sorted(files):
                if Path(f).suffix in KEEP_SUFFIXES:
                    z.write(Path(root) / f, str(rel / f))
    return out


if __name__ == "__main__":
    path = build()
    with zipfile.ZipFile(path) as z:
        n = len(z.namelist())
    print(f"wrote {path.relative_to(ROOT)} ({n} entries, "
          f"{path.stat().st_size / 1e6:.2f} MB)")
