"""Build the pinned upstream C++ simulator locally without changing its physics."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import urllib.request
from .paths import ROOT, REFERENCE, RUNTIME, DEPS, BINARY, UPSTREAM_COMMIT


def _download(url, target):
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())


def ensure_runtime(build=True):
    if not REFERENCE.exists():
        if not build:
            raise FileNotFoundError("Run python -m firehose_reproduction setup first")
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/aidan-curtis/firehose.git", str(REFERENCE)], check=True)
        subprocess.run(["git", "-C", str(REFERENCE), "checkout", UPSTREAM_COMMIT], check=True)
    revision = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("Firehose reference checkout does not match the pinned commit")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # The C++ binary expects local input/output folders. Source stays untouched.
    if not (RUNTIME / "cell2fire").exists():
        shutil.copytree(REFERENCE / "cell2fire", RUNTIME / "cell2fire")
    if not (RUNTIME / "data").exists():
        (RUNTIME / "data").symlink_to(REFERENCE / "data", target_is_directory=True)
    # Adopt the already compiled development build, if present.
    existing = RUNTIME / "Cell2FireC/Cell2Fire"
    if existing.exists() and not BINARY.exists():
        shutil.copy2(existing, BINARY)
    if not BINARY.exists():
        if not build:
            raise FileNotFoundError("Cell2Fire binary is missing: run the setup command")
        DEPS.mkdir(parents=True, exist_ok=True)
        eigen = DEPS / "eigen-3.4.0"
        if not eigen.exists():
            archive = DEPS / "eigen.tar.gz"
            _download("https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz", archive)
            with tarfile.open(archive) as handle:
                handle.extractall(DEPS, filter="data")
        boost = DEPS / "boost/usr/include"
        if not boost.exists():
            archive = DEPS / "boost.deb"
            _download("https://deb.debian.org/debian/pool/main/b/boost1.83/libboost1.83-dev_1.83.0-4.2_amd64.deb", archive)
            payload = subprocess.check_output(["ar", "p", str(archive), "data.tar.xz"])
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as handle:
                handle.extractall(DEPS / "boost", filter="data")
        subprocess.run(["make", "-C", str(BINARY.parent), "-f", "Makefile_UBUNTU",
                        f"CFLAGS=-std=c++11 -O3 -I{eigen} -I{boost}", "-j2"], check=True)
    manifest = {"upstream_commit": revision, "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
                "build": "Original C++ source, Linux Makefile; local Eigen3.4.0 and Boost1.83 headers",
                "source_modified": False}
    (RUNTIME / "build_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(ensure_runtime(), indent=2))
