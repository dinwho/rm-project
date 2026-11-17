"""Pinned upstream checkout and isolated, writable simulator runtime."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "e49a52a9a987a1585d25ddca2f711222ca500ffb"
REFERENCE = ROOT / "artifacts/references/firehose"
RUNTIME = ROOT / "artifacts/firehose_runtime"
DEPS = ROOT / "artifacts/deps"
BINARY = RUNTIME / "cell2fire/Cell2FireC/Cell2Fire"
