"""Shared fingerprints and atomic writes for generated artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

import numpy as np


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def arrays_fingerprint(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(json.dumps([array.dtype.str, array.shape]).encode("ascii") + b"\0")
        digest.update(array)
    return digest.hexdigest()


def read_npz_cache(path: Path, keys: Iterable[str]) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in keys}
    except (OSError, ValueError, KeyError, EOFError, TypeError, BadZipFile):
        return None


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-") as tmp:
        staging = Path(tmp) / path.name
        np.savez_compressed(staging, **arrays)
        staging.replace(path)


@contextmanager
def publish_directory(path: Path) -> Iterator[Path]:
    """Expose the final run only after all files have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-") as tmp:
        staging = Path(tmp)
        yield staging
        if path.exists():
            raise FileExistsError(path)
        os.rename(staging, path)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
