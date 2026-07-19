"""Audit-grade hashing and environment fingerprint utilities.

Scientific intent:
- Ensure that every data artifact produced by the pipeline can be verified
  and referenced immutably.
- Support reproducibility by recording environment metadata.

This module is part of the project's reproducibility contract:
runs/<run_id>/checksums.sha256 must exist for a run to be considered valid.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash for a file (streamed).

    Parameters
    ----------
    path:
        File path to hash.
    chunk_size:
        Bytes per read; larger improves performance on big parquet files.

    Returns
    -------
    str
        SHA256 hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums_sha256(file_paths: Iterable[Path], out_path: Path) -> dict[str, str]:
    """Write checksums.sha256 in standard format.

    Parameters
    ----------
    file_paths : Iterable[Path]
        List of file paths to hash.
    out_path : Path
        Output path for the checksum file.

    Returns
    -------
    dict[str, str]
        Dictionary mapping relative paths to SHA256 hashes.

    Format: <sha256>  <relative_path>

    Scientific rule:
    - relative_path MUST be relative to the directory containing checksums.sha256
      (i.e., the run directory), to keep artifacts portable across machines.

    """
    out_path = out_path.resolve()
    base_dir = out_path.parent.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {}
    lines: list[str] = []

    for p in file_paths:
        p = Path(p).resolve()
        digest = sha256_file(p)

        try:
            rel = str(p.relative_to(base_dir))
        except ValueError:
            # If artifact is outside the run directory, still record it,
            # but this should be avoided in this project.
            rel = str(p)

        mapping[rel] = digest
        lines.append(f"{digest}  {rel}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mapping


def get_pip_freeze() -> list[str]:
    """Return installed-package pins for environment traceability.

    Uses importlib.metadata (no subprocess): robust in uv/venv environments where
    the `pip` module itself is absent (audit finding: silent empty pip_freeze).
    """
    try:
        import importlib.metadata as _md

        pins = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in _md.distributions()
            if d.metadata and d.metadata.get("Name")
        )
        if pins:
            return pins
    except Exception:
        pass
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        return [line.strip() for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def environment_fingerprint() -> dict:
    """Capture environment metadata for manifest.json.

    Returns
    -------
    dict
        Dictionary containing environment metadata (platform, python version, etc.).

    Notes
    -----
    - GPU/CUDA fingerprint will be added later in training scripts.
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture": platform.machine(),
        "pip_freeze": get_pip_freeze(),
    }


def save_json(obj: dict, path: Path) -> None:
    """Write a JSON file with stable formatting.

    Parameters
    ----------
    obj : dict
        Dictionary to serialize.
    path : Path
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


# ─── Merkle tree + hash chain (provenance sealing) ───────────────────────────
# The tree hashes groups of artifacts (code, generation records, figures); the
# chain folds the per-stage roots left-to-right so that ANY byte change in ANY
# stage changes the final ROOT. The ROOT is what the Zenodo Version DOI anchors.


def sha256_hex(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(*parts: str) -> str:
    """SHA-256 of the UTF-8 concatenation of text parts.

    Identical scheme to the pilot harness (`pilot_common.sha256_text`), so the
    per-record `record_sha256` can be re-derived and checked against the data.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def merkle_root(leaves: list[str]) -> str:
    """Binary Merkle root over hex leaf digests (SHA-256).

    Odd levels duplicate the last node (Bitcoin-style). Empty -> sha256(b'').
    The leaves must be passed in a FIXED, documented order (e.g. sorted by path,
    or the on-disk record order); the root is order-sensitive by construction.
    Intent here is tamper-evidence over a KNOWN, fixed artifact set, not defence
    against an adversary crafting a second pre-image of the whole set.
    """
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256((level[i] + level[i + 1]).encode("utf-8")).hexdigest()
                 for i in range(0, len(level), 2)]
    return level[0]


def hash_chain(stage_hexes: list[str]) -> tuple[list[str], str]:
    """Left-fold hash chain over per-stage hex digests.

    link[0] = sha256(stage[0]); link[i] = sha256(link[i-1] || stage[i]).
    Returns (links, root). The root is the chain tip: changing any stage hash
    (hence any underlying byte) changes it. Deterministic; no timestamps.
    """
    links: list[str] = []
    prev = ""
    for i, s in enumerate(stage_hexes):
        material = s if i == 0 else prev + s
        prev = hashlib.sha256(material.encode("utf-8")).hexdigest()
        links.append(prev)
    root = links[-1] if links else hashlib.sha256(b"").hexdigest()
    return links, root
