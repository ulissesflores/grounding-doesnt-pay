"""Seal the replication artifacts into a Merkle tree + hash chain.

Two seals, deliberately distinct (see PROVENANCE.md):

  * GENERATION is frozen evidence. The 180 records are what the Grok-4.20 API
    returned on 2026-07-08; an LLM at temperature 0.9 is NOT bit-reproducible, so
    the chain proves the records were NOT ALTERED since capture — not that they
    can be regenerated. We additionally re-derive each `record_sha256` from its
    content, tying the hash to the bytes.
  * DERIVATION (scores, figures, paper numbers) IS reproducible: given the frozen
    generation + this code + a compatible environment, `score_pilot.py` reproduces
    `pilot_scores.json` (modulo last-decimal embedding noise, per the paper).

The per-stage Merkle roots are folded left-to-right into an `artifact_root`
(the chain tip). Any byte change in any sealed file changes it. The environment
fingerprint is recorded as CONTEXT (the pins the numbers were derived under) but
is NOT folded into `artifact_root`, so the root stays portable — any reviewer on
any machine can recompute and verify it. The Zenodo Version DOI of the release
anchors `artifact_root`; that DOI is the paper<->repo reference.

Usage: python scripts/build_chain.py
Writes: PROVENANCE.json (canonical, repo root) and a sealed runs/<id>/.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from hash_utils import (
    environment_fingerprint,
    hash_chain,
    merkle_root,
    save_json,
    sha256_file,
    sha256_hex,
    sha256_text,
)

ROOT = Path(__file__).resolve().parent.parent

# Sealed artifact groups, in a fixed, documented order. Leaves within a group are
# sorted by relative path (generation keeps on-disk record order). The chain folds
# the groups in THIS order (causal flow: code -> prereg -> generation -> ...).
CODE_GLOBS = ["code/*.py", "code/*.json", "scripts/*.py", "tests/*.py"]
PREREG = "data/PREREGISTRATION.md"
GENERATION = "data/pilot_gen_full.jsonl"
SCORES = ["data/pilot_scores.json", "data/pilot_posthoc.json", "data/pilot_report.md",
          "data/robustness/rescore_baseline.json", "data/robustness/rescore_multilingual.json",
          "data/robustness/rescore_minstop.json", "data/robustness/rescore_summary.json",
          "data/robustness/ROBUSTNESS-FINDINGS.md"]
FIGURES = ["output/figures/fig1_clusters_per_1k.png", "output/figures/fig2_caching_breakeven.png"]
PAPER = ["docs/paper/paper-final.pdf", "docs/paper/paper-final.docx"]


def _leaves(rel_paths: list[str]) -> dict[str, str]:
    """Ordered {relative_path: sha256} for a fixed file list (skips missing)."""
    out: dict[str, str] = {}
    for rel in sorted(rel_paths):
        p = ROOT / rel
        if p.exists():
            out[rel] = sha256_file(p)
    return out


def _glob_leaves(globs: list[str]) -> dict[str, str]:
    rels = sorted({str(p.relative_to(ROOT)) for g in globs for p in ROOT.glob(g)
                   if "__pycache__" not in p.parts})
    return {rel: sha256_file(ROOT / rel) for rel in rels}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, cwd=ROOT).decode().strip()
    except Exception:
        return "UNKNOWN"


def _generation_stage() -> dict:
    """Seal the frozen generation: file hash + Merkle root of the per-record
    hashes, each re-derived from its content and checked (hash <-> bytes tie)."""
    jsonl = ROOT / GENERATION
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    recomputed, mismatches = [], []
    for r in records:
        h = sha256_text(r["system_prompt"], r["user_prompt"], r["output_text"])
        recomputed.append(h)
        if h != r["record_sha256"]:
            mismatches.append(r.get("call_id", "?"))
    return {
        "name": "generation",
        "kind": "frozen-evidence",
        "note": "LLM output (Grok-4.20 @ T=0.9) — immutable, NOT bit-reproducible; chain proves no tampering, not regeneration",
        "file_sha256": sha256_file(jsonl),
        "n_records": len(records),
        "record_merkle_root": merkle_root(recomputed),
        "records_reverified": not mismatches,
        "record_hash_mismatches": mismatches,
        # the stage hash binds the file, the record tree, and the re-derivation result
        "hash": sha256_text(sha256_file(jsonl), merkle_root(recomputed), "ok" if not mismatches else "MISMATCH"),
    }


def _merkle_stage(name: str, kind: str, leaves: dict[str, str]) -> dict:
    return {"name": name, "kind": kind, "merkle_root": merkle_root(list(leaves.values())),
            "leaves": leaves, "hash": merkle_root(list(leaves.values()))}


def compute_provenance() -> dict:
    """Pure computation of the provenance tree + chain from the on-disk files.
    No side effects — build_chain writes it, verify_chain recomputes and compares."""
    env = environment_fingerprint()
    env_hash = sha256_hex(json.dumps(env, sort_keys=True, ensure_ascii=False).encode("utf-8"))

    # Figures are NOT sealed into artifact_root: they are a reproducible derivation
    # whose PNG bytes vary across matplotlib/freetype versions, and the published
    # figures are already fixed inside the sealed paper PDF. We record their hashes
    # as derived-artifact context, outside the root.
    stages = [
        _merkle_stage("code", "derivation-input", _glob_leaves(CODE_GLOBS)),
        _merkle_stage("preregistration", "frozen-before-collection", _leaves([PREREG])),
        _generation_stage(),
        _merkle_stage("scores", "reproducible-derivation", _leaves(SCORES)),
        _merkle_stage("paper", "artifact", _leaves(PAPER)),
    ]
    stage_hashes = [s["hash"] for s in stages]
    links, artifact_root = hash_chain(stage_hashes)
    for s, link in zip(stages, links, strict=False):
        s["chain_link"] = link

    return {
        "provenance_version": "1.1",
        "algorithm": "sha256",
        "paper": "Grounding Doesn't Pay: A Token-Matched Negative Result on Creative Diversity",
        "semantics": {
            "generation": "frozen evidence — immutable, not bit-reproducible (LLM API @ T=0.9); the chain proves the records were not altered, not that they regenerate",
            "derivation": "reproducible — scores re-derive from the frozen generation under the declared reproduction environment (exact cluster counts depend on the sklearn/sentence-transformers pins; signs, orderings and CIs are robust)",
            "artifact_root": "portable, machine-independent tamper-evidence over code + preregistration + generation + scores + paper; anchored by the Zenodo Version DOI",
            "figures": "NOT in artifact_root — reproducible derivation, byte-variable across matplotlib versions; the published figures are fixed inside the sealed paper PDF",
        },
        # The pins the scores reproduce under (NOT the machine that signed the seal).
        "reproduction_environment": {
            "declared_requirements": ["requirements.txt", "requirements-full.txt"],
            "note": "scores are frozen evidence from an earlier session; reproduction runs under these declared pins. sentence-transformers/all-MiniLM-L6-v2 reproduces the exact numbers; a TF-IDF fallback (no torch) reproduces the verdict (the null).",
        },
        "sealed_by_interpreter": {"python": env.get("python_version"),
                                  "platform": env.get("platform"), "sha256": env_hash,
                                  "note": "metadata about the signing interpreter only; NOT the environment of derivation and NOT folded into artifact_root"},
        "derived_artifacts_not_sealed": _leaves(FIGURES),
        "stages": stages,
        "chain_order": [s["name"] for s in stages],
        "artifact_root": artifact_root,
        "doi_anchor": "set at release: the Zenodo Version DOI binds this artifact_root",
    }


def main() -> None:
    provenance = compute_provenance()
    artifact_root = provenance["artifact_root"]
    stages = provenance["stages"]
    save_json(provenance, ROOT / "PROVENANCE.json")

    # sealed run (canon requires runs/<id>/manifest.json with pip_freeze). Run
    # build_chain.py with the scoring venv's interpreter so this captures the
    # reproduction environment, not an unrelated system Python.
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git": {"repository_url": "set at release", "commit": _git_commit(), "tag": None},
        "environment": environment_fingerprint(),
        "artifact_root": artifact_root,
        "chain_algorithm": "sha256 Merkle-per-stage + left-fold chain (scripts/build_chain.py)",
        "notes": "grounding-doesnt-pay replication seal; generation frozen (LOCK), scores reproducible under the declared pins",
    }
    save_json(manifest, run_dir / "manifest.json")
    save_json(provenance, run_dir / "chain.json")

    sealed = (list(_glob_leaves(CODE_GLOBS)) + [PREREG, GENERATION] + SCORES + PAPER)
    lines = [f"{sha256_file(ROOT / rel)}  {rel}" for rel in sealed if (ROOT / rel).exists()]
    (run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gen = next(s for s in stages if s["name"] == "generation")
    print(f"[OK] artifact_root = {artifact_root}")
    print(f"[OK] generation: {gen['n_records']} records, hashes re-verified = {gen['records_reverified']}")
    print(f"[OK] PROVENANCE.json + runs/{run_id}/ (manifest, chain, checksums)")


if __name__ == "__main__":
    main()
