# Provenance: how this package proves nothing was altered

This replication package is sealed with a **Merkle tree + hash chain**. The seal
makes two distinct, honestly-scoped guarantees. Read this before trusting any
number in the paper.

## The two seals (and why they differ)

| Stage | What it is | What the seal proves |
|---|---|---|
| **Generation** | The 180 raw records the Grok-4.20 API returned on 2026-07-08 (`data/pilot_gen_full.jsonl`) | **Immutability.** An LLM at temperature 0.9 is *not* bit-reproducible, so the seal proves these records were **not altered since capture** — not that you can regenerate them. This is the pre-registration LOCK. |
| **Derivation** | Everything computed *from* the frozen generation: `pilot_scores.json`, the robustness re-scores, the figures, the paper numbers | **Reproducibility.** Given the frozen generation + this code + a compatible environment, `code/score_pilot.py` reproduces the scores (modulo last-decimal embedding noise; the signs, orderings, and CI conclusions are robust). |

Conflating the two would be an over-claim. The generation is **frozen evidence**;
the derivation is **reproducible**. The chain encodes exactly that.

## The structure

Each stage is hashed into a Merkle root over its files (leaves = per-file
SHA-256, in a fixed documented order). The per-stage roots are then folded
left-to-right into a single **`artifact_root`** — the chain tip:

```
link[0]   = sha256( code_hash )
link[1]   = sha256( link[0] ‖ preregistration_hash )
link[2]   = sha256( link[1] ‖ generation_hash )      ← generation sealed here
link[3]   = sha256( link[2] ‖ scores_hash )
link[4]   = sha256( link[3] ‖ figures_hash )
artifact_root = sha256( link[4] ‖ paper_hash )
```

Changing **one byte** of any sealed file changes its stage hash, which changes
every downstream link, which changes `artifact_root`. The full tree, every leaf,
and the chain are recorded in [`PROVENANCE.json`](PROVENANCE.json).

The **generation** stage additionally re-derives each record's `record_sha256`
from its own content — `sha256(system_prompt ‖ user_prompt ‖ output_text)` — and
checks it, so the hash is tied to the bytes, not just asserted alongside them.

## Environment is context, not part of the root

`artifact_root` is deliberately **environment-independent**: the machine's
`pip_freeze` is recorded (as the pins the numbers were *derived* under) but is
**not** folded into the root. That keeps the root **portable** — any reviewer on
any machine recomputes the same root. A differing environment is a `[WARN]` in
the verifier (derived scores may differ at the last decimal), never a failure.

## The DOI is the reference

At release the Zenodo **Version DOI** anchors this `artifact_root`. The paper
cites that Version DOI (not the concept DOI), so paper ↔ repo is a single
verifiable link: the cited DOI names a deposit whose `artifact_root` you can
recompute from the files.

## Verify it yourself

```bash
python scripts/verify_chain.py    # exit 0 = intact, exit 1 = tampered
```

It recomputes the whole tree and chain from the on-disk files and fails loudly,
naming the drifted stage, if `artifact_root` does not match the seal. This is
proof by recomputation, not trust.
