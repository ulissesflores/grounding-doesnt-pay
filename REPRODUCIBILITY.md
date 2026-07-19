# Reproducibility

Two tracks. Track 1 confirms the published artifact is intact and every paper
number ties to the frozen data — no heavy dependencies. Track 2 re-derives the
scores from the raw generation. Generation itself is frozen and never re-run
(see [`PROVENANCE.md`](PROVENANCE.md)).

## Track 1 — verify integrity + numbers + figures (light)

Needs Python 3.11+ and `matplotlib` (the tests use the standard library only).

```bash
pip install -r requirements.txt
python run_all.py
```

This runs, in order:

1. `scripts/verify_chain.py` — recompute the Merkle tree + hash chain and confirm
   `artifact_root` matches the seal; re-tie every generation record hash to its
   content. Exit 1 if anything was altered.
2. `tests/test_paper_numbers.py` — assert every headline number, CI, ratio,
   per-query value, the §5 caching break-evens, and the §4.5 encoder/stopword
   robustness numbers equal the frozen JSON.
3. `scripts/make_figures.py` — regenerate `output/figures/fig1,fig2.png` from the
   locked scores.

## Track 2 — re-derive the scores from raw generation (heavy)

Re-embeds and re-clusters the 180 frozen records, reproducing `pilot_scores.json`
and the encoder-robustness re-scores.

```bash
pip install -r requirements-full.txt
python run_all.py --with-rescore
```

> The sentence encoder and clustering can differ at the last decimal across
> hardware and library versions. What the paper claims — the **signs, orderings,
> and confidence-interval conclusions** — is robust to that noise. The clustering
> stack is pinned exactly (`requirements.lock`); the MiniLM encoder is pinned by
> range (`sentence-transformers`) rather than by a recorded torch build, so a
> **bit-for-bit** match of `pilot_scores.json` is not guaranteed across machines —
> it was reproduced once under the original environment (`test_robustness_baseline`
> confirms the frozen counts). The **verdict** (the null) reproduces even under the
> TF-IDF fallback, which needs no encoder at all.

## What is frozen and cannot be re-run here

`code/run_pilot_c1c7.py` produced the raw records via OpenRouter `x-ai/grok-4.20`
(non-reasoning) and needs the original domain corpus, router, and a live API key.
**The data are frozen; no new generation is part of this artifact** (the
pre-registration LOCK). The generation script ships as *evidence of method*, not
as a step to run — and the provenance seal proves the shipped records are exactly
what was captured.

## Layout

| Path | Contents |
|---|---|
| `code/` | Harness verbatim: `run_pilot_c1c7.py` (generation), `score_pilot.py` (pre-registered scoring), `posthoc_analysis.py`, `pilot_common.py`, `pilot_config.json`. |
| `data/` | Frozen: `pilot_gen_full.jsonl` (180 records, SHA-256 each), `pilot_scores.json`, `pilot_posthoc.json`, `PREREGISTRATION.md`, `robustness/`. |
| `scripts/` | `make_figures.py`, `rescore_robustness.py`, `hash_utils.py`, `build_chain.py` (author seal), `verify_chain.py` (reviewer proof). |
| `tests/` | `test_paper_numbers.py` — every number ↔ data. |
| `output/figures/` | `fig1_*`, `fig2_*` regenerated from the scores. |
| `docs/paper/` | The published paper (`paper-final.pdf`, `paper-final.docx`). |
| `runs/<id>/` | Sealed run: `manifest.json` (environment + git + `artifact_root`), `chain.json`, `checksums.sha256`. |
| `PROVENANCE.json` | The canonical Merkle tree + hash chain (see `PROVENANCE.md`). |
