#!/usr/bin/env python3
"""LOCK-safe robustness re-scores (F5.5 blind-review, findings C4 + C5).

Re-analyses the FROZEN 180 generation records (data/pilot_gen_full.jsonl) with the
*same* pre-registered pipeline (code/score_pilot.py :: score), swapping exactly one
knob at a time. NO new API/generation — this only re-embeds and re-clusters ideas
already collected, so it does not touch the LOCK.

Variants
--------
  baseline    : original config verbatim -> MUST reproduce data/pilot_scores.json
  multilingual: swap the English all-MiniLM encoder for a multilingual one
                (paraphrase-multilingual-MiniLM-L12-v2) on the PT-BR ideas   [C4]
  minstop     : minimal domain-blind stopword list -- strip ONLY the non-content
                scaffolding (structural words, mode names, persona/style code
                labels) and KEEP domain-descriptor vocabulary, to test whether
                stripping domain words biased the null against C7               [C5]

Usage:  python scripts/rescore_robustness.py
Writes: output/robustness/rescore_<variant>.json  +  prints a comparison table.

The FROZEN, sealed re-scores live in data/robustness/ (part of the provenance
chain). This script writes freshly-computed copies to output/robustness/ (a
non-sealed working directory) so re-running it can never mutate the seal; compare
the two directories to confirm reproduction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
DATA = ROOT / "data"
FROZEN = DATA / "robustness"          # sealed frozen re-scores (read-only reference)
OUT = ROOT / "output" / "robustness"  # non-sealed working output (safe to overwrite)
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE))
import pilot_common as pc  # noqa: E402
import score_pilot as sp  # noqa: E402


def load_records() -> list[dict]:
    return [json.loads(ln) for ln in (DATA / "pilot_gen_full.jsonl").open()]


def base_cfg() -> dict:
    return pc.load_config(str(CODE / "pilot_config.json"))


# ── minimal stopword set (finding C5) ────────────────────────────────────────
# Strip only pure scaffolding + persona/style *labels* (never organic idea
# content); KEEP domain-descriptor words (arquitetura, acustica, ventilacao,
# urbanismo, paisagismo, iluminacao, sustentavel, ...) that grounded ideas may
# legitimately carry. This is the charitable-to-C7 direction: if C7's cluster
# count rises materially, the original strip was load-bearing.
STRUCTURAL = {"dominio", "modo", "persona", "style", "spec", "substrato",
              "generic", "sem", "campo", "area"}
MODES = {"humano", "biomimetico", "biomimetica"}
# persona code-names (C7) + style labels (C2) — pure seeded labels
PERSONA_CODES = {"theoros", "incisus", "mutator", "vates", "silenus"}


def minimal_stopset(records: list[dict]) -> set[str]:
    """Structural + mode + persona/style-label tokens only; keep domain words.

    The C2 style label is the token right after 'style/o ' (e.g. 'absurdista');
    the C7 persona is the uppercase code after 'persona-'. Domain descriptor
    tokens (after 'dominio-NN-') are deliberately KEPT.
    """
    import re
    word = re.compile(r"[a-zA-ZÀ-ÿ]+")
    stop = set(STRUCTURAL) | set(MODES) | set(PERSONA_CODES)
    for r in records:
        sid = r.get("source_identity", "").lower()
        # C2 style label: 'style/o <label>' possibly multiword ('o arquiteto do caos')
        if sid.startswith("style/"):
            for tok in word.findall(sid.split("/", 1)[1]):
                if tok not in {"o", "a", "do", "da", "de"}:
                    stop.add(tok)
    return {w for w in stop if len(w) > 1}


def run_variant(name: str, records: list[dict]) -> dict:
    cfg = base_cfg()
    emb = cfg["scoring"]["embedding"]
    if name == "baseline":
        pass
    elif name == "multilingual":
        emb["primary_model"] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    elif name == "minstop":
        # feed a precomputed minimal stopset; disable source-vocab harvesting
        emb["domain_blind"]["strip_source_vocabulary"] = False
        emb["domain_blind"]["extra_stopwords"] = sorted(minimal_stopset(records))
    else:
        raise SystemExit(f"unknown variant {name}")
    res = sp.score(records, cfg)
    (OUT / f"rescore_{name}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def summarize(name: str, res: dict) -> dict:
    m = res["metrics"]
    bd = res["bootstrap_diff"]
    def g(a, k):
        return m[a]["clusters_per_1k_input_tokens"][k]
    return {
        "variant": name,
        "backend": res["meta"]["embedding_backend"],
        "detail": res["meta"]["embedding_detail"][:48],
        "clusters_agg": {a: m[a]["clusters"]["agglomerative"] for a in m},
        "clusters_hdb": {a: m[a]["clusters"]["hdbscan"] for a in m},
        "per1k_agg": {a: round(g(a, "agglomerative"), 3) for a in m},
        "per1k_hdb": {a: round(g(a, "hdbscan"), 3) for a in m},
        "cos_dist": {a: round(m[a]["mean_pairwise_cosine_distance"], 3) for a in m},
        "rho": {a: (round(m[a]["rho"], 4) if m[a]["rho"] is not None else None) for a in m},
        "C7_last_tokens_agg": min(m, key=lambda a: g(a, "agglomerative")) == "C7-proxy",
        "C7_last_tokens_hdb": min(m, key=lambda a: g(a, "hdbscan")) == "C7-proxy",
        "C7vsC1_tok_agg": _c(bd, "C7-proxy_vs_C1", "agglomerative", "tokens_matched"),
        "C7vsC1_tok_hdb": _c(bd, "C7-proxy_vs_C1", "hdbscan", "tokens_matched"),
        "C7vsC2_prop_agg": _c(bd, "C7-proxy_vs_C2", "agglomerative", "proposals_matched"),
        "C7vsC2_prop_hdb": _c(bd, "C7-proxy_vs_C2", "hdbscan", "proposals_matched"),
        "C7vsC1_prop_agg": _c(bd, "C7-proxy_vs_C1", "agglomerative", "proposals_matched"),
        "ordering_invariant": res["ordering"]["invariant"],
    }


def _c(bd, contrast, algo, view):
    d = bd[contrast][algo][view]
    return [round(d["mean_diff"], 3), round(d["ci_low"], 3), round(d["ci_high"], 3)]


def main() -> int:
    records = load_records()
    print(f"loaded {len(records)} frozen records\n")
    sums = []
    for name in ("baseline", "multilingual", "minstop"):
        print(f"--- {name} ---")
        res = run_variant(name, records)
        s = summarize(name, res)
        sums.append(s)
        print(f"  backend={s['backend']} ({s['detail']})")
        print(f"  clusters agg C1/C2/C7 = {list(s['clusters_agg'].values())}"
              f"  hdb = {list(s['clusters_hdb'].values())}")
        print(f"  per-1k agg = {s['per1k_agg']}")
        print(f"  C7 last tokens-matched: agg={s['C7_last_tokens_agg']} hdb={s['C7_last_tokens_hdb']}")
        print(f"  C7-vs-C1 tokens agg={s['C7vsC1_tok_agg']} hdb={s['C7vsC1_tok_hdb']}")
        print(f"  C7-vs-C2 proposals agg={s['C7vsC2_prop_agg']} hdb={s['C7vsC2_prop_hdb']}")
        print(f"  cos-dist={s['cos_dist']} rho={s['rho']}\n")

    # Baseline reproduction check vs the frozen scores
    frozen = json.loads((DATA / "pilot_scores.json").read_text())
    b = next(s for s in sums if s["variant"] == "baseline")
    ok = (b["clusters_agg"] == {a: frozen["metrics"][a]["clusters"]["agglomerative"] for a in frozen["metrics"]}
          and b["clusters_hdb"] == {a: frozen["metrics"][a]["clusters"]["hdbscan"] for a in frozen["metrics"]})
    print(f"BASELINE reproduces frozen agglomerative+hdbscan cluster counts: {ok}")
    (OUT / "rescore_summary.json").write_text(json.dumps(sums, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}/rescore_summary.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
