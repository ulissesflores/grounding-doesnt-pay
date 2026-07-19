"""Post-hoc exploratory analysis (NOT pre-registered) — Grounding Doesn't Pay pilot.

This script does NOT touch the pre-registered gate, DV, threshold or config.
It reuses the exact scoring functions from `score_pilot.py` and reports three
robustness/exploratory views that reviewers of the null will ask for:

  A. Per-query split — does the tokens-matched verdict (C7-proxy last) hold
     within EACH query separately, not only pooled?
  B. Budget-unit robustness — the decisive verdict under input-tokens (the
     pre-registered denominator) vs total-tokens (input+output) vs actual
     USD cost. Cluster counts are fixed; only the denominator changes.
  C. C2-vs-C1 contrast — the pre-registration only bootstrapped C7-vs-C1 and
     C7-vs-C2. C2 is the tokens-matched WINNER, so its edge over C1 deserves
     a CI (positive finding, reported honestly alongside the null).

Everything here is EXPLORATORY and labeled as such. The pre-registered gate
decision comes from `score_pilot.py` only. Run:

    python code/posthoc_analysis.py --in data/pilot_gen_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc
import score_pilot as sp

ARMS = ["C1", "C2", "C7-proxy"]


def build_arm_labels(records: list[dict], cfg: dict) -> dict:
    """Embed (domain-blind) + cluster each arm, reusing score_pilot functions.

    Returns per-arm labels (both algos), n_ideas, and token/cost budgets.
    Embedding is per-idea (MiniLM encodes each text independently), so a
    subset embeds to the SAME vectors it would in the full run.
    """
    stopset = sp.build_stopset(records, cfg)
    db = cfg["scoring"]["embedding"]["domain_blind"]
    per_arm: dict[str, dict] = {}

    for arm in ARMS:
        ideas: list[str] = []
        seen: set[str] = set()
        in_tok = out_tok = 0
        cost = 0.0
        for r in (x for x in records if x["arm"] == arm):
            in_tok += int(r.get("input_tokens", 0))
            out_tok += int(r.get("output_tokens", 0))
            cost += float(r.get("cost_usd", 0.0) or 0.0)
            for idea in sp.extract_ideas(r.get("output_text", ""),
                                         cfg["scoring"]["atomic_extraction"]["min_chars"]):
                key = idea.lower().strip()
                if cfg["scoring"]["atomic_extraction"]["exact_dedup"] and key in seen:
                    continue
                seen.add(key)
                ideas.append(idea)
        per_arm[arm] = {"ideas": ideas, "n_ideas": len(ideas),
                        "input_tokens": in_tok, "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok, "cost_usd": cost}

    # Embed each arm's ideas (domain-blind), cluster with both algorithms.
    for arm in ARMS:
        texts = per_arm[arm]["ideas"]
        n = len(texts)
        if n == 0:
            per_arm[arm]["labels"] = {"hdbscan": np.zeros(0, int),
                                      "agglomerative": np.zeros(0, int)}
            continue
        cleaned = [sp.domain_blind(t, stopset, db.get("lowercase", True)) or t.lower()
                   for t in texts]
        emb, _, _ = sp.embed_corpus(cleaned, cfg)
        dist = sp.cosine_distance_matrix(emb)
        per_arm[arm]["labels"] = sp.cluster_arm(dist, cfg)
    return per_arm


def clusters(per_arm: dict, arm: str) -> dict:
    lab = per_arm[arm]["labels"]
    return {"hdbscan": sp.n_distinct(lab["hdbscan"]),
            "agglomerative": sp.n_distinct(lab["agglomerative"])}


def boot_diff_ci(per_arm, a, b, algo, denom_key, rng, resamples, level):
    """Bootstrap CI of (clusters_a/denom_a - clusters_b/denom_b).

    denom_key: 'n_ideas' (proposals), 'input_tokens', 'total_tokens', 'cost_usd'.
    Token/cost denominators are scaled to per-1k-token / per-dollar units.
    """
    la, lb = per_arm[a]["labels"][algo], per_arm[b]["labels"][algo]
    da = sp.boot_distinct(la, resamples, rng)
    dbb = sp.boot_distinct(lb, resamples, rng)
    if denom_key == "n_ideas":
        na, nb = per_arm[a]["n_ideas"], per_arm[b]["n_ideas"]
    elif denom_key == "cost_usd":
        na, nb = per_arm[a]["cost_usd"], per_arm[b]["cost_usd"]
    else:  # token denominators -> per 1k
        na, nb = per_arm[a][denom_key] / 1000.0, per_arm[b][denom_key] / 1000.0
    if not na or not nb:
        return None
    diff = da / na - dbb / nb
    lo, hi = sp.ci(diff, level)
    return {"mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
            "denom_a": na, "denom_b": nb}


def point_ratio(per_arm, a, b, algo, denom_key):
    """Budget gap (denom_a/denom_b) vs cluster gain (clusters_a/clusters_b)."""
    ca = clusters(per_arm, a)[algo]
    cb = clusters(per_arm, b)[algo]
    da = per_arm[a][denom_key]
    dbv = per_arm[b][denom_key]
    return {"clusters_a": ca, "clusters_b": cb,
            "cluster_ratio_a_over_b": (ca / cb) if cb else None,
            "budget_ratio_a_over_b": (da / dbv) if dbv else None}


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-hoc exploratory analysis")
    ap.add_argument("--in", dest="infile", default="results/pilot_gen_full.jsonl")
    ap.add_argument("--config", default=str(pc.CONFIG_PATH))
    args = ap.parse_args()

    cfg = pc.load_config(args.config)
    path = Path(args.infile)
    if not path.is_absolute() and not path.exists():
        path = pc.RESULTS_DIR / path.name  # fallback to the legacy results dir
    records = pc.read_jsonl(path)

    boot = cfg["scoring"]["bootstrap"]
    resamples, level, seed = boot["resamples"], boot["ci"], boot["seed"]

    out: dict = {"note": "POST-HOC EXPLORATORY — NOT pre-registered. The gate "
                         "decision is from score_pilot.py only. These views answer "
                         "reviewer robustness questions on already-collected data.",
                 "n_records": len(records)}

    # ── A. Per-query split ────────────────────────────────────────────────
    queries = sorted({r["query_id"] for r in records})
    out["per_query"] = {}
    for q in queries:
        sub = [r for r in records if r["query_id"] == q]
        pa = build_arm_labels(sub, cfg)
        rng = np.random.default_rng(seed)
        c7c1 = boot_diff_ci(pa, "C7-proxy", "C1", "agglomerative", "input_tokens",
                            rng, resamples, level)
        c7_last_agg = clusters(pa, "C7-proxy")["agglomerative"] / (pa["C7-proxy"]["input_tokens"] / 1000)
        c1_agg = clusters(pa, "C1")["agglomerative"] / (pa["C1"]["input_tokens"] / 1000)
        c2_agg = clusters(pa, "C2")["agglomerative"] / (pa["C2"]["input_tokens"] / 1000)
        out["per_query"][q] = {
            "n_records": len(sub),
            "clusters_per_1k_input_tok_agg": {"C1": c1_agg, "C2": c2_agg, "C7-proxy": c7_last_agg},
            "C7_is_last_tokens_matched": c7_last_agg < c1_agg and c7_last_agg < c2_agg,
            "boot_C7_vs_C1_input_tokens": c7c1,
            "verdict": "AMPUTATE-consistent" if (c7c1 and c7c1["ci_high"] < 0) else "check",
        }

    # ── B. Budget-unit robustness (full set) ──────────────────────────────
    pa_full = build_arm_labels(records, cfg)
    rng = np.random.default_rng(seed)
    out["budget_unit_robustness"] = {"algorithm": "agglomerative", "contrasts": {}}
    for a, b in (("C7-proxy", "C1"), ("C7-proxy", "C2")):
        entry = {}
        for denom, unit in (("input_tokens", "per_1k_input_tok"),
                            ("total_tokens", "per_1k_total_tok"),
                            ("cost_usd", "per_usd")):
            rng2 = np.random.default_rng(seed)
            entry[unit] = {
                "point": point_ratio(pa_full, a, b, "agglomerative", denom),
                "boot": boot_diff_ci(pa_full, a, b, "agglomerative", denom,
                                     rng2, resamples, level),
            }
        out["budget_unit_robustness"]["contrasts"][f"{a}_vs_{b}"] = entry

    # ── C. C2-vs-C1 contrast (the positive finding) ───────────────────────
    out["C2_vs_C1"] = {"algorithm": "agglomerative"}
    for denom, unit in (("n_ideas", "proposals_matched"),
                        ("input_tokens", "tokens_matched_per_1k")):
        rng3 = np.random.default_rng(seed)
        out["C2_vs_C1"][unit] = boot_diff_ci(pa_full, "C2", "C1", "agglomerative",
                                             denom, rng3, resamples, level)

    out_path = pc.RESULTS_DIR / "pilot_posthoc.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console summary
    print("=== POST-HOC EXPLORATORY (not pre-registered) ===\n")
    print("A. Per-query split (clusters/1k-input-tok, agglomerative):")
    for q, d in out["per_query"].items():
        cc = d["clusters_per_1k_input_tok_agg"]
        print(f"  {q:<18} C1={cc['C1']:.3f}  C2={cc['C2']:.3f}  C7={cc['C7-proxy']:.3f}"
              f"  | C7 last? {d['C7_is_last_tokens_matched']}  | {d['verdict']}")
    print("\nB. Budget-unit robustness (C7-proxy vs C1, agglomerative):")
    for unit, e in out["budget_unit_robustness"]["contrasts"]["C7-proxy_vs_C1"].items():
        p, bt = e["point"], e["boot"]
        print(f"  {unit:<18} budget_gap={p['budget_ratio_a_over_b']:.2f}x  "
              f"cluster_gain={p['cluster_ratio_a_over_b']:.2f}x  "
              f"diff={bt['mean_diff']:.3f} CI[{bt['ci_low']:.3f},{bt['ci_high']:.3f}]")
    print("\nC. C2-vs-C1 (the positive finding):")
    for unit in ("proposals_matched", "tokens_matched_per_1k"):
        bt = out["C2_vs_C1"][unit]
        print(f"  {unit:<22} diff={bt['mean_diff']:.3f} "
              f"CI[{bt['ci_low']:.3f},{bt['ci_high']:.3f}]")
    print(f"\n[posthoc] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
