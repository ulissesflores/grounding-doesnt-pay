#!/usr/bin/env python3
"""Reproducibility guard: every number printed in the paper must equal the
frozen pilot data. Runs with the standard library only (no heavy deps), so
anyone can verify the paper<->data tie without loading the embedding model.

Each check pairs a value as *written in the paper* (docs/paper/paper-final.pdf)
with the path into data/pilot_scores.json or data/pilot_posthoc.json it must
match. A drift in either direction fails loudly. Run:

    python3 -m pytest tests/test_paper_numbers.py       # if pytest is present
    python3 tests/test_paper_numbers.py                 # plain, no pytest
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SCORES = json.loads((DATA / "pilot_scores.json").read_text())
POSTHOC = json.loads((DATA / "pilot_posthoc.json").read_text())

M = SCORES["metrics"]
BD = SCORES["bootstrap_diff"]
BUR = POSTHOC["budget_unit_robustness"]["contrasts"]
N_IDEAS = 180


def close(a: float, b: float, tol: float = 5e-3) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ── §4.1 main table (agglomerative) ──────────────────────────────────────────

def test_main_table_agglomerative():
    exp = {
        # arm: (input_tok, clusters_agg, clusters_per_idea_agg, per_1k_agg, cos, rho)
        "C1": (18810, 99, 0.550, 5.263, 0.606, None),
        "C2": (19076, 145, 0.806, 7.601, 0.593, -0.012),
        "C7-proxy": (127470, 161, 0.894, 1.263, 0.681, -0.014),
    }
    for arm, (itok, clu, cpi, per1k, cos, rho) in exp.items():
        md = M[arm]
        assert md["input_tokens"] == itok, arm
        assert md["clusters"]["agglomerative"] == clu, arm
        assert close(md["clusters_per_idea"]["agglomerative"], cpi), arm
        assert close(md["clusters_per_1k_input_tokens"]["agglomerative"], per1k), arm
        assert close(md["mean_pairwise_cosine_distance"], cos), arm
        if rho is not None:
            assert close(md["rho"], rho, tol=2e-2), arm


def test_ratios_678_and_163():
    # paper §4.1: 127,470 ≈ 6.8× 18,810 for only 1.63× the raw clusters (161 vs 99)
    assert close(M["C7-proxy"]["input_tokens"] / M["C1"]["input_tokens"], 6.78, tol=2e-3)
    assert close(161 / 99, 1.63, tol=3e-3)


# ── §4.2 the invariant ───────────────────────────────────────────────────────

def test_hdbscan_per_1k_order():
    h = {a: M[a]["clusters_per_1k_input_tokens"]["hdbscan"] for a in M}
    assert close(h["C1"], 0.585) and close(h["C2"], 0.105) and close(h["C7-proxy"], 0.071)
    assert h["C7-proxy"] == min(h.values())          # C7 last under HDBSCAN
    agg = {a: M[a]["clusters_per_1k_input_tokens"]["agglomerative"] for a in M}
    assert agg["C7-proxy"] == min(agg.values())      # C7 last under agglomerative


def test_c7_vs_c1_tokens_negative_both_algorithms():
    agg = BD["C7-proxy_vs_C1"]["agglomerative"]["tokens_matched"]
    hdb = BD["C7-proxy_vs_C1"]["hdbscan"]["tokens_matched"]
    assert close(agg["mean_diff"], -2.894) and agg["ci_high"] < 0
    assert close(hdb["mean_diff"], -0.511) and hdb["ci_high"] < 0
    assert close(agg["ci_low"], -3.286) and close(agg["ci_high"], -2.502)
    assert close(hdb["ci_low"], -0.522) and close(hdb["ci_high"], -0.461)


# ── §4.4 two-algorithm contrast table ────────────────────────────────────────

def test_section_4_4_contrast_table():
    a1 = BD["C7-proxy_vs_C1"]["agglomerative"]
    h1 = BD["C7-proxy_vs_C1"]["hdbscan"]
    a2 = BD["C7-proxy_vs_C2"]["agglomerative"]
    h2 = BD["C7-proxy_vs_C2"]["hdbscan"]
    # tokens-matched
    assert close(a2["tokens_matched"]["mean_diff"], -4.246)
    assert close(h2["tokens_matched"]["mean_diff"], -0.033)
    assert h2["tokens_matched"]["ci_low"] < 0 < h2["tokens_matched"]["ci_high"]  # crosses 0
    # proposals-matched: agglom C7 wins vs C1, HDBSCAN C7 loses vs C1
    assert close(a1["proposals_matched"]["mean_diff"], 0.194) and a1["proposals_matched"]["ci_low"] > 0
    assert close(h1["proposals_matched"]["mean_diff"], -0.012) and h1["proposals_matched"]["ci_high"] < 0
    # proposals-matched vs C2: agglom crosses 0, HDBSCAN C7 wins
    assert close(a2["proposals_matched"]["mean_diff"], 0.045)
    assert a2["proposals_matched"]["ci_low"] < 0 < a2["proposals_matched"]["ci_high"]
    assert close(h2["proposals_matched"]["mean_diff"], 0.038) and h2["proposals_matched"]["ci_low"] > 0
    # raw cluster counts row
    assert [M[a]["clusters"]["hdbscan"] for a in ("C1", "C2", "C7-proxy")] == [11, 2, 9]


def test_ordering_not_invariant():
    assert SCORES["ordering"]["invariant"] is False


# ── §4.5 robustness (budget units + per query) ───────────────────────────────

def test_budget_unit_robustness():
    c = BUR["C7-proxy_vs_C1"]
    assert close(c["per_1k_input_tok"]["point"]["budget_ratio_a_over_b"], 6.78, tol=2e-3)
    assert close(c["per_1k_total_tok"]["point"]["budget_ratio_a_over_b"], 5.24, tol=2e-3)
    assert close(c["per_usd"]["point"]["budget_ratio_a_over_b"], 4.35, tol=3e-3)
    assert close(c["per_1k_total_tok"]["boot"]["mean_diff"], -1.936)
    for u in ("per_1k_input_tok", "per_1k_total_tok", "per_usd"):
        assert c[u]["boot"]["ci_high"] < 0            # every unit's CI below zero


def test_per_query_c7_last():
    pq = POSTHOC["per_query"]
    ab = pq["aut_brick"]["clusters_per_1k_input_tok_agg"]
    wc = pq["water_compartment"]["clusters_per_1k_input_tok_agg"]
    assert close(ab["C1"], 6.33) and close(ab["C2"], 7.91) and close(ab["C7-proxy"], 1.25)
    assert close(wc["C1"], 4.39) and close(wc["C2"], 7.82) and close(wc["C7-proxy"], 1.35)
    assert ab["C7-proxy"] == min(ab.values()) and wc["C7-proxy"] == min(wc.values())


# ── §4.7 positive finding (C2 vs C1) ─────────────────────────────────────────

def test_c2_beats_c1_both_accountings():
    c = POSTHOC["C2_vs_C1"]
    assert close(c["proposals_matched"]["mean_diff"], 0.149) and c["proposals_matched"]["ci_low"] > 0
    assert close(c["tokens_matched_per_1k"]["mean_diff"], 1.350) and c["tokens_matched_per_1k"]["ci_low"] > 0


# ── §3 sampling counts / §5 caching derivation ───────────────────────────────

def test_source_identity_counts():
    assert M["C7-proxy"]["n_source_identities"] == 53
    assert M["C2"]["n_source_identities"] == 39


def test_caching_breakeven_and_floor():
    # §5 formula, constants derived from logged tokens (b, S) + declared prices.
    b = M["C1"]["input_tokens"] / N_IDEAS
    S = M["C7-proxy"]["input_tokens"] / N_IDEAS - b
    cpi = M["C7-proxy"]["clusters_per_idea"]["agglomerative"]
    c1 = M["C1"]["clusters_per_1k_input_tokens"]["agglomerative"]
    c2 = M["C2"]["clusters_per_1k_input_tokens"]["agglomerative"]

    def c7_eff(R):
        return cpi * 1000.0 / (b + S * (1.15 / R + 0.10))

    assert close(c7_eff(1.0), 1.041, tol=1e-2)         # no-cache row
    floor = c7_eff(1e9)
    assert close(floor, 5.425, tol=1e-2)               # perfect-cache floor
    assert floor < c2                                   # never reaches C2
    # break-even vs C1 ≈ 137 (bisection)
    lo, hi = 1.0, 1e6
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if c7_eff(mid) < c1 else (lo, mid)
    assert 120 <= (lo * hi) ** 0.5 <= 155              # order-of-magnitude ~137


# ── §8 cost + §3 per-call tokens (paper coverage, finding E5) ─────────────────

def test_cost_and_per_call_tokens():
    # §8 "total pilot cost ≈ $0.26" derived from the raw generation log.
    recs = [json.loads(ln) for ln in (DATA / "pilot_gen_full.jsonl").read_text().splitlines() if ln.strip()]
    total_cost = sum(float(r.get("cost_usd", 0.0)) for r in recs)
    assert close(total_cost, 0.26, tol=2e-2), total_cost          # $0.259 rounds to $0.26
    # §3/§5 "~2.1k tokens per call": C7's input tokens over its 60 calls.
    c7_calls = sum(1 for r in recs if r["arm"] == "C7-proxy")
    assert c7_calls == 60
    assert close(M["C7-proxy"]["input_tokens"] / c7_calls, 2124.5, tol=1e-3)


# ── §5 caching at the model's own 0.16× rate (finding A5) ─────────────────────

def test_caching_model_rate_floor_never_reaches_c1():
    b = M["C1"]["input_tokens"] / N_IDEAS
    S = M["C7-proxy"]["input_tokens"] / N_IDEAS - b
    cpi = M["C7-proxy"]["clusters_per_idea"]["agglomerative"]
    c1 = M["C1"]["clusters_per_1k_input_tokens"]["agglomerative"]

    def floor(read):  # R -> inf, net write premium = 1.25 - read
        return cpi * 1000.0 / (b + S * read)

    assert close(floor(0.10), 5.425, tol=1e-2) and floor(0.10) > c1   # 0.10× overtakes C1
    assert close(floor(0.16), 4.448, tol=1e-2) and floor(0.16) < c1   # 0.16× never reaches C1


# ── §4.5 encoder/stopword robustness re-scores (findings C4, C5) ──────────────

def _rescore(name: str) -> dict:
    return json.loads((DATA / "robustness" / f"rescore_{name}.json").read_text())


def test_robustness_baseline_reproduces_frozen():
    b = _rescore("baseline")["metrics"]
    for a in ("C1", "C2", "C7-proxy"):
        assert b[a]["clusters"]["agglomerative"] == M[a]["clusters"]["agglomerative"], a
        assert b[a]["clusters"]["hdbscan"] == M[a]["clusters"]["hdbscan"], a


def test_robustness_invariant_survives_multilingual():
    ml = _rescore("multilingual")
    m, bd = ml["metrics"], ml["bootstrap_diff"]
    def per1k(a, k):
        return m[a]["clusters_per_1k_input_tokens"][k]
    # C7 last on tokens-matched under BOTH algorithms (the invariant)
    assert per1k("C7-proxy", "agglomerative") == min(per1k(a, "agglomerative") for a in m)
    assert per1k("C7-proxy", "hdbscan") == min(per1k(a, "hdbscan") for a in m)
    # C7 loses to C1 tokens-matched (CI < 0); grounding still helps per proposal (CI > 0)
    tok = bd["C7-proxy_vs_C1"]["agglomerative"]["tokens_matched"]
    prop = bd["C7-proxy_vs_C1"]["agglomerative"]["proposals_matched"]
    assert tok["ci_high"] < 0 and close(tok["mean_diff"], -1.858, tol=2e-2)
    assert prop["ci_low"] > 0 and close(prop["mean_diff"], 0.260, tol=2e-2)
    # the encoder FLIP we refuse to lean on: C7 beats C2 per proposal under multilingual
    c2p = bd["C7-proxy_vs_C2"]["agglomerative"]["proposals_matched"]
    assert c2p["ci_low"] > 0 and close(c2p["mean_diff"], 0.104, tol=2e-2)


def test_robustness_minstop_leaves_c7_unchanged():
    ms = _rescore("minstop")["metrics"]
    assert ms["C7-proxy"]["clusters"]["agglomerative"] == 161   # stripping not load-bearing for C7
    # C7 still last on tokens-matched (the null does not rest on the stopword list)
    per1k = {a: ms[a]["clusters_per_1k_input_tokens"]["agglomerative"] for a in ms}
    assert per1k["C7-proxy"] == min(per1k.values())


# ── plain-python runner (no pytest needed) ───────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
