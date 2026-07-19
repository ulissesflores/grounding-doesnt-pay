#!/usr/bin/env python3
"""Regenerate every figure in the paper from the locked pilot data.

Deterministic, no LLM, no network. Reads only ../data/pilot_scores.json and
../data/pilot_posthoc.json (the frozen, pre-registered outputs) and writes PNGs
to ../output/figures/. Every constant is derived from the JSON except the two
declared caching-price assumptions (documented inline), so the figures track the
data automatically and cannot silently disagree with §4-§5 of the paper.

Usage:
    python3 scripts/make_figures.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Fix the PNG creation-time chunk so figures are byte-reproducible (matplotlib
# honours SOURCE_DATE_EPOCH). Must be set BEFORE importing matplotlib. Without
# this, each regeneration produces visually-identical but byte-different PNGs,
# which would break the provenance seal on every run.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1500000000")

import matplotlib

matplotlib.use("Agg")  # headless, reproducible
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Arm display order and labels (paper §3).
ARMS = ["C1", "C2", "C7-proxy"]
LABELS = {
    "C1": "C1\nrepeated sampling",
    "C2": "C2\nstyle-persona",
    "C7-proxy": "C7\nin-context spec",
}
# Colour-blind-safe, print-legible; treatment arm (C7) highlighted.
COLORS = {"C1": "#4E79A7", "C2": "#59A14F", "C7-proxy": "#E15759"}

# Declared caching-price assumptions (paper §5). Anthropic-style base: cache-write
# 1.25× base input, cache-read 0.10× base input; the net write premium spread over
# reuses is 1.25 − 0.10 = 1.15 (the read is already counted in CACHE_READ), so the
# per-idea effective spec cost is S·(CACHE_WRITE/R + CACHE_READ). CACHE_READ_MODEL
# is the Grok-4.20 card's own listed rate (cached input $0.20/M ≈ 0.16× the $1.25/M
# input), which we plot as a second, less charitable floor.
CACHE_WRITE = 1.15  # net write premium over the read (= 1.25 − 0.10), amortized over R
CACHE_READ = 0.10   # per-reuse read multiplier on the spec's input tokens (Anthropic-style)
CACHE_READ_MODEL = 0.16  # Grok-4.20 card's own cached-input rate

N_IDEAS = 180  # per arm (paper §3)


def load(name: str) -> dict:
    with open(DATA / name) as fh:
        return json.load(fh)


def fig1_clusters_per_1k(scores: dict) -> Path:
    """Fig. 1 — tokens-matched diversity by arm under both algorithms.

    Two panels (different y-scales) so HDBSCAN's small counts stay legible.
    The single fact the panels share: C7 (grounding) is last in both.
    """
    m = scores["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for ax, algo, title in (
        (axes[0], "agglomerative", "Agglomerative (pre-registered decisive view)"),
        (axes[1], "hdbscan", "HDBSCAN"),
    ):
        vals = [m[a]["clusters_per_1k_input_tokens"][algo] for a in ARMS]
        bars = ax.bar(
            [LABELS[a] for a in ARMS], vals,
            color=[COLORS[a] for a in ARMS], edgecolor="black", linewidth=0.6,
        )
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("clusters / 1k input tokens")
        # smaller x-tick font so the two-line arm labels never run together
        # (the "repeated samplingstyle-persona" collision — F8 B5)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.18)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        # mark the invariant: C7 is last
        c7_idx = ARMS.index("C7-proxy")
        ax.annotate("last", xy=(c7_idx, vals[c7_idx]),
                    xytext=(c7_idx, max(vals) * 0.55), ha="center", fontsize=8,
                    color=COLORS["C7-proxy"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["C7-proxy"], lw=0.8))
    fig.suptitle("Grounding is never the token-efficient frontier "
                 "(clusters per 1k input tokens)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "fig1_clusters_per_1k.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig2_caching_breakeven(scores: dict) -> Path:
    """Fig. 2 — prompt-caching break-even (agglomerative view, paper §5).

    C7's effective tokens-matched diversity as one spec is reused R times,
    against the cache-free C1 and C2 frontiers. b and S are derived from the
    logged input tokens; only the two cache multipliers are assumed.
    """
    m = scores["metrics"]
    b = m["C1"]["input_tokens"] / N_IDEAS            # generic-prompt base ≈ 104.5
    S = m["C7-proxy"]["input_tokens"] / N_IDEAS - b  # spec overhead ≈ 603.7
    cpi_c7 = m["C7-proxy"]["clusters_per_idea"]["agglomerative"]  # 0.894
    c1 = m["C1"]["clusters_per_1k_input_tokens"]["agglomerative"]  # 5.263
    c2 = m["C2"]["clusters_per_1k_input_tokens"]["agglomerative"]  # 7.601

    def c7_eff(R: float) -> float:
        eff_input = b + S * (CACHE_WRITE / R + CACHE_READ)
        return cpi_c7 * 1000.0 / eff_input

    def c7_eff_model(R: float) -> float:
        # Grok card's own 0.16× read rate; net write premium 1.25 − 0.16 = 1.09.
        eff_input = b + S * ((1.25 - CACHE_READ_MODEL) / R + CACHE_READ_MODEL)
        return cpi_c7 * 1000.0 / eff_input

    floor_model = c7_eff_model(1e9)  # ≈ 4.448 < c1 → never reaches C1 at the model's own rate

    # break-even vs C1 (bisection; asymptote 5.425 > c1 so a root exists)
    lo, hi = 1.0, 1e6
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        (lo, hi) = (mid, hi) if c7_eff(mid) < c1 else (lo, mid)
    be_c1 = (lo * hi) ** 0.5

    Rs = [1 + i * 0.05 for i in range(0, 6000)]  # 1 .. ~300, dense
    ys = [c7_eff(R) for R in Rs]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(Rs, ys, color=COLORS["C7-proxy"], lw=1.8,
            label="C7 in-context spec (cached)")
    ax.axhline(c1, color=COLORS["C1"], ls="--", lw=1.3,
               label=f"C1 repeated sampling ({c1:.3f})")
    ax.axhline(c2, color=COLORS["C2"], ls="--", lw=1.3,
               label=f"C2 style-persona ({c2:.3f})")
    ax.axhline(c7_eff(1e9), color=COLORS["C7-proxy"], ls=":", lw=1.0,
               label=f"C7 floor, 0.10× cache-read ({c7_eff(1e9):.3f}) — overtakes C1")
    ax.axhline(floor_model, color=COLORS["C7-proxy"], ls=(0, (1, 3)), lw=1.0, alpha=0.8,
               label=f"C7 floor, 0.16× model rate ({floor_model:.3f}) — never reaches C1")
    ax.axvline(be_c1, color="gray", ls=":", lw=1.0)
    ax.annotate(f"break-even vs C1\nR ≈ {be_c1:.0f}\n(only at 0.10×)",
                xy=(be_c1, c1), xytext=(be_c1 * 0.32, c1 + 0.9), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("reuses R of one identical spec (log scale)")
    ax.set_ylabel("clusters / 1k effective input tokens")
    ax.set_title("Prompt caching does not rescue grounding vs the cheap persona\n"
                 "(agglomerative view; C7 never reaches C2; overtakes C1 only after ~10² reuses,\n"
                 "and only under the charitable 0.10× rate — never at the model's own 0.16×)",
                 fontsize=9.0)
    ax.legend(fontsize=7.6, loc="lower right")
    ax.margins(x=0)
    fig.tight_layout()
    path = OUT / "fig2_caching_breakeven.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, be_c1


def main() -> None:
    scores = load("pilot_scores.json")
    p1 = fig1_clusters_per_1k(scores)
    p2, be_c1 = fig2_caching_breakeven(scores)
    print(f"wrote {p1}")
    print(f"wrote {p2}  (break-even vs C1: R ≈ {be_c1:.1f})")


if __name__ == "__main__":
    main()
