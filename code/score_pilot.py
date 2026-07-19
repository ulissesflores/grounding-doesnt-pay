"""Pre-registered scoring for the Grounding Doesn't Pay pilot (100% local, no LLM).

Pipeline (all pre-registered in pilot_config.json):
  1. Extract atomic ideas (1 proposition/line), exact dedup per arm.
  2. Embed domain-blind with sentence-transformers/all-MiniLM-L6-v2
     (TF-IDF fallback if the model cannot be loaded — documented in output).
  3. Cluster each arm with TWO algorithms: HDBSCAN (min_cluster_size=3) and
     Agglomerative (cosine, distance_threshold=0.35). Validity = arm ordering
     invariant across the two.
  4. Primary DV: clusters / idea, in both budget views:
        - proposals-matched  : clusters / n_ideas
        - tokens-matched     : clusters / (input_tokens / 1000)   (the decisive one)
  5. Secondaries: raw cluster count, mean pairwise cosine distance, fluency.
  6. rho: mean pairwise correlation of cluster-coverage between source
     identities, per arm.
  7. Bootstrap (10k) CIs on the between-arm difference of the primary DV.
  8. Read the result against the pre-registered gate rule.

Usage
-----
    # reproduce the pre-registered scores from the frozen generation:
    python code/score_pilot.py --in data/pilot_gen_full.jsonl

    # smoke-test the scoring pipeline on ~30 mock ideas (no cost, no LLM):
    python code/score_pilot.py --mock
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc

# ─── Atomic extraction ───────────────────────────────────────────────────────

_MARKER_RE = re.compile(r"^\s*(?:\d+[.)\-]|[-*•])\s+(.*)$")


def extract_ideas(text: str, min_chars: int = 3) -> list[str]:
    """Extract atomic ideas: one proposition per marked line."""
    ideas: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _MARKER_RE.match(s)
        if m:
            ideas.append(m.group(1).strip())
    if not ideas:  # fallback: model ignored numbering
        ideas = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    return [i for i in ideas if len(i) >= min_chars]


# ─── Domain-blind cleaning ───────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")


def build_stopset(records: list[dict], cfg: dict) -> set[str]:
    """Seeded vocabulary to strip before embedding (so we don't measure it)."""
    db = cfg["scoring"]["embedding"]["domain_blind"]
    stop = {w.lower() for w in db.get("extra_stopwords", [])}
    if db.get("strip_source_vocabulary", True):
        for r in records:
            for tok in _WORD_RE.findall(r.get("source_identity", "").lower()):
                if len(tok) > 2:
                    stop.add(tok)
    return stop


def domain_blind(text: str, stopset: set[str], lowercase: bool = True) -> str:
    t = text.lower() if lowercase else text
    return " ".join(w for w in _WORD_RE.findall(t) if w not in stopset)


# ─── Embedding backend (with fallback) ───────────────────────────────────────


def _load_st_model(name: str, timeout: float):
    """Load a sentence-transformers model in a thread, bounded by timeout."""
    box: dict[str, Any] = {}

    def _load() -> None:
        try:
            from sentence_transformers import SentenceTransformer

            box["model"] = SentenceTransformer(name)
        except Exception as e:
            box["error"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=_load, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return None, f"timeout after {timeout}s"
    if "error" in box:
        return None, box["error"]
    return box.get("model"), None


def embed_corpus(texts: list[str], cfg: dict) -> tuple[np.ndarray, str, str]:
    """Embed all texts (L2-normalized). Returns (matrix, backend, detail)."""
    emb = cfg["scoring"]["embedding"]
    model, err = _load_st_model(emb["primary_model"], emb.get("load_timeout_seconds", 120))
    if model is not None:
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float64), "sentence-transformers", emb["primary_model"]

    # Fallback: TF-IDF (sklearn), rows L2-normalized by default (norm="l2").
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(min_df=1)
    tfidf = vec.fit_transform(texts)
    detail = f"tfidf (MiniLM unavailable: {err}); vocab={len(vec.vocabulary_)}"
    return np.asarray(tfidf.toarray(), dtype=np.float64), "tfidf-sklearn", detail


def cosine_distance_matrix(mat: np.ndarray) -> np.ndarray:
    """Cosine distance among L2-normalized rows; clipped to [0, 2]."""
    sim = mat @ mat.T
    dist = 1.0 - sim
    np.clip(dist, 0.0, 2.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return (dist + dist.T) / 2.0  # enforce symmetry


# ─── Clustering ──────────────────────────────────────────────────────────────


def cluster_arm(dist: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Cluster one arm with HDBSCAN and Agglomerative on a precomputed matrix."""
    from sklearn.cluster import AgglomerativeClustering

    cl = cfg["scoring"]["clustering"]
    n = dist.shape[0]
    out: dict[str, np.ndarray] = {}

    # HDBSCAN (native in sklearn >= 1.3)
    mcs = cl["hdbscan_min_cluster_size"]
    if n >= mcs:
        from sklearn.cluster import HDBSCAN

        hdb = HDBSCAN(min_cluster_size=mcs, metric="precomputed")
        out["hdbscan"] = hdb.fit_predict(dist)
    else:
        out["hdbscan"] = np.full(n, -1, dtype=int)  # too few points -> all noise

    # Agglomerative (cosine distance, average linkage, fixed threshold)
    if n >= 2:
        agg = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=cl["agglomerative_linkage"],
            distance_threshold=cl["agglomerative_distance_threshold"],
        )
        out["agglomerative"] = agg.fit_predict(dist)
    else:
        out["agglomerative"] = np.zeros(n, dtype=int)
    return out


def n_distinct(labels: np.ndarray) -> int:
    return len({int(x) for x in labels if int(x) != -1})


# ─── rho (source-identity decorrelation) ─────────────────────────────────────


def compute_rho(labels: np.ndarray, identities: list[str]) -> float | None:
    """Mean pairwise correlation of cluster-coverage vectors across identities.

    Low rho = identities cover DIFFERENT clusters (decorrelation working).
    Returns None when fewer than 2 distinct identities carry ideas (e.g. C1).
    """
    uniq_ids = sorted(set(identities))
    if len(uniq_ids) < 2:
        return None
    clusters = sorted({int(x) for x in labels if int(x) != -1})
    if not clusters:
        return None
    idx = {c: j for j, c in enumerate(clusters)}
    mat = np.zeros((len(uniq_ids), len(clusters)))
    id_row = {i: r for r, i in enumerate(uniq_ids)}
    for lab, ident in zip(labels, identities, strict=True):
        if int(lab) in idx:
            mat[id_row[ident], idx[int(lab)]] += 1
    corrs: list[float] = []
    for a in range(len(uniq_ids)):
        for b in range(a + 1, len(uniq_ids)):
            va, vb = mat[a], mat[b]
            if va.std() == 0 or vb.std() == 0:
                continue  # constant vector -> correlation undefined
            corrs.append(float(np.corrcoef(va, vb)[0, 1]))
    if not corrs:
        return None
    return float(np.mean(corrs))


# ─── Bootstrap ───────────────────────────────────────────────────────────────


def boot_distinct(labels: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the distinct-cluster count (resample ideas)."""
    n = len(labels)
    if n == 0:
        return np.zeros(resamples)
    out = np.empty(resamples)
    for i in range(resamples):
        sample = labels[rng.integers(0, n, n)]
        out[i] = n_distinct(sample)
    return out


def ci(arr: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo = float(np.percentile(arr, (1 - level) / 2 * 100))
    hi = float(np.percentile(arr, (1 + level) / 2 * 100))
    return lo, hi


# ─── Core scoring ────────────────────────────────────────────────────────────


def score(records: list[dict], cfg: dict) -> dict:
    """Run the full scoring pipeline over generation records."""
    arms_order = cfg["generation"]["arms"]
    min_chars = cfg["scoring"]["atomic_extraction"]["min_chars"]
    exact_dedup = cfg["scoring"]["atomic_extraction"]["exact_dedup"]

    # 1. Extract + dedup atomic ideas per arm; keep source identity + tokens.
    per_arm: dict[str, dict[str, Any]] = {}
    stopset = build_stopset(records, cfg)
    db = cfg["scoring"]["embedding"]["domain_blind"]

    for arm in arms_order:
        ideas: list[str] = []
        idents: list[str] = []
        seen: set[str] = set()
        input_tokens = 0
        raw_ideas = 0
        for r in (x for x in records if x["arm"] == arm):
            input_tokens += int(r.get("input_tokens", 0))
            for idea in extract_ideas(r.get("output_text", ""), min_chars):
                raw_ideas += 1
                key = idea.lower().strip()
                if exact_dedup and key in seen:
                    continue
                seen.add(key)
                ideas.append(idea)
                idents.append(r.get("source_identity", "generic"))
        per_arm[arm] = {
            "ideas": ideas,
            "identities": idents,
            "input_tokens": input_tokens,
            "raw_ideas": raw_ideas,
            "n_ideas": len(ideas),
        }

    # 2. Embed the pooled corpus (shared space), domain-blind.
    all_ideas: list[str] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for arm in arms_order:
        n = per_arm[arm]["n_ideas"]
        slices[arm] = slice(cursor, cursor + n)
        all_ideas.extend(per_arm[arm]["ideas"])
        cursor += n

    cleaned = [
        domain_blind(t, stopset, db.get("lowercase", True)) or t.lower()
        for t in all_ideas
    ]
    if all_ideas:
        emb_all, backend, backend_detail = embed_corpus(cleaned, cfg)
    else:
        emb_all, backend, backend_detail = np.zeros((0, 1)), "none", "no ideas"

    # 3-6. Per-arm clustering + metrics.
    boot_seed = cfg["scoring"]["bootstrap"]["seed"]
    resamples = cfg["scoring"]["bootstrap"]["resamples"]
    rng = np.random.default_rng(boot_seed)

    metrics: dict[str, dict[str, Any]] = {}
    boot_distinct_cache: dict[str, dict[str, np.ndarray]] = {}

    for arm in arms_order:
        info = per_arm[arm]
        emb_arm = emb_all[slices[arm]] if info["n_ideas"] else np.zeros((0, 1))
        n_ideas = info["n_ideas"]
        toks_1k = info["input_tokens"] / 1000.0

        if n_ideas >= 1:
            dist = cosine_distance_matrix(emb_arm)
            labels = cluster_arm(dist, cfg)
            iu = np.triu_indices(n_ideas, k=1)
            mean_pw = float(dist[iu].mean()) if len(iu[0]) else 0.0
        else:
            dist = np.zeros((0, 0))
            labels = {"hdbscan": np.zeros(0, int), "agglomerative": np.zeros(0, int)}
            mean_pw = 0.0

        nc_h = n_distinct(labels["hdbscan"])
        nc_a = n_distinct(labels["agglomerative"])

        def dv(nc: int, n_ideas: int = n_ideas, toks_1k: float = toks_1k) -> dict[str, float | None]:
            return {
                "proposals_matched": (nc / n_ideas) if n_ideas else None,
                "tokens_matched": (nc / toks_1k) if toks_1k else None,
            }

        metrics[arm] = {
            "n_ideas": n_ideas,
            "raw_ideas": info["raw_ideas"],
            "input_tokens": info["input_tokens"],
            "n_source_identities": len(set(info["identities"])),
            "clusters": {"hdbscan": nc_h, "agglomerative": nc_a},
            "clusters_per_idea": {
                "hdbscan": dv(nc_h)["proposals_matched"],
                "agglomerative": dv(nc_a)["proposals_matched"],
            },
            "clusters_per_1k_input_tokens": {
                "hdbscan": dv(nc_h)["tokens_matched"],
                "agglomerative": dv(nc_a)["tokens_matched"],
            },
            "mean_pairwise_cosine_distance": mean_pw,
            "fluency_n_ideas": n_ideas,
            "rho": compute_rho(labels["agglomerative"], info["identities"]),
        }

        boot_distinct_cache[arm] = {
            algo: boot_distinct(labels[algo], resamples, rng)
            for algo in ("hdbscan", "agglomerative")
        }

    # 7. Bootstrap CIs on between-arm differences of the primary DV.
    contrasts = cfg["scoring"]["bootstrap"]["contrasts"]
    ci_level = cfg["scoring"]["bootstrap"]["ci"]
    bootstrap_out: dict[str, Any] = {}
    for a, b in contrasts:
        name = f"{a}_vs_{b}"
        bootstrap_out[name] = {}
        for algo in ("hdbscan", "agglomerative"):
            da, db_ = boot_distinct_cache[a][algo], boot_distinct_cache[b][algo]
            na, nb = per_arm[a]["n_ideas"], per_arm[b]["n_ideas"]
            ta = per_arm[a]["input_tokens"] / 1000.0
            tb = per_arm[b]["input_tokens"] / 1000.0
            # proposals-matched diff
            if na and nb:
                diff_prop = da / na - db_ / nb
                p_mean, (p_lo, p_hi) = float(diff_prop.mean()), ci(diff_prop, ci_level)
            else:
                p_mean = p_lo = p_hi = None
            # tokens-matched diff (the decisive view)
            if ta and tb:
                diff_tok = da / ta - db_ / tb
                t_mean, (t_lo, t_hi) = float(diff_tok.mean()), ci(diff_tok, ci_level)
            else:
                t_mean = t_lo = t_hi = None
            bootstrap_out[name][algo] = {
                "proposals_matched": {"mean_diff": p_mean, "ci_low": p_lo, "ci_high": p_hi},
                "tokens_matched": {"mean_diff": t_mean, "ci_low": t_lo, "ci_high": t_hi},
            }

    # Ordering invariance across the two algorithms (proposals view).
    def order(view_algo: str) -> list[str]:
        return sorted(
            arms_order,
            key=lambda a: (metrics[a]["clusters_per_idea"][view_algo] or -1),
            reverse=True,
        )

    order_hdb = order("hdbscan")
    order_agg = order("agglomerative")
    ordering_invariant = order_hdb == order_agg

    gate = read_gate(metrics, bootstrap_out, ordering_invariant, cfg)

    return {
        "meta": {
            "scored_at": datetime.now(UTC).isoformat(),
            "n_records": len(records),
            "embedding_backend": backend,
            "embedding_detail": backend_detail,
            "domain_blind_stopwords": sorted(stopset),
            "clustering": cfg["scoring"]["clustering"],
            "bootstrap_resamples": resamples,
        },
        "arms": arms_order,
        "metrics": metrics,
        "ordering": {
            "hdbscan": order_hdb,
            "agglomerative": order_agg,
            "invariant": ordering_invariant,
        },
        "bootstrap_diff": bootstrap_out,
        "gate": gate,
    }


# ─── Gate reading ────────────────────────────────────────────────────────────


def read_gate(metrics, bootstrap, ordering_invariant, cfg) -> dict:
    """Apply the pre-registered decision rule using the tokens-matched view."""
    # Display algorithm for the gate reading. The pre-registration names the
    # tokens-matched BUDGET view as decisive and pre-registers TWO algorithms
    # symmetrically (validity = ordering invariance across them); it does NOT
    # name an algorithm as primary. Agglomerative is a post-collection display
    # choice (HDBSCAN's counts are near-degenerate at this n); HDBSCAN is
    # reported alongside. See PREREGISTRATION.md and paper §4.4.
    algo = "agglomerative"
    c7_c1 = bootstrap.get("C7-proxy_vs_C1", {}).get(algo, {}).get("tokens_matched", {})
    c7_c2 = bootstrap.get("C7-proxy_vs_C2", {}).get(algo, {}).get("tokens_matched", {})

    def beats(d: dict) -> bool | None:
        if d.get("ci_low") is None:
            return None
        return d["ci_low"] > 0

    def approx(d: dict) -> bool | None:
        if d.get("ci_low") is None:
            return None
        return d["ci_low"] <= 0 <= d["ci_high"]

    c7_beats_c1 = beats(c7_c1)
    c7_beats_c2 = beats(c7_c2)
    c7_approx_c2 = approx(c7_c2)

    if c7_beats_c1 is None:
        decision = "INCONCLUSIVE"
        reading = "Insufficient data to evaluate the tokens-matched contrast."
    elif not c7_beats_c1:
        decision = "AMPUTATE"
        # Distinguish a strict loss (CI entirely < 0) from a statistical tie
        # (CI includes 0). Both satisfy "C7 does NOT beat C1" -> AMPUTATE, but
        # the wording must not misreport a strict loss as a tie.
        ci_hi = c7_c1.get("ci_high")
        if ci_hi is not None and ci_hi < 0:
            ci_phrase = (
                "bootstrap CI of the difference lies entirely below 0 -> C7-proxy "
                "is strictly WORSE than C1"
            )
        else:
            ci_phrase = (
                "bootstrap CI of the difference includes 0 -> C7-proxy is "
                "statistically tied with C1 (does not beat it)"
            )
        reading = (
            f"C7-proxy does NOT beat C1 on tokens-matched ({ci_phrase}). "
            "Structural coupling confirmed -> amputation validated: DO NOT "
            "build the Cupula; report the null."
        )
    elif c7_beats_c1 and c7_beats_c2:
        decision = "PROCEED"
        reading = (
            "C7-proxy beats BOTH C1 and C2 on tokens-matched -> domain beats "
            "style AND scale -> proceed to the narrow ablation (k-curve, human "
            "anchor, OSF pre-registration)."
        )
    elif c7_beats_c1 and c7_approx_c2:
        decision = "PERSONA_NOT_DOMAIN"
        reading = (
            "C7-proxy beats C1 but is statistically ~= C2 -> the effect is "
            "persona/style, not grounded domain -> the domain-specific thesis "
            "falls (publish as a negative result)."
        )
    else:
        decision = "INCONCLUSIVE"
        reading = "C7-proxy beats C1 but the C7-vs-C2 contrast is ambiguous."

    return {
        "decision_view": "tokens_matched",
        "algorithm": algo,
        "ordering_invariant": ordering_invariant,
        "c7_vs_c1_tokens": c7_c1,
        "c7_vs_c2_tokens": c7_c2,
        "decision": decision,
        "reading": reading,
        "caveat": (
            "Validity requires ordering invariance across HDBSCAN and "
            "Agglomerative; if False, treat the reading as provisional."
        ),
    }


# ─── Report ──────────────────────────────────────────────────────────────────


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_report(res: dict, cfg: dict, out_path: Path, is_mock: bool) -> None:
    m = res["metrics"]
    arms = res["arms"]
    gate = res["gate"]
    lines: list[str] = []

    lines.append("# Pilot Scores (C1 / C2 / C7-proxy)")
    lines.append("")
    lines.append(f"> Pre-registration: `{cfg['preregistration']}`. "
                 f"Scored {res['meta']['scored_at']}.")
    if is_mock:
        lines.append(">")
        lines.append("> [!WARNING]")
        lines.append("> These numbers come from a **MOCK** idea set generated to prove the "
                     "scoring pipeline. They are NOT a real gate decision.")
    lines.append("")

    # TL;DR
    lines.append("## TL;DR")
    lines.append("")
    lines.append("| | |")
    lines.append("|--|--|")
    lines.append(f"| Embedding backend | `{res['meta']['embedding_backend']}` |")
    lines.append(f"| Clustering | HDBSCAN(min_cluster_size={cfg['scoring']['clustering']['hdbscan_min_cluster_size']}) "
                 f"+ Agglomerative(cosine, thr={cfg['scoring']['clustering']['agglomerative_distance_threshold']}) |")
    lines.append(f"| Ordering invariant across algos | {res['ordering']['invariant']} |")
    lines.append(f"| Bootstrap resamples | {res['meta']['bootstrap_resamples']:,} |")
    lines.append(f"| Gate decision | **{gate['decision']}** |")
    lines.append("")

    # Main table
    lines.append("## Primary metrics")
    lines.append("")
    header = ("| Arm | Ideas | Input tok | Clusters (H/A) | clusters/idea (H/A) "
              "| clusters/1k-tok (H/A) | mean cos-dist | rho |")
    sep = "|" + "---|" * 8
    lines.append(header)
    lines.append(sep)
    for arm in arms:
        a = m[arm]
        cl = a["clusters"]
        cpi = a["clusters_per_idea"]
        cpt = a["clusters_per_1k_input_tokens"]
        lines.append(
            f"| {arm} | {a['n_ideas']} | {a['input_tokens']:,} "
            f"| {cl['hdbscan']}/{cl['agglomerative']} "
            f"| {_fmt(cpi['hdbscan'])}/{_fmt(cpi['agglomerative'])} "
            f"| {_fmt(cpt['hdbscan'])}/{_fmt(cpt['agglomerative'])} "
            f"| {_fmt(a['mean_pairwise_cosine_distance'])} "
            f"| {_fmt(a['rho'])} |"
        )
    lines.append("")
    lines.append("H = HDBSCAN, A = Agglomerative. rho = mean pairwise cluster-coverage "
                 "correlation between source identities (low = decorrelation working; "
                 "n/a for C1's single distribution).")
    lines.append("")

    # Bootstrap contrasts
    lines.append("## Bootstrap contrasts (Agglomerative, 95% CI of the difference)")
    lines.append("")
    lines.append("| Contrast | View | mean diff | CI low | CI high | C7 wins? |")
    lines.append("|---|---|---|---|---|---|")
    for name, byalgo in res["bootstrap_diff"].items():
        d = byalgo["agglomerative"]
        for view in ("proposals_matched", "tokens_matched"):
            dd = d[view]
            wins = ("yes" if (dd["ci_low"] is not None and dd["ci_low"] > 0)
                    else ("no" if dd["ci_low"] is not None else "n/a"))
            lines.append(
                f"| {name} | {view} | {_fmt(dd['mean_diff'])} "
                f"| {_fmt(dd['ci_low'])} | {_fmt(dd['ci_high'])} | {wins} |"
            )
    lines.append("")

    # Gate
    lines.append("## Gate decision (pre-registered rule)")
    lines.append("")
    lines.append(f"> **{gate['decision']}** — {gate['reading']}")
    lines.append("")
    lines.append(f"- Decision view: `{gate['decision_view']}` | algorithm: `{gate['algorithm']}`")
    lines.append(f"- Ordering invariant across algorithms: `{gate['ordering_invariant']}`")
    lines.append(f"- {gate['caveat']}")
    lines.append("")
    lines.append("Pre-registered branches:")
    lines.append("")
    lines.append(f"- **AMPUTATE** — {cfg['gate_rule']['amputate']}")
    lines.append(f"- **PROCEED** — {cfg['gate_rule']['proceed']}")
    lines.append(f"- **PERSONA_NOT_DOMAIN** — {cfg['gate_rule']['persona_not_domain']}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Mock data ───────────────────────────────────────────────────────────────


def mock_records() -> list[dict]:
    """~30 mock ideas across 3 arms, shaped like real generation records.

    C7-proxy is deliberately more diverse (spread across topics) and carries
    larger input-token counts (the spec cost); C1 is the tightest; C2 in
    between — so the pipeline exercises every branch of the math.
    """
    def rec(arm, idx, ident, ideas, in_tok):
        body = "\n".join(f"{i+1}. {t}" for i, t in enumerate(ideas))
        return {
            "arm": arm,
            "query_id": "aut_brick",
            "call_index": idx,
            "source_identity": ident,
            "output_text": body,
            "input_tokens": in_tok,
            "output_tokens": 40,
        }

    recs: list[dict] = []
    # C1 — generic, clustered around a few obvious uses (low diversity)
    c1 = [
        ["use a brick as a doorstop", "use a brick to prop a door open", "hold a door with a brick"],
        ["use a brick as a paperweight", "weigh down papers with a brick", "keep papers flat with a brick"],
        ["build a small wall with bricks", "stack bricks into a wall", "make a low garden wall"],
    ]
    for i, ideas in enumerate(c1):
        recs.append(rec("C1", i, "generic", ideas, 150))
    # C2 — style personas, moderately diverse
    c2 = [
        (["a brick as abstract sculpture", "paint the brick as protest art", "brick as minimalist decor"], "style/o artista"),
        (["grind brick into red pigment", "crush brick for a running track", "brick dust as polishing grit"], "style/o cientista sem campo"),
        (["a brick as a status symbol paperweight", "sell the brick as art", "brick as a conversation piece"], "style/o provocador"),
    ]
    for i, (ideas, ident) in enumerate(c2):
        recs.append(rec("C2", i, ident, ideas, 200))
    # C7-proxy — grounded domains, high topical spread + high input tokens
    c7 = [
        (["brick as thermal mass in a passive-solar wall", "brick as trombe-wall heat store", "brick lattice for night cooling"], "dominio-01/humano/THEOROS"),
        (["crushed brick as pozzolanic aggregate", "recycled brick fines in low-carbon mortar", "brick rubble as permeable sub-base"], "dominio-07/humano/INCISUS"),
        (["brick as acoustic diffuser on a facade", "perforated brick as a resonant absorber", "brick baffle for reverberation control"], "dominio-08/humano/MUTATOR"),
        (["brick planter for green-wall irrigation", "brick as a mycelium-growth substrate", "porous brick as a rainwater wick"], "dominio-06/biomimetico/VATES"),
        (["brick as ballast for a floating structure", "brick as a mold for concrete casting", "brick as riprap against scour"], "dominio-03/humano/SILENUS"),
    ]
    for i, (ideas, ident) in enumerate(c7):
        recs.append(rec("C7-proxy", i, ident, ideas, 2100))
    return recs


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-registered scoring (local, no LLM)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="infile", help="generation JSONL to score")
    g.add_argument("--mock", action="store_true", help="score a ~30-idea mock set")
    ap.add_argument("--config", default=str(pc.CONFIG_PATH))
    ap.add_argument("--out-scores", default=str(pc.RESULTS_DIR / "pilot_scores.json"))
    ap.add_argument("--out-report", default=str(pc.RESULTS_DIR / "pilot_report.md"))
    args = ap.parse_args()

    cfg = pc.load_config(args.config)

    if args.mock:
        records = mock_records()
        out_scores = pc.RESULTS_DIR / "pilot_scores_mock.json"
        out_report = pc.RESULTS_DIR / "pilot_report_mock.md"
    else:
        path = Path(args.infile)
        if not path.is_absolute() and not path.exists():
            path = pc.RESULTS_DIR / path.name  # fallback to the legacy results dir
        records = pc.read_jsonl(path)
        out_scores = Path(args.out_scores)
        out_report = Path(args.out_report)

    if not records:
        print("[score_pilot] no records to score.", file=sys.stderr)
        return 1

    res = score(records, cfg)
    out_scores.parent.mkdir(parents=True, exist_ok=True)
    out_scores.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(res, cfg, out_report, is_mock=args.mock)

    # Console summary
    print(f"[score_pilot] backend={res['meta']['embedding_backend']} "
          f"({res['meta']['embedding_detail']})")
    print(f"[score_pilot] records={len(records)} "
          f"ordering_invariant={res['ordering']['invariant']}")
    print("\n  arm         ideas  in_tok   clus(H/A)  cpi(A)   c/1k-tok(A)  rho")
    for arm in res["arms"]:
        a = res["metrics"][arm]
        print(f"  {arm:<11} {a['n_ideas']:>5}  {a['input_tokens']:>6}  "
              f"{a['clusters']['hdbscan']:>2}/{a['clusters']['agglomerative']:<2}    "
              f"{_fmt(a['clusters_per_idea']['agglomerative']):>6}   "
              f"{_fmt(a['clusters_per_1k_input_tokens']['agglomerative']):>8}   "
              f"{_fmt(a['rho'])}")
    print(f"\n[score_pilot] GATE: {res['gate']['decision']}")
    print(f"  {res['gate']['reading']}")
    print(f"\n[score_pilot] wrote {out_scores}")
    print(f"[score_pilot] wrote {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
