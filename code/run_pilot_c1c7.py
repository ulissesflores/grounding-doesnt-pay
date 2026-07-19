"""Pre-registered generation harness (arms C1 / C2 / C7-proxy) — FROZEN EVIDENCE.

This is the script that produced the 180 frozen records in
``data/pilot_gen_full.jsonl``. It is included as evidence of method and is **not
runnable from this package**: it depends on the private domain corpus and router
used to collect the data, and on a live OpenRouter key. No new generation is part
of this artifact (pre-registration LOCK); the shipped records are the data, and
``scripts/verify_chain.py`` proves they are exactly what was captured. To reproduce
the *scoring* from the frozen records, use ``code/score_pilot.py`` (see README).

It generated ideas for the three pre-registered arms under matched budget, using
OpenRouter ``x-ai/grok-4.20`` with reasoning disabled (non-reasoning). Every call
was INDEPENDENT (no shared history, no debate, no memory) and every record logs
input and output tokens and a SHA-256 of prompt+output. The OpenRouter API key was
read from the environment (``OPENROUTER_API_KEY``).

Arms
----
* C1  repeated sampling: one generic creative prompt, no persona / no domain.
* C2  style-persona: a generic style label (no knowledge substrate).
* C7-proxy  domain: the full real spec (domain x mode x persona) injected.

The script is deterministic given SEED (identity + call ordering). OpenRouter
does not accept a sampling seed in the router payload, so run-to-run token
sampling still varies — the pre-registration explicitly allows this and fixes
N and order instead.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _resolve_openrouter_key() -> tuple[str, str]:
    """Resolve a live OpenRouter key from the environment.

    Returns (key, source). Empty key means nothing was found. (This harness is
    frozen evidence and is not run from this package; the key path is kept minimal.)
    """
    val = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if val:
        return val, "env"
    return "", "none"

# Make the pilot package importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pilot_common as pc

# The project src/ must be importable; add it defensively.
sys.path.insert(0, str(pc.SPLIT_ROOT / "src"))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_call_plan(config: dict, mode: str) -> list[dict]:
    """Deterministically build the ordered list of calls to make.

    mode: "smoke" (2 calls/arm, first query only) or "full" (30 calls/arm/query
    over both queries).
    """
    gen = config["generation"]
    seed = config["seed"]
    arms = gen["arms"]
    queries_order = gen["queries_order"]
    queries = config["queries"]
    prompts = config["prompts"]

    if mode == "smoke":
        calls_per = gen["smoke_calls_per_arm"]
        active_queries = queries_order[:1]  # first query only
    else:
        calls_per = gen["calls_per_arm_per_query"]
        active_queries = queries_order

    style_pool = config["arms"]["C2"]["style_personas"]
    c7_identities = pc.discover_c7_identities(config)
    if not c7_identities:
        raise RuntimeError("No C7-proxy identities discovered — check arcabouco/ specs.")

    plan: list[dict] = []
    for query_id in active_queries:
        q = queries[query_id]
        user_prompt = prompts["user_template"].format(brief=q["brief"])
        q_index = queries_order.index(query_id)
        # Per-query deterministic RNG so arms sample reproducibly and distinctly.
        rng = random.Random(seed + q_index)

        # Pre-sample identities / labels for this query.
        c7_sample = rng.sample(c7_identities, min(calls_per, len(c7_identities)))
        n_labels = min(calls_per, len(style_pool))
        c2_sample = rng.sample(style_pool, n_labels)

        for arm in arms:
            for call_index in range(calls_per):
                if arm == "C1":
                    system_prompt = prompts["c1_system"]
                    source_identity = "generic"
                elif arm == "C2":
                    label = c2_sample[call_index % len(c2_sample)]
                    system_prompt = prompts["c2_system_template"].format(style=label)
                    source_identity = f"style/{label}"
                elif arm == "C7-proxy":
                    ident = c7_sample[call_index % len(c7_sample)]
                    system_prompt = prompts["c7_system_template"].format(
                        domain=ident["domain"],
                        mode=ident["mode"],
                        persona=ident["persona"],
                        spec=ident["spec_text"],
                    )
                    source_identity = ident["identity_tag"]
                else:
                    raise ValueError(f"Unknown arm: {arm}")

                call_id = f"{arm}|{query_id}|{call_index:02d}"
                plan.append(
                    {
                        "call_id": call_id,
                        "arm": arm,
                        "query_id": query_id,
                        "query_object": q["object"],
                        "query_brief": q["brief"],
                        "call_index": call_index,
                        "source_identity": source_identity,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    }
                )
    return plan


async def run(config: dict, mode: str, out_path: Path) -> dict:
    """Execute the plan, appending each record to the JSONL as it completes."""
    from parallax.config import settings
    from parallax.utils.llm_router import get_router

    gen = config["generation"]
    provider = gen["provider"]
    if provider == "openrouter":
        key, source = _resolve_openrouter_key()
        if not key:
            raise RuntimeError(
                "No OPENROUTER_API_KEY found in the environment — generation cannot run."
            )
        settings.openrouter_api_key = key
        print(f"[run_pilot] openrouter key source: {source}")
    elif provider == "cerebras":
        if not settings.cerebras_api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY is not set in the environment — generation cannot run."
            )
    else:
        raise RuntimeError(f"Unsupported generation provider: {provider}")

    plan = build_call_plan(config, mode)
    done = pc.existing_call_ids(out_path)  # resume support
    router = get_router()

    totals = {
        "calls": 0,
        "skipped": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "model_versions": set(),
    }

    print(
        f"[run_pilot] mode={mode} planned_calls={len(plan)} "
        f"already_done={len(done & {c['call_id'] for c in plan})} "
        f"out={out_path}"
    )

    for i, call in enumerate(plan, 1):
        if call["call_id"] in done:
            totals["skipped"] += 1
            continue
        t0 = time.time()
        resp = None
        last_err: Exception | None = None
        for attempt in range(1, 4):  # bounded retry — Grok occasionally stalls (ReadTimeout)
            try:
                resp = await router.call(
                    provider=gen["provider"],
                    model=gen["model"],
                    messages=[{"role": "user", "content": call["user_prompt"]}],
                    system=call["system_prompt"],
                    max_tokens=gen["max_tokens"],
                    temperature=gen["temperature"],
                    reasoning=gen.get("reasoning"),
                )
                break
            except Exception as e:  # noqa: BLE001 — transient provider stall, retry
                last_err = e
                print(
                    f"  [{i:>3}/{len(plan)}] {call['call_id']} attempt {attempt} "
                    f"failed: {type(e).__name__}; retrying"
                )
                await asyncio.sleep(2 * attempt)
        if resp is None:
            raise RuntimeError(
                f"call {call['call_id']} failed after 3 attempts: {last_err}"
            )
        elapsed = round(time.time() - t0, 3)

        record = {
            **{k: call[k] for k in (
                "call_id", "arm", "query_id", "query_object", "query_brief",
                "call_index", "source_identity", "system_prompt", "user_prompt",
            )},
            "provider": gen["provider"],
            "model": gen["model"],
            "model_api_id_config": gen["model_api_id"],
            "model_version_returned": resp.api_model or resp.model,
            "temperature": gen["temperature"],
            "max_tokens": gen["max_tokens"],
            "output_text": resp.text,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "elapsed_s": elapsed,
            "prompt_sha256": pc.sha256_text(call["system_prompt"], call["user_prompt"]),
            "output_sha256": pc.sha256_text(resp.text),
            "record_sha256": pc.sha256_text(
                call["system_prompt"], call["user_prompt"], resp.text
            ),
            "timestamp": _now_iso(),
        }
        pc.append_jsonl(out_path, record)

        totals["calls"] += 1
        totals["input_tokens"] += resp.input_tokens
        totals["output_tokens"] += resp.output_tokens
        totals["cost_usd"] += resp.cost_usd
        totals["model_versions"].add(resp.model)
        print(
            f"  [{i:>3}/{len(plan)}] {call['call_id']:<36} "
            f"in={resp.input_tokens:>5} out={resp.output_tokens:>4} "
            f"{elapsed:>5.2f}s"
        )

    await router.close()
    totals["model_versions"] = sorted(totals["model_versions"])
    return totals


def main() -> int:
    ap = argparse.ArgumentParser(description="Generation harness (frozen evidence)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="2 calls/arm, first query only")
    group.add_argument("--full", action="store_true", help="180 calls (pre-registered run)")
    ap.add_argument("--config", default=str(pc.CONFIG_PATH))
    args = ap.parse_args()

    config = pc.load_config(args.config)
    mode = "smoke" if args.smoke else "full"
    out_path = pc.RESULTS_DIR / config["artifacts"]["generation_jsonl"].format(
        mode=mode
    ).replace("results/", "")

    t0 = time.time()
    totals = asyncio.run(run(config, mode, out_path))
    wall = round(time.time() - t0, 1)

    print("\n[run_pilot] DONE")
    print(f"  new_calls        : {totals['calls']}")
    print(f"  skipped (resume) : {totals['skipped']}")
    print(f"  input_tokens     : {totals['input_tokens']:,}")
    print(f"  output_tokens    : {totals['output_tokens']:,}")
    print(f"  est_cost_usd     : ${totals['cost_usd']:.4f}")
    print(f"  model_versions   : {totals['model_versions']}")
    print(f"  wall_clock       : {wall}s")
    print(f"  jsonl            : {out_path}")
    if mode == "smoke":
        print("\n  --smoke proved the wiring. Run --full only after authorization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
