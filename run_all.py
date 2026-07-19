"""One-command replication for the Grounding Doesn't Pay pilot.

Default (light, stdlib + matplotlib): verify the integrity chain, check every
paper number against the frozen data, regenerate the figures. This confirms the
published artifact is internally consistent and untampered.

    python run_all.py

Opt-in (heavy, needs sentence-transformers + hdbscan + sklearn): re-embed and
re-cluster the frozen generation to reproduce pilot_scores.json and the encoder
robustness re-scores. This is the reproducible-derivation seal in action.

    python run_all.py --with-rescore

Generation itself is NOT re-runnable here: it is frozen evidence from the
Grok-4.20 API (see PROVENANCE.md). No new API calls are ever made.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(rel: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / rel), *args]
    print(f"\n=== {rel} {' '.join(args)}".rstrip())
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replicate the pilot from the frozen data.")
    ap.add_argument("--with-rescore", action="store_true",
                    help="also re-embed/re-cluster (heavy deps) to reproduce the scores")
    args = ap.parse_args()

    run("scripts/verify_chain.py")          # integrity: nothing was altered
    run("tests/test_paper_numbers.py")      # every headline number ties to the data
    run("scripts/make_figures.py")          # regenerate both figures from the scores

    if args.with_rescore:
        run("code/score_pilot.py", "--in", "data/pilot_gen_full.jsonl")
        run("scripts/rescore_robustness.py")

    print("\nAll replication steps passed. Generation is frozen; only derivation was re-run.")


if __name__ == "__main__":
    main()
