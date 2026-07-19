"""Verify the replication chain: recompute the Merkle tree + hash chain from the
on-disk files and confirm it matches the sealed PROVENANCE.json.

This is the executable proof — not trust. It fails loudly (exit != 0) if:
  * any sealed file's bytes changed (artifact_root mismatch), or
  * any generation record's content no longer matches its record_sha256.

The environment fingerprint is compared as CONTEXT: a difference is a WARNING,
not a failure, because a compatible-but-not-identical environment can reproduce
the derived scores (the artifact_root is deliberately environment-independent).

Usage: python scripts/verify_chain.py   # exit 0 = intact, exit 1 = tampered
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from build_chain import compute_provenance
from hash_utils import environment_fingerprint, sha256_hex

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sealed_path = ROOT / "PROVENANCE.json"
    if not sealed_path.exists():
        print("[FAIL] PROVENANCE.json missing — run scripts/build_chain.py first")
        return 1
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    now = compute_provenance()

    ok = True

    # 1) artifact_root — the portable tamper-evidence seal
    if now["artifact_root"] == sealed["artifact_root"]:
        print(f"[PASS] artifact_root matches ({now['artifact_root'][:16]}…)")
    else:
        ok = False
        print("[FAIL] artifact_root MISMATCH — a sealed file changed")
        print(f"       sealed:      {sealed['artifact_root']}")
        print(f"       recomputed:  {now['artifact_root']}")
        # localize the drift to a stage
        for s_now, s_old in zip(now["stages"], sealed["stages"], strict=False):
            if s_now["hash"] != s_old["hash"]:
                print(f"       -> stage '{s_now['name']}' differs")

    # 2) generation records re-tie to their content (hash <-> bytes)
    gen = next(s for s in now["stages"] if s["name"] == "generation")
    if gen["records_reverified"]:
        print(f"[PASS] generation: {gen['n_records']} records re-verified (hash ties to content)")
    else:
        ok = False
        print(f"[FAIL] generation: record hash mismatches: {gen['record_hash_mismatches'][:5]}")

    # 3) environment — context only, never a failure. artifact_root is
    # machine-independent by design, so a different interpreter/pins cannot
    # invalidate the seal; reproduction of the exact scores depends on the
    # declared pins (see PROVENANCE.md), which is a separate, documented claim.
    env_now = environment_fingerprint()
    env_now_hash = sha256_hex(json.dumps(env_now, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    sealed_by = sealed.get("sealed_by_interpreter", {})
    if env_now_hash == sealed_by.get("sha256"):
        print("[INFO] running under the same interpreter that signed the seal")
    else:
        print("[INFO] different interpreter than the one that signed the seal — expected;")
        print("       artifact_root is environment-independent, so the seal still holds.")

    print("\n" + ("=" * 56))
    print("CHAIN INTACT" if ok else "CHAIN VIOLATED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
