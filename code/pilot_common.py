"""Shared helpers for the pre-registered pilot harness.

Kept dependency-light on purpose: only stdlib here. Heavy scientific
dependencies (numpy / scikit-learn / sentence-transformers) live in
``score_pilot.py``; the OpenRouter generation path lives in
``run_pilot_c1c7.py``.

Everything in this file is deterministic and free of side effects beyond
reading the pre-registration config and the domain specs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# ─── Paths ───────────────────────────────────────────────────────────────────
# The generation harness (run_pilot_c1c7.py) resolved a private domain corpus via
# these roots; it is frozen evidence and not run from this package. The scorer only
# needs PILOT_DIR/RESULTS_DIR, and score_pilot.py accepts an explicit --in path.

PILOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PILOT_DIR / "results"
SPLIT_ROOT = PILOT_DIR.parent.parent
PROJECT_ROOT = SPLIT_ROOT.parent
ARCABOUCO_DIR = PROJECT_ROOT / "arcabouco"

CONFIG_PATH = PILOT_DIR / "pilot_config.json"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load the frozen pre-registration config."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Hashing ─────────────────────────────────────────────────────────────────


def sha256_text(*parts: str) -> str:
    """SHA-256 of the concatenation of the given text parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


# ─── Domain spec discovery + parsing ─────────────────────────────────────────

# Each specs file concatenates 5 persona blocks of the shape:
#   ---
#   id: dominio-NN-slug/persona-PERSONA/modo-MODE
#   ... frontmatter ...
#   ---
#   ## IDENTIDADE ... (body) ...
# The body runs until the next `---\nid: dominio` block or EOF.
_PERSONA_BLOCK_RE = re.compile(
    r"---\s*\n(id: dominio.*?)\n---\s*\n(.*?)(?=\n---\s*\nid: dominio|\Z)",
    re.DOTALL,
)
_FM_FIELD_RE = re.compile(r"^([a-z_]+):\s*(.+?)\s*$", re.MULTILINE)


def _parse_frontmatter(fm_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for m in _FM_FIELD_RE.finditer(fm_text):
        fields[m.group(1)] = m.group(2)
    return fields


def parse_specs_file(path: Path) -> list[dict[str, str]]:
    """Parse a specs-*.md file into its persona blocks.

    Returns a list of dicts with keys: id, domain, domain_n, mode, persona,
    body, full_text (frontmatter + body).
    """
    text = path.read_text(encoding="utf-8")
    blocks: list[dict[str, str]] = []
    for m in _PERSONA_BLOCK_RE.finditer(text):
        fm_text, body = m.group(1), m.group(2).strip()
        fm = _parse_frontmatter(fm_text)
        blocks.append(
            {
                "id": fm.get("id", ""),
                "domain": fm.get("domain", ""),
                "domain_n": fm.get("domain_n", ""),
                "mode": fm.get("mode", ""),
                "persona": fm.get("persona", ""),
                "body": body,
                "full_text": fm_text.strip() + "\n\n" + body,
            }
        )
    return blocks


def discover_c7_identities(config: dict[str, Any]) -> list[dict[str, str]]:
    """Discover every available (domain x mode x persona) identity for C7-proxy.

    A domain qualifies only if BOTH the human and biomimetic spec files exist.
    This is intentionally dynamic: the pre-registration mentions "11 domains"
    but on disk only the domains that carry both specs-humano.md AND
    specs-biomimetico.md are usable. Each identity carries the full spec text
    of its persona block.
    """
    c7 = config["arms"]["C7-proxy"]
    root = PROJECT_ROOT / c7["domains_root"]
    spec_files = c7["spec_filenames"]
    wanted_personas = set(c7["personas"])

    identities: list[dict[str, str]] = []
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        human = domain_dir / spec_files["humano"]
        bio = domain_dir / spec_files["biomimetico"]
        if not (human.exists() and bio.exists()):
            continue  # domain lacks the required specs -> skip
        for mode, spec_path in (("humano", human), ("biomimetico", bio)):
            for block in parse_specs_file(spec_path):
                if block["persona"] not in wanted_personas:
                    continue
                identities.append(
                    {
                        "identity_tag": (
                            f"{domain_dir.name}/modo-{mode}/"
                            f"persona-{block['persona']}"
                        ),
                        "domain_dir": domain_dir.name,
                        "domain": block["domain"],
                        "mode": mode,
                        "persona": block["persona"],
                        "spec_text": block["full_text"],
                    }
                )
    return identities


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL file, flushing immediately (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def existing_call_ids(path: Path) -> set[str]:
    """Return the set of call_id values already present (for crash resume)."""
    if not path.exists():
        return set()
    return {r.get("call_id", "") for r in read_jsonl(path)}
