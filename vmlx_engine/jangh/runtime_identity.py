"""Persisted-state identity for the canonical JANGH numerical runtime."""

import hashlib
import os
from pathlib import Path

# Freeze diagnostic routing controls alongside persisted-state identity.
DECODE_ROT = os.environ.get("JANGTQ2_DECODE_ROT", "host").strip().lower()
PREFILL = os.environ.get("JANGTQ2_PREFILL", "").strip().lower()
EXPERT_TILES = os.environ.get("JANGH_EXPERT_TILES", "0").strip()
if EXPERT_TILES not in {"0", "1"}:
    raise ValueError("JANGH_EXPERT_TILES must be 0 or 1")
if DECODE_ROT not in {"host", "kernel"}:
    raise ValueError("JANGTQ2_DECODE_ROT must be host or kernel")
if PREFILL not in {"", "steel", "nax"}:
    raise ValueError("JANGTQ2_PREFILL must be steel, nax, or unset")


def runtime_identity() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return (
        "jangh=" + digest.hexdigest()
        + ";decode_rot=" + DECODE_ROT + ";prefill=" + (PREFILL or "auto")
        + ";expert_tiles=" + EXPERT_TILES
    )
