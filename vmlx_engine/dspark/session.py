"""Multi-turn prefix reuse for the DSpark lane (MiniCPM-V-4.7).

The DSpark route ran every request from an empty cache: each turn of a conversation re-prefilled the whole history
and stored nothing for the next turn. This module applies the DFlash2 lane's session design (dflash2_runtime /
dflash2_session_disk) to DSpark, reusing its store, cut finders, cache-clone helper and SSD tier:

- entries: {kind, tokens, cache_len, target_cache, draft_cache, model_key}; ``draft_cache`` is the drafter's
  context (one mlx-lm KVCache per drafter layer, ``DSparkDrafter.context_caches``), so a resumed request keeps the
  conversation history the drafter attends to, not only the new turn;
- snapshots before the last assistant generation tag ("boundary": reasoning templates re-render the previous turn,
  so the end-of-turn sequence is rarely a prefix of the next prompt, the prompt up to the latest user message is)
  and at the end of the system message ("system": a new chat with the same system prompt and tools); plus the
  end-of-turn entry (exactly the tokens the caches hold, ``iter_dspark(session=...)``);
- SSD tier (same managed pool and budget as the block cache), with the SSD-only RAM release policy: an entry is
  dropped from RAM once its SSD copy is durable, and a later request restores it from SSD (RAM stays at the
  model's size). Image and video requests key their entries with salted placeholder ids (models/mllm.py
  ``_dspark_salted_media_keys``) and carry the canvas ``rope_delta``; audio requests take the native path.
VMLX_DSPARK_PREFIX_REUSE=0 disables reuse; VMLX_DSPARK_SSD=0 keeps the RAM tier only.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_OFF = {"0", "false", "off", "no"}
_DEFAULTS = {"root": None, "max_size_bytes": 0, "enabled": False}
STORE = None
SSD = None


def _store():
    global STORE
    if STORE is None:
        from ..dflash2_runtime import _DFlash2SessionStore

        STORE = _DFlash2SessionStore(max_entries=4)
    return STORE


def reuse_enabled() -> bool:
    return os.environ.get("VMLX_DSPARK_PREFIX_REUSE", "1").strip().lower() not in _OFF


def configure_defaults(*, root: Optional[str], max_size_bytes: int, enabled: bool) -> None:
    """CLI: where a DSpark session SSD tier would live (the block cache pool). Attached when a drafter loads."""
    _DEFAULTS.update(root=root, max_size_bytes=int(max_size_bytes or 0), enabled=bool(enabled))


def attach_ssd(target_path: str, drafter_path: str) -> None:
    global SSD
    if not _DEFAULTS["enabled"] or not _DEFAULTS["root"]:
        return
    if os.environ.get("VMLX_DSPARK_SSD", "1").strip().lower() in _OFF:
        logger.info("DSpark session SSD tier off (VMLX_DSPARK_SSD=0)")
        return
    try:
        from ..dflash2_session_disk import DFlash2SessionSSD, model_identity

        SSD = DFlash2SessionSSD(root=_DEFAULTS["root"], max_size_bytes=_DEFAULTS["max_size_bytes"],
                                model_key="dspark|" + model_identity(str(target_path), str(drafter_path)))
        logger.info("DSpark session SSD tier: %s", SSD.store.directory)
    except Exception:
        SSD = None
        logger.warning("DSpark session SSD tier unavailable; RAM tier only", exc_info=True)


def snapshot_cuts(tokenizer: Any, prompt: list, cache_len: int) -> list:
    """[(kind, cut)] for this prompt: the end of each leading system block (dflash2_runtime._system_block_cuts), then
    before the last assistant generation tag."""
    from ..dflash2_runtime import _assistant_tag_cut, _system_block_cuts

    if not reuse_enabled():
        return []
    boundary = _assistant_tag_cut(tokenizer, prompt, cache_len)
    systems = [c for c in _system_block_cuts(tokenizer, prompt, cache_len) if boundary is None or c < boundary]
    return [("system", c) for c in systems] + ([("boundary", boundary)] if boundary is not None else [])


def lookup(model_key: Any, prompt: list, tokenizer: Any, make_target: Callable, make_draft: Callable):
    """The stored entry covering the longest prefix of ``prompt`` (RAM, then SSD), or None. Ownership moves to the
    caller (the entry's caches are consumed by the request)."""
    if not reuse_enabled():
        return None
    resume = _store().take_matching(model_key, prompt)
    if resume is not None:
        resume.setdefault("source", "dspark-ram")
        return resume
    if SSD is None:
        return None
    try:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    except Exception:
        im_start = None
    eos = getattr(getattr(tokenizer, "stopping_criteria", None), "eos_token_ids", None) or \
        ([tokenizer.eos_token_id] if getattr(tokenizer, "eos_token_id", None) is not None else [])
    resume = SSD.take_matching(prompt, im_start_id=int(im_start) if isinstance(im_start, int) and im_start >= 0 else None,
                               eos_ids=eos, make_target=make_target, make_draft=make_draft)
    if resume is not None:
        resume["model_key"] = model_key
        resume["source"] = "dspark-ssd"
        if resume.get("draft_cache") is None:
            return None                    # an entry without drafter context cannot resume DSpark exactly aligned
    return resume


def ssd_parent(resume: Optional[dict]) -> Optional[dict]:
    """The SSD entry a cache built on ``resume`` extends (for delta entries), as the DFlash2 lane does."""
    if resume is None:
        return None
    if resume.get("ssd_ref"):
        return dict(resume["ssd_ref"], receipt=None)
    receipt = resume.get("_ssd_receipt")
    if getattr(receipt, "vmlx_ssd_ref", None):
        return dict(receipt.vmlx_ssd_ref, receipt=receipt)
    return None


def chain_after(parent: Optional[dict], receipt) -> Optional[dict]:
    """The SSD parent for the NEXT entry a request stores, after storing one with ``parent``. When ``parent`` could not
    be used (none, or the delta chain is at MAX_CHAIN) that entry was written in full: make it the parent of the
    request's later entries, so a boundary + turn pair costs one full write and one small delta instead of two full
    writes (MiMo 98k: 2 x 2.3 GB, 0.49 + 0.70 s, and the next request's admission waited 1.16 s). A usable parent is
    kept (chain depth grows by one per request, as before)."""
    from ..dflash2_session_disk import MAX_CHAIN

    usable = bool(parent) and int(parent.get("depth", 0)) + 1 <= MAX_CHAIN
    if usable:
        return parent
    ref = getattr(receipt, "vmlx_ssd_ref", None)
    if not ref:
        return parent
    try:
        if receipt.done() and not (receipt.result() or {}).get("durable", True):
            return parent                                  # refused / failed write: never a parent
    except Exception:  # noqa: BLE001
        return parent
    return dict(ref, receipt=receipt)


def clone(caches: list, factory: Callable) -> Optional[list]:
    from ..dflash2_runtime import _clone_cache_shells

    return _clone_cache_shells(caches, factory)


def store(entry: dict, parent: Optional[dict] = None) -> Future:
    """RAM store + SSD write-behind with the SSD-only RAM release policy (dflash2_runtime._store_session)."""
    from ..glm_dflash2 import ssd_only_ram_release_enabled

    st = _store()
    entry.setdefault("draft_delta", True)      # the drafter context is append-only and token-aligned: SSD delta
    ssd_only = SSD is not None and ssd_only_ram_release_enabled()
    if not ssd_only:
        st.put(entry)
    if SSD is None:
        receipt = Future()
        receipt.set_result({"outcome": "stored", "durable": False, "retained_tokens": int(entry["cache_len"]),
                            "detail": "dspark-session-ram"})
        receipt.vmlx_lookup_served_from_ram = True
        return receipt
    if parent:
        entry["ssd_parent"] = parent
    try:
        receipt = SSD.put(entry)
        entry.pop("ssd_parent", None)
        if ssd_only:
            refused = receipt.done() and not (receipt.result() or {}).get("durable")
            if refused:
                st.put(entry)
                st.attach_receipt(entry, receipt)
            receipt.vmlx_lookup_served_from_ram = bool(refused)
            return receipt
        receipt.vmlx_lookup_served_from_ram = True
        st.attach_receipt(entry, receipt)
        return receipt
    except Exception:
        logger.warning("DSpark SSD snapshot failed; RAM entry kept", exc_info=True)
        entry.pop("ssd_parent", None)
        if ssd_only:
            st.put(entry)
        receipt = Future()
        receipt.set_result({"outcome": "failed", "durable": False, "retained_tokens": 0,
                            "detail": "DSpark SSD snapshot failed"})
        return receipt


def clear(include_ssd: bool = False) -> int:
    _store().clear()
    if include_ssd and SSD is not None:
        return SSD.clear()
    return 0


def cache_stats() -> dict:
    """Administrative view of this lane's existing store and shared SSD budget."""
    result = {"ram": _store().stats(), "ssd": None}
    if SSD is not None:
        from dataclasses import asdict

        budget = asdict(SSD.store.budget.refresh_health())
        budget["root"] = str(SSD.store.root)
        result["ssd"] = dict(SSD.stats, pending_writes=SSD._q.unfinished_tasks, global_budget=budget)
    return result
