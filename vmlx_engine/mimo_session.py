"""Multi-turn prefix reuse for the MiMo-V2.6 DFlash lane (RAM + SSD), the DSpark lane's design (dspark/session.py).

The ported lane kept two RAM-only prompt snapshots (``MiMoPrefixStore``): no SSD tier, nothing survived a restart, a
third conversation evicted the first, and the end-of-turn state was never stored, so each follow-up re-prefilled the
previous answer. Here the lane stores what the DFlash2 / DSpark lanes store, in the same managed SSD pool:

- entries ``{kind, tokens, cache_len, target_cache, draft_context, model_key}``: the target cache holds exactly
  ``cache_len`` positions; ``draft_context`` is the drafter's context = the target layer taps of the last
  ``min(window, cache_len)`` positions (rows ``(n, n_taps * hidden)``, at most 1024 x 20480 bf16 = 42 MB). Taps are a
  pure function of the target's prefix, so a restore re-projects them through the drafter (fc / hidden_norm / per-layer
  K,V) exactly as the live lane does; no drafter cache layout is persisted;
- snapshots at the end of the system message ("system") and before the last assistant generation tag ("boundary"),
  plus the end-of-turn entry ("turn"); media requests key their entries with salted placeholder ids;
- SSD write-behind with the SSD-only RAM release policy (glm_dflash2.ssd_only_ram_release_enabled): an entry leaves
  RAM once its SSD copy is durable; the end-of-turn receipt is the request's persistence future (publish early,
  settled before the next admission). Full-attention layers (KVCache) are stored as deltas against the parent entry;
  the 39 sliding-window layers (RotatingKVCache, <= 128 + verify rows each) are stored whole.

Rotating caches: a RotatingKVCache past its window returns its LIVE buffer from ``state``, and single-token updates
write into it in place, so a shared-array clone (dflash2_runtime._clone_cache_shells) would keep changing after the
snapshot. ``clone`` copies those layers (``* 1``); full-attention layers share an immutable slice.

Chunk checkpoints (2026-10-16): a prompt that diverges INSIDE the last user message (another question about the same
pasted document, an edited / regenerated message) matched no entry -- system / boundary / turn entries all end at message
edges -- and re-prefilled everything (20,752-token follow-up sharing 20,740 tokens: 0 restored, 25.6 s). With the SSD
tier attached, prefill also stops at every absolute multiple of CHUNK_EVERY (default 4096, the prefill step) past the
restored prefix and below the boundary; each "chunk" entry is stored at once (``ChunkWriter``: the clone is not held for
the whole answer) as a delta against the PREVIOUS chunk, so a long paste writes each row once (a delta against the
request's resume parent would rewrite every earlier row: quadratic); MAX_CHAIN still forces a full entry every 8 links.
The SSD cut finder probes the same multiples. RAM-only (no SSD) keeps the old cuts: the 4-entry RAM tier would churn.
VMLX_MIMO_DFLASH_CHUNK_SNAPSHOT_EVERY=0 disables chunk entries.

VMLX_MIMO_DFLASH_PREFIX_REUSE=0 disables reuse; VMLX_MIMO_DFLASH_SSD=0 keeps the RAM tier only.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_OFF = {"0", "false", "off", "no"}
STORE = None
SSD = None


def _store():
    global STORE
    if STORE is None:
        from .dflash2_runtime import _DFlash2SessionStore

        STORE = _DFlash2SessionStore(max_entries=4)
    return STORE


def reuse_enabled() -> bool:
    return os.environ.get("VMLX_MIMO_DFLASH_PREFIX_REUSE", "1").strip().lower() not in _OFF


def attach_ssd(target_path: str, drafter_path: str) -> None:
    """Same managed pool and budget as the block cache (dspark.session.configure_defaults is called by the CLI)."""
    global SSD
    from .dspark.session import _DEFAULTS

    if not _DEFAULTS["enabled"] or not _DEFAULTS["root"]:
        return
    if os.environ.get("VMLX_MIMO_DFLASH_SSD", "1").strip().lower() in _OFF:
        logger.info("MiMo DFlash session SSD tier off (VMLX_MIMO_DFLASH_SSD=0)")
        return
    try:
        from .dflash2_session_disk import DFlash2SessionSSD, model_identity

        SSD = DFlash2SessionSSD(root=_DEFAULTS["root"], max_size_bytes=_DEFAULTS["max_size_bytes"],
                                model_key="mimo-dflash|" + model_identity(str(target_path), str(drafter_path)))
        logger.info("MiMo DFlash session SSD tier: %s", SSD.store.directory)
    except Exception:
        SSD = None
        logger.warning("MiMo DFlash session SSD tier unavailable; RAM tier only", exc_info=True)


def chunk_every() -> int:
    try:
        return max(0, int(os.environ.get("VMLX_MIMO_DFLASH_CHUNK_SNAPSHOT_EVERY", "4096")))
    except ValueError:
        return 4096


def chunk_cuts(cache_len: int, upper: int, every: int) -> list:
    """Absolute multiples of ``every`` strictly between the restored prefix and ``upper``."""
    if every <= 0:
        return []
    first = (int(cache_len) // every + 1) * every
    return list(range(first, int(upper), every))


def snapshot_cuts(tokenizer: Any, prompt: list, cache_len: int) -> list:
    from .dspark.session import snapshot_cuts as _cuts

    if not reuse_enabled():
        return []
    cuts = _cuts(tokenizer, prompt, cache_len)
    if SSD is not None:
        upper = next((c for k, c in cuts if k == "boundary"), len(prompt))
        taken = {c for _, c in cuts}
        cuts = sorted(cuts + [("chunk", c) for c in chunk_cuts(cache_len, upper, chunk_every()) if c not in taken],
                      key=lambda kc: kc[1])
    return cuts


class ChunkWriter:
    """Stores chunk entries as they are cut (each a delta against the previous chunk) and hands the request's
    deferred entries (boundary / turn) the closest stored chunk below them as their parent."""

    def __init__(self, parent: Optional[dict]):
        self.parent = parent
        self.refs: list = []                                   # [(cut, parent dict)] in cut order

    def put(self, entry: dict):
        receipt = store(entry, parent=self.parent)
        ref = getattr(receipt, "vmlx_ssd_ref", None)
        try:
            refused = receipt.done() and not (receipt.result() or {}).get("durable", True)
        except Exception:  # noqa: BLE001
            refused = True
        if ref and not refused:
            self.parent = dict(ref, receipt=receipt)
            self.refs.append((int(entry["cache_len"]), self.parent))
        return receipt

    def parent_for(self, cut: int, parent: Optional[dict]) -> Optional[dict]:
        below = [p for c, p in self.refs if c < int(cut)]
        if below and (parent is None or int(below[-1]["len"]) > int(parent.get("len", 0))):
            return below[-1]
        return parent


def ssd_parent(resume: Optional[dict]) -> Optional[dict]:
    from .dspark.session import ssd_parent as _parent

    return _parent(resume)


def chain_after(parent: Optional[dict], receipt) -> Optional[dict]:
    from .dspark.session import chain_after as _chain

    return _chain(parent, receipt)


def clone(caches: list, factory: Callable) -> Optional[list]:
    """Independent snapshot of MiMo's target cache (see the module docstring for why rotating layers are copied)."""
    from mlx_lm.models.cache import RotatingKVCache

    try:
        fresh = factory()
        for dst, src in zip(fresh, caches):
            state = src.state
            if isinstance(src, RotatingKVCache):
                window = int(src.max_size)
                keys = src.keys
                if (keys is not None and int(src.keep) == 0 and int(keys.shape[2]) > window
                        and int(src._idx) == int(keys.shape[2])):
                    # Right after a multi-row prefill chunk the ring is in concat mode holding window - 1 + chunk
                    # rows in temporal order; only the last `window` rows are ever attended again (the next update
                    # drops the oldest). Keep those: a 2048-row chunk otherwise put ~434 MB of dead SWA rows into
                    # every boundary snapshot (500 MB entries vs 282 MB end-of-turn entries on the 2026-10-15 chain).
                    k, v = keys[..., -window:, :] * 1, src.values[..., -window:, :] * 1
                    dst.state = [k, v]
                    dst.meta_state = tuple(str(x) for x in (src.keep, window, src.offset, window))
                    continue
                state = [None if a is None else a * 1 for a in state]
            dst.state = list(state) if isinstance(state, (list, tuple)) else state
            if isinstance(src, RotatingKVCache):
                dst.meta_state = src.meta_state
        return fresh
    except Exception:
        logger.info("MiMo DFlash cache clone failed", exc_info=True)
        return None


def lookup(model_key: Any, prompt: list, tokenizer: Any, make_target: Callable, eos_ids) -> Optional[dict]:
    """The stored entry covering the longest prefix of ``prompt`` (RAM, then SSD), or None; ownership moves to the
    caller. An entry without its drafter context is a miss (the drafter could not resume aligned)."""
    if not reuse_enabled():
        return None
    resume = _store().take_matching(model_key, prompt)
    if resume is not None:
        resume.setdefault("source", "mimo-dflash-ram")
        return resume
    if SSD is None:
        return None
    try:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    except Exception:
        im_start = None
    resume = SSD.take_matching(prompt, im_start_id=int(im_start) if isinstance(im_start, int) and im_start >= 0 else None,
                               eos_ids=list(eos_ids), make_target=make_target, make_draft=list,
                               every=chunk_every())
    if resume is not None:
        resume["model_key"] = model_key
        resume["source"] = "mimo-dflash-ssd"
        if resume.get("draft_context") is None:
            return None
    return resume


def store(entry: dict, parent: Optional[dict] = None) -> Future:
    """RAM store + SSD write-behind with the SSD-only RAM release policy (as dspark.session.store)."""
    from .glm_dflash2 import ssd_only_ram_release_enabled

    st = _store()
    entry.setdefault("draft_cache", None)
    ssd_only = SSD is not None and ssd_only_ram_release_enabled()
    if not ssd_only:
        st.put(entry)
    if SSD is None:
        receipt = Future()
        receipt.set_result({"outcome": "stored", "durable": False, "retained_tokens": int(entry["cache_len"]),
                            "detail": "mimo-dflash-session-ram"})
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
        logger.warning("MiMo DFlash SSD snapshot failed; RAM entry kept", exc_info=True)
        entry.pop("ssd_parent", None)
        if ssd_only:
            st.put(entry)
        receipt = Future()
        receipt.set_result({"outcome": "failed", "durable": False, "retained_tokens": 0,
                            "detail": "MiMo DFlash SSD snapshot failed"})
        return receipt


def clear(include_ssd: bool = False) -> int:
    _store().clear()
    if include_ssd and SSD is not None:
        return SSD.clear()
    return 0


def cache_stats() -> dict:
    result = {"ram": _store().stats(), "ssd": None}
    if SSD is not None:
        from dataclasses import asdict

        budget = asdict(SSD.store.budget.refresh_health())
        budget["root"] = str(SSD.store.root)
        result["ssd"] = dict(SSD.stats, pending_writes=SSD._q.unfinished_tasks, global_budget=budget)
    return result
