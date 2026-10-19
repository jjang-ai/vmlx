# SPDX-License-Identifier: Apache-2.0
"""SSD (L2) tier for the DFlash2 session store.

WHY THIS EXISTS
---------------
A bundle that ships a DFlash2 drafter (Qwen3.8-27B JANG_4D / JANGH2) runs on
SimpleEngine (``--no-continuous-batching``), so the scheduler's prefix cache
and block-disk SSD tier are not in play.  Its only multiturn reuse was the
in-RAM ``_DFlash2SessionStore`` (4 entries): lost on restart, and a 5th
conversation evicted the 1st.  This module writes every stored session entry
to the managed SSD pool and reads it back on a RAM miss.

WHAT IS STORED (one safetensors file + JSON sidecar per entry)
--------------------------------------------------------------
Exactly what the RAM store holds: the target prompt cache (full-attention KV
and GatedDeltaNet recurrent state, via each cache object's ``state`` /
``meta_state``), the drafter cache when it was kept, the drafter's bridging
hidden slice (``draft_hidden_gap``), the token ids and ``cache_len``.  Full
precision, no re-quantisation: a restored entry is bit-identical to the RAM
entry it was written from.

HOW A PROMPT FINDS AN ENTRY
---------------------------
The RAM store matches "stored tokens are a prefix of the prompt".  The disk
store is keyed by the exact token prefix (sha256 of the uint32 ids), so the
lookup must guess where an entry could end.  Entries only ever end at two
places, both visible in the next prompt:
  * ``boundary`` entries end right BEFORE an ``<|im_start|>`` (the assistant
    generation tag of an earlier prompt), and
  * ``turn`` entries end right AFTER an EOS id (``<|im_end|>``) that closed a
    generated reply.
So the candidates are those positions in the new prompt, tried longest first.
The stored ids are compared with the prompt after load (hash collisions and
torn files are misses, never wrong state).

COST / TRADE-OFFS
-----------------
* Writes happen on a background thread after the arrays are evaluated on the
  generation thread. Final content is delivered before SimpleEngine waits
  for the turn's write receipt and publishes the terminal event. A reserved
  queue slot protects the turn snapshot from prefill snapshots; queue
  refusals and write failures never report durable storage.
* Up to three writes per turn (system + boundary + turn entry). They are needed: reasoning
  templates strip <think> from history, so the next USER turn only matches
  the boundary entry, while a TOOL continuation matches the turn entry.

DELTA ENTRIES (v4, 2026-10-12)
------------------------------
Full-attention KV is append-only, so an entry whose cache was built ON an entry that is already on SSD (the
request's previous snapshot, or the entry it resumed from) stores only the KV rows its parent does not have,
``[parent_len, cache_len)``, plus the fixed-size state (GatedDeltaNet conv/recurrent, the drafter's rotating
window, the drafter context rows) whole. Before this every entry held its whole prefix: Qwen3.8 27B JANGH2 wrote
5.8 GB per entry at 84.7k tokens, three per cold turn and two (11.6 GB) for a 31-token continuation, and each
write made a contiguous copy of the live KV. ``meta["parent"]`` names the parent's signature and length; a restore
loads the chain root-first, compares every link's token ids with the prompt, and assembles each KV layer once, with
headroom so the first append is in place. A missing or failed parent makes the child a miss (the writer refuses to
publish a child whose parent did not become durable), never wrong state. Chains are cut to a full entry every
``MAX_CHAIN`` links. Delta rows are COPIED when packed: a view would keep the live KV buffer referenced, and MLX's
copy-on-write would then duplicate the whole buffer on the next decode append.
* Size is bounded by the same aggregate SSD budget as every other managed
  cache (``--block-disk-cache-max-gb`` / ``--block-disk-cache-max-percent``),
  with LRU eviction by the pool.  ``--disable-block-disk-cache`` turns it off.
* The namespace includes both bundles' file fingerprints (size + mtime of
  weights/configs) and the runtime package fingerprint, so a re-quantised
  bundle or upgraded runtime cannot replay old state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from array import array
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# v2 (2026-10-07): v1 entries of hybrid targets were written while snapshots
# aliased the live GatedDeltaNet state (recurrent state advanced past the cut;
# restores changed answers -- dflash2_runtime._clone_cache_shells). v1 also had
# no rope_delta metadata. Never read v1; its namespace is cleared at startup.
# v3 (2026-10-08): v2 drafter caches rebuilt on resume were written with a
# forced absolute offset over a physical buffer and could hold up to 255 zero
# K/V rows (dflash2_runtime._patch_rotating_cache_resume_math). Never read v2.
# v4 (2026-10-12): delta entries (``meta["parent"]``); v3 readers would restore a delta as a full cache. Never read v3.
SCHEMA = "dflash2_session_v4"
LEGACY_SCHEMAS = ("dflash2_session_v1", "dflash2_session_v2", "dflash2_session_v3")
MAX_CHAIN = 8                   # links before a full entry is written again (bounds restore reads and dependencies)
_DELTA_CLASSES = ("KVCache",)   # append-only caches whose state is (keys, values) with the token axis at dim 2


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(array("I", tokens).tobytes()).hexdigest()


def signature_for(tokens: list[int]) -> str:
    return f"{len(tokens)}:{_token_digest(tokens)}"


def _read_header_meta(path) -> dict:
    """The ``dflash2`` metadata of a stored entry from its safetensors header alone (no tensor reads), plus the names
    of its stored arrays under ``_arrays``."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n <= 64 << 20:
            raise ValueError(f"implausible safetensors header length {n}")
        head = json.loads(f.read(n))
    meta = json.loads((head.get("__metadata__") or {})["dflash2"])
    meta["_arrays"] = [k for k in head if k != "__metadata__"]
    return meta


def _parent_key_length(parent: dict, tokens: list[int]) -> Optional[int]:
    """Validate the lookup prefix independently of its cached row count.

    A DFlash2 turn signs confirmed tokens but caches confirmed[:-1]. Its
    pending final token belongs to the key and is evaluated on resume.
    Require a strictly shorter key to prevent overwriting a parent with a
    delta that depends on that same file.
    """
    try:
        sig = parent["sig"]
        rows = int(parent["len"])
        key_len = int(sig.split(":", 1)[0])
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if not 0 < rows <= key_len <= rows + 1 or key_len >= len(tokens):
        return None
    return key_len if signature_for(tokens[:key_len]) == sig else None


def candidate_cuts(prompt: list[int], im_start_id: Optional[int], eos_ids: Iterable[int], every: int = 0) -> list[int]:
    """Prefix lengths at which a stored entry can end, longest first. ``every`` > 0 adds the absolute multiples of
    ``every`` (lanes that store periodic mid-message checkpoints, e.g. mimo_session chunk entries)."""
    eos = {int(t) for t in eos_ids}
    cuts = set(range(int(every), len(prompt), int(every))) if every and int(every) > 0 else set()
    for i, t in enumerate(prompt):
        if im_start_id is not None and t == im_start_id and i > 0:
            cuts.add(i)                    # boundary: tokens[:i], i = tag index
        if t in eos:
            cuts.add(i + 1)                # turn: tokens end with the EOS id
    return sorted((c for c in cuts if 0 < c < len(prompt)), reverse=True)


def _has_meta_setter(cache: Any) -> bool:
    prop = getattr(type(cache), "meta_state", None)
    return isinstance(prop, property) and prop.fset is not None


def _detached(a):
    """A new buffer with identical bits (x * 1 keeps -0.0; bool via logical_and)."""
    import mlx.core as mx

    return mx.logical_and(a, True) if a.dtype == mx.bool_ else a * 1


def _delta_layer(c, state, base: int) -> bool:
    return (base > 0 and type(c).__name__ in _DELTA_CLASSES and isinstance(state, (list, tuple)) and len(state) == 2
            and all(a is not None and a.ndim == 4 for a in state) and int(state[0].shape[2]) > base
            and int(getattr(c, "offset", -1)) == int(state[0].shape[2]))


def _pack(prefix: str, caches: list, arrays: dict, layers: list, base: int = 0) -> None:
    """base > 0: KV layers keep only rows [base, offset) (copied, see DELTA ENTRIES); other layers whole."""
    for i, c in enumerate(caches):
        state = list(c.state) if isinstance(c.state, (list, tuple)) else [c.state]
        nones = []
        delta = _delta_layer(c, state, base)
        for j, a in enumerate(state):
            if a is None:
                nones.append(j)
            else:
                arrays[f"{prefix}{i}.{j}"] = _detached(a[:, :, base:, :]) if delta else a
        meta = None
        if _has_meta_setter(c):
            m = c.meta_state
            meta = list(m) if isinstance(m, (list, tuple)) else m
        info = {"cls": type(c).__name__, "n": len(state), "none": nones,
                "tuple": isinstance(c.state, (list, tuple)), "meta": meta}
        if delta:
            info["delta"] = int(base)
        layers.append(info)


def _unpack(prefix: str, factory: Callable[[], list], arrays: dict, layers: list) -> Optional[list]:
    fresh = factory()
    if len(fresh) != len(layers):
        return None
    for i, (dst, info) in enumerate(zip(fresh, layers)):
        if type(dst).__name__ != info["cls"]:
            return None
        state = [None if j in info["none"] else arrays[f"{prefix}{i}.{j}"] for j in range(info["n"])]
        dst.state = state if info["tuple"] else state[0]
        if info["meta"] is not None:
            dst.meta_state = tuple(info["meta"]) if isinstance(info["meta"], list) else info["meta"]
    return fresh


def _assemble_kv(pieces: list, step: int):
    """Concatenate one KV layer's chain pieces (root first) along the token axis into ONE buffer with headroom up to
    the next ``step`` multiple, so the first append after the restore writes in place instead of copying the whole
    buffer to grow it. Evaluated per layer so the loaded pieces are freed before the next layer is read."""
    import mlx.core as mx

    n = sum(int(p.shape[2]) for p in pieces)
    pad = (-n) % step if step > 0 else 0
    if pad == 0 and step > 0:
        pad = step
    parts = list(pieces)
    if pad:
        b, h, _, d = pieces[0].shape
        parts.append(mx.zeros((b, h, pad, d), dtype=pieces[0].dtype))
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
    mx.eval(out)
    return out, n


def _unpack_chain(prefix: str, key: str, factory: Callable[[], list], links: list) -> Optional[list]:
    """Rebuild a cache list from a root-first chain of (arrays, meta) links: append-only KV layers concatenate the
    root's rows and every delta; every other layer comes whole from the leaf. Layout mismatches are misses."""
    fresh = factory()
    leaf_arrays, leaf_meta = links[-1]
    leaf_layers = leaf_meta[key]
    if len(fresh) != len(leaf_layers):
        return None
    for i, (dst, info) in enumerate(zip(fresh, leaf_layers)):
        if type(dst).__name__ != info["cls"]:
            return None
        kv_layer = type(dst).__name__ in _DELTA_CLASSES and info["n"] == 2 and not info["none"]
        if kv_layer:
            pieces_k, pieces_v, expect = [], [], 0
            for arrays, meta in links:
                linfo = meta[key][i]
                if linfo["cls"] != info["cls"]:
                    return None
                base = int(linfo.get("delta", 0))
                if base != expect:                          # a gap or overlap in the chain: never guess
                    return None
                k, v = arrays[f"{prefix}{i}.0"], arrays[f"{prefix}{i}.1"]
                pieces_k.append(k); pieces_v.append(v)
                expect = base + int(k.shape[2])
            step = int(getattr(dst, "step", 256) or 256)
            keys, n = _assemble_kv(pieces_k, step)
            values, _ = _assemble_kv(pieces_v, step)
            del pieces_k, pieces_v
            dst.keys, dst.values, dst.offset = keys, values, n
            continue
        if "delta" in info:
            return None
        state = [None if j in info["none"] else leaf_arrays[f"{prefix}{i}.{j}"] for j in range(info["n"])]
        dst.state = state if info["tuple"] else state[0]
        if info["meta"] is not None:
            dst.meta_state = tuple(info["meta"]) if isinstance(info["meta"], list) else info["meta"]
    return fresh


class DFlash2SessionSSD:
    """Write-behind SSD tier keyed by exact token prefix (see module doc)."""

    queue_depth = 2

    def __init__(self, *, root: str | Path, max_size_bytes: int, model_key: str):
        from .utils.omni_session_disk_store import OmniSessionDiskStore

        self.store = OmniSessionDiskStore(
            root=root, model_key=model_key, max_size_bytes=int(max_size_bytes), schema=SCHEMA
        )
        self.legacy_cleared = 0
        for legacy in LEGACY_SCHEMAS:
            # Through the store's own API so the pool's budget ledger stays in
            # sync; only when that namespace exists (constructing creates it).
            legacy_ns = Path(root).expanduser().resolve() / hashlib.sha256(
                f"{model_key}:{legacy}".encode()).hexdigest()[:16]
            if not (legacy_ns / "native_sessions").is_dir():
                continue
            try:
                old = OmniSessionDiskStore(root=root, model_key=model_key,
                                           max_size_bytes=int(max_size_bytes), schema=legacy)
                try:
                    self.legacy_cleared += int(old.clear() or 0)
                finally:
                    old.close()
            except Exception:
                logger.warning("DFlash2 SSD: could not clear legacy %s entries", legacy, exc_info=True)
        if self.legacy_cleared:
            logger.info("DFlash2 SSD: cleared %d bytes of legacy session entries (schema %s)",
                        self.legacy_cleared, ",".join(LEGACY_SCHEMAS))
        self.stats = {"writes": 0, "write_bytes": 0, "write_s": 0.0, "write_errors": 0, "delta_writes": 0,
                      "dropped": 0, "hits": 0, "hit_tokens": 0, "load_s": 0.0, "misses": 0, "chain_misses": 0,
                      "same_prefix_kept": 0, "deep_parent_full_writes": 0}
        # Two prefill snapshots plus a reserved final-turn slot. Requests
        # settle their terminal receipt before releasing the generation lock.
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=self.queue_depth + 1)
        self._worker = threading.Thread(target=self._run, name="dflash2-ssd-writer", daemon=True)
        self._worker.start()

    # ---- write ---------------------------------------------------------------
    @staticmethod
    def _usable_parent_static(entry: dict) -> Optional[dict]:
        """``entry["ssd_parent"]`` (``{"sig", "len", "depth", "receipt"}``: the SSD entry this cache was built on)
        when a delta against it is sound: a strict prefix by signature, chain not too deep, its write not failed."""
        parent = entry.get("ssd_parent")
        if not parent or parent.get("delta_compatible") is False or not parent.get("generation"):
            return None
        n, tokens = int(parent.get("len", 0)), entry["tokens"]
        if not 0 < n < int(entry["cache_len"]) or n > len(tokens) or int(parent.get("depth", 0)) + 1 > MAX_CHAIN:
            return None
        if _parent_key_length(parent, tokens) is None:
            return None
        receipt = parent.get("receipt")
        if receipt is not None and receipt.done():
            try:
                if not (receipt.result() or {}).get("durable"):
                    return None
            except Exception:
                return None
        return parent

    def _chain_depth(self, sig: str, cache_len: Optional[int]) -> Optional[tuple[int, dict]]:
        """(links - 1, leaf metadata) of the stored chain ending at ``sig`` read from headers only, or None when the
        entry or any parent is missing, a link's row count is not what its child recorded, or the walk is longer than
        the loader accepts (take_matching / _load_chain: MAX_CHAIN + 1 parents).
        A None cache_len accepts the leaf's stored row count; parent links still require exact counts."""
        depth, cur_sig, cur_len, leaf = 0, sig, int(cache_len) if cache_len is not None else None, None
        expected_generation = None
        try:
            for _ in range(MAX_CHAIN + 2):
                meta = self.store.load(cur_sig, _read_header_meta)
                if meta is None or (cur_len is not None and int(meta["cache_len"]) != cur_len):
                    return None
                if expected_generation is not None and meta.get("generation") != expected_generation:
                    return None
                leaf = leaf or meta
                parent = meta.get("parent")
                if not parent:
                    return depth, leaf
                expected_generation = parent.get("generation")
                if not expected_generation:
                    return None
                depth += 1
                cur_sig, cur_len = parent["sig"], int(parent["len"])
        except Exception:
            return None
        return None

    def _usable_parent(self, entry: dict) -> Optional[dict]:
        """As _usable_parent_static, and for a parent that is already a durable file (restored from SSD, or a write
        this process saw settle), the depth that counts is its ACTUAL chain on disk, read from headers: a parent file
        replaced by a later write of the same prefix could make a recorded depth understate the walk, and a delta past
        the loader's limit is a silent miss (2026-10-19: an 8-deep entry walked 11 links). Parents still queued in this
        process keep their recorded depth (their own write settles first, FIFO)."""
        parent = self._usable_parent_static(entry)
        if parent is None:
            return None
        receipt = parent.get("receipt")
        if receipt is not None and not receipt.done():
            return parent
        walked = self._chain_depth(parent["sig"], int(parent["len"]))
        if (walked is None or walked[0] + 1 > MAX_CHAIN
                or walked[1].get("generation") != parent["generation"]):
            self.stats["deep_parent_full_writes"] += 1
            logger.info("DFlash2 SSD: parent %d-token entry %s; writing a whole entry", int(parent["len"]),
                        "chain unreadable" if walked is None else
                        "generation changed" if walked[1].get("generation") != parent["generation"] else
                        "already %d deep on disk" % walked[0])
            return None
        return dict(parent, depth=max(int(parent.get("depth", 0)), walked[0]))

    def _stored(self, sig: str) -> bool:
        try:
            return self.store.paths(sig)[0].is_file()
        except Exception:
            return False

    def _kept_same_prefix(self, entry: dict, sig: str) -> Optional[Future]:
        """A durable receipt for an entry whose exact prefix is already stored with an intact chain, or None.

        Rewriting that file would REPLACE an entry other entries may name as their parent; a delta replacement
        lengthens every such chain (it now walks through this entry's new parents) and past MAX_CHAIN + 1 links the
        loader drops it as a chain miss. Measured 2026-10-19 (MiMo in-app chain, second session over the same
        prefixes): turns re-stored the 9,187- and 18,577-token entries with new parents, the previous session's
        27,159-token boundary then walked 11 links and was skipped, and the next turn restored 24,576 instead of
        27,1xx tokens (+2.6k-token re-prefill). The old prefix remains a valid standalone snapshot, so the
        stored file is kept; it is rewritten only when it is less complete than this entry (drafter context or state
        missing). The kept file is NOT a delta parent for the incoming live cache: equal tokens may have different
        KV bits after a different prefill chunking. A later actual restore can safely use it as a parent."""
        # The lookup key includes confirmed tokens, while an AR/speculative turn
        # may leave its last token pending. Preserve the existing snapshot even
        # when its row count differs by that pending token: replacing it would
        # invalidate descendants bound to its exact generation.
        walked = self._chain_depth(sig, None)
        if walked is None:
            return None
        depth, meta = walked
        stored_rows = int(meta["cache_len"])
        if not 0 < stored_rows <= len(entry["tokens"]) <= stored_rows + 1:
            return None
        arrays = set(meta.get("_arrays") or ())
        if entry.get("draft_context") is not None and "draft_context" not in arrays:
            return None
        if entry.get("draft_cache") is not None and meta.get("draft") is None:
            return None
        self.stats["same_prefix_kept"] += 1
        logger.info("DFlash2 SSD kept %s entry: %d tokens already durable (same prefix, chain %d)",
                    entry.get("kind", "turn"), stored_rows, depth)
        receipt = Future()
        receipt.vmlx_ssd_ref = {"sig": sig, "len": stored_rows, "depth": depth,
                                "generation": meta.get("generation"), "delta_compatible": False}
        receipt.set_running_or_notify_cancel()
        receipt.set_result({"outcome": "stored", "durable": True, "retained_tokens": stored_rows,
                            "detail": "DFlash2 SSD entry already durable (same prefix kept)"})
        return receipt

    def put(self, entry: dict) -> Future:
        """Snapshot ``entry`` (a RAM-store dict) and queue it for SSD (a delta when ``entry["ssd_parent"]`` allows).

        The receipt carries ``vmlx_ssd_ref`` = this entry's ``{"sig", "len", "depth"}`` for a later child."""
        import mlx.core as mx

        sig = signature_for([int(t) for t in entry["tokens"]])
        kept = self._kept_same_prefix(entry, sig)
        if kept is not None:
            return kept
        arrays: dict = {}
        target_layers: list = []
        # A same-prefix file that exists but could not be kept (broken / too-long chain, less complete) is replaced
        # by a WHOLE entry: entries naming it as their parent then walk a shorter chain, never a longer one.
        parent = None if self._stored(sig) else self._usable_parent(entry)
        _pack("t", entry["target_cache"], arrays, target_layers, int(parent["len"]) if parent else 0)
        if parent is not None and not any("delta" in info for info in target_layers):
            parent = None                       # no append-only layer used it: a whole entry, no dependency
        draft_layers = None
        if entry.get("draft_cache") is not None:
            # A drafter cache that never ran (e.g. a max_tokens=1 request) has empty layers whose
            # `state` raises; store the entry without drafter state (it is rebuilt on resume).
            try:
                d_arrays: dict = {}
                d_layers: list = []
                # DSpark (entry["draft_delta"]): the drafter context is an append-only KVCache aligned with the
                # target's token positions, so it is a delta against the same parent; without this every entry
                # rewrote the whole context (35B: 10 KB/token, 140 MB of a 284 MB entry at 14k tokens). DFlash2
                # drafter state (rotating window, a context that can lag the target by one row) stays whole.
                d_base = int(parent["len"]) if parent is not None and entry.get("draft_delta") else 0
                _pack("d", entry["draft_cache"], d_arrays, d_layers, d_base)
            except (AttributeError, TypeError):
                d_layers = None
            if d_layers is not None:
                arrays.update(d_arrays)
                draft_layers = d_layers
                if entry.get("draft_hidden_gap") is not None:
                    arrays["gap"] = entry["draft_hidden_gap"]
        tokens = [int(t) for t in entry["tokens"]]
        if entry.get("draft_context") is not None:
            arrays["draft_context"] = entry["draft_context"]
        arrays["tokens"] = mx.array(tokens, dtype=mx.uint32)
        # Evaluate on the generation thread; the writer only copies bytes.
        mx.eval(list(arrays.values()))
        depth = int(parent.get("depth", 0)) + 1 if parent else 0
        # Token signatures identify lookup prefixes, not exact numerical state. Bind
        # children to this particular snapshot so a later replacement can only
        # invalidate a child, never splice different KV bits into its delta.
        generation = uuid.uuid4().hex
        meta = {"generation": generation, "kind": entry.get("kind", "turn"), "cache_len": int(entry["cache_len"]),
                "rope_delta": entry.get("rope_delta"),
                "target": target_layers, "draft": draft_layers,
                "parent": {"sig": parent["sig"], "len": int(parent["len"]), "depth": depth,
                           "generation": parent["generation"]} if parent else None}
        receipt = Future()
        receipt.vmlx_ssd_ref = {"sig": signature_for(tokens), "len": int(entry["cache_len"]), "depth": depth,
                                "generation": generation}
        # Cancelling an HTTP consumer must not cancel an already-owned write.
        receipt.set_running_or_notify_cancel()
        try:
            if meta["kind"] != "turn" and self._q.qsize() >= self.queue_depth:
                raise queue.Full
            self._q.put_nowait((signature_for(tokens), arrays, meta, receipt,
                                (parent or {}).get("receipt")))
        except queue.Full:
            self.stats["dropped"] += 1
            logger.info("DFlash2 SSD write skipped (writer busy); entry stays in RAM only")
            receipt.set_result({"outcome": "refused", "durable": False,
                                "retained_tokens": 0, "detail": "DFlash2 SSD writer queue full"})
        return receipt

    def _run(self) -> None:
        import mlx.core as mx

        while True:
            item = self._q.get()
            sig, arrays, meta, receipt = item[:4]
            parent_receipt = item[4] if len(item) > 4 else None
            del item
            t0 = time.perf_counter()
            outcome = {"outcome": "failed", "durable": False,
                       "retained_tokens": 0, "detail": "DFlash2 SSD write failed"}
            try:
                parent = meta.get("parent")
                if parent is not None and not self._parent_durable(parent["sig"], parent_receipt):
                    # FIFO: the parent was queued first, so it is settled by now. A child of a failed or evicted
                    # parent would restore nothing; refuse it (its RAM entry stays, see dflash2_runtime).
                    raise OSError("DFlash2 SSD parent entry is not durable")
                if parent is not None:
                    current = self.store.load(parent["sig"], _read_header_meta)
                    if current is None or current.get("generation") != parent["generation"]:
                        raise OSError("DFlash2 SSD parent entry was replaced")
                path = self.store.save(
                    sig,
                    lambda tmp: mx.save_safetensors(str(tmp), arrays, {"dflash2": json.dumps(meta)}),
                )
                self.stats["writes"] += 1
                self.stats["write_bytes"] += path.stat().st_size
                self.stats["write_s"] += time.perf_counter() - t0
                if parent is not None:
                    self.stats["delta_writes"] = self.stats.get("delta_writes", 0) + 1
                    self._touch(parent["sig"])  # the pool's LRU must not evict a parent before its new child
                logger.info("DFlash2 SSD stored %s entry: %s tokens, %.0f MB, %.2fs%s",
                            meta["kind"], sig.split(":")[0], path.stat().st_size / 1e6,
                            time.perf_counter() - t0,
                            "" if parent is None else " (delta from %d tokens, chain %d)" % (parent["len"], parent["depth"]))
                outcome = {"outcome": "stored", "durable": True,
                           "retained_tokens": meta["cache_len"],
                           "detail": "DFlash2 SSD " + meta["kind"] + " snapshot"}
            except Exception:
                self.stats["write_errors"] += 1
                logger.warning("DFlash2 SSD write failed", exc_info=True)
            finally:
                # A daemon waiting in get() otherwise retains the last full
                # snapshot indefinitely through these loop-local references.
                # Release completed payloads before signalling the write done.
                del arrays, meta, parent_receipt
                self._q.task_done()
                if not receipt.cancelled():
                    receipt.set_result(outcome)

    def _parent_durable(self, sig: str, parent_receipt) -> bool:
        if parent_receipt is not None:
            if not parent_receipt.done():
                return False
            try:
                if not (parent_receipt.result() or {}).get("durable"):
                    return False
            except Exception:
                return False
        data, side = self.store.paths(sig)
        return data.is_file() and side.is_file()

    def _touch(self, sig: str) -> None:
        import os

        try:
            for path in self.store.paths(sig):
                os.utime(path, None)
        except OSError:
            pass

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait for queued writes (tests / shutdown)."""
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    def wait_pending_writes(self) -> None:
        """Settle every queued write (tests and shutdown paths)."""
        self._q.join()

    # ---- read ----------------------------------------------------------------
    def take_matching(self, prompt: list[int], *, im_start_id: Optional[int], eos_ids: Iterable[int],
                      make_target: Callable[[], list], make_draft: Callable[[], list], every: int = 0) -> Optional[dict]:
        import mlx.core as mx

        t0 = time.perf_counter()
        for cut in candidate_cuts(prompt, im_start_id, eos_ids, every):
            prefix = prompt[:cut]
            loaded = self.store.load(signature_for(prefix), lambda p: mx.load(str(p), return_metadata=True))
            if loaded is None:
                continue
            arrays, metadata = loaded
            try:
                meta = json.loads(metadata["dflash2"])
                if arrays["tokens"].tolist() != prefix or int(meta["cache_len"]) >= len(prompt):
                    continue
                links = self._load_chain(arrays, meta, prefix)
                if links is None:
                    self.stats["chain_misses"] += 1
                    continue
                target = _unpack_chain("t", "target", make_target, links)
                if target is None:
                    continue
                draft = None
                if meta.get("draft") is not None:
                    if any("delta" in info for info in meta["draft"]):
                        draft = _unpack_chain("d", "draft", make_draft, links)     # DSpark drafter-context delta
                        if draft is None:
                            continue
                    else:
                        draft = _unpack("d", make_draft, arrays, meta["draft"])
                del links
                mx.eval([a for k, a in arrays.items() if not k.startswith("t")])
            except Exception:
                logger.warning("DFlash2 SSD entry unreadable; treating as a miss", exc_info=True)
                continue
            dt = time.perf_counter() - t0
            self.stats["hits"] += 1
            self.stats["hit_tokens"] += int(meta["cache_len"])
            self.stats["load_s"] += dt
            return {"kind": meta["kind"], "tokens": prefix, "cache_len": int(meta["cache_len"]),
                    "rope_delta": meta.get("rope_delta"),
                    "target_cache": target, "draft_cache": draft,
                    "draft_hidden_gap": arrays.get("gap") if draft is not None else None,
                    "draft_context": arrays.get("draft_context"),
                    "load_s": dt,
                    # this entry is on SSD: a cache built on it can be stored as a delta against it
                    "ssd_ref": {"sig": signature_for(prefix), "len": int(meta["cache_len"]),
                                "depth": int((meta.get("parent") or {}).get("depth", 0)),
                                "generation": meta.get("generation")}}
        self.stats["misses"] += 1
        return None

    def _load_chain(self, arrays: dict, meta: dict, prefix: list[int]) -> Optional[list]:
        """Root-first (arrays, meta) links of a (delta) entry; None if any parent is missing, unreadable, not a token
        prefix of this entry, or the chain is longer than its own depth says."""
        import mlx.core as mx

        links = [(arrays, meta)]
        cur = meta
        while cur.get("parent"):
            parent = cur["parent"]
            if len(links) > MAX_CHAIN + 1:
                return None
            n = int(parent["len"])
            key_len = _parent_key_length(parent, prefix)
            if key_len is None or n >= int(cur["cache_len"]):
                return None
            loaded = self.store.load(parent["sig"], lambda p: mx.load(str(p), return_metadata=True))
            if loaded is None:
                return None
            p_arrays, p_metadata = loaded
            p_meta = json.loads(p_metadata["dflash2"])
            if (not parent.get("generation") or p_meta.get("generation") != parent["generation"]
                    or int(p_meta["cache_len"]) != n or p_arrays["tokens"].tolist() != prefix[:key_len]):
                return None
            links.insert(0, (p_arrays, p_meta))
            cur = p_meta
        return links

    def clear(self) -> int:
        # A timed-out writer can still publish its atomic rename. Deleting
        # now would report success and then let that write resurrect an entry.
        if not self.flush(10.0):
            raise TimeoutError("DFlash2 cache clear waiting for pending SSD writes; retry when idle")
        return self.store.clear()


def model_identity(target_path: str, draft_path: str) -> str:
    """Namespace key: both bundles' file fingerprints + runtime packages."""
    from .model_bundle_integrity import _bundle_fingerprint
    from .prefix_cache import runtime_cache_fingerprint

    parts = []
    for label, p in (("target", target_path), ("draft", draft_path)):
        root = Path(p).expanduser().resolve()
        parts.append(f"{label}={root}:{_bundle_fingerprint(root)[:16]}")
    parts.append(runtime_cache_fingerprint())
    return "|".join(parts)
