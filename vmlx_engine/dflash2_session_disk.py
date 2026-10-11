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
SCHEMA = "dflash2_session_v3"
LEGACY_SCHEMAS = ("dflash2_session_v1", "dflash2_session_v2")


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(array("I", tokens).tobytes()).hexdigest()


def signature_for(tokens: list[int]) -> str:
    return f"{len(tokens)}:{_token_digest(tokens)}"


def candidate_cuts(prompt: list[int], im_start_id: Optional[int], eos_ids: Iterable[int]) -> list[int]:
    """Prefix lengths at which a stored entry can end, longest first."""
    eos = {int(t) for t in eos_ids}
    cuts = set()
    for i, t in enumerate(prompt):
        if im_start_id is not None and t == im_start_id and i > 0:
            cuts.add(i)                    # boundary: tokens[:i], i = tag index
        if t in eos:
            cuts.add(i + 1)                # turn: tokens end with the EOS id
    return sorted((c for c in cuts if 0 < c < len(prompt)), reverse=True)


def _has_meta_setter(cache: Any) -> bool:
    prop = getattr(type(cache), "meta_state", None)
    return isinstance(prop, property) and prop.fset is not None


def _pack(prefix: str, caches: list, arrays: dict, layers: list) -> None:
    for i, c in enumerate(caches):
        state = list(c.state) if isinstance(c.state, (list, tuple)) else [c.state]
        nones = []
        for j, a in enumerate(state):
            if a is None:
                nones.append(j)
            else:
                arrays[f"{prefix}{i}.{j}"] = a
        meta = None
        if _has_meta_setter(c):
            m = c.meta_state
            meta = list(m) if isinstance(m, (list, tuple)) else m
        layers.append({"cls": type(c).__name__, "n": len(state), "none": nones,
                       "tuple": isinstance(c.state, (list, tuple)), "meta": meta})


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
        self.stats = {"writes": 0, "write_bytes": 0, "write_s": 0.0, "write_errors": 0,
                      "dropped": 0, "hits": 0, "hit_tokens": 0, "load_s": 0.0, "misses": 0}
        # Two prefill snapshots plus a reserved final-turn slot. Requests
        # settle their terminal receipt before releasing the generation lock.
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=self.queue_depth + 1)
        self._worker = threading.Thread(target=self._run, name="dflash2-ssd-writer", daemon=True)
        self._worker.start()

    # ---- write ---------------------------------------------------------------
    def put(self, entry: dict) -> Future:
        """Snapshot ``entry`` (a RAM-store dict) and queue it for SSD."""
        import mlx.core as mx

        arrays: dict = {}
        target_layers: list = []
        _pack("t", entry["target_cache"], arrays, target_layers)
        draft_layers = None
        if entry.get("draft_cache") is not None:
            # A drafter cache that never ran (e.g. a max_tokens=1 request) has empty layers whose
            # `state` raises; store the entry without drafter state (it is rebuilt on resume).
            try:
                d_arrays: dict = {}
                d_layers: list = []
                _pack("d", entry["draft_cache"], d_arrays, d_layers)
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
        meta = {"kind": entry.get("kind", "turn"), "cache_len": int(entry["cache_len"]),
                "rope_delta": entry.get("rope_delta"),
                "target": target_layers, "draft": draft_layers}
        receipt = Future()
        # Cancelling an HTTP consumer must not cancel an already-owned write.
        receipt.set_running_or_notify_cancel()
        try:
            if meta["kind"] != "turn" and self._q.qsize() >= self.queue_depth:
                raise queue.Full
            self._q.put_nowait((signature_for(tokens), arrays, meta, receipt))
        except queue.Full:
            self.stats["dropped"] += 1
            logger.info("DFlash2 SSD write skipped (writer busy); entry stays in RAM only")
            receipt.set_result({"outcome": "refused", "durable": False,
                                "retained_tokens": 0, "detail": "DFlash2 SSD writer queue full"})
        return receipt

    def _run(self) -> None:
        import mlx.core as mx

        while True:
            sig, arrays, meta, receipt = self._q.get()
            t0 = time.perf_counter()
            outcome = {"outcome": "failed", "durable": False,
                       "retained_tokens": 0, "detail": "DFlash2 SSD write failed"}
            try:
                path = self.store.save(
                    sig,
                    lambda tmp: mx.save_safetensors(str(tmp), arrays, {"dflash2": json.dumps(meta)}),
                )
                self.stats["writes"] += 1
                self.stats["write_bytes"] += path.stat().st_size
                self.stats["write_s"] += time.perf_counter() - t0
                logger.info("DFlash2 SSD stored %s entry: %s tokens, %.0f MB, %.2fs",
                            meta["kind"], sig.split(":")[0], path.stat().st_size / 1e6,
                            time.perf_counter() - t0)
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
                del arrays, meta
                self._q.task_done()
                if not receipt.cancelled():
                    receipt.set_result(outcome)

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait for queued writes (tests / shutdown)."""
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    def wait_pending_writes(self, backlog_only: bool = False) -> None:
        """Settle writes left by a disconnected request before the next one.

        backlog_only: wait only while snapshots are still queued behind the
        writer (so the next request's own snapshots are not refused for a full
        queue); a single write already in progress does not hold the request."""
        if backlog_only and self._q.qsize() == 0:
            return
        self._q.join()

    # ---- read ----------------------------------------------------------------
    def take_matching(self, prompt: list[int], *, im_start_id: Optional[int], eos_ids: Iterable[int],
                      make_target: Callable[[], list], make_draft: Callable[[], list]) -> Optional[dict]:
        import mlx.core as mx

        t0 = time.perf_counter()
        for cut in candidate_cuts(prompt, im_start_id, eos_ids):
            prefix = prompt[:cut]
            loaded = self.store.load(signature_for(prefix), lambda p: mx.load(str(p), return_metadata=True))
            if loaded is None:
                continue
            arrays, metadata = loaded
            try:
                meta = json.loads(metadata["dflash2"])
                if arrays["tokens"].tolist() != prefix or int(meta["cache_len"]) >= len(prompt):
                    continue
                target = _unpack("t", make_target, arrays, meta["target"])
                if target is None:
                    continue
                draft = None
                if meta.get("draft") is not None:
                    draft = _unpack("d", make_draft, arrays, meta["draft"])
                mx.eval([a for a in arrays.values()])
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
                    "load_s": dt}
        self.stats["misses"] += 1
        return None

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
