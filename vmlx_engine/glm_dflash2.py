"""DFlash2 speculative decoding for GLM-5.3-Flash (glm5_next) targets (2026-10-09).

Drafter: incoai/GLM-5.3-Flash-DFlash2 (z-lab DFlash2: 5 sliding-window qwen3 layers, block 8, two-tap grouped dynamic
conv, candidate selector top-16), loaded with ``dflash.model_mlx.load_draft`` from its ORIGINAL BF16 files and
optionally affine-8 quantized at load (``quantize_draft``); it shares the target's embedding and lm_head (``bind``).

GLM target contract (SGLang PR 36708 ``_prepare_aux_hidden_state`` + its unit test): the drafter's context taps are the
COMPLETED outputs of target layers ``dflash_config.target_layer_ids`` = [5, 14, 24, 33, 42], each contracted over the
4 mHC streams by an unweighted mean (the same contraction GLM applies before its final norm).

Verify / rollback: the verify forward runs [last committed token] + drafts with ``n_confirmed=1``, so every KDA layer
records the exact recurrent + conv state after each position (``Glm5KDACache.set_speculative_states``) and a partial
rejection restores the state after the accepted prefix (``rollback_speculative(rejected)``); MLA/DSA caches trim
(an incomplete indexer pool is dropped and recomputed from the packed history on the next append).
Cache invariant (as upstream dflash): the last emitted token is sampled but never forwarded.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn


class _MeanStreamTap:
    """Wraps one decoder layer; when armed, stores mean over the mHC stream axis of its output: (B, T, 4, D) -> (B, T, D)."""

    def __init__(self, layer, slot: int, store: list, armed: dict):
        self._layer, self._slot, self._store, self._armed = layer, slot, store, armed

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        if self._armed["on"]:
            self._store[self._slot] = mx.mean(out, axis=2)
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


class GLMDFlashTarget:
    """Speculative-decoding view of a loaded glm5_next model (VLM wrapper or text Model)."""

    def __init__(self, model, layer_ids):
        lm = model.language_model if hasattr(model, "language_model") else model
        self.lm, self.inner = lm, lm.model
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self._store: list = [None] * len(self.layer_ids)
        self._armed = {"on": False}
        layers = self.inner.layers
        for slot, lid in enumerate(self.layer_ids):
            cur = layers[lid]
            if isinstance(cur, _MeanStreamTap):        # re-wrap safely (adapter rebuilt for the same model)
                cur = cur._layer
            layers[lid] = _MeanStreamTap(cur, slot, self._store, self._armed)

    def close(self):
        layers = self.inner.layers
        for lid in self.layer_ids:
            if isinstance(layers[lid], _MeanStreamTap):
                layers[lid] = layers[lid]._layer

    # dflash.bind() looks for .model.embed_tokens and .lm_head
    @property
    def model(self):
        return self.inner

    @property
    def lm_head(self):
        return self.lm.lm_head

    def make_cache(self):
        return self.lm.make_cache()

    def forward(self, ids: mx.array, cache, verify: bool, taps: bool = True, last_row: bool = False,
                inputs_embeds: mx.array | None = None):
        self._armed["on"] = taps
        try:
            h = self.inner(ids, cache=cache, n_confirmed=1 if (verify and ids.shape[1] > 1) else 0, return_pre_norm=True,
                           inputs_embeds=inputs_embeds)
        finally:
            self._armed["on"] = False
        if last_row:
            h = h[:, -1:]
        return self.lm.lm_head(self.inner.norm(h))

    def hidden(self) -> mx.array:
        return mx.concatenate(self._store, axis=-1)

    @staticmethod
    def rollback(cache, rejected: int) -> None:
        from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache

        for c in cache:
            if isinstance(c, Glm5KDACache):
                if rejected > 0:
                    if not c.rollback_speculative(rejected):
                        raise RuntimeError("glm dflash2: KDA speculative rollback failed (no recorded states)")
                else:
                    c.commit_speculative()
            elif rejected > 0:
                c.trim(rejected)


def load_glm_drafter(path: str, quantize_bits: int = 8, group_size: int = 64):
    """incoai DFlash2 drafter from its original files; affine `quantize_bits` at load (0 = keep BF16)."""
    from pathlib import Path

    import dflash.model_mlx as runtime

    original = runtime.snapshot_download
    if Path(path).expanduser().is_dir():                   # local files: never resolve as a Hub repo id
        runtime.snapshot_download = lambda model_id, **_kw: model_id
    try:
        draft = runtime.load_draft(str(Path(path).expanduser()))
    finally:
        runtime.snapshot_download = original
    if quantize_bits:
        quantize_draft(draft, quantize_bits, group_size)
    return draft


def quantize_draft(draft, bits: int = 8, group_size: int = 64) -> int:
    """Affine-quantize the drafter's Linear layers (attention, MLP, fc, selector projection) in place; the selector
    codebooks (embeddings) and the dynamic-conv parameters stay BF16. Returns the number of quantized modules."""
    count = {"n": 0}

    def pred(path, m):
        ok = isinstance(m, nn.Linear) and m.weight.shape[-1] % group_size == 0
        count["n"] += int(ok)
        return ok
    nn.quantize(draft, group_size=group_size, bits=bits, class_predicate=pred)
    mx.eval(draft.parameters())
    return count["n"]


@dataclass
class SpecStats:
    prompt_tokens: int = 0
    prefill_s: float = 0.0
    tokens: int = 0
    decode_s: float = 0.0
    rounds: int = 0
    accepted: list = field(default_factory=list)
    ar_steps: int = 0                       # adaptive mode: tokens produced by plain target steps
    cached_tokens: int = 0                  # prompt tokens restored from the lane's prefix store
    cache_source: str = ""                  # "ram" | "disk" for a restored prefix (usage cache_detail)
    vision_tower: bool = False              # an image plan's embeddings were materialized for this prefill
    prefill_compute_s: float = 0.0          # suffix prefill + prompt state + first sample, after lookup/restore
    blocks: list = field(default_factory=list)   # block size used by each verify round
    output_token_ids: list = field(default_factory=list)

    @property
    def tok_per_round(self) -> float:
        return (sum(self.accepted) + len(self.accepted)) / max(1, len(self.accepted))

    @property
    def decode_tps(self) -> float:
        return self.tokens / max(self.decode_s, 1e-9)


def _prefill(target: GLMDFlashTarget, prompt: mx.array, cache, step: int, hidden_limit: int | None, taps: bool = True,
             embeds: mx.array | None = None):
    """embeds: (1, prompt.size, D) merged input embeddings for this segment (image features at their placeholders),
    or None to embed the token ids."""
    chunks, start, logits = [], 0, None
    from vmlx_engine.utils.generation_cancellation import raise_if_generation_cancelled

    while start < prompt.size:
        raise_if_generation_cancelled()          # cooperative cancel at each prefill chunk (SimpleEngine scope)
        rem = prompt.size - start
        end = start + (1 if rem == 1 else min(step, rem - 1)) if taps else start + min(step, rem)
        logits = target.forward(prompt[None, start:end], cache, verify=False, taps=taps, last_row=True,
                                inputs_embeds=None if embeds is None else embeds[:, start:end])
        if taps:
            h = target.hidden()
            if hidden_limit is not None:
                if chunks:
                    h = mx.concatenate((chunks[0], h), axis=1)
                chunks = [h[:, -hidden_limit:]]
            else:
                chunks.append(h)
        mx.eval([c.state for c in cache], *(chunks[-1:] if chunks else []))
        start = end
    hidden = None
    if taps:
        hidden = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)
    return logits, hidden


def generate_ar(target: GLMDFlashTarget, prompt_ids, max_tokens: int, eos: set, temperature: float = 0.0,
                top_p: float = 1.0, prefill_step: int = 2048, stats: SpecStats | None = None) -> Iterator[int]:
    """Plain autoregressive control on the SAME forward path (no taps), for in-process A/B."""
    from dflash.model_mlx import make_sampler

    sampler = make_sampler(temperature, top_p, 0)
    prompt = mx.array(prompt_ids); cache = target.make_cache(); st = stats or SpecStats()
    t0 = time.perf_counter()
    logits, _ = _prefill(target, prompt, cache, prefill_step, None, taps=False)
    tok = sampler(logits[:, -1:]); mx.eval(tok)
    st.prompt_tokens, st.prefill_s = prompt.size, time.perf_counter() - t0
    t1 = time.perf_counter(); n = 0
    from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
    # pipelined like the served loop: the next step's graph is built (inside the productive-AR scope, which selects the
    # AR-only kernels) and queued BEFORE the current token is read back
    while True:
        if n + 1 < max_tokens:
            with affine_moe_ar_scope():
                logits = target.forward(tok.reshape(1, 1), cache, verify=False, taps=False)
            nxt = sampler(logits[:, -1:]); mx.async_eval(nxt)
        t = int(tok.item()); n += 1; yield t
        if t in eos or n >= max_tokens:
            break
        tok = nxt
    st.tokens, st.decode_s = n, time.perf_counter() - t1


class _PrefixSnapshot:
    __slots__ = ("cache", "hidden", "logits", "receipt", "parent", "depth")

    def __init__(self, cache, hidden, logits):
        self.cache, self.hidden, self.logits = cache, hidden, logits
        self.receipt = None                     # SSD write receipt (concurrent Future) while the copy is in flight
        self.parent = None                      # delta snapshots: {"name", "n", "depth"} of the snapshot they extend
        self.depth = 0


# Delta snapshots (2026-10-12). The lane snapshots the prompt every PREFIX_SNAPSHOT_EVERY tokens, at the stable root
# and at the prompt end, and every snapshot held the WHOLE MLA prefix: GLM-5.3 JANGHT2.4 wrote 35 GB for one cold
# 64.7k-token prompt (33 snapshots; ~10.6 KB/token of MLA, plus per snapshot 147.5 MB of KDA state + conv tails and
# 83.9 MB of drafter context = 2,047 rows x 5 tapped layers x 4096 bf16) and 0.63 GB for a 35-token continuation. The MLA layers' state is append-only along each slot's token axis, so a snapshot that extends the
# request's previous snapshot (or the one it restored) stores only the new rows of every MLA slot, COPIED (a view would
# pin the live buffer), and the KDA layers whole. ``parent`` names that snapshot; a restore loads the chain root first,
# compares every link's token ids, and concatenates each slot. A missing parent is a miss, never wrong state; the
# writer refuses a child whose parent did not become durable; every GLM_PREFIX_MAX_CHAIN links a whole snapshot is
# written. Used with an SSD tier under the SSD-only policy (no RAM copies to keep whole).
GLM_PREFIX_MAX_CHAIN = max(1, int(os.environ.get("VMLX_GLM_DFLASH2_PREFIX_CHAIN", "16")))


def _mla_seq_axes(absorbed: bool) -> tuple:
    """Token axis of each Glm5MLACache state slot (None = not sequence-shaped)."""
    # absorbed: latent [B,1,T,R], packed DSA history [B,T,2D], pool keys [B,T//kpool,D], reserved (empty)
    # materialized: keys/values [B,H,T,D], packed [B,T,.], pool keys [B,T//kpool,D]
    return (2, 1, 1, None) if absorbed else (2, 2, 1, 1)


def _layer_seq_lens(c) -> "list[int] | None":
    """Per-slot logical lengths of an MLA cache (KDA layers: None)."""
    if type(c).__name__ != "Glm5MLACache":
        return None
    axes = _mla_seq_axes(bool(getattr(c, "absorbed", False)))
    state = list(c.state)
    return [0 if (a is None or v is None) else int(v.shape[a]) for v, a in zip(state, axes)]


class _MLADelta:
    """Disk-only stand-in for one Glm5MLACache snapshot holding only the rows its parent snapshot lacks."""

    type_name = "Glm5MLACache"
    __slots__ = ("state", "meta_state", "delta")

    def __init__(self, state, meta_state, delta):
        self.state, self.meta_state, self.delta = state, meta_state, delta


def ssd_only_ram_release_enabled() -> bool:
    """SSD caching means RAM holds the model, not the conversation (2026-10-12).

    With an SSD tier configured, a RAM snapshot is kept only until its SSD copy is durable, then dropped; a later
    turn restores from SSD. Before this the GLM lane kept up to 4 snapshots (3 GiB cap) and the generic DFlash2 lane
    up to 4 (no byte cap) resident while idle: GLM-5.3 JANGHT2.4 sat at +7 GiB above its weights after a 17k chat.
    VMLX_SSD_ONLY_RAM_RELEASE=0 keeps the RAM copies (A/B and diagnostics)."""
    return os.environ.get("VMLX_SSD_ONLY_RAM_RELEASE", "1").strip().lower() not in ("0", "false", "no", "off")


def _receipt_durable(receipt) -> bool:
    if receipt is None or not receipt.done():
        return False
    try:
        result = receipt.result()
    except Exception:
        return False
    return bool(isinstance(result, dict) and result.get("durable"))


def _detached_copy(v):
    return mx.logical_and(v, True) if v.dtype == mx.bool_ else v * 1      # a new buffer, logical size only


def _clone_caches(cache, detach_mla: bool = False):
    """Isolated GLM native-state copies (the prefix caches' clone contract).

    detach_mla=True (snapshots taken DURING a live prefill): MLA state is really copied at its logical size. Sharing
    storage would mark the live cache's preallocated capacity buffers copy-on-write, and its next append then copies the
    whole buffer: boundary snapshots every 2048 tokens made an 8k cold prefill 15.7 -> 27.7 s (2026-10-10). KDA state is
    replaced functionally on every step, so it is shared safely either way."""
    from vmlx_engine.models.glm5_next.glm5_next import Glm5MLACache, clone_glm5_next_layer_cache
    out = []
    for c in cache:
        if detach_mla and isinstance(c, Glm5MLACache):
            out.append(clone_glm5_next_layer_cache(c, copy_fn=_detached_copy, copy_is_detached=True))
        else:
            out.append(clone_glm5_next_layer_cache(c, copy_fn=lambda v: v))
    return out


def _cache_nbytes(cache) -> int:
    return sum(int(v.nbytes) for c in cache for v in list(c.state) if v is not None)


class PrefixStore:
    """In-RAM prompt-prefix snapshots for the GLM DFlash2 lane (2026-10-10).

    The lane rebuilt the whole history every turn (served multi-turn: 18 s TTFT at 8k tokens, while the batched AR
    lane restored its prefix cache in 0.3 s). After each prompt prefill this keeps the target's KDA/MLA caches, the
    drafter's context rows (its sliding window of tapped hidden states) and the last logits, keyed by the prompt token
    ids. A request whose prompt starts with a stored prompt restores that state and prefills only the suffix. The
    drafter cache itself is rebuilt from the context rows exactly as after a cold prefill, so no drafter state is kept.
    Also snapshots every PREFIX_SNAPSHOT_EVERY (2048) absolute prompt tokens during prefill, for mid-history
    divergence. LRU over `max_entries` snapshots (VMLX_GLM_DFLASH2_PREFIX_ENTRIES, default 4) and a total cap
    (VMLX_GLM_DFLASH2_PREFIX_MAX_GB, default 3; shared KDA arrays are counted per snapshot, so this over-counts)."""

    def __init__(self, max_entries: int = 4, max_bytes: float = 3 * 2 ** 30, disk: "PrefixDiskTier | None" = None):
        from collections import OrderedDict
        self._chain = None                      # delta snapshots: the snapshot this request's cache extends
        self.delta_writes = 0
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = float(max_bytes)
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        self.ram_released = 0
        self.disk = disk
        self.last_persistence = None
        self.last_source = ""                   # "ram" | "disk" | "" (miss) for the latest lookup

    def release_durable_ram(self) -> tuple[int, int]:
        """Model thread only. Drop RAM snapshots whose SSD copy is durable (SSD-only policy).

        Returns (released, still_pending): pending = SSD writes not finished yet, whose RAM copy is still needed."""
        if self.disk is None or not ssd_only_ram_release_enabled():
            return 0, 0
        released = pending = 0
        for key in [k for k, snap in self.entries.items() if snap.receipt is not None]:
            snap = self.entries[key]
            if _receipt_durable(snap.receipt):
                del self.entries[key]
                released += 1
            elif not snap.receipt.done():
                pending += 1
        self.ram_released += released
        return released, pending

    def _delta_mode(self) -> bool:
        return self.disk is not None and ssd_only_ram_release_enabled()

    def _note_chain(self, key, snap) -> None:
        self._chain = {"name": PrefixDiskTier.digest(key), "n": len(key), "depth": int(getattr(snap, "depth", 0) or 0),
                       "lens": [_layer_seq_lens(c) for c in snap.cache], "receipt": getattr(snap, "receipt", None)}

    def lookup(self, ids):
        self.release_durable_ram()
        self._chain = None                      # a new request: its cache extends what this lookup restores
        best = None
        for key in self.entries:
            n = len(key)
            if n <= len(ids) and (best is None or n > len(best)) and tuple(ids[:n]) == key:
                best = key
        if self.disk is not None:
            # a longer prefix on SSD (written by an earlier server process, or evicted from RAM) beats a RAM hit
            found = self.disk.lookup(ids, min_len=len(best) if best is not None else 0)
            if found is not None:
                n, snap = found
                key = tuple(int(t) for t in ids[:n])
                if not ssd_only_ram_release_enabled():      # SSD-only: a snapshot read from SSD is not kept in RAM
                    self.entries[key] = snap
                    self._evict()
                    if key in self.entries:
                        self.entries.move_to_end(key)
                self.hits += 1
                self.last_source = "disk"
                self._note_chain(key, snap)
                return n, snap
        if best is None:
            self.misses += 1
            self.last_source = ""
            return None
        self.entries.move_to_end(best); self.hits += 1
        self.last_source = "ram"
        self._note_chain(best, self.entries[best])
        return len(best), self.entries[best]

    def _delta_snapshot(self, key, cache, hidden, logits):
        """A snapshot of ``cache`` storing, per MLA layer, only the rows past the chain parent (copied); None when it
        cannot extend the parent (no parent, chain too deep, not a token prefix, or a slot did not grow)."""
        chain = self._chain
        if (
            chain is None
            or not 0 < int(chain["n"]) < len(key)
            or int(chain["depth"]) + 1 > GLM_PREFIX_MAX_CHAIN
            or PrefixDiskTier.digest(key[: int(chain["n"])]) != chain["name"]
            or len(chain["lens"]) != len(cache)
        ):
            return None
        receipt = chain.get("receipt")
        if receipt is not None and receipt.done() and not _receipt_durable(receipt):
            return None
        from vmlx_engine.models.glm5_next.glm5_next import clone_glm5_next_layer_cache
        layers = []
        for c, base in zip(cache, chain["lens"]):
            if type(c).__name__ != "Glm5MLACache":
                layers.append(clone_glm5_next_layer_cache(c, copy_fn=lambda v: v))   # KDA: whole (immutable arrays)
                continue
            cur = _layer_seq_lens(c)
            if base is None or len(base) != len(cur) or any(b > n for b, n in zip(base, cur)):
                return None
            axes = _mla_seq_axes(bool(getattr(c, "absorbed", False)))
            state = []
            for v, a, b in zip(list(c.state), axes, base):
                if v is None or a is None:
                    state.append(v)
                    continue
                index = [slice(None)] * v.ndim
                index[a] = slice(int(b), None)
                state.append(_detached_copy(v[tuple(index)]))
            layers.append(_MLADelta(state, c.meta_state, list(base)))
        snap = _PrefixSnapshot(layers, hidden, logits)
        snap.parent = {"name": chain["name"], "n": int(chain["n"]), "depth": int(chain["depth"])}
        snap.depth = int(chain["depth"]) + 1
        return snap

    def put(self, ids, cache, hidden, logits):
        key = tuple(int(t) for t in ids)
        if self._delta_mode():
            # SSD-only + SSD tier: no RAM copy at all; the snapshot goes to SSD (a delta when it extends the chain).
            snap = self._delta_snapshot(key, cache, hidden, logits)
            parent_receipt = (self._chain or {}).get("receipt") if snap is not None else None
            if snap is None:
                snap = _PrefixSnapshot(_clone_caches(cache, detach_mla=True), hidden, logits)
            else:
                self.delta_writes += 1
            mx.eval([v for c in snap.cache for v in list(c.state) if v is not None])
            self.last_persistence = self.disk.put(key, snap, parent_receipt=parent_receipt)
            snap.receipt = self.last_persistence
            lens = [_layer_seq_lens(c) for c in cache]
            self._chain = {"name": PrefixDiskTier.digest(key), "n": len(key), "depth": snap.depth, "lens": lens,
                           "receipt": self.last_persistence}
            return
        snap = _PrefixSnapshot(_clone_caches(cache, detach_mla=True), hidden, logits)
        mx.eval([c.state for c in snap.cache])
        self.entries[key] = snap
        self.entries.move_to_end(key)
        self._evict()
        if self.disk is not None:
            self.last_persistence = self.disk.put(key, snap)
            snap.receipt = self.last_persistence
            self.release_durable_ram()

    def _evict(self):
        while len(self.entries) > self.max_entries or (len(self.entries) > 1 and self.nbytes() > self.max_bytes):
            self.entries.popitem(last=False)
            self.evictions += 1

    def nbytes(self) -> int:
        return sum(_cache_nbytes(e.cache) for e in self.entries.values())


class PrefixDiskTier:
    """SSD tier of the GLM DFlash2 prefix store (2026-10-10): every snapshot the RAM store keeps is also written to
    `root` by a background thread (one safetensors file per snapshot: each layer's native cache state + typed
    meta_state, the drafter context rows, the last logits and the prompt ids), so a restarted server restores long
    histories instead of re-prefilling them. A RAM miss looks here for the longest stored prefix; the file's ids are
    compared token by token before use. Each bundle has a namespace within the shared pool;
    aggregate LRU and process-safe publication are owned by its budget coordinator."""

    FORMAT = "glm-dflash2-prefix-v2"      # v2: delta snapshots (layers carry their delta bases, meta a parent)

    def __init__(self, root, max_bytes: float, model_key: str = "glm-prefix"):
        import math
        import queue
        import threading
        from .utils.glm_prefix_disk_store import GlmPrefixDiskStore

        self.max_bytes = float(max_bytes)
        self._catalog = GlmPrefixDiskStore(
            root, model_key, int(max_bytes) if math.isfinite(max_bytes) else 0,
        )
        self.dir = self._catalog.directory
        self.hits = self.misses = self.writes = self.evictions = self.errors = 0
        self.chain_misses = 0
        self._lock = threading.Lock()
        self.index: dict[str, dict] = {}
        self._load_index()
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="glm-dflash2-prefix-disk", daemon=True)
        self._thread.start()

    @staticmethod
    def digest(ids) -> str:
        import hashlib

        import numpy as np
        return hashlib.sha256(np.asarray(list(ids), dtype=np.int64).tobytes()).hexdigest()[:40]

    def _load_index(self):
        records = self._catalog.records()
        with self._lock:
            self.index = records

    def nbytes(self) -> int:
        with self._lock:
            return sum(e["bytes"] for e in self.index.values())

    def put(self, key: tuple, snap: "_PrefixSnapshot", parent_receipt=None):
        from concurrent.futures import Future

        receipt = Future()
        name = self.digest(key)
        with self._lock:
            known = name in self.index
        # Another pool owner can evict a previously indexed snapshot. Confirm
        # the finalized pair under the shared lock before claiming durability.
        try:
            retained = known and self._catalog.contains(name, len(key))
        except (OSError, ValueError, KeyError, TypeError):
            retained = False
        if retained:
            with self._lock:
                if name in self.index:
                    self.index[name]["used"] = time.time()
            receipt.set_result({"outcome": "already_durable", "durable": True,
                                "retained_tokens": len(key), "detail": "glm-dflash2-prefix-disk"})
            return receipt
        self._queue.put((name, key, snap, receipt, parent_receipt))
        return receipt

    def flush(self) -> None:
        self._queue.join()

    def clear(self) -> int:
        """Clear this bundle's indexed snapshots while serving is excluded."""
        self.flush()  # A queued write must not recreate an entry after Clear.
        removed = self._catalog.clear()
        with self._lock:
            self.index.clear()
        return removed

    def close(self) -> None:
        """Drain queued snapshots and release the writer at model teardown.

        The serving lifecycle has already stopped generation before calling
        this method. Normal generation continues to enqueue without waiting.
        """
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self._catalog.close()

    def _writer(self):
        while True:
            item = self._queue.get()
            receipt = None
            try:
                if item is None:
                    return
                name, key, snap, receipt = item[:4]
                parent_receipt = item[4] if len(item) > 4 else None
                parent = getattr(snap, "parent", None)
                if parent is not None and not self._parent_durable(parent, parent_receipt):
                    # FIFO: the parent was queued first and is settled now. A child of a failed or evicted parent
                    # would restore nothing: refuse it.
                    raise OSError("parent prefix snapshot is not durable")
                parent_receipt = None
                result = self._write(name, key, snap)
                # Drop the snapshot before publishing durability: once the receipt resolves, the RAM store releases its
                # copy (SSD-only policy) and that must free the arrays. Loop-locals held here while blocked in get()
                # otherwise kept the LAST snapshot resident until the next write (GLM-5.3: +1.7-2.8 GiB at idle,
                # growing with context); the generic DFlash2 writer drops its payload the same way.
                item = snap = key = None
                receipt.set_result(result)
            except Exception as e:                         # never fail a generation over the SSD tier
                self.errors += 1
                import logging
                logging.getLogger(__name__).warning("glm dflash2 prefix disk write failed: %s", e)
                item = snap = key = None
                if receipt is not None and not receipt.done():
                    receipt.set_result({"outcome": "failed", "durable": False,
                                        "retained_tokens": 0, "detail": str(e)[:200]})
            finally:
                item = snap = key = None
                self._queue.task_done()

    def _parent_durable(self, parent: dict, parent_receipt) -> bool:
        if parent_receipt is not None:
            if not parent_receipt.done() or not _receipt_durable(parent_receipt):
                return False
        try:
            return bool(self._catalog.contains(parent["name"], int(parent["n"])))
        except Exception:
            return False

    def _write(self, name, key, snap):
        import json
        # int64: image prefixes are keyed with per-item ids above the int32 range (_dflash2_rounds key_ids).
        arrays = {"ids": mx.array(list(key), dtype=mx.int64), "hidden": snap.hidden, "logits": snap.logits}
        layers = []
        for i, c in enumerate(snap.cache):
            state = list(c.state)
            layers.append([getattr(c, "type_name", type(c).__name__), [str(m) for m in c.meta_state],
                           [v is not None for v in state], getattr(c, "delta", None)])
            for j, v in enumerate(state):
                if v is not None:
                    arrays[f"c{i}.{j}"] = v
        parent = getattr(snap, "parent", None)
        self._catalog.save(name, len(key), lambda tmp: mx.save_safetensors(
            str(tmp), arrays, metadata={"format": self.FORMAT, "layers": json.dumps(layers),
                                        "parent": json.dumps(parent)},
        ))
        self._load_index()
        with self._lock:
            self.writes += 1
            retained = name in self.index
            return {"outcome": "stored" if retained else "refused", "durable": retained,
                    "retained_tokens": len(key) if retained else 0,
                    "detail": "glm-dflash2-prefix-disk" if retained else "snapshot not retained by shared SSD budget"}

    def lookup(self, ids, min_len: int = 0):
        try:
            self._load_index()
        except OSError as e:
            import logging
            logging.getLogger(__name__).warning("glm dflash2 prefix catalog unavailable: %s", e)
            self.errors += 1
            self.misses += 1
            return None
        with self._lock:
            cands = sorted({e["n"] for e in self.index.values() if min_len < e["n"] <= len(ids)}, reverse=True)
            by_name = dict(self.index)
        for n in cands:
            name = self.digest(ids[:n])
            if name not in by_name:
                continue
            snap = self._read(name, ids[:n])
            if snap is not None:
                with self._lock:
                    if name in self.index:
                        self.index[name]["used"] = time.time()
                self.hits += 1
                return n, snap
        self.misses += 1
        return None

    def _read_link(self, name, ids, leaf: bool = True):
        """One materialized link: (arrays, layers, parent) of snapshot ``name`` for exactly ``ids``, + its file identity.

        A parent link (``leaf=False``) contributes only its MLA token rows to the assembled snapshot: its KDA state,
        drafter context and logits are superseded by the leaf's. Those arrays are dropped unread (``mx.load`` is lazy, so
        an array that is never evaluated is never read from the file) — on GLM-5.3 every link carries ~232 MB of KDA
        state and drafter context, and materializing it for each of ~15 parents made a 65k chained restore read
        several GB it then discarded."""
        import json

        seen = {}

        def read(path):
            seen["identity"] = self._catalog.file_identity(path)
            arrays, meta = mx.load(str(path), return_metadata=True)
            if meta.get("format") != self.FORMAT or arrays["ids"].tolist() != [int(t) for t in ids]:
                raise ValueError("prefix snapshot does not match")
            layers = json.loads(meta["layers"])
            if not leaf:
                keep = set()
                for i, entry in enumerate(layers):
                    if entry[0] != "Glm5MLACache":
                        continue
                    absorbed = tuple(entry[1][:2]) == ("glm5_next_native_v2", "mla_absorbed")
                    keep.update(f"c{i}.{j}" for j, axis in enumerate(_mla_seq_axes(absorbed))
                                if axis is not None and entry[2][j])
                arrays = {k: v for k, v in arrays.items() if k in keep}
            # MLX loads lazily. Keep the eviction lock through materialization,
            # not merely until the file's header has been read.
            mx.eval(list(arrays.values()))
            return arrays, layers, json.loads(meta.get("parent") or "null")

        try:
            return self._catalog.load(name, len(ids), read), seen.get("identity")
        except Exception as e:
            e.vmlx_failed_identity = seen.get("identity")   # the caller removes the unreadable file
            raise

    @staticmethod
    def _assemble(links):
        """Root-first links -> a _PrefixSnapshot: MLA slots concatenated along their token axes (each delta must start
        exactly where the chain so far ends), every other layer whole from the leaf."""
        from vmlx_engine.models.glm5_next import glm5_next as G
        classes = {"Glm5KDACache": G.Glm5KDACache, "Glm5MLACache": G.Glm5MLACache}
        leaf_arrays, leaf_layers, _ = links[-1]
        if any(len(layers) != len(leaf_layers) for _, layers, _ in links):
            raise ValueError("prefix snapshot chain layer count differs")
        cache = []
        for i, entry in enumerate(leaf_layers):
            cls_name, meta_state, present = entry[0], entry[1], entry[2]
            cls = classes[cls_name]
            chained = cls_name == "Glm5MLACache" and any(layers[i][3] is not None for _, layers, _ in links[1:])
            if not chained:
                state = [leaf_arrays[f"c{i}.{j}"] if p else None for j, p in enumerate(present)]
                cache.append(cls.from_state(state, tuple(meta_state)))
                continue
            if links[0][1][i][3] is not None or any(layers[i][0] != cls_name for _, layers, _ in links):
                raise ValueError("prefix snapshot chain does not start from a whole MLA layer")
            absorbed = tuple(meta_state[:2]) == ("glm5_next_native_v2", "mla_absorbed")
            axes = _mla_seq_axes(absorbed)
            state = []
            for j, axis in enumerate(axes):
                if axis is None:
                    state.append(leaf_arrays[f"c{i}.{j}"] if present[j] else None)
                    continue
                pieces, length = [], 0
                for k, (arrays, layers, _) in enumerate(links):
                    layer = layers[i]
                    if k > 0 and layer[3] is not None and int(layer[3][j]) != length:
                        raise ValueError("prefix snapshot delta does not continue its parent")
                    if layer[2][j]:
                        piece = arrays[f"c{i}.{j}"]
                        pieces.append(piece)
                        length += int(piece.shape[axis])
                state.append(None if not pieces else pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=axis))
            cache.append(cls.from_state(state, tuple(meta_state)))      # validates every length against the offset
        mx.eval([v for c in cache for v in list(c.state) if v is not None])
        snap = _PrefixSnapshot(cache, leaf_arrays["hidden"], leaf_arrays["logits"])
        snap.depth = len(links) - 1
        return snap

    def _read(self, name, ids):
        failed = None                                  # (name, n, identity) of the link that could not be used
        try:
            links = []
            cur_name, cur_ids = name, [int(t) for t in ids]
            while True:
                failed = (cur_name, len(cur_ids), None)
                link, identity = self._read_link(cur_name, cur_ids, leaf=not links)
                if link is None:
                    if links:
                        self.chain_misses += 1        # parent gone (evicted): the leaf alone restores nothing
                    return None
                failed = (cur_name, len(cur_ids), identity)
                links.insert(0, link)
                parent = link[2]
                if not parent:
                    break
                if len(links) > GLM_PREFIX_MAX_CHAIN + 1:
                    raise ValueError("prefix snapshot chain is longer than its limit")
                n = int(parent["n"])
                if not 0 < n < len(cur_ids):
                    raise ValueError("prefix snapshot parent length is invalid")
                cur_ids = cur_ids[:n]
                cur_name = str(parent["name"])
                if self.digest(cur_ids) != cur_name:
                    raise ValueError("prefix snapshot parent is not a prefix of it")
            return self._assemble(links)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("glm dflash2 prefix snapshot %s unusable (%s); ignored", name, e)
            self.errors += 1
            if failed is not None:
                bad_name, bad_n, identity = failed
                identity = identity or getattr(e, "vmlx_failed_identity", None)
                if identity is not None:
                    try:
                        self._catalog.remove(bad_name, bad_n, identity)
                    except OSError:
                        logging.getLogger(__name__).warning("glm dflash2 failed snapshot cleanup deferred", exc_info=True)
            return None


_PREFIX_STORES: dict = {}
PREFIX_SNAPSHOT_EVERY = int(os.environ.get("VMLX_GLM_DFLASH2_PREFIX_EVERY", "2048"))
PREFIX_RAM_ENTRIES_DEFAULT = 4
_PREFIX_DISK = {"root": None, "max_bytes": 0.0, "model_key": None}


def session_cache_stats() -> dict:
    """Metadata-only native counters plus the shared aggregate SSD budget."""
    stores = list(_PREFIX_STORES.values())
    keys = [key for store in stores for key in list(store.entries)]
    result = {"runtime": "glm-dflash2", "ram": {
        "entries": len(keys),
        "max_entries": sum(store.max_entries for store in stores),
        "tokens": sum(len(key) for key in keys),
        "tokens_note": "sum_across_entries; prefixes_may_overlap",
    }, "ssd": None}
    disks = list({id(store.disk): store.disk for store in stores if store.disk is not None}.values())
    if not _PREFIX_DISK["root"] and not disks:
        return result
    cap = _PREFIX_DISK["max_bytes"]
    import math
    from dataclasses import asdict

    limit = int(cap) if math.isfinite(cap) else 0
    budget = (asdict(disks[0]._catalog.budget.refresh_health()) if disks else {
        "bytes_after": 0, "max_size_bytes": limit, "accounted": False,
        "telemetry_stale": True, "compliant": False,
    })
    budget["root"] = _PREFIX_DISK["root"]
    result["ssd"] = {
        "hits": sum(d.hits for d in disks), "misses": sum(d.misses for d in disks),
        "writes": sum(d.writes for d in disks), "write_errors": sum(d.errors for d in disks),
        "pending_writes": sum(d._queue.unfinished_tasks for d in disks),
        "budget_scope": "shared", "shared_pool": True,
        "global_budget": budget,
    }
    return result


def clear_sessions(include_ssd: bool) -> int:
    """Called on the model worker with the engine generation lock held."""
    removed = 0
    for store in list(_PREFIX_STORES.values()):
        if include_ssd and store.disk is not None:
            removed += store.disk.clear()
        store.entries.clear()
    return removed


def _bundle_fingerprint(model_path, draft_path=None) -> str:
    """Startup-only identity for native state and drafter-dependent hidden taps."""
    import hashlib
    from pathlib import Path
    from .dflash2_session_disk import model_identity

    h = hashlib.sha256(PrefixDiskTier.FORMAT.encode())
    p = Path(model_path).expanduser().resolve()
    # Use the shared target/drafter/runtime contract. The hidden rows stored
    # beside target state depend on the drafter's selected target layers.
    # A target-only namespace could silently restore incompatible taps.
    h.update(model_identity(str(p), str(draft_path or p)).encode())
    h.update(str(p).encode())
    for name in ("config.json", "jang_config.json", "model.safetensors.index.json"):
        f = p / name
        if f.exists():
            h.update(f.read_bytes())
    for f in sorted(p.glob("*.safetensors")):
        s = f.stat()
        h.update(f"{f.name}:{s.st_size}:{int(s.st_mtime)}".encode())
    return h.hexdigest()[:16]


def configure_prefix_disk(enabled: bool, block_cache_dir, max_gb, model_path, draft_path=None) -> None:
    """CLI hook (non-batched launches): the DFlash2 lane's SSD tier follows the server's block disk cache flags
    (on unless --disable-block-disk-cache; --disable-prefix-cache governs the batched scheduler only; --block-disk-cache-dir;
    --block-disk-cache-max-gb,
    where 0 = unlimited as for the block store). New snapshots use the managed
    shared pool. Legacy sibling directories are preserved, never silently reused.
    VMLX_GLM_DFLASH2_DISK_DIR (a directory, or 0 = off) and VMLX_GLM_DFLASH2_DISK_MAX_GB override."""
    from pathlib import Path
    env_dir = os.environ.get("VMLX_GLM_DFLASH2_DISK_DIR")
    root = None
    if env_dir is not None:
        root = None if env_dir in ("", "0") else Path(env_dir).expanduser()
    elif enabled:
        base = Path(block_cache_dir or Path.home() / ".cache" / "vmlx-engine" / "block-cache").expanduser()
        root = base
    env_gb = os.environ.get("VMLX_GLM_DFLASH2_DISK_MAX_GB")
    gb = float(env_gb) if env_gb else float(max_gb if max_gb is not None else 10.0)
    _PREFIX_DISK.update(root=str(root) if root is not None else None,
                        model_key=_bundle_fingerprint(model_path, draft_path) if root is not None and model_path else None,
                        max_bytes=gb * 2 ** 30 if gb > 0 else float("inf"))
    close_prefix_stores()


def close_prefix_stores() -> None:
    """Called only between model lifetimes, never on a generation path."""
    for store in list(_PREFIX_STORES.values()):
        if store.disk is not None:
            store.disk.close()
    _PREFIX_STORES.clear()


def shutdown_runtime() -> None:
    """Finish SSD writes and remove adapters retaining an unloaded model."""
    close_prefix_stores()
    for target in list(_TARGETS.values()):
        target.close()
    _TARGETS.clear()


def prefix_store_for(target) -> "PrefixStore":
    key = id(target)
    st = _PREFIX_STORES.get(key)
    if st is None:
        disk = PrefixDiskTier(_PREFIX_DISK["root"], _PREFIX_DISK["max_bytes"],
                              _PREFIX_DISK["model_key"]) if _PREFIX_DISK["root"] else None
        # Default 4 (was 12): resident snapshots cost the next prefill. GLM-5.3 JANGHT2.4, 8-turn chat, request 8 with
        # 11,147 cached: ENTRIES 12 -> 3.68 s, 4 -> 3.46 s, 2 -> 3.39 s (3 alternating rounds each, box 2026-10-11);
        # after a RAM clear an engine that served 7 turns is exactly as fast as a fresh one. Linear chats only need
        # the newest snapshot; older branch points restore from the SSD tier, which keeps every snapshot.
        st = _PREFIX_STORES[key] = PrefixStore(int(os.environ.get("VMLX_GLM_DFLASH2_PREFIX_ENTRIES", str(PREFIX_RAM_ENTRIES_DEFAULT))),
                                                float(os.environ.get("VMLX_GLM_DFLASH2_PREFIX_MAX_GB", "3")) * 2 ** 30,
                                                disk=disk)
    return st


def release_durable_prefix_ram() -> tuple[int, int]:
    """Model thread only: SSD-only release across every GLM prefix store. Returns (released, still_pending)."""
    released = pending = 0
    for store in list(_PREFIX_STORES.values()):
        r, p = store.release_durable_ram()
        released += r
        pending += p
    return released, pending


class AdaptiveSpec:
    """Online speculation policy (2026-10-10): draft only when it pays, at the block size that pays most.

    Per draft position i (1-based) it keeps decayed counts of how often position i was reached (all earlier drafts
    accepted) and accepted, i.e. the conditional acceptance p_i. Expected tokens of a b-token verify round:
    E(b) = 1 + sum_{i<b} prod_{j<=i} p_j. Costs are measured: an EMA of the wall time of a round per block size (draft +
    verify, synchronized by the acceptance read-back) and of one plain target step. Each decision picks
    argmax_b E(b)/cost(b) if that beats 1/cost(AR) by `margin`, else an AR burst; after `probe_every` AR tokens one
    round is forced at the best estimated block so p_i keeps tracking the text (code drafts well, prose does not).
    Unmeasured block costs are interpolated linearly from the measured ones (prior: AR x (1 + slope (b-1)); slope 0.22
    = measured GLM verify cost per extra row after the 2026-10-10 verify kernels: 2 rows 1.18x, 4 rows 1.61x, 8 rows
    2.55x an AR step). Every `explore_every` rounds the next round tries a neighbour of the best block (b+1, then
    b-1) so the cost/acceptance tables are filled by measurement, not by the prior (with a 0.35 prior the policy
    locked onto blocks 2-3 while block 4 was 6-12% faster).
    """

    def __init__(self, bmax: int, bmin: int = 2, decay: float = 0.97, prior_p: float = 0.6, prior_n: float = 2.0,
                 probe_every: int = 24, margin: float = 0.05, ar_burst: int = 8, slope: float = 0.22,
                 explore_every: int = 6, costs: dict | None = None, ar_reprobe_rounds: int | None = None,
                 ar_reprobe_tokens: int | None = None, ar_min_sample_tokens: int | None = None,
                 ar_outlier_ratio: float = 3.0):
        self.bmax, self.bmin = int(bmax), int(bmin)
        self.decay, self.probe_every, self.margin, self.ar_burst = decay, probe_every, margin, ar_burst
        self.n_try = [prior_n] * (self.bmax + 1)
        self.n_acc = [prior_n * prior_p] * (self.bmax + 1)
        # Costs are a property of the model + kernels, not of the text: ``costs`` is the process-wide table for this
        # target (adaptive_costs). Per request they were re-measured from scratch, so EVERY request opened with an
        # 8-token AR burst (t_ar unknown) and re-seeded its round costs: GLM-5.3 and MiMo stats showed ar_tokens = 9 /
        # 8 on every request of a 13-turn chain at 76-97 % acceptance. Acceptance (p_i) stays per request.
        self._costs = costs if costs is not None else {}
        self._shared_costs = costs is not None
        self.t_round: dict[int, float] = self._costs.setdefault("rounds", {})
        self.ar_since_probe = 0
        self.slope, self.explore_every = slope, explore_every
        self.rounds_since_explore = 0
        self._explore_flip = False
        # AR cost hygiene (MiMo quality-v2 run 2026-10-16: the shared table held t_ar = 88 ms against a real ~25 ms,
        # so AR was never chosen again -- ar_tokens 0 on every 10-33k-token reasoning turn, where spec rounds were only
        # break-even). (1) a burst cut short by EOS / max_tokens pays two pipelined launches + a sync over 1-2 tokens,
        # 2-4x the real per-token cost: such bursts do not update t_ar; (2) once spec wins, nothing re-measured AR, so
        # a stale (or context-drifted) t_ar lasted the life of the process: after ar_reprobe_rounds spec rounds an
        # AR burst of the normal length re-measures it and replaces most of the old value (4-token probes read
        # 27-32 ms against 21-27 for 8-token bursts in the 2026-10-16 A/B: launch overhead over fewer tokens);
        # (3) the 88 ms itself came from ONE burst in one request (~240 ms per token, a ~2 s stall: t_ar 22 -> 88
        # in a single 0.7/0.3 blend): an ordinary burst sample above ar_outlier_ratio x t_ar is not blended in.
        # Re-probes are always taken, so a genuine increase still lands within ar_reprobe_rounds.
        # VMLX_DFLASH_AR_REPROBE_ROUNDS=0 / VMLX_DFLASH_AR_MIN_SAMPLE_TOKENS=1 restore the old behaviour (A/B control)
        if ar_reprobe_rounds is None:
            ar_reprobe_rounds = int(os.environ.get("VMLX_DFLASH_AR_REPROBE_ROUNDS", "256"))
        if ar_min_sample_tokens is None:
            ar_min_sample_tokens = int(os.environ.get("VMLX_DFLASH_AR_MIN_SAMPLE_TOKENS", "3"))
        if ar_reprobe_tokens is None:
            ar_reprobe_tokens = ar_burst
        self.ar_reprobe_rounds, self.ar_reprobe_tokens = int(ar_reprobe_rounds), int(ar_reprobe_tokens)
        self.ar_outlier_ratio = float(ar_outlier_ratio)
        self.ar_outliers_skipped = 0
        self.ar_min_sample_tokens = int(ar_min_sample_tokens)
        self.rounds_since_ar = 0
        self._reprobe_pending = False

    @property
    def t_ar(self):
        return self._costs.get("ar")

    @t_ar.setter
    def t_ar(self, value) -> None:
        self._costs["ar"] = value

    def p(self, i: int) -> float:
        return self.n_acc[i] / max(self.n_try[i], 1e-9)

    def expected(self, b: int) -> float:
        e, c = 1.0, 1.0
        for i in range(1, b):
            c *= self.p(i); e += c
        return e

    def cost(self, b: int) -> float:
        if b in self.t_round:
            return self.t_round[b]
        known = sorted(self.t_round)
        if len(known) >= 2:
            (b0, c0), (b1, c1) = (known[0], self.t_round[known[0]]), (known[-1], self.t_round[known[-1]])
            slope = max((c1 - c0) / max(b1 - b0, 1), 0.0)
            return max(c0 + slope * (b - b0), 1e-6)
        base = self.t_ar if self.t_ar is not None else 1.0
        if len(known) == 1:
            k = known[0]
            return self.t_round[k] * (1 + self.slope * (b - 1)) / (1 + self.slope * (k - 1))
        return base * (1 + self.slope * (b - 1))

    def choose(self) -> tuple[str, int]:
        cands = range(self.bmin, self.bmax + 1)
        best = max(cands, key=lambda b: self.expected(b) / self.cost(b))
        if self.t_ar is None:                       # need one AR measurement before comparing
            return ("ar", self.ar_burst)
        if not self.t_round:                        # and one round
            return ("spec", best)
        if self.ar_reprobe_rounds > 0 and self.rounds_since_ar >= self.ar_reprobe_rounds:
            self.rounds_since_ar = 0
            self._reprobe_pending = True
            return ("ar", self.ar_reprobe_tokens)
        spec_rate = self.expected(best) / self.cost(best)
        # hysteresis: AR only when it is estimated at least `margin` faster than the best block (round costs carry
        # read-back overhead that AR bursts hide; switching on a near-tie loses)
        if spec_rate * (1.0 + self.margin) >= 1.0 / self.t_ar or self.ar_since_probe >= self.probe_every:
            self.ar_since_probe = 0
            self.rounds_since_explore += 1
            if self.rounds_since_explore >= self.explore_every:
                self.rounds_since_explore = 0
                self._explore_flip = not self._explore_flip
                nb = best + 1 if self._explore_flip else best - 1
                if self.bmin <= nb <= self.bmax:
                    return ("spec", nb)
            return ("spec", best)
        return ("ar", self.ar_burst)

    def update_spec(self, b: int, accepted: int, dt: float, after_ar: bool = False) -> None:
        """after_ar: the round right after an AR burst feeds the drafter all of the burst's context rows and is
        slower than a steady-state round; its acceptance counts, its time does not."""
        self.rounds_since_ar += 1
        for i in range(1, self.bmax + 1):
            self.n_try[i] *= self.decay; self.n_acc[i] *= self.decay
        for i in range(1, b):
            if i <= accepted + 1:
                self.n_try[i] += 1.0
            if i <= accepted:
                self.n_acc[i] += 1.0
        if self._shared_costs:
            warm = self._costs.setdefault("warm_rounds", set())
            if b not in warm:
                # first round at this block size in the shared table: drafter / verify kernels compile inside it; a
                # stored outlier would steer every later request toward AR (as _BlockChooser's warm set)
                warm.add(b)
                return
        if not after_ar or b not in self.t_round:
            self.t_round[b] = dt if b not in self.t_round else 0.7 * self.t_round[b] + 0.3 * dt

    def update_ar(self, dt_per_token: float, tokens: int, requested: int | None = None) -> None:
        """``requested``: the burst length asked for. A burst that ended early (EOS / max_tokens) pays its pipeline
        launches and the final sync over fewer tokens than planned (the cut that set MiMo's t_ar to 28.5 ms was 5 of
        8 tokens, above the 3-token floor, and that seed then held for the life of the process): it is not an AR
        cost sample. (A process-wide re-probe counter was measured too and dropped: on GLM, where drafting beats AR on
        every prompt, the extra forced re-probes cost 2-3.5 % on prose and sampled decode, 2026-10-18 release audit.)"""
        self.ar_since_probe += tokens
        self.rounds_since_ar = 0
        reprobe, self._reprobe_pending = self._reprobe_pending, False
        if int(tokens) < self.ar_min_sample_tokens or (requested is not None and int(tokens) < int(requested)):
            if reprobe:
                self.rounds_since_ar = self.ar_reprobe_rounds   # the re-measurement did not happen: re-probe next
            return                                  # EOS / max_tokens-cut burst: launch overhead, not the AR cost
        if self._shared_costs and not self._costs.get("ar_warm"):
            # the first AR burst of the process can include 1-row kernel compiles: a stored outlier would keep the
            # shared table claiming AR is slow (and AR fallback off) for every later request
            self._costs["ar_warm"] = True
            if self.t_ar is None:
                self.ar_since_probe = 0
                return
        if self.t_ar is None:
            self.t_ar = dt_per_token
        elif not reprobe and self.ar_outlier_ratio > 0 and dt_per_token > self.ar_outlier_ratio * self.t_ar:
            self.ar_outliers_skipped += 1                # one stalled burst must not freeze the policy on spec
        elif reprobe:
            self.t_ar = 0.3 * self.t_ar + 0.7 * dt_per_token
        else:
            self.t_ar = 0.7 * self.t_ar + 0.3 * dt_per_token


def adaptive_costs(target, bmax: int) -> dict:
    """AdaptiveSpec cost table for one target model and block range, shared by every request on that target (see
    AdaptiveSpec). Kept on the target object itself: a process-wide dict keyed by id() would hand a later target that
    reuses a freed id the old model's costs."""
    tables = getattr(target, "_vmlx_adaptive_costs", None)
    if not isinstance(tables, dict):
        tables = {}
        object.__setattr__(target, "_vmlx_adaptive_costs", tables)
    return tables.setdefault(int(bmax), {})


def _dflash2_rounds(target: GLMDFlashTarget, draft, prompt_ids, max_tokens: int, eos: set, temperature: float = 0.0,
                    top_p: float = 1.0, top_k: int = 0, block_size: int | None = None, prefill_step: int = 2048,
                    stats: SpecStats | None = None, controls=None, adaptive: bool = False,
                    prefix_store: "PrefixStore | None" = None, key_ids=None, media: dict | None = None,
                    stable_root: int = 0):
    """Yields (new_tokens, drafted) per cycle: first the token sampled from the prompt (drafted 0), then each verify
    round's accepted drafts + bonus token (drafted = verify width - 1). Upstream dflash loop with GLM verify/rollback.
    controls: optional vmlx_engine.dflash2_sampling.DFlash2SamplingControls (min_p, logit_bias, penalties) applied to
    the TARGET distribution of every verified row, as the Qwen lane does (rejection sampling stays exact).
    media: an image plan from MLXMultimodalLM._dflash2_media_plan. The prompt rows up to its last image are prefilled
    from the VLM's merged input embeddings (vision features at their placeholders); text rows embed their ids.
    key_ids: the prefix-store key (the plan's per-item media key: every image placeholder run replaced by ids derived
    from THAT image's pixels), so a later turn restores through identical earlier images and diverges at a new one.
    A restore that already covers every image never runs the vision tower. Without per-item keys, prefixes are reused
    and stored only up to the first image."""
    import dflash.model_mlx as runtime
    from dflash.model_mlx import (DFlash2DraftModel, _rejection_sample, _sample_probs, _sampling_probs,
                                  _trim_recent_cache, make_sampler)

    sampler = make_sampler(temperature, top_p, top_k)
    bs0 = int(block_size or draft.config.block_size)
    mask_id = int(draft.config.mask_token_id)
    prompt = mx.array(prompt_ids); st = stats if stats is not None else SpecStats()
    dcache = draft.make_cache(); draft.bind(target)
    hidden_limit = draft.config.sliding_window - 1 if all(t == "sliding_attention" for t in draft.config.layer_types) else None
    t0 = time.perf_counter()
    lookup_ids = [int(t) for t in (key_ids if key_ids is not None else prompt_ids)]
    if len(lookup_ids) != len(prompt_ids):
        raise ValueError("glm dflash2: prefix key and prompt lengths differ")
    store_limit = None
    if media is not None and key_ids is None:
        store_limit = int(media.get("first_media_index", 0))      # unkeyed images: text before the first one only
    media_end = int(media.get("media_end", 0)) if media is not None else 0
    hit = None
    if prefix_store is not None:
        hit = prefix_store.lookup(lookup_ids if store_limit is None else lookup_ids[:store_limit])
        if store_limit is not None:
            prefix_store.last_persistence = None
    reused, cache = 0, None
    if hit is not None:
        reused, snap = hit
        cache = _clone_caches(snap.cache)
    if cache is None:
        cache = target.make_cache()
    # pp/s receipt clock: lookup and restore above are excluded (the panel contract), the vision tower, suffix
    # prefill, boundary snapshots and the first sample below are included.
    compute_t0 = time.perf_counter()
    if reused == prompt.size:                       # whole prompt already prefilled
        logits, hidden = snap.logits, snap.hidden
    else:
        # Prefill the suffix; with a prefix store, stop at every absolute multiple of PREFIX_SNAPSHOT_EVERY and keep a
        # snapshot there too, so a later prompt that diverges mid-history (same long prefix, different last turn)
        # restores the last boundary before the divergence instead of starting cold. KDA recurrent state cannot be
        # truncated, so only stored boundaries are restorable.
        every = PREFIX_SNAPSHOT_EVERY if prefix_store is not None else 0
        pos, hidden = reused, (snap.hidden if reused else None)
        embeds = None
        if media is not None and reused < media_end:
            if media.get("embeds") is None:
                media.update(media["materialize"]())
            embeds = media["embeds"]
        st.vision_tower = media is not None and embeds is not None
        while pos < prompt.size:
            end = min(prompt.size, (pos // every + 1) * every) if every else prompt.size
            # Stable root: also stop (and snapshot) at the end of the system turn (system prompt + tools), so a new
            # chat with the same system prompt restores all of it, not just the last 2048-token boundary inside it.
            if every and pos < stable_root < end:
                end = stable_root
            seg = embeds[:, pos:end] if (embeds is not None and pos < media_end) else None
            logits, h = _prefill(target, prompt[pos:end], cache, prefill_step, hidden_limit, embeds=seg)
            hidden = h if hidden is None else mx.concatenate([hidden, h], axis=1)
            if hidden_limit is not None:
                hidden = hidden[:, -hidden_limit:]
            pos = end
            if every and pos < prompt.size and (store_limit is None or pos <= store_limit):
                mx.eval(logits, hidden, [c.state for c in cache])
                prefix_store.put(lookup_ids[:pos], cache, hidden, logits[:, -1:])
    st.cached_tokens = reused
    st.cache_source = prefix_store.last_source if (prefix_store is not None and reused) else ""
    if prefix_store is not None and store_limit is None:
        mx.eval(logits, hidden, [c.state for c in cache])
        prefix_store.put(lookup_ids, cache, hidden, logits[:, -1:])
    for c in dcache:
        c.offset = prompt.size - hidden.shape[1]
    history = list(prompt_ids)
    first = logits[:, -1:]
    if controls is not None:
        first = controls.process(first, history)
        tok = runtime._sample_probs(controls.probabilities(runtime, first, temperature, top_p, top_k)) if temperature > 0 \
            else mx.argmax(first, axis=-1)
    else:
        tok = sampler(first)
    mx.eval(tok, hidden)
    st.prompt_tokens, st.prefill_s = prompt.size, time.perf_counter() - t0
    st.prefill_compute_s = time.perf_counter() - compute_t0
    t1 = time.perf_counter()
    tokens = [int(tok.item())]; n = 1
    # Reuse the generation's existing history; export a copy only at terminal.
    st.output_token_ids = tokens
    try:
        yield [tokens[-1]], 0
        if tokens[-1] in eos:
            return
        policy = AdaptiveSpec(bs0, costs=adaptive_costs(target, bs0)) if adaptive else None

        def sample_row(lg, hist):
            last = lg[:, -1:]
            if controls is not None:
                last = controls.process(last, hist)
                return (runtime._sample_probs(controls.probabilities(runtime, last, temperature, top_p, top_k))
                        if temperature > 0 else mx.argmax(last, axis=-1))
            return sampler(last)

        def ar_step(tok_arr):
            from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
            with affine_moe_ar_scope():
                lg = target.forward(tok_arr, cache, verify=False, taps=True)
            return lg, target.hidden()

        def ar_burst(k):
            """k plain target steps (pipelined when no history-dependent controls); keeps the drafter's tap rows.
            Invariant kept: the last emitted token is not forwarded (an extra step launched past EOS / max_tokens only
            touches this generation's private cache)."""
            emitted, rows = [], []
            pipelined = controls is None
            lg, h = ar_step(mx.array([[tokens[-1]]])); nx = sample_row(lg, history + tokens); mx.async_eval(nx, h)
            for j in range(k):
                launched = None
                if pipelined and j + 1 < k:
                    lg2, h2 = ar_step(nx.reshape(1, 1)); nx2 = sample_row(lg2, None); mx.async_eval(nx2, h2)
                    launched = (nx2, h2)
                t = int(nx.item()); emitted.append(t); rows.append(h)
                if t in eos or n + len(emitted) >= max_tokens or j + 1 >= k:
                    break
                if launched is None:
                    lg, h = ar_step(nx.reshape(1, 1)); nx = sample_row(lg, history + tokens + emitted)
                    mx.async_eval(nx, h)
                else:
                    nx, h = launched
            return emitted, rows

        after_ar = False
        while n < max_tokens:
            mode, b = policy.choose() if policy is not None else ("spec", bs0)
            if mode == "ar":
                after_ar = True
                ta = time.perf_counter()
                new, rows = ar_burst(min(b, max_tokens - n))
                policy.update_ar((time.perf_counter() - ta) / max(len(new), 1), len(new), requested=b)
                hidden = mx.concatenate([hidden] + rows, axis=1)
                if hidden_limit is not None and hidden.shape[1] > hidden_limit:
                    drop = hidden.shape[1] - hidden_limit          # outside the drafter's sliding window anyway
                    hidden = hidden[:, drop:]
                    for c in dcache:
                        c.offset += drop
                st.ar_steps += len(new)
                stop = next((i for i, t in enumerate(new) if t in eos), None)
                if stop is not None:
                    new = new[: stop + 1]
                tokens.extend(new); n += len(new)
                yield new, 0
                if stop is not None:
                    break
                continue
            bs = min(b, max_tokens - n + 1)
            if bs <= 1:
                break
            tr = time.perf_counter()
            block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
            if isinstance(draft, DFlash2DraftModel):
                d_tok, d_idx, d_prob = draft.propose(block, hidden, dcache, temperature, logits_start=1)
            else:
                dl = draft(block, hidden, dcache, logits_start=1)
                d_idx = None
                if temperature > 0:
                    # DRAW the drafts from q: rejection sampling is exact only for d ~ q. The argmax here (2026-10-16
                    # audit) proposed a point mass while passing the full q, inflating acceptance min(1, p/q) toward
                    # the drafter's top token (DFlash2DraftModel's selector, the production path, already samples)
                    d_prob = _sampling_probs(dl, temperature, top_p, top_k); d_tok = _sample_probs(d_prob)
                else:
                    d_prob = None; d_tok = mx.argmax(dl, axis=-1)
            if (trim_n := dcache[0].offset - (prompt.size + n - 1)) > 0:
                _trim_recent_cache(dcache, trim_n)
            mx.async_eval(d_tok)
            verify_in = mx.concatenate([mx.array([[tokens[-1]]]), d_tok], axis=1)
            logits = target.forward(verify_in, cache, verify=True)
            hidden = target.hidden()
            if controls is not None:
                logits = controls.process(logits, history + tokens, d_tok)
            if temperature > 0:
                t_prob = controls.probabilities(runtime, logits, temperature, top_p, top_k) if controls is not None else \
                    _sampling_probs(logits, temperature, top_p, top_k)
                mx.async_eval(t_prob, hidden)
            else:
                t_tok = mx.argmax(logits, axis=-1); mx.async_eval(t_tok, hidden)
            dl_ = d_tok[0].tolist()
            if temperature > 0:
                accepted, bonus = _rejection_sample(d_tok, t_prob, d_prob, d_idx)
            else:
                tl_ = t_tok[0].tolist()
                accepted = next((i for i in range(len(dl_)) if dl_[i] != tl_[i]), len(dl_))
                bonus = tl_[accepted]
            new = (dl_[:accepted] + [bonus])[: max_tokens - n]
            st.rounds += 1; st.accepted.append(accepted); st.blocks.append(bs)
            if policy is not None:
                policy.update_spec(bs, accepted, time.perf_counter() - tr, after_ar=after_ar)
            after_ar = False
            stop = next((i for i, t in enumerate(new) if t in eos), None)
            if stop is not None:
                new = new[: stop + 1]
            tokens.extend(new); n += len(new)
            yield new, bs - 1
            if stop is not None:
                break
            target.rollback(cache, bs - accepted - 1)
            hidden = hidden[:, : accepted + 1, :]
    finally:
        st.tokens, st.decode_s = n, time.perf_counter() - t1


def generate_dflash2(target: GLMDFlashTarget, draft, prompt_ids, max_tokens: int, eos: set, temperature: float = 0.0,
                     top_p: float = 1.0, block_size: int | None = None, prefill_step: int = 2048,
                     stats: SpecStats | None = None, adaptive: bool = False,
                     prefix_store: "PrefixStore | None" = None, key_ids=None,
                     media: dict | None = None, stable_root: int = 0) -> Iterator[int]:
    """DFlash2 speculative generation as a token iterator. Greedy output == generate_ar's (also with adaptive=True:
    AR bursts and verify rounds both emit the target's argmax)."""
    for new, _ in _dflash2_rounds(target, draft, prompt_ids, max_tokens, eos, temperature, top_p, 0, block_size,
                                  prefill_step, stats, adaptive=adaptive, prefix_store=prefix_store,
                                  key_ids=key_ids, media=media, stable_root=stable_root):
        yield from new


_TARGETS: dict = {}


def target_for(model, layer_ids) -> GLMDFlashTarget:
    """One tap adapter per loaded model (the taps wrap the model's layers in place)."""
    key = id(model)
    t = _TARGETS.get(key)
    if t is None or t.layer_ids != tuple(int(i) for i in layer_ids):
        t = GLMDFlashTarget(model, layer_ids); _TARGETS[key] = t
    return t


def is_glm5_next(model) -> bool:
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None) if cfg is not None and not isinstance(cfg, dict) else (cfg or {}).get("model_type")
    return mt == "glm5_next" or getattr(model, "model_type", None) == "glm5_next"


def _glm_stable_root(tokenizer, prompt_ids) -> int:
    """Tokens before the first `<|user|>`: GLM's system turn ([gMASK]<sop><|system|>Reasoning Effort... + the tool
    block + the system message). A new chat with the same system prompt, tools and effort shares exactly these.
    0 when the token is unknown, absent, or the root is shorter than 256 tokens (the 2048 boundaries cover it)."""
    try:
        uid = tokenizer.convert_tokens_to_ids("<|user|>")
    except Exception:
        return 0
    if not isinstance(uid, int) or uid < 0 or uid == getattr(tokenizer, "unk_token_id", None):
        return 0
    try:
        root = list(prompt_ids).index(uid)
    except ValueError:
        return 0
    return root if root >= 256 else 0


def stream_glm_dflash2(model, tokenizer, draft, prompt: str, *, max_tokens: int, temperature: float, top_p: float = 1.0,
                       top_k: int = 0, stop=None, prompt_tokens=None, media=None, sampling_controls=None):
    """Server lane: yields upstream ``dflash.model_mlx.GenerationResponse`` chunks (text segment, tokens, accepted = tokens
    emitted this cycle, drafted = verify width - 1) for a glm5_next target. ``media``: an image plan (see
    ``_dflash2_rounds``) with ``prompt_tokens`` set to its processor-expanded ids."""
    from dflash.model_mlx import GenerationResponse
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if media is not None and prompt_tokens is None:
        raise ValueError("glm5_next DFlash2 media plan without its processor-expanded prompt tokens")
    tw = tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)
    if prompt_tokens is None:
        specials = tuple(t for t in ("[gMASK]", getattr(tw, "bos_token", None)) if t)
        prompt_tokens = tw.encode(prompt, add_special_tokens=not any(prompt.startswith(t) for t in specials))
    eos = set(getattr(tw, "eos_token_ids", None) or ()) | {154820, 154827, 154829}
    detok = tw.detokenizer; detok.reset()
    if stop:
        from vmlx_engine.dflash2_runtime import _StopDetokenizer
        detok = _StopDetokenizer(detok, [stop] if isinstance(stop, str) else list(stop))
    target = target_for(model, draft.config.target_layer_ids)
    st = SpecStats(); tic = time.perf_counter(); n = 0
    store = (prefix_store_for(target)
             if os.environ.get("VMLX_GLM_DFLASH2_PREFIX_CACHE", "1") != "0" else None)
    from vmlx_engine.utils.prefill_step import prefill_step_size
    for new, drafted in _dflash2_rounds(target, draft, list(prompt_tokens), int(max_tokens), eos, float(temperature),
                                        float(top_p), int(top_k), prefill_step=prefill_step_size(),
                                        stats=st, controls=sampling_controls,
                                        adaptive=os.environ.get("VMLX_GLM_DFLASH2_ADAPTIVE", "1") != "0",
                                        prefix_store=store,
                                        key_ids=media.get("key_tokens") if media is not None else None,
                                        media=media,
                                        stable_root=_glm_stable_root(tw, prompt_tokens) if store is not None else 0):
        if n == 0:
            tic = time.perf_counter()
        for t in new:
            if t not in eos:                  # stop tokens end the turn; their text (<|user|>) is not content
                detok.add_token(t)
        n += len(new)
        finished = new[-1] in eos or n >= max_tokens or getattr(detok, "matched", False)
        reason = None
        if finished:
            detok.finalize(); reason = "stop" if (new[-1] in eos or getattr(detok, "matched", False)) else "length"
        r = GenerationResponse(detok.last_segment, list(new), len(new), st.prompt_tokens,
                               st.prompt_tokens / max(st.prefill_s, 1e-9), n, n / max(time.perf_counter() - tic, 1e-9),
                               mx.get_peak_memory() / 1e9, reason)
        r.drafted = int(drafted)
        r.cached_tokens = int(st.cached_tokens)        # prompt tokens restored from the prefix store (usage)
        # usage.vmlx_prefill: the chat footer's pp/s. Same contract as the Qwen lane (dflash2_runtime) and the batched
        # engines: tokens the model computed for this prompt, model prefill + prompt state (+ vision) seconds.
        r.prefill_usage = {
            "tokens": int(st.prompt_tokens) - int(st.cached_tokens),
            "seconds": float(st.prefill_compute_s),
            "scope": "model_prefill_and_prompt_state",
            "path": "glm_dflash2" + ("_with_vision" if st.vision_tower else ""),
        }
        if finished:
            r.output_token_ids = list(st.output_token_ids)
        # usage prompt_tokens_details.cache_detail, like the batched lanes' "native-glm+disk" / "block-disk"
        r.cache_detail = {"ram": "glm-dflash2-prefix", "disk": "glm-dflash2-prefix+disk"}.get(st.cache_source, "")
        if finished and media is not None:
            import logging
            logging.getLogger(__name__).info(
                "GLM DFlash2 image prefill: prompt_tokens=%d cached=%d source=%s media_end=%d key=%s vision_tower=%s",
                st.prompt_tokens, st.cached_tokens, st.cache_source or "-", int(media.get("media_end", 0)),
                "per-item" if media.get("key_tokens") is not None else "text-before-first-image",
                "ran" if st.vision_tower else "skipped")
        if finished and store is not None:
            # Publish the terminal immediately; SimpleEngine waits for this
            # request's SSD receipt only at the next admission or teardown.
            r.persistence_future = store.last_persistence
        yield r
        if finished:
            return
