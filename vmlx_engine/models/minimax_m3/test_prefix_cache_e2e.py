"""End-to-end prefix-cache TIER test for the MiniMax-M3 MSA dual cache.

Drives the REAL vMLX cache tiers — no reimplementation of the slice/restore
logic. If any of the additive ("minimax_m3", k, v, idx) branches in
prefix_cache.py / block_disk_store.py are wrong, these decodes diverge.

  PART 1  (L1, real model)  prefill -> store_cache (splits the live cache into
           paged blocks, carving k/v/idx per block) -> reconstruct_cache
           (concatenates the blocks back into a MiniMaxM3SparseCache) -> decode.
           Asserts the reconstructed-cache decode is bit-identical to the
           live-cache decode. Proves the indexer keys survive block split +
           reconstruction (without idx, the sparse-layer block selection drifts).

  PART 2  (L2 SSD, real disk)  write_block_async -> shutdown(flush) -> reopen
           store -> read_block. A true round-trip through safetensors + the
           SQLite index on mixed dense-kv + sparse-minimax_m3 layers. Proves the
           "cache pool" persists and restores all three components from disk.

Run on the other Mac:
  python test_prefix_cache_e2e.py <bundle>           # PART 1 + PART 2
  python test_prefix_cache_e2e.py <bundle> --disk-only   # PART 2 only (no model)

Created by Jinho Jang (eric@jangq.ai).
"""
import sys
import os
import tempfile
import time
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minimax_m3 as M  # noqa: E402
from cache import restore_minimax_m3_sparse  # noqa: F401,E402

from vmlx_engine.prefix_cache import BlockAwarePrefixCache  # noqa: E402
from vmlx_engine.paged_cache import PagedCacheManager  # noqa: E402
from vmlx_engine.block_disk_store import (  # noqa: E402
    BlockDiskStore,
    _serialize_block,
    _deserialize_block,
)

BUNDLE = sys.argv[1]
DISK_ONLY = "--disk-only" in sys.argv


# ── helpers ──────────────────────────────────────────────────────────
def _copy_state(state):
    """Force independent copies so later decode on the live cache can't alias
    (KVCache.state returns slices/views of the live buffers)."""
    if not isinstance(state, (tuple, list)):
        return state
    out = []
    for x in state:
        if x is not None and hasattr(x, "shape"):
            y = x * 1
            mx.eval(y)
            out.append(y)
        else:
            out.append(x)
    return tuple(out)


def extract_states(cache):
    """Replicate scheduler._extract_cache_states for a flat per-layer cache."""
    return [
        {
            "state": _copy_state(c.state),
            "meta_state": c.meta_state,
            "class_name": type(c).__name__,
        }
        for c in cache
    ]


def decode_from(model, cache, first_logits, n):
    nxt = int(mx.argmax(first_logits[0, -1]))
    toks = [nxt]
    for _ in range(n - 1):
        lg = model(mx.array([[toks[-1]]]), cache=cache)
        mx.eval(lg)
        toks.append(int(mx.argmax(lg[0, -1])))
    return toks


def maxabs(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


# ── PART 2: L2 disk round-trip (fast, deterministic, no model) ───────
def part2_disk_roundtrip():
    print("\n=== PART 2: L2 SSD round-trip (write_block_async -> read_block) ===")
    B, H, S, D = 1, 4, 70, 128  # 70 tok > block_size 64 → exercises real shapes
    Hi = 1
    mx.random.seed(0)
    cache_data = []
    for i in range(6):
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        if i < 3:  # dense full-attn layers → standard kv tuple
            cache_data.append(("kv", k, v))
        else:      # MSA sparse layers → minimax_m3 tuple with idx_keys
            idx = mx.random.normal((B, Hi, S, D)).astype(mx.float16)
            cache_data.append(("minimax_m3", k, v, idx))
    mx.eval(*[t for e in cache_data for t in e[1:] if t is not None])

    # (a) pure codec round-trip via a real safetensors file
    tensors, dtype, n_layers = _serialize_block(cache_data)
    assert n_layers == len(cache_data), (n_layers, len(cache_data))
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "blk.safetensors")
        mx.save_safetensors(fp, dict(tensors))
        loaded = mx.load(fp)
    restored = _deserialize_block(loaded, dtype)
    assert len(restored) == len(cache_data), (len(restored), len(cache_data))
    codec_maxdiff = 0.0
    for orig, rec in zip(cache_data, restored):
        assert orig[0] == rec[0], f"tag mismatch {orig[0]} != {rec[0]}"
        for a, b in zip(orig[1:], rec[1:]):
            if a is None:
                assert b is None
                continue
            codec_maxdiff = max(codec_maxdiff, maxabs(a, b))
    print(f"  codec: {len(cache_data)} layers (3 kv + 3 minimax_m3), "
          f"maxdiff={codec_maxdiff:.4g}, tags preserved")
    assert codec_maxdiff == 0.0, "codec round-trip not exact"

    # (b) true disk path: async write -> flush -> reopen -> read
    with tempfile.TemporaryDirectory() as td:
        store = BlockDiskStore(cache_dir=td, max_size_gb=1.0,
                               expected_num_layers=len(cache_data))
        bh = bytes.fromhex("ab" * 32)
        store.write_block_async(bh, cache_data, token_count=S)
        store.shutdown()  # flush queue + finalize SQLite index synchronously

        store2 = BlockDiskStore(cache_dir=td, max_size_gb=1.0,
                                expected_num_layers=len(cache_data))
        assert store2.has_block(bh), "block missing from L2 index after flush"
        disk = store2.read_block(bh)
        store2.shutdown()
    assert disk is not None and len(disk) == len(cache_data)
    disk_maxdiff = 0.0
    m3_layers = 0
    for orig, rec in zip(cache_data, disk):
        assert orig[0] == rec[0], f"disk tag mismatch {orig[0]} != {rec[0]}"
        if rec[0] == "minimax_m3":
            m3_layers += 1
            assert rec[3] is not None, "idx_keys lost on disk round-trip"
        for a, b in zip(orig[1:], rec[1:]):
            if a is None:
                assert b is None
                continue
            disk_maxdiff = max(disk_maxdiff, maxabs(a, b))
    print(f"  disk:  read back {len(disk)} layers, {m3_layers} minimax_m3 with "
          f"idx_keys intact, maxdiff={disk_maxdiff:.4g}")
    assert disk_maxdiff == 0.0, "disk round-trip not exact"
    print("PART 2 PASS — L2 SSD persists/restores k, v AND idx_keys exactly")


# ── PART 1: L1 prefix reuse through the real BlockAwarePrefixCache ───
def part1_prefix_reuse():
    print("=== PART 1: L1 prefix reuse (store_cache -> reconstruct_cache) ===")
    from runtime import load_minimax_m3
    t0 = time.time()
    model, tok, _eos = load_minimax_m3(BUNDLE)
    print(f"  model loaded in {time.time()-t0:.0f}s")

    # prompt long enough for several 64-tok blocks (multi-block concat path)
    prompt = (
        "def merge_sort(arr):\n"
        "    \"\"\"Sort a list using the classic merge sort algorithm. "
        "Split the list in half, recursively sort each half, then merge the "
        "two sorted halves back together in ascending order. This function "
        "demonstrates the divide-and-conquer paradigm and runs in O(n log n) "
        "time on any input list, regardless of its initial ordering or the "
        "distribution of duplicate values it happens to contain.\"\"\"\n"
        "    if len(arr) <= 1:\n        return arr\n"
        "    mid = len(arr) // 2\n"
        "    left = merge_sort(arr[:mid])\n"
        "    right = merge_sort(arr[mid:])\n"
    )
    ids = tok.encode(prompt)
    print(f"  prompt = {len(ids)} tokens (~{(len(ids)+63)//64} blocks of 64)")

    # live cache A: prefill. Snapshot the FROZEN prefill state BEFORE decoding
    # (decode mutates A in place), then take the reference continuation.
    A = model.make_cache()
    logitsA = model(mx.array([ids]), cache=A)
    mx.eval(logitsA)
    cache_data = extract_states(A)          # frozen at len(ids) tokens
    n_sparse = sum(isinstance(c, M.MiniMaxM3SparseCache) for c in A)
    print(f"  live cache: {len(A)} layers, {n_sparse} MSA sparse "
          f"(idx_keys len={A[3].idx_keys.shape[2]}, kv offset={A[3].offset})")
    ref = decode_from(model, A, logitsA, 8)
    print(f"  ref tokens (live continuation) = {ref}")

    # reconstruct from paged blocks (L1 RAM reuse; disk path proven in PART 2).
    pcm = PagedCacheManager(block_size=64, max_blocks=4096, disk_store=None)
    bac = BlockAwarePrefixCache(model=model, paged_cache_manager=pcm,
                                model_path=BUNDLE)
    bt = bac.store_cache("req0", ids, cache_data)
    assert bt is not None, "store_cache returned None"
    print(f"  stored: {bt.num_tokens} tokens across {len(bt.block_ids)} blocks")
    try:
        print(f"  allowed_n_kv_heads = {sorted(bac._get_allowed_n_kv_heads())}")
    except Exception as e:
        print(f"  allowed_n_kv_heads raised: {e}")
    for bid in bt.block_ids:
        blk = pcm.allocated_blocks.get(bid)
        cd = blk.cache_data
        tags = [e[0] for e in cd][:6] if cd else None
        print(f"  block {bid}: cache_data={'None' if cd is None else len(cd)} "
              f"first6 tags={tags}")
    rebuilt = bac.reconstruct_cache(bt)

    assert rebuilt is not None, "reconstruct_cache returned None (cache miss)"
    assert len(rebuilt) == len(A), (len(rebuilt), len(A))
    # NOTE: count by type NAME, not isinstance — the engine reconstructs via the
    # fully-qualified vmlx_engine.models.minimax_m3.cache package, while this
    # standalone test loaded the class via a bare sys.path import, so the two
    # MiniMaxM3SparseCache class objects differ in identity (same name). In the
    # real server both paths share the package, so this is purely a test quirk.
    def is_m3(c):
        return type(c).__name__ == "MiniMaxM3SparseCache"
    n_sparse_r = sum(is_m3(c) for c in rebuilt)
    assert n_sparse_r == n_sparse, (n_sparse_r, n_sparse)
    # idx_keys must be restored on every sparse layer
    for li, c in enumerate(rebuilt):
        if is_m3(c):
            assert c.idx_keys is not None, f"layer {li} idx_keys not restored"
            assert c.idx_keys.shape[2] == c.offset, (
                f"layer {li} idx/kv length mismatch "
                f"{c.idx_keys.shape[2]} != {c.offset}")
    print(f"  reconstructed: {len(rebuilt)} layers, {n_sparse_r} MSA sparse, "
          f"idx_keys aligned to kv offset on all sparse layers")

    # decode from the reconstructed cache; must match the live continuation.
    # rebuilt holds the full prefix (offset == len(ids)); feeding ref[0] must
    # yield ref[1], feeding ref[1] must yield ref[2], ... exactly as the live
    # cache did. (No extra prefill — that's the whole point of reuse.)
    got = [ref[0]]
    for _ in range(7):
        lg = model(mx.array([[got[-1]]]), cache=rebuilt)
        mx.eval(lg)
        got.append(int(mx.argmax(lg[0, -1])))

    match = got == ref
    print(f"  ref     tokens: {ref}")
    print(f"  rebuilt tokens: {got}")
    if not match:
        # find first divergence
        for i, (a, b) in enumerate(zip(ref, got)):
            if a != b:
                print(f"  FIRST DIVERGENCE at step {i}: {a} != {b}")
                break
    assert match, "reconstructed-cache decode diverged from live-cache decode"
    print("PART 1 PASS — prefix split + reconstruct preserves MSA decode exactly")


if __name__ == "__main__":
    part2_disk_roundtrip()  # fast gate first
    if not DISK_ONLY:
        print()
        part1_prefix_reuse()
    print("\nALL PREFIX-CACHE TIER TESTS PASSED")
