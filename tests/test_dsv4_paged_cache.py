# SPDX-License-Identifier: Apache-2.0
"""DSV4 paged/L2 cache contract tests.

DSV4 uses a composite cache, not a plain KV cache:
DeepseekV4Cache.state = (local_swa_state, compressor_state, indexer_state).
The block cache must preserve that nested state exactly enough for
reconstruction and L2 disk promotion.
"""

from __future__ import annotations

import pytest


mx = pytest.importorskip("mlx.core")


def _make_dsv4_state_cache():
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache

    c = DeepseekV4Cache(sliding_window=128, compress_ratio=4)
    local_k = mx.ones((1, 1, 7, 512), dtype=mx.float16)
    local_v = mx.ones((1, 1, 7, 512), dtype=mx.float16) * 2
    comp_buf = mx.ones((1, 3, 512), dtype=mx.float16) * 3
    comp_gate = mx.ones((1, 3, 512), dtype=mx.float16) * 4
    comp_pool = mx.ones((1, 2, 512), dtype=mx.float16) * 5
    idx_buf = mx.ones((1, 3, 512), dtype=mx.float16) * 6
    idx_gate = mx.ones((1, 3, 512), dtype=mx.float16) * 7
    idx_pool = mx.ones((1, 2, 512), dtype=mx.float16) * 8
    c.state = (
        (local_k, local_v),
        (comp_buf, comp_gate, comp_pool),
        (idx_buf, idx_gate, idx_pool),
    )
    c.meta_state = ("0", "128", "7", "7")
    return c


def _state_dict(c):
    return {
        "state": c.state,
        "meta_state": c.meta_state,
        "class_name": "DeepseekV4Cache",
        "compress_ratio": 4,
        "sliding_window": 128,
    }


def test_dsv4_block_slice_uses_deepseek_v4_tag_only_on_terminal_block():
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=PagedCacheManager(4, 8))
    c = _make_dsv4_state_cache()

    non_last = pc._extract_block_tensor_slice([_state_dict(c)], 0, 4, is_last_block=False)
    last = pc._extract_block_tensor_slice([_state_dict(c)], 4, 7, is_last_block=True)

    assert non_last[0][0] == "deepseek_v4_pending"
    assert non_last[0][2]["compress_ratio"] == 4
    assert last is not None
    assert last[0][0] == "deepseek_v4"
    assert last[0][4]["compress_ratio"] == 4
    assert last[0][4]["sliding_window"] == 128


def test_dsv4_block_disk_serialization_round_trips_nested_state():
    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block

    c = _make_dsv4_state_cache()
    block = [(
        "deepseek_v4",
        c.state,
        c.meta_state,
        "DeepseekV4Cache",
        {"compress_ratio": 4, "sliding_window": 128},
    )]

    tensors, dtype, num_layers = _serialize_block(block)
    restored = _deserialize_block(dict(tensors), dtype)

    assert num_layers == 1
    assert restored[0][0] == "deepseek_v4"
    assert restored[0][4] == {"compress_ratio": 4, "sliding_window": 128}
    local_state, compressor_state, indexer_state = restored[0][1]
    assert local_state[0].shape == (1, 1, 7, 512)
    assert compressor_state[2].shape == (1, 2, 512)
    assert indexer_state[2].shape == (1, 2, 512)


def test_dsv4_pending_marker_round_trips_for_l2_chain_blocks():
    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block

    block = [(
        "deepseek_v4_pending",
        "DeepseekV4Cache",
        {"compress_ratio": 128, "sliding_window": 128},
    )]

    tensors, dtype, num_layers = _serialize_block(block)
    restored = _deserialize_block(dict(tensors), dtype)

    assert num_layers == 1
    assert restored == [(
        "deepseek_v4_pending",
        "DeepseekV4Cache",
        {"compress_ratio": 128, "sliding_window": 128},
    )]


def test_dsv4_l2_pending_chain_without_terminal_is_unsafe():
    from types import SimpleNamespace

    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    pending = SimpleNamespace(cache_data=[
        ("kv", "k", "v"),
        (
            "deepseek_v4_pending",
            "DeepseekV4Cache",
            {"compress_ratio": 4, "sliding_window": 128},
        ),
    ])
    terminal = SimpleNamespace(cache_data=[
        (
            "deepseek_v4",
            ("local", "compressor", "indexer"),
            ("0", "128", "7", "7"),
            "DeepseekV4Cache",
            {"compress_ratio": 4, "sliding_window": 128},
        )
    ])

    assert BlockAwarePrefixCache._dsv4_l2_chain_missing_terminal_state([pending])
    assert not BlockAwarePrefixCache._dsv4_l2_chain_missing_terminal_state(
        [pending, terminal]
    )


def test_dsv4_in_memory_pending_chain_without_terminal_is_a_miss():
    from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(block_size=4, max_blocks=8)
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)

    tokens = [11, 12, 13, 14]
    block = paged.allocate_block()
    block.token_count = len(tokens)
    block.cache_data = [(
        "deepseek_v4_pending",
        "DeepseekV4Cache",
        {"compress_ratio": 4, "sliding_window": 128},
    )]
    block_hash = compute_block_hash(None, tokens)
    block.block_hash = block_hash
    paged.cached_block_hash_to_block.insert(block_hash, block)

    table, remaining = pc.fetch_cache("dsv4-pending-only", tokens + [15])

    assert table is None
    assert remaining == tokens + [15]


def test_dsv4_paged_reconstruct_returns_deepseek_cache_not_ssm_partial():
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=PagedCacheManager(4, 8))
    c = _make_dsv4_state_cache()
    tokens = [11, 12, 13, 14, 15, 16, 17]
    table = pc.store_cache("dsv4-test", tokens, [_state_dict(c)])

    rebuilt = pc.reconstruct_cache(table)

    assert rebuilt is not None
    assert len(rebuilt) == 1
    assert isinstance(rebuilt[0], DeepseekV4Cache)
    assert rebuilt[0].compress_ratio == 4
    assert rebuilt[0].state[1][2].shape == (1, 2, 512)


def test_dsv4_store_does_not_reuse_legacy_content_hash_for_repeated_blocks():
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=PagedCacheManager(4, 8))
    c = _make_dsv4_state_cache()
    # Two identical 4-token chunks under different prefix history. Legacy
    # content-only hashes would collapse these onto one block, which is invalid
    # for DSV4 because CSA/HSA pool state depends on the whole prefix.
    tokens = [1, 2, 3, 4, 1, 2, 3, 4, 5, 6]

    table = pc.store_cache("dsv4-repeated", tokens, [_state_dict(c)])

    assert table is not None
    assert table.num_tokens == len(tokens)
    assert len(table.block_ids) == 3
    assert len(set(table.block_ids)) == 3


def test_dsv4_fetch_prefers_n_minus_one_terminal_partial_after_restart():
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(block_size=4, max_blocks=8)
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    c = _make_dsv4_state_cache()
    # Scheduler stores DSV4 under N-1 prompt tokens so the last prompt token can
    # be re-fed for first-token logits. Store 6 tokens, then fetch with the full
    # 7-token prompt.
    stored_tokens = [10, 11, 12, 13, 14, 15]
    full_prompt_tokens = stored_tokens + [16]
    table = pc.store_cache("dsv4-nminus-store", stored_tokens, [_state_dict(c)])
    assert table is not None

    # Simulate process restart: full blocks are discoverable from L2/index, but
    # in-memory knowledge of terminal partial sizes is gone. The full-N lookup
    # would see one full block and miss a 3-token partial; the N-1 lookup must
    # still find the 2-token terminal partial carrying deepseek_v4 state.
    paged._partial_block_sizes.clear()

    hit_table, remaining = pc.fetch_cache("dsv4-nminus-fetch", full_prompt_tokens)
    rebuilt = pc.reconstruct_cache(hit_table)

    assert hit_table is not None
    assert hit_table.num_tokens == len(stored_tokens)
    assert remaining == [16]
    assert rebuilt is not None
    assert len(rebuilt) == 1
    assert isinstance(rebuilt[0], DeepseekV4Cache)


# ============================================================================
# DSV4 SWA+CSA+HSA cache truncation guard (scheduler.py:1964) regression tests
# ============================================================================

def _bootstrap_dsv4_cache_offset(c, offset_value):
    """DeepseekV4Cache.offset is a @property delegating to self.local.offset.
    The property may be settable; if not, we set the underlying RotatingKVCache."""
    try:
        c.local.offset = offset_value
    except Exception:
        pass
    return c


def test_dsv4_truncation_refuses_post_generation_cache():
    """Pin the SWA+CSA+HSA truncation guard.

    `_truncate_cache_to_prompt_length` MUST return None for DSV4 when the
    live cache has advanced past the prompt boundary (current_len >
    target_len). Storing the trimmed state would corrupt next-turn decode
    because:
      - SWA RotatingKVCache cannot be rewound after wrap (offset > max_size)
      - CSA pool buffers are cumulative across the entire window
      - HSA indexer pool is cumulative; trim drops trailing rows but
        boundary may not align with prompt/output split

    Regression target: scheduler.py:1964 DSV4 branch must `return None` when
    to_trim > 0 unless `VMLX_DSV4_TRUST_TRIMMED_CACHE=1`.
    """
    import os
    from unittest.mock import MagicMock
    from vmlx_engine.scheduler import Scheduler

    # Ensure the unsafe override is OFF for this test
    prev = os.environ.pop("VMLX_DSV4_TRUST_TRIMMED_CACHE", None)
    try:
        c = _make_dsv4_state_cache()
        # Simulate post-generation: local SWA has wrapped (offset=600 > 128).
        # `local.offset` is read by the guard.
        # _make_dsv4_state_cache builds with sliding_window=128.
        # We need the live cache to claim it has advanced past prompt.
        # Mock its `offset` attribute directly.
        c.local.offset = 600  # post-generation, wrapped
        # `layer_cache.offset` is read for current_len.
        # DeepseekV4Cache.offset returns local.offset by convention.
        target_len = 28  # prompt-only target
        # _truncate_cache_to_prompt_length is @staticmethod — call directly.
        result = Scheduler._truncate_cache_to_prompt_length([c], target_len)
        assert result is None, (
            "DSV4 truncation guard MUST return None when to_trim>0; "
            "guard at scheduler.py:1964 is regressed."
        )
    finally:
        if prev is not None:
            os.environ["VMLX_DSV4_TRUST_TRIMMED_CACHE"] = prev


def test_dsv4_truncation_allows_zero_trim_clean_state():
    """Pin: the guard does NOT block clean-boundary stores.

    When target_len == current_len (no trim needed), the cache is at the
    prompt boundary (no decode happened yet, or caller captured a clean
    snapshot). Guard should return a valid truncated list.
    """
    from vmlx_engine.scheduler import Scheduler

    c = _make_dsv4_state_cache()
    # Simulate fresh-prefilled cache (no decode yet): offset == target.
    c.local.offset = 28  # at prompt boundary, no wrap
    # _truncate_cache_to_prompt_length is @staticmethod — call directly.
    result = Scheduler._truncate_cache_to_prompt_length([c], 28)
    # to_trim == 0 → guard does not fire, returns the rebuilt cache list
    assert result is not None
    assert len(result) == 1


def test_dsv4_unsafe_override_in_cache_scope_key():
    """Pin: when VMLX_DSV4_TRUST_TRIMMED_CACHE is set, the dsv4 cache scope
    key includes that env so debug runs don't share namespace with safe runs.

    Regression target: scheduler.py:~595 dsv4_scope must include
    `dsv4_unsafe_trim={0,1}` so block-disk caches written under `=1` cannot
    be replayed when the override is later disabled.
    """
    # We assert the source contains the scope key contribution, not that the
    # full block_scope_key is computed (that requires a full Scheduler
    # instance with live model state).
    import inspect
    from vmlx_engine import scheduler

    src = inspect.getsource(scheduler)
    assert "dsv4_unsafe_trim" in src, (
        "scheduler.py block_scope_key MUST include dsv4_unsafe_trim={0,1} "
        "so VMLX_DSV4_TRUST_TRIMMED_CACHE=1 debug runs don't share L2 disk "
        "namespace with default safe runs."
    )
    assert "VMLX_DSV4_TRUST_TRIMMED_CACHE" in src
