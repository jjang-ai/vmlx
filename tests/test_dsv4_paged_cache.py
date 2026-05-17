# SPDX-License-Identifier: Apache-2.0
"""DSV4 paged/L2 cache contract tests.

DSV4 uses a composite cache, not a plain KV cache:
DeepseekV4Cache.state = (local_swa_state, compressor_state, indexer_state).
The block cache must preserve that nested state exactly enough for
reconstruction and L2 disk promotion.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

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


def test_pool_quantized_v4_cache_is_detected_as_dsv4_composite():
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from vmlx_engine.prefix_cache import _is_dsv4_cache_class
    from vmlx_engine.scheduler import Scheduler

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)

    assert Scheduler._is_dsv4_cache_object(cache)
    assert _is_dsv4_cache_class("PoolQuantizedV4Cache")


def test_pool_quantized_v4_cache_does_not_route_to_hybrid_ssm():
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.scheduler import Scheduler

    class _Model:
        def make_cache(self):
            return [KVCache(), PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)]

    assert Scheduler._model_uses_dsv4_cache(_Model())
    assert not Scheduler._is_hybrid_model(_Model())


def test_dsv4_pool_quant_default_is_correctness_safe_opt_in(monkeypatch):
    """DSV4 pool quant must not auto-enable just because the class exists."""
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _configure_dsv4_pool_quant_default,
    )

    monkeypatch.delenv("DSV4_LONG_CTX", raising=False)
    monkeypatch.delenv("DSV4_POOL_QUANT", raising=False)

    assert _configure_dsv4_pool_quant_default() == "0"
    assert os.environ["DSV4_LONG_CTX"] == "1"
    assert os.environ["DSV4_POOL_QUANT"] == "0"

    monkeypatch.setenv("DSV4_POOL_QUANT", "1")
    assert _configure_dsv4_pool_quant_default() == "1"


def test_dsv4_cli_cache_policy_uses_ds4_page_sized_blocks(caplog):
    """DSV4 serve auto-config must not inherit generic 64-token pages.

    DeepSeek-V4 Flash uses a composite SWA+CSA/HCA decode cache; the paged
    prefix layer stores prompt-boundary snapshots of that composite state,
    not plain KV. The production serve path therefore needs the DS4/vLLM-style
    256-token block size even when a stale panel/session config still carries
    the generic 64-token default.
    """
    from vmlx_engine.cli import _apply_dsv4_cache_policy

    args = SimpleNamespace(
        enable_prefix_cache=True,
        disable_prefix_cache=False,
        use_paged_cache=False,
        paged_cache_block_size=64,
        enable_block_disk_cache=True,
    )

    changed = _apply_dsv4_cache_policy(args, logger=__import__("logging").getLogger("test"))

    assert args.use_paged_cache is True
    assert args.paged_cache_block_size == 256
    assert "paged=required_for_dsv4_composite" in changed
    assert "block_size=64->256" in changed


def test_dsv4_runtime_policy_applies_to_bench_like_cli_args(tmp_path, monkeypatch):
    """Bench/CLI paths must not bypass the serve-time DSV4 cache gates."""
    from vmlx_engine.cli import _apply_dsv4_runtime_policy

    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v4"}')
    args = SimpleNamespace(
        model=str(tmp_path),
        continuous_batching=False,
        enable_prefix_cache=True,
        disable_prefix_cache=False,
        use_paged_cache=False,
        paged_cache_block_size=64,
        enable_block_disk_cache=True,
        kv_cache_quantization="q4",
        kv_cache_quantization_explicit=True,
        max_num_seqs=9,
        prefill_batch_size=2048,
        prefill_step_size=4096,
        completion_batch_size=99,
        no_memory_aware_cache=True,
        enable_disk_cache=True,
        enable_jit=True,
        smelt=True,
        flash_moe=True,
        distributed=True,
        speculative_model="draft",
    )
    monkeypatch.delenv("DSV4_LONG_CTX", raising=False)
    monkeypatch.delenv("DSV4_POOL_QUANT", raising=False)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")

    applied, changes = _apply_dsv4_runtime_policy(
        args,
        logger=__import__("logging").getLogger("test"),
        clamp_max_num_seqs=True,
    )

    assert applied is True
    assert args.use_paged_cache is True
    assert args.paged_cache_block_size == 256
    assert args.kv_cache_quantization == "none"
    assert args.kv_cache_quantization_explicit is True
    assert args.max_num_seqs == 1
    assert args.continuous_batching is True
    assert args.disable_prefix_cache is False
    assert args.prefill_batch_size == 1
    assert args.completion_batch_size == 1
    assert args.no_memory_aware_cache is False
    assert args.enable_disk_cache is False
    assert args.enable_jit is False
    assert args.smelt is False
    assert args.flash_moe is False
    assert args.distributed is False
    assert args.speculative_model is None
    assert os.environ["DSV4_LONG_CTX"] == "1"
    assert os.environ["DSV4_POOL_QUANT"] == "0"
    assert os.environ["VMLX_DISABLE_TQ_KV"] == "1"
    assert "VMLINUX_FORCE_TQ_AUTO" not in os.environ
    assert "continuous_batching=off->on" in changes
    assert "paged=required_for_dsv4_composite" in changes
    assert "block_size=64->256" in changes
    assert "max_num_seqs=9->1" in changes
    assert "prefill_batch_size=2048->1" in changes
    assert "completion_batch_size=99->1" in changes
    assert "no_memory_aware_cache=off" in changes
    assert "legacy_disk_cache=off" in changes
    assert "enable_jit=off" in changes
    assert "smelt=off" in changes
    assert "flash_moe=off" in changes
    assert "distributed=off" in changes
    assert "speculative_model=off" in changes


def test_panel_suppresses_generic_kv_quantization_controls_for_dsv4():
    """The app UI/launch preview must not advertise generic KV q4/q8 for DSV4."""
    from pathlib import Path

    form = Path("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").read_text()
    settings = Path("panel/src/renderer/src/components/sessions/SessionSettings.tsx").read_text()
    sessions = Path("panel/src/main/sessions.ts").read_text()

    assert "const effectiveStoredCacheQuantization = dsv4Active ? 'auto'" in form
    assert "disabled={effectivelyNoBatching || prefixOff || dsv4Active}" in form
    assert "!dsv4Active && config.kvCacheQuantization" in settings
    assert "detectedFamily !== 'deepseek-v4' && config.kvCacheQuantization" in sessions
    assert "if (family === 'deepseek_v4') return 'deepseek-v4'" in form
    assert "if (family === 'deepseek_v4') return 'deepseek-v4'" in settings
    assert "if (family === 'deepseek_v4') return 'deepseek-v4'" in sessions


def test_panel_suppresses_generic_batch_and_chunk_controls_for_dsv4():
    """DSV4 app launches must not pass misleading generic batch/chunk flags."""
    from pathlib import Path

    form = Path("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").read_text()
    settings = Path("panel/src/renderer/src/components/sessions/SessionSettings.tsx").read_text()
    sessions = Path("panel/src/main/sessions.ts").read_text()

    assert "const effectiveMaxNumSeqs = dsv4Active ? 1 : config.maxNumSeqs" in form
    assert "const effectivePrefillBatchSize = dsv4Active ? 1 : config.prefillBatchSize" in form
    assert "const effectiveCompletionBatchSize = dsv4Active ? 1 : config.completionBatchSize" in form
    assert "if (!dsv4Active && config.prefillBatchSize" in sessions
    assert "if (!dsv4Active && config.prefillStepSize" in sessions
    assert "if (!dsv4Active && config.completionBatchSize" in sessions
    assert "if (!dsv4Active && config.prefillBatchSize" in settings
    assert "if (!dsv4Active && config.prefillStepSize" in settings
    assert "if (!dsv4Active && config.completionBatchSize" in settings


def test_dsv4_ui_forces_native_cache_stack_and_hides_unsafe_runtime_controls():
    """DSV4 settings must show only production-safe controls or disabled values."""
    from pathlib import Path

    form = Path("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").read_text()

    assert "const effectiveContinuousBatching = dsv4Active ? true : config.continuousBatching" in form
    assert "const prefixOff = dsv4Active ? false : !config.enablePrefixCache" in form
    assert "const multimodalActive = !dsv4Active" in form
    assert "checked={effectiveContinuousBatching}" in form
    assert "checked={dsv4Active ? true : config.enablePrefixCache}" in form
    assert "hidden={isImage || dsv4Active}" in form
    assert "const showVideoControls = !dsv4Active" in form
    assert "checked={effectiveSmeltActive}" in form
    assert "disabled={dsv4Active || effectiveFlashMoeActive}" in form
    assert "checked={effectiveFlashMoeActive}" in form
    assert "disabled={dsv4Active || effectiveSmeltActive || effectiveDistributedActive}" in form
    assert "!dsv4Active && !smeltActive && !detectedForceTextOnly && config.isMultimodal === true" in form
    assert "!dsv4Active && !smeltActive && !detectedForceTextOnly && config.isMultimodal === false" in form


def test_dsv4_launch_filters_stale_saved_and_additional_args():
    """Saved sessions and raw additionalArgs must not reintroduce invalid DSV4 flags."""
    from pathlib import Path

    settings = Path("panel/src/renderer/src/components/sessions/SessionSettings.tsx").read_text()
    sessions = Path("panel/src/main/sessions.ts").read_text()

    for source in (settings, sessions):
        assert "const cacheStackActive = dsv4Active ? true : config.continuousBatching !== false" in source
        assert "const prefixCacheOff = dsv4Active ? false" in source
        assert "const effectiveSmelt = !!(config as any).smelt && !dsv4Active" in source
        assert "const isVLM = dsv4Active || effectiveSmelt" in source
        assert "const effectiveDistributed = requestedDistributed && !dsv4Active" in source
        assert "const effectiveFlashMoe = requestedFlashMoe && !effectiveDistributed && !dsv4Active" in source
        assert "if (!dsv4Active && config.speculativeModel)" in source
        assert "DSV4_ADDITIONAL_ARG_BLOCKLIST" in source
        assert "--no-continuous-batching" in source
        assert "--disable-prefix-cache" in source
        assert "--kv-cache-quantization" in source
        assert "--enable-jit" in source
        assert "--smelt" in source
        assert "--flash-moe" in source
        assert "--distributed" in source
        assert "--speculative-model" in source
        assert "--stream-interval" in source
        assert "--tool-call-parser" in source
        assert "--reasoning-parser" in source
    assert "config.isMultimodal = false" in sessions


def test_dsv4_block_l2_namespace_includes_paged_block_size():
    """DSV4 L2 namespaces must not mix 64-token and 256-token block records."""
    import inspect
    from vmlx_engine.scheduler import Scheduler

    source = inspect.getsource(Scheduler.__init__)

    assert ":dsv4_paged_block_size={self.config.paged_cache_block_size}" in source


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


def test_dsv4_numpy_disk_slice_keeps_composite_layers_when_only_two_kv_sources():
    import numpy as np

    from vmlx_engine.prefix_cache import _numpy_block_slice

    keys = mx.ones((1, 1, 7, 8), dtype=mx.float16)
    values = mx.ones((1, 1, 7, 8), dtype=mx.float16) * 2
    kv_state = {
        "state": (keys, values),
        "class_name": "KVCache",
        "meta_state": (7,),
    }
    dsv4_state = _state_dict(_make_dsv4_state_cache())
    cache_data = [kv_state, kv_state] + [dsv4_state for _ in range(41)]
    np_sources = {
        0: (np.array(keys), np.array(values), keys.dtype),
        1: (np.array(keys), np.array(values), keys.dtype),
    }

    non_terminal = _numpy_block_slice(
        cache_data, np_sources, 0, 4, is_last_block=False
    )
    terminal = _numpy_block_slice(
        cache_data, np_sources, 4, 7, is_last_block=True
    )

    assert non_terminal is not None
    assert terminal is not None
    assert len(non_terminal) == 43
    assert len(terminal) == 43
    assert sum(1 for entry in non_terminal if entry[0] == "kv") == 2
    assert sum(1 for entry in non_terminal if entry[0] == "deepseek_v4_pending") == 41
    assert sum(1 for entry in terminal if entry[0] == "kv") == 2
    assert sum(1 for entry in terminal if entry[0] == "deepseek_v4") == 41
    local_state, compressor_state, indexer_state = terminal[2][1]
    assert len(local_state) == 2
    assert len(compressor_state) == 3
    assert len(indexer_state) == 3
    assert compressor_state[2].shape == (1, 2, 512)
    assert indexer_state[2].shape == (1, 2, 512)


def test_dsv4_block_disk_log_summarizes_native_composite_tags(caplog):
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    class _DummyDisk:
        def write_block_async(self, *_args, **_kwargs):
            return None

    caplog.set_level("INFO", logger="vmlx_engine.prefix_cache")
    paged = PagedCacheManager(block_size=4, max_blocks=8, disk_store=_DummyDisk())
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    c = _make_dsv4_state_cache()

    pc.store_cache(
        "dsv4-log-summary",
        [11, 12, 13, 14, 15, 16, 17],
        [_state_dict(c)],
    )

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("deepseek_v4_pending=1" in msg for msg in messages)
    assert any("deepseek_v4=1" in msg for msg in messages)


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


def test_dsv4_frugal_store_keeps_terminal_composite_block_in_ram(monkeypatch):
    """Immediate same-process DSV4 hits must not depend on async L2 visibility.

    DSV4 non-terminal blocks are only pending markers. The terminal block is
    the one that carries the full SWA+CSA/HCA composite state. If frugal mode
    skips that terminal in-RAM mirror, an immediate repeat can find the block
    table but reconstruct None until the async block-disk write becomes visible.
    """
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    class _DummyDisk:
        def write_block_async(self, *_args, **_kwargs):
            return None

    monkeypatch.delenv("VMLX_PAGED_FRUGAL", raising=False)
    paged = PagedCacheManager(block_size=4, max_blocks=8, disk_store=_DummyDisk())
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)

    c = _make_dsv4_state_cache()
    table = pc.store_cache(
        "dsv4-frugal-terminal",
        [11, 12, 13, 14, 15, 16, 17],
        [_state_dict(c)],
    )

    assert table is not None
    for block_id in table.block_ids:
        assert paged.allocated_blocks[block_id].cache_data is not None
    terminal_block = paged.allocated_blocks[table.block_ids[-1]]
    assert terminal_block.cache_data[0][0] == "deepseek_v4"
    assert pc.reconstruct_cache(table) is not None


def test_dsv4_store_does_not_reuse_legacy_content_hash_for_repeated_blocks():
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=PagedCacheManager(4, 8))
    c = _make_dsv4_state_cache()
    # Two identical 4-token chunks under different prefix history. Legacy
    # content-only hashes would collapse these onto one block, which is invalid
    # for DSV4 because CSA/HCA pool state depends on the whole prefix.
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


def test_dsv4_trim_block_table_to_terminal_state_keeps_safe_prefix_only():
    """Memory-fit DSV4 reuse may shrink only to a terminal composite block.

    A block carrying ``deepseek_v4_pending`` is not a usable final cache state:
    it has SWA/front-layer pieces but not the CSA/HCA composite pools. A block
    carrying ``deepseek_v4`` is a clean terminal checkpoint and can be reused as
    the shortened cached prefix.
    """
    from vmlx_engine.paged_cache import BlockTable, PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(block_size=4, max_blocks=8)
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    table = pc.store_cache(
        "dsv4-terminal-trim",
        [1, 2, 3, 4, 5, 6],
        [_state_dict(_make_dsv4_state_cache())],
    )
    assert table is not None
    assert len(table.block_ids) == 2

    # Simulate a later cache hit that appended another pending block after the
    # terminal checkpoint. The safe shrink target is the terminal checkpoint,
    # not the later pending block.
    trailing = paged.allocate_block()
    assert trailing is not None
    trailing.token_count = 4
    trailing.cache_data = [("deepseek_v4_pending", "DeepseekV4Cache")]
    trailing.ref_count = 1
    paged.allocated_blocks[trailing.block_id] = trailing

    live = BlockTable(
        request_id="dsv4-live",
        block_ids=[*table.block_ids, trailing.block_id],
        num_tokens=10,
    )
    paged.request_tables["dsv4-live"] = live

    trimmed = pc.trim_block_table_to_terminal_state(
        "dsv4-live",
        target_tokens=10,
        tag="deepseek_v4",
    )

    assert trimmed is not None
    assert trimmed.block_ids == table.block_ids
    assert trimmed.num_tokens == 6
    assert trailing.ref_count == 0


def test_dsv4_trim_block_table_to_terminal_state_refuses_pending_only_prefix():
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(block_size=4, max_blocks=8)
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    table = pc.store_cache(
        "dsv4-pending-only-trim",
        [1, 2, 3, 4, 5, 6],
        [_state_dict(_make_dsv4_state_cache())],
    )
    assert table is not None

    trimmed = pc.trim_block_table_to_terminal_state(
        "dsv4-pending-only-trim",
        target_tokens=4,
        tag="deepseek_v4",
    )

    assert trimmed is None


def test_dsv4_storage_quantization_is_forced_off_for_composite_cache():
    """DSV4 prefix/paged/L2 storage must keep the native composite cache.

    DeepseekV4Cache already contains SWA local cache plus compressed CSA/HCA
    pools. The only DSV4-supported compression layer is the native
    PoolQuantizedV4Cache pool codec; generic QuantizedKVCache must not wrap
    the local SWA state at prefix/paged/L2 boundaries.
    """
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache
    from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache
    from vmlx_engine.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_cache_bits = 4
    scheduler._kv_cache_group_size = 64

    source = _make_dsv4_state_cache()
    stored = scheduler._quantize_cache_for_storage([source])

    assert stored[0] is source
    assert isinstance(stored[0], DeepseekV4Cache)
    assert isinstance(stored[0].local, RotatingKVCache)
    assert not isinstance(stored[0].local, QuantizedKVCache)
    assert not hasattr(stored[0], "_vmlx_dsv4_local_quant_meta")


def test_dsv4_scheduler_forces_generic_kv_quantization_off():
    """SchedulerConfig q4/q8 must not enable generic KV quant for DSV4."""
    from types import SimpleNamespace

    from jang_tools.dsv4.mlx_model import DeepseekV4Cache
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    class _Tokenizer:
        eos_token_id = 1
        name_or_path = "DeepSeek-V4-Flash-JANGTQ"

        def encode(self, *_args, **_kwargs):
            return [1]

    class _Model:
        args = SimpleNamespace(model_type="deepseek_v4", kv_lora_rank=512)
        config = {"model_type": "deepseek_v4"}

        def make_cache(self):
            return [DeepseekV4Cache(sliding_window=128, compress_ratio=4)]

    config = SchedulerConfig(
        enable_prefix_cache=True,
        use_paged_cache=True,
        kv_cache_quantization="q4",
        model_path="/models/DeepSeek-V4-Flash-JANGTQ",
    )
    scheduler = Scheduler(_Model(), _Tokenizer(), config)

    assert scheduler._uses_dsv4_cache
    assert scheduler.config.kv_cache_quantization == "none"
    assert scheduler._kv_cache_bits == 0


def test_dsv4_native_pool_codec_stays_distinct_from_generic_kv_quant():
    """Pin the intended split: native pool quant yes, generic KV quant no."""
    import inspect
    from vmlx_engine.scheduler import Scheduler

    init_src = inspect.getsource(Scheduler.__init__)
    quant_src = inspect.getsource(Scheduler._quantize_cache_for_storage)

    assert "DSV4 composite cache detected" in init_src
    assert "DSV4_POOL_QUANT" in init_src
    assert "wrap any component in generic QuantizedKVCache" in quant_src


def test_dsv4_serve_path_forces_generic_kv_quantization_off():
    """The CLI/app serve path must not pass q4/q8 generic KV quant to DSV4."""
    import inspect
    from vmlx_engine import cli

    serve_src = inspect.getsource(cli.serve_command)
    policy_src = inspect.getsource(cli._apply_dsv4_runtime_policy)

    assert "_apply_dsv4_runtime_policy(args, logger)" in serve_src
    assert 'args.kv_cache_quantization = "none"' in policy_src
    assert "DSV4-Flash native SWA+CSA/HCA cache owns cache" in policy_src
    assert 'os.environ["VMLX_DISABLE_TQ_KV"] = "1"' in policy_src


def test_dsv4_cached_prefix_kickoff_avoids_cross_thread_mx_eval():
    """DSV4 cache-hit kickoff must not use mx.eval on worker-thread tensors."""
    import inspect
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    source = inspect.getsource(DSV4BatchGenerator._prefill_last_logits)

    sync_idx = source.find('if hasattr(mx, "synchronize")')
    fallback_idx = source.find("mx.eval(last_logits)")

    assert sync_idx != -1
    assert fallback_idx != -1
    assert sync_idx < fallback_idx


def test_minimax_ling_do_not_use_long_repetition_context():
    """MiniMax/Ling prompts contain EOS/turn sentinels that must not be
    penalized through a widened 512-token repetition window."""
    import inspect
    from vmlx_engine.scheduler import Scheduler

    class LingModel:
        config = {"model_type": "bailing_hybrid"}

    class MiniMaxModel:
        config = {"model_type": "minimax_m2"}

    assert Scheduler._detect_model_type_for_runtime(LingModel()) == "bailing_hybrid"
    assert Scheduler._detect_model_type_for_runtime(MiniMaxModel()) == "minimax_m2"

    source = inspect.getsource(Scheduler.__init__)
    assignment = source.split("self._long_repetition_context =", 1)[1].split(
        "self._tq_active", 1
    )[0]
    assert '"deepseek_v4"' in assignment
    assert '"minimax_m2"' not in assignment
    assert '"bailing_hybrid"' not in assignment
    assert '"bailing_moe_v2_5"' not in assignment
    source = inspect.getsource(Scheduler._create_batch_generator)
    assert "_rep_context_size = 512 if self._long_repetition_context else 20" in source


def test_hybrid_ssm_rederive_uses_n_minus_one_cache_key():
    """Hybrid SSM companion must align with paged KV's N-1 cache key.

    The paged cache stores prompt[:-1] so cache hits re-feed the final prompt
    token. Storing clean SSM companion at the full-N prompt length guarantees
    every Ling/Bailing hit misses SSM and falls back to full prefill.
    """
    import inspect
    from vmlx_engine.scheduler import Scheduler

    source = inspect.getsource(Scheduler._cleanup_finished)
    assert "companion_tokens = (" in source
    assert "all_tokens[:-1]" in source
    assert "(list(companion_tokens), companion_len, request_id)" in source

    import vmlx_engine.scheduler as scheduler_mod

    assert scheduler_mod.SSM_REDERIVE_MIN_TOKENS == 1


def test_hybrid_ssm_companion_fetch_is_worker_deferred():
    """Hybrid SSM companion clone must not run on the API thread.

    Deferred re-derive stores MLX arrays on the scheduler worker stream.
    Fetching/cloning those arrays in add_request() makes valid companion
    entries look like misses due MLX's thread-local stream guard.
    """
    import inspect
    from vmlx_engine.scheduler import Scheduler

    add_src = inspect.getsource(Scheduler.add_request)
    schedule_src = inspect.getsource(Scheduler._schedule_waiting)
    finalize_src = inspect.getsource(Scheduler._finalize_hybrid_paged_cache_on_worker)

    assert "_hybrid_prompt_cache_needs_worker_ssm = True" in add_src
    assert "_paged_block_table_needs_worker_reconstruct = True" in add_src
    assert "_ssm_state_cache.fetch(" not in add_src
    assert "reconstruct_cache(block_table)" in schedule_src
    assert "_hybrid_prompt_cache_needs_worker_ssm" in schedule_src
    assert "_finalize_hybrid_paged_cache_on_worker" in schedule_src
    assert "_ssm_state_cache.fetch(" in finalize_src
    assert "hybrid paged HIT" in finalize_src


def test_hybrid_ssm_l2_is_model_scoped_and_block_disk_backed():
    """Hybrid SSM L2 must be wired with block-disk, not a hidden global env."""
    import inspect
    from vmlx_engine.scheduler import Scheduler

    init_src = inspect.getsource(Scheduler.__init__)
    stats_src = inspect.getsource(Scheduler._get_ssm_cache_stats)

    assert "compute_model_cache_key(" in init_src
    assert "model_key=_ssm_model_key" in init_src
    assert "attach_disk_store(_ssm_disk)" in init_src
    assert 'os.path.join(cache_dir, "ssm_companion")' in init_src
    assert "Hybrid SSM companion L2 enabled" in init_src
    assert "disk_enabled" in stats_src
    assert "disk.stats()" in stats_src


def test_bailing_mla_cache_uses_expanded_attention_heads():
    """Ling/Bailing MLA stores full per-head KV, not H=1 compressed latents."""
    from types import SimpleNamespace

    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.scheduler import Scheduler

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="bailing_hybrid",
            kv_lora_rank=512,
            num_attention_heads=32,
            num_key_value_heads=1,
        )
    )

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    assert scheduler._detect_n_kv_heads() == 32

    cache = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=PagedCacheManager(block_size=64, max_blocks=2),
    )
    assert cache._get_n_kv_heads() == 32
    assert cache._get_allowed_n_kv_heads() == {32}


def test_gemma4_nested_text_config_exposes_mixed_kv_head_counts():
    """Gemma 4 VLM wrappers store SWA/full KV heads under config.text_config."""
    from types import SimpleNamespace

    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.scheduler import Scheduler

    text_cfg = SimpleNamespace(
        model_type="gemma4_text",
        num_key_value_heads=8,
        num_global_key_value_heads=2,
    )
    model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4", text_config=text_cfg))

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    assert scheduler._detect_n_kv_heads() == 8

    cache = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=PagedCacheManager(block_size=64, max_blocks=2),
    )
    assert cache._get_n_kv_heads() == 8
    assert cache._get_allowed_n_kv_heads() == {2, 8}


# ============================================================================
# DSV4 SWA+CSA+HCA cache truncation guard (scheduler.py:1964) regression tests
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
    """Pin the SWA+CSA+HCA truncation guard.

    `_truncate_cache_to_prompt_length` MUST return None for DSV4 when the
    live cache has advanced past the prompt boundary (current_len >
    target_len). Storing the trimmed state would corrupt next-turn decode
    because:
      - SWA RotatingKVCache cannot be rewound after wrap (offset > max_size)
      - CSA pool buffers are cumulative across the entire window
      - HCA/indexer pool is cumulative; trim drops trailing rows but
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

    `_truncate_cache_to_prompt_length` stores prompt_len - 1 tokens, because
    cache hits re-feed the last prompt token for first-token logits. When a
    caller already passes an N-1 prompt-boundary snapshot, the guard should
    return a valid truncated list.
    """
    from vmlx_engine.scheduler import Scheduler

    c = _make_dsv4_state_cache()
    # Simulate a clean N-1 prompt-boundary snapshot: offset == prompt_len - 1.
    c.local.offset = 27
    # _truncate_cache_to_prompt_length is @staticmethod — call directly.
    result = Scheduler._truncate_cache_to_prompt_length([c], 28)
    # to_trim == 0 -> guard does not fire, returns the rebuilt cache list
    assert result is not None
    assert len(result) == 1


def test_dsv4_length_capped_clean_snapshot_is_cacheable():
    """Length-capped DSV4 output can still donate a prompt snapshot.

    The unsafe case is trimming the live post-generation DeepseekV4Cache.
    DSV4BatchGenerator now captures an N-1 prompt-boundary snapshot before
    decode, so capped generations with that snapshot must still reach paged/L2
    storage.
    """
    import inspect
    from vmlx_engine import scheduler

    src = (
        inspect.getsource(scheduler.Scheduler.step)
        + inspect.getsource(scheduler.Scheduler._cleanup_finished)
    )

    assert "_extracted_cache_from_prompt_snapshot" in src
    assert "RequestStatus.FINISHED_LENGTH_CAPPED" in src
    assert (
        'and not getattr(request, "_extracted_cache_from_prompt_snapshot", False)'
        in src
    )


def test_dsv4_cache_hit_store_skips_sync_full_reprefill_when_snapshot_missing():
    """DSV4 cache-hit kickoff must not synchronously re-prefill long prompts.

    On a paged-prefix hit, DSV4BatchGenerator starts from a restored terminal
    DeepseekV4Cache checkpoint and processes only the remaining prompt tail.
    That path can finish with ``prompt_cache_snapshot=None``. The live cache is
    then post-decode-contaminated and must not be trimmed; but re-prefilling the
    entire expanded prompt before returning the response makes long-context
    "cache hits" slow again. Keep the existing terminal N-1 cache point and skip
    synchronous extension-store until there is an async store path.
    """
    import inspect
    from vmlx_engine import scheduler

    src = inspect.getsource(scheduler.Scheduler._process_batch_responses)

    assert "DSV4 prefix cache store skipped" in src
    assert "avoiding synchronous full" in src
    assert "cached_tokens" in src
    assert "clean prompt-boundary re-prefill" in src
    assert "dsv4_key_tokens" in src
    assert "_prefill_for_prompt_only_cache" in src
    helper_src = inspect.getsource(scheduler.Scheduler._prefill_for_prompt_only_cache)
    assert "chunk_size = len(prompt_tokens) if self._uses_dsv4_cache else 2048" in helper_src


def test_dsv4_long_prefill_guard_describes_single_shot_default():
    """The DSV4 long-prefill guard must not claim chunked prefill is default.

    DSV4BatchGenerator deliberately defaults to single-shot prefill because
    arbitrary chunking has not been proven safe for SWA+CSA/HCA composite state.
    """
    import inspect

    from vmlx_engine import scheduler

    guard_src = inspect.getsource(scheduler.Scheduler.add_request)

    assert "defaults to single-shot" in guard_src
    assert "uses chunked prefill" not in guard_src


def test_dsv4_prompt_only_prefill_collects_composite_state_without_values_attr():
    """DeepseekV4Cache has `.keys` but no top-level `.values` property.

    The prompt-only re-derive path must collect the nested composite state tree
    instead of treating DSV4 cache objects as plain KVCache instances. Otherwise
    cache-hit turns decode correctly but cannot donate the extended prefix.
    """
    from types import SimpleNamespace

    from vmlx_engine.scheduler import Scheduler

    class _Model:
        def make_cache(self):
            return [_make_dsv4_state_cache()]

        def __call__(self, input_ids, cache=None):
            return SimpleNamespace(logits=input_ids)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = _Model()
    scheduler._uses_dsv4_cache = True

    cache = scheduler._prefill_for_prompt_only_cache([1, 2, 3])

    assert cache is not None
    assert type(cache[0]).__name__ == "DeepseekV4Cache"


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
