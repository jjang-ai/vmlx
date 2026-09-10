# SPDX-License-Identifier: Apache-2.0
"""DSV4 paged/L2 cache contract tests.

DSV4 uses a composite cache, not a plain KV cache:
DeepseekV4Cache.state = (local_swa_state, compressor_state, indexer_state).
The block cache must preserve that nested state exactly enough for
reconstruction and L2 disk promotion.
"""

from __future__ import annotations

import os
import json
from types import SimpleNamespace

import pytest


mx = pytest.importorskip("mlx.core")


def _restore_process_env_after_test(request, *names):
    """Restore variables mutated directly by CLI policy helpers."""
    originals = {name: os.environ.get(name) for name in names}

    def _restore():
        for name, value in originals.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    request.addfinalizer(_restore)


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


def _make_pool_quantized_dsv4_state_cache():
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    c = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    base = _make_dsv4_state_cache()
    c.state = base.state
    c.meta_state = base.meta_state
    return c


def _state_dict(c):
    state = (
        c.storage_state
        if type(c).__name__ == "PoolQuantizedV4Cache"
        and hasattr(c, "storage_state")
        else c.state
    )
    result = {
        "state": state,
        "meta_state": c.meta_state,
        "class_name": type(c).__name__,
        "compress_ratio": 4,
        "sliding_window": 128,
        "pool_quant": type(c).__name__ == "PoolQuantizedV4Cache",
    }
    if type(c).__name__ == "PoolQuantizedV4Cache" and hasattr(c, "storage_state"):
        result["pool_storage_schema"] = state[0]
    return result


def _make_encoded_pool_quantized_dsv4_cache(monkeypatch):
    import jang_tools.dsv4.pool_quant_cache as pool_quant_cache
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pool_quant_cache, "_POOL_BF16_MAX_BYTES", 1)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    local_k = mx.random.normal((1, 1, 7, 64), dtype=mx.bfloat16)
    local_v = mx.random.normal((1, 1, 7, 64), dtype=mx.bfloat16)
    cache.local.state = (local_k, local_v)
    cache.meta_state = ("0", "128", "7", "7")
    cache.update_pool(
        mx.random.normal((1, 129, 64), dtype=mx.bfloat16),
        "compressor_state",
    )
    cache.update_pool(
        mx.random.normal((1, 129, 64), dtype=mx.bfloat16),
        "indexer_state",
    )
    return cache


def _assert_pool_q_segments_equal(source, restored):
    for source_branch, restored_branch in (
        (source.compressor_state, restored.compressor_state),
        (source.indexer_state, restored.indexer_state),
    ):
        source_segments = source_branch._pooled_q_segments
        restored_segments = restored_branch._pooled_q_segments
        assert len(restored_segments) == len(source_segments)
        for source_segment, restored_segment in zip(
            source_segments, restored_segments
        ):
            for source_leaf, restored_leaf in zip(
                source_segment[:3], restored_segment[:3]
            ):
                assert mx.array_equal(source_leaf, restored_leaf).item()
            assert source_segment[3:] == restored_segment[3:]


def test_pool_quantized_v4_cache_is_detected_as_dsv4_composite():
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from vmlx_engine.prefix_cache import _is_dsv4_cache_class
    from vmlx_engine.scheduler import Scheduler

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)

    assert Scheduler._is_dsv4_cache_object(cache)
    assert _is_dsv4_cache_class("PoolQuantizedV4Cache")


def test_dsv4_prompt_snapshot_preserves_pool_quantized_cache_type():
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    cache = _make_pool_quantized_dsv4_state_cache()
    snapshot = DSV4BatchGenerator._snapshot_dsv4_cache([cache])

    assert snapshot is not None
    assert len(snapshot) == 1
    assert isinstance(snapshot[0], PoolQuantizedV4Cache)
    assert snapshot[0].compress_ratio == 4
    assert snapshot[0].state[1][2].shape == (1, 2, 512)


def test_dsv4_prompt_snapshot_preserves_encoded_pool_segments(monkeypatch):
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    cache = _make_encoded_pool_quantized_dsv4_cache(monkeypatch)
    snapshot = DSV4BatchGenerator._snapshot_dsv4_cache([cache])

    assert snapshot is not None
    _assert_pool_q_segments_equal(cache, snapshot[0])


def test_dsv4_prompt_snapshot_preserves_zero_ratio_swa_ring():
    from mlx_lm.models.cache import RotatingKVCache
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    cache = RotatingKVCache(max_size=8, keep=0)
    keys = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
    values = keys + 100
    cache.update_and_fetch(keys, values)
    # Cross the in-place wrap so the physical order and insertion pointer are
    # both meaningful state, not just a temporally ordered short prefix.
    cache.update_and_fetch(mx.array([[[[12.0]]]]), mx.array([[[[112.0]]]]))
    mx.eval(cache.keys, cache.values)

    snapshot = DSV4BatchGenerator._snapshot_dsv4_cache([cache])

    assert snapshot is not None
    assert len(snapshot) == 1
    restored = snapshot[0]
    assert isinstance(restored, RotatingKVCache)
    assert restored.meta_state == cache.meta_state
    assert restored.keys.tolist() == cache.keys.tolist()
    assert restored.values.tolist() == cache.values.tolist()


def test_dsv4_extraction_reports_pool_quantized_native_codec():
    from vmlx_engine.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    extracted = scheduler._extract_cache_states(
        [_make_pool_quantized_dsv4_state_cache()]
    )

    assert extracted[0]["class_name"] == "PoolQuantizedV4Cache"
    assert extracted[0]["pool_quant"] is True


def test_dsv4_extraction_uses_lossless_pool_storage_state(monkeypatch):
    from jang_tools.dsv4.pool_quant_cache import POOL_STORAGE_SCHEMA
    from vmlx_engine.scheduler import Scheduler

    cache = _make_encoded_pool_quantized_dsv4_cache(monkeypatch)
    scheduler = Scheduler.__new__(Scheduler)
    extracted = scheduler._extract_cache_states([cache])

    assert extracted[0]["pool_storage_schema"] == POOL_STORAGE_SCHEMA
    assert extracted[0]["state"][0] == POOL_STORAGE_SCHEMA
    # Reading extraction metadata must not materialize a retained BF16 pool.
    assert cache.compressor_state._pooled_bf16 is None
    assert cache.compressor_state._pooled_q_segments


def test_pool_quantized_v4_cache_does_not_route_to_hybrid_ssm():
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.scheduler import Scheduler

    class _Model:
        def make_cache(self):
            return [KVCache(), PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)]

    assert Scheduler._model_uses_dsv4_cache(_Model())
    assert not Scheduler._is_hybrid_model(_Model())


def test_dsv4_pool_quant_default_preserves_legacy_fallback_and_env_override(
    monkeypatch, request
):
    """The direct loader's pool codec default is independent of prefix reuse."""
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _configure_dsv4_pool_quant_default,
    )

    _restore_process_env_after_test(request, "DSV4_LONG_CTX", "DSV4_POOL_QUANT")
    monkeypatch.delenv("DSV4_LONG_CTX", raising=False)
    monkeypatch.delenv("DSV4_POOL_QUANT", raising=False)

    assert _configure_dsv4_pool_quant_default() == "1"
    assert os.environ["DSV4_LONG_CTX"] == "1"
    assert os.environ["DSV4_POOL_QUANT"] == "1"

    monkeypatch.setenv("DSV4_POOL_QUANT", "0")
    assert _configure_dsv4_pool_quant_default() == "0"


def test_dsv4_pool_quant_default_reads_bundle_cache_stamp(
    monkeypatch, request, tmp_path
):
    """Direct CLI/load paths use the same model-owned pool codec default as UI."""
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _configure_dsv4_pool_quant_default,
    )

    _restore_process_env_after_test(request, "DSV4_LONG_CTX", "DSV4_POOL_QUANT")
    monkeypatch.delenv("DSV4_LONG_CTX", raising=False)
    monkeypatch.delenv("DSV4_POOL_QUANT", raising=False)
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"cache": {"pool_quant_default": False}})
    )

    assert _configure_dsv4_pool_quant_default(str(tmp_path)) == "0"
    assert os.environ["DSV4_POOL_QUANT"] == "0"

    monkeypatch.delenv("DSV4_POOL_QUANT", raising=False)
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"cache": {"pool_quant_default": True}})
    )
    assert _configure_dsv4_pool_quant_default(str(tmp_path)) == "1"


def test_dsv4_cli_cache_policy_keeps_prefix_and_independent_ram_l2_tiers():
    """DSV4 uses native 256-token transport without forcing either tier off."""
    from vmlx_engine.cli import _apply_dsv4_cache_policy

    args = SimpleNamespace(
        enable_prefix_cache=True,
        disable_prefix_cache=False,
        use_paged_cache=True,
        paged_cache_block_size=64,
        max_cache_blocks=1000,
        max_cache_blocks_explicit=False,
        enable_block_disk_cache=True,
        dsv4_enable_prefix_cache=False,
    )

    changed = _apply_dsv4_cache_policy(args, logger=__import__("logging").getLogger("test"))

    assert args.enable_prefix_cache is True
    assert args.disable_prefix_cache is False
    assert args.use_paged_cache is True
    assert args.enable_block_disk_cache is True
    assert args.paged_cache_block_size == 256
    assert args.max_cache_blocks == 4097
    assert changed == ("block_size=64->256", "max_blocks=1000->4097")


def test_dsv4_cli_cache_policy_preserves_explicit_max_cache_blocks():
    """An explicit user block budget wins over the DSV4 family default."""
    from vmlx_engine.cli import _apply_dsv4_cache_policy

    args = SimpleNamespace(
        enable_prefix_cache=True,
        disable_prefix_cache=False,
        use_paged_cache=False,
        paged_cache_block_size=256,
        max_cache_blocks=2048,
        max_cache_blocks_explicit=True,
        enable_block_disk_cache=True,
    )

    changed = _apply_dsv4_cache_policy(
        args, logger=__import__("logging").getLogger("test")
    )

    assert args.max_cache_blocks == 2048
    assert not any(item.startswith("max_blocks=") for item in changed)


def test_dsv4_cli_cache_policy_does_not_force_paged_ram_on():
    """SSD-only DSV4 keeps paged RAM off while using native 256-token blocks."""
    from vmlx_engine.cli import _apply_dsv4_cache_policy

    args = SimpleNamespace(
        enable_prefix_cache=True,
        disable_prefix_cache=False,
        use_paged_cache=False,
        paged_cache_block_size=64,
        enable_block_disk_cache=True,
        dsv4_enable_prefix_cache=True,
    )

    changed = _apply_dsv4_cache_policy(args, logger=__import__("logging").getLogger("test"))

    assert args.enable_prefix_cache is True
    assert args.disable_prefix_cache is False
    assert args.use_paged_cache is False
    assert args.enable_block_disk_cache is True
    assert args.paged_cache_block_size == 256
    assert "block_size=64->256" in changed


def test_dsv4_cli_cache_policy_explicit_prefix_off_disables_both_storage_tiers():
    """Turning the owning prefix index off also disables RAM and SSD payloads."""
    from vmlx_engine.cli import _apply_dsv4_cache_policy

    args = SimpleNamespace(
        enable_prefix_cache=True,
        disable_prefix_cache=True,
        use_paged_cache=True,
        paged_cache_block_size=64,
        enable_block_disk_cache=True,
    )

    changed = _apply_dsv4_cache_policy(
        args, logger=__import__("logging").getLogger("test")
    )

    assert args.use_paged_cache is False
    assert args.enable_block_disk_cache is False
    assert args.paged_cache_block_size == 256
    assert "block_size=64->256" in changed
    assert "paged=disabled_without_prefix" in changed
    assert "L2 disk=disabled_without_prefix" in changed


def test_dsv4_runtime_policy_applies_to_bench_like_cli_args(
    tmp_path, monkeypatch, request
):
    """Bench/CLI paths share native DSV4 topology and preserve cache controls."""
    from vmlx_engine.cli import _apply_dsv4_runtime_policy

    _restore_process_env_after_test(
        request,
        "DSV4_LONG_CTX",
        "DSV4_POOL_QUANT",
        "VMLX_DISABLE_TQ_KV",
    )
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v4"}')
    args = SimpleNamespace(
        model=str(tmp_path),
        continuous_batching=False,
        enable_prefix_cache=True,
        disable_prefix_cache=False,
        use_paged_cache=True,
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
        dsv4_enable_prefix_cache=False,
    )
    monkeypatch.delenv("DSV4_LONG_CTX", raising=False)
    monkeypatch.delenv("DSV4_POOL_QUANT", raising=False)
    monkeypatch.delenv("VMLX_DSV4_ENABLE_PREFIX_CACHE", raising=False)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")

    applied, changes = _apply_dsv4_runtime_policy(
        args,
        logger=__import__("logging").getLogger("test"),
        clamp_max_num_seqs=True,
    )

    assert applied is True
    assert args.enable_prefix_cache is True
    assert args.disable_prefix_cache is False
    assert args.use_paged_cache is True
    assert args.enable_block_disk_cache is True
    assert args.paged_cache_block_size == 256
    assert args.kv_cache_quantization == "none"
    assert args.kv_cache_quantization_explicit is True
    assert args.max_num_seqs == 1
    assert args.continuous_batching is True
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
    assert os.environ["DSV4_POOL_QUANT"] == "1"
    assert os.environ["VMLX_DISABLE_TQ_KV"] == "1"
    assert "VMLX_FORCE_TQ_AUTO" not in os.environ
    assert "continuous_batching=off->on" in changes
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


def test_dsv4_runtime_policy_respects_explicit_prefix_cache_disable(monkeypatch):
    """Explicit prefix Off wins and disables both DSV4 storage tiers."""
    from vmlx_engine.cli import _apply_dsv4_runtime_policy

    args = SimpleNamespace(
        continuous_batching=True,
        enable_prefix_cache=True,
        disable_prefix_cache=True,
        use_paged_cache=True,
        paged_cache_block_size=64,
        enable_block_disk_cache=True,
        kv_cache_quantization="none",
        kv_cache_quantization_explicit=False,
        max_num_seqs=1,
        prefill_batch_size=1,
        completion_batch_size=1,
        no_memory_aware_cache=False,
        enable_disk_cache=False,
        enable_jit=False,
        smelt=False,
        flash_moe=False,
        distributed=False,
        speculative_model=None,
        dsv4_enable_prefix_cache=True,
    )
    monkeypatch.delenv("DSV4_LONG_CTX", raising=False)
    monkeypatch.delenv("DSV4_POOL_QUANT", raising=False)
    monkeypatch.delenv("VMLX_DSV4_ENABLE_PREFIX_CACHE", raising=False)

    applied, changes = _apply_dsv4_runtime_policy(
        args,
        logger=__import__("logging").getLogger("test"),
        clamp_max_num_seqs=True,
    )

    assert applied is True
    assert args.disable_prefix_cache is True
    assert args.use_paged_cache is False
    assert args.enable_block_disk_cache is False
    assert args.paged_cache_block_size == 256
    assert "paged=disabled_without_prefix" in changes
    assert "L2 disk=disabled_without_prefix" in changes
    assert "block_size=64->256" in changes


def test_panel_suppresses_generic_kv_quantization_controls_for_dsv4():
    """The app UI/launch preview must not advertise generic KV q4/q8 for DSV4."""
    from pathlib import Path

    form = Path("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").read_text()
    settings = Path("panel/src/renderer/src/components/sessions/SessionSettings.tsx").read_text()
    sessions = Path("panel/src/main/sessions.ts").read_text()

    assert "const effectiveStoredCacheQuantization = 'auto'" in form
    assert '<select value={effectiveStoredCacheQuantization} className="cfg-input" disabled>' in form, (
        "the stored-codec selector must stay visibly disabled at auto for "
        "every family; a re-enabled selector would reintroduce lossy stored "
        "prefixes"
    )
    assert 'className="cfg-input" disabled' in form
    # Generic KV quantization emission was retired entirely: NO family gets a
    # panel-pushed --kv-cache-quantization anymore (the stored/live cache is
    # exact and architecture-native everywhere), which strictly covers the
    # original DSV4/M3/openPangu suppression this test pinned. The flag only
    # survives in the additional-args blocklists so a hand-typed value cannot
    # sneak back in through that door either.
    assert "parts.push('--kv-cache-quantization'" not in settings
    assert "push('--kv-cache-quantization'" not in sessions
    assert "'--kv-cache-quantization',\n" in settings  # blocklisted for additional args
    # The engine->registry family alias used to be a byte-identical private copy
    # in all three of these files, spanning the main AND renderer processes. It
    # now lives once in src/shared/detectedFamilyNames.ts. Asserting the body in
    # each file again would pin exactly the duplication that was removed — and
    # a wrong spelling here (deepseek_v4 vs deepseek-v4) is what silently no-op'd
    # two earlier fixes, so check the rule at its owner and that each consumer
    # imports rather than redefines it.
    shared = (
        Path(__file__).resolve().parents[1]
        / "panel/src/shared/detectedFamilyNames.ts"
    ).read_text()
    # The if-chain became a table when four more engine spellings were added
    # (qwen3.5 / qwen3.5-moe / qwen3-next / nemotron-h were missing, which is
    # what left remote sessions on those families at the generic timeout).
    # Assert the MAPPING, not the syntax that expresses it.
    assert "deepseek_v4: 'deepseek-v4'" in shared
    for name, source in (("form", form), ("settings", settings), ("sessions", sessions)):
        assert "detectedFamilyNames" in source, (
            f"{name} stopped importing the shared family-alias rule"
        )
        assert "function normalizeDetectedFamilyName" not in source, (
            f"{name} grew its own copy of the family-alias rule back"
        )


def test_panel_names_dsv4_cache_as_native_composite_not_generic_paged_kv():
    """DSV4 UI must not make the internal paged-prefix path look like generic KV."""
    from pathlib import Path

    form = Path("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").read_text()
    en_catalog = Path("panel/src/renderer/src/i18n/locales/en.json").read_text()

    assert "<Section title={t('sessions.config.prefixCache')} sectionKey=\"prefixCache\"" in form
    assert "<CheckField label={t('sessions.config.pagedKVCache')}" not in form
    assert "const pagedCacheToggleLabel = dsv4Active" not in form
    # The "no hidden toggle" reassurance is user-visible copy, so the i18n pass
    # moved it into the locale catalog. The invariant is unchanged: the form
    # must still render it for DSV4 and the English entry must still make the
    # no-second-toggle promise.
    assert "{dsv4Active && <InfoNote text={t('sessions.config.dsv4PrefixReuseNote')} />}" in form
    assert "there is no separate hidden DSV4 cache toggle" in en_catalog
    # Stale DSV4-branded cache names must stay dead in the source AND in the
    # catalog — labels live in en.json now, so a resurrected key/value pair is
    # how this copy would reach users today.
    for stale in (
        "DSV4 Native Cache",
        "DSV4 Native Composite Prefix Cache",
        "restored SWA+CSA/HCA state has not proven output-equivalent",
    ):
        assert stale not in form
        assert stale not in en_catalog


def test_panel_suppresses_dsv4_batch_sizes_but_passes_real_prefill_step():
    """DSV4 batch sizes stay fixed at one; its memory step remains effective."""
    from pathlib import Path

    form = Path("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").read_text()
    settings = Path("panel/src/renderer/src/components/sessions/SessionSettings.tsx").read_text()
    sessions = Path("panel/src/main/sessions.ts").read_text()

    assert "const effectiveMaxNumSeqs = dsv4Active ? 1 : config.maxNumSeqs" in form
    assert "const effectivePrefillBatchSize = dsv4Active ? 1 : config.prefillBatchSize" in form
    assert "const effectiveCompletionBatchSize = dsv4Active ? 1 : config.completionBatchSize" in form
    assert "const prefillBatchSize = finitePositiveInteger(config.prefillBatchSize)" in sessions
    assert "if (!dsv4Active && prefillBatchSize != null)" in sessions
    assert "const prefillStepSize = finitePositiveInteger(config.prefillStepSize)" in sessions
    assert "if (prefillStepSize != null)" in sessions
    assert "if (!dsv4Active && prefillStepSize != null)" not in sessions
    assert "const completionBatchSize = finitePositiveInteger(config.completionBatchSize)" in sessions
    assert "if (!dsv4Active && completionBatchSize != null)" in sessions
    assert "const prefillBatchSize = finitePositiveInteger(config.prefillBatchSize)" in settings
    assert "if (!dsv4Active && prefillBatchSize != null)" in settings
    assert "const prefillStepSize = finitePositiveInteger(config.prefillStepSize)" in settings
    assert "if (prefillStepSize != null)" in settings
    assert "if (!dsv4Active && prefillStepSize != null)" not in settings
    assert "const completionBatchSize = finitePositiveInteger(config.completionBatchSize)" in settings
    assert "if (!dsv4Active && completionBatchSize != null)" in settings


def test_dsv4_ui_exposes_native_composite_reuse_without_a_second_toggle():
    """DSV4 uses the shared tier controls without inventing a second cache."""
    from pathlib import Path

    form = Path("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").read_text()
    en_catalog = Path("panel/src/renderer/src/i18n/locales/en.json").read_text()

    assert "const effectiveContinuousBatching = dsv4Active ? true : config.continuousBatching" in form
    assert "dsv4PrefixCache: false" in form
    assert "dsv4PoolQuant: false" not in form
    assert "dsv4CompositeCacheOptIn" not in form
    assert "const prefixOff = !effectivePrefixCacheEnabled" in form
    assert "const multimodalActive = !dsv4Active" in form
    assert "checked={effectiveContinuousBatching}" in form
    assert "checked={effectivePrefixCacheEnabled}" in form
    assert "cacheControlUpdatesForDsv4CompositeToggle" not in form
    assert "applyDsv4CompositeCacheToggle" not in form
    assert "cacheControlUpdatesForDsv4PoolQuantToggle" not in form
    assert "applyDsv4PoolQuantToggle" not in form
    assert "disabled={!cachePolicy.blockDiskCacheVisible || cachePolicy.blockDiskCacheDisabled || exactTypedPromptDiskCache}" in form
    assert "cacheControlUpdatesForBlockDiskToggle(v, cacheControlState)" in form
    # The no-second-toggle copy is rendered via the locale catalog since the
    # i18n pass; the form must reference the key and English must still carry
    # the promise this test is named after.
    assert "{dsv4Active && <InfoNote text={t('sessions.config.dsv4PrefixReuseNote')} />}" in form
    assert "there is no separate hidden DSV4 cache toggle" in en_catalog
    assert "checked={config.dsv4PrefixCache !== false}" not in form
    # Stale second-cache branding must not return in the source or as a
    # catalog entry — en.json is where a resurrected label would render from.
    for stale in (
        "restored SWA+CSA/HCA state has not proven output-equivalent",
        "DSV4 Composite Prefix Cache",
        "DSV4 Native Cache",
        "DSV4 Pool Quantization",
    ):
        assert stale not in form
        assert stale not in en_catalog
    assert "checked={dsv4Active ? true : config.enablePrefixCache}" not in form
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
    cache_helper = Path("panel/src/shared/cacheLaunchArgs.ts").read_text()
    cache_policy = Path("panel/src/shared/cacheControlPolicy.ts").read_text()

    for source in (settings, sessions):
        # DFlash2 speculative sessions run the SimpleEngine lane, which
        # deliberately keeps the paged cache stack off (its multiturn reuse
        # lives in the dflash2_runtime session store instead).
        assert (
            "const cacheStackActive = dsv4Active\n"
            "      ? true\n"
            "      : dflash2Speculative\n"
            "        ? false\n"
            "        : config.continuousBatching !== false"
        ) in source or (
            "const cacheStackActive = dsv4Active\n"
            "    ? true\n"
            "    : dflash2Speculative\n"
            "      ? false\n"
            "      : config.continuousBatching !== false"
        ) in source
        assert "buildCacheLaunchArgs" in source
        assert "const prefixCacheOff = dsv4Active ? false" not in source
        assert "const effectiveSmelt = !!(config as any).smelt && !dsv4Active" in source
        assert "const isVLM = dsv4Active || effectiveSmelt" in source
        assert "const effectiveDistributed = requestedDistributed && !dsv4Active" in source
        assert "const effectiveFlashMoe = requestedFlashMoe && !effectiveDistributed && !dsv4Active" in source
        assert "compatibleExternalSpeculative" in source
        assert "if (compatibleExternalSpeculative)" in source
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
        assert "dsv4PrefixCacheOptIn" not in source
    # The renderer preview pushes buildCacheLaunchArgs output directly and no
    # longer destructures prefixCacheOff; only the main-process launcher needs
    # the policy field for its DSV4 log line and gating.
    assert "const prefixCacheOff = cacheLaunchPolicy.prefixCacheOff" in sessions
    assert "const usePagedCache = cacheLaunchPolicy.effectiveUsePagedCache" in sessions
    assert "resolveCacheLaunchPolicy" in cache_helper
    assert "architectureRequiresPagedCache" in cache_policy
    assert "config.isMultimodal = false" in sessions
    assert "--dsv4-enable-prefix-cache" in sessions


def test_dsv4_cache_ui_uses_shared_cache_owner_without_duplicate_labels():
    """The DSV4 product surface uses shared controls for its typed cache."""
    from pathlib import Path

    form = Path("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").read_text()
    en_catalog = Path("panel/src/renderer/src/i18n/locales/en.json").read_text()

    assert 'label="DSV4 Native Composite Prefix Cache"' not in form
    assert 'label="DSV4 CSA/HCA Pool Codec"' not in form
    # Labels render from the locale catalog since the i18n pass. The invariant
    # ("the SHARED tier controls exist, spelled the shared way") now means: the
    # form references the shared keys, and en.json still spells them without a
    # DSV4 prefix.
    assert "label={t('sessions.cache.blockDiskCache')}" in form
    assert '"blockDiskCache": "Block Disk Cache (SSD / L2)"' in en_catalog
    assert "DSV4 Block Disk Cache (SSD / L2)" not in form
    assert "DSV4 Block Disk Cache (SSD / L2)" not in en_catalog

    # DSV4 uses the shared prefix/SSD controls; only the incompatible
    # generic stored-KV codec selector stays disabled. There is no second
    # DSV4-specific prefix toggle.
    assert "<CheckField label={t('sessions.config.enablePrefixCache')}" in form
    assert '"enablePrefixCache": "Enable Prefix Cache"' in en_catalog
    assert "!dsv4Active && (\n          <CheckField label={t('sessions.config.enablePrefixCache')}" not in form
    assert "<CheckField label={t('sessions.config.pagedKVCache')}" not in form
    assert "checked={cachePolicy.blockDiskCacheChecked}" in form
    assert "sessions.config.blockDiskPureSsdNote" in form
    assert 'className="cfg-input" disabled' in form
    assert "const effectiveStoredCacheQuantization = 'auto'" in form
    assert '<select value={effectiveStoredCacheQuantization} className="cfg-input" disabled>' in form, (
        "the stored-codec selector must stay visibly disabled at auto for "
        "every family; a re-enabled selector would reintroduce lossy stored "
        "prefixes"
    )

    # A stale label would reach users through a catalog entry now, so check
    # both surfaces.
    for stale_label in (
        "DSV4 Native Cache",
        "DSV4 Composite Prefix Cache",
        "DSV4 Pool Quantization",
        "DSV4 Flash composite prefix cache is disabled",
    ):
        assert stale_label not in form
        assert stale_label not in en_catalog


def test_dsv4_product_ui_uses_shared_cache_defaults_and_cli_env_stays_explicit():
    """Product sessions default shared tiers while pool codec remains explicit."""
    from pathlib import Path

    policy = Path("panel/src/shared/cacheControlPolicy.ts").read_text()
    dsv4_env = Path("panel/src/shared/dsv4Env.ts").read_text()
    sessions = Path("panel/src/main/sessions.ts").read_text()

    assert "Dsv4" not in policy
    assert "dsv4PrefixCache" not in policy
    assert "dsv4PoolQuant" not in policy

    assert "dsv4PrefixCache" not in dsv4_env
    assert "dsv4PoolQuant?:" not in dsv4_env
    assert "dsv4PoolQuantDefault?: boolean" in dsv4_env
    assert "typeof options.dsv4PoolQuantDefault === 'boolean'" in dsv4_env
    assert "options.dsv4PoolQuantDefault ? '1' : '0'" in dsv4_env
    assert "env.VMLX_DSV4_ENABLE_PREFIX_CACHE" not in dsv4_env

    assert "const dsv4PrefixOptIn = false" not in sessions
    assert "config.dsv4PoolQuant = false" not in sessions
    assert "config.enablePrefixCache = false" not in sessions
    assert "if (config.enablePrefixCache === undefined)" in sessions
    assert "config.enablePrefixCache = true" in sessions
    assert "if (config.usePagedCache === undefined)" in sessions
    assert "config.usePagedCache = false" in sessions
    assert "if (config.enableBlockDiskCache === undefined)" in sessions
    assert "config.enableBlockDiskCache = true" in sessions
    assert "config.dsv4PoolQuant = detected.dsv4PoolQuantDefault" in sessions
    assert "delete config.dsv4PoolQuant" in sessions
    assert "dsv4PoolQuantDefault: freshDetectedConfig?.dsv4PoolQuantDefault" in sessions


def test_session_preview_and_real_launch_share_dsv4_and_image_sanitizers():
    """Preview must sanitize the same stale CLI args as the real launcher."""
    import re
    from pathlib import Path

    sessions = Path("panel/src/main/sessions.ts").read_text()
    settings = Path("panel/src/renderer/src/components/sessions/SessionSettings.tsx").read_text()

    shared = Path("panel/src/shared/launchArgValues.ts").read_text()

    def extract_set(source: str, name: str) -> set[str]:
        match = re.search(
            rf"(?:export )?const {name} = new Set\(\[\n(?P<body>.*?)\n\]\)",
            source,
            re.S,
        )
        assert match, f"missing {name}"
        return set(re.findall(r"'(--[^']+)'", match.group("body")))

    # ADDITIONAL_ARG_VALUE_FLAGS moved out of both files: it was a 66-entry list
    # duplicated across the argv BUILDER and the CLI-preview renderer, and now
    # lives once in src/shared/launchArgValues.ts with both importing it.
    # Asserting the two copies match would pin the duplication that was removed,
    # so read the rule from its owner and check the consumers import it.
    assert extract_set(shared, "ADDITIONAL_ARG_VALUE_FLAGS")
    for consumer in (sessions, settings):
        assert "launchArgValues" in consumer
        assert "const ADDITIONAL_ARG_VALUE_FLAGS" not in consumer

    # These two are still per-file and MUST stay in agreement.
    for set_name in (
        "IMAGE_ADDITIONAL_ARG_BLOCKLIST",
        "DSV4_ADDITIONAL_ARG_BLOCKLIST",
    ):
        assert extract_set(settings, set_name) == extract_set(sessions, set_name)

    dsv4_blocklist = extract_set(sessions, "DSV4_ADDITIONAL_ARG_BLOCKLIST")
    for flag in (
        "--dsv4-enable-prefix-cache",
        "--disable-prefix-cache",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        "--block-disk-cache-max-gb",
        "--kv-cache-quantization",
        "--max-tokens",
        "--max-prompt-tokens",
        "--image-mode",
        "--image-quantize",
        "--mflux-class",
        "--native-mtp-depth",
        "--native-mtp-sampling-policy",
        "--omni-backend",
        "--enable-jit",
        "--default-temperature",
        "--default-repetition-penalty",
        "--tool-call-parser",
        "--reasoning-parser",
    ):
        assert flag in dsv4_blocklist


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


def test_dsv4_pool_q8_disk_tree_and_reconstruction_are_lossless(
    monkeypatch,
):
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.scheduler import Scheduler

    source = _make_encoded_pool_quantized_dsv4_cache(monkeypatch)
    scheduler = Scheduler.__new__(Scheduler)
    extracted = scheduler._extract_cache_states([source])
    paged = PagedCacheManager(block_size=4, max_blocks=8)
    prefix = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    table = prefix.store_cache(
        "dsv4-lossless-pool-l2",
        [11, 12, 13, 14, 15, 16, 17],
        extracted,
    )
    assert table is not None

    terminal = paged.allocated_blocks[table.block_ids[-1]]
    tensors, dtype, num_layers = _serialize_block(terminal.cache_data)
    terminal.cache_data = _deserialize_block(dict(tensors), dtype)
    rebuilt = prefix.reconstruct_cache(table)

    assert num_layers == 1
    assert rebuilt is not None
    assert isinstance(rebuilt[0], PoolQuantizedV4Cache)
    _assert_pool_q_segments_equal(source, rebuilt[0])


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

    # Per-block cache bookkeeping is intentionally DEBUG-only: INFO logging on
    # high-cardinality/eval workloads used to spend more time formatting one
    # line per miss/write than reporting actionable runtime state.  Keep the
    # native-composite tag contract pinned without re-promoting hot-path noise.
    caplog.set_level("DEBUG", logger="vmlx_engine.prefix_cache")
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


def test_dsv4_disk_only_pending_chain_without_terminal_is_a_miss():
    from types import SimpleNamespace

    from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    class _DiskOnlyPendingDsv4:
        def __init__(self, cache_data):
            self.cache_data = cache_data

        def read_block(self, _block_hash):
            return self.cache_data

    tokens = [11, 12, 13, 14]
    pending_cache = [(
        "deepseek_v4_pending",
        "DeepseekV4Cache",
        {"compress_ratio": 4, "sliding_window": 128},
    )]
    disk = _DiskOnlyPendingDsv4(pending_cache)

    assert BlockAwarePrefixCache._dsv4_l2_chain_missing_terminal_state(
        [SimpleNamespace(cache_data=None, block_hash=b"dsv4-pending")],
        disk,
    )

    paged = PagedCacheManager(
        block_size=4,
        max_blocks=8,
        disk_store=disk,
    )
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)

    block = paged.allocate_block()
    block.token_count = len(tokens)
    block.cache_data = None
    block.cache_data_from_disk = False
    block_hash = compute_block_hash(None, tokens)
    block.block_hash = block_hash
    paged.cached_block_hash_to_block.insert(block_hash, block)

    table, remaining = pc.fetch_cache("dsv4-disk-only-pending", tokens + [15])

    assert table is None
    assert remaining == tokens + [15]


def test_dsv4_metadata_only_validation_reads_l2_payload_once_before_reconstruct():
    from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    terminal_payload = [(
        "deepseek_v4",
        ("local", "compressor", "indexer"),
        ("0", "128", "4", "4"),
        "DeepseekV4Cache",
        {"compress_ratio": 4, "sliding_window": 128},
    )]

    tokens = [11, 12, 13, 14]
    block_hash = compute_block_hash(None, tokens)

    class _CountingDisk:
        def __init__(self):
            self.reads = {}

        def read_block(self, requested_hash):
            self.reads[requested_hash] = self.reads.get(requested_hash, 0) + 1
            return terminal_payload if requested_hash == block_hash else None

    disk = _CountingDisk()
    paged = PagedCacheManager(
        block_size=4,
        max_blocks=8,
        disk_store=disk,
    )
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)
    block = paged.allocate_block()
    block.token_count = len(tokens)
    block.cache_data = None
    block.cache_data_from_disk = False
    block.block_hash = block_hash
    paged.cached_block_hash_to_block.insert(block_hash, block)

    table, remaining = cache.fetch_cache(
        "dsv4-metadata-only-validation",
        tokens + [15],
    )

    assert table is not None
    assert table.num_tokens == len(tokens)
    assert remaining == [15]
    assert disk.reads[block_hash] == 1
    # Validation must not turn an immutable inspection payload into resident
    # live state. The worker still performs the authoritative reconstruction.
    assert block.cache_data is None


def test_dsv4_disk_backed_terminal_chain_with_composite_state_is_a_hit():
    from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(block_size=4, max_blocks=8)
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)

    tokens = [11, 12, 13, 14]
    block = paged.allocate_block()
    block.token_count = len(tokens)
    block.cache_data = [(
        "deepseek_v4",
        ("local", "compressor", "indexer"),
        ("0", "128", "7", "7"),
        "DeepseekV4Cache",
        {"compress_ratio": 4, "sliding_window": 128},
    )]
    block.cache_data_from_disk = True
    block_hash = compute_block_hash(None, tokens)
    block.block_hash = block_hash
    paged.cached_block_hash_to_block.insert(block_hash, block)

    table, remaining = pc.fetch_cache("dsv4-disk-terminal", tokens + [15])

    assert table is not None
    assert table.block_ids == [block.block_id]
    assert table.num_tokens == len(tokens)
    assert remaining == [15]


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


def test_dsv4_paged_reconstruct_preserves_pool_quantized_cache_type():
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=PagedCacheManager(4, 8))
    c = _make_pool_quantized_dsv4_state_cache()
    tokens = [11, 12, 13, 14, 15, 16, 17]
    table = pc.store_cache("dsv4-pool-quant-test", tokens, [_state_dict(c)])

    rebuilt = pc.reconstruct_cache(table)

    assert rebuilt is not None
    assert len(rebuilt) == 1
    assert isinstance(rebuilt[0], PoolQuantizedV4Cache)
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

        def has_block(self, _block_hash):
            return True

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
    # Once the L2 write is readable, reconstruction keeps the native composite
    # payload as a genuine RAM-tier entry but removes its temporary protection so
    # the configured byte-budget LRU can evict it later.
    for block_id in table.block_ids:
        block = paged.allocated_blocks[block_id]
        assert block.cache_data is not None
        assert block.cache_data_from_disk is False
        assert block.keep_resident is False


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


def test_dsv4_pool_quant_appends_only_new_pool_rows(monkeypatch):
    """Bundled JANG pool quant must not requantize old DSV4 pool rows."""
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    quant_shapes = []
    original_quant = pq._quant_pool

    def recording_quant(pool, *args, **kwargs):
        quant_shapes.append(tuple(pool.shape))
        return original_quant(pool, *args, **kwargs)

    monkeypatch.setattr(pq, "_quant_pool", recording_quant)
    # Lower the small-pool BF16 threshold so this focused test exercises the
    # quantized append path without allocating a multi-MB pool.
    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 64)

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    first = mx.ones((1, 3, 16), dtype=mx.bfloat16)
    second = mx.ones((1, 1, 16), dtype=mx.bfloat16) * 2

    pool_a = cache.update_pool_view(first, "compressor_state")
    pool_b = cache.update_pool_view(second, "compressor_state")

    assert tuple(pool_b.shape) == (1, 4, 16)
    assert quant_shapes == [(1, 3, 16), (1, 1, 16)]


def test_dsv4_pool_quant_uses_tiled_view_without_retained_full_bf16(monkeypatch):
    """Pool-on DSV4 retains q8 while attention reads bounded BF16 tiles."""
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    dequant_count = 0
    original_dequant = pq._dequant_pool

    def recording_dequant(qpool, *args, **kwargs):
        nonlocal dequant_count
        dequant_count += 1
        return original_dequant(qpool, *args, **kwargs)

    monkeypatch.setattr(pq, "_dequant_pool", recording_dequant)
    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    first = mx.ones((1, 3, 16), dtype=mx.bfloat16)
    second = mx.ones((1, 1, 16), dtype=mx.bfloat16) * 2

    pool_a = cache.update_pool_view(first, "compressor_state")
    pool_b = cache.update_pool_view(second, "compressor_state")
    retained_bytes = cache.nbytes
    tiles = list(pool_b.iter_dequantized_tiles(max_rows=2))
    mx.eval([tile for _start, tile in tiles])

    assert tuple(pool_a.shape) == (1, 4, 16)
    assert tuple(pool_b.shape) == (1, 4, 16)
    assert [start for start, _tile in tiles] == [0, 2]
    assert all(int(tile.shape[1]) <= 2 for _start, tile in tiles)
    assert dequant_count > 0
    assert cache.nbytes == retained_bytes
    assert cache.compressor_state._pooled_bf16 is None
    assert "_pooled_attention_view" not in vars(cache.compressor_state)
    assert cache.nbytes < 4 * 16 * 2


def test_dsv4_timing_probe_is_env_gated_and_covers_cache_boundaries():
    """DSV4 speed work needs boundary timings, not sampler/cache guesses."""
    import inspect

    from vmlx_engine.scheduler import Scheduler
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    gen_src = inspect.getsource(DSV4BatchGenerator)
    sched_src = inspect.getsource(Scheduler)

    assert "VMLINUX_DSV4_TRACE_TIMINGS" in gen_src
    assert "VMLINUX_DSV4_TRACE_TIMINGS" in sched_src
    assert "DSV4 timing" in gen_src
    assert "DSV4 timing" in sched_src
    for marker in (
        "prefill_head",
        "prompt_snapshot",
        "prefill_last",
        "cache_hit_tail_prefill",
        "decode_model",
        "sample_materialize",
    ):
        assert marker in gen_src
    for marker in ("reconstruct_cache", "extract_cache_states", "store_cache"):
        assert marker in sched_src


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


def test_dsv4_cli_cache_summary_names_native_composite_cache():
    """Startup summary must not advertise DSV4 as generic paged KV."""
    from types import SimpleNamespace

    from vmlx_engine.cli import _cache_stack_summary_lines

    lines = _cache_stack_summary_lines(
        SimpleNamespace(
            use_paged_cache=True,
            paged_cache_block_size=256,
            max_cache_blocks=4097,
            enable_block_disk_cache=True,
            block_disk_cache_max_gb=10,
        ),
        dsv4_model=True,
    )

    joined = "\n".join(lines)
    assert "DSV4 native composite in-memory prefix cache" in joined
    assert "deepseek_v4_v10_delta" in joined
    assert "max_blocks=4097" in joined
    assert "usable_blocks=4096" in joined
    assert "capacity=1048576 tokens" in joined
    assert "generic paged KV" in joined
    assert "Paged cache:" not in joined


def test_dsv4_cli_disk_only_summary_reports_usable_index_capacity():
    """Disk-only DSV4 summaries exclude the manager's reserved null block."""
    from types import SimpleNamespace

    from vmlx_engine.cli import _cache_stack_summary_lines

    lines = _cache_stack_summary_lines(
        SimpleNamespace(
            use_paged_cache=False,
            paged_cache_block_size=256,
            max_cache_blocks=4097,
            enable_block_disk_cache=True,
            block_disk_cache_max_gb=10,
        ),
        dsv4_model=True,
    )

    joined = "\n".join(lines)
    assert "max_index_blocks=4097" in joined
    assert "usable_blocks=4096" in joined
    assert "indexed_capacity=1048576 tokens" in joined


def test_dsv4_scheduler_log_names_native_composite_block_index():
    """Scheduler log should not call DSV4's typed block transport generic paged KV."""
    import inspect

    from vmlx_engine.scheduler import Scheduler

    source = inspect.getsource(Scheduler.__init__)
    assert "DSV4 native composite block index enabled" in source
    assert "not generic paged KV" in source
    assert "deepseek_v4_v10_delta" in source
    assert 'f"Paged cache enabled: block_size=' in source


def test_dsv4_health_reports_rotating_swa_layers_as_ratio_zero(monkeypatch):
    """Instantiated DSV4 RotatingKVCache layers are ratio-zero, not unknown."""
    from types import SimpleNamespace

    from vmlx_engine.server import _native_cache_status

    class RotatingKVCache:
        pass

    rotating = RotatingKVCache()
    composite = SimpleNamespace(
        compress_ratio=4,
        local=SimpleNamespace(max_size=128),
    )
    scheduler = SimpleNamespace(
        _uses_dsv4_cache=True,
        _model_type_for_runtime="deepseek_v4",
        model=SimpleNamespace(make_cache=lambda: [rotating, composite]),
        block_aware_cache=None,
        paged_cache_manager=None,
        disk_cache=None,
    )
    monkeypatch.setenv("DSV4_POOL_QUANT", "0")

    status = _native_cache_status(scheduler)

    assert status["compress_ratios"] == [0, 4]
    assert status["compress_ratio_counts"] == {"0": 1, "4": 1}
    assert status["layer_cache_roles"]["ratio_0"] == "swa_local_only"


def test_dsv4_activation_qat_partitions_prefix_and_l2_model_identity(monkeypatch):
    """QAT-on cache tensors must never refault into a QAT-off graph."""
    from types import SimpleNamespace

    from vmlx_engine.prefix_cache import compute_model_cache_key

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="deepseek_v4",
            num_hidden_layers=43,
            num_attention_heads=64,
            num_key_value_heads=1,
            hidden_size=4096,
        )
    )
    monkeypatch.setenv("DSV4_ACTIVATION_QAT", "0")
    qat_off_key = compute_model_cache_key(model)
    monkeypatch.setenv("DSV4_ACTIVATION_QAT", "1")
    qat_on_key = compute_model_cache_key(model)

    assert qat_off_key != qat_on_key
    monkeypatch.setenv("DSV4_ACTIVATION_QAT", "false")
    assert compute_model_cache_key(model) == qat_off_key


def test_dsv4_health_reports_qat_truth_and_native_memory_separately(monkeypatch):
    """Health distinguishes JANG QAT attestation, native state, and retained L1."""
    from types import SimpleNamespace

    from vmlx_engine.server import _native_cache_status

    class RotatingKVCache:
        pass

    class PoolQuantizedV4Cache:
        def __init__(self, ratio):
            self.compress_ratio = ratio
            self.local = SimpleNamespace(max_size=128)

    config = SimpleNamespace(
        model_type="deepseek_v4",
        num_hidden_layers=3,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        hidden_size=4096,
        sliding_window=128,
        index_head_dim=128,
        torch_dtype="bfloat16",
        max_position_embeddings=1_048_576,
        compress_ratios=[0, 4, 128],
    )
    model = SimpleNamespace(
        # The engine may wrap the language model in an unrelated outer config;
        # telemetry must select the recognized DSV4 config, not the wrapper.
        config=SimpleNamespace(model_type="wrapper"),
        language_model=SimpleNamespace(config=config),
        make_cache=lambda: [
            RotatingKVCache(),
            PoolQuantizedV4Cache(4),
            PoolQuantizedV4Cache(128),
        ],
        _vmlx_dsv4_activation_qat_status={
            "requested": True,
            "effective": True,
            "observed": True,
            "attested": True,
            "e4m3_kv_pool_observed": True,
            "hadamard_fp4_indexer_observed": True,
            "implementation_available": True,
            "fused_e4m3_available": True,
            "fused_indexer_available": True,
            "fp32_compressor_staging_unconditional": True,
            "attestation_scope": "transform_family_dispatch_not_every_call_site",
            "transform_families": ["e4m3", "hadamard_fp4"],
        },
    )
    batch_generator = SimpleNamespace(
        _requests=[],
        prompt_snapshot_last_estimated_bytes=123_456,
    )
    scheduler = SimpleNamespace(
        _uses_dsv4_cache=True,
        _model_type_for_runtime="deepseek_v4",
        model=model,
        block_aware_cache=None,
        paged_cache_manager=None,
        disk_cache=None,
        running={
            "request": SimpleNamespace(
                num_prompt_tokens=4096,
                num_output_tokens=64,
                num_tokens=4160,
                prompt_token_ids=[],
            )
        },
        _last_cache_execution={"prompt_tokens": 2048},
        batch_generator=batch_generator,
    )
    monkeypatch.setenv("DSV4_POOL_QUANT", "1")
    monkeypatch.setenv("DSV4_ACTIVATION_QAT", "1")

    status = _native_cache_status(scheduler)

    assert status["activation_qat"]["requested"] is True
    assert status["activation_qat"]["effective"] is True
    assert status["activation_qat"]["observed"] is True
    assert status["activation_qat"]["attested"] is True
    assert status["activation_qat"]["matches_request"] is True
    assert status["activation_qat"]["fp32_compressor_staging"] == {
        "controlled_by_toggle": False,
        "observed": True,
        "reason": "separate_precision_contract",
    }
    memory = status["native_state_memory"]
    assert memory["basis"] == "architecture_estimate_not_allocator_measurement"
    assert memory["retained_l1_reported_separately"] is True
    assert memory["pool_quant_observed"] is True
    assert memory["current_prompt"]["tokens"] == 4096
    assert memory["current_sequence"]["tokens"] == 4160
    assert memory["current_sequence_basis"] == "prompt_plus_generated_tokens"
    assert memory["last_prompt"]["tokens"] == 2048
    assert memory["max_context"]["tokens"] == 1_048_576
    assert memory["max_context"]["total_bytes"] < 15 * 1024**3
    assert memory["last_prompt_snapshot"]["estimated_bytes"] == 123_456
    assert "prefix_cache_l1" in memory["excludes"]


def test_dsv4_health_does_not_echo_qat_request_as_runtime_observation(monkeypatch):
    """An old JANG runtime without attestation remains explicitly unknown."""
    from types import SimpleNamespace

    from vmlx_engine.server import _native_cache_status

    model = SimpleNamespace(make_cache=lambda: [])
    scheduler = SimpleNamespace(
        _uses_dsv4_cache=True,
        _model_type_for_runtime="deepseek_v4",
        model=model,
        block_aware_cache=None,
        paged_cache_manager=None,
        disk_cache=None,
        running={},
        batch_generator=None,
    )
    monkeypatch.setenv("DSV4_ACTIVATION_QAT", "1")

    qat = _native_cache_status(scheduler)["activation_qat"]

    assert qat["requested"] is True
    assert qat["effective"] is None
    assert qat["observed"] is None
    assert qat["attested"] is None
    assert qat["matches_request"] is False


def test_dsv4_health_accepts_attested_identity_bypass_as_qat_off(monkeypatch):
    """Observed False is the correct enabled state when both paths bypass QAT."""
    from types import SimpleNamespace

    from vmlx_engine.server import _dsv4_activation_qat_status

    model = SimpleNamespace(
        _vmlx_dsv4_activation_qat_status={
            "requested": False,
            "effective": False,
            "observed": False,
            "attested": True,
            "e4m3_kv_pool_observed": False,
            "hadamard_fp4_indexer_observed": False,
            "implementation_available": True,
            "fused_e4m3_available": True,
            "fused_indexer_available": True,
            "fp32_compressor_staging_unconditional": True,
            "attestation_scope": "transform_family_dispatch_not_every_call_site",
            "transform_families": ["e4m3", "hadamard_fp4"],
        }
    )
    monkeypatch.setenv("DSV4_ACTIVATION_QAT", "0")

    status = _dsv4_activation_qat_status(model)

    assert status["requested"] is False
    assert status["runtime_requested"] is False
    assert status["effective"] is False
    assert status["observed"] is False
    assert status["attested"] is True
    assert status["matches_request"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("implementation_available", False),
        ("attested", False),
        ("e4m3_kv_pool_observed", False),
        ("hadamard_fp4_indexer_observed", None),
    ],
)
def test_dsv4_health_rejects_contradictory_qat_attestation(
    monkeypatch, field, value
):
    """A request match requires every transform-family attestation to agree."""
    from types import SimpleNamespace

    from vmlx_engine.server import _dsv4_activation_qat_status

    runtime_status = {
        "requested": True,
        "effective": True,
        "observed": True,
        "attested": True,
        "e4m3_kv_pool_observed": True,
        "hadamard_fp4_indexer_observed": True,
        "implementation_available": True,
    }
    runtime_status[field] = value
    monkeypatch.setenv("DSV4_ACTIVATION_QAT", "1")

    status = _dsv4_activation_qat_status(
        SimpleNamespace(_vmlx_dsv4_activation_qat_status=runtime_status)
    )

    assert status["matches_request"] is False


def test_dsv4_native_memory_prefers_dimensional_inner_config(monkeypatch):
    """A DSV4-labelled wrapper must not hide its estimator-ready text config."""
    from types import SimpleNamespace

    from vmlx_engine.server import _dsv4_native_state_memory_status

    inner = SimpleNamespace(
        model_type="deepseek_v4",
        num_hidden_layers=3,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        hidden_size=4096,
        max_position_embeddings=4096,
        compress_ratios=[0, 4, 128],
    )
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v4", text_config=inner),
        make_cache=lambda: [],
    )
    scheduler = SimpleNamespace(
        _uses_dsv4_cache=True,
        _model_type_for_runtime="deepseek_v4",
        model=model,
        block_aware_cache=None,
        paged_cache_manager=None,
        disk_cache=None,
        running={
            "request": SimpleNamespace(
                num_prompt_tokens=1024,
                num_output_tokens=0,
                num_tokens=1024,
                prompt_token_ids=[],
            )
        },
        batch_generator=SimpleNamespace(
            _requests=[], prompt_snapshot_last_estimated_bytes=0
        ),
        _last_cache_execution=None,
    )
    memory = _dsv4_native_state_memory_status(
        scheduler, model, pool_quant_observed=True
    )

    assert memory["available"] is True
    assert memory["current_prompt"]["tokens"] == 1024


def test_dsv4_native_memory_fails_closed_when_estimator_cannot_size(monkeypatch):
    """Positive token counts cannot report available with null estimates."""
    from types import SimpleNamespace

    from vmlx_engine.server import _dsv4_native_state_memory_status

    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))
    scheduler = SimpleNamespace(
        model=model,
        running={
            "request": SimpleNamespace(
                num_prompt_tokens=1024,
                num_output_tokens=0,
                num_tokens=1024,
                prompt_token_ids=[],
            )
        },
        batch_generator=SimpleNamespace(
            _requests=[], prompt_snapshot_last_estimated_bytes=0
        ),
        _last_cache_execution=None,
    )

    memory = _dsv4_native_state_memory_status(
        scheduler, model, pool_quant_observed=True
    )

    assert memory["available"] is False
    assert memory["reason"] == "native_estimator_unavailable"
    assert "current_prompt" in memory["unavailable_estimates"]


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


def test_dsv4_repetition_penalty_uses_generated_only_prompt_context():
    """DSV4 exact-copy/code prompts must not penalize every prompt token.

    Normal mlx_lm generate_step semantics apply repetition penalty to the final
    prompt token plus generated tokens. DSV4 still passes the full original
    prompt into its custom generator for cache-hit parity, so the scheduler must
    pass a wrapped per-request processor and must not install unwrapped global
    processors on the DSV4 generator.
    """
    import inspect
    from vmlx_engine.scheduler import Scheduler

    create_src = inspect.getsource(Scheduler._create_batch_generator)
    dsv4_block = create_src[
        create_src.index("return DSV4BatchGenerator("):
        create_src.index("except Exception as _dsv4_err:")
    ]
    assert "logits_processors=None" in dsv4_block
    assert "logits_processors=logits_processors" not in dsv4_block

    schedule_src = inspect.getsource(Scheduler._schedule_waiting)
    dsv4_insert_block = schedule_src[
        schedule_src.index('== "DSV4BatchGenerator"'):
        schedule_src.index("else:", schedule_src.index('== "DSV4BatchGenerator"'))
    ]
    assert 'insert_kwargs["all_tokens"] = [request.prompt_token_ids]' in dsv4_insert_block
    assert 'insert_kwargs["prompt_snapshot_tail_tokens"]' in dsv4_insert_block
    assert schedule_src.count('insert_kwargs["prompt_snapshot_tail_tokens"]') == 2
    assert "request_processors = self._request_logits_processors(" in dsv4_insert_block
    assert "request, list(request.prompt_token_ids)" in dsv4_insert_block
    assert 'insert_kwargs["logits_processors"]' in dsv4_insert_block

    seen = {}

    def processor(tokens, logits):
        seen["tokens"] = list(tokens)
        return logits

    wrapped = Scheduler._wrap_generated_only_logits_processor(
        processor,
        skip_prefix_tokens=3,
    )
    wrapped([10, 11, 12, 13, 14, 15], object())
    assert seen["tokens"] == [13, 14, 15]


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
    # The worker finalizer serves both Paged On and block-disk-only hybrid
    # backends, so the production diagnostic intentionally uses the broader
    # block-cache label.
    assert "hybrid block-cache HIT" in finalize_src


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


def test_dsv4_v9_paged_restore_is_not_unconditionally_rejected():
    """The v9 native-state namespace may reach normal paged reconstruction.

    v9 invalidates the old lossy pool records, preserves ratio-zero rotating
    SWA, and transports q8 pool segments without materialize/requantize.  A
    blanket DSV4 rejection here would silently discard every otherwise-valid
    exact or partial L2 hit and force full prefill.
    """
    from vmlx_engine.scheduler import Scheduler

    import inspect

    add_request = inspect.getsource(Scheduler.add_request)
    assert not hasattr(Scheduler, "_dsv4_paged_hit_requires_full_prefill")
    assert "rejecting unsafe DSV4 paged/L2 extension" not in add_request
    assert "if block_table and block_table.num_tokens > 0:" in add_request


def test_failed_generation_prefix_trim_never_stores_original_state_under_shorter_key():
    """Every cache family must fail closed when key/state alignment fails."""
    import inspect

    from vmlx_engine.scheduler import Scheduler

    cleanup = inspect.getsource(Scheduler._cleanup_finished)
    failed_trim_branch = cleanup.index("if trunc_ok and truncated_dicts:")
    fail_closed = cleanup.index("cache_data = None", failed_trim_branch)
    family_specific_log = cleanup.index(
        "if not self._uses_dsv4_cache:", fail_closed
    )
    store_guard = cleanup.index("if cache_data is not None:", family_specific_log)

    # cache_data is discarded before selecting the family-specific diagnostic,
    # so this is a generic correctness boundary rather than a DSV4-only guard.
    assert failed_trim_branch < fail_closed < family_specific_log < store_guard
    assert "key/state equivalence is unknown for every" in cleanup[
        failed_trim_branch:family_specific_log
    ]
    assert "self.block_aware_cache.store_cache(" not in cleanup[
        fail_closed:store_guard
    ]


def _run_cleanup_publication_probe(
    monkeypatch,
    *,
    request_id,
    prompt_token_ids,
    gen_prompt_len,
    extracted_cache,
    uses_dsv4_cache,
    is_hybrid=False,
    reject_publication=True,
):
    """Run the real cleanup path and retain every attempted publication."""
    import vmlx_engine.scheduler as scheduler_mod
    from vmlx_engine.request import Request, RequestStatus, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    monkeypatch.setattr(scheduler_mod, "clear_mlx_memory_cache", lambda log=None: None)

    request = Request(
        request_id=request_id,
        prompt=list(prompt_token_ids),
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.status = RequestStatus.RUNNING
    request._gen_prompt_len = gen_prompt_len
    request._extracted_cache = extracted_cache

    published = []

    class _Paged:
        max_resident_bytes = 0

        @staticmethod
        def release_request_refs(_table):
            return None

        @staticmethod
        def detach_request(_request_id):
            return None

    class _BlockAware:
        _request_tables = {}
        paged_cache = _Paged()

        @staticmethod
        def store_cache(*args, **kwargs):
            published.append((args, kwargs))
            if reject_publication:
                raise AssertionError("unalignable state reached store_cache")
            return []

    scheduler = object.__new__(Scheduler)
    scheduler.running = {request.request_id: request}
    scheduler.requests = {request.request_id: request}
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.finished_req_ids = set()
    scheduler.batch_generator = None
    scheduler.stop_tokens = set()
    scheduler.block_aware_cache = _BlockAware()
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None
    scheduler.disk_cache = None
    scheduler.config = type("_Config", (), {"enable_prefix_cache": True})()
    scheduler._ssm_state_cache = None
    scheduler._kv_cache_bits = 0
    scheduler._is_hybrid = is_hybrid
    scheduler._uses_dsv4_cache = uses_dsv4_cache
    scheduler._uses_zaya_cache = False
    scheduler._mixed_attention_cache_model = False
    scheduler._pld_pending = {}
    scheduler._pld_ngram_indices = {}
    scheduler._pick_cache_type_for_request = lambda _request: "user"
    scheduler._retarget_ssm_rederive_to_paged_boundary = lambda *_args, **_kwargs: None
    scheduler._dsv4_trace_timing = lambda *_args, **_kwargs: None
    scheduler._cleanup_detokenizer = lambda _request_id: None
    scheduler._materialize_deferred_prompt_cache = lambda _request_id, _request: None
    scheduler.model = object()

    Scheduler._cleanup_finished(scheduler, {request.request_id})

    return request, published


def test_generation_prefix_cleanup_rejects_nonpositive_cache_boundary(monkeypatch):
    """A recognized state still cannot be published under an empty key."""
    state = (
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
    )
    request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="nonpositive-generation-prefix-boundary",
        prompt_token_ids=[10, 90],
        gen_prompt_len=1,
        extracted_cache=[
            {"state": state, "meta_state": (2,), "class_name": "KVCache"}
        ],
        uses_dsv4_cache=False,
    )

    assert published == []
    assert request._extracted_cache is None


def test_generation_prefix_cleanup_aligns_nested_cache_list(monkeypatch):
    """Nested CacheList payloads of plain KV subs are recursively aligned.

    This used to fail closed, which cost every CacheList-per-layer family
    (deepseek_v32, longcat_flash, longcat_flash_ngram) all prefix reuse: the
    store refused each layer, so no entry was ever published.
    """
    request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="nonempty-cache-list",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": None,
                "meta_state": None,
                "class_name": "CacheList",
                "sub_caches": [
                    {
                        "state": (
                            mx.zeros((1, 1, 5, 8), dtype=mx.float16),
                            mx.zeros((1, 1, 5, 8), dtype=mx.float16),
                        ),
                        "meta_state": (5,),
                        "class_name": "KVCache",
                    }
                ],
            }
        ],
        uses_dsv4_cache=False,
        reject_publication=False,
    )

    assert published, "aligned CacheList must reach store_cache"
    (_req_id, key_tokens, cache_data), _kwargs = published[0]
    # prompt_tokens[:-gen_prompt_len] is 4 long, N-1 target is 3. The stored
    # key and the sliced sub-state must agree -- storing 5 tokens of KV under a
    # 3-token key is the corruption this alignment exists to prevent.
    assert len(key_tokens) == 3
    sub = cache_data[0]["sub_caches"][0]
    assert int(sub["state"][0].shape[2]) == 3
    assert int(sub["state"][1].shape[2]) == 3
    assert sub["meta_state"] == ("3",)


def test_generation_prefix_cleanup_rejects_cache_list_with_cumulative_sub(monkeypatch):
    """A cumulative sub inside a CacheList still fails closed.

    falcon_h1/baichuan_m1 build CacheList(ArraysCache, KVCache). Recurrent
    state has no token boundary to slice to, so publishing the layer would
    restore an SSM state that ran past the key it is stored under.
    """
    request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="cumulative-cache-list",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": None,
                "meta_state": None,
                "class_name": "CacheList",
                "sub_caches": [
                    {
                        "state": (mx.zeros((1, 4, 8), dtype=mx.float16),),
                        "meta_state": (),
                        "class_name": "ArraysCache",
                    },
                    {
                        "state": (
                            mx.zeros((1, 1, 5, 8), dtype=mx.float16),
                            mx.zeros((1, 1, 5, 8), dtype=mx.float16),
                        ),
                        "meta_state": (5,),
                        "class_name": "KVCache",
                    },
                ],
            }
        ],
        uses_dsv4_cache=False,
    )

    assert published == []
    assert request._extracted_cache is None


def test_generation_prefix_cleanup_rejects_unknown_nonempty_state(monkeypatch):
    """Unknown nonempty layer formats may not ride a shortened cache key."""
    request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="unknown-nonempty-state",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": {"opaque": mx.zeros((5, 8), dtype=mx.float16)},
                "meta_state": None,
                "class_name": "FutureOpaqueCache",
            }
        ],
        uses_dsv4_cache=False,
    )

    assert published == []
    assert request._extracted_cache is None


def test_generation_prefix_cleanup_rejects_unknown_rank_five_tuple(monkeypatch):
    """A tuple is positional only for the rank-3/4 formats the store owns."""
    state = (
        mx.zeros((1, 1, 1, 5, 8), dtype=mx.float16),
        mx.zeros((1, 1, 1, 5, 8), dtype=mx.float16),
    )
    request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="unknown-rank-five-state",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": state,
                "meta_state": None,
                "class_name": "FutureRankFiveCache",
            }
        ],
        uses_dsv4_cache=False,
    )

    assert published == []
    assert request._extracted_cache is None


def test_generation_prefix_cleanup_publishes_aligned_plain_kv(monkeypatch):
    """Fail-closed handling must preserve the recognized positional KV lane."""
    state = (
        mx.arange(40, dtype=mx.float16).reshape(1, 1, 5, 8),
        mx.arange(40, dtype=mx.float16).reshape(1, 1, 5, 8),
    )
    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="aligned-plain-kv",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {"state": state, "meta_state": (5,), "class_name": "KVCache"}
        ],
        uses_dsv4_cache=False,
        reject_publication=False,
    )

    assert len(published) == 1
    args, _kwargs = published[0]
    assert args[1] == [10, 11, 12]
    stored_keys, stored_values = args[2][0]["state"]
    assert stored_keys.shape[-2] == 3
    assert stored_values.shape[-2] == 3


def test_generation_prefix_cleanup_rejects_plain_kv_shorter_than_key(monkeypatch):
    """A positional KV state shorter than the cache key must fail closed."""
    state = (
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
    )
    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="plain-kv-shorter-than-key",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {"state": state, "meta_state": (2,), "class_name": "KVCache"}
        ],
        uses_dsv4_cache=False,
    )

    assert published == []


def test_generation_prefix_cleanup_handles_safe_and_wrapped_rotating_kv(
    monkeypatch,
):
    """Direct rotating metadata trims only before the circular buffer wraps."""
    state = (
        mx.zeros((1, 1, 5, 8), dtype=mx.float16),
        mx.zeros((1, 1, 5, 8), dtype=mx.float16),
    )
    _request, safe_publication = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="safe-rotating-kv",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": state,
                "meta_state": ("0", "8", "5", "5"),
                "class_name": "RotatingKVCache",
            }
        ],
        uses_dsv4_cache=False,
        reject_publication=False,
    )

    assert len(safe_publication) == 1
    safe_layer = safe_publication[0][0][2][0]
    assert safe_layer["state"][0].shape[-2] == 3
    assert safe_layer["meta_state"] == ("0", "8", "3", "3")

    _request, wrapped_publication = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="wrapped-rotating-kv",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": state,
                "meta_state": ("0", "4", "5", "1"),
                "class_name": "RotatingKVCache",
            }
        ],
        uses_dsv4_cache=False,
    )

    assert wrapped_publication == []


def test_generation_prefix_cleanup_rejects_missing_or_malformed_rotating_meta(
    monkeypatch,
):
    """Rotating temporal order cannot be inferred from absent metadata."""
    state = (
        mx.zeros((1, 1, 5, 8), dtype=mx.float16),
        mx.zeros((1, 1, 5, 8), dtype=mx.float16),
    )
    for request_id, meta_state in (
        ("missing-rotating-meta", ()),
        ("malformed-rotating-meta", ("keep", "max", "offset", "idx")),
        ("malformed-rotating-idx", ("0", "8", "5", "bogus")),
        ("divergent-rotating-idx", ("0", "8", "5", "4")),
    ):
        _request, published = _run_cleanup_publication_probe(
            monkeypatch,
            request_id=request_id,
            prompt_token_ids=[10, 11, 12, 13, 90, 91],
            gen_prompt_len=2,
            extracted_cache=[
                {
                    "state": state,
                    "meta_state": meta_state,
                    "class_name": "RotatingKVCache",
                }
            ],
            uses_dsv4_cache=False,
        )
        assert published == []


def _quantized_kv_state(length: int):
    return tuple(
        mx.zeros((1, 1, length, width), dtype=mx.float16)
        for width in (8, 2, 2)
    )


def test_generation_prefix_cleanup_publishes_exact_quantized_kv(monkeypatch):
    """Quantized K/V components align and slice to the exact key length."""
    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="aligned-quantized-kv",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": (_quantized_kv_state(5), _quantized_kv_state(5)),
                "meta_state": ("5", "64", "4"),
                "class_name": "QuantizedKVCache",
            }
        ],
        uses_dsv4_cache=False,
        reject_publication=False,
    )

    assert len(published) == 1
    layer = published[0][0][2][0]
    assert all(part.shape[-2] == 3 for part in layer["state"][0])
    assert all(part.shape[-2] == 3 for part in layer["state"][1])
    assert layer["meta_state"] == ("3", "64", "4")


def test_generation_prefix_cleanup_rejects_misaligned_quantized_kv(monkeypatch):
    """A single shorter value component invalidates the quantized layer."""
    bad_values = list(_quantized_kv_state(5))
    bad_values[1] = mx.zeros((1, 1, 2, 2), dtype=mx.float16)
    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="misaligned-quantized-kv",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": (_quantized_kv_state(5), tuple(bad_values)),
                "meta_state": ("5", "64", "4"),
                "class_name": "QuantizedKVCache",
            }
        ],
        uses_dsv4_cache=False,
    )

    assert published == []


def test_generation_prefix_cleanup_rejects_misaligned_minimax_m3_state(
    monkeypatch,
):
    """MiniMax-M3 keys, values, and lightning-indexer keys must agree."""
    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="misaligned-minimax-m3-cache",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": (
                    mx.zeros((1, 1, 5, 8), dtype=mx.float16),
                    mx.zeros((1, 1, 4, 8), dtype=mx.float16),
                    mx.zeros((1, 1, 5, 4), dtype=mx.float16),
                ),
                "meta_state": ("5",),
                "class_name": "MiniMaxM3SparseCache",
            }
        ],
        uses_dsv4_cache=False,
    )

    assert published == []


def test_generation_prefix_cleanup_rejects_dsv4_state_shorter_than_key(
    monkeypatch,
):
    """DSV4 native offset must reach the exact reusable key boundary."""
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache

    cache = DeepseekV4Cache(sliding_window=128, compress_ratio=4)
    keys = mx.zeros((1, 1, 2, 8), dtype=mx.float16)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    assert cache.offset == 2

    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="dsv4-state-shorter-than-key",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[_state_dict(cache)],
        uses_dsv4_cache=True,
    )

    assert published == []


def test_generation_prefix_cleanup_rejects_dsv4_local_length_meta_mismatch(
    monkeypatch,
):
    """DSV4 metadata cannot claim tokens absent from its local K/V state."""
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache

    cache = DeepseekV4Cache(sliding_window=128, compress_ratio=4)
    keys = mx.zeros((1, 1, 2, 8), dtype=mx.float16)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    state_dict = _state_dict(cache)
    state_dict["meta_state"] = ("0", "128", "3", "3")

    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="dsv4-local-length-meta-mismatch",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[state_dict],
        uses_dsv4_cache=True,
    )

    assert published == []


def test_generation_prefix_cleanup_publishes_true_no_state_placeholder(monkeypatch):
    """A layer with no state and no nested payload remains usable."""
    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="true-no-state-placeholder",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {
                "state": None,
                "meta_state": None,
                "class_name": "EmptyCachePlaceholder",
            }
        ],
        uses_dsv4_cache=False,
        reject_publication=False,
    )

    assert len(published) == 1
    assert published[0][0][1] == [10, 11, 12]


def test_generation_prefix_cleanup_keeps_hybrid_kv_and_skips_cumulative_state(
    monkeypatch,
):
    """Hybrid KV blocks publish while cumulative SSM state stays companion-owned."""
    kv_state = (
        mx.arange(40, dtype=mx.float16).reshape(1, 1, 5, 8),
        mx.arange(40, dtype=mx.float16).reshape(1, 1, 5, 8),
    )
    _request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="hybrid-kv-with-external-ssm-companion",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[
            {"state": kv_state, "meta_state": (5,), "class_name": "KVCache"},
            {
                "state": [mx.zeros((1, 8, 5), dtype=mx.float16)],
                "meta_state": None,
                "class_name": "MambaCache",
            },
        ],
        uses_dsv4_cache=False,
        is_hybrid=True,
        reject_publication=False,
    )

    assert len(published) == 1
    args, kwargs = published[0]
    assert args[1] == [10, 11, 12]
    assert args[2][0]["state"][0].shape[-2] == 3
    assert args[2][1]["class_name"] == "MambaCache"
    assert args[2][1]["state"] is None
    assert kwargs["store_cumulative_state"] is False


def test_non_dsv4_cleanup_drops_unalignable_state_before_paged_or_l2_publication(
    monkeypatch,
):
    """Unknown non-DSV4 state must not be published under a shorter key."""
    request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="generic-unalignable-store",
        prompt_token_ids=[10, 11, 12, 13, 90, 91],
        gen_prompt_len=2,
        extracted_cache=[object()],
        uses_dsv4_cache=False,
    )

    assert published == []
    assert request._extracted_cache is None


def test_dsv4_cleanup_drops_real_wrapped_state_before_shortened_key_publication(
    monkeypatch,
):
    """A wrapped DeepseekV4Cache state dict must fail closed in cleanup."""
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache

    cache = DeepseekV4Cache(sliding_window=128, compress_ratio=4)
    # Populate through the real RotatingKVCache update path so the serialized
    # metadata describes a genuinely wrapped state, rather than a fabricated
    # offset on an otherwise short cache.
    for start in range(0, 160, 32):
        keys = mx.full((1, 1, 32, 8), start, dtype=mx.float16)
        cache.update_and_fetch(keys, keys + 1)
    mx.eval(cache.state)

    wrapped_state = _state_dict(cache)
    assert cache.offset == 160
    assert cache.offset > cache.local.max_size
    assert int(wrapped_state["meta_state"][2]) == cache.offset

    # After the two generation-prefix tokens are removed, the reusable key
    # would represent 97 tokens (98 prompt tokens minus the N-1 re-feed token),
    # while the wrapped state represents 160. DeepseekV4Cache cannot safely
    # rewind that SWA+CSA/HCA state to this boundary.
    request, published = _run_cleanup_publication_probe(
        monkeypatch,
        request_id="dsv4-wrapped-state-store",
        prompt_token_ids=list(range(100)),
        gen_prompt_len=2,
        extracted_cache=[wrapped_state],
        uses_dsv4_cache=True,
    )

    assert published == []
    assert request._extracted_cache is None


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
    assert "_deferred_prompt_cache" in src
    assert '"family": "DSV4"' in src
    assert "_prefill_for_prompt_only_cache" not in src
    cleanup_src = inspect.getsource(
        scheduler.Scheduler._materialize_deferred_prompt_cache
    )
    assert "_prefill_for_prompt_only_cache" in cleanup_src
    helper_src = inspect.getsource(scheduler.Scheduler._prefill_for_prompt_only_cache)
    assert "dsv4_effective_prefill_step" in helper_src
    assert "chunk_size = len(prompt_tokens) if self._uses_dsv4_cache" not in helper_src


def test_terminal_cache_capture_requires_enabled_instantiated_store():
    """A generator cache object cannot trigger a throwaway second prefill.

    The live DSV4 Electron cache-Off row exposed ``prompt_cache`` on every
    terminal response.  Without this ownership gate, the mixed-SWA branch
    scheduled a full clean re-prefill even though no prefix/L2 backend existed.
    """
    from vmlx_engine.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    scheduler.config = SimpleNamespace(enable_prefix_cache=False)
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None
    scheduler.disk_cache = None
    request = SimpleNamespace(_bypass_prefix_cache=False)
    response = SimpleNamespace(prompt_cache=object())

    assert not Scheduler._terminal_cache_capture_enabled(
        scheduler,
        request,
        response,
    )

    scheduler.config.enable_prefix_cache = True
    assert not Scheduler._terminal_cache_capture_enabled(
        scheduler,
        request,
        response,
    )

    scheduler.memory_aware_cache = object()
    assert Scheduler._terminal_cache_capture_enabled(
        scheduler,
        request,
        response,
    )

    request._bypass_prefix_cache = True
    assert not Scheduler._terminal_cache_capture_enabled(
        scheduler,
        request,
        response,
    )


def test_dsv4_short_prompt_snapshot_skip_does_not_sync_reprefill_for_store():
    """Short-prompt snapshot skips must also skip sync re-prefill store.

    If the generator omits the prompt-boundary snapshot because the prompt is
    below the DSV4 snapshot threshold, the scheduler must not immediately
    replace that saved time with a clean prompt-only re-prefill during cleanup.
    The short request should simply skip donating a DSV4 prefix block.
    """
    import inspect
    from vmlx_engine import scheduler

    src = inspect.getsource(scheduler.Scheduler._process_batch_responses)

    assert "DSV4 prefix cache store skipped" in src
    assert "prompt below snapshot/store threshold" in src
    assert "dsv4_prompt_snapshot_min_tokens" in src


def test_dsv4_short_prompt_store_skip_does_not_warn_cannot_produce_cache():
    """The by-design short-prompt skip must not double-log as a WARNING.

    The threshold branch already emits a single INFO and sets
    _dsv4_short_prompt_store_skipped; the terminal else-chain must honor that
    flag instead of emitting the false-alarm "Cannot produce prompt-only
    cache" WARNING for every short DSV4 request.
    """
    import inspect
    from vmlx_engine import scheduler

    src = inspect.getsource(scheduler.Scheduler._process_batch_responses)

    assert src.count("_dsv4_short_prompt_store_skipped") >= 2
    warn_idx = src.index("Cannot produce prompt-only cache")
    guard_idx = src.rindex("_dsv4_short_prompt_store_skipped")
    assert guard_idx < warn_idx


def test_dsv4_generator_skips_prompt_snapshot_when_cache_store_disabled(monkeypatch):
    """No-cache DSV4 requests must not deep-copy composite cache snapshots.

    The prompt-boundary snapshot is only useful when paged/L2 prefix storage can
    consume it. Capturing it on plain no-cache chat adds a large synchronous
    cache copy before decoding and is visible as a DSV4 speed regression.
    """
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    class _Model:
        def make_cache(self):
            return [object()]

        def __call__(self, ids, cache=None):
            model_calls.append(ids.tolist()[0])
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    calls = []
    model_calls = []

    def _snapshot(_cache):
        calls.append("snapshot")
        return ["snapshot"]

    monkeypatch.setattr(DSV4BatchGenerator, "_snapshot_dsv4_cache", staticmethod(_snapshot))
    gen = DSV4BatchGenerator(_Model(), capture_prompt_snapshot=False)
    gen._warmed_up = True
    gen.insert([[42, 43, 44]], max_tokens=[2])

    prompt_responses, generation_responses = gen.next()

    assert prompt_responses
    assert not generation_responses
    assert calls == []
    assert model_calls == [[42, 43, 44]]
    assert prompt_responses[0].prompt_cache_snapshot is None


def test_dsv4_generator_captures_prompt_snapshot_when_cache_store_enabled(monkeypatch):
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.setenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", "0")

    class DeepseekV4Cache:
        """Mutable production-shaped composite cache test double."""

        def export_block_delta(self, start, end, **_kwargs):
            return {"start_token": start, "end_token": end}

    class _Model:
        def make_cache(self):
            # Production DSV4 cache layers are mutable Python objects and the
            # native block-delta collector attaches transient capture state.
            return [DeepseekV4Cache()]

        def __call__(self, ids, cache=None):
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    calls = []

    def _snapshot(_cache):
        calls.append("snapshot")
        return ["snapshot"]

    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_snapshot_admissible_dsv4_cache",
        staticmethod(_snapshot),
    )
    gen = DSV4BatchGenerator(_Model(), capture_prompt_snapshot=True)
    gen._warmed_up = True
    gen.insert([[42, 43]], max_tokens=[2])

    prompt_responses, _ = gen.next()

    assert calls == ["snapshot"]
    assert prompt_responses[0].prompt_cache_snapshot == ["snapshot"]


def test_dsv4_generator_snapshot_excludes_generation_rail_tokens(monkeypatch):
    """Native deltas end at the rail-stripped N-1 cache-key boundary."""
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.setenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", "0")
    model_calls = []

    class DeepseekV4Cache:
        def export_block_delta(self, start, end, **_kwargs):
            return {"start_token": start, "end_token": end}

    class _Model:
        def make_cache(self):
            return [DeepseekV4Cache()]

        def __call__(self, ids, cache=None):
            model_calls.append(ids.tolist()[0])
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    # This fake record has no native append-safe anchor payload. That contract
    # has separate real-topology coverage; this row owns only the terminal
    # generation-rail boundary.
    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_capture_dsv4_append_safe_checkpoint",
        classmethod(lambda cls, cache, target, **_kw: None),
    )
    gen = DSV4BatchGenerator(_Model(), capture_prompt_snapshot=True)
    gen._warmed_up = True
    prompt = list(range(368))
    gen.insert(
        [prompt],
        max_tokens=[2],
        prompt_snapshot_tail_tokens=[3],
    )

    prompt_responses, _ = gen.next()

    assert sum(len(call) for call in model_calls[:-1]) == 365
    assert len(model_calls[-1]) == 3
    snapshot = prompt_responses[0].prompt_cache_snapshot
    assert snapshot[0]["dsv4_record_intervals"] == ((0, 256), (256, 365))


def test_dsv4_generator_rejects_oversize_nested_snapshot_before_copy(monkeypatch):
    """Finite block budgets reject delta capture before record allocation."""
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.setenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", "0")

    class _CompositeCache:
        state = (
            (mx.zeros((8,), dtype=mx.float32), None),
            (mx.zeros((16,), dtype=mx.float32),),
            (mx.zeros((32,), dtype=mx.float32),),
        )

    class _Model:
        args = {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "sliding_window": 128,
            "index_head_dim": 32,
            "torch_dtype": "bfloat16",
            "compress_ratios": [0, 4],
        }

        def make_cache(self):
            return [_CompositeCache()]

        def __call__(self, ids, cache=None):
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    calls = []

    def _snapshot(_cache):
        calls.append("snapshot")
        return ["snapshot"]

    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_snapshot_dsv4_cache",
        staticmethod(_snapshot),
    )
    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_reset_dsv4_block_records",
        classmethod(
            lambda cls, cache, start: (_ for _ in ()).throw(
                AssertionError("oversize capture must be rejected before records")
            )
        ),
    )
    gen = DSV4BatchGenerator(
        _Model(),
        capture_prompt_snapshot=True,
        prompt_snapshot_max_bytes=1,
    )
    gen._warmed_up = True
    gen.insert([[42, 43]], max_tokens=[2])

    prompt_responses, _ = gen.next()

    assert calls == []
    assert prompt_responses[0].prompt_cache_snapshot is None
    assert gen.prompt_snapshot_last_estimated_bytes > 1
    assert gen.prompt_snapshot_oversize_skips == 1


def test_dsv4_snapshot_estimator_counts_encoded_pool_without_dequantizing(
    monkeypatch,
):
    import jang_tools.dsv4.pool_quant_cache as pool_quant_cache

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    cache = _make_encoded_pool_quantized_dsv4_cache(monkeypatch)

    def _unexpected_dequant(_qpool):
        raise AssertionError("snapshot admission must not dequantize DSV4 pools")

    with monkeypatch.context() as context:
        context.setattr(pool_quant_cache, "_dequant_pool", _unexpected_dequant)
        estimated = DSV4BatchGenerator._estimate_dsv4_cache_nbytes([cache])

    assert estimated == cache.nbytes
    assert cache.compressor_state._pooled_bf16 is None
    assert cache.compressor_state._pooled_q_segments


def test_dsv4_generator_rejects_snapshot_that_exceeds_metal_headroom(monkeypatch):
    import mlx.core as mx

    from vmlx_engine.utils import memory_limits
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.setenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", "0")
    monkeypatch.setattr(
        memory_limits,
        "get_effective_metal_working_set_bytes",
        lambda _mx: (990, 1000),
    )
    monkeypatch.setattr(memory_limits, "get_metal_ws_guard_threshold", lambda: 100.0)

    class _CompositeCache:
        state = ((mx.zeros((8,), dtype=mx.float32),), None, None)

    class _Model:
        def make_cache(self):
            return [_CompositeCache()]

        def __call__(self, ids, cache=None):
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    gen = DSV4BatchGenerator(
        _Model(), capture_prompt_snapshot=True, prompt_snapshot_max_bytes=None
    )
    gen._warmed_up = True
    gen.insert([[42, 43]], max_tokens=[2])

    prompt_responses, _ = gen.next()

    assert prompt_responses[0].prompt_cache_snapshot is None
    assert gen.prompt_snapshot_last_estimated_bytes == 32
    assert gen.prompt_snapshot_last_headroom_bytes == 10
    assert gen.prompt_snapshot_headroom_skips == 1
    assert gen.prompt_snapshot_oversize_skips == 1


def test_scheduler_passes_snapshot_budget_to_dsv4_and_surfaces_telemetry():
    import inspect

    from vmlx_engine import scheduler

    create_src = inspect.getsource(scheduler.Scheduler._create_batch_generator)
    dsv4_block = create_src[
        create_src.index("return DSV4BatchGenerator("):
        create_src.index("except Exception as _dsv4_err:")
    ]
    assert "prompt_snapshot_max_bytes=_prompt_snapshot_max_bytes" in dsv4_block

    stats_src = inspect.getsource(scheduler.Scheduler.get_stats)
    cache_stats_src = inspect.getsource(scheduler.Scheduler.get_cache_stats)
    for source in (stats_src, cache_stats_src):
        assert '"DSV4BatchGenerator"' in source
        assert '"prompt_snapshot_last_estimated_bytes"' in source
        assert '"prompt_snapshot_headroom_skips"' in source


def test_scheduler_snapshot_budget_includes_block_aware_ram_and_l2():
    from vmlx_engine.scheduler import _prompt_snapshot_backend_limit_bytes

    legacy_memory = SimpleNamespace(
        get_stats=lambda: {"max_bytes": 1000}
    )
    legacy_disk = SimpleNamespace(max_size_bytes=2000)
    paged = SimpleNamespace(
        disk_only=False,
        max_resident_bytes=3000,
        _disk_store=SimpleNamespace(max_size_bytes=4000),
    )
    block_cache = SimpleNamespace(paged_cache=paged)

    assert _prompt_snapshot_backend_limit_bytes(
        memory_aware_cache=legacy_memory,
        disk_cache=legacy_disk,
        block_aware_cache=block_cache,
    ) == 4000
    assert _prompt_snapshot_backend_limit_bytes(
        block_aware_cache=SimpleNamespace(
            paged_cache=SimpleNamespace(
                disk_only=True,
                max_resident_bytes=0,
                _disk_store=SimpleNamespace(max_size_bytes=5000),
            )
        )
    ) == 5000

    # Preserve the legacy meaning: an explicitly unbounded disk backend wins.
    assert _prompt_snapshot_backend_limit_bytes(
        memory_aware_cache=legacy_memory,
        disk_cache=SimpleNamespace(max_size_bytes=0),
    ) is None


def test_dsv4_cache_hit_extends_clean_snapshot_from_uncached_tail(monkeypatch):
    """A DSV4 hit must donate the expanded N-1 terminal state.

    Electron tool loops append function-call and function-output records after
    a cached user prefix.  The generator must snapshot after feeding only that
    new tail (except the final prompt token), so a later process can restore
    the post-tool terminal state without a full prompt re-prefill.
    """
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.setenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", "0")

    model_calls = []

    class _Model:
        def __call__(self, ids, cache=None):
            model_calls.append(ids.tolist()[0])
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    snapshots = []

    def _snapshot(cache):
        snapshots.append(cache)
        return ["extended-snapshot"]

    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_snapshot_dsv4_cache",
        staticmethod(_snapshot),
    )
    # Delta transport has dedicated real-topology coverage.  This test isolates
    # the cache-hit extension/snapshot boundary, so do not ask its minimal fake
    # cache layer to manufacture a native rotating anchor.
    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_capture_dsv4_terminal_anchor",
        classmethod(lambda cls, cache, target_token, **_kw: None),
    )
    restored_cache = [SimpleNamespace()]
    gen = DSV4BatchGenerator(_Model(), capture_prompt_snapshot=True)
    gen._warmed_up = True
    gen.insert(
        [[70, 71, 72]],
        max_tokens=[2],
        caches=[restored_cache],
        all_tokens=[[10, 11, 12, 70, 71, 72]],
    )

    prompt_responses, generation_responses = gen.next()

    assert prompt_responses
    assert not generation_responses
    assert model_calls == [[70, 71], [72]]
    assert snapshots == [restored_cache]
    assert prompt_responses[0].prompt_cache_snapshot == ["extended-snapshot"]


def test_dsv4_cache_hit_extension_excludes_generation_rail_tokens(monkeypatch):
    """A partial hit snapshots before the ordinary N-1 token and rail."""
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.setenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", "0")
    model_calls = []
    snapshots = []

    class _Model:
        def __call__(self, ids, cache=None):
            model_calls.append(ids.tolist()[0])
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_snapshot_dsv4_cache",
        staticmethod(lambda cache: snapshots.append(cache) or ["snapshot"]),
    )
    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_capture_dsv4_terminal_anchor",
        classmethod(lambda cls, cache, target_token, **_kw: None),
    )
    restored_cache = [SimpleNamespace()]
    gen = DSV4BatchGenerator(_Model(), capture_prompt_snapshot=True)
    gen._warmed_up = True
    gen.insert(
        [[70, 71, 72, 73, 74]],
        max_tokens=[2],
        caches=[restored_cache],
        all_tokens=[[10, 11, 12, 70, 71, 72, 73, 74]],
        prompt_snapshot_tail_tokens=[3],
    )

    prompt_responses, _ = gen.next()

    assert model_calls == [[70, 71], [72, 73, 74]]
    assert snapshots == [restored_cache]
    assert prompt_responses[0].prompt_cache_snapshot == ["snapshot"]


def test_dsv4_exact_n_minus_one_hit_skips_zero_delta_full_snapshot(monkeypatch):
    """An exact terminal hit must not duplicate the full composite cache."""
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.setenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", "0")

    model_calls = []

    class _Model:
        def __call__(self, ids, cache=None):
            model_calls.append(ids.tolist()[0])
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    def _unexpected_snapshot(_cache):
        raise AssertionError("exact N-1 hit must not copy a zero-delta snapshot")

    monkeypatch.setattr(
        DSV4BatchGenerator,
        "_snapshot_admissible_dsv4_cache",
        _unexpected_snapshot,
    )
    restored_cache = [object()]
    gen = DSV4BatchGenerator(_Model(), capture_prompt_snapshot=True)
    gen._warmed_up = True
    gen.insert(
        [[72]],
        max_tokens=[2],
        caches=[restored_cache],
        all_tokens=[[10, 11, 12, 70, 71, 72]],
    )

    prompt_responses, generation_responses = gen.next()

    assert prompt_responses
    assert not generation_responses
    assert model_calls == [[72]]
    assert prompt_responses[0].prompt_cache_snapshot is None


def test_dsv4_generator_skips_prompt_snapshot_for_short_cache_store_prompt_by_default(monkeypatch):
    """Short DSV4 prompts must not pay the composite snapshot store cost.

    Live timing showed a 21-token prompt spending ~14s in the prompt-boundary
    snapshot before decode. Prefix/L2 store still has value for long prompts,
    but tiny prompts should decode immediately and let the safe scheduler
    fallback skip or re-derive cache state instead of blocking the user path.

    With the extended store enabled (the default) short prompts DO arm the
    cheap delta-transport capture so the decode-time chain can grow past a
    block boundary; this test pins the kill-switch contract.
    """
    import mlx.core as mx

    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    monkeypatch.delenv("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", raising=False)
    monkeypatch.delenv("VMLINUX_DSV4_PROMPT_SNAPSHOT_MIN_TOKENS", raising=False)
    monkeypatch.setenv("VMLX_DSV4_EXTENDED_STORE", "0")

    class _Model:
        def make_cache(self):
            return [object()]

        def __call__(self, ids, cache=None):
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    calls = []

    def _snapshot(_cache):
        calls.append("snapshot")
        return ["snapshot"]

    monkeypatch.setattr(DSV4BatchGenerator, "_snapshot_dsv4_cache", staticmethod(_snapshot))
    gen = DSV4BatchGenerator(_Model(), capture_prompt_snapshot=True)
    gen._warmed_up = True
    gen.insert([[42, 43, 44]], max_tokens=[2])

    prompt_responses, _ = gen.next()

    assert calls == []
    assert prompt_responses[0].prompt_cache_snapshot is None


def test_dsv4_short_prompt_arms_extended_capture_when_extended_store_enabled():
    """Short prompts must still seed the extended decode-time chain.

    A 74-token prompt with a 2.5k-token generation was observed live storing
    nothing: the snapshot min-tokens gate skipped the two-phase capture, so
    _arm_extended_capture bailed on prompt_snapshot=None and the next turn
    re-prefilled the whole conversation. Both the cold and cache-hit snapshot
    gates must relax the min-tokens threshold when the extended store is
    enabled, and the scheduler must keep skipping tiny prompt-only snapshot
    stores when the chain never crossed a block boundary.
    """
    import inspect

    from vmlx_engine import scheduler
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    gen_src = inspect.getsource(DSV4BatchGenerator.next)
    assert gen_src.count("or self._extended_store_enabled") == 2

    sched_src = inspect.getsource(scheduler.Scheduler)
    assert "_dsv4_snapshot_store_below_threshold" in sched_src


def test_dsv4_long_prefill_guard_describes_bounded_chunk_default():
    """The DSV4 long-prefill guard must describe the current bounded default.

    Single-shot remains available for diagnostics, but it is no longer the
    production default after installed-app cache-hit validation.
    """
    import inspect

    from vmlx_engine import scheduler

    guard_src = inspect.getsource(scheduler.Scheduler.add_request)

    assert "defaults to bounded" in guard_src
    assert "DSV4_PREFILL_STEP_SIZE=0" in guard_src
    assert "dsv4_max_prefill_tokens()" in guard_src
    assert 'DSV4_MAX_PREFILL_TOKENS", "32768"' not in guard_src


def test_dsv4_long_prefill_ceiling_is_explicit_opt_in(monkeypatch):
    from vmlx_engine.utils.dsv4_batch_generator import dsv4_max_prefill_tokens

    monkeypatch.delenv("DSV4_MAX_PREFILL_TOKENS", raising=False)
    assert dsv4_max_prefill_tokens() == 0

    monkeypatch.setenv("DSV4_MAX_PREFILL_TOKENS", "32768")
    assert dsv4_max_prefill_tokens() == 32_768

    monkeypatch.setenv("DSV4_MAX_PREFILL_TOKENS", "invalid")
    assert dsv4_max_prefill_tokens() == 0


def test_dsv4_batch_generator_prefill_step_default_and_legacy_override(monkeypatch):
    from vmlx_engine.utils.dsv4_batch_generator import (
        DSV4BatchGenerator,
        dsv4_effective_prefill_step,
    )

    monkeypatch.delenv("DSV4_PREFILL_STEP_SIZE", raising=False)
    gen = DSV4BatchGenerator(object(), prefill_step_size=2048)
    assert gen.prefill_step_size == 2048

    monkeypatch.setenv("DSV4_PREFILL_STEP_SIZE", "1024")
    gen = DSV4BatchGenerator(object(), prefill_step_size=2048)
    assert gen.prefill_step_size == 1024

    monkeypatch.setenv("DSV4_PREFILL_STEP_SIZE", "0")
    gen = DSV4BatchGenerator(object(), prefill_step_size=2048)
    assert gen.prefill_step_size == 1 << 30

    assert dsv4_effective_prefill_step(2048, 12_288) == 2048
    # Attention sub-chunking (default on) bounds per-layer attention width,
    # so the long-context 512 clamp only applies when it is disabled.
    monkeypatch.delenv("DSV4_ATTN_SUBCHUNK", raising=False)
    assert dsv4_effective_prefill_step(2048, 12_289) == 2048
    assert dsv4_effective_prefill_step(2048, 32_768) == 2048
    monkeypatch.setenv("DSV4_ATTN_SUBCHUNK", "0")
    assert dsv4_effective_prefill_step(2048, 12_289) == 512
    assert dsv4_effective_prefill_step(2048, 32_768) == 512
    monkeypatch.delenv("DSV4_ATTN_SUBCHUNK", raising=False)
    assert dsv4_effective_prefill_step(256, 32_768) == 256
    assert (
        dsv4_effective_prefill_step(
            1 << 30,
            32_768,
            single_shot=True,
        )
        == 32_768
    )


def test_dsv4_prompt_only_prefill_uses_adaptive_long_context_chunks(monkeypatch):
    from types import SimpleNamespace

    from vmlx_engine.scheduler import Scheduler

    monkeypatch.delenv("DSV4_PREFILL_STEP_SIZE", raising=False)
    calls = []

    class _Model:
        def make_cache(self):
            return [_make_dsv4_state_cache()]

        def __call__(self, input_ids, cache=None):
            calls.append(int(input_ids.shape[-1]))
            return SimpleNamespace(logits=input_ids)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = _Model()
    scheduler.config = SimpleNamespace(prefill_step_size=2048)
    scheduler._uses_dsv4_cache = True

    # Legacy clamp path: with attention sub-chunking disabled, long-context
    # prefill falls back to bounded 512-token chunks.
    monkeypatch.setenv("DSV4_ATTN_SUBCHUNK", "0")
    cache = scheduler._prefill_for_prompt_only_cache(list(range(12_289)))

    assert cache is not None
    assert calls
    assert max(calls) == 512
    assert sum(calls) == 12_289

    # Default path: sub-chunking active keeps the configured wide step.
    monkeypatch.delenv("DSV4_ATTN_SUBCHUNK", raising=False)
    calls.clear()
    cache = scheduler._prefill_for_prompt_only_cache(list(range(12_289)))

    assert cache is not None
    assert calls
    assert max(calls) == 2048
    assert sum(calls) == 12_289


def test_dsv4_prompt_only_prefill_materializes_encoded_pool_without_dequantizing(
    monkeypatch,
):
    from types import SimpleNamespace

    import jang_tools.dsv4.pool_quant_cache as pool_quant_cache

    from vmlx_engine.scheduler import Scheduler

    encoded = _make_encoded_pool_quantized_dsv4_cache(monkeypatch)

    class _Model:
        def make_cache(self):
            return [encoded]

        def __call__(self, input_ids, cache=None):
            return SimpleNamespace(logits=input_ids)

    def _unexpected_dequant(_qpool):
        raise AssertionError("prompt-only prefill must not dequantize DSV4 pools")

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = _Model()
    scheduler.config = SimpleNamespace(prefill_step_size=2048)
    scheduler._uses_dsv4_cache = True
    with monkeypatch.context() as context:
        context.setattr(pool_quant_cache, "_dequant_pool", _unexpected_dequant)
        cache = scheduler._prefill_for_prompt_only_cache([1, 2, 3])

    assert cache == [encoded]


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


# ── DSV4 shadow re-key (idle-time predicted-transcript store) ────────────


def _dsv4_shadow_native_prefill_fixture():
    from vmlx_engine.scheduler import Scheduler
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    model_calls = []

    class DeepseekV4Cache:
        def __init__(self):
            self.offset = 0
            self.compress_ratio = 4

        def export_block_delta(
            self,
            start,
            end,
            *,
            block_size,
            anchor_interval_blocks,
            force_anchor=False,
        ):
            assert self.offset == end
            periodic = end % (block_size * anchor_interval_blocks) == 0
            anchored = bool(force_anchor or periodic)
            return {
                "start_token": start,
                "end_token": end,
                "block_size": block_size,
                "anchor_interval_blocks": anchor_interval_blocks,
                "anchor": (
                    {
                        "tokens": end,
                        "periodic": periodic,
                        "terminal": bool(force_anchor),
                    }
                    if anchored
                    else None
                ),
            }

    class _Model:
        def make_cache(self):
            return [DeepseekV4Cache()]

        def __call__(self, input_ids, cache=None):
            width = int(input_ids.shape[-1])
            model_calls.append(width)
            cache[0].offset += width
            return mx.zeros((1, width, 2), dtype=mx.float32)

    model = _Model()
    generator = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    generator.model = model
    generator.prefill_step_size = 2048
    generator._prefill_single_shot = False
    generator._sync = lambda: None
    generator._prefill_should_clear_cache = lambda: False
    generator._delta_capture_admitted = lambda cache, tokens: True

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    scheduler.batch_generator = generator
    scheduler._uses_dsv4_cache = True
    return scheduler, model_calls


def test_dsv4_shadow_prefill_captures_zero_origin_native_chain():
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    scheduler, model_calls = _dsv4_shadow_native_prefill_fixture()

    cache = scheduler._prefill_for_prompt_only_cache(
        list(range(326)),
        capture_dsv4_deltas=True,
    )

    assert cache is not None
    # The owning generator splits at the final preceding full block so it can
    # stamp an append-safe checkpoint while the live cache is exactly at 256.
    assert model_calls == [256, 70]
    transport = DSV4BatchGenerator._dsv4_delta_transport(cache)
    assert transport is not None
    assert transport[0]["dsv4_record_intervals"] == ((0, 256), (256, 326))
    records = transport[0]["dsv4_block_records"]
    assert records[0]["anchor"]["append_safe"] is True
    assert records[1]["anchor"]["terminal"] is True


def test_dsv4_shadow_native_prefill_yields_before_first_chunk():
    scheduler, model_calls = _dsv4_shadow_native_prefill_fixture()

    cache = scheduler._prefill_for_prompt_only_cache(
        list(range(3000)),
        should_stop=lambda: True,
        capture_dsv4_deltas=True,
    )

    assert cache is None
    assert model_calls == []


def test_dsv4_shadow_native_prefill_yields_between_chunks():
    scheduler, model_calls = _dsv4_shadow_native_prefill_fixture()
    polls = iter((False, True))

    cache = scheduler._prefill_for_prompt_only_cache(
        list(range(3000)),
        should_stop=lambda: next(polls),
        capture_dsv4_deltas=True,
    )

    assert cache is None
    assert model_calls == [2048]


def _think_ids():
    from vmlx_engine.utils.dsv4_batch_generator import (
        DSV4_EOS_ID,
        THINK_CLOSE_ID,
        THINK_OPEN_ID,
    )

    return THINK_OPEN_ID, THINK_CLOSE_ID, DSV4_EOS_ID


def test_dsv4_shadow_rekey_predicts_visible_transcript_chain():
    """Reasoning-dropping turn: predicted chain must be exactly
    prompt[:-1] + </think> + visible answer + eos (trailing output eos
    normalized to one)."""
    from vmlx_engine.scheduler import Scheduler

    topen, tclose, eos = _think_ids()
    prompt = [10, 11, 12, topen]
    output = [500, 501, tclose, 600, 601, 602, eos]

    predicted = Scheduler._dsv4_predict_shadow_rekey_tokens(prompt, output)

    assert predicted == [10, 11, 12, tclose, 600, 601, 602, eos]


def test_dsv4_shadow_rekey_appends_eos_when_output_omits_it():
    from vmlx_engine.scheduler import Scheduler

    topen, tclose, eos = _think_ids()
    predicted = Scheduler._dsv4_predict_shadow_rekey_tokens(
        [10, topen], [500, tclose, 600, 601]
    )

    assert predicted == [10, tclose, 600, 601, eos]


def test_dsv4_shadow_rekey_skips_reasoning_keeping_render():
    """THINK_OPEN in the fed history (tools-on render keeps prior
    reasoning) means the extended store already covers the next turn; a
    stripped shadow chain would never match."""
    from vmlx_engine.scheduler import Scheduler

    topen, tclose, eos = _think_ids()
    prompt = [10, topen, 500, tclose, 600, 11, topen]
    output = [700, tclose, 800, eos]

    assert Scheduler._dsv4_predict_shadow_rekey_tokens(prompt, output) is None


def test_dsv4_shadow_rekey_skips_ineligible_turns():
    from vmlx_engine.scheduler import Scheduler

    topen, tclose, eos = _think_ids()
    # No thinking rail armed.
    assert (
        Scheduler._dsv4_predict_shadow_rekey_tokens(
            [10, 11], [500, tclose, 600]
        )
        is None
    )
    # No reasoning transition in output.
    assert (
        Scheduler._dsv4_predict_shadow_rekey_tokens([10, topen], [500, 501])
        is None
    )
    # Empty visible answer (reasoning ran to eos).
    assert (
        Scheduler._dsv4_predict_shadow_rekey_tokens(
            [10, topen], [500, tclose, eos]
        )
        is None
    )
    # Empty inputs.
    assert Scheduler._dsv4_predict_shadow_rekey_tokens([], []) is None


def _dsv4_shadow_queue_fixture(*, tools_present: bool):
    from vmlx_engine.scheduler import Scheduler

    topen, tclose, eos = _think_ids()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._uses_dsv4_cache = True
    scheduler.block_aware_cache = object()
    scheduler.config = SimpleNamespace(enable_prefix_cache=True)
    scheduler._pick_cache_type_for_request = lambda _request: "assistant"
    idle_registrations = []
    scheduler._ensure_dsv4_shadow_rekey_idle_task = (
        lambda: idle_registrations.append("registered")
    )
    request = SimpleNamespace(
        prompt_token_ids=[10] * 300 + [topen],
        output_token_ids=[500, tclose, 600, eos],
        _vmlx_tools_present=tools_present,
        _cache_extra_keys=None,
    )
    return scheduler, request, idle_registrations


def test_dsv4_shadow_rekey_skips_tools_on_request():
    scheduler, request, idle_registrations = _dsv4_shadow_queue_fixture(
        tools_present=True
    )

    scheduler._queue_dsv4_shadow_rekey("tools-on", request, "stop")

    assert getattr(scheduler, "_dsv4_shadow_rekey_queue", []) == []
    assert idle_registrations == []


def test_dsv4_shadow_rekey_still_queues_tools_off_request():
    scheduler, request, idle_registrations = _dsv4_shadow_queue_fixture(
        tools_present=False
    )

    scheduler._queue_dsv4_shadow_rekey("tools-off", request, "stop")

    assert len(scheduler._dsv4_shadow_rekey_queue) == 1
    assert scheduler._dsv4_shadow_rekey_queue[0][1] == "tools-off"
    assert idle_registrations == ["registered"]


def test_dsv4_shadow_rekey_drain_race_requeues_without_store():
    from vmlx_engine.scheduler import IdleTaskResult, Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._uses_dsv4_cache = True
    scheduler.block_aware_cache = object()
    scheduler.config = SimpleNamespace(enable_prefix_cache=True)
    scheduler._dsv4_shadow_rekey_task_queued = True
    entry = ([1] * 300, "race", "assistant", None)
    scheduler._dsv4_shadow_rekey_queue = [entry]
    foreground_checks = iter((False, True))
    scheduler._foreground_pending = lambda: next(foreground_checks)
    stores = []

    def _prefill(tokens, *, should_stop, capture_dsv4_deltas):
        assert tokens == entry[0]
        assert capture_dsv4_deltas is True
        assert should_stop() is True
        return None

    scheduler._prefill_for_prompt_only_cache = _prefill
    scheduler._store_dsv4_shadow_chain = lambda *args: stores.append(args)

    result = scheduler._drain_one_dsv4_shadow_rekey()

    assert result is IdleTaskResult.PARKED
    assert scheduler._dsv4_shadow_rekey_queue == [entry]
    assert stores == []


@pytest.mark.asyncio
async def test_dsv4_llm_tools_present_reaches_engine_core_both_paths():
    from vmlx_engine.engine.batched import BatchedEngine

    calls = []

    class _Engine:
        async def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return SimpleNamespace(
                output_text="ok",
                output_token_ids=[1],
                prompt_tokens=1,
                completion_tokens=1,
                cached_tokens=0,
                cache_detail="",
                finish_reason="stop",
                logprobs=None,
            )

        async def add_request(self, **kwargs):
            calls.append(("stream", kwargs))
            return "stream-request"

        async def stream_outputs(self, request_id):
            assert request_id == "stream-request"
            yield SimpleNamespace(
                output_text="ok",
                new_text="ok",
                prompt_tokens=1,
                completion_tokens=1,
                cached_tokens=0,
                cache_detail="",
                finished=True,
                finish_reason="stop",
                logprobs=None,
            )

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _Engine()

    output = await engine.generate(
        prompt="hello",
        max_tokens=1,
        _vmlx_tools_present=True,
    )
    streamed = [
        item
        async for item in engine.stream_generate(
            prompt="hello",
            max_tokens=1,
            _vmlx_tools_present=True,
        )
    ]

    assert output.text == "ok"
    assert streamed[-1].text == "ok"
    assert [kind for kind, _kwargs in calls] == ["generate", "stream"]
    for _kind, kwargs in calls:
        assert kwargs["tools_present"] is True
        assert "_vmlx_tools_present" not in kwargs


@pytest.mark.asyncio
async def test_dsv4_tools_present_preserved_for_mllm_both_paths():
    from vmlx_engine.engine.batched import BatchedEngine

    calls = []

    class _Scheduler:
        async def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return SimpleNamespace(
                output_text="ok",
                output_token_ids=[1],
                prompt_tokens=1,
                completion_tokens=1,
                cached_tokens=0,
                cache_detail="",
                finish_reason="stop",
                logprobs=None,
            )

        async def add_request_async(self, **kwargs):
            calls.append(("stream", kwargs))
            return "mllm-stream"

        async def stream_outputs(self, request_id):
            assert request_id == "mllm-stream"
            yield SimpleNamespace(
                output_text="ok",
                new_text="ok",
                prompt_tokens=1,
                completion_tokens=1,
                cached_tokens=0,
                cache_detail="",
                finished=True,
                finish_reason="stop",
            )

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = True
    engine._mllm_scheduler = _Scheduler()

    output = await engine.generate(
        prompt="hello",
        max_tokens=1,
        _vmlx_tools_present=True,
    )
    streamed = [
        item
        async for item in engine.stream_generate(
            prompt="hello",
            max_tokens=1,
            _vmlx_tools_present=True,
        )
    ]

    assert output.text == "ok"
    assert streamed[-1].text == "ok"
    assert [kind for kind, _kwargs in calls] == ["generate", "stream"]
    for _kind, kwargs in calls:
        assert kwargs["_vmlx_tools_present"] is True
        assert "tools_present" not in kwargs


@pytest.mark.asyncio
async def test_dsv4_engine_core_attaches_tools_present_default_and_true():
    import asyncio

    from vmlx_engine.engine_core import EngineCore

    captured = []
    core = EngineCore.__new__(EngineCore)
    core.config = SimpleNamespace(stream_interval=1)
    core.scheduler = SimpleNamespace(add_request=captured.append)
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}
    core._terminal_cleanup_complete = asyncio.Event()
    core._terminal_cleanup_complete.set()

    await core.add_request(prompt=[1], request_id="default")
    await core.add_request(
        prompt=[1], request_id="tools", tools_present=True
    )

    assert captured[0]._vmlx_tools_present is False
    assert captured[1]._vmlx_tools_present is True


def test_dsv4_shadow_rekey_env_gates():
    """VMLX_DSV4_SHADOW_REKEY default-on kill switch + cap parsing."""
    from vmlx_engine.utils.dsv4_batch_generator import (
        DEFAULT_DSV4_SHADOW_REKEY_MAX_TOKENS,
        dsv4_shadow_rekey_enabled,
        dsv4_shadow_rekey_max_tokens,
    )

    for name in ("VMLX_DSV4_SHADOW_REKEY", "VMLX_DSV4_SHADOW_REKEY_MAX_TOKENS"):
        os.environ.pop(name, None)
    try:
        assert dsv4_shadow_rekey_enabled() is True
        assert (
            dsv4_shadow_rekey_max_tokens()
            == DEFAULT_DSV4_SHADOW_REKEY_MAX_TOKENS
        )
        os.environ["VMLX_DSV4_SHADOW_REKEY"] = "0"
        assert dsv4_shadow_rekey_enabled() is False
        os.environ["VMLX_DSV4_SHADOW_REKEY_MAX_TOKENS"] = "8192"
        assert dsv4_shadow_rekey_max_tokens() == 8192
        os.environ["VMLX_DSV4_SHADOW_REKEY_MAX_TOKENS"] = "bogus"
        assert (
            dsv4_shadow_rekey_max_tokens()
            == DEFAULT_DSV4_SHADOW_REKEY_MAX_TOKENS
        )
    finally:
        for name in (
            "VMLX_DSV4_SHADOW_REKEY",
            "VMLX_DSV4_SHADOW_REKEY_MAX_TOKENS",
        ):
            os.environ.pop(name, None)


def test_dsv4_shadow_rekey_idle_task_contract():
    """Source contract: the drain task must follow the vmlx#245 idle-task
    discipline (park on foreground, requeue on mid-prefill park, one entry
    per iteration) and the store must settle refs to cached-but-free."""
    import inspect
    from vmlx_engine.scheduler import Scheduler

    drain_src = inspect.getsource(Scheduler._drain_one_dsv4_shadow_rekey)
    assert "_foreground_pending" in drain_src
    assert "IdleTaskResult.PARKED" in drain_src
    assert "queue.insert(" in drain_src
    assert "_prefill_for_prompt_only_cache" in drain_src
    assert "clear_mlx_memory_cache" in drain_src
    assert "capture_dsv4_deltas=True" in drain_src

    store_src = inspect.getsource(Scheduler._store_dsv4_shadow_chain)
    assert "_dsv4_delta_transport" in store_src
    assert "expected_end = len(key_tokens)" in store_src
    assert "if not complete" in store_src
    assert "_quantize_cache_for_storage" not in store_src
    assert "store_cache" in store_src
    assert "release_request_refs" in store_src
    assert "detach_request" in store_src
    assert "enforce_byte_budget" in store_src

    queue_src = inspect.getsource(Scheduler._queue_dsv4_shadow_rekey)
    assert 'finish_reason != "stop"' in queue_src
    assert "_bypass_prefix_cache" in queue_src
    assert "_vmlx_tools_present" in queue_src
    assert "dsv4_shadow_rekey_enabled" in queue_src
    assert "dsv4_prompt_snapshot_min_tokens" in queue_src

    # The completion path must actually queue the shadow re-key.
    resp_src = inspect.getsource(Scheduler._process_batch_responses)
    assert "_queue_dsv4_shadow_rekey" in resp_src


class TestTerminalAnchorTailBudget:
    """A terminal anchor must be usable for a multi-token generation rail.

    _dsv4_checkpoint_boundary admitted a terminal anchor only when the request
    needed at most ONE further token. DSV4's rail is three, so the visible-answer
    pass could never restore the anchor its own first pass had just stored: it
    matched 127,444, fell back to the block-aligned 127,232 checkpoint, and
    re-prefilled 215 tokens to change the single rail token that differs.

    Live A/B at 127k, same prompt:
      off: cached 127,232  fresh 215  wall 21.97s  stall 5.66s  decode mean  8.8
      on : cached 127,444  fresh   3  wall 13.74s  stall 1.64s  decode mean 16.0
    decode p50 was 24.7 vs 24.9 — steady decode untouched. Output was
    byte-identical (hash 9884a471bb9bc9a5 both ways) while cached_tokens
    differed, proving the restore path was exercised.
    """

    def test_defaults_to_the_generation_rail_length(self):
        from vmlx_engine.prefix_cache import _dsv4_terminal_anchor_tail_budget

        # 3-token rail plus one token of margin.
        assert _dsv4_terminal_anchor_tail_budget() >= 3

    def test_env_zero_restores_the_historical_single_token_bound(self, monkeypatch):
        from vmlx_engine.prefix_cache import _dsv4_terminal_anchor_tail_budget

        monkeypatch.setenv("VMLX_DSV4_TERMINAL_ANCHOR_TAIL", "0")
        assert _dsv4_terminal_anchor_tail_budget() == 0

    def test_env_override_is_honoured(self, monkeypatch):
        from vmlx_engine.prefix_cache import _dsv4_terminal_anchor_tail_budget

        monkeypatch.setenv("VMLX_DSV4_TERMINAL_ANCHOR_TAIL", "9")
        assert _dsv4_terminal_anchor_tail_budget() == 9

    def test_garbage_and_negative_fall_back_to_the_default(self, monkeypatch):
        from vmlx_engine.prefix_cache import _dsv4_terminal_anchor_tail_budget

        for bad in ("garbage", "-3", ""):
            monkeypatch.setenv("VMLX_DSV4_TERMINAL_ANCHOR_TAIL", bad)
            value = _dsv4_terminal_anchor_tail_budget()
            assert value >= 0, f"{bad!r} produced {value}"

    def test_budget_is_actually_applied_to_allow_terminal(self):
        """Structural pin: the budget must gate allow_terminal, not sit unused."""
        import inspect

        from vmlx_engine.prefix_cache import BlockAwarePrefixCache

        src = inspect.getsource(BlockAwarePrefixCache)
        assert "_dsv4_terminal_anchor_tail_budget()" in src
        assert "allow_terminal = len(request_tokens) - matched <= max(1," in src
