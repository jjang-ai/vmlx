# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM Scheduler cache infrastructure.

Covers:
- MLLMSchedulerConfig cache field parity with SchedulerConfig
- Cache init chain (paged > memory-aware > legacy)
- _ensure_batch_generator clears all cache modes
- _cleanup_finished stores to all cache paths
- HybridSSMStateCache companion cache
- Metal optimization settings
- get_stats() cache reporting
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig, MLLMScheduler
from vmlx_engine.mllm_batch_generator import (
    _prefix_hit_tail_and_cached_tokens as _mllm_prefix_hit_tail_and_cached_tokens,
)


def test_mllm_scheduler_does_not_shadow_hashlib_in_init():
    """Regression: paged block-disk init uses hashlib before non-paged L2 setup."""
    import inspect

    source = inspect.getsource(MLLMScheduler.__init__)

    assert "import hashlib" not in source
    assert "hashlib.sha256" in source


def test_batched_template_tool_call_arguments_are_mappings():
    """OpenAI tool-call history uses JSON strings; Jinja templates need dicts."""
    from vmlx_engine.engine.batched import BatchedEngine

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "record_fact",
                        "arguments": "{\"value\":\"blue-cat\"}",
                    },
                }
            ],
        }
    ]

    normalized = BatchedEngine._normalize_tool_call_arguments_for_template(messages)

    args = normalized[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {"value": "blue-cat"}
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == "{\"value\":\"blue-cat\"}"


def test_mllm_tight_memory_text_prefill_chunks_short_tool_prompts():
    """Tight-memory text-only MLLM prefill must not fall back to one-shot LM."""
    import inspect
    import vmlx_engine.mllm_batch_generator as mbg

    source = inspect.getsource(mbg)

    assert "VMLINUX_TIGHT_MEMORY_PREFILL_STEP_SIZE" in source
    assert "seq_len > _tight_text_prefill_step_size + 1" in source
    assert "chunk_size = min(_tight_text_prefill_step_size" in source


def test_mimo_v2_cache_extraction_preserves_swa_kv_heads():
    """MiMo full/SWA layers have different legal KV head counts.

    The primary config has num_key_value_heads=4, while rotating SWA layers use
    swa_num_key_value_heads=8. Prefix-cache storage must not normalize the SWA
    layer down to 4 heads, or the next cache hit crashes when RotatingKVCache
    tries to concatenate 4-head cached state with 8-head live suffix KV.
    """
    import mlx.core as mx

    class FullKV:
        state = (
            mx.zeros((1, 4, 5, 192), mx.float32),
            mx.zeros((1, 4, 5, 192), mx.float32),
        )
        meta_state = (5,)

    class RotatingKVCache:
        state = (
            mx.zeros((1, 8, 5, 192), mx.float32),
            mx.zeros((1, 8, 5, 192), mx.float32),
        )
        meta_state = (128, 0, 5, 5)

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="mimo_v2",
            num_key_value_heads=4,
            swa_num_key_value_heads=8,
        )
    )

    extracted = scheduler._extract_cache_states([FullKV(), RotatingKVCache()])

    assert extracted[0]["state"][0].shape[1] == 4
    assert extracted[1]["class_name"] == "RotatingKVCache"
    assert extracted[1]["state"][0].shape[1] == 8


def test_mllm_scheduler_clamps_active_turboquant_kv_to_single_sequence():
    """VLM TurboQuant KV must not advertise multi-seq batching it cannot run."""

    def _turboquant_make_cache():
        return []

    class LanguageModel:
        make_cache = staticmethod(_turboquant_make_cache)

    class VLMModel:
        language_model = LanguageModel()
        config = object()

    config = MLLMSchedulerConfig(
        enable_prefix_cache=False,
        kv_cache_quantization="none",
        max_num_seqs=256,
        prefill_batch_size=8,
        completion_batch_size=32,
    )

    scheduler = MLLMScheduler(VLMModel(), processor=object(), config=config)

    assert scheduler._tq_active is True
    assert scheduler.config.max_num_seqs == 1
    assert scheduler.config.prefill_batch_size == 1
    assert scheduler.config.completion_batch_size == 1


def test_mllm_scheduler_preserves_turboquant_batch_api_config():
    """VLM TurboQuant KV may batch when jang_tools exposes batch API v1."""

    class TurboQuantKVCache:
        _vmlx_batch_api = "turboquant_kv_v1"

        def extend(self, other):
            return None

        def filter(self, keep):
            return None

        def extract(self, idx):
            return self

        def prepare(self, *args, **kwargs):
            return None

        def finalize(self):
            return None

    def _turboquant_make_cache():
        return [TurboQuantKVCache()]

    class LanguageModel:
        make_cache = staticmethod(_turboquant_make_cache)

    class VLMModel:
        language_model = LanguageModel()
        config = object()

    config = MLLMSchedulerConfig(
        enable_prefix_cache=False,
        kv_cache_quantization="none",
        max_num_seqs=8,
        prefill_batch_size=16,
        completion_batch_size=32,
    )

    scheduler = MLLMScheduler(VLMModel(), processor=object(), config=config)

    assert scheduler._tq_active is True
    assert scheduler._tq_batch_api is True
    assert scheduler.config.max_num_seqs == 8
    assert scheduler.config.prefill_batch_size == 16
    assert scheduler.config.completion_batch_size == 32


def test_mllm_merge_preserves_turboquant_batch_api_cache():
    """MLLM batch merge must call TurboQuantKVCache.extend, not BatchKVCache."""
    from vmlx_engine.mllm_batch_generator import _merge_caches

    class TurboQuantKVCache:
        _vmlx_batch_api = "turboquant_kv_v1"

        def __init__(self, name):
            self.name = name
            self.extended = []

        def extend(self, other):
            self.extended.append(other.name)

        def filter(self, keep):
            return None

        def extract(self, idx):
            return self

        def prepare(self, *args, **kwargs):
            return None

        def finalize(self):
            return None

    merged = _merge_caches([
        [TurboQuantKVCache("a")],
        [TurboQuantKVCache("b")],
    ])

    assert type(merged[0]).__name__ == "TurboQuantKVCache"
    assert merged[0].name == "a"
    assert merged[0].extended == ["b"]


def test_mllm_audio_payload_prefill_uses_model_wrapper_not_text_fast_path():
    """Audio media must not be dropped by the text-only language-model shortcut."""
    import mlx.core as mx
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    class BadLanguageModel:
        def __call__(self, *args, **kwargs):
            raise AssertionError("audio media request used text-only language_model")

    class CapturingModel:
        def __init__(self):
            self.kwargs = None

        def __call__(self, input_ids, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(logits=mx.zeros((1, input_ids.shape[-1], 4)))

    model = CapturingModel()
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.model = model
    generator.language_model = BadLanguageModel()
    generator._is_hybrid = False
    generator._model_type = "mimo_v2"
    generator.prefill_step_size = 128

    request = SimpleNamespace(
        input_ids=mx.array([1, 151669, 2]),
        pixel_values=None,
        attention_mask=None,
        image_grid_thw=None,
        audio_codes=mx.array([[1, 2], [3, 4]], dtype=mx.int32),
        audio_embeds=None,
        audio_features=None,
        extra_kwargs={},
        vision_encoded=False,
    )

    logits = generator._run_vision_encoding_inner(request, cache=[])

    assert logits.shape == (1, 3, 4)
    assert model.kwargs["audio_codes"].tolist() == [[1, 2], [3, 4]]
    assert request.vision_encoded is True


def test_mllm_processor_audio_outputs_are_promoted_to_request_fields():
    """Processor-returned audio tensors must not stay buried in extra_kwargs."""
    import inspect
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    source = inspect.getsource(MLLMBatchGenerator._preprocess_request)

    assert 'request.extra_kwargs.pop("audio_codes", None)' in source
    assert 'request.extra_kwargs.pop("audio_embeds", None)' in source
    assert 'request.extra_kwargs.pop("audio_features", None)' in source
    assert "request.audio_codes = _ensure_mx_array(" in source
    assert "request.audio_embeds = _ensure_mx_array(" in source
    assert "request.audio_features = _ensure_mx_array(" in source


def test_mllm_processor_direct_forwards_raw_audio_to_processor():
    """Raw audio request inputs must reach processors that expose audio kwargs."""
    from vmlx_engine.mllm_batch_generator import _call_processor_direct

    captured = {}

    class Processor:
        def __call__(
            self,
            *,
            text,
            padding,
            return_tensors,
            add_special_tokens,
            audio=None,
            audios=None,
        ):
            captured.update(
                {
                    "text": text,
                    "padding": padding,
                    "return_tensors": return_tensors,
                    "add_special_tokens": add_special_tokens,
                    "audio": audio,
                    "audios": audios,
                }
            )
            return {"input_ids": [[1, 2, 3]], "audio_codes": [[4, 5]]}

    result = _call_processor_direct(
        Processor(),
        prompts=["describe audio"],
        images=None,
        videos=None,
        audio=["/tmp/vmlx-audio.wav"],
        add_special_tokens=False,
    )

    assert captured["audio"] == ["/tmp/vmlx-audio.wav"]
    assert captured["audios"] == ["/tmp/vmlx-audio.wav"]
    assert result["audio_codes"] == [[4, 5]]


def test_mllm_scheduler_and_batched_engine_route_raw_audio_requests():
    """OpenAI audio parts must reach MLLM scheduler, not stop at API parsing."""
    import inspect
    from dataclasses import fields

    from vmlx_engine.engine.batched import BatchedEngine
    from vmlx_engine.mllm_scheduler import MLLMRequest, MLLMScheduler

    assert "audio" in {field.name for field in fields(MLLMRequest)}
    add_source = inspect.getsource(MLLMScheduler.add_request)
    assert "audio: Optional[List[Any]] = None" in add_source
    assert "audio=audio" in add_source
    assert "not images and not videos and not audio" in add_source
    schedule_source = inspect.getsource(MLLMScheduler._schedule_waiting)
    assert "audio=request.audio" in schedule_source

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "UklGRg==",
                        "format": "wav",
                    },
                },
            ],
        }
    ]
    extracted = BatchedEngine._extract_audio_content(messages)
    assert extracted == [{"data": "UklGRg==", "format": "wav"}]

    engine_source = inspect.getsource(BatchedEngine)
    assert "extracted_audio = self._extract_audio_content(messages)" in engine_source
    assert "audio=all_audio if all_audio else None" in engine_source


def test_mllm_exact_prefix_hit_with_generation_suffix_refeeds_last_prompt_token():
    """VLM memory/legacy hits store N-1 KV, so suffix-only prefill is wrong."""

    remaining, cached_tokens = _mllm_prefix_hit_tail_and_cached_tokens(
        token_list=[10, 11, 12, 13],
        remaining=[],
        gen_prompt_suffix=[90, 91],
    )

    assert remaining == [13, 90, 91]
    assert cached_tokens == 3


def test_mllm_exact_prefix_hit_without_suffix_counts_n_minus_one_cached_tokens():
    remaining, cached_tokens = _mllm_prefix_hit_tail_and_cached_tokens(
        token_list=[10, 11, 12, 13],
        remaining=[],
        gen_prompt_suffix=[],
    )

    assert remaining == []
    assert cached_tokens == 3


# ============================================================
# Config parity tests
# ============================================================


class TestMLLMSchedulerConfigParity:
    """Verify all cache-related fields exist on MLLMSchedulerConfig."""

    def test_memory_aware_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "use_memory_aware_cache")
        assert hasattr(config, "cache_memory_mb")
        assert hasattr(config, "cache_memory_percent")
        assert hasattr(config, "cache_ttl_minutes")

    def test_memory_aware_defaults(self):
        config = MLLMSchedulerConfig()
        assert config.use_memory_aware_cache is True
        assert config.cache_memory_mb is None
        assert config.cache_memory_percent == 0.20
        assert config.cache_ttl_minutes == 0

    def test_legacy_prefix_cache_field(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "prefix_cache_size")
        assert config.prefix_cache_size == 100

    def test_ssm_companion_budget_defaults(self):
        config = MLLMSchedulerConfig()
        assert config.ssm_state_cache_size == 8
        assert config.ssm_state_cache_max_mb == 512

    def test_disk_cache_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "enable_disk_cache")
        assert hasattr(config, "disk_cache_dir")
        assert hasattr(config, "disk_cache_max_gb")
        assert config.enable_disk_cache is False
        assert config.disk_cache_dir is None
        assert config.disk_cache_max_gb == 10.0

    def test_block_disk_cache_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "enable_block_disk_cache")
        assert hasattr(config, "block_disk_cache_dir")
        assert hasattr(config, "block_disk_cache_max_gb")
        assert config.enable_block_disk_cache is False
        assert config.block_disk_cache_dir is None
        assert config.block_disk_cache_max_gb == 10.0

    def test_model_path_field(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "model_path")
        assert config.model_path is None

    def test_kv_cache_quantization_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "kv_cache_quantization")
        assert hasattr(config, "kv_cache_group_size")
        assert config.kv_cache_quantization == "none"
        assert config.kv_cache_group_size == 64

    def test_paged_cache_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "enable_prefix_cache")
        assert hasattr(config, "use_paged_cache")
        assert hasattr(config, "paged_cache_block_size")
        assert hasattr(config, "max_cache_blocks")
        assert config.enable_prefix_cache is True
        assert config.use_paged_cache is True
        assert config.paged_cache_block_size == 64
        assert config.max_cache_blocks == 1000

    def test_custom_values(self):
        config = MLLMSchedulerConfig(
            cache_memory_mb=2048,
            cache_memory_percent=0.30,
            cache_ttl_minutes=5.0,
            prefix_cache_size=200,
            enable_disk_cache=True,
            disk_cache_dir="/tmp/test-cache",
            disk_cache_max_gb=20.0,
            model_path="/models/test-vlm",
        )
        assert config.cache_memory_mb == 2048
        assert config.cache_memory_percent == 0.30
        assert config.cache_ttl_minutes == 5.0
        assert config.prefix_cache_size == 200
        assert config.enable_disk_cache is True
        assert config.disk_cache_dir == "/tmp/test-cache"
        assert config.disk_cache_max_gb == 20.0
        assert config.model_path == "/models/test-vlm"

    def test_all_config_fields_match_scheduler_config(self):
        """Ensure MLLM config has all cache-related fields from SchedulerConfig."""
        from vmlx_engine.scheduler import SchedulerConfig

        cache_field_names = {
            "enable_prefix_cache",
            "use_paged_cache",
            "paged_cache_block_size",
            "max_cache_blocks",
            "kv_cache_quantization",
            "kv_cache_group_size",
            "use_memory_aware_cache",
            "cache_memory_mb",
            "cache_memory_percent",
            "cache_ttl_minutes",
            "ssm_state_cache_size",
            "ssm_state_cache_max_mb",
            "prefix_cache_size",
            "enable_disk_cache",
            "disk_cache_dir",
            "disk_cache_max_gb",
            "enable_block_disk_cache",
            "block_disk_cache_dir",
            "block_disk_cache_max_gb",
            "model_path",
        }

        mllm_field_names = {f.name for f in fields(MLLMSchedulerConfig)}
        scheduler_field_names = {f.name for f in fields(SchedulerConfig)}

        for field_name in cache_field_names:
            assert field_name in scheduler_field_names, (
                f"Field '{field_name}' missing from SchedulerConfig"
            )
            assert field_name in mllm_field_names, (
                f"Field '{field_name}' missing from MLLMSchedulerConfig"
            )


# ============================================================
# Config forwarding from batched.py
# ============================================================


class TestConfigForwarding:
    """Verify BatchedEngine._start_mllm() forwards all settings."""

    def test_mllm_config_fields_forwarded(self):
        """Check that _start_mllm creates MLLMSchedulerConfig with all fields."""
        from vmlx_engine.engine.batched import BatchedEngine
        from vmlx_engine.scheduler import SchedulerConfig
        import inspect

        # Get the source of _start_mllm to verify forwarding
        source = inspect.getsource(BatchedEngine._start_mllm)

        forwarded_fields = [
            "use_memory_aware_cache",
            "cache_memory_mb",
            "cache_memory_percent",
            "cache_ttl_minutes",
            "ssm_state_cache_size",
            "ssm_state_cache_max_mb",
            "prefix_cache_size",
            "enable_disk_cache",
            "disk_cache_dir",
            "disk_cache_max_gb",
            "enable_block_disk_cache",
            "block_disk_cache_dir",
            "block_disk_cache_max_gb",
            "model_path",
        ]

        for field_name in forwarded_fields:
            assert field_name in source, (
                f"Field '{field_name}' not forwarded in _start_mllm()"
            )

    def test_batched_engine_logs_effective_mllm_batch_sizes_after_clamps(self):
        """Startup log must not report requested batch sizes after scheduler clamps."""
        from vmlx_engine.engine.batched import BatchedEngine
        import inspect

        source = inspect.getsource(BatchedEngine._start_mllm)

        log_tail = source.split("MLLM Scheduler started with continuous batching:", 1)[1]
        assert "self._mllm_scheduler.config.max_num_seqs" in log_tail
        assert "self._mllm_scheduler.config.prefill_batch_size" in log_tail
        assert "self._mllm_scheduler.config.completion_batch_size" in log_tail


# ============================================================
# HybridSSMStateCache tests
# ============================================================


class TestHybridSSMStateCache:
    """Tests for the companion SSM state cache."""

    def test_import(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache
        cache = HybridSSMStateCache(max_entries=10)
        assert cache is not None

    def test_store_and_fetch(self):
        # Updated 2026-04-08 (Agent 3, REQ-A3-001): fetch() now returns
        # a (states, is_complete) tuple per the SSMCompanionCache extraction.
        # The legacy `result is ssm_states` identity assertion was wrong even
        # before the API change because fetch() deep-copies per session
        # 2026-03-28b root cause fix. New assertion verifies tuple shape +
        # default is_complete=True; for the canonical deep-copy independence
        # check see tests/test_ssm_companion_cache.py.
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        tokens = [1, 2, 3, 4, 5]
        ssm_states = [MagicMock(), MagicMock()]

        cache.store(tokens, 5, ssm_states)
        result = cache.fetch(tokens, 5)

        assert result is not None
        states, is_complete = result
        assert is_complete is True
        assert len(states) == 2

    def test_fetch_miss(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        result = cache.fetch([1, 2, 3], 3)
        assert result is None

    def test_lru_eviction(self):
        # Updated 2026-04-08 (Agent 3, REQ-A3-001): fetch() returns tuple.
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=2)

        cache.store([1, 2], 2, ["state_a"])
        cache.store([3, 4], 2, ["state_b"])
        cache.store([5, 6], 2, ["state_c"])  # Should evict [1,2]

        assert cache.fetch([1, 2], 2) is None  # Evicted
        r_b = cache.fetch([3, 4], 2)
        assert r_b is not None and r_b[0] == ["state_b"]
        r_c = cache.fetch([5, 6], 2)
        assert r_c is not None and r_c[0] == ["state_c"]

    def test_lru_access_refresh(self):
        # Updated 2026-04-08 (Agent 3, REQ-A3-001): fetch() returns tuple.
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=2)

        cache.store([1, 2], 2, ["state_a"])
        cache.store([3, 4], 2, ["state_b"])

        # Access [1,2] to refresh its position
        cache.fetch([1, 2], 2)

        # Now store [5,6] — should evict [3,4] (oldest), not [1,2]
        cache.store([5, 6], 2, ["state_c"])

        r_a = cache.fetch([1, 2], 2)
        assert r_a is not None and r_a[0] == ["state_a"]
        assert cache.fetch([3, 4], 2) is None  # Evicted
        r_c = cache.fetch([5, 6], 2)
        assert r_c is not None and r_c[0] == ["state_c"]

    def test_clear(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        cache.store([1, 2, 3], 3, ["state"])
        cache.clear()
        assert cache.fetch([1, 2, 3], 3) is None

    def test_prefix_keying(self):
        """Verify that cache is keyed by prefix, not full token list."""
        # Updated 2026-04-08 (Agent 3, REQ-A3-001): fetch() returns tuple.
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        tokens_full = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Store with 5-token prefix
        cache.store(tokens_full, 5, ["state_5"])

        # Fetch with same 5-token prefix but different suffix
        tokens_different_suffix = [1, 2, 3, 4, 5, 99, 98, 97]
        result = cache.fetch(tokens_different_suffix, 5)
        assert result is not None
        states, is_complete = result
        assert states == ["state_5"]
        assert is_complete is True

    def test_different_lengths_different_keys(self):
        """Same prefix tokens but different num_tokens → different keys."""
        # Updated 2026-04-08 (Agent 3, REQ-A3-001): fetch() returns tuple.
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        tokens = [1, 2, 3, 4, 5]

        cache.store(tokens, 3, ["state_3"])
        cache.store(tokens, 5, ["state_5"])

        r3 = cache.fetch(tokens, 3)
        r5 = cache.fetch(tokens, 5)
        assert r3 is not None and r3[0] == ["state_3"]
        assert r5 is not None and r5[0] == ["state_5"]

    def test_fetch_longest_prefix_records_match_and_miss_diagnostics(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        cache.store([1, 2, 3, 4], 4, ["state_4"])

        hit = cache.fetch_longest_prefix([1, 2, 3, 4, 99], 5)
        assert hit is not None
        assert hit[0] == 4
        assert cache.last_prefix_lookup == {
            "max_len": 5,
            "candidate_lengths": [4],
            "matched": True,
            "checkpoint_tokens": 4,
            "is_complete": True,
        }

        miss = cache.fetch_longest_prefix([9, 2, 3, 4], 4)
        assert miss is None
        assert cache.last_prefix_lookup == {
            "max_len": 4,
            "candidate_lengths": [4],
            "matched": False,
            "reason": "prefix_hash_mismatch",
            "store_size": 1,
        }


# ============================================================
# MLLMBatchGenerator cache params tests
# ============================================================


class TestBatchGeneratorCacheParams:
    """Test that MLLMBatchGenerator accepts all cache params."""

    def test_absolute_text_position_ids_continue_from_cache_offset(self):
        """Text-only cache-hit tails must use absolute mRoPE positions.

        Qwen3.6 hybrid resumed a 627-token cached prefix, then passed only
        the 30-token tail to the language model. With `_rope_deltas` reset,
        mlx-vlm recomputed position_ids from tail length starting at zero.
        That made cached prefill diverge from full prefill even though KV+SSM
        cache state existed.
        """
        import mlx.core as mx
        from vmlx_engine.mllm_batch_generator import _absolute_text_position_ids

        class OffsetCache:
            offset = 627

        class InnerModel:
            fa_idx = 0

        language_model = SimpleNamespace(model=InnerModel())

        pos = _absolute_text_position_ids(
            mx.array([[11, 12, 13, 14]]),
            [OffsetCache()],
            language_model,
        )

        assert pos is not None
        assert pos.shape == (3, 1, 4)
        assert pos[0, 0].tolist() == [627, 628, 629, 630]

    def test_absolute_text_position_ids_zero_offset_is_still_explicit(self):
        """Qwen text-only MLLM prefill should not depend on cached rope deltas."""
        import mlx.core as mx
        from vmlx_engine.mllm_batch_generator import _absolute_text_position_ids

        class OffsetCache:
            offset = 0

        class InnerModel:
            fa_idx = 0

        language_model = SimpleNamespace(model=InnerModel())

        pos = _absolute_text_position_ids(
            mx.array([[21, 22, 23]]),
            [OffsetCache()],
            language_model,
        )

        assert pos is not None
        assert pos.shape == (3, 1, 3)
        assert pos[0, 0].tolist() == [0, 1, 2]

    def test_lm_supports_position_ids_uses_call_signature(self):
        from vmlx_engine.mllm_batch_generator import _lm_supports_position_ids

        class StepLikeLM:
            def __call__(self, inputs, cache=None):
                return inputs

        class QwenLikeLM:
            def __call__(self, inputs, cache=None, position_ids=None):
                return inputs

        assert _lm_supports_position_ids(StepLikeLM()) is False
        assert _lm_supports_position_ids(QwenLikeLM()) is True

        class StepLikeLMWithKwargs:
            def __call__(self, inputs, cache=None, **kwargs):
                return inputs

        assert _lm_supports_position_ids(StepLikeLMWithKwargs()) is True

    def test_lm_supports_return_logits_uses_var_kwargs(self):
        from vmlx_engine.mllm_batch_generator import _lm_supports_return_logits

        class StepLikeLM:
            def __call__(self, inputs, cache=None):
                return inputs

        class QwenLikeLM:
            def __call__(self, inputs, cache=None, return_logits=True):
                return inputs

        class StepLikeLMWithKwargs:
            def __call__(self, inputs, cache=None, **kwargs):
                return inputs

        assert _lm_supports_return_logits(StepLikeLM()) is False
        assert _lm_supports_return_logits(QwenLikeLM()) is True
        assert _lm_supports_return_logits(StepLikeLMWithKwargs()) is True

    def test_init_signature(self):
        """Verify constructor accepts all cache params."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        sig = inspect.signature(MLLMBatchGenerator.__init__)
        param_names = list(sig.parameters.keys())

        expected = [
            "memory_aware_cache",
            "prefix_cache",
            "disk_cache",
            "kv_cache_bits",
            "kv_cache_group_size",
        ]
        for name in expected:
            assert name in param_names, f"Missing param: {name}"

    def test_accepts_scheduler_owned_ssm_l2_store(self):
        """Hybrid MLLM prefix hits are only real across restart when the
        generator's SSM companion cache receives the scheduler-owned disk L2.

        Qwen3.6 JANGTQ live proof showed a false-positive restart hit: paged
        KV blocks restored from BlockDiskStore, but the companion SSM cache had
        no disk store and forced a full prefill. Pin the constructor contract
        that lets MLLMScheduler attach the matching SSM L2 namespace.
        """
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        disk_store = MagicMock()
        generator = MLLMBatchGenerator(
            model=MagicMock(),
            processor=MagicMock(),
            paged_cache_manager=MagicMock(),
            block_aware_cache=MagicMock(),
            ssm_state_disk_store=disk_store,
            ssm_state_cache_model_key="qwen36-test-key",
        )

        cache = generator._ssm_state_cache
        assert cache is not None
        assert cache.disk_enabled is True
        assert cache.model_key == "qwen36-test-key"


# ============================================================
# MLLMBatch extract_cache contiguous tests
# ============================================================


class TestMLLMBatchExtractContiguous:
    """Test contiguous enforcement in extract_cache."""

    def test_extract_calls_contiguous(self):
        """Verify extract_cache makes keys/values contiguous."""
        import mlx.core as mx
        from vmlx_engine.mllm_batch_generator import MLLMBatch

        # Create a mock batch with a mock cache that has extract()
        mock_kv = MagicMock()
        mock_kv.keys = mx.zeros((1, 4, 8, 64))  # Already contiguous
        mock_kv.values = mx.zeros((1, 4, 8, 64))

        mock_cache = MagicMock()
        mock_cache.extract.return_value = mock_kv

        batch = MLLMBatch(
            uids=[0],
            request_ids=["req-1"],
            y=mx.array([0]),
            logprobs=[mx.zeros(10)],
            max_tokens=[100],
            num_tokens=[0],
            cache=[mock_cache],
            requests=[],
        )

        result = batch.extract_cache(0)
        assert len(result) == 1
        mock_cache.extract.assert_called_once_with(0)
        # The result should have keys/values set
        layer = result[0]
        assert layer.keys is not None
        assert layer.values is not None


# ============================================================
# _ensure_batch_generator clears all cache modes
# ============================================================


class TestEnsureBatchGeneratorCacheClearing:
    """Test that _ensure_batch_generator updates sampler in place (preserves caches)."""

    def test_updates_sampler_in_place(self):
        """Verify sampler updated in place without cache invalidation."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        assert "batch_generator.sampler" in source, \
            "Must update sampler in place on existing generator"

    def test_preserves_caches_on_param_change(self):
        """Verify caches NOT cleared when sampling params change."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        # The old code had cache clearing; now it should NOT clear caches
        # when only sampling params change (per-request samplers handle it)
        assert "_current_sampler_params" in source, \
            "Must track current sampler params"

    def test_does_not_clear_caches_on_temp_change(self):
        """Verify temperature change does NOT wipe all prefix caches."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        # When generator already exists, should NOT close and recreate
        lines = source.split('\n')
        # Check that the in-place update path exists (batch_generator.sampler = ...)
        has_inplace = any('batch_generator.sampler' in line for line in lines)
        assert has_inplace, "Must have in-place sampler update path"


class TestHybridSSMBlockDiskWiring:
    """Regression pins for hybrid SSM companion L2 on the MLLM path."""

    def test_scheduler_creates_matching_ssm_companion_l2_for_block_disk(self):
        """Block-disk cache for hybrid VLMs must include the companion SSM
        disk store, not just paged KV blocks. Otherwise /v1/cache/stats can
        report block hits while generation still pays full SSM prefill.
        """
        config = MLLMSchedulerConfig(
            enable_prefix_cache=True,
            use_paged_cache=True,
            enable_block_disk_cache=True,
            block_disk_cache_dir="/tmp/vmlx-test-block-cache",
            block_disk_cache_max_gb=0.001,
        )
        source_init = Path("vmlx_engine/mllm_scheduler.py").read_text()
        import inspect
        source_ensure = inspect.getsource(MLLMScheduler._ensure_batch_generator)

        assert "SSMCompanionDiskStore" in source_init
        assert "_ssm_companion_disk_store" in source_init
        assert "ssm_companion" in source_init
        assert "ssm_state_disk_store" in source_ensure
        assert config.enable_block_disk_cache is True

    def test_scheduler_init_log_distinguishes_prompt_and_block_l2(self):
        """Startup logs must not hide block-disk L2 behind prompt L2 status."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        log_section = source[source.find("MLLM Scheduler initialized") :]

        assert "_block_disk_l2_enabled" in source
        assert "prompt_l2=" in log_section
        assert "block_l2=" in log_section
        assert "disk_l2=" not in log_section


# ============================================================
# _cleanup_finished cache store paths
# ============================================================


class TestCleanupFinishedCacheStore:
    """Verify _cleanup_finished stores to all 3 cache paths."""

    def test_has_paged_store_path(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "block_aware_cache" in source
        assert "store_cache" in source

    def test_has_memory_aware_store_path(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "memory_aware_cache" in source
        assert "memory_aware_cache.store" in source.replace("self.", "")

    def test_has_legacy_store_path(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "prefix_cache" in source
        assert "prefix_cache.store_cache" in source.replace("self.", "")

    def test_has_disk_cache_l2_writes(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "disk_cache" in source

    def test_scheduler_trace_logs_cleanup_subsegments(self, monkeypatch, caplog):
        import logging

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=1,
            _bypass_prefix_cache=True,
            _extracted_cache=None,
            _added_stop_tokens=set(),
        )
        scheduler.running = {"trace-cleanup": request}
        scheduler.block_aware_cache = None
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.batch_generator = None
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = None
        scheduler.requests = {"trace-cleanup": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        monkeypatch.setenv("VMLINUX_MLLM_SCHEDULER_TRACE", "1")

        with patch("vmlx_engine.mllm_scheduler.clear_mlx_memory_cache") as clear_memory, caplog.at_level(
            logging.INFO, logger="vmlx_engine.mllm_scheduler"
        ):
            scheduler._cleanup_finished({"trace-cleanup"})

        clear_memory.assert_called_once()
        assert "VMLINUX_MLLM_CLEANUP_TRACE finished=1" in caplog.text
        assert "cache_store_ms=" in caplog.text
        assert "bookkeeping_ms=" in caplog.text
        assert "clear_memory_ms=" in caplog.text

    def test_mixed_swa_full_prefix_hit_skips_redundant_clean_store(self):
        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=32,
            _bypass_prefix_cache=False,
            _cached_tokens=4,
            _extracted_cache=lambda: ["dirty-live-cache"],
            _extracted_tokens=[10, 11, 12, 13, 14],
            _added_stop_tokens=set(),
        )

        scheduler.running = {"req-hit": request}
        scheduler.block_aware_cache = MagicMock()
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.disk_cache = None
        scheduler._is_hybrid = False
        scheduler._kv_cache_bits = 0
        scheduler._uses_zaya_cache = False
        scheduler._mixed_attention_cache_model = True
        scheduler._mllm_request_has_media_cache_context = MagicMock(return_value=False)
        scheduler._mllm_media_prefix_cache_allowed = MagicMock(return_value=False)
        scheduler._truncate_hybrid_cache = MagicMock(return_value=["prompt-cache"])
        scheduler._validate_cache = MagicMock(return_value=True)
        scheduler._extract_cache_states = MagicMock(return_value=["state"])
        scheduler.batch_generator = SimpleNamespace(
            _prefill_for_clean_path_dependent_cache=MagicMock(
                return_value=["clean-cache"]
            )
        )
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = MagicMock()
        scheduler.requests = {"req-hit": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        scheduler._cleanup_finished({"req-hit"})

        scheduler.batch_generator._prefill_for_clean_path_dependent_cache.assert_not_called()
        scheduler.block_aware_cache.store_cache.assert_not_called()

    def test_media_request_skips_memory_aware_token_only_store(self):
        """Image/video VLM requests must not be stored under text-only keys."""

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=1,
            _has_history=False,
            _extracted_cache=lambda: ["live-cache"],
            _extracted_tokens=[10, 11, 12, 13, 14],
            _added_stop_tokens=set(),
            images=["image-a.png"],
            videos=None,
        )

        scheduler.running = {"req-media-memory": request}
        scheduler.block_aware_cache = None
        scheduler.memory_aware_cache = MagicMock()
        scheduler.prefix_cache = None
        scheduler.disk_cache = MagicMock()
        scheduler._is_hybrid = False
        scheduler._kv_cache_bits = 0
        scheduler._truncate_hybrid_cache = MagicMock(return_value=["prompt-cache"])
        scheduler._validate_cache = MagicMock(return_value=True)
        scheduler.batch_generator = MagicMock()
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = None
        scheduler.requests = {"req-media-memory": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        scheduler._cleanup_finished({"req-media-memory"})

        scheduler.disk_cache.store.assert_not_called()
        scheduler.memory_aware_cache.store.assert_not_called()

    def test_media_placeholder_skips_legacy_token_only_store(self):
        """Text containing media placeholder ids must not populate legacy cache."""

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=1,
            _has_history=False,
            _extracted_cache=lambda: ["live-cache"],
            _extracted_tokens=[10, 999, 12, 13, 14],
            _added_stop_tokens=set(),
            images=None,
            videos=None,
        )

        scheduler.running = {"req-media-prefix": request}
        scheduler.block_aware_cache = None
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = MagicMock()
        scheduler.disk_cache = MagicMock()
        scheduler._is_hybrid = False
        scheduler._kv_cache_bits = 0
        scheduler._truncate_hybrid_cache = MagicMock(return_value=["prompt-cache"])
        scheduler._validate_cache = MagicMock(return_value=True)
        scheduler.batch_generator = SimpleNamespace(
            _tokens_contain_media_placeholders=lambda toks: 999 in toks
        )
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = None
        scheduler.requests = {"req-media-prefix": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        scheduler._cleanup_finished({"req-media-prefix"})

        scheduler.disk_cache.store.assert_not_called()
        scheduler.prefix_cache.store_cache.assert_not_called()

    @pytest.mark.parametrize("cache_mode", ["paged", "memory", "legacy"])
    def test_media_cache_store_skip_still_cleans_finished_request(self, cache_mode):
        """Media cache-safety skip must not leave a phantom running request."""

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=1,
            _has_history=False,
            _extracted_cache=lambda: ["live-cache"],
            _extracted_tokens=[10, 11, 12, 13, 14],
            _added_stop_tokens=set(),
            images=["image-a.png"],
            videos=None,
        )

        scheduler.running = {"req-media-cleanup": request}
        scheduler.block_aware_cache = MagicMock() if cache_mode == "paged" else None
        if scheduler.block_aware_cache is not None:
            scheduler.block_aware_cache._request_tables = {}
        scheduler.memory_aware_cache = MagicMock() if cache_mode == "memory" else None
        scheduler.prefix_cache = MagicMock() if cache_mode == "legacy" else None
        scheduler.disk_cache = MagicMock()
        scheduler._is_hybrid = False
        scheduler._kv_cache_bits = 0
        scheduler._truncate_hybrid_cache = MagicMock(return_value=["prompt-cache"])
        scheduler._validate_cache = MagicMock(return_value=True)
        scheduler._extract_cache_states = MagicMock(return_value=["state"])
        scheduler.batch_generator = None
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = MagicMock() if cache_mode == "paged" else None
        scheduler.requests = {"req-media-cleanup": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        scheduler._cleanup_finished({"req-media-cleanup"})

        if scheduler.block_aware_cache is not None:
            scheduler.block_aware_cache.store_cache.assert_not_called()
        if scheduler.memory_aware_cache is not None:
            scheduler.memory_aware_cache.store.assert_not_called()
        if scheduler.prefix_cache is not None:
            scheduler.prefix_cache.store_cache.assert_not_called()
        scheduler.disk_cache.store.assert_not_called()
        assert "req-media-cleanup" not in scheduler.running
        assert "req-media-cleanup" not in scheduler.requests
        scheduler._cleanup_detokenizer.assert_called_once_with("req-media-cleanup")

    def test_short_single_turn_outputs_still_store_prompt_cache(self):
        """The first turn of a future chat can be a one-token answer.

        MLLM used to skip prefix-cache storage for <=3 output tokens when the
        request had no history. That makes turn 2 semantically coherent because
        history is still in the prompt, but it forces a full re-prefill and
        reports cached_tokens=0 for Gemma/MiniMax/Nemotron-style MLLM routes.
        Cacheability is a prompt property; benchmarks that need isolation must
        use the explicit bypass flag.
        """

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=1,
            _has_history=False,
            _extracted_cache=lambda: ["live-cache"],
            _extracted_tokens=[10, 11, 12, 13, 14],
            _added_stop_tokens=set(),
        )

        scheduler.running = {"req-1": request}
        scheduler.block_aware_cache = MagicMock()
        scheduler.block_aware_cache._request_tables = {}
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.disk_cache = None
        scheduler._is_hybrid = False
        scheduler._kv_cache_bits = 0
        scheduler._truncate_hybrid_cache = MagicMock(return_value=["prompt-cache"])
        scheduler._validate_cache = MagicMock(return_value=True)
        scheduler._extract_cache_states = MagicMock(return_value=["state"])
        scheduler.batch_generator = None
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = MagicMock()
        scheduler.requests = {"req-1": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        scheduler._cleanup_finished({"req-1"})

        scheduler.block_aware_cache.store_cache.assert_called_once_with(
            "req-1",
            [10, 11, 12, 13],
            ["state"],
            cache_extra_keys=None,
        )

    def test_memory_aware_store_uses_n_minus_one_key_for_truncated_payload(self):
        """Memory-aware VLM cache keys must match the N-1 KV payload length."""

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=1,
            _has_history=False,
            _extracted_cache=lambda: ["live-cache"],
            _extracted_tokens=[10, 11, 12, 13, 14],
            _added_stop_tokens=set(),
        )

        scheduler.running = {"req-memory": request}
        scheduler.block_aware_cache = None
        scheduler.memory_aware_cache = MagicMock()
        scheduler.prefix_cache = None
        scheduler.disk_cache = MagicMock()
        scheduler._is_hybrid = False
        scheduler._kv_cache_bits = 0
        scheduler._truncate_hybrid_cache = MagicMock(return_value=["prompt-cache"])
        scheduler._validate_cache = MagicMock(return_value=True)
        scheduler.batch_generator = None
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = None
        scheduler.requests = {"req-memory": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        scheduler._cleanup_finished({"req-memory"})

        scheduler.disk_cache.store.assert_called_once_with(
            [10, 11, 12, 13, 14],
            ["prompt-cache"],
        )
        scheduler.memory_aware_cache.store.assert_called_once_with(
            [10, 11, 12, 13],
            ["prompt-cache"],
        )

    def test_legacy_prefix_store_uses_n_minus_one_key_for_truncated_payload(self):
        """Legacy VLM prefix keys must also be N-1 for exact-hit correctness."""

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=1,
            _has_history=False,
            _extracted_cache=lambda: ["live-cache"],
            _extracted_tokens=[10, 11, 12, 13, 14],
            _added_stop_tokens=set(),
        )

        scheduler.running = {"req-prefix": request}
        scheduler.block_aware_cache = None
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = MagicMock()
        scheduler.disk_cache = MagicMock()
        scheduler._is_hybrid = False
        scheduler._kv_cache_bits = 0
        scheduler._truncate_hybrid_cache = MagicMock(return_value=["prompt-cache"])
        scheduler._validate_cache = MagicMock(return_value=True)
        scheduler.batch_generator = None
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = None
        scheduler.requests = {"req-prefix": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        scheduler._cleanup_finished({"req-prefix"})

        scheduler.disk_cache.store.assert_called_once_with(
            [10, 11, 12, 13, 14],
            ["prompt-cache"],
        )
        scheduler.prefix_cache.store_cache.assert_called_once_with(
            [10, 11, 12, 13],
            ["prompt-cache"],
        )

    def test_uses_extracted_tokens_not_prompt_token_ids(self):
        """Ensure memory-aware and legacy paths use _extracted_tokens."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        # Should NOT reference prompt_token_ids (doesn't exist on MLLMRequest)
        lines = source.split('\n')
        for line in lines:
            if 'prompt_token_ids' in line and 'prompt_token_ids' not in line.lstrip().startswith('#'):
                # Only OK if it's in a comment
                stripped = line.lstrip()
                if not stripped.startswith('#'):
                    pytest.fail(f"Found prompt_token_ids reference: {line.strip()}")


# ============================================================
# Metal GC timer
# ============================================================


class TestMetalGCTimer:
    """Test Metal GC timer in scheduler."""

    def test_gc_timer_fields_exist(self):
        """Verify GC timer fields are on MLLMScheduler source."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "_last_metal_gc_time" in source
        assert "_metal_gc_interval" in source

    def test_step_has_periodic_gc(self):
        """Verify step() includes periodic Metal GC."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.step)
        assert "clear_mlx_memory_cache" in source
        assert "_metal_gc_interval" in source

    def test_cleanup_has_idle_gc(self):
        """Verify _cleanup_finished clears Metal cache when idle."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "clear_mlx_memory_cache" in source


# ============================================================
# get_stats cache reporting
# ============================================================


class TestGetStatsCacheReporting:
    """Test that get_stats reports all cache modes."""

    def test_get_stats_reports_cache_modes(self):
        """Verify get_stats source includes all cache mode reporting."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.get_stats)
        assert "paged_cache" in source
        assert "memory_aware_cache" in source
        assert "prefix_cache" in source
        assert "disk_cache" in source


# ============================================================
# Hybrid model detection
# ============================================================


class TestHybridModelDetection:
    """Test _is_hybrid_model static method."""

    def test_non_hybrid_returns_false(self):
        """Pure KVCache model should not be detected as hybrid."""
        from mlx_lm.models.cache import KVCache

        model = MagicMock()
        model.make_cache.return_value = [KVCache(), KVCache(), KVCache()]

        assert MLLMScheduler._is_hybrid_model(model) is False

    def test_no_make_cache_returns_false(self):
        model = MagicMock(spec=[])  # No make_cache attribute
        assert MLLMScheduler._is_hybrid_model(model) is False


# ============================================================
# Integration: cache init chain
# ============================================================


class TestCacheInitChain:
    """Test the 3-tier cache init chain in MLLMScheduler.__init__."""

    def test_init_source_has_three_tiers(self):
        """Verify init chain covers all 3 tiers."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)

        # Paged tier
        assert "PagedCacheManager" in source
        assert "BlockAwarePrefixCache" in source

        # Memory-aware tier
        assert "MemoryAwarePrefixCache" in source
        assert "MemoryCacheConfig" in source

        # Legacy tier
        assert "PrefixCacheManager" in source

    def test_init_source_has_disk_cache_l2(self):
        """Verify disk cache L2 initialization is present."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "DiskCacheManager" in source

    def test_init_source_has_block_disk_store(self):
        """Verify block disk store wiring is present."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "BlockDiskStore" in source


# ============================================================
# Process prompts cache fetch paths
# ============================================================


class TestProcessPromptsCacheFetch:
    """Test that _process_prompts handles all cache fetch paths."""

    def test_has_paged_fetch(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "block_aware_cache" in source
        assert "fetch_cache" in source

    def test_has_memory_aware_fetch(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "memory_aware_cache" in source

    def test_has_legacy_fetch(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "prefix_cache" in source

    def test_has_disk_cache_l2_fallback(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "disk_cache" in source

    def test_has_hybrid_ssm_state_fetch(self):
        """Verify hybrid models check companion SSM state cache."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "_ssm_state_cache" in source
        assert "HYBRID cache HIT" in source

    def test_has_ssm_state_capture(self):
        """Verify SSM state is captured at prompt boundary during prefill."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "_ssm_state_cache.store" in source
        assert "Captured SSM state" in source

    def test_hybrid_ssm_capture_stores_block_aligned_checkpoints(self):
        """Hybrid paged cache hits are block-aligned, so the companion SSM
        cache must store a matching block-aligned checkpoint in addition to the
        full clean prompt boundary.
        """
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        run_source = inspect.getsource(MLLMBatchGenerator._run_vision_encoding_inner)
        process_source = inspect.getsource(MLLMBatchGenerator._process_prompts)

        assert "_ssm_capture_boundaries_for" in run_source
        assert "_ssm_block_aligned_boundary" in inspect.getsource(
            MLLMBatchGenerator._ssm_capture_boundaries_for
        )
        assert "_inline_ssm_checkpoints" in process_source
        assert "for _inline_boundary, _inline_tokens, _inline_layers in" in process_source

    def test_hybrid_ssm_kv_only_miss_marks_checkpoint_for_next_hit(self):
        """A KV-only hybrid hit should teach the next prefill its missing boundary.

        Long Responses/tool chains often share only a small paged prefix
        (64/128/320 tokens). If the companion SSM cache only stores near the
        full prompt boundary, every later KV hit remains unusable and TTFT
        regresses to full prefill forever.
        """
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        run_source = inspect.getsource(MLLMBatchGenerator._run_vision_encoding_inner)
        process_source = inspect.getsource(MLLMBatchGenerator._process_prompts)

        assert "_mark_required_ssm_checkpoint" in process_source
        assert "_ssm_capture_boundaries_for" in run_source
        assert "_ssm_required_checkpoint_tokens" in inspect.getsource(
            MLLMBatchGenerator._ssm_capture_boundaries_for
        )

    def test_hybrid_ssm_required_checkpoint_boundary_is_captured(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._is_hybrid = True
        generator.block_aware_cache = SimpleNamespace(block_size=64)
        request = SimpleNamespace(_ssm_required_checkpoint_tokens=128)

        boundaries = generator._ssm_capture_boundaries_for(
            request,
            seq_len=400,
            has_images=False,
            clean_boundary=300,
        )

        assert boundaries == [128, 256, 300]

    def test_hybrid_ssm_required_checkpoint_boundary_converts_from_absolute_prefix(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._is_hybrid = True
        generator.block_aware_cache = SimpleNamespace(block_size=64)
        request = SimpleNamespace(
            _cached_tokens=64,
            _ssm_required_checkpoint_tokens=320,
        )

        boundaries = generator._ssm_capture_boundaries_for(
            request,
            seq_len=400,
            has_images=False,
            clean_boundary=300,
        )

        assert boundaries == [256, 300]

    def test_hybrid_ssm_required_checkpoint_ignored_for_media_or_invalid_hits(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._is_hybrid = True
        generator.block_aware_cache = SimpleNamespace(block_size=64)
        request = SimpleNamespace(_ssm_required_checkpoint_tokens=400)

        assert generator._ssm_capture_boundaries_for(
            request,
            seq_len=400,
            has_images=False,
            clean_boundary=300,
        ) == [256, 300]
        assert generator._ssm_capture_boundaries_for(
            request,
            seq_len=400,
            has_images=True,
            clean_boundary=300,
        ) == []

    def test_hybrid_ssm_required_checkpoint_respects_inline_capture_killswitch(self, monkeypatch):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._is_hybrid = True
        generator.block_aware_cache = SimpleNamespace(block_size=64)
        request = SimpleNamespace(_ssm_required_checkpoint_tokens=128)

        monkeypatch.setenv("VMLX_DISABLE_SSM_INLINE_CAPTURE", "1")

        assert generator._ssm_capture_boundaries_for(
            request,
            seq_len=400,
            has_images=False,
            clean_boundary=300,
        ) == []

    def test_mark_required_ssm_checkpoint_records_absolute_boundary(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        request = SimpleNamespace(_cached_tokens=256)

        generator._mark_required_ssm_checkpoint(request, 128)

        assert request._ssm_required_checkpoint_tokens == 128
        assert request._cached_tokens == 0

    def test_mark_required_ssm_checkpoint_can_preserve_partial_resume(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        request = SimpleNamespace(_cached_tokens=64)

        generator._mark_required_ssm_checkpoint(
            request,
            320,
            reset_cached_tokens=False,
        )

        assert request._ssm_required_checkpoint_tokens == 320
        assert request._cached_tokens == 64

    def test_hybrid_ssm_inline_capture_materializes_snapshot_before_phase_b(self):
        """Inline SSM checkpoints must not stay as lazy views of live cache.

        Qwen3.6 hybrid cache hits reused an inline checkpoint captured before
        the generation-prompt suffix, but the snapshot was not forced before
        phase-B prefill continued mutating the live SSM cache. The next turn
        then had cached_tokens>0 but answered from the previous instruction.
        """
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._maybe_capture_clean_ssm_boundary)

        assert "_inline_materialize" in source
        assert "mx.eval(*_inline_materialize)" in source


# ============================================================
# Metal cache limit
# ============================================================


class TestMetalCacheLimit:
    """Test Metal cache limit tuning in batch generator."""

    def test_init_has_cache_limit(self):
        """Verify batch generator sets Metal cache limit."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator.__init__)
        assert "set_cache_limit" in source
        assert "_old_cache_limit" in source

    def test_close_restores_cache_limit(self):
        """Verify close() restores old cache limit."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator.close)
        assert "_old_cache_limit" in source
        assert "set_cache_limit" in source

    def test_tight_memory_mllm_prefill_drains_allocator(self):
        """Tight-memory MLLM rows must drain allocator state between prefills."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        init_source = inspect.getsource(MLLMBatchGenerator.__init__)
        process_source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        next_source = inspect.getsource(MLLMBatchGenerator._next)
        drain_source = inspect.getsource(
            MLLMBatchGenerator._drain_tight_memory_allocator
        )

        assert "_tight_memory_prefill_drain = safety_limit < base_limit" in init_source
        assert '_drain_tight_memory_allocator("before_prefill")' in process_source
        assert '_drain_tight_memory_allocator("after_batch_finish")' in next_source
        assert "clear_mlx_memory_cache" in drain_source
        assert "mx.synchronize" in drain_source

    def test_mllm_store_detects_rotating_cache_layout_from_live_cache(self):
        """MiMo wrappers can hide cache_subtype; live RotatingKVCache must win."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)

        assert "RotatingKVCache" in source
        assert "Detected mixed-SWA VLM cache layout" in source
        assert "_uses_mixed_attention_cache = True" in source
        assert "_prefill_for_clean_path_dependent_cache" in source
