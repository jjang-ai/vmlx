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

import json
import re

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig, MLLMScheduler
from vmlx_engine.mllm_batch_generator import (
    MLLMBatchGenerator,
    _disk_prefix_hit_tail_and_cached_tokens as _mllm_disk_prefix_hit_tail_and_cached_tokens,
    _fetch_request_ssm_longest_prefix,
    _paged_reconstruct_disk_source,
    _prefix_hit_tail_and_cached_tokens as _mllm_prefix_hit_tail_and_cached_tokens,
    _record_request_ssm_exact_lookup,
    _trace_mimo_v2_generated_token,
)
from vmlx_engine.utils.ssm_companion_cache import make_ssm_prefix_lookup


def test_mllm_scheduler_does_not_shadow_hashlib_in_init():
    """Regression: paged block-disk init uses hashlib before non-paged L2 setup."""
    import inspect

    source = inspect.getsource(MLLMScheduler.__init__)

    assert "import hashlib" not in source
    assert "hashlib.sha256" in source


def test_mllm_stop_releases_owned_block_disk_store_lease():
    import asyncio

    disk_store = MagicMock()
    manager = SimpleNamespace(_disk_store=disk_store)
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._processing_task = None
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler._terminal_cleanup_complete.set()
    scheduler._running = False
    scheduler.batch_generator = None
    scheduler.paged_cache_manager = manager
    scheduler._ssm_companion_disk_store = MagicMock()
    scheduler._block_disk_l2_enabled = True

    asyncio.run(scheduler.stop())

    disk_store.shutdown.assert_called_once_with()
    assert manager._disk_store is None
    assert scheduler._ssm_companion_disk_store is None
    assert scheduler._block_disk_l2_enabled is False


def test_mllm_worker_reconstruct_promotes_lazy_l2_source_to_request_detail():
    """An indexed paged hit whose payload refaults from L2 is a disk hit."""
    cache = SimpleNamespace(_last_reconstruct_disk_blocks=3)

    assert _paged_reconstruct_disk_source(
        fetch_disk_hit=False,
        block_aware_cache=cache,
        reconstructed=["kv"],
    ) == (True, 3)
    assert _paged_reconstruct_disk_source(
        fetch_disk_hit=True,
        block_aware_cache=SimpleNamespace(_last_reconstruct_disk_blocks=0),
        reconstructed=["kv"],
    ) == (True, 0)
    assert _paged_reconstruct_disk_source(
        fetch_disk_hit=False,
        block_aware_cache=cache,
        reconstructed=None,
    ) == (False, 0)


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
    # 2026-08-16 (ledger row 157): the tight step is PROJECTED from the
    # attention-score budget (max_prefill_chunk_tokens) rather than collapsed
    # to a flat 64 — measured 2.93x on dots3, and probing for it is itself
    # expensive because every chunk re-streams the expert set. The env
    # override must still be honoured exactly when an operator sets it.
    assert "max_prefill_chunk_tokens(" in source
    assert "_tight_env_step" in source
    # 2026-08-16 (ledger row 157): the tight step is now the conservative
    # FIRST chunk and the measured fitter may grow back toward the configured
    # step (a smaller chunk does not reduce weight streaming, it multiplies
    # it — measured 2.77x on dots3). The protection this test exists for is
    # unchanged: the tight path must still CHUNK rather than fall through to
    # the one-shot LM call, and the growth ceiling must derive from the
    # configured step with the tight step seeding it.
    assert "chunk_size = min(_chunk_ceiling" in source
    assert "_chunk_ceiling = (" in source
    assert "_adaptive_chunk_cap = _tight_text_prefill_step_size" in source
    assert "VMLX_TIGHT_PREFILL_ADAPTIVE_GROWTH" in source


def test_mimo_v2_generator_detects_inner_language_model_type_for_processors():
    """JANG VLM wrappers can have blank outer model_type; inner runtime wins."""
    import math

    import mlx.core as mx

    class Tokenizer:
        eos_token_id = 9

        def encode(self, text, add_special_tokens=False):
            if text == "<think>":
                return [1]
            if text == "</think>":
                return [2]
            if text == "<|im_end|>":
                return [9]
            return [3]

    model = SimpleNamespace(
        config=SimpleNamespace(model_type=""),
        language_model=SimpleNamespace(config=SimpleNamespace(model_type="mimo_v2")),
    )
    generator = MLLMBatchGenerator(model=model, processor=SimpleNamespace(tokenizer=Tokenizer()))

    assert generator._model_type == "mimo_v2"

    request = SimpleNamespace(enable_thinking=False, output_tokens=[])
    processors = generator._mimo_v2_thinking_off_logits_processors(request)
    logits = mx.zeros((1, 12))

    no_think = processors[0](mx.array([42]), logits)
    assert math.isinf(float(no_think[0, 1])) and float(no_think[0, 1]) < 0
    assert math.isinf(float(no_think[0, 2])) and float(no_think[0, 2]) < 0

    first_token = processors[1](mx.array([42]), logits)
    assert math.isinf(float(first_token[0, 9])) and float(first_token[0, 9]) < 0

    request.output_tokens = [7]
    later_token = processors[1](mx.array([42, 7]), logits)
    assert float(later_token[0, 9]) == 0.0


def test_mimo_v2_token_trace_is_opt_in_and_reports_decoded_stop(caplog, monkeypatch):
    """MiMo token tracing must be diagnostic-only and include stop evidence."""
    import logging

    import mlx.core as mx

    class Tokenizer:
        def decode(self, ids, skip_special_tokens=False):
            assert skip_special_tokens is False
            return {
                9: "<|im_end|>",
                3: "3",
                1: "1",
            }.get(ids[0], str(ids[0]))

    generator = SimpleNamespace(
        _model_type="mimo_v2",
        processor=SimpleNamespace(tokenizer=Tokenizer()),
        stop_tokens={9},
        _mimo_v2_thinking_off_token_ids={
            "think_ids": {1, 2},
            "eos_ids": {9},
        },
    )
    request = SimpleNamespace(
        request_id="json-stop",
        enable_thinking=False,
        output_tokens=[],
    )

    caplog.set_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator")
    _trace_mimo_v2_generated_token(
        generator,
        request,
        9,
        phase="prefill_sample",
        finish_reason="stop",
    )
    assert "MiMo-V2 token trace" not in caplog.text

    monkeypatch.setenv("VMLINUX_MIMO_V2_TOKEN_TRACE", "1")
    monkeypatch.setenv("VMLINUX_MIMO_V2_TOKEN_TRACE_TOPK", "2")
    _trace_mimo_v2_generated_token(
        generator,
        request,
        9,
        phase="prefill_sample",
        finish_reason="stop",
        logprobs=mx.array([0.0, 0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]),
    )

    assert "MiMo-V2 token trace" in caplog.text
    assert "request=json-stop" in caplog.text
    assert "decoded='<|im_end|>'" in caplog.text
    assert "is_stop=True" in caplog.text
    assert "eos_ids=[9]" in caplog.text
    assert "top_tokens=[{'id': 9, 'decoded': '<|im_end|>'" in caplog.text
    assert "{'id': 1, 'decoded': '1'" in caplog.text


def test_batched_mllm_generate_forwards_enable_thinking_to_scheduler():
    """Non-streaming batched MLLM requests must preserve thinking-off state."""
    import asyncio

    from vmlx_engine.engine.batched import BatchedEngine

    captured = {}

    class Scheduler:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text="ACK",
                output_token_ids=[1],
                prompt_tokens=1,
                completion_tokens=1,
                cached_tokens=0,
                cache_detail="",
                finish_reason="stop",
            )

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = True
    engine._mllm_scheduler = Scheduler()

    output = asyncio.run(
        engine.generate(
            prompt="rendered",
            max_tokens=8,
            temperature=0,
            top_p=1,
            enable_thinking=False,
        )
    )

    assert output.text == "ACK"
    assert captured["enable_thinking"] is False


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
        request_id="audio-prefill-wrapper-test",
        input_ids=mx.array([1, 151669, 2]),
        pixel_values=None,
        attention_mask=None,
        image_grid_thw=None,
        video_pixel_values=None,
        video_grid_thw=None,
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
    assert 'request.extra_kwargs.pop("input_features", None)' in source
    assert 'request.extra_kwargs.pop("input_features_mask", None)' in source
    assert "request.audio_codes = _ensure_mx_array(" in source
    assert "request.audio_embeds = _ensure_mx_array(" in source
    assert "request.audio_features = _ensure_mx_array(" in source
    assert "request.audio_features_mask = _ensure_mx_array(" in source


def test_mllm_processor_direct_loads_audio_paths_for_non_mimo_processor(tmp_path):
    """Gemma-style processors receive waveform arrays, not temp audio paths."""
    import math
    import struct
    import wave

    import numpy as np

    from vmlx_engine.mllm_batch_generator import _call_processor_direct

    wav_path = tmp_path / "audio-present.wav"
    sample_rate = 16000
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for idx in range(sample_rate // 10):
            value = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * idx / sample_rate))
            frames.extend(struct.pack("<h", value))
        wav.writeframes(bytes(frames))

    captured = {}

    class Processor:
        sampling_rate = sample_rate

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
        audio=[str(wav_path)],
        add_special_tokens=False,
    )

    assert len(captured["audio"]) == 1
    assert isinstance(captured["audio"][0], np.ndarray)
    assert captured["audio"][0].dtype == np.float32
    assert captured["audio"][0].ndim == 1
    assert captured["audio"][0].size > 0
    assert captured["audios"] == captured["audio"]
    assert result["audio_codes"] == [[4, 5]]


def test_mllm_gemma_input_features_forward_as_raw_audio_features():
    """Gemma processors emit input_features that must reach Gemma's embedder."""
    from types import SimpleNamespace

    import mlx.core as mx

    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    class CapturingModel:
        def __init__(self):
            self.kwargs = None

        def __call__(self, input_ids, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(logits=mx.zeros((1, input_ids.shape[-1], 4)))

    model = CapturingModel()
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.model = model
    generator.language_model = None
    generator._is_hybrid = False
    generator._model_type = "gemma4"
    generator.prefill_step_size = 128

    request = SimpleNamespace(
        request_id="gemma-audio-input-features-test",
        input_ids=mx.array([1, 258881, 2]),
        pixel_values=None,
        attention_mask=None,
        image_grid_thw=None,
        video_pixel_values=None,
        video_grid_thw=None,
        audio_codes=None,
        audio_embeds=None,
        audio_features=mx.ones((1, 2, 640)),
        audio_features_mask=mx.array([[True, False]]),
        audio_features_are_raw_input_features=True,
        extra_kwargs={},
        vision_encoded=False,
    )

    logits = generator._run_vision_encoding_inner(request, cache=[])

    assert logits.shape == (1, 3, 4)
    assert "input_features" in model.kwargs
    assert "input_features_mask" in model.kwargs
    assert "audio_embeds" not in model.kwargs
    assert model.kwargs["input_features"].shape == (1, 2, 640)
    assert model.kwargs["input_features_mask"].tolist() == [[True, False]]
    assert request.vision_encoded is True


def test_gemma4_wrapper_does_not_forward_processor_padding_mask_to_language_model():
    """Continuous batching must match mlx-vlm's Gemma media mask contract."""
    import mlx.core as mx

    from vmlx_engine.models.gemma4_unified_register import (
        register_gemma4_unified_runtime,
    )

    register_gemma4_unified_runtime()
    from mlx_vlm.models.gemma4_unified.gemma4_unified import (
        _language_model_mask,
    )

    processor_padding_mask = mx.ones((1, 165), dtype=mx.bool_)
    assert _language_model_mask(processor_padding_mask) is None

    explicit_attention_mask = mx.zeros((1, 1, 4, 4))
    assert _language_model_mask(explicit_attention_mask) is explicit_attention_mask


def test_mllm_processor_direct_omits_invalid_audios_alias_for_mimo_v2_processor():
    """MiMo-V2 processors warn and ignore unknown `audios`; use `audio` only."""
    from vmlx_engine.mllm_batch_generator import _call_processor_direct

    captured = {}

    class Processor:
        __module__ = "mlx_vlm.models.mimo_v2.processing_mimo_v2"

        def __call__(self, **kwargs):
            captured.update(kwargs)
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
    assert "audios" not in captured
    assert result["audio_codes"] == [[4, 5]]


def test_mllm_raw_audio_without_processor_payload_fails_loudly(tmp_path):
    """Audio requests must not silently continue if no audio token is present."""
    from types import SimpleNamespace

    import pytest

    from vmlx_engine.errors import UnsupportedMediaModalityError
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

    wav = tmp_path / "blue.wav"
    wav.write_bytes(b"RIFF----WAVEfmt ")

    class Processor:
        def __call__(self, **kwargs):
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="mimo_v2",
            processor_config={"audio_token_id": 151669},
        ),
        language_model=SimpleNamespace(),
    )
    generator = MLLMBatchGenerator(model=model, processor=Processor())
    request = MLLMBatchRequest(
        uid=1,
        request_id="audio-no-payload",
        prompt="transcribe audio",
        audio=[str(wav)],
    )

    with pytest.raises(UnsupportedMediaModalityError) as exc:
        generator._preprocess_request(request)

    assert exc.value.modality == "audio"
    assert exc.value.family == "mimo_v2"
    assert "contains no audio token" in exc.value.detail


def test_mllm_mimo_raw_audio_bridge_populates_audio_codes(tmp_path, monkeypatch):
    """MiMo raw audio should build audio_codes before the generic fail-loud guard."""
    from types import SimpleNamespace

    import mlx.core as mx

    import vmlx_engine.mllm_batch_generator as module
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

    wav = tmp_path / "blue.wav"
    wav.write_bytes(b"RIFF----WAVEfmt ")
    captured = {}

    class Processor:
        name_or_path = str(tmp_path)

        def __call__(self, **kwargs):
            return {
                "input_ids": [[1, 151669, 2]],
                "attention_mask": [[1, 1, 1]],
            }

    def fake_bridge(**kwargs):
        captured.update(kwargs)
        return mx.ones((10, 20), dtype=mx.int32)

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="mimo_v2",
            processor_config={"audio_token_id": 151669},
            audio_config={"group_size": 4},
        ),
        language_model=SimpleNamespace(),
    )
    monkeypatch.setattr(module, "_build_mimo_audio_codes_from_paths", fake_bridge)

    generator = MLLMBatchGenerator(model=model, processor=Processor())
    request = MLLMBatchRequest(
        uid=1,
        request_id="audio-bridge",
        prompt="transcribe audio <|audio_pad|>",
        audio=[str(wav)],
    )

    generator._preprocess_request(request)

    assert captured["audio_paths"] == [str(wav)]
    assert captured["model"] is model
    assert request.audio_codes.shape == (10, 20)
    assert request.input_ids.tolist() == [[1, 151669, 151669, 151669, 2]]
    assert request.attention_mask.tolist() == [[1, 1, 1, 1, 1]]


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


def test_mllm_disk_prefix_hit_refeeds_the_last_matched_token():
    """The MLLM disk L2 payload covers matched[:-1], so matched[-1] must lead.

    This test previously asserted the OPPOSITE (cached == len(matched), tail
    starting after matched) on the premise that the MLLM disk store writes a
    payload as long as its key. Both MLLM store branches disprove that: the
    plain VLM path truncates through _truncate_hybrid_cache(prompt_len - 1),
    and the mixed-SWA / ZAYA clean path re-prefills token_list[:prompt_len-1] --
    and both write under the FULL N-token key. Dropping matched[-1] meant its
    KV was never computed, so a warm disk-prefix turn answered differently from
    the same turn cold.
    """

    remaining, cached_tokens = _mllm_disk_prefix_hit_tail_and_cached_tokens(
        token_list=[10, 11, 12, 13, 14, 15],
        matched_tokens=[10, 11, 12, 13],
        gen_prompt_suffix=[90, 91],
    )

    assert remaining == [13, 14, 15, 90, 91]
    assert cached_tokens == 3


def test_mllm_disk_exact_hit_with_generation_suffix_refeeds_boundary():
    remaining, cached_tokens = _mllm_disk_prefix_hit_tail_and_cached_tokens(
        token_list=[10, 11, 12, 13],
        matched_tokens=[10, 11, 12, 13],
        gen_prompt_suffix=[90, 91],
    )

    assert remaining == [13, 90, 91]
    assert cached_tokens == 3


def test_mllm_and_text_disk_helpers_agree_at_a_non_block_aligned_length():
    """The two lanes must not be able to disagree again.

    The original defect survived because each scheduler carried its own copy
    and only one got 12ee1c8ee. Lengths here are deliberately indivisible by
    64 or 256 -- the aligned lengths every prior test used are exactly where
    an off-by-one is invisible.
    """
    from vmlx_engine.scheduler import Scheduler

    fetch = list(range(1361))
    matched = list(range(1003))
    suffix = [90, 91]

    text_tail, text_cached = Scheduler._disk_prefix_hit_tail_and_cached_tokens(
        fetch_tokens=fetch, matched_tokens=matched, gen_prompt_suffix=suffix
    )
    mllm_tail, mllm_cached = _mllm_disk_prefix_hit_tail_and_cached_tokens(
        token_list=fetch, matched_tokens=matched, gen_prompt_suffix=suffix
    )

    assert (mllm_tail, mllm_cached) == (text_tail, text_cached)
    assert mllm_cached == 1002
    assert mllm_tail[0] == matched[-1] == 1002
    assert mllm_tail[-2:] == suffix


def test_mllm_disk_boundary_swap_refeeds_the_swapped_token():
    """A boundary-swap match hands back the CURRENT prompt's final token.

    disk_cache.fetch_longest_prefix can match a stored key whose last token
    differs from the query's (a thinking sentinel toggling between turns) and
    returns the query's token in matched[-1]. Dropping it silently served the
    previous turn's thinking mode.
    """
    remaining, cached_tokens = _mllm_disk_prefix_hit_tail_and_cached_tokens(
        token_list=[10, 11, 12, 777, 14],
        matched_tokens=[10, 11, 12, 777],
        gen_prompt_suffix=[],
    )

    assert remaining[0] == 777, "the swapped boundary token was dropped"
    assert remaining == [777, 14]
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
        # 0.15, matching the CLI --cache-memory-percent default and the app's
        # cacheMemoryPercent: 15. This class is named ConfigParity, so assert
        # the parity rather than a frozen literal — the MLLM default sat at
        # 0.20 while both other surfaces said 15, which is the divergence this
        # test exists to catch.
        _cli = Path(__file__).resolve().parents[1] / "vmlx_engine" / "cli.py"
        _src = _cli.read_text(encoding="utf-8")
        _i = _src.index('"--cache-memory-percent"')
        _m = re.search(r"default=([0-9.]+)", _src[_i : _i + 400])
        assert _m, "could not read the CLI --cache-memory-percent default"
        assert config.cache_memory_percent == float(_m.group(1))
        assert config.cache_ttl_minutes == 0

    def test_legacy_prefix_cache_field(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "prefix_cache_size")
        assert config.prefix_cache_size == 100

    def test_ssm_companion_budget_defaults(self):
        config = MLLMSchedulerConfig()
        assert config.ssm_state_cache_size == 0
        assert config.ssm_state_cache_max_mb == 0

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
        assert config.use_paged_cache is False
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
            "candidate_count": 1,
            "candidate_lengths_truncated": False,
            "attempted_candidate_lengths": [5, 4],
            "attempted_candidate_count": 2,
            "attempted_candidate_lengths_truncated": False,
            "matched": True,
            "checkpoint_tokens": 4,
            "is_complete": True,
            "source": "l1_or_l2",
            "reason": "matched",
            "store_size": 1,
        }

        miss = cache.fetch_longest_prefix([9, 2, 3, 4], 4)
        assert miss is None
        assert cache.last_prefix_lookup == {
            "max_len": 4,
            "candidate_lengths": [4],
            "candidate_count": 1,
            "candidate_lengths_truncated": False,
            "attempted_candidate_lengths": [4],
            "attempted_candidate_count": 1,
            "attempted_candidate_lengths_truncated": False,
            "matched": False,
            "checkpoint_tokens": 0,
            "is_complete": False,
            "source": "none",
            "reason": "prefix_hash_mismatch",
            "store_size": 1,
        }


def test_request_ssm_lookup_overwrites_success_miss_disabled_and_exception():
    request = SimpleNamespace(
        request_id="req-current",
        _cache_execution={
            "ssm_prefix_lookup": {
                "request_id": "req-stale",
                "debug_path": "/private/stale",
            }
        },
    )

    class _Cache:
        size = 2
        mode = "hit"
        last_prefix_lookup = None

        def fetch_longest_prefix(
            self,
            token_ids,
            max_len,
            cache_extra_keys=None,
        ):
            if self.mode == "exception":
                raise RuntimeError("/private/cache/secret")
            matched = self.mode == "hit"
            self.last_prefix_lookup = make_ssm_prefix_lookup(
                max_len=max_len,
                candidate_lengths=[4],
                attempted_candidate_lengths=[max_len, 4],
                matched=matched,
                checkpoint_tokens=4 if matched else 0,
                is_complete=matched,
                source="l1_or_l2" if matched else "none",
                reason="matched" if matched else "candidate_fetch_miss",
                store_size=self.size,
            )
            return (4, ["state"], True) if matched else None

    cache = _Cache()
    result, lookup = _fetch_request_ssm_longest_prefix(
        request,
        cache,
        enabled=True,
        token_ids=[1, 2, 3, 4, 5],
        max_len=5,
    )
    assert result == (4, ["state"], True)
    assert lookup["matched"] is True
    assert request._cache_execution["ssm_prefix_lookup"] == lookup

    cache.mode = "miss"
    _, lookup = _fetch_request_ssm_longest_prefix(
        request,
        cache,
        enabled=True,
        token_ids=[9, 2, 3, 4, 5],
        max_len=5,
    )
    assert lookup["matched"] is False
    assert lookup["reason"] == "candidate_fetch_miss"

    _, lookup = _fetch_request_ssm_longest_prefix(
        request,
        cache,
        enabled=False,
        token_ids=[1, 2, 3, 4, 5],
        max_len=5,
    )
    assert lookup["reason"] == "ssm_prefix_resume_disabled"
    assert lookup["attempted_candidate_lengths"] == []

    cache.mode = "exception"
    _, lookup = _fetch_request_ssm_longest_prefix(
        request,
        cache,
        enabled=True,
        token_ids=[1, 2, 3, 4, 5],
        max_len=5,
    )
    assert lookup == {
        "max_len": 5,
        "candidate_lengths": [],
        "candidate_count": 0,
        "candidate_lengths_truncated": False,
        "attempted_candidate_lengths": [5],
        "attempted_candidate_count": 1,
        "attempted_candidate_lengths_truncated": False,
        "matched": False,
        "checkpoint_tokens": 0,
        "is_complete": False,
        "source": "none",
        "reason": "lookup_exception",
        "store_size": 2,
        "request_id": "req-current",
    }
    assert "/private/" not in repr(request._cache_execution)
    assert "req-stale" not in repr(request._cache_execution)


def test_request_ssm_lookup_guards_telemetry_properties_and_preserves_hit():
    class _RaisingTelemetryCache:
        mode = "last_lookup"

        @property
        def size(self):
            raise RuntimeError("/private/size-secret")

        @property
        def last_prefix_lookup(self):
            raise RuntimeError("/private/lookup-secret")

        def fetch_longest_prefix(self, *args, **kwargs):
            return (4, ["state"], True)

    request = SimpleNamespace(
        request_id="proof-request",
        _cache_execution={"ssm_prefix_lookup": {"request_id": "stale"}},
    )
    cache = _RaisingTelemetryCache()
    result, lookup = _fetch_request_ssm_longest_prefix(
        request,
        cache,
        enabled=True,
        token_ids=[1, 2, 3, 4, 5],
        max_len=5,
    )
    assert result == (4, ["state"], True)
    assert lookup["reason"] == "malformed_lookup"
    assert lookup["store_size"] == 0
    assert "/private/" not in repr(request._cache_execution)
    assert "stale" not in repr(request._cache_execution)

    class _RaisingFetchProperty:
        size = 3

        @property
        def fetch_longest_prefix(self):
            raise RuntimeError("/private/fetch-secret")

    result, lookup = _fetch_request_ssm_longest_prefix(
        request,
        _RaisingFetchProperty(),
        enabled=True,
        token_ids=[1, 2, 3, 4, 5],
        max_len=5,
    )
    assert result is None
    assert lookup["reason"] == "lookup_exception"
    assert lookup["store_size"] == 3
    assert lookup["attempted_candidate_lengths"] == []
    assert lookup["attempted_candidate_count"] == 0
    assert "/private/" not in repr(request._cache_execution)


def test_request_ssm_lookup_rejects_stale_well_formed_prior_result():
    request = SimpleNamespace(
        request_id="req-current",
        _cache_execution={"request_id": "req-current"},
    )

    class _StaleLookupCache:
        size = 1
        last_prefix_lookup = make_ssm_prefix_lookup(
            max_len=4,
            candidate_lengths=[4],
            attempted_candidate_lengths=[4],
            matched=True,
            checkpoint_tokens=4,
            is_complete=True,
            source="exact_boundary_l1_or_l2",
            reason="matched",
        )

        def fetch_longest_prefix(self, *args, **kwargs):
            return None

    result, lookup = _fetch_request_ssm_longest_prefix(
        request,
        _StaleLookupCache(),
        enabled=True,
        token_ids=[1, 2, 3, 4, 5],
        max_len=5,
    )
    assert result is None
    assert lookup["max_len"] == 5
    assert lookup["matched"] is False
    assert lookup["reason"] == "malformed_lookup"
    assert lookup["request_id"] == "req-current"
    assert request._cache_execution["ssm_prefix_lookup"] == lookup


def test_production_exact_ssm_lookup_attaches_actual_hit_and_miss():
    import inspect

    production_source = inspect.getsource(MLLMBatchGenerator._process_prompts)
    exact_fetch = production_source.index(
        "_entry = self._ssm_state_cache.fetch("
    )
    exact_attestation = production_source.index(
        "_record_request_ssm_exact_lookup("
    )
    exact_branch = production_source.index("if _entry is None:", exact_fetch)
    assert exact_fetch < exact_attestation < exact_branch
    assert "exact_boundary_already_missed=True" in production_source

    request = SimpleNamespace(
        request_id="req-exact",
        _cache_execution={"request_id": "req-exact"},
    )
    cache = SimpleNamespace(size=2)

    hit = _record_request_ssm_exact_lookup(
        request,
        cache,
        max_len=64,
        matched=True,
        is_complete=True,
    )
    assert hit["max_len"] == 64
    assert hit["candidate_lengths"] == [64]
    assert hit["attempted_candidate_lengths"] == [64]
    assert hit["matched"] is True
    assert hit["checkpoint_tokens"] == 64
    assert hit["source"] == "exact_boundary_l1_or_l2"

    miss = _record_request_ssm_exact_lookup(
        request,
        cache,
        max_len=64,
        matched=False,
        is_complete=False,
    )
    assert miss["max_len"] == 64
    assert miss["candidate_lengths"] == []
    assert miss["attempted_candidate_lengths"] == [64]
    assert miss["matched"] is False
    assert miss["checkpoint_tokens"] == 0
    assert miss["reason"] == "candidate_fetch_miss"
    assert request._cache_execution["ssm_prefix_lookup"] == miss


def test_request_ssm_lookup_hashes_malicious_request_identity_everywhere():
    malicious_prefix = "../private/cache/\n"
    malicious = malicious_prefix + ("x" * (319 - len(malicious_prefix)))
    assert len(malicious) == 319
    request = SimpleNamespace(
        request_id=malicious,
        _cache_execution={"request_id": malicious},
    )

    class _Cache:
        size = 0
        last_prefix_lookup = make_ssm_prefix_lookup(
            max_len=5,
            attempted_candidate_lengths=[5],
            reason="candidate_fetch_miss",
        )

        def fetch_longest_prefix(self, *args, **kwargs):
            return None

    _, lookup = _fetch_request_ssm_longest_prefix(
        request,
        _Cache(),
        enabled=True,
        token_ids=[1, 2, 3, 4, 5],
        max_len=5,
    )
    payload = json.dumps(request._cache_execution)
    assert malicious not in payload
    assert "../private" not in payload
    assert "\n" not in payload
    assert lookup["request_id"].startswith("opaque-")
    assert request._cache_execution["request_id"] == lookup["request_id"]


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

    def test_custom_root_runtime_wiring_shares_one_aggregate_budget(
        self,
        tmp_path,
    ):
        """VLM custom roots must budget hashed KV and typed SSM as one tree."""
        import asyncio

        class KVCache:
            pass

        class MambaCache:
            pass

        class LanguageModel:
            config = SimpleNamespace(model_type="test-hybrid-vlm")

            def make_cache(self):
                return [KVCache(), MambaCache()]

        root = tmp_path / "custom-block-root"
        model = SimpleNamespace(
            language_model=LanguageModel(),
            config=SimpleNamespace(model_type="test-hybrid-vlm"),
        )
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token_id=0, eos_token_ids={0})
        )
        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(
                model_path="example/hybrid-vlm",
                enable_prefix_cache=True,
                use_paged_cache=False,
                use_memory_aware_cache=True,
                enable_block_disk_cache=True,
                block_disk_cache_dir=str(root),
                block_disk_cache_max_gb=0.001,
                max_cache_blocks=8,
                kv_cache_quantization="q4",
            ),
        )
        try:
            block_store = scheduler.paged_cache_manager._disk_store
            companion = scheduler._ssm_companion_disk_store
            assert block_store.cache_dir.parent == root
            assert len(block_store.cache_dir.name) == 12
            assert block_store.global_cache_root == root
            assert companion.directory == block_store.cache_dir / "ssm_companion"
            assert companion._global_budget is block_store.global_budget
        finally:
            asyncio.run(scheduler.stop())

    def test_scheduler_allows_authoritative_block_l2_with_paged_ram_off(self):
        """Hybrid VLMs may use SSD blocks plus typed SSM state without RAM pages."""
        source = Path("vmlx_engine/mllm_scheduler.py").read_text()

        assert "self.config.use_paged_cache or self.config.enable_block_disk_cache" in source
        assert "block_disk_only = bool(" in source
        assert "disk_only=block_disk_only" in source
        assert "MLLM block disk-only prefix backend" in source
        assert "refusing to substitute a RAM backend" in source

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

    def test_cleanup_releases_paged_request_refs_before_detach(self):
        """Completed VLM prefixes must become cached-but-free LRU blocks.

        Merely detaching the request table leaves ref_count=1 on every block,
        permanently filling max_cache_blocks and preventing later prefixes
        from being admitted or evicting older entries.
        """
        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=1,
            _bypass_prefix_cache=True,
            _extracted_cache=None,
            _added_stop_tokens=set(),
        )
        block_table = object()
        scheduler.running = {"paged-cleanup": request}
        scheduler.block_aware_cache = SimpleNamespace(
            _request_tables={
                "paged-cleanup": SimpleNamespace(block_table=block_table)
            }
        )
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.batch_generator = None
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = MagicMock()
        scheduler.requests = {"paged-cleanup": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        with patch("vmlx_engine.mllm_scheduler.clear_mlx_memory_cache"):
            scheduler._cleanup_finished({"paged-cleanup"})

        scheduler.paged_cache_manager.release_request_refs.assert_called_once_with(
            block_table
        )
        scheduler.paged_cache_manager.detach_request.assert_called_once_with(
            "paged-cleanup"
        )
        assert "paged-cleanup" not in scheduler.block_aware_cache._request_tables

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

    def test_mixed_swa_tight_memory_store_uses_clean_prefill_when_headroom_is_safe(self):
        """MiMo JANGTQ should keep release-gate cache reuse with healthy headroom."""

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        request = SimpleNamespace(
            num_output_tokens=3,
            _bypass_prefix_cache=False,
            _cached_tokens=0,
            _extracted_cache=lambda: ["dirty-live-cache"],
            _extracted_tokens=list(range(667)),
            _added_stop_tokens=set(),
        )

        scheduler.running = {"req-mimo-headroom": request}
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
            _tight_memory_prefill_drain=True,
            _prefill_for_clean_path_dependent_cache=MagicMock(
                return_value=["clean-cache"]
            ),
        )
        scheduler.stop_tokens = set()
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.paged_cache_manager = MagicMock()
        scheduler.requests = {"req-mimo-headroom": request}
        scheduler.finished_req_ids = set()
        scheduler._cleanup_detokenizer = MagicMock()

        with patch(
            "vmlx_engine.mllm_scheduler.get_effective_metal_working_set_bytes",
            return_value=(76 * 1024**3, 107 * 1024**3),
        ):
            scheduler._cleanup_finished({"req-mimo-headroom"})

        scheduler.batch_generator._prefill_for_clean_path_dependent_cache.assert_called_once()
        args = scheduler.batch_generator._prefill_for_clean_path_dependent_cache.call_args.args
        assert len(args[0]) == 666
        scheduler.block_aware_cache.store_cache.assert_called_once()

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
            [10, 11, 12, 13],
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

    def test_dsv4_contract_is_observed_from_instantiated_cache_graph(self):
        """MLLM native validators must follow cache objects, not model names."""
        from vmlx_engine.utils.cache_types import detect_dsv4_cache_contract

        class KVCache:
            pass

        class DeepseekV4Cache:
            pass

        class DerivedDsv4(DeepseekV4Cache):
            pass

        class CacheList:
            def __init__(self, caches):
                self.caches = caches

        class Model:
            def __init__(self, cache):
                self._cache = cache

            def make_cache(self):
                return self._cache

        assert detect_dsv4_cache_contract(Model([KVCache()])) is False
        assert detect_dsv4_cache_contract(
            Model([CacheList([KVCache(), DerivedDsv4()])])
        ) is True
        assert detect_dsv4_cache_contract(object()) is None

    def test_mllm_passes_observed_dsv4_contract_to_block_cache(self):
        """Regression: omitting this kwarg doubled every Qwen SSD block read."""
        import inspect

        source = inspect.getsource(MLLMScheduler.__init__)

        assert "self._uses_dsv4_cache = detect_dsv4_cache_contract(lang_model)" in source
        assert "uses_dsv4_cache=self._uses_dsv4_cache" in source


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

    def test_mllm_tight_memory_mixed_swa_skips_clean_prefill_store(self):
        """Tight-memory MiMo keeps a bounded clean-store safety policy."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)

        assert "tight_memory_clean_store_disabled" in source
        assert "_tight_memory_prefill_drain" in source
        assert "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE" in source
        assert "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE_MAX_TOKENS" in source
        assert '"128"' in source
        assert "tight-memory clean prompt" in source
        assert "prefill disabled to avoid Metal OOM" in source


class RotatingKVCache:
    pass


class TestMLLMMixedSWACleanStorePolicy:
    class FakeBlockAwareCache:
        def __init__(self, store_result=None):
            self.stores = []
            self._request_tables = {}
            self.store_result = store_result

        def store_cache(
            self, request_id, token_ids, cache_states, cache_extra_keys=None
        ):
            self.stores.append(
                (request_id, list(token_ids), list(cache_states), cache_extra_keys)
            )
            return self.store_result

    def _scheduler(self, tokens, monkeypatch, store_result=None):
        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.block_aware_cache = self.FakeBlockAwareCache(store_result)
        scheduler.paged_cache_manager = MagicMock()
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.disk_cache = None
        scheduler.batch_generator = SimpleNamespace(
            stop_tokens=set(),
            _tight_memory_prefill_drain=True,
            _prefill_for_clean_path_dependent_cache=MagicMock(
                return_value=[RotatingKVCache()]
            ),
        )
        scheduler.stop_tokens = set()
        request = SimpleNamespace(
            request_id="mimo-clean",
            _extracted_tokens=list(tokens),
            _extracted_cache=[RotatingKVCache()],
            _added_stop_tokens=set(),
            num_output_tokens=1,
            _cached_tokens=0,
        )
        scheduler.running = {"mimo-clean": request}
        scheduler.request_id_to_uid = {"mimo-clean": 3}
        scheduler.uid_to_request_id = {3: "mimo-clean"}
        scheduler.requests = dict(scheduler.running)
        scheduler.finished_req_ids = set()
        scheduler._is_hybrid = True
        scheduler._uses_zaya_cache = False
        scheduler._mixed_attention_cache_model = False
        scheduler._kv_cache_bits = 0
        scheduler._cleanup_detokenizer = MagicMock()
        scheduler._mllm_request_has_media_cache_context = MagicMock(return_value=False)
        scheduler._mllm_media_prefix_cache_allowed = MagicMock(return_value=False)
        scheduler._validate_cache = MagicMock(return_value=True)
        scheduler._extract_cache_states = MagicMock(
            return_value=[{"class_name": "RotatingKVCache"}]
        )
        monkeypatch.delenv("VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE", raising=False)
        monkeypatch.delenv(
            "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE_MAX_TOKENS",
            raising=False,
        )
        return scheduler

    def test_tight_memory_mixed_swa_stores_short_clean_prompt_by_default(self, monkeypatch):
        scheduler = self._scheduler([10, 11, 12, 13], monkeypatch)

        scheduler._cleanup_finished({"mimo-clean"})

        # Third/fourth args are the incremental-store base: None/0 here because
        # this fixture's block cache has no prior chain to extend.
        scheduler.batch_generator._prefill_for_clean_path_dependent_cache.assert_called_once_with(
            [10, 11, 12], None, 0
        )
        assert scheduler.block_aware_cache.stores == [
            ("mimo-clean", [10, 11, 12], [{"class_name": "RotatingKVCache"}], None)
        ]

    def test_partial_paged_store_logs_actual_retained_tokens(self, monkeypatch, caplog):
        import logging

        scheduler = self._scheduler(
            [10, 11, 12, 13],
            monkeypatch,
            store_result=SimpleNamespace(num_tokens=2, block_ids=[1]),
        )

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_scheduler"):
            scheduler._cleanup_finished({"mimo-clean"})

        # the ` prefix_key=<hash>` identity suffix is graded by test_cache_identity_telemetry
        assert (
            "VLM Scheduler stored paged Prefix Cache for mimo-clean: "
            "1 layers, retained_tokens=2, block_table_blocks=1, "
            "requested_cache_key_tokens=3"
        ) in [m.split(" prefix_key=")[0] for m in caplog.messages]
        assert all(
            "truncated to 3 tokens" not in message for message in caplog.messages
        )

    def test_missing_paged_store_receipt_is_not_logged_as_success(
        self, monkeypatch, caplog
    ):
        import logging

        scheduler = self._scheduler([10, 11, 12, 13], monkeypatch)

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_scheduler"):
            scheduler._cleanup_finished({"mimo-clean"})

        assert any(
            "returned incomplete retention receipt for mimo-clean" in message
            for message in caplog.messages
        )
        assert all(
            "VLM Scheduler stored paged Prefix Cache for mimo-clean" not in message
            for message in caplog.messages
        )

    def test_zero_token_paged_store_receipt_is_not_logged_as_success(
        self, monkeypatch, caplog
    ):
        import logging

        scheduler = self._scheduler(
            [10, 11, 12, 13],
            monkeypatch,
            store_result=SimpleNamespace(num_tokens=0, block_ids=[]),
        )

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_scheduler"):
            scheduler._cleanup_finished({"mimo-clean"})

        assert any(
            "retained no cache-key tokens for mimo-clean: 1 layers, "
            "block_table_blocks=0, requested_cache_key_tokens=3" in message
            for message in caplog.messages
        )
        assert all(
            "VLM Scheduler stored paged Prefix Cache for mimo-clean" not in message
            for message in caplog.messages
        )

    @pytest.mark.parametrize(
        "store_result",
        [
            SimpleNamespace(num_tokens="invalid", block_ids=[1]),
            SimpleNamespace(num_tokens=2, block_ids=object()),
        ],
    )
    def test_malformed_paged_store_receipt_does_not_raise_after_storage(
        self, monkeypatch, caplog, store_result
    ):
        import logging

        scheduler = self._scheduler(
            [10, 11, 12, 13],
            monkeypatch,
            store_result=store_result,
        )

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_scheduler"):
            scheduler._cleanup_finished({"mimo-clean"})

        assert not any(
            "Failed to store VLM paged cache" in message for message in caplog.messages
        )
        assert any(
            "returned incomplete retention receipt for mimo-clean" in message
            for message in caplog.messages
        )
        assert all(
            "VLM Scheduler stored paged Prefix Cache for mimo-clean" not in message
            for message in caplog.messages
        )

    def test_tight_memory_mixed_swa_skips_clean_prompt_above_configured_cap(
        self, monkeypatch
    ):
        scheduler = self._scheduler([10, 11, 12, 13], monkeypatch)
        monkeypatch.setenv(
            "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE_MAX_TOKENS",
            "2",
        )

        scheduler._cleanup_finished({"mimo-clean"})

        scheduler.batch_generator._prefill_for_clean_path_dependent_cache.assert_not_called()
        assert scheduler.block_aware_cache.stores == []

    def test_tight_memory_mixed_swa_force_env_overrides_cap(self, monkeypatch):
        scheduler = self._scheduler([10, 11, 12, 13], monkeypatch)
        monkeypatch.setenv(
            "VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE_MAX_TOKENS",
            "2",
        )
        monkeypatch.setenv("VMLINUX_MLLM_TIGHT_MEMORY_CLEAN_PREFILL_STORE", "1")

        scheduler._cleanup_finished({"mimo-clean"})

        # Third/fourth args are the incremental-store base: None/0 here because
        # this fixture's block cache has no prior chain to extend.
        scheduler.batch_generator._prefill_for_clean_path_dependent_cache.assert_called_once_with(
            [10, 11, 12], None, 0
        )


class TestCleanRederiveChunking:
    """Gemma 4 mixed-SWA lost prefix caching entirely to an SSM-only guard.

    ``_prefill_for_clean_path_dependent_cache`` is the ONLY way the mixed-SWA
    VLM path produces a storable cache — a None return means "skip the store".
    It applied an O(seq_len^2) dense-attention estimate written for recurrent
    one-shot re-derives to attention-only stacks too, predicting 8.9 GB at just
    ~12k tokens on a 30-head backbone. Live on the box: 0 cached tokens on every
    turn of a 6-turn 74k conversation, TTFT climbing 3.1s -> 54.8s.
    """

    def test_mixed_swa_attention_stack_is_chunk_safe(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from vmlx_engine.mllm_batch_generator import (
            _cache_requires_one_shot_rederive,
        )

        gemma_like = [RotatingKVCache(max_size=512), KVCache()] * 4
        assert _cache_requires_one_shot_rederive(gemma_like) is False

    def test_recognised_recurrent_slot_is_chunk_safe(self, monkeypatch):
        """ArraysCache is the GatedDelta slot the byte-exactness A/B covered."""
        import importlib

        from mlx_lm.models.cache import ArraysCache, KVCache

        import vmlx_engine.mllm_batch_generator as mbg

        assert mbg._cache_requires_one_shot_rederive(
            [KVCache(), ArraysCache(4)]
        ) is False

        # The opt-out must still restore the strict rule.
        monkeypatch.setenv("VMLX_CHUNKED_SSM_REDERIVE", "0")
        strict = importlib.reload(mbg)
        try:
            assert strict._cache_requires_one_shot_rederive(
                [KVCache(), ArraysCache(4)]
            ) is True
        finally:
            monkeypatch.delenv("VMLX_CHUNKED_SSM_REDERIVE", raising=False)
            importlib.reload(mbg)

    def test_resume_from_a_recurrent_base_stays_strict(self):
        """Chunking a fresh re-derive is a different question from resuming.

        Whether the forward pass may be chunked says nothing about whether a
        RESTORED recurrent base is a safe thing to resume from, so that call
        site keeps the strict rule regardless of the chunking default.
        """
        from mlx_lm.models.cache import ArraysCache, KVCache

        from vmlx_engine.mllm_batch_generator import (
            _cache_requires_one_shot_rederive,
        )

        assert _cache_requires_one_shot_rederive(
            [KVCache(), ArraysCache(4)], ignore_chunk_override=True
        ) is True

    def test_unclassifiable_slot_fails_closed_to_one_shot(self):
        """Chunking state we cannot classify could store silently wrong data."""
        from vmlx_engine.mllm_batch_generator import (
            _cache_requires_one_shot_rederive,
        )

        assert _cache_requires_one_shot_rederive([object()]) is True

    def test_nested_cache_lists_are_inspected(self):
        from mlx_lm.models.cache import ArraysCache, KVCache

        from vmlx_engine.mllm_batch_generator import (
            _cache_requires_one_shot_rederive,
        )

        class _Wrapper:
            def __init__(self, caches):
                self.caches = caches

        assert _cache_requires_one_shot_rederive(
            [_Wrapper([KVCache(), KVCache()])]
        ) is False
        # Recognised recurrent slots nested inside a wrapper are chunk-safe
        # like any other; the strict rule still reaches them via the override.
        assert _cache_requires_one_shot_rederive(
            [_Wrapper([KVCache(), ArraysCache(4)])]
        ) is False
        assert _cache_requires_one_shot_rederive(
            [_Wrapper([KVCache(), ArraysCache(4)])], ignore_chunk_override=True
        ) is True

    def test_attention_stack_is_prefilled_in_prefill_step_chunks(self):
        """The whole point: a long attention-only prompt must NOT be rejected."""
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.prefill_step_size = 1024
        gen._cache_model = MagicMock(
            make_cache=lambda: [RotatingKVCache(max_size=512), KVCache()]
        )
        calls = []

        def _forward(input_ids, cache=None):
            calls.append(int(input_ids.shape[1]))
            return None

        gen.language_model = MagicMock(side_effect=_forward)

        tokens = list(range(12224))  # the length that used to predict 8.9 GB
        result = gen._prefill_for_clean_path_dependent_cache(tokens)

        assert result is not None, "long attention-only prompt was rejected"
        assert len(calls) == 12, f"expected 12 chunks of 1024, got {calls[:3]}..."
        assert sum(calls) == len(tokens)
        assert max(calls) <= 1024


class TestMllmIndexCapacityParity:
    """--max-cache-blocks counts BLOCKS, and MLLM blocks are 4x smaller.

    The shared default of 1000 indexes 256k tokens on the text path's 256-token
    block but only 64k on the MLLM path's 64-token block, silently capping
    reuse far below Gemma 4's context window. Measured on the box: a 77k Gemma
    prompt reported 0 cached tokens on an exact repeat and ran SLOWER than a
    cold prefill (82.5s vs 55.7s), while the same probe at 28k reused 28,199
    tokens and cut TTFT 9.10s -> 0.98s.
    """

    def test_default_matches_text_path_token_capacity(self):
        from vmlx_engine.mllm_scheduler import (
            DEFAULT_MAX_CACHE_BLOCKS,
            TEXT_REFERENCE_BLOCK_SIZE,
            resolve_mllm_index_blocks,
        )

        text_tokens = DEFAULT_MAX_CACHE_BLOCKS * TEXT_REFERENCE_BLOCK_SIZE
        for block_size in (64, 128, 256):
            blocks = resolve_mllm_index_blocks(block_size, DEFAULT_MAX_CACHE_BLOCKS)
            assert blocks * block_size == text_tokens, (
                f"block_size={block_size} indexes {blocks * block_size} tokens, "
                f"text path indexes {text_tokens}"
            )

    def test_explicit_operator_value_is_never_rescaled(self):
        from vmlx_engine.mllm_scheduler import resolve_mllm_index_blocks

        assert resolve_mllm_index_blocks(64, 4097) == 4097
        assert resolve_mllm_index_blocks(64, 100) == 100

    def test_larger_than_reference_block_is_left_alone(self):
        from vmlx_engine.mllm_scheduler import (
            DEFAULT_MAX_CACHE_BLOCKS,
            resolve_mllm_index_blocks,
        )

        assert resolve_mllm_index_blocks(512, DEFAULT_MAX_CACHE_BLOCKS) == (
            DEFAULT_MAX_CACHE_BLOCKS
        )

    def test_degenerate_block_size_does_not_explode(self):
        from vmlx_engine.mllm_scheduler import (
            DEFAULT_MAX_CACHE_BLOCKS,
            resolve_mllm_index_blocks,
        )

        assert resolve_mllm_index_blocks(0, DEFAULT_MAX_CACHE_BLOCKS) == (
            DEFAULT_MAX_CACHE_BLOCKS
        )
        assert resolve_mllm_index_blocks(-8, DEFAULT_MAX_CACHE_BLOCKS) == (
            DEFAULT_MAX_CACHE_BLOCKS
        )


class TestCleanStoreExtendsStoredChain:
    """The clean re-derive used to re-prefill the ENTIRE N-1 prompt each turn.

    Measured on Gemma 4 (box, engine 55b496bdb): TTFT 51-92s at 86k-123k even
    though only ~12k tokens were fresh, because turn N inherited turn N-1's
    ~30s store. Reconstructing the stored chain costs far less than recomputing
    it (0.98s at 28k, 5.25s at 77k vs ~30s), so the store should forward only
    the delta.
    """

    @staticmethod
    def _gen(cache_slots, prefill_step_size=1024):
        from unittest.mock import MagicMock

        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.prefill_step_size = prefill_step_size
        gen._cache_model = MagicMock(make_cache=lambda: list(cache_slots))
        gen.calls = []
        gen.language_model = MagicMock(
            side_effect=lambda input_ids, cache=None: gen.calls.append(
                int(input_ids.shape[1])
            )
        )
        return gen

    def test_only_the_delta_is_forwarded(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        gen = self._gen([RotatingKVCache(max_size=512), KVCache()])
        base = [RotatingKVCache(max_size=512), KVCache()]

        result = gen._prefill_for_clean_path_dependent_cache(
            list(range(20_000)), base, 16_000
        )

        assert result is base, "should extend the supplied base in place"
        assert sum(gen.calls) == 4000, (
            f"forwarded {sum(gen.calls)} tokens, expected only the 4000-token delta"
        )
        assert max(gen.calls) <= 1024

    def test_no_base_still_prefills_everything(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        gen = self._gen([RotatingKVCache(max_size=512), KVCache()])
        gen._prefill_for_clean_path_dependent_cache(list(range(5000)))
        assert sum(gen.calls) == 5000

    def test_recurrent_base_is_refused(self):
        """Resuming mid-sequence is a chunk boundary; SSM state cannot take one."""
        from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

        gen = self._gen([RotatingKVCache(max_size=512), KVCache()])
        recurrent_base = [KVCache(), ArraysCache(4)]

        gen._prefill_for_clean_path_dependent_cache(
            list(range(20_000)), recurrent_base, 16_000
        )

        assert sum(gen.calls) == 20_000, "recurrent base must not be resumed from"

    def test_degenerate_coverage_falls_back_to_full_prefill(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        for covered in (0, -5, 20_000, 25_000):
            gen = self._gen([RotatingKVCache(max_size=512), KVCache()])
            base = [RotatingKVCache(max_size=512), KVCache()]
            gen._prefill_for_clean_path_dependent_cache(
                list(range(20_000)), base, covered
            )
            assert sum(gen.calls) == 20_000, f"covered={covered} mishandled"


class TestCleanStoreBaseLookup:
    """The scheduler-side half: find and release the stored chain safely."""

    @staticmethod
    def _scheduler(block_table, reconstructed):
        from unittest.mock import MagicMock

        from vmlx_engine.mllm_scheduler import MLLMScheduler

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.block_aware_cache = MagicMock()
        sched.block_aware_cache.fetch_cache.return_value = (block_table, [])
        sched.block_aware_cache.reconstruct_cache.return_value = reconstructed
        return sched

    def test_returns_base_and_releases_refs(self):
        from unittest.mock import MagicMock

        table = MagicMock(num_tokens=16_000)
        sentinel = ["cache"]
        sched = self._scheduler(table, sentinel)

        base, covered = sched._clean_store_base_from_stored_chain(
            "req-1", list(range(20_000)), None
        )

        assert base is sentinel
        assert covered == 16_000
        sched.block_aware_cache.release_cache.assert_called_once()

    def test_full_coverage_returns_no_base(self):
        """Nothing left to forward — the caller's own skip path handles it."""
        from unittest.mock import MagicMock

        sched = self._scheduler(MagicMock(num_tokens=20_000), ["cache"])
        assert sched._clean_store_base_from_stored_chain(
            "req-1", list(range(20_000)), None
        ) == (None, 0)

    def test_short_chain_is_not_worth_reconstructing(self):
        from unittest.mock import MagicMock

        sched = self._scheduler(MagicMock(num_tokens=3_000), ["cache"])
        assert sched._clean_store_base_from_stored_chain(
            "req-1", list(range(20_000)), None
        ) == (None, 0)

    def test_lookup_failure_falls_back_and_still_releases(self):
        from unittest.mock import MagicMock

        from vmlx_engine.mllm_scheduler import MLLMScheduler

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.block_aware_cache = MagicMock()
        sched.block_aware_cache.fetch_cache.side_effect = RuntimeError("boom")

        assert sched._clean_store_base_from_stored_chain(
            "req-1", list(range(20_000)), None
        ) == (None, 0)
        sched.block_aware_cache.release_cache.assert_called_once()

    def test_missing_cache_is_handled(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.block_aware_cache = None
        assert sched._clean_store_base_from_stored_chain(
            "req-1", [1, 2, 3], None
        ) == (None, 0)


class TestChunkedSSMRederiveDefault:
    """Recurrent slots are chunk-safe by default.

    Requiring one contiguous pass made the re-derive decline whenever the
    predicted attention buffer exceeded the Metal single-buffer limit -- only
    ~12.5k tokens on a ~30-head model -- and a declined re-derive means the
    store is skipped, so long documents were re-prefilled in full on every
    follow-up.
    """

    def _requires_one_shot(self, monkeypatch, value):
        import importlib

        import vmlx_engine.mllm_batch_generator as mbg

        monkeypatch.delenv("VMLX_CHUNKED_SSM_REDERIVE", raising=False)
        if value is not None:
            monkeypatch.setenv("VMLX_CHUNKED_SSM_REDERIVE", value)
        mbg = importlib.reload(mbg)
        try:
            class _Recurrent:
                # mlx-lm recurrent caches carry their rolling tensors here; a
                # slot without it is unclassifiable and must stay one-shot.
                state = None

            return mbg._cache_requires_one_shot_rederive([_Recurrent()])
        finally:
            monkeypatch.delenv("VMLX_CHUNKED_SSM_REDERIVE", raising=False)
            importlib.reload(mbg)

    def test_recurrent_slot_is_chunk_safe_by_default(self, monkeypatch):
        assert self._requires_one_shot(monkeypatch, None) is False

    def test_opt_out_restores_the_one_shot_rule(self, monkeypatch):
        for value in ("0", "false", "no", "off"):
            assert self._requires_one_shot(monkeypatch, value) is True, value

    def test_typo_does_not_silently_restore_one_shot(self, monkeypatch):
        """An unrecognised value must not quietly disable long-context stores."""
        assert self._requires_one_shot(monkeypatch, "maybe") is False


class TestNullContentAssistantTurnDoesNotCrashTemplate:
    """A replayed empty assistant turn must not 500.

    A reply that spends its whole budget on reasoning finishes with `length` and
    empty content -- allowed by the never-empty contract. Replaying it used to
    reach the chat template with no `content` key at all (model_dump drops None),
    and Mistral's template raises rather than rendering an empty turn.
    """

    def test_server_fills_empty_content_for_contentless_assistant_turns(self):
        # The inline filter became the shared _drop_contentless_assistant_turns
        # helper so every dialect's request prep (chat, responses, anthropic,
        # streaming ollama) applies the same drop. Pin the helper's behavior
        # instead of the old inline source shape.
        from vmlx_engine.server import _drop_contentless_assistant_turns

        kept = _drop_contentless_assistant_turns(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": ""},
                {"role": "assistant"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
            ]
        )
        assert kept == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        ], "a tool-call-only assistant turn is valid and must NOT be dropped"

    def test_responses_path_drops_contentless_assistant_turns_too(self):
        """The app talks to /v1/responses, which builds messages separately.

        Fixing only the chat-completions loop would leave the surface users
        actually hit still returning 500.
        """
        import inspect

        from vmlx_engine.server import _responses_input_to_messages

        source = inspect.getsource(_responses_input_to_messages)
        assert "_drop_contentless_assistant_turns(" in source, (
            "the responses builder must apply the shared contentless drop"
        )

    def test_responses_builder_drops_the_empty_turn_in_practice(self):
        from vmlx_engine.server import _responses_input_to_messages

        out = _responses_input_to_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "Name one colour."},
            ]
        )
        assert [m["role"] for m in out] == ["user", "user"], out


class TestVmlx91ResumeAlignmentGuard:
    """MLLM resume must refuse a KV/SSM pairing that is not exactly aligned.

    KV block tables trim to whole blocks; SSM state is cumulative at the
    checkpoint length. Accepting KV@aligned<checkpoint re-feeds the gap
    tokens through layers whose state already absorbed them (the LLM
    scheduler pins the same contract via checkpoint_len == aligned_len).
    """

    def test_resume_site_gates_on_exact_alignment(self):
        from pathlib import Path

        source = Path("vmlx_engine/mllm_batch_generator.py").read_text()
        marker = '"reason": "kv_ssm_checkpoint_misaligned"'
        assert marker in source, "misalignment refusal is missing from the MLLM resume"
        window = source[source.index(marker) - 1500:source.index(marker) + 2400]
        assert "int(trimmed.num_tokens) != int(_ck_len or 0)" in window
        assert "release_cache(req.request_id)" in window
        assert "_adjust_paged_hit_credit(req.request_id, 0)" in window
