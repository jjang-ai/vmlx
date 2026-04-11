# SPDX-License-Identifier: Apache-2.0
"""Strict cache-bypass enforcement suite.

These tests enforce two guarantees that must hold for every model family
regardless of whether it's JANG, VL, or hybrid SSM:

1. **Cache bypass is absolute.** When a request carries `cache_salt` or
   `skip_prefix_cache=True`, no prefix cache layer may return stored state
   and no prefix cache layer may store new state. This applies to the paged
   cache, memory-aware cache, legacy prefix cache, disk L2, block disk store,
   AND the SSM companion cache for hybrid models.

2. **Multi-turn context is preserved WITHOUT bypass.** A follow-up request
   within the same session must correctly consult the cache (`_bypass = False`
   code path). The bypass gates must not accidentally disable the happy path.

These tests are source-level: they assert that specific gating expressions
appear in the right files at the right sites. They run in <1s because they
don't load any model weights, and they catch regressions BEFORE a release.
A future refactor that accidentally drops a bypass gate will fail these
tests immediately.
"""
from __future__ import annotations

import pytest

from vmlx_engine.api.models import ChatCompletionRequest, CompletionRequest
from vmlx_engine.request import Request, SamplingParams


# ---------------------------------------------------------------------------
# API model layer: cache_salt / skip_prefix_cache field acceptance
# ---------------------------------------------------------------------------


class TestAPIModelFields:
    """The OpenAI-compatible request models must accept the new fields."""

    def test_chat_request_accepts_cache_salt_string(self):
        r = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            cache_salt="benchmark-run-42",
        )
        assert r.cache_salt == "benchmark-run-42"
        assert r.skip_prefix_cache is None

    def test_chat_request_accepts_skip_prefix_cache_bool(self):
        r = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            skip_prefix_cache=True,
        )
        assert r.skip_prefix_cache is True
        assert r.cache_salt is None

    def test_chat_request_default_values_do_not_bypass(self):
        r = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert r.cache_salt is None
        assert r.skip_prefix_cache is None

    def test_completion_request_accepts_cache_salt(self):
        r = CompletionRequest(model="test", prompt="hi", cache_salt="run-1")
        assert r.cache_salt == "run-1"

    def test_completion_request_accepts_skip_prefix_cache(self):
        r = CompletionRequest(model="test", prompt="hi", skip_prefix_cache=True)
        assert r.skip_prefix_cache is True


# ---------------------------------------------------------------------------
# Server helper: _compute_bypass_prefix_cache
# ---------------------------------------------------------------------------


class TestComputeBypassFlag:
    """The helper that decides whether a request bypasses cache must
    return True ONLY when the request explicitly asked for it."""

    def test_none_request_returns_false(self):
        from vmlx_engine.server import _compute_bypass_prefix_cache
        assert _compute_bypass_prefix_cache(None) is False

    def test_no_bypass_fields_returns_false(self):
        from vmlx_engine.server import _compute_bypass_prefix_cache

        class R:
            cache_salt = None
            skip_prefix_cache = None

        assert _compute_bypass_prefix_cache(R()) is False

    def test_empty_cache_salt_returns_false(self):
        """Empty-string salt must not trigger bypass — it's semantically
        the same as no salt at all. Otherwise clients that default-construct
        the field to '' would accidentally opt into bypass mode."""
        from vmlx_engine.server import _compute_bypass_prefix_cache

        class R:
            cache_salt = ""
            skip_prefix_cache = None

        assert _compute_bypass_prefix_cache(R()) is False

    def test_non_empty_cache_salt_returns_true(self):
        from vmlx_engine.server import _compute_bypass_prefix_cache

        class R:
            cache_salt = "abc-123"
            skip_prefix_cache = None

        assert _compute_bypass_prefix_cache(R()) is True

    def test_explicit_skip_prefix_cache_returns_true(self):
        from vmlx_engine.server import _compute_bypass_prefix_cache

        class R:
            cache_salt = None
            skip_prefix_cache = True

        assert _compute_bypass_prefix_cache(R()) is True

    def test_skip_prefix_cache_false_returns_false(self):
        from vmlx_engine.server import _compute_bypass_prefix_cache

        class R:
            cache_salt = None
            skip_prefix_cache = False

        assert _compute_bypass_prefix_cache(R()) is False

    def test_cache_salt_and_skip_both_set_returns_true(self):
        from vmlx_engine.server import _compute_bypass_prefix_cache

        class R:
            cache_salt = "x"
            skip_prefix_cache = True

        assert _compute_bypass_prefix_cache(R()) is True


# ---------------------------------------------------------------------------
# Request object: _bypass_prefix_cache attribute
# ---------------------------------------------------------------------------


class TestRequestBypassAttribute:
    """The Request class must carry the bypass flag through the scheduler
    pipeline. Default must be False so normal requests use cache normally."""

    def _make_request(self) -> Request:
        return Request(
            request_id="test-req",
            prompt="hello",
            sampling_params=SamplingParams(),
        )

    def test_default_bypass_is_false(self):
        req = self._make_request()
        assert req._bypass_prefix_cache is False

    def test_set_bypass_to_true(self):
        req = self._make_request()
        req._bypass_prefix_cache = True
        assert req._bypass_prefix_cache is True

    def test_bypass_survives_attribute_access_with_default(self):
        """getattr with a `False` default must return the real value, not
        the default — otherwise scheduler-level `getattr(request,
        '_bypass_prefix_cache', False)` would always evaluate to False."""
        req = self._make_request()
        req._bypass_prefix_cache = True
        assert getattr(req, "_bypass_prefix_cache", False) is True


# ---------------------------------------------------------------------------
# Source-level gating assertions: scheduler + mllm_scheduler + mllm_batch_generator
# ---------------------------------------------------------------------------


class TestSchedulerBypassGating:
    """Source-level assertions that every cache path in scheduler.py,
    mllm_scheduler.py, and mllm_batch_generator.py gates on
    `_bypass_prefix_cache` either directly (at fetch sites) or via
    `_skip_cache_store` (at store sites).

    These run without any model weights. They catch a future refactor
    that accidentally drops a gate — which is the exact class of bug
    we've already seen once and must prevent from recurring.
    """

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_scheduler_schedule_has_bypass_gate(self):
        src = self._read("vmlx_engine/scheduler.py")
        # The main _schedule_request path must declare the bypass variable
        assert "_bypass = bool(getattr(request, \"_bypass_prefix_cache\"" in src, (
            "scheduler.py lost the _bypass variable declaration"
        )
        # block_aware_cache fetch must be gated
        assert (
            "self.block_aware_cache is not None and not _bypass" in src
        ), "scheduler.py block_aware_cache fetch is no longer gated on _bypass"
        # memory_aware_cache fetch must be gated
        assert (
            "self.memory_aware_cache is not None and not _bypass" in src
        ), "scheduler.py memory_aware_cache fetch is no longer gated on _bypass"
        # legacy prefix_cache fetch must be gated
        assert (
            "self.prefix_cache is not None and not _bypass" in src
        ), "scheduler.py legacy prefix_cache fetch is no longer gated on _bypass"
        # disk L2 fallback must be gated
        assert (
            "self.disk_cache is not None and not _bypass" in src
        ), "scheduler.py disk L2 fetch is no longer gated on _bypass"

    def test_scheduler_store_path_honors_bypass(self):
        src = self._read("vmlx_engine/scheduler.py")
        # _skip_cache_store must get forced to True when bypass is set
        assert 'getattr(request, "_bypass_prefix_cache", False):' in src, (
            "scheduler.py store path no longer reads _bypass_prefix_cache"
        )
        # And the line immediately after must set _skip_cache_store = True
        idx = src.index('getattr(request, "_bypass_prefix_cache", False):')
        # Search forward within a small window for the assignment
        window = src[idx : idx + 200]
        assert "_skip_cache_store = True" in window, (
            "scheduler.py bypass check no longer forces _skip_cache_store = True"
        )

    def test_mllm_scheduler_store_path_honors_bypass(self):
        src = self._read("vmlx_engine/mllm_scheduler.py")
        assert (
            "getattr(request, '_bypass_prefix_cache', False):" in src
        ), "mllm_scheduler.py store path no longer checks _bypass_prefix_cache"
        idx = src.index("getattr(request, '_bypass_prefix_cache', False):")
        window = src[idx : idx + 200]
        assert "_skip_cache_store = True" in window, (
            "mllm_scheduler.py bypass check no longer forces _skip_cache_store = True"
        )

    def test_mllm_scheduler_add_request_reads_bypass(self):
        src = self._read("vmlx_engine/mllm_scheduler.py")
        assert 'kwargs.get("bypass_prefix_cache", False)' in src, (
            "mllm_scheduler.add_request no longer reads bypass_prefix_cache from kwargs"
        )
        assert "request._bypass_prefix_cache = True" in src, (
            "mllm_scheduler.add_request no longer attaches _bypass_prefix_cache to request"
        )

    def test_mllm_batch_generator_gates_all_three_fetch_paths(self):
        src = self._read("vmlx_engine/mllm_batch_generator.py")
        # Must have the bypass variable
        assert "_mllm_bypass" in src, (
            "mllm_batch_generator.py lost the _mllm_bypass variable"
        )
        # Must check _bypass_prefix_cache on the request
        assert "_bypass_prefix_cache" in src, (
            "mllm_batch_generator.py no longer reads _bypass_prefix_cache"
        )
        # Must gate at least 3 fetch paths (paged, memory-aware, disk L2)
        assert src.count("not _mllm_bypass") >= 3, (
            "mllm_batch_generator.py must gate paged + memory-aware + disk-L2 "
            f"fetches — found only {src.count('not _mllm_bypass')} gates"
        )

    def test_engine_core_add_request_accepts_bypass(self):
        src = self._read("vmlx_engine/engine_core.py")
        assert "bypass_prefix_cache: bool = False" in src, (
            "engine_core.add_request no longer accepts bypass_prefix_cache parameter"
        )
        assert "request._bypass_prefix_cache = True" in src, (
            "engine_core.add_request no longer sets _bypass_prefix_cache on Request"
        )

    def test_batched_engine_threads_bypass_to_engine(self):
        src = self._read("vmlx_engine/engine/batched.py")
        # Must pop from kwargs at top of generate() and stream_generate()
        pop_count = src.count('kwargs.pop("_bypass_prefix_cache"')
        assert pop_count >= 2, (
            f"batched.py must pop _bypass_prefix_cache in at least 2 methods "
            f"(generate + stream_generate) — found {pop_count}"
        )
        # Must forward as bypass_prefix_cache=bypass_prefix_cache to both
        # LLM and MLLM paths in both generate and stream_generate
        fwd_count = src.count("bypass_prefix_cache=bypass_prefix_cache")
        assert fwd_count >= 4, (
            f"batched.py must forward bypass to LLM + MLLM paths in both "
            f"generate() and stream_generate() — found only {fwd_count} forwards"
        )

    def test_simple_engine_eats_bypass_kwarg(self):
        """SimpleEngine has no prefix cache but must still pop the kwarg so
        it doesn't leak into mlx_lm.generate which would reject unknown kwargs."""
        src = self._read("vmlx_engine/engine/simple.py")
        pop_count = src.count('kwargs.pop("_bypass_prefix_cache"')
        assert pop_count >= 4, (
            f"SimpleEngine must pop _bypass_prefix_cache in all 4 methods "
            f"(generate, stream_generate, chat, stream_chat) — found {pop_count}"
        )


# ---------------------------------------------------------------------------
# Server: gateway forwarding of cache_salt → chat_kwargs
# ---------------------------------------------------------------------------


class TestServerForwarding:
    """Each API gateway handler in server.py must forward the bypass flag
    into the kwargs it passes to engine.chat/stream_chat/generate."""

    def _read_server(self):
        with open("vmlx_engine/server.py") as f:
            return f.read()

    def test_helper_exists(self):
        src = self._read_server()
        assert "def _compute_bypass_prefix_cache(" in src, (
            "server.py lost the _compute_bypass_prefix_cache helper"
        )

    def test_all_gateway_forward_sites_set_bypass_kwarg(self):
        """Every `_resolve_repetition_penalty(...)` forward site must be
        paired with a `_compute_bypass_prefix_cache(...)` check that sets
        `_bypass_prefix_cache` in the same kwargs dict. This ensures
        Anthropic, Ollama, OpenAI chat, Responses API, and both streaming
        and non-streaming completions all honor the flag."""
        src = self._read_server()
        call_count = src.count("_compute_bypass_prefix_cache(")
        # 6 forward sites + 1 helper definition = 7 total
        assert call_count >= 7, (
            f"Expected at least 7 _compute_bypass_prefix_cache references "
            f"(definition + 6 forward sites) — found {call_count}"
        )
        assert '_msg_kwargs["_bypass_prefix_cache"] = True' in src, (
            "Anthropic forward site lost the _bypass_prefix_cache assignment"
        )
        chat_kwargs_assigns = src.count('chat_kwargs["_bypass_prefix_cache"] = True')
        assert chat_kwargs_assigns >= 3, (
            f"Need at least 3 chat_kwargs assignments (ollama + openai + "
            f"responses) — found {chat_kwargs_assigns}"
        )
        gen_kwargs_assigns = src.count('gen_kwargs["_bypass_prefix_cache"] = True')
        assert gen_kwargs_assigns >= 2, (
            f"Need at least 2 gen_kwargs assignments (completions non-stream "
            f"+ streaming) — found {gen_kwargs_assigns}"
        )


# ---------------------------------------------------------------------------
# Multi-turn coherence: the happy path without bypass must still work
# ---------------------------------------------------------------------------


class TestHappyPathStillUsesCache:
    """Negative-space test: bypass gates must not leak into the non-bypass
    happy path. Source-level check that the gating is conditional, not
    unconditional."""

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_scheduler_still_does_fetch_when_not_bypassed(self):
        """The gate is `and not _bypass` — without this modifier the fetch
        would always be skipped. Check that the old unconditional form
        isn't present (regression guard)."""
        src = self._read("vmlx_engine/scheduler.py")
        # These patterns would indicate a broken gate that always bypasses
        forbidden = [
            "if self.block_aware_cache is not None:\n            # Use paged",
            "if False:  # bypass",
        ]
        for pattern in forbidden:
            assert pattern not in src, (
                f"scheduler.py contains broken gate pattern: {pattern!r} — "
                f"the happy (non-bypass) path would be dead"
            )

    def test_default_request_does_not_bypass(self):
        """Construct a fresh Request with no bypass → scheduler's
        `getattr(req, '_bypass_prefix_cache', False)` returns False → fetch
        paths run as normal."""
        from vmlx_engine.request import Request, SamplingParams
        req = Request(
            request_id="happy-path",
            prompt="normal request",
            sampling_params=SamplingParams(),
        )
        assert getattr(req, "_bypass_prefix_cache", False) is False

    def test_chat_request_without_cache_salt_does_not_bypass(self):
        """Building a ChatCompletionRequest without cache_salt or
        skip_prefix_cache must not accidentally trigger bypass."""
        from vmlx_engine.server import _compute_bypass_prefix_cache
        r = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        )
        assert _compute_bypass_prefix_cache(r) is False


# ---------------------------------------------------------------------------
# Mixed-attention auto-bypass (Gemma 4 sliding + full pattern)
# ---------------------------------------------------------------------------


class TestMixedAttentionAutoBypass:
    """Gemma 4 and similar models with interleaved sliding-window + full
    attention layers drift under the prefix-cache reconstruct path and
    produce word loops on the 3rd multi-turn request. The schedulers
    detect this at init and transparently bypass the prefix cache on
    every request.
    """

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_llm_scheduler_has_mixed_attention_helper(self):
        src = self._read("vmlx_engine/scheduler.py")
        assert "def _model_has_mixed_attention(" in src, (
            "scheduler.py lost the _model_has_mixed_attention helper"
        )
        assert "_force_bypass_prefix_cache" in src, (
            "scheduler.py lost the _force_bypass_prefix_cache flag"
        )
        assert 'request._bypass_prefix_cache = True' in src, (
            "scheduler.add_request no longer forces the per-request bypass flag"
        )

    def test_mllm_scheduler_has_mixed_attention_helper(self):
        src = self._read("vmlx_engine/mllm_scheduler.py")
        assert "def _model_has_mixed_attention(" in src, (
            "mllm_scheduler.py lost the _model_has_mixed_attention helper"
        )
        assert "_force_bypass_prefix_cache" in src, (
            "mllm_scheduler.py lost the _force_bypass_prefix_cache flag"
        )

    def test_mixed_attention_helper_detects_gemma4_layer_types(self):
        """Feed the helper a mock model whose args.layer_types look like
        Gemma 4 (25 sliding + 5 full) and assert it returns True."""
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            layer_types = (["sliding_attention"] * 25) + (["full_attention"] * 5)

        class _Model:
            args = _Args()

        assert Scheduler._model_has_mixed_attention(_Model()) is True

    def test_mixed_attention_helper_ignores_uniform_models(self):
        """Uniform-attention models (Llama, Qwen) must return False so
        they keep the full prefix-cache behaviour."""
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            layer_types = ["full_attention"] * 32

        class _Model:
            args = _Args()

        assert Scheduler._model_has_mixed_attention(_Model()) is False

    def test_mixed_attention_helper_handles_text_config(self):
        """VLM wrappers store the attention layout under text_config."""
        from vmlx_engine.scheduler import Scheduler

        class _TextCfg:
            layer_types = (["sliding_attention"] * 2) + (["full_attention"] * 1)

        class _Cfg:
            text_config = _TextCfg()

        class _Model:
            config = _Cfg()

        assert Scheduler._model_has_mixed_attention(_Model()) is True


# ---------------------------------------------------------------------------
# RotatingKVCache meta_state truncation: keep + max_size must NOT be lost
# ---------------------------------------------------------------------------


class TestRotatingKVCacheMetaStateTruncation:
    """Gemma 4 uses RotatingKVCache for sliding layers with
    meta_state = (keep, max_size, offset, _idx). The scheduler's
    gen_prompt_len truncation path previously assumed slot 0 was ``offset``
    and silently overwrote ``keep`` with the truncated sequence length —
    this is the bug that prompted the RotatingKVCache investigation.
    The helper below must preserve keep + max_size and only touch
    offset/_idx.
    """

    def test_rotating_kv_cache_preserves_keep_and_max_size(self):
        from vmlx_engine.scheduler import _rebuild_meta_state_after_truncation
        meta = _rebuild_meta_state_after_truncation(
            "RotatingKVCache",
            (str(0), str(1024), str(150), str(150)),
            safe_len=93,
        )
        assert meta == ("0", "1024", "93", "93"), (
            f"RotatingKVCache meta_state slot 0 (keep) must stay at 0, "
            f"slot 1 (max_size) must stay at 1024; got {meta}"
        )

    def test_rotating_kv_cache_wrapped_refuses_store(self):
        """A circular buffer that has wrapped (offset > max_size) cannot
        be truncated by a head-aligned slice — refuse to store rather
        than corrupt."""
        from vmlx_engine.scheduler import _rebuild_meta_state_after_truncation
        meta = _rebuild_meta_state_after_truncation(
            "RotatingKVCache",
            (str(0), str(1024), str(2000), str(800)),
            safe_len=93,
        )
        assert meta is None

    def test_standard_kv_cache_meta_state_unchanged(self):
        from vmlx_engine.scheduler import _rebuild_meta_state_after_truncation
        meta = _rebuild_meta_state_after_truncation(
            "KVCache",
            (str(150),),
            safe_len=93,
        )
        assert meta == ("93",)

    def test_quantized_kv_cache_preserves_group_size_and_bits(self):
        from vmlx_engine.scheduler import _rebuild_meta_state_after_truncation
        meta = _rebuild_meta_state_after_truncation(
            "QuantizedKVCache",
            (str(150), str(64), str(8)),
            safe_len=93,
        )
        assert meta == ("93", "64", "8"), (
            f"QuantizedKVCache must preserve (group_size, bits); got {meta}"
        )


# ---------------------------------------------------------------------------
# DeepseekV32Attention MLA absorb fp32-SDPA fix (GLM-5.1, DeepSeek-V3.2-Exp)
# ---------------------------------------------------------------------------


class TestDeepseekV32AbsorbFp32Patch:
    """The bundled mlx_lm/models/deepseek_v32.py must keep the L==1 absorb-
    branch SDPA inputs cast to fp32, otherwise GLM-5.1 / glm_moe_dsa decode
    drifts ~7.0 in logit magnitude per token vs the prefill path and
    produces repetition loops ('1.1.1.1...', 'precedence precedence...').

    Source-level guard: a future mlx_lm bump that overwrites the bundled
    file will revert this patch silently. These tests fail immediately when
    that happens, so the regression is caught before a release ships.
    """

    def _read(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def test_bundled_deepseek_v32_has_fp32_absorb_fix(self):
        src = self._read(
            "panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/deepseek_v32.py"
        )
        # The fp32 cast lines must be present in the L==1 branch.
        assert "q_sdpa = q_nope.astype(mx.float32)" in src, (
            "deepseek_v32.py lost the q_nope→fp32 cast on the L==1 branch — "
            "GLM-5.1 / DeepSeek-V3.2 will repetition-loop on decode"
        )
        assert "k_sdpa = k.astype(mx.float32)" in src
        assert "v_sdpa = v.astype(mx.float32)" in src
        assert "mask_sdpa = pe_scores.astype(mx.float32)" in src
        # The else branch must alias the variables (no fp32 on prefill path).
        assert "q_sdpa, k_sdpa, v_sdpa, mask_sdpa = q_nope, k, v, pe_scores" in src
        # Output must be cast back to bf16 before unembed_out.
        assert "output = output.astype(kv_latent.dtype)" in src, (
            "deepseek_v32.py lost the output→bf16 cast-back before unembed_out"
        )
        # SDPA must use the _sdpa variables, not the originals.
        assert "q_sdpa, k_sdpa, v_sdpa, cache=cache, scale=self.scale, mask=mask_sdpa" in src

    def test_only_l_eq_1_branch_is_touched_not_prefill(self):
        """Negative-space guard: the prefill (L != 1) branch must be
        unchanged so we don't accidentally slow down every other MLA model
        (Mistral 4, Qwen3.5 MLA variants, DeepSeek V3 / V2)."""
        src = self._read(
            "panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/deepseek_v32.py"
        )
        # Prefill branch still uses the original logic
        assert "k = self.embed_q(kv_latent, transpose=False)" in src
        assert "v = self.unembed_out(kv_latent)" in src

    def test_bundled_deepseek_v32_module_imports(self):
        """Smoke test: the patched module must still be valid Python."""
        import importlib
        mod = importlib.import_module("mlx_lm.models.deepseek_v32")
        assert hasattr(mod, "DeepseekV32Attention"), "module schema regressed"
        assert hasattr(mod, "Model"), "module schema regressed"
        import inspect
        attn_src = inspect.getsource(mod.DeepseekV32Attention.__call__)
        assert "q_sdpa" in attn_src, (
            "loaded mlx_lm has unpatched deepseek_v32.py — bundled and "
            "active site-packages may have drifted; reapply patch"
        )

    def test_bundled_mistral4_has_fp32_absorb_fix(self):
        """Mistral 4's MLA absorb decode path needs the same fp32-SDPA cast
        as deepseek_v32 — same bug, same shape (kv_lora_rank contraction at
        bf16). Without it, JANG_2L (2-bit) profiles produce 'stern bard bard'
        word loops on the very first decode.
        """
        src = self._read(
            "panel/bundled-python/python/lib/python3.12/site-packages/mlx_lm/models/mistral4.py"
        )
        assert "q_sdpa = q_nope.astype(mx.float32)" in src, (
            "mistral4.py lost the q_nope→fp32 cast on the L==1 branch — "
            "JANG 2L Mistral 4 119B will repetition-loop on decode"
        )
        assert "k_sdpa = k.astype(mx.float32)" in src
        assert "v_sdpa = v.astype(mx.float32)" in src
        assert "mask_sdpa = pe_scores.astype(mx.float32)" in src
        assert "q_sdpa, k_sdpa, v_sdpa, mask_sdpa = q_nope, k, v, pe_scores" in src
        assert "output = output.astype(kv_latent.dtype)" in src
        assert "q_sdpa, k_sdpa, v_sdpa, cache=cache, scale=self.scale, mask=mask_sdpa" in src


# ---------------------------------------------------------------------------
# Mistral-Small-4-119B JANG model_type promotion (mistral3 wrapper hides MLA)
# ---------------------------------------------------------------------------


class TestMistral4ModelTypePromotion:
    """Mistral-Small-4-119B's HF config.json has top-level model_type=mistral3
    (the VLM wrapper) but text_config.model_type=mistral4 (MLA inner). Loading
    via the VLM wrapper used the standard q_proj/k_proj/v_proj attention from
    mlx_vlm/models/mistral3/language.py, which has nowhere to land the JANG
    MLA weights → modules kept random init → 'armanarmanarman' / 'Bub Bub'
    token soup. Earlier was tracked as a JANG quality issue, but it's a
    loader regression. Promotion + LM-strip + re-quantize landed 2026-04-11.
    """

    def _read(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def test_v2_llm_loader_has_mistral4_promotion(self):
        src = self._read("vmlx_engine/utils/jang_loader.py")
        assert 'Mistral 4 model_type promotion' in src
        assert 'top mistral3 + text_config' in src
        assert "_flat = dict(_tc_for_model_type)" in src
        assert "_flat.setdefault(\"model_type\", \"mistral4\")" in src

    def test_v2_llm_loader_has_lm_strip(self):
        src = self._read("vmlx_engine/utils/jang_loader.py")
        assert "_needs_mistral4_lm_strip" in src
        assert "Mistral 4 LM-prefix strip" in src

    def test_v2_llm_loader_has_post_promo_requantize(self):
        src = self._read("vmlx_engine/utils/jang_loader.py")
        assert "Re-quantized" in src
        assert "_renamed_quant_paths" in src
        assert "_post_promo_predicate" in src
        # The predicate must also pre-register embed_q / unembed_out paths
        # since mlx_lm/mistral4.py:sanitize splits kv_b_proj into them.
        assert ".embed_q" in src
        assert ".unembed_out" in src
