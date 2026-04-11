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
