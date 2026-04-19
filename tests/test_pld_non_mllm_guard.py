# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for vmlx#92 — PLD speculative decode must short-circuit
when the BatchGenerator isn't MLLMBatchGenerator.

Before the guard, `_try_speculative_decode` reached `self.batch_generator.
active_batch`, raised AttributeError, and the finally-block's emergency
re-insert then corrupted the batch so the next `step()` died with
`<class 'list'> does not yet support batching with history` — triggering
a full paged-cache wipe on recovery. Guarding BEFORE `remove()` keeps the
batch state clean and lets the retrospective prompt-lookup analyzer
continue to run normally.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestPLDNonMLLMGuard:
    """vmlx#92 — short-circuit PLD-spec when active_batch is absent."""

    def _make_scheduler_shim(self, batch_generator):
        """Build the minimum attribute surface `_try_speculative_decode`
        touches before the guard fires: temperature gate, n-gram cache,
        uid map, spec-attempt counter, batch_generator."""
        from vmlx_engine.scheduler import Scheduler

        sched = Scheduler.__new__(Scheduler)
        sched._pld_spec_max_temp = 0.8
        sched._pld_ngram_indices = {}
        sched._pld_spec_attempts = 0
        sched.request_id_to_uid = {"req-1": 42}
        sched.batch_generator = batch_generator
        return sched

    def _make_request(self):
        """Request stub carrying the minimum fields the early gates read."""
        req = SimpleNamespace()
        req.sampling_params = SimpleNamespace(
            temperature=0.0,
            max_tokens=128,
        )
        req.prompt_token_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        # Force the n-gram lookup to find drafts: repeat a 3-gram so the
        # n-gram index surfaces the continuation, and ensure we're past the
        # `remaining <= 1` gate.
        req.output_token_ids = [20, 21, 22, 23, 20, 21, 22]
        req.num_output_tokens = len(req.output_token_ids)
        return req

    def test_returns_empty_on_non_mllm_generator(self):
        """No `.active_batch` on the generator → early `[]` return, no
        batch-state mutations."""
        # Deliberately use an object WITHOUT an `active_batch` attribute —
        # mirrors mlx-lm's plain BatchGenerator.
        bg = MagicMock(spec=["insert", "remove", "step"])
        sched = self._make_scheduler_shim(bg)
        req = self._make_request()

        result = sched._try_speculative_decode("req-1", req, last_token=23)

        assert result == []
        assert not bg.remove.called, (
            "non-MLLM guard must short-circuit BEFORE remove() — "
            "otherwise batch state is corrupted on the next step()"
        )
        assert not bg.insert.called

    def test_does_not_count_as_spec_attempt(self):
        """Attempts counter must not advance on non-MLLM early-return,
        otherwise [PLD:*] summaries report phantom attempts that never
        actually ran verification."""
        bg = MagicMock(spec=["insert", "remove", "step"])
        sched = self._make_scheduler_shim(bg)
        req = self._make_request()

        sched._try_speculative_decode("req-1", req, last_token=23)
        sched._try_speculative_decode("req-1", req, last_token=23)

        assert sched._pld_spec_attempts == 0

    def test_mllm_generator_not_short_circuited_by_guard(self):
        """Sanity check — a generator that DOES have `.active_batch`
        isn't short-circuited by the new guard. (It may still return []
        further down for other reasons, but the guard itself must not
        fire.)"""
        ab = SimpleNamespace(uids=[], logprobs=[])
        bg = MagicMock(spec=["insert", "remove", "step", "active_batch"])
        bg.active_batch = ab
        sched = self._make_scheduler_shim(bg)
        req = self._make_request()

        # uid not in active_batch.uids → downstream raises RuntimeError,
        # caught by the try/except, returns []. The point is that the
        # guard itself didn't fire — execution reached the `ab.uids`
        # check, proving the hasattr path allowed through.
        result = sched._try_speculative_decode("req-1", req, last_token=23)

        assert result == []
        # Attempts counter advances because we got past the guard and
        # entered the real speculation path (before it raised downstream).
        assert sched._pld_spec_attempts == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
