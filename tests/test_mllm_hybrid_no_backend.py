"""A hybrid without a restorable SSD backend must not fall back to RAM."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx_lm.models.cache import ArraysCache, KVCache

from vmlx_engine.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache, Glm5MLACache
from vmlx_engine.persistence_outcome import LEDGER


@pytest.mark.parametrize("cache_kind", ["glm5", "hybrid", "kv"])
@pytest.mark.parametrize("requested", [True, False])
def test_no_backend_does_not_retain_unrestorable_hybrid_state(
    tmp_path, cache_kind, requested
):
    class LanguageModel:
        config = SimpleNamespace(model_type="test", num_hidden_layers=2)

        def make_cache(self):
            if cache_kind == "glm5":
                return [Glm5KDACache(), Glm5MLACache()]
            if cache_kind == "hybrid":
                return [ArraysCache(size=2), KVCache()]
            return [KVCache(), KVCache()]

    model = SimpleNamespace(
        language_model=LanguageModel(), config=LanguageModel.config
    )
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=0, eos_token_ids={0})
    )
    scheduler = MLLMScheduler(
        model=model,
        processor=processor,
        config=MLLMSchedulerConfig(
            enable_prefix_cache=requested,
            use_paged_cache=False,
            enable_block_disk_cache=False,
            use_memory_aware_cache=False,
            enable_disk_cache=True,
            disk_cache_dir=str(tmp_path / "prompt-cache"),
        ),
    )
    try:
        if cache_kind != "kv":
            assert scheduler.prefix_cache is None
            assert scheduler.memory_aware_cache is None
            assert scheduler.disk_cache is None
            assert scheduler.block_aware_cache is None
            assert scheduler.config.enable_prefix_cache is False
            assert scheduler._prefix_cache_requested is requested
            assert bool(scheduler._prefix_cache_unavailable_reason) is requested
            if requested:
                request_id = f"no-backend-{cache_kind}"
                extract = Mock(side_effect=AssertionError("must not resolve payload"))
                request = SimpleNamespace(
                    num_output_tokens=1,
                    _extracted_cache=extract,
                    _extracted_tokens=[1, 2, 3],
                    _added_stop_tokens=set(),
                )
                scheduler.running[request_id] = request
                scheduler.requests[request_id] = request
                scheduler._cleanup_finished({request_id})
                extract.assert_not_called()
                assert request._extracted_cache is None
                outcome = LEDGER.take(request_id)
                assert outcome["outcome"] == "skipped"
                assert outcome["durable"] is False
                assert outcome["retained_tokens"] == 0
                assert "no RAM fallback" in outcome["detail"]
        elif requested:
            # An ordinary KV CLI legacy backend is independently supported.
            assert scheduler.prefix_cache is not None
            assert scheduler.disk_cache is not None
            assert scheduler.config.enable_prefix_cache is True
        else:
            assert scheduler.prefix_cache is None
            assert scheduler.disk_cache is None
    finally:
        asyncio.run(scheduler.stop())
