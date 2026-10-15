"""The batched engine's tight-memory text step projects a chunk x context attention-score buffer
(prefill_admission.max_prefill_chunk_tokens). MiMo-V2.6 JANGHT on the NAX 192/128 kernel never builds that buffer and
declares it (_vmlx_prefill_scores_materialized = False): its text prefill must then run at the configured step. The
control (no declaration) keeps the projection: 1024-row chunks at 32k context with 17 GB of headroom."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest


def _run(monkeypatch, declared):
    seen = {}

    class FakeModel:
        _mimo_v26_runtime = True

    class FakeLM:
        def __call__(self, inputs, cache=None, inputs_embeds=None):
            raise AssertionError("the MiMo text prefill owns this forward")

    model = FakeModel()
    if declared is not None:
        model._vmlx_prefill_scores_materialized = declared

    def fake_prefill(request, input_ids, cache, kwargs, *, step):
        seen["step"] = step
        return SimpleNamespace(logits=mx.zeros((1, 1, 5)))

    fake_self = SimpleNamespace(
        _is_hybrid=False, _model_type="mimo_v2", _tight_memory_prefill_drain=True, prefill_step_size=2048,
        language_model=FakeLM(), model=model, _steady_cache_limit=None,
        _turn_peak_walk_admit=lambda final_ctx, request=None: None, _prefill_mimo_v26_text=fake_prefill,
    )
    monkeypatch.setattr("vmlx_engine.mllm_batch_generator.get_effective_metal_working_set_bytes",
                        lambda _mx: (90 * 10 ** 9, 107 * 10 ** 9))
    monkeypatch.setattr("vmlx_engine.mllm_batch_generator._infer_attention_heads_for_hybrid_oom_guard",
                        lambda lm: 64)
    monkeypatch.delenv("VMLINUX_TIGHT_MEMORY_PREFILL_STEP_SIZE", raising=False)
    request = MLLMBatchRequest(uid=1, request_id="mimo-text", prompt="p")
    request.input_ids = mx.zeros((1, 32000), dtype=mx.int32)
    request.extra_kwargs = {}
    request._original_token_ids = list(range(32000))
    MLLMBatchGenerator._run_vision_encoding_inner(fake_self, request, cache=[object()])
    return seen["step"]


def test_declared_fused_attention_runs_the_configured_step(monkeypatch):
    assert _run(monkeypatch, False) == 2048


@pytest.mark.parametrize("declared", [None, True])
def test_undeclared_or_materialized_keeps_the_projection(monkeypatch, declared):
    step = _run(monkeypatch, declared)
    assert 64 <= step < 2048
