import math

import mlx.core as mx

from vmlx_engine.native_mtp_acceptance import accept_lp_for
from vmlx_engine.sampling import make_sampler


def test_compact_top_k_sampler_accepts_logits_and_returns_top_k_token():
    sampler = make_sampler(temp=1.0, top_p=1.0, top_k=3)
    assert getattr(sampler, "_vmlx_accepts_logits", False) is True
    assert getattr(sampler, "_vmlx_is_greedy", False) is False
    assert sampler.temp == 1.0
    assert sampler.top_p == 1.0
    assert sampler.min_p == 0.0
    assert sampler.top_k == 3

    logits = mx.array([[0.1, 5.0, -2.0, 4.0, 3.0, -1.0]], dtype=mx.float32)
    seen = set()
    for _ in range(32):
        token = int(sampler(logits).item())
        seen.add(token)
        assert token in {1, 3, 4}
    assert seen


def test_compact_top_k_publishes_its_exact_acceptance_distribution():
    sampler = make_sampler(temp=0.5, top_p=0.80, top_k=3)
    logits = mx.array([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    accepted_lp = accept_lp_for(sampler, logprobs)
    mx.eval(accepted_lp)

    probabilities = mx.exp(accepted_lp)
    assert bool(mx.allclose(probabilities.sum(), mx.array(1.0)))
    # top-k excludes ids 3/4; top-p at this distribution keeps ids 0/1.
    assert math.isinf(float(accepted_lp[0, 2]))
    assert math.isinf(float(accepted_lp[0, 3]))
    assert math.isinf(float(accepted_lp[0, 4]))


def test_argmax_sampler_accepts_logits():
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0)
    assert getattr(sampler, "_vmlx_accepts_logits", False) is True
    assert getattr(sampler, "_vmlx_is_greedy", False) is True
    token = int(sampler(mx.array([[0.1, 0.2, 9.0]], dtype=mx.float32)).item())
    assert token == 2


def test_seeded_argmax_sampler_keeps_mtp_on_greedy_identity_path():
    from types import SimpleNamespace

    from vmlx_engine.text_mtp.lane import batch_is_greedy

    sampler = make_sampler(temp=0.0, seed=731)
    assert getattr(sampler, "_vmlx_is_greedy", False) is True
    batch = SimpleNamespace(samplers=[sampler], fallback_sampler=None)
    assert batch_is_greedy(batch) is True
