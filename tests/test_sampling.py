import mlx.core as mx

from vmlx_engine.sampling import make_sampler


def test_compact_top_k_sampler_accepts_logits_and_returns_top_k_token():
    sampler = make_sampler(temp=1.0, top_p=1.0, top_k=3)
    assert getattr(sampler, "_vmlx_accepts_logits", False) is True

    logits = mx.array([[0.1, 5.0, -2.0, 4.0, 3.0, -1.0]], dtype=mx.float32)
    seen = set()
    for _ in range(32):
        token = int(sampler(logits).item())
        seen.add(token)
        assert token in {1, 3, 4}
    assert seen


def test_argmax_sampler_accepts_logits():
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0)
    assert getattr(sampler, "_vmlx_accepts_logits", False) is True
    token = int(sampler(mx.array([[0.1, 0.2, 9.0]], dtype=mx.float32)).item())
    assert token == 2
