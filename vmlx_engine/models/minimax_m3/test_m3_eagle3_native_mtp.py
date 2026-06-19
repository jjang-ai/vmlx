"""Regression gates for MiniMax-M3 EAGLE3 native-MTP scaffolding."""

import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minimax_m3 as M
from cache import truncate_minimax_m3_cache


def _tiny_args():
    return M.ModelArgs(
        hidden_size=32,
        num_hidden_layers=4,
        intermediate_size=16,
        dense_intermediate_size=64,
        shared_intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rotary_dim=4,
        vocab_size=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=[0, 0, 0, 0],
        sparse_attention_freq=[0, 0, 0, 0],
    )


def main():
    model = M.Model(_tiny_args())
    ids = mx.array([[1, 2, 3]])

    logits = model(ids)
    logits_aux, aux = model(ids, return_aux=True, aux_layers=(0, 2))
    mx.eval(logits, logits_aux, *aux)

    assert logits.shape == (1, 3, 64), logits.shape
    assert logits_aux.shape == logits.shape, logits_aux.shape
    assert len(aux) == 2, len(aux)
    assert aux[0].shape == (1, 3, 32), aux[0].shape
    assert aux[1].shape == (1, 3, 32), aux[1].shape

    cache = model.make_cache()
    _ = model(ids, cache=cache)
    mx.eval(_)
    assert cache[0].offset == 3, cache[0].offset
    truncate_minimax_m3_cache(cache, 1)
    assert cache[0].offset == 1, cache[0].offset

    print("OK: MiniMax-M3 EAGLE3 aux taps and cache truncation scaffolding hold")


if __name__ == "__main__":
    main()
