"""Regression gate: MiniMax-M3 must NEVER have its custom MSA make_cache replaced
by the JANG loader's flat TurboQuantKVCache patch.

M3's MiniMaxM3SparseCache carries an append-only Lightning-Indexer key buffer
(idx_keys) that sparse-attention block selection is recomputed from each step.
A flat TurboQuantKVCache has no idx_keys lane, so swapping make_cache() would
silently drop the indexer state -> broken block selection / garbage output.
This pins the explicit M3 skip added to jang_loader._patch_turboquant_make_cache
(mirrors the MLA CacheList skip). M3 is positional/append-only but structurally
TQ-incompatible, like DSV4/ZAYA have their own typed caches.
"""
import os, sys
sys.path.insert(0, "/Users/example/mlx/vllm-mlx")
os.environ.pop("VMLX_DISABLE_TQ_KV", None)

from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache
from vmlx_engine.utils.model_inspector import is_mla_model

SENTINEL = ["NATIVE_MSA_CACHE"]
TQ_ENABLED = {"turboquant": {"enabled": True, "default_key_bits": 3,
              "default_value_bits": 3, "critical_key_bits": 4, "critical_value_bits": 4,
              "critical_layers": [0, 1, 2, -3, -2, -1], "seed": 42}}


class _FakeM3:
    def __init__(self):
        self.layers = [object()] * 32
    def make_cache(self):
        return SENTINEL


def _assert_intact(model_type, text_type):
    cfg = {"model_type": model_type,
           "text_config": {"model_type": text_type, "num_hidden_layers": 32}}
    # MLA skip must NOT be the reason M3 is protected — prove is_mla_model is False.
    assert is_mla_model(cfg) is False, f"{model_type} unexpectedly detected as MLA"
    m = _FakeM3()
    _patch_turboquant_make_cache(m, TQ_ENABLED, cfg)
    assert m.make_cache() is SENTINEL, (
        f"M3 ({model_type}/{text_type}) make_cache was CLOBBERED by TQ patch — "
        "idx_keys MSA cache would be destroyed")
    print(f"PASS: M3 ({model_type}/{text_type}) MSA make_cache intact with TQ enabled")


# Every model_type spelling M3 ships under (root VL wrapper + inner LM + missing text).
_assert_intact("minimax_m3_vl", "minimax_m3")
_assert_intact("minimax_m3", "minimax_m3")
_assert_intact("minimax_m3_vl", "")

print("OK: MiniMax-M3 TurboQuant make_cache skip holds (MSA/idx_keys preserved)")
