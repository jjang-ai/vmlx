"""DeepSeek-V4.1 CLI policy and DSV4 generator sniff (CPU only, no model)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from vmlx_engine import cli
from vmlx_engine.scheduler import Scheduler


def _cls(name, module):
    return type(name, (), {"__module__": module})


def test_dsv4_sniff_matches_v4_modules_never_v41():
    assert Scheduler._is_dsv4_model_class(_cls("Model", "jang_tools.dsv4.mlx_model"))
    assert Scheduler._is_dsv4_model_class(_cls("Model", "mlx_lm.models.deepseek_v4"))  # was missed
    assert Scheduler._is_dsv4_model_class(_cls("DeepseekV4Model", "x.y"))
    assert not Scheduler._is_dsv4_model_class(_cls("Model", "jang_tools.dsv4_1.mlx_model"))
    assert not Scheduler._is_dsv4_model_class(_cls("Model", "mlx_lm.models.deepseek_v41"))
    assert not Scheduler._is_dsv4_model_class(_cls("Model", "mlx_lm.models.llama"))


def test_dsv41_policy_clamps_single_sequence_and_unsupported_flags():
    args = SimpleNamespace(continuous_batching=False, max_num_seqs=8, kv_cache_quantization="q4",
                           enable_jit=True, smelt=True, flash_moe=False, distributed=False,
                           speculative_model="draft")
    ok, changes = cli._apply_dsv41_runtime_policy(args, logging.getLogger("t"))
    assert ok
    assert args.max_num_seqs == 1 and args.continuous_batching is True
    assert args.kv_cache_quantization == "none" and args.enable_jit is False and args.smelt is False
    assert args.speculative_model is None
    assert "max_num_seqs=8->1" in changes
    assert cli._SLOW_FAMILY_TIMEOUTS["deepseek_v4_1"] == 900
