"""Per-arch model paths and metadata for the thinking-template render audit.

Each ThinkingTemplateModel describes one local bundle the audit can probe:

- arch_name:           Human-readable label, used as the pytest test ID.
- family:              vmlx_engine.model_configs family_name.
- model_path:          Absolute path on local disk. Test is skipped if missing.
- think_in_template:   Mirror of vmlx_engine.model_configs[family].think_in_template.
                       True  = template honors enable_thinking natively.
                       False = template does not; engine previously injected
                               an empty <think></think> at the prompt tail.
- sample_user_message: Plain user content used as the test prompt.

The 94b16d22 commit removed the engine-side empty <think></think> inject from
vmlx_engine/engine/{batched,simple}.py (4 sites). The audit verifies that
removal does not regress enable_thinking=False honor for any arch — and if it
does, the fix is a per-bundle chat_template.jinja patch, not engine-side
re-inject (per spec §8.3, §17.3, working principle "no guards or
monkeypatches").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ThinkingTemplateModel:
    arch_name: str
    family: str
    model_path: Path
    think_in_template: bool
    sample_user_message: str


_SAMPLE = "What is 2+2?"


# At-risk archs: think_in_template=False in vmlx_engine.model_configs.
# These templates historically did NOT emit <think>...</think> themselves;
# the engine injected an empty pair at prompt tail. Inject was removed in
# 94b16d22 — these are the regression candidates.
AT_RISK_MODELS: List[ThinkingTemplateModel] = [
    ThinkingTemplateModel(
        arch_name="ling-2.6-flash-jangtq2",
        family="bailing_hybrid",
        model_path=Path("/Users/eric/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK"),
        think_in_template=False,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="nemotron-omni-nano-jangtq",
        family="nemotron_h",
        model_path=Path("/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK"),
        think_in_template=False,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="gemma-4-26b-jang4m",
        family="gemma4",
        model_path=Path("/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK"),
        think_in_template=False,
        sample_user_message=_SAMPLE,
    ),
]


# Sanity archs: think_in_template=True. The bundle's chat template is
# expected to handle enable_thinking natively, so removal of the engine
# inject should be a no-op. We Layer-1 verify; Layer-2 not required.
SANITY_MODELS: List[ThinkingTemplateModel] = [
    ThinkingTemplateModel(
        arch_name="dsv4-flash-jangtq",
        family="deepseek_v4",
        model_path=Path("/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="minimax-m2.7-jangtq-k",
        family="minimax_m2",
        model_path=Path("/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="qwen3.6-27b-jang4m",
        family="qwen3_5",
        model_path=Path("/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="qwen3.6-35b-a3b-jangtq",
        family="qwen3_5_moe",
        model_path=Path("/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
    ThinkingTemplateModel(
        arch_name="kimi-k2.6-small-jangtq",
        family="kimi_k25",
        model_path=Path("/Users/eric/models/JANGQ/Kimi-K2.6-Small-JANGTQ"),
        think_in_template=True,
        sample_user_message=_SAMPLE,
    ),
]


ALL_MODELS: List[ThinkingTemplateModel] = AT_RISK_MODELS + SANITY_MODELS


def available_models() -> List[ThinkingTemplateModel]:
    """Return the subset of ALL_MODELS whose model_path exists on disk."""
    return [m for m in ALL_MODELS if m.model_path.is_dir()]
