# SPDX-License-Identifier: Apache-2.0
"""Family-neutral runtime-acceleration capability and attestation contract.

This module deliberately has no MLX imports.  It describes what a family can
request and merges that configuration with model-instance attestations recorded
by the owning loader.  A requested environment flag is never reported as an
installed or exercised kernel without a runtime attestation.
"""

from __future__ import annotations

import os
from typing import Any, Iterable


SCHEMA = "vmlx-runtime-acceleration-v1"
_FALSE_VALUES = {"", "0", "false", "no", "off", "disabled", "none"}


_FAMILY_ALIASES = {
    "qwen4_exp": "qwen4_exp",
    "qwen4_exp_text": "qwen4_exp",
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_5_moe_text": "qwen3_5_moe",
    "glm5_next": "glm5_next",
    "deepseek_v4": "deepseek_v4",
    "ernie4_5_moe": "ernie4_5",
    "ernie4_5": "ernie4_5",
}


def _feature(
    feature_id: str,
    *,
    label: str,
    kind: str,
    scopes: tuple[str, ...],
    default: bool | str,
    env: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": feature_id,
        "label": label,
        "kind": kind,
        "scopes": scopes,
        "default": default,
        "env": env,
    }


_FAMILIES: dict[str, dict[str, Any]] = {
    "qwen4_exp": {
        "display_family": "Qwen3.8-Flash-Next",
        "native_state": ["QSA", "GDN", "PLE", "n-gram", "MoE"],
        "features": [
            _feature(
                "projection_groups",
                label="exact quantized projection grouping",
                kind="load_time_graph",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "hyper_connection_fusion",
                label="hyper-connection projection folding and compilation",
                kind="load_time_graph",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "affine_moe",
                label="full affine MoE decode fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN4_AFFINE_MOE", "VMLINUX_QWEN4_AFFINE_MOE"),
            ),
            _feature(
                "affine_moe_pair",
                label="affine MoE gate/up and weighted-down fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
                env=("VMLX_QWEN4_FUSED_MOE_PAIR",),
            ),
            _feature(
                "gdn_conv_state",
                label="GDN convolution and state-shift fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN4_FUSED_GDN_CONV",),
            ),
            _feature(
                "ple_conv_state",
                label="PLE dilated-convolution and state fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN4_FUSED_PLE_CONV",),
            ),
            _feature(
                "qsa_sparse_score",
                label="QSA sparse-index score fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN4_FUSED_QSA_SCORE",),
            ),
            _feature(
                "gated_rmsnorm",
                label="small-row gated RMSNorm fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_FUSED_GATED_RMSNORM",),
            ),
            _feature(
                "ple_parallel_read",
                label="parallel SSD PLE row reads",
                kind="storage_io",
                scopes=("prefill", "ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN4_PLE_PARALLEL_READ",),
            ),
        ],
    },
    "qwen3_5": {
        "display_family": "Qwen3.8-27B / Qwen3.5 hybrid",
        "native_state": ["attention", "GDN", "SSM companion state"],
        "features": [
            _feature(
                "projection_groups",
                label="exact GDN decode projection grouping",
                kind="load_time_graph",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "vendored_hybrid_runtime",
                label="vMLX hybrid GDN/attention runtime",
                kind="runtime",
                scopes=("prefill", "ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "gdn_conv_state",
                label="GDN K4 convolution and state-shift fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN35_FUSED_GDN_CONV",),
            ),
            _feature(
                "gdn_gate_terms",
                label="joint GDN beta and decay preparation",
                kind="mlx_compiled_graph",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN35_FUSED_GDN_GATES",),
            ),
            _feature(
                "mtp_verify_qmm",
                label="four-row native-MTP verifier QMM",
                kind="metal_kernel",
                scopes=("mtp_verify",),
                default=False,
                env=("DFLASH_VERIFY_QMM",),
            ),
        ],
    },
    "qwen3_5_moe": {
        "display_family": "Qwen3.5 MoE hybrid",
        "native_state": ["attention", "GDN", "SSM companion state", "MoE"],
        "features": [
            _feature(
                "projection_groups",
                label="exact GDN decode projection grouping",
                kind="load_time_graph",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "vendored_hybrid_runtime",
                label="vMLX hybrid GDN/attention/MoE runtime",
                kind="runtime",
                scopes=("prefill", "ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "gdn_conv_state",
                label="GDN K4 convolution and state-shift fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN35_FUSED_GDN_CONV",),
            ),
            _feature(
                "gdn_gate_terms",
                label="joint GDN beta and decay preparation",
                kind="mlx_compiled_graph",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_QWEN35_FUSED_GDN_GATES",),
            ),
            _feature(
                "mtp_verify_qmm",
                label="four-row native-MTP verifier QMM",
                kind="metal_kernel",
                scopes=("mtp_verify",),
                default=False,
                env=("DFLASH_VERIFY_QMM",),
            ),
        ],
    },
    "glm5_next": {
        "display_family": "GLM-5.3-Flash",
        "native_state": ["KDA", "DSA", "mHC", "MoE"],
        "features": [
            _feature(
                "projection_groups",
                label="exact KDA and dense projection grouping",
                kind="load_time_graph",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "startup_warmup",
                label="request-independent first-forward warmup",
                kind="load_time_graph",
                scopes=("model_load",),
                default=True,
            ),
            _feature(
                "dsa_pool_cache",
                label="typed completed DSA pool-key reuse",
                kind="state_algorithm",
                scopes=("prefill", "ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "vectorized_kda_verify",
                label="batched KDA speculative verification with exact rollback",
                kind="state_algorithm",
                scopes=("mtp_verify",),
                default=True,
                env=(
                    "VMLX_GLM5_VECTOR_KDA_VERIFY",
                    "VMLINUX_GLM5_VECTOR_KDA_VERIFY",
                ),
            ),
            _feature(
                "aligned_mtp_head_cache",
                label="verifier-confirmed GLM MTP-head cache alignment",
                kind="state_algorithm",
                scopes=("mtp_decode", "mtp_verify"),
                default=False,
                env=(
                    "VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE",
                    "VMLINUX_GLM5_ALIGNED_MTP_HEAD_CACHE",
                ),
            ),
            _feature(
                "mtp_prompt_priming",
                label="GLM MTP-head prompt-history priming",
                kind="state_algorithm",
                scopes=("prefill", "mtp_decode"),
                default=False,
                env=(
                    "VMLX_GLM5_MTP_PROMPT_PRIMING",
                    "VMLINUX_GLM5_MTP_PROMPT_PRIMING",
                ),
            ),
            _feature(
                "affine_moe_pair",
                label="affine MoE gate/up and weighted-down fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_GLM5_FUSED_MOE_PAIR",),
            ),
            _feature(
                "kda_conv_state",
                label="KDA q/k/v convolution and state fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
                env=("VMLX_GLM5_FUSED_KDA_CONV",),
            ),
            _feature(
                "kda_recurrent_step",
                label="KDA recurrent-state update fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
                env=("VMLX_GLM5_FUSED_KDA_STEP",),
            ),
            _feature(
                "mhc_transform",
                label="mHC decode transform fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
                env=("VMLX_GLM5_FUSED_MHC",),
            ),
            _feature(
                "mhc_verify_transform",
                label="mHC multi-row speculative-verifier transform fusion",
                kind="metal_kernel",
                scopes=("mtp_verify",),
                default=True,
                env=(
                    "VMLX_GLM5_FUSED_MHC_VERIFY",
                    "VMLINUX_GLM5_FUSED_MHC_VERIFY",
                ),
            ),
            _feature(
                "hc_place",
                label="hyper-connection placement fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_GLM5_FUSED_HC_PLACE",),
            ),
            _feature(
                "hc_place_verify",
                label="multi-row speculative-verifier hyper-connection placement fusion",
                kind="metal_kernel",
                scopes=("mtp_verify",),
                default=True,
                env=(
                    "VMLX_GLM5_FUSED_HC_PLACE_VERIFY",
                    "VMLINUX_GLM5_FUSED_HC_PLACE_VERIFY",
                ),
            ),
            _feature(
                "dsa_sparse_score",
                label="DSA sparse-index score fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_GLM5_FUSED_DSA_SCORE",),
            ),
            _feature(
                "dsa_bf16_state",
                label="DSA indexer BF16 state and scoring",
                kind="dtype_policy",
                scopes=("prefill", "ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_GLM5_DSA_BF16",),
            ),
            _feature(
                "gated_rmsnorm",
                label="small-row gated RMSNorm fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_FUSED_GATED_RMSNORM",),
            ),
        ],
    },
    "deepseek_v4": {
        "display_family": "DeepSeek-V4-Flash",
        "native_state": ["DSA", "CSA", "HCA", "MoE"],
        "features": [
            _feature(
                "runtime_patch",
                label="native DSV4 MLA/DSA/mHC runtime",
                kind="runtime",
                scopes=("prefill", "ar_decode", "mtp_decode"),
                default=True,
            ),
            _feature(
                "indexer_scores",
                label="DSA/CSA prefill indexer score reduction",
                kind="metal_kernel",
                scopes=("prefill",),
                default=False,
                env=("VMLX_DSV4_INDEXER_KERNEL",),
            ),
            _feature(
                "fused_moe_pair",
                label="validated pair-SwiGLU MoE decode fusion",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=True,
                env=("VMLX_DSV4_FUSED_MOE_PAIR",),
            ),
            _feature(
                "affine_moe",
                label="affine routed-expert fast path",
                kind="metal_kernel",
                scopes=("ar_decode", "mtp_decode"),
                default=False,
                env=("VMLX_DSV4_AFFINE_MOE_FASTPATH",),
            ),
            _feature(
                "lm_head",
                label="quantized or exact-cached vocabulary head",
                kind="projection",
                scopes=("ar_decode", "mtp_verify"),
                default="qmm",
                env=("VMLX_DSV4_LM_HEAD_MODE",),
            ),
            _feature(
                "rope_cache",
                label="bounded exact RoPE table sharing",
                kind="state_algorithm",
                scopes=("prefill", "ar_decode", "mtp_decode"),
                default=True,
                env=("VMLX_DSV4_ROPE_CACHE",),
            ),
        ],
    },
    "ernie4_5": {
        "display_family": "ERNIE-4.5",
        "native_state": ["MoE"],
        "features": [
            _feature(
                "mtp_prompt_priming",
                label="ERNIE MTP-head prompt-history priming",
                kind="state_algorithm",
                scopes=("prefill", "mtp_decode"),
                default=True,
                # Same order as batch_generator._TEXT_PROMPT_PRIMING_FLAGS so the
                # report and the runtime gate agree when both aliases are set.
                env=(
                    "VMLINUX_ERNIE45_MTP_PROMPT_PRIMING",
                    "VMLX_ERNIE45_MTP_PROMPT_PRIMING",
                ),
            ),
        ],
    },
}


def canonical_acceleration_family(model_type: str | None) -> str | None:
    if not model_type:
        return None
    return _FAMILY_ALIASES.get(str(model_type).strip().lower())


def acceleration_family_from_config(config: dict[str, Any]) -> str | None:
    candidates = [config.get("model_type")]
    for key in ("text_config", "llm_config", "language_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("model_type"))
    for candidate in candidates:
        family = canonical_acceleration_family(candidate)
        if family is not None:
            return family
    return None


def _requested(spec: dict[str, Any]) -> tuple[bool, str | bool, str]:
    default = spec["default"]
    for env_name in spec["env"]:
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name].strip()
        if isinstance(default, str):
            mode = raw.lower()
            return mode not in _FALSE_VALUES, mode, env_name
        return raw.lower() not in _FALSE_VALUES, raw, env_name
    if isinstance(default, str):
        mode = default.lower()
        return mode not in _FALSE_VALUES, mode, "default"
    return bool(default), bool(default), "default"


def _feature_state(
    requested: bool,
    runtime: dict[str, Any] | None,
) -> str:
    if not requested:
        return "disabled"
    if runtime is None:
        return "configured_unattested"
    observed_calls = int(runtime.get("observed_calls", runtime.get("calls", 0)) or 0)
    installed_value = runtime.get("installed")
    installed = (
        installed_value
        if isinstance(installed_value, bool)
        else int(installed_value or 0) > 0
    )
    if installed and observed_calls > 0:
        return "active_observed"
    if installed:
        return "installed_unobserved"
    if runtime.get("reason") or runtime.get("disabled_reason"):
        return "refused_or_fallback"
    return "configured_unattested"


def build_acceleration_contract(
    family: str | None,
    runtime_attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = canonical_acceleration_family(family) or family
    definition = _FAMILIES.get(str(canonical)) if canonical else None
    if definition is None:
        return {
            "schema": SCHEMA,
            "family": canonical,
            "known_family": False,
            "native_state": [],
            "features": [],
            "summary": {
                "requested": 0,
                "installed": 0,
                "observed": 0,
                "source_only_or_unattested": 0,
            },
        }

    runtime_features = (
        runtime_attestation.get("features", {})
        if isinstance(runtime_attestation, dict)
        else {}
    )
    features = []
    for spec in definition["features"]:
        requested, selected, source = _requested(spec)
        runtime = runtime_features.get(spec["id"])
        if not isinstance(runtime, dict):
            runtime = None
        row = {
            "id": spec["id"],
            "label": spec["label"],
            "kind": spec["kind"],
            "scopes": list(spec["scopes"]),
            "default": spec["default"],
            "env": list(spec["env"]),
            "requested": requested,
            "selection": selected,
            "selection_source": source,
            "state": _feature_state(requested, runtime),
        }
        if runtime is not None:
            row["runtime"] = dict(runtime)
        features.append(row)

    return {
        "schema": SCHEMA,
        "family": canonical,
        "display_family": definition["display_family"],
        "known_family": True,
        "native_state": list(definition["native_state"]),
        "features": features,
        "summary": {
            "requested": sum(bool(row["requested"]) for row in features),
            "installed": sum(
                row["state"] in {"installed_unobserved", "active_observed"}
                for row in features
            ),
            "observed": sum(row["state"] == "active_observed" for row in features),
            "source_only_or_unattested": sum(
                row["state"] == "configured_unattested" for row in features
            ),
        },
    }


def record_acceleration_attestation(
    model: Any,
    family: str,
    features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge loader-owned facts onto one model instance without executing it."""

    existing = getattr(model, "_vmlx_acceleration_attestation", None)
    current_features: dict[str, dict[str, Any]] = {}
    if isinstance(existing, dict) and isinstance(existing.get("features"), dict):
        current_features.update(
            {
                key: dict(value)
                for key, value in existing["features"].items()
                if isinstance(value, dict)
            }
        )
    for key, value in features.items():
        prior = current_features.get(key, {})
        current_features[key] = {**prior, **dict(value)}
    attestation = {
        "schema": SCHEMA,
        "family": canonical_acceleration_family(family) or family,
        "features": current_features,
    }
    try:
        setattr(model, "_vmlx_acceleration_attestation", attestation)
    except (AttributeError, TypeError):
        # Status recording is observability-only.  Opaque loader wrappers and
        # test doubles may intentionally reject new attributes; never make
        # model loading depend on the telemetry carrier.
        pass
    return attestation


def find_acceleration_attestation(
    candidates: Iterable[Any],
) -> dict[str, Any] | None:
    for candidate in candidates:
        value = getattr(candidate, "_vmlx_acceleration_attestation", None)
        if isinstance(value, dict) and value.get("schema") == SCHEMA:
            return {
                **value,
                "features": {
                    key: dict(item)
                    for key, item in value.get("features", {}).items()
                    if isinstance(item, dict)
                },
            }
    return None


__all__ = [
    "SCHEMA",
    "acceleration_family_from_config",
    "build_acceleration_contract",
    "canonical_acceleration_family",
    "find_acceleration_attestation",
    "record_acceleration_attestation",
]
