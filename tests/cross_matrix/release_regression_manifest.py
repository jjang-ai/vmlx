"""Release-regression manifest for current vMLX Python/Electron work.

This is the durable index for the broad regression suite Eric requested. It is
intentionally explicit: each row names the user-visible or runtime contract it
protects, the commands that currently cover it, and whether the row is no-heavy,
live-model, or packaged-app only.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from tests.cross_matrix.run_model_family_detection_contract import (
    REQUIRED_ROWS as EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
)

EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS = (
    "jang_and_jangtq_detection",
    "ling_bailing_hybrid_loader_repairs",
    "mxfp4_detection",
    "mxfp8_detection",
    "plain_mlx_4bit_detection",
    "dropped_mtp_detection",
    "preserved_mtp_detection",
    "cache_profile_detection",
    "not_path_name_only",
)

EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS = (
    "dsv4_native_composite_cache_status",
    "zaya_typed_cca_status",
    "hybrid_ssm_partial_reuse",
    "generic_tq_not_applied_to_hybrid_ssm",
    "turboquant_kv_runtime_contract",
    "turboquant_disk_roundtrip",
    "dsv4_terminal_composite_contracts",
    "prompt_disk_l2_backfill_contracts",
    "cache_detail_telemetry_contracts",
    "panel_cache_launch_policy",
    "legacy_count_floor_still_nontrivial",
)


REQUIRED_RELEASE_DOMAINS = {
    "chat_ui_settings",
    "generation_defaults",
    "parser_registry",
    "reasoning_template",
    "tool_calls",
    "api_surface",
    "cache_architecture",
    "model_artifact_detection",
    "model_family_detection",
    "native_mtp",
    "mcp",
    "vl_media",
    "packaging_release",
    "performance",
    "release_surface",
    "pr_intake",
}

CURRENT_POST_BUDGET_EDGE_ARTIFACTS = {
    "chat-settings-max-output-context-ui": "build/current-max-output-context-contract-20260523-profile-chat-cap.json",
    "panel-session-cache-settings-family-gating": "build/current-panel-settings-contract-proof-20260523-post-budget-edge.json",
    "generation-defaults-no-hidden-forcing": "build/current-generation-defaults-contract-20260523-post-budget-edge.json",
    "parser-registry-tool-reasoning-parity": "build/current-parser-registry-contract-20260523-post-budget-edge.json",
    "reasoning-template-no-think-tag-leak": "build/current-reasoning-template-contract-20260523-post-budget-edge.json",
    "tool-call-loop-parser-cleanup": "build/current-tool-call-contract-20260523-post-budget-edge.json",
    "api-chat-responses-anthropic-ollama-parity": "build/current-api-surface-contract-20260523-post-budget-edge.json",
    "cache-architecture-family-classification": "build/current-cache-architecture-contract-20260523-post-budget-edge.json",
    "model-artifact-format-detection": "build/current-model-artifact-format-contract-20260523-post-budget-edge.json",
    "model-family-detection-noheavy": "build/current-model-family-detection-contract-20260523-post-budget-edge.json",
    "native-mtp-d3-effect-policy": "build/current-native-mtp-contract-20260523-post-budget-edge.json",
    "mcp-policy-ui-gateway": "build/current-mcp-policy-contract-20260523-post-budget-edge.json",
    "vl-media-cache-tool-followup": "build/current-vl-media-cache-contract-20260523-post-budget-edge.json",
    "packaged-release-integrity": "build/current-packaged-integrity-contract-20260523-post-budget-edge-refreshed.json",
    "public-release-surface-preflight": "build/current-release-surface-contract-20260523-post-budget-edge.json",
}

CURRENT_REGRESSION_SUITE_ARTIFACT = (
    "build/current-regression-suite-20260523-profile-chat-cap-clean-jang.json"
)

EXPECTED_CURRENT_OPEN_REQUIREMENTS = [
    "Qwen 27B JANG_4M prompt-processing speed floor is release-cleared",
    "DSV4 long-output/code/file-generation quality is release-cleared",
]


_ROWS: list[dict[str, Any]] = [
    {
        "id": "chat-settings-max-output-context-ui",
        "domain": "chat_ui_settings",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Server Default Max Output Tokens maps to --max-tokens",
            "Max Context Tokens maps to --max-prompt-tokens",
            "Chat Max Output Tokens remains a per-chat/API override for non-streaming and streaming",
            "Legacy /v1/completions max_tokens remains a per-request output cap for non-streaming and streaming",
            "Prompt/context aliases clamp to server max-prompt-tokens without rewriting output caps",
            "Ollama gateway non-stream num_predict remains request-scoped across chat, templated generate, and raw generate",
            "Ollama gateway malformed num_predict values are omitted instead of poisoning max_tokens",
            "Ollama gateway malformed context values are omitted instead of poisoning max_prompt_tokens",
            "External coding-tool configs keep context window and output limit separate",
            "new-chat model-owned maxTokens cannot be replaced by inherited per-chat output caps",
            "server startup maxTokens and chat maxTokens remain independent when both are set",
            "persisted chat maxTokens cannot relaunch server with a new startup maxTokens",
            "per-chat maxTokens below or above the server startup default remain request-scoped",
            "explicit per-request Chat/Responses output caps can be below or above Server Default Max Output Tokens",
            "Explicit Chat/Responses output caps do not mutate the server startup default used by later Auto requests",
            "Streaming Chat/Responses output caps do not mutate the server startup default used by later Auto requests",
            "later Auto Chat/Responses requests still use the original server startup default",
            "Responses maxTokens below or above the server startup default remain request scoped",
            "Responses Auto does not synthesize max_output_tokens",
            "DB-cleared NULL chat maxTokens stays Auto for Chat Completions and Responses",
            "Raw API non-positive max_tokens/max_output_tokens are rejected rather than becoming hidden Auto defaults or empty generations",
            "Auto chat Max Tokens omits per-request output caps so server startup defaults can apply",
            "new chat output caps are not inherited or made sticky on default-profile or same-model clean chat creation",
            "chat reset does not convert model max_new_tokens into a sticky per-chat maxTokens override",
            "string-shaped legacy session maxTokens values are cleared before launch",
            "server startup maxTokens explicitness survives wake/reload and CLI implicit startup paths",
            "API gateway output-budget and context-budget paths are source-hashed",
            "DSV4 request-budget helper is source-hashed with the max-output boundary gate",
            "Casual preset maxTokens is documented as an explicit server output cap and does not change model-owned defaults or context",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-recheck-server-chat-output-compat.json",
        ],
        "artifacts": [
            "build/current-max-output-context-contract-20260523-profile-chat-cap.json",
            "build/current-max-output-context-contract-20260523-dsv4-budget-edge.json",
            "build/current-max-output-context-contract-20260522-recheck-server-chat-output-compat.json",
            "build/current-max-output-context-contract-20260522-persisted-chat-output-cap.json",
            "build/current-max-output-context-contract-20260522-new-chat-output-cap-nonsticky.json",
            "build/current-max-output-context-contract-20260522-chat-auto-server-default.json",
            "build/current-max-output-context-contract-20260522-responses-output-boundary.json",
            "build/current-max-output-context-contract-20260522-request-default-mutation.json",
            "build/current-max-output-context-contract-20260523-streaming-mutation.json",
            "build/current-max-output-context-contract-20260523-ollama-nonstream-caps.json",
            "build/current-max-output-context-contract-20260523-null-chat-cap.json",
            "build/current-max-output-context-contract-20260522-reset-policy-string-legacy.json",
            "build/current-max-output-context-contract-20260522-api-validator-caps.json",
            "build/current-max-output-context-contract-20260522-gateway-dsv4-budget-hash.json",
            "build/current-max-output-context-contract-20260522-casual-server-output-cap.json",
        ],
    },
    {
        "id": "panel-session-cache-settings-family-gating",
        "domain": "cache_architecture",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "DSV4 pool quant and native prefix controls stay DSV4-only",
            "DSV4 native composite prefix cache remains diagnostic opt-in",
            "generic KV quantization is suppressed for DSV4 native composite cache",
            "non-DSV4 JANG/JANGTQ/MXFP cache toggles keep normal prefix/paged/L2/KV semantics",
            "launch-memory admission is warning-only for lazy-mmap JANG/JANGTQ bundles and does not hard-block on macOS cache pressure",
            "What's New update notice is localized across all five locales and mentions MCP, MTP, latest.json, Developer ID, L2 disk cache, and max_tokens",
            "panel typecheck and model-family registry stay green with those controls",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_noheavy_panel_settings_contract.py --out build/current-panel-settings-contract-proof-20260522-recheck-update-notice-i18n.json",
        ],
        "artifacts": [
            "build/current-panel-settings-contract-proof-20260523-post-budget-edge.json",
            "build/current-panel-settings-contract-proof-20260522-launch-memory-warning.json",
            "build/current-panel-settings-contract-proof-20260522-recheck-update-notice-i18n.json",
        ],
    },
    {
        "id": "generation-defaults-no-hidden-forcing",
        "domain": "generation_defaults",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "generation_config.json defaults are surfaced without copying into sticky overrides",
            "jang_config.json sampling defaults are surfaced without hidden sampler forcing",
            "request/API overrides remain explicit and win over startup defaults",
            "bundle max_new_tokens is preserved for omitted request budgets without hidden sampler or repetition floors",
            "server default output cap is not a request ceiling for explicit Chat Completions or Responses requests",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_generation_defaults_contract.py --out build/current-generation-defaults-contract-20260522-recheck-no-hidden-forcing.json",
        ],
        "artifacts": [
            "build/current-generation-defaults-contract-20260523-post-budget-edge.json",
            "build/current-generation-defaults-contract-20260521.json",
            "build/current-generation-defaults-contract-20260522-recheck-no-hidden-forcing.json",
        ],
    },
    {
        "id": "parser-registry-tool-reasoning-parity",
        "domain": "parser_registry",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Panel-emitted tool parser ids are accepted by the engine",
            "Panel-emitted reasoning parser ids are accepted by the engine",
            "CLI accepts every registered reasoning parser emitted by model families",
            "Reasoning parser dropdown covers every parser the panel registry can emit",
            "MiniMax minimax_m2 regression stays covered",
            "Qwen2/Qwen2-VL, Gemma 3, and GLM base stay off reasoning rails",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_parser_registry_contract.py --out build/current-parser-registry-contract-20260522-non-reasoning-boundaries.json",
        ],
        "artifacts": [
            "build/current-parser-registry-contract-20260523-post-budget-edge.json",
            "build/current-parser-registry-contract-20260522-reasoning-dropdown.json",
            "build/current-parser-registry-contract-20260522-non-reasoning-boundaries.json",
        ],
    },
    {
        "id": "reasoning-template-no-think-tag-leak",
        "domain": "reasoning_template",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Reasoning on/off request wiring stays explicit",
            "Server-side reasoning parsers do not leak think tags into visible content",
            "Client streaming fallback does not double-extract reasoning_content",
            "Interleaved reasoning segments render without corrupting token counts",
            "DSV4 requested reasoning rails are preserved and DSV4 tool calls are not forced onto a hidden direct rail",
            "MiniMax and Ling family-specific reasoning boundaries stay explicit without sampling floors",
            "tool follow-up reasoning state resets before subsequent visible content or tool parsing",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_reasoning_template_contract.py --out build/current-reasoning-template-contract-20260522-recheck-parser-rails.json",
        ],
        "artifacts": [
            "build/current-reasoning-template-contract-20260523-post-budget-edge.json",
            "build/current-reasoning-template-contract-20260521.json",
            "build/current-reasoning-template-contract-20260522-recheck-parser-rails.json",
        ],
    },
    {
        "id": "tool-call-loop-parser-cleanup",
        "domain": "tool_calls",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Tool parser residue is rejected instead of executed",
            "DSV4 degraded DSML variants from live default-cache runs are repaired only when schema-valid",
            "live DSV4 write_file DSML degradation is repaired schema-safely",
            "maxToolIterations caps tool loops",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_tool_call_contract.py --out build/current-tool-call-contract-20260522-dsv4-live-write-file-repair.json",
        ],
        "artifacts": [
            "build/current-tool-call-contract-20260523-post-budget-edge.json",
            "build/current-tool-call-contract-20260521.json",
            "build/current-tool-call-contract-20260522-dsv4-live-write-file-repair.json",
            "build/current-dsv4-default-cache-tool-loop/result.json",
            "build/current-dsv4-default-cache-tool-loop-poolon-materialized-parserfix-20260522/result.json",
        ],
    },
    {
        "id": "api-chat-responses-anthropic-ollama-parity",
        "domain": "api_surface",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "OpenAI Chat Completions sampling/default propagation for non-streaming and streaming",
            "OpenAI Responses sampling/default propagation for non-streaming and streaming",
            "OpenAI legacy Completions max_tokens/default propagation for non-streaming and streaming",
            "Anthropic adapter bundle defaults and streaming max_tokens override semantics",
            "Prompt/context alias clamp semantics across OpenAI, Anthropic, and Ollama-compatible surfaces",
            "Ollama adapter streaming/done behavior, streaming num_predict output-cap overrides, malformed num_predict omission, and malformed context omission",
            "Auto chat Max Tokens omits per-request output caps so server startup defaults can apply on Chat Completions and Responses",
            "Chat Completions and Responses streaming usage preserve cached_tokens plus cache_detail through finish chunks",
            "server cache and DSV4 tool/parser surfaces stay named in the API surface contract",
            "panel request builders omit invalid persisted maxTokens instead of poisoning Chat Completions or Responses",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_api_surface_contract.py --out build/current-api-surface-contract-20260522-recheck-endpoint-assembly.json",
        ],
        "artifacts": [
            "build/current-api-surface-contract-20260523-post-budget-edge.json",
            "build/current-api-surface-contract-20260522-stream-cache-detail.json",
            "build/current-api-surface-contract-20260522-recheck-endpoint-assembly.json",
        ],
    },
    {
        "id": "cache-architecture-family-classification",
        "domain": "cache_architecture",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Plain KV, hybrid SSM/Mamba, MLA, DSV4 composite, ZAYA CCA, and MLLM media-salt cache contracts stay classified per family",
            "Generic TurboQuant KV is not applied to DSV4 native composite or hybrid SSM paths",
            "Bundled JANG DSV4 pool quant codec appends only newly generated CSA/HCA pool rows instead of requantizing the whole accumulated pool",
            "DSV4 pool quant reads reuse a materialized pool view instead of dequantizing and concatenating historical CSA/HCA pool segments on every read",
            "DSV4 short prompts skip synchronous composite prompt snapshots and do not replace that saved time with a synchronous prompt-only re-prefill store",
            "DSV4 panel env mapping enables pool quant only from explicit DSV4 config while the default remains off",
            "DSV4 timing probe covers prefix-cache replay and cold-store boundaries before speed/cache root-cause claims",
            "Cache detail telemetry reports paged, typed native, and TQ/L2 state",
            "Panel session launch builder preserves DSV4 default and diagnostic prefix-cache policy, DSV4-only native cache controls, Qwen3.6 hybrid and Mamba paged-cache forcing, and regular KV stale saved false semantics",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_cache_architecture_contract.py --out build/current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json",
        ],
        "artifacts": [
            "build/current-cache-architecture-contract-20260523-post-budget-edge.json",
            "build/current-cache-architecture-contract-20260521.json",
            "build/current-cache-architecture-contract-20260522-panel-cache-launch.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-quant-append.json",
            "build/current-cache-architecture-contract-20260522-dsv4-timing.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-env-gate.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-materialized-cache.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json",
            "build/current-cache-architecture-contract-20260522-dsv4-short-snapshot-threshold.json",
            "build/current-dsv4-prefix-enabled-two-turn-threshold-live-20260522.json",
        ],
    },
    {
        "id": "model-artifact-format-detection",
        "domain": "model_artifact_detection",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "JANG, JANGTQ/MXTQ, Ling/Bailing hybrid loader repairs, plain MLX 4bit, MXFP4, MXFP8, dropped-MTP, and preserved-MTP artifacts are not inferred from names alone",
            "generic affine JANG loader accepts weight_format=affine bundles without misclassifying them as JANGTQ/MXFP/plain MLX",
            "MXFP4/MXFP8 VLM loader paths quantize with the declared quantization mode instead of falling back to affine or JANGTQ handling",
            "JANGTQ_K mixed routed bits use the down-projection bit width for gather_dn",
            "Ling/Bailing flat 2D switch_mlp repair and correct 3D no-op stay covered",
            "Qwen plain MLX 4bit stays distinct from JANG, JANGTQ/MXTQ, MXFP4, and MXFP8",
            "native-MTP detection uses real indexed weights rather than path names",
            "family-specific loader wrappers for JANGTQ text, JANGTQ VLM, Kimi VLM, DSV4, ZAYA, Laguna, and Mistral stay source-hashed in the artifact gate",
            "Registry/family detection uses bundle config and capability metadata",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_model_artifact_format_contract.py --out build/current-model-artifact-format-contract-20260522-recheck-jang-mxfp-mtp-loaders.json",
        ],
        "artifacts": [
            "build/current-model-artifact-format-contract-20260523-post-budget-edge.json",
            "build/current-model-artifact-format-contract-20260522-recheck-jang-mxfp-mtp-loaders.json",
            "build/current-model-artifact-format-contract-20260522-affine-jang-loader.json",
            "build/current-model-artifact-format-contract-20260522-mxfp-vlm-loader.json",
            "build/current-model-artifact-format-contract-20260522-loader-wrapper-hashes.json",
        ],
    },
    {
        "id": "model-family-detection-noheavy",
        "domain": "model_family_detection",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Named DSV4, ZAYA, ZAYA1-VL, Ling/Bailing, Nemotron, Qwen 3.6 VL/video/hybrid, plain MLX 4bit, MXFP4, MXFP8, native-MTP, MiniMax, and Hy3 rows keep expected parser/cache/modality policy",
            "Qwen dense linear-attention wrappers stay hybrid cache through the engine registry, not panel-only matching",
            "Qwen MoE text linear-attention wrappers stay hybrid cache through the engine registry, not panel-only matching",
            "Qwen 3.6 release rows intentionally keep qwen3_5/qwen3.5 family aliases so VL/video/hybrid, MXFP, JANG, JANGTQ, and parser policy stay aligned",
            "base Nemotron-H registry rows stay hybrid cache before stale Omni sidecar overrides are considered",
            "stale ZAYA converter stamps cannot disable reasoning, swap the qwen3 parser, or reenable think_in_template",
            "ZAYA1-VL JANGTQ_K/JANGTQ2/JANGTQ4 qwen3 reasoning rails preserve VL and typed CCA detection",
            "Hy3 JANGTQ_K Low/High reasoning contract stays on Hunyuan tools with qwen3 reasoning",
            "affine-JANG Qwen native-MTP VL/video artifacts stay multimodal when indexed MTP and vision tensors exist",
            "Decode-speed rows keep JANG-only, JANGTQ/MXTQ, plain MLX 4bit, MXFP4, and MXFP8 speed thresholds distinct while staying aligned with engine registry parser, modality, and cache metadata for existing local models",
            "JANG-only MX matmul rows stay text-only launch rows without --is-mllm, startup --max-tokens, MXFP, JANGTQ, or forced JANGTQ environment knobs",
            "Decode-speed rows keep DSV4 native composite separate from generic JANGTQ/MXTQ rows",
            "Decode-speed health checks reject DSV4 native or ZAYA typed cache health for plain KV JANG/JANGTQ/MXFP rows",
            "Every decode-speed row with a declared tool or reasoning parser uses a registered engine parser and CLI-accepted parser choice even when that local model path is absent",
            "Decode-speed launch commands preserve row parser/modality policy and strip source-path or forced JANGTQ acceleration environment overrides",
            "Panel session launch builder preserves MiniMax minimax_m2 parser launch, Qwen3.6 hybrid cache forces paged cache, ZAYA qwen3 reasoning parser and model-owned no-thinking defaults, DSV4 stale cache/additionalArgs suppression, and native-MTP D3 launch policy",
            "engine, panel, and session launch wiring all run in the same family launch-policy matrix",
            "ZAYA, ZAYA1-VL, Ling/Bailing, Nemotron-H, Qwen 3.6, MiniMax, Hy3, and DSV4 rows keep aligned parser/cache/modality policy",
            "session launch strips stale DSV4 additionalArgs and preserves native-MTP D3 only where supported",
            "Decode-speed matrix includes large external Mistral JANGTQ, Mistral MXFP4, and GPT-OSS rows with parser/modality launch policy pinned",
            "Decode-speed matrix includes external Nemotron 3 JANGTQ2 and MXFP4 rows with Nemotron parser/reasoning launch policy pinned",
            "Existing local high-risk DSV4, Qwen JANG/JANGTQ/MXFP/4bit/MTP, Hy3, and Nemotron Omni/Nano JANGTQ/MXFP rows match the current engine registry parser, cache, and modality policy without loading weights",
            "Panel detection matches the same current local high-risk paths, including native-MTP Qwen affine-JANG VL routing, so UI launch policy does not diverge from engine/API routing",
            "This is source/static compatibility proof only; live multi-turn output quality remains a separate live row",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_model_family_detection_contract.py --out build/current-model-family-detection-contract-20260522-recheck-launch-policy-matrix.json",
        ],
        "artifacts": [
            "build/current-model-family-detection-contract-20260523-post-budget-edge.json",
            "build/current-model-family-detection-contract-20260522-recheck-launch-policy-matrix.json",
            "build/current-model-family-detection-contract-20260522-jang-only-mx-matmul-policy.json",
            "build/current-model-family-detection-contract-20260522-qwen-nemotron-hybrid-cache.json",
            "build/current-model-family-detection-contract-20260522-zaya-stale-stamp.json",
            "build/current-model-family-detection-contract-20260522-plain-kv-cache-health.json",
            "build/current-model-family-detection-contract-20260522-panel-launch-wiring.json",
            "build/current-model-family-detection-contract-20260522-zaya-hy3-qwen-vl-profile-rows.json",
            "build/current-model-family-detection-contract-20260522-local-artifact-registry.json",
            "build/current-model-family-detection-contract-20260522-panel-local-paths.json",
            "build/current-model-family-detection-contract-20260522-qwen36-alias.json",
        ],
    },
    {
        "id": "native-mtp-d3-effect-policy",
        "domain": "native_mtp",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Native MTP D3 launch policy is wired where supported",
            "Native MTP controls are hidden/suppressed for DSV4 and unsupported families",
            "DSV4 stale additionalArgs cannot reenable Native MTP, deterministic MTP sampling, hidden default sampling, or startup max-token overrides",
            "Config-only MTP bundles without indexed mtp.* tensors do not activate native MTP or expose Native MTP launch controls",
            "Runtime-active MTP still requires live equivalence/speed rows before release claims",
            "Qwen native-MTP live decode speed and output equivalence are proven by same-artifact AR-vs-MTP A/B",
            "Qwen prompt-processing floor remains a separate performance row; deterministic MTP decode alone is not enough",
            "The same-artifact AR-vs-MTP live A/B proves MTP decode is not the MLLM prefill bottleneck; native MTP D3 is faster than the AR baseline with identical output",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_native_mtp_contract.py --out build/current-native-mtp-contract-20260522-dsv4-additional-args.json",
            ".venv/bin/python bench/native_mtp_speed_ab.py /Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP --served-name qwen27-jang4m-mtp-ab --port 8798 --cache off --max-num-seqs 1 --max-tokens 320 --repeats 1 --warmup 0 --load-timeout-s 420 --out build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-20260523 --disable-prompt-reuse",
        ],
        "artifacts": [
            "build/current-native-mtp-contract-20260523-post-budget-edge.json",
            "build/current-native-mtp-contract-20260522-dsv4-additional-args.json",
            "build/current-native-mtp-contract-20260522-config-only.json",
            "build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-20260523/result.json",
        ],
    },
    {
        "id": "mcp-policy-ui-gateway",
        "domain": "mcp",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "MCP autodiscovery, redaction, security policy, gateway routing, and panel config source stay covered",
            "MCP required marker checks fail the gate if named engine or panel rows disappear",
            "mcp.json/jsonc import keeps secrets out of managed metadata",
            "built-in Electron tools remain separate from MCP execution and request policy",
            "ambiguous multi-session MCP requests are rejected unless a model alias is explicit",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_mcp_policy_contract.py --out build/current-mcp-policy-contract-20260522-recheck-ui-gateway.json",
        ],
        "artifacts": [
            "build/current-mcp-policy-contract-20260523-post-budget-edge.json",
            "build/current-mcp-policy-contract-20260521.json",
            "build/current-mcp-policy-contract-20260522-recheck-ui-gateway.json",
        ],
    },
    {
        "id": "vl-media-cache-tool-followup",
        "domain": "vl_media",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "VLM media request serialization, media cache salting, and tool follow-up paths stay source-covered",
            "Panel family detection keeps ZAYA-VL, Qwen VL/video/hybrid/indexed-MTP, MXFP4/MXFP8 VLM, and Nemotron stale-Omni sidecar routing covered",
            "Engine family detection keeps affine-JANG Qwen VLM-looking artifacts on the text loader until the M-RoPE path is fixed",
            "Qwen3.6 VL JANG indexed-MTP artifacts stay multimodal only when indexed MTP and vision tensors exist",
            "MXTQ/JANGTQ and MXFP4/MXFP8 Qwen VLM rows preserve multimodal launch policy",
            "hybrid SSM companion cache, deferred rederive, and generic TQ suppression stay covered for VLM paths",
            "read_video and image/video tool-result follow-up build real multimodal content parts without leaking media bytes into plain tool text",
            "Still-image live rows do not imply video/audio/Omni clearance",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_vl_media_cache_contract.py --out build/current-vl-media-cache-contract-20260522-recheck-qwen-zaya-media.json",
        ],
        "artifacts": [
            "build/current-vl-media-cache-contract-20260523-post-budget-edge.json",
            "build/current-vl-media-cache-contract-20260522-panel-family.json",
            "build/current-vl-media-cache-contract-20260522-recheck-qwen-zaya-media.json",
        ],
    },
    {
        "id": "packaged-release-integrity",
        "domain": "packaging_release",
        "mode": "packaged",
        "heavy": False,
        "proves": [
            "Version triples, bundled Python hash parity, packaged app signature, and objective proof digest gate release attempts",
            "Direct release-gate runs refresh and enforce the objective proof digest instead of relying only on the umbrella suite",
            "verify-bundled checks packaged Python imports, source hash parity, and relocatable console scripts",
            "packaged integrity uses a clean JANG source path for bundled jang_tools source-hash checks",
            "bundled critical jang_tools files match source content and console-script shebangs are relocatable",
            "dry release gate fails only on the known DSV4 objective row while version, typecheck, bundled import, and digest refresh steps pass",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_packaged_integrity_contract.py --jang-tools-source /Users/example/jang/.worktrees/vmlx-release-clean-b5f66a7/jang-tools --out build/current-packaged-integrity-contract-20260522-recheck-bundled-release-gate.json",
        ],
        "artifacts": [
            "build/current-packaged-integrity-contract-20260523-post-budget-edge-refreshed.json",
            "build/current-packaged-integrity-contract-20260522-recheck-bundled-release-gate.json",
            "build/current-packaged-integrity-contract-20260522-objective-gate-enforced.json",
            "build/current-packaged-integrity-contract-20260521.json",
        ],
    },
    {
        "id": "public-release-surface-preflight",
        "domain": "release_surface",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "local latest.json updater manifest does not advance ahead of the source version",
            "source version consistent across pyproject.toml and panel/package.json",
            "if latest.json is bumped to the source version, it carries a matching URL, SHA256, and notes version",
            "complete post-release updater state is accepted while incomplete bumped latest.json state is rejected",
            "after release, latest.json may equal the source version only when the published updater state is complete",
            "PyPI, GitHub release, and updater feeds remain explicit operator checks before public release",
            "live public release-surface mode checks raw GitHub latest.json, mlx.studio/update/latest.json, PyPI files, and GitHub release DMG asset digest after a release is cut",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-recheck-updater-i18n.json",
            ".venv/bin/python tests/cross_matrix/run_release_surface_contract.py --live-public --out build/current-release-surface-contract-20260522-live-public-v1548.json",
        ],
        "artifacts": [
            "build/current-release-surface-contract-20260523-post-budget-edge.json",
            "build/current-release-surface-contract-20260522-live-public-v1548.json",
            "build/current-release-surface-contract-20260522-post-release-updater.json",
            "build/current-release-surface-contract-20260522-recheck-updater-i18n.json",
            "build/current-release-surface-contract-20260521.json",
            "latest.json",
        ],
    },
    {
        "id": "pr-intake-credit-triage",
        "domain": "pr_intake",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Open PRs are reviewed for merge/adapt/reject decision",
            "Contributor credit is tracked even when an idea is adapted instead of merged",
        ],
        "commands": [
            "gh pr list --repo jjang-ai/vmlx --state open --limit 100 --json number,title,author,createdAt,updatedAt,isDraft,headRefName,baseRefName,mergeable,reviewDecision,url,labels,statusCheckRollup",
        ],
        "artifacts": [
            "docs/internal/pr-triage/2026-05-21-open-pr-intake.md",
            "build/current-gh-prs-vmlx-open-20260521.json",
        ],
    },
    {
        "id": "dsv4-long-output-quality-live",
        "domain": "tool_calls",
        "mode": "live",
        "heavy": True,
        "proves": [
            "DSV4 long-output/code/file-generation exactness and identifier integrity pass on a real model server",
            "This remains open until a source/rebuilt-body or equivalent live proof passes",
        ],
        "commands": [
            "uv run --extra dev python tests/cross_matrix/run_production_family_audit.py --rows dsv4_jang_local --live --out build/current-production-family-audit-live-dsv4-jang-local-20260522.json",
        ],
        "artifacts": [
            "build/current-production-family-audit-live-dsv4-jang-local-20260522-after-stream-cache-detail.json",
            "build/current-production-family-audit-live-dsv4-jang-local-20260522.json",
            "build/current-dsv4-identifier-count-ablation-20260521/result.json",
            "build/current-dsv4-long-output-quality-clearance-20260521.json",
        ],
    },
    {
        "id": "model-family-live-multiturn-soak",
        "domain": "reasoning_template",
        "mode": "live",
        "heavy": True,
        "proves": [
            "Hy3, MiniMax, Qwen 3.6 hybrid, ZAYA, ZAYA1-VL, Ling, Nemotron, and DSV4 family rows need real multi-turn UI/API soaks before broad production claims",
            "Qwen MTP/VL/video rows are no-heavy family-detection covered until dedicated live-audit rows exist",
            "No source-only test may claim output-coherence clearance for every family",
        ],
        "commands": [
            "uv run --extra dev python tests/cross_matrix/run_production_family_audit.py --rows hy3_preview_jangtq2,minimax_m27_tq_k,qwen36_moe_tq4,zaya_jangtq2,zaya_vl_jangtq4,ling_flash_tq,nemotron_omni_tq2,dsv4_jang_local --live --out build/current-production-family-audit-live-multifamily-soak-20260522.json",
        ],
        "artifacts": [
            "build/current-production-family-audit-live-multifamily-soak-20260522.json",
        ],
    },
    {
        "id": "qwen-jang-mx-live-speed-review",
        "domain": "performance",
        "mode": "live",
        "heavy": True,
        "proves": [
            "Qwen3.6-27B JANG_4M live decode speed is measured from the packaged Python engine, not inferred from source-only rows",
            "Qwen/JANG speed artifacts record the MLX and MLX-metal wheel tags so compat-vs-native wheel performance cannot be hidden",
            "selective live TurboQuant for Qwen attention KV layers is compatible only when native cache health proves SSM companion state remains full precision",
            "Native-wheel qwen27_jang4m text-loader clears the PP floor, but native-MTP MLLM/VL prefill remains below the floor on both native and compat wheels",
            "Compat bundled MTP prefill is retained as the user-facing packaged regression diagnostic and must not be hidden behind source-only rows",
            "SingleBatchGenerator honors the explicit prefill keep-alloc CLI/env path so single-sequence JANG runs can test allocator reuse without changing defaults",
            "Qwen 27B prompt-processing floor compares isolated native-MTP/VL prefill against the isolated text-loader baseline instead of blaming MTP decode",
            "MLLM prefill trace shows preprocessing, cache lookup, sampler call, cache submit, SSM capture, and cache merge are not the current PP bottleneck",
            "This row prevents broad JANG/MX matmul speed claims from hiding prompt-processing regressions",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --port 8790 --out build/current-decode-speed-live-qwen27-jang4m-20260522-hybrid-tq-review.json --timeout 420",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --port 8791 --python .venv/bin/python --out build/current-decode-speed-live-qwen27-jang4m-source-keepalloc-20260522.json --timeout 420 --serve-extra-arg=--prefill-keep-alloc",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --python /Volumes/vMLX-Tahoe-Speed-Proof/vMLX.app/Contents/Resources/bundled-python/python/bin/python3 --port 8794 --timeout 360 --out build/current-decode-speed-live-qwen27-jang4m-packaged-tahoe-dmg-20260522.json",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --port 8799 --timeout 420 --out build/current-decode-speed-live-qwen27-jang4m-text-baseline-20260523.json --serve-extra-arg=--disable-prefix-cache",
            "VMLINUX_MLLM_PREFILL_TRACE=1 .venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --port 8798 --timeout 420 --out build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace3-20260523.json --serve-extra-arg=--disable-prefix-cache",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --python .venv/bin/python --port 8796 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json --serve-extra-arg=--disable-prefix-cache",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --python .venv/bin/python --port 8796 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --python panel/bundled-python/python/bin/python3 --port 8796 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json",
            "VMLINUX_MLLM_PREFILL_TRACE=1 .venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --python .venv/bin/python --port 8796 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json",
        ],
        "artifacts": [
            "build/current-decode-speed-live-qwen27-jang4m-20260522-hybrid-tq-review.json",
            "build/current-decode-speed-live-qwen27-jang4m-source-keepalloc-20260522.json",
            "build/current-decode-speed-live-qwen27-jang4m-packaged-keepalloc-20260522.json",
            "build/current-decode-speed-live-qwen27-jang4m-packaged-tahoe-dmg-20260522.json",
            "build/current-decode-speed-live-qwen27-jang4m-text-baseline-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-source-bypass-fix-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace3-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json",
        ],
    },
]


def build_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "rows": deepcopy(_ROWS),
    }


def validate_current_proof_sweep_artifacts(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    not_pass: list[dict[str, str]] = []

    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = root / artifact
        if not path.exists():
            missing.append(artifact)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            status = str(payload.get("status"))
        except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
            status = f"load_error:{type(exc).__name__}"
        if status != "pass":
            not_pass.append({"artifact": artifact, "status": status})

    regression_suite = _validate_current_regression_suite_artifact(root)
    model_family_matrix = _validate_current_model_family_matrix_artifact(root)
    model_artifact_matrix = _validate_current_model_artifact_matrix_artifact(root)
    cache_architecture_matrix = _validate_current_cache_architecture_matrix_artifact(root)
    regression_suite_ok = (
        regression_suite["status"] == "pass"
        and not regression_suite["failed_steps"]
        and not regression_suite["unexpected_open_requirements"]
        and not regression_suite["missing_expected_open_requirements"]
    )
    model_family_matrix_ok = (
        model_family_matrix["status"] == "pass"
        and not model_family_matrix["missing_rows"]
        and not model_family_matrix["missing_expected_rows"]
    )
    model_artifact_matrix_ok = (
        model_artifact_matrix["status"] == "pass"
        and not model_artifact_matrix["missing_markers"]
        and not model_artifact_matrix["failed_checks"]
        and not model_artifact_matrix["missing_expected_checks"]
    )
    cache_architecture_matrix_ok = (
        cache_architecture_matrix["status"] == "pass"
        and not cache_architecture_matrix["missing_markers"]
        and not cache_architecture_matrix["missing_api_checks"]
        and not cache_architecture_matrix["missing_api_command_markers"]
        and not cache_architecture_matrix["missing_panel_markers"]
        and not cache_architecture_matrix["failed_checks"]
        and not cache_architecture_matrix["missing_expected_checks"]
    )

    return {
        "status": "pass"
        if (
            not missing
            and not not_pass
            and regression_suite_ok
            and model_family_matrix_ok
            and model_artifact_matrix_ok
            and cache_architecture_matrix_ok
        )
        else "fail",
        "missing": missing,
        "not_pass": not_pass,
        "regression_suite": regression_suite,
        "model_family_matrix": model_family_matrix,
        "model_artifact_matrix": model_artifact_matrix,
        "cache_architecture_matrix": cache_architecture_matrix,
    }


def _validate_current_regression_suite_artifact(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "missing",
        "failed_steps": [],
        "open_requirements": [],
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS.copy(),
    }
    path = root / CURRENT_REGRESSION_SUITE_ARTIFACT
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    open_requirements = [str(item) for item in payload.get("open_requirements", [])]
    failed_steps = [str(item) for item in payload.get("failed_steps", [])]
    unexpected_open_requirements = [
        item for item in open_requirements if item not in EXPECTED_CURRENT_OPEN_REQUIREMENTS
    ]
    missing_expected_open_requirements = [
        item for item in EXPECTED_CURRENT_OPEN_REQUIREMENTS if item not in open_requirements
    ]

    result.update(
        {
            "status": str(payload.get("status")),
            "failed_steps": failed_steps,
            "open_requirements": open_requirements,
            "unexpected_open_requirements": unexpected_open_requirements,
            "missing_expected_open_requirements": missing_expected_open_requirements,
        }
    )
    return result


def _validate_current_cache_architecture_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "missing_api_checks": [],
        "missing_api_command_markers": [],
        "missing_panel_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "missing_api_checks": [str(item) for item in payload.get("missing_api_checks", [])],
            "missing_api_command_markers": [
                str(item) for item in payload.get("missing_api_command_markers", [])
            ],
            "missing_panel_markers": [
                str(item) for item in payload.get("missing_panel_markers", [])
            ],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_model_artifact_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_markers = [str(item) for item in payload.get("missing_markers", [])]
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": missing_markers,
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_model_family_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "matched_rows": [],
        "missing_rows": [],
        "missing_expected_rows": list(EXPECTED_CURRENT_MODEL_FAMILY_ROWS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    matched_rows = [str(item) for item in payload.get("matched_rows", [])]
    missing_rows = [str(item) for item in payload.get("missing_rows", [])]
    missing_expected_rows = [
        row for row in EXPECTED_CURRENT_MODEL_FAMILY_ROWS if row not in matched_rows
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "matched_rows": matched_rows,
            "missing_rows": missing_rows,
            "missing_expected_rows": missing_expected_rows,
        }
    )
    return result
