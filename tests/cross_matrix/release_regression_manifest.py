"""Release-regression manifest for current vMLX Python/Electron work.

This is the durable index for the broad regression suite Eric requested. It is
intentionally explicit: each row names the user-visible or runtime contract it
protects, the commands that currently cover it, and whether the row is no-heavy,
live-model, or packaged-app only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


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
    "release_surface",
    "pr_intake",
}


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
            "Ollama gateway malformed num_predict values are omitted instead of poisoning max_tokens",
            "Ollama gateway malformed context values are omitted instead of poisoning max_prompt_tokens",
            "External coding-tool configs keep context window and output limit separate",
            "new-chat model-owned maxTokens cannot be replaced by inherited per-chat output caps",
            "server startup maxTokens and chat maxTokens remain independent when both are set",
            "persisted chat maxTokens cannot relaunch server with a new startup maxTokens",
            "per-chat maxTokens below or above the server startup default remain request-scoped",
            "Explicit Chat/Responses output caps do not mutate the server startup default used by later Auto requests",
            "Responses maxTokens below or above the server startup default remain request scoped",
            "Responses Auto does not synthesize max_output_tokens",
            "Raw API non-positive max_tokens/max_output_tokens are rejected rather than becoming hidden Auto defaults or empty generations",
            "Auto chat Max Tokens omits per-request output caps so server startup defaults can apply",
            "new chat output caps are not inherited or made sticky on default-profile or same-model clean chat creation",
            "chat reset does not convert model max_new_tokens into a sticky per-chat maxTokens override",
            "string-shaped legacy session maxTokens values are cleared before launch",
            "API gateway output-budget and context-budget paths are source-hashed",
            "DSV4 request-budget helper is source-hashed with the max-output boundary gate",
            "Casual preset maxTokens is documented as an explicit server output cap and does not change model-owned defaults or context",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-casual-server-output-cap.json",
        ],
        "artifacts": [
            "build/current-max-output-context-contract-20260522-persisted-chat-output-cap.json",
            "build/current-max-output-context-contract-20260522-new-chat-output-cap-nonsticky.json",
            "build/current-max-output-context-contract-20260522-chat-auto-server-default.json",
            "build/current-max-output-context-contract-20260522-responses-output-boundary.json",
            "build/current-max-output-context-contract-20260522-request-default-mutation.json",
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
            "panel typecheck and model-family registry stay green with those controls",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_noheavy_panel_settings_contract.py --out build/current-panel-settings-contract-proof-20260522-launch-memory-warning.json",
        ],
        "artifacts": [
            "build/current-panel-settings-contract-proof-20260522-launch-memory-warning.json",
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
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_generation_defaults_contract.py --out build/current-generation-defaults-contract-20260521.json",
        ],
        "artifacts": [
            "build/current-generation-defaults-contract-20260521.json",
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
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_reasoning_template_contract.py --out build/current-reasoning-template-contract-20260521.json",
        ],
        "artifacts": [
            "build/current-reasoning-template-contract-20260521.json",
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
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_api_surface_contract.py --out build/current-api-surface-contract-20260522-stream-cache-detail.json",
        ],
        "artifacts": [
            "build/current-api-surface-contract-20260522-stream-cache-detail.json",
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
            "DSV4 panel env mapping enables pool quant only from explicit DSV4 config while the default remains off",
            "DSV4 timing probe covers prefix-cache replay and cold-store boundaries before speed/cache root-cause claims",
            "Cache detail telemetry reports paged, typed native, and TQ/L2 state",
            "Panel session launch builder preserves DSV4 default and diagnostic prefix-cache policy, DSV4-only native cache controls, Qwen3.6 hybrid and Mamba paged-cache forcing, and regular KV stale saved false semantics",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_cache_architecture_contract.py --out build/current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json",
        ],
        "artifacts": [
            "build/current-cache-architecture-contract-20260521.json",
            "build/current-cache-architecture-contract-20260522-panel-cache-launch.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-quant-append.json",
            "build/current-cache-architecture-contract-20260522-dsv4-timing.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-env-gate.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-materialized-cache.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json",
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
            "family-specific loader wrappers for JANGTQ text, JANGTQ VLM, Kimi VLM, DSV4, ZAYA, Laguna, and Mistral stay source-hashed in the artifact gate",
            "Registry/family detection uses bundle config and capability metadata",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_model_artifact_format_contract.py --out build/current-model-artifact-format-contract-20260522-affine-jang-loader.json",
        ],
        "artifacts": [
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
            "Decode-speed matrix includes large external Mistral JANGTQ, Mistral MXFP4, and GPT-OSS rows with parser/modality launch policy pinned",
            "Decode-speed matrix includes external Nemotron 3 JANGTQ2 and MXFP4 rows with Nemotron parser/reasoning launch policy pinned",
            "Existing local high-risk DSV4, Qwen JANG/JANGTQ/MXFP/4bit/MTP, Hy3, and Nemotron Omni/Nano JANGTQ/MXFP rows match the current engine registry parser, cache, and modality policy without loading weights",
            "Panel detection matches the same current local high-risk paths, including native-MTP Qwen affine-JANG VL routing, so UI launch policy does not diverge from engine/API routing",
            "This is source/static compatibility proof only; live multi-turn output quality remains a separate live row",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_model_family_detection_contract.py --out build/current-model-family-detection-contract-20260522-panel-local-paths.json",
        ],
        "artifacts": [
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
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_native_mtp_contract.py --out build/current-native-mtp-contract-20260522-dsv4-additional-args.json",
        ],
        "artifacts": [
            "build/current-native-mtp-contract-20260522-dsv4-additional-args.json",
            "build/current-native-mtp-contract-20260522-config-only.json",
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
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_mcp_policy_contract.py --out build/current-mcp-policy-contract-20260521.json",
        ],
        "artifacts": [
            "build/current-mcp-policy-contract-20260521.json",
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
            "Still-image live rows do not imply video/audio/Omni clearance",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_vl_media_cache_contract.py --out build/current-vl-media-cache-contract-20260522-panel-family.json",
        ],
        "artifacts": [
            "build/current-vl-media-cache-contract-20260522-panel-family.json",
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
        ],
        "commands": [
            "VMLINUX_JANG_TOOLS_SOURCE=/Users/example/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools VMLX_JANG_TOOLS_SOURCE=/Users/example/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools .venv/bin/python tests/cross_matrix/run_packaged_integrity_contract.py --out build/current-packaged-integrity-contract-20260522-objective-gate-enforced.json",
        ],
        "artifacts": [
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
            "if latest.json is bumped to the source version, it carries a matching URL, SHA256, and notes version",
            "after release, latest.json may equal the source version only when the published updater state is complete",
            "PyPI, GitHub release, and updater feeds remain explicit operator checks before public release",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-post-release-updater.json",
        ],
        "artifacts": [
            "build/current-release-surface-contract-20260522-post-release-updater.json",
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
]


def build_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "rows": deepcopy(_ROWS),
    }
