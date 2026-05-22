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
            "Auto chat Max Tokens omits per-request output caps so server startup defaults can apply",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-20260522-chat-auto-server-default.json",
        ],
        "artifacts": [
            "build/current-max-output-context-contract-20260522-chat-auto-server-default.json",
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
            "Reasoning parser dropdown covers every parser the panel registry can emit",
            "MiniMax minimax_m2 regression stays covered",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_parser_registry_contract.py --out build/current-parser-registry-contract-20260522-reasoning-dropdown.json",
        ],
        "artifacts": [
            "build/current-parser-registry-contract-20260522-reasoning-dropdown.json",
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
            "maxToolIterations caps tool loops",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_tool_call_contract.py --out build/current-tool-call-contract-20260521.json",
        ],
        "artifacts": [
            "build/current-tool-call-contract-20260521.json",
            "build/current-dsv4-default-cache-tool-loop/result.json",
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
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_api_surface_contract.py --out build/current-api-surface-contract-20260522-chat-auto-server-default.json",
        ],
        "artifacts": [
            "build/current-api-surface-contract-20260522-chat-auto-server-default.json",
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
            "Cache detail telemetry reports paged, typed native, and TQ/L2 state",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_cache_architecture_contract.py --out build/current-cache-architecture-contract-20260521.json",
        ],
        "artifacts": [
            "build/current-cache-architecture-contract-20260521.json",
        ],
    },
    {
        "id": "model-artifact-format-detection",
        "domain": "model_artifact_detection",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "JANG, JANGTQ/MXTQ, MXFP4, MXFP8, dropped-MTP, and preserved-MTP artifacts are not inferred from names alone",
            "Registry/family detection uses bundle config and capability metadata",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_model_artifact_format_contract.py --out build/current-model-artifact-format-contract-20260521.json",
        ],
        "artifacts": [
            "build/current-model-artifact-format-contract-20260521.json",
        ],
    },
    {
        "id": "model-family-detection-noheavy",
        "domain": "model_family_detection",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Named DSV4, ZAYA, ZAYA1-VL, Ling/Bailing, Nemotron, Qwen 3.6 VL/video/hybrid, MXFP4, MXFP8, native-MTP, MiniMax, and Hy3 rows keep expected parser/cache/modality policy",
            "Decode-speed rows keep JANG-only, JANGTQ/MXTQ, MXFP4, and MXFP8 speed thresholds distinct while staying aligned with engine registry parser, modality, and cache metadata for existing local models",
            "Decode-speed rows keep DSV4 native composite separate from generic JANGTQ/MXTQ rows",
            "Decode-speed health checks reject DSV4 native or ZAYA typed cache health for plain KV JANG/JANGTQ/MXFP rows",
            "Every decode-speed row with a declared tool or reasoning parser uses a registered engine parser and CLI-accepted parser choice even when that local model path is absent",
            "Decode-speed launch commands preserve row parser/modality policy and strip source-path or forced JANGTQ acceleration environment overrides",
            "Decode-speed matrix includes large external Mistral JANGTQ, Mistral MXFP4, and GPT-OSS rows with parser/modality launch policy pinned",
            "Decode-speed matrix includes external Nemotron 3 JANGTQ2 and MXFP4 rows with Nemotron parser/reasoning launch policy pinned",
            "This is source/static compatibility proof only; live multi-turn output quality remains a separate live row",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_model_family_detection_contract.py --out build/current-model-family-detection-contract-20260522-plain-kv-cache-health.json",
        ],
        "artifacts": [
            "build/current-model-family-detection-contract-20260522-plain-kv-cache-health.json",
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
            "Runtime-active MTP still requires live equivalence/speed rows before release claims",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_native_mtp_contract.py --out build/current-native-mtp-contract-20260521.json",
        ],
        "artifacts": [
            "build/current-native-mtp-contract-20260521.json",
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
            "Still-image live rows do not imply video/audio/Omni clearance",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_vl_media_cache_contract.py --out build/current-vl-media-cache-contract-20260521.json",
        ],
        "artifacts": [
            "build/current-vl-media-cache-contract-20260521.json",
        ],
    },
    {
        "id": "packaged-release-integrity",
        "domain": "packaging_release",
        "mode": "packaged",
        "heavy": False,
        "proves": [
            "Version triples, bundled Python hash parity, packaged app signature, and objective proof digest gate release attempts",
            "verify-bundled checks packaged Python imports, source hash parity, and relocatable console scripts",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_packaged_integrity_contract.py --out build/current-packaged-integrity-contract-20260521.json",
        ],
        "artifacts": [
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
            "PyPI, GitHub release, and updater feeds remain explicit operator checks before public release",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260521.json",
        ],
        "artifacts": [
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
