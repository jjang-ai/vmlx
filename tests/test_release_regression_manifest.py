import json
import re
from pathlib import Path
import shlex
import hashlib

from tests.cross_matrix.run_release_regression_manifest import build_manifest_artifact
from tests.cross_matrix.release_regression_manifest import (
    CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS,
    CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS,
    CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS,
    CURRENT_DEV_UI_PROOF_ARTIFACTS,
    CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS,
    CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
    CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
    CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT,
    CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT,
    CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT,
    CURRENT_POST_BUDGET_EDGE_ARTIFACTS,
    CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_NOCACHE_NO_KVQ_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_PROFILE_DIAGNOSTIC_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_SINK_AB_ARTIFACT,
    CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
    CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS,
    CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS,
    CURRENT_REGRESSION_SUITE_ARTIFACT,
    CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
    CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
    EXPECTED_CURRENT_API_SURFACE_CHECKS,
    EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS,
    EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS,
    EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS,
    EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS,
    EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS,
    EXPECTED_CURRENT_MCP_POLICY_CHECKS,
    EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS,
    EXPECTED_CURRENT_NATIVE_MTP_CHECKS,
    EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS,
    EXPECTED_CURRENT_OPEN_REQUIREMENTS,
    EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS,
    EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
    EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS,
    EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS,
    EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS,
    EXPECTED_CURRENT_TOOL_CALL_CHECKS,
    EXPECTED_CURRENT_VL_MEDIA_CHECKS,
    REQUIRED_RELEASE_DOMAINS,
    REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES,
    REQUIRED_REAL_UI_LIVE_MODEL_SURFACES,
    REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY,
    _required_live_smoke_request_labels,
    _current_regression_suite_state_is_acceptable,
    _annotate_real_ui_unblocked_non_mimo_status,
    _current_release_blocker_ledger,
    _live_smoke_gap_is_expected_mimo_open,
    _real_ui_named_tool_probe_semantics_ok,
    _validate_current_issue175_179_release_boundary_audit,
    _validate_current_real_ui_live_model_matrix,
    _validate_current_step37_vlm_runtime_audit,
    build_manifest,
    validate_current_proof_sweep_artifacts,
)

_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _passing_cache_family_matrix():
    return {
        row: {
            "status": "pass",
            "checks": {},
            "missing_markers": [],
            "missing_api_checks": [],
            "missing_api_command_markers": [],
            "missing_panel_markers": [],
        }
        for row in EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
    }


def _passing_cache_architecture_payload():
    return {
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS},
        "missing_markers": [],
        "missing_api_checks": [],
        "missing_api_command_markers": [],
        "missing_panel_markers": [],
        "cache_family_matrix": _passing_cache_family_matrix(),
    }


def _passing_generation_defaults_family_matrix():
    return {
        row: {
            "status": "pass",
            "checks": {},
            "missing_markers": [],
        }
        for row in EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS
    }


def _passing_generation_defaults_payload():
    return {
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS},
        "missing_markers": [],
        "generation_defaults_family_matrix": _passing_generation_defaults_family_matrix(),
    }


def _write_current_objective_digest(
    root: Path,
    *,
    open_requirements: list[str] | None = None,
    missing_evidence: list[str] | None = None,
) -> None:
    artifact = root / "build/current-objective-proof-audit-20260531-nemotron-exact-finalizer-ledger.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    open_rows = (
        EXPECTED_CURRENT_OPEN_REQUIREMENTS
        if open_requirements is None
        else open_requirements
    )
    requirements = [
        {
            "requirement": "Cache architecture matrix is release-proven",
            "status": "pass",
            "details": {
                "evidence_files_present": True,
                "missing_evidence": [],
            },
        },
    ]
    requirements.extend(
        {
            "requirement": requirement,
            "status": "open",
            "details": {
                "evidence_files_present": not missing_evidence,
                "missing_evidence": missing_evidence or [],
            },
        }
        for requirement in open_rows
    )
    artifact.write_text(json.dumps({"requirements": requirements}) + "\n", encoding="utf-8")


def _write_passing_covered_live_smoke_artifacts(root: Path) -> None:
    for row_id, artifact in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.items():
        row = dict(CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS[row_id])
        result = {
            "model": row["name"],
            "row": row,
            "status": "pass",
            "failures": [],
            "command": [
                "panel/bundled-python/python/bin/python3.12",
                "-B",
                "-s",
                "-m",
                "vmlx_engine.cli",
                "serve",
                row["name"],
            ],
        }
        result["requests"] = [
            {"label": label, "validation_failures": []}
            for label in _required_live_smoke_request_labels(row_id, result)
        ]
        for request in result["requests"]:
            if request["label"] == "text_cache_repeat_1":
                request["usage"] = {"prompt_tokens_details": None}
                request["cache_summary"] = {
                    "has_cache_hit": False,
                    "cache_hit_tokens": 0,
                }
            if request["label"] == "text_cache_repeat_2":
                request["usage"] = {
                    "prompt_tokens_details": {
                        "cached_tokens": 10,
                        "cache_detail": "paged",
                    }
                }
                request["cache_summary"] = {
                    "has_cache_hit": True,
                    "cache_hit_tokens": 10,
                }
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "completed": 1,
                    "failed": 0,
                    "results": [result],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _write_passing_dev_ui_proof_artifacts(root: Path) -> None:
    for key, artifact in CURRENT_DEV_UI_PROOF_ARTIFACTS.items():
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith("screenshot"):
            path.write_bytes(_MINIMAL_PNG)
            continue
        path.write_text(
            json.dumps(
                {
                    "repoDir": str(root.resolve()),
                    "panelDir": str((root / "panel").resolve()),
                    "appLogTail": [
                        "dev server running for the electron renderer process at:",
                        "  \u279c  Local:   http://localhost:5173/",
                        "start electron app...",
                    ],
                    "screenshots": {
                        "chatSettings": str(
                            (
                                root
                                / CURRENT_DEV_UI_PROOF_ARTIFACTS[
                                    "chat_settings_screenshot"
                                ]
                            ).resolve()
                        ),
                        "serverCacheSettings": str(
                            (
                                root
                                / CURRENT_DEV_UI_PROOF_ARTIFACTS[
                                    "server_cache_screenshot"
                                ]
                            ).resolve()
                        ),
                    },
                    "mockResponsesRequests": 2,
                    "finalAssistantContent": "Done after tools.",
                    "rawThinkTagLeak": False,
                    "persistedToolCount": 14,
                    "runToolExecutingToResultMs": 1000,
                    "eventCounts": {
                        "tool": 14,
                        "reasoningDone": 2,
                        "complete": 1,
                    },
                    "toolPhasesById": {
                        tool_id: ["calling", "executing", "result"]
                        for tool_id in (
                            "call_live_run",
                            "call_live_list",
                            "call_live_image",
                            "call_live_video",
                        )
                    },
                    "followupTailContentTypes": ["text", "image_url", "video_url"],
                    "initialPayloadKeys": {
                        "temperature": False,
                        "top_p": False,
                        "top_k": False,
                        "max_output_tokens": False,
                    },
                    "chatSettingsUi": {
                        "visible": True,
                        "labels": [
                            "Enable Built-in Coding Tools",
                            "Working Directory",
                            "Shell",
                            "Search",
                            "Utilities",
                            "Hide Tool Status",
                        ],
                        "checked": {
                            "builtinToolsEnabled": True,
                            "shellEnabled": True,
                            "fileToolsEnabled": False,
                            "gitEnabled": False,
                            "hideToolStatus": True,
                        },
                    },
                    "serverCacheUi": {
                        "visible": True,
                        "labels": [
                            "Enable Prefix Cache",
                            "Use Paged KV Cache",
                            "Block Disk Cache (L2)",
                            "Enable Disk Cache",
                            "Stored Cache Quantization",
                        ],
                        "afterBlockDiskToggle": {
                            "enablePrefixCache": True,
                            "usePagedCache": True,
                            "enableBlockDiskCache": True,
                            "enableDiskCache": False,
                        },
                        "afterDiskToggle": {
                            "enablePrefixCache": True,
                            "usePagedCache": False,
                            "enableBlockDiskCache": False,
                            "enableDiskCache": True,
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _write_passing_real_ui_live_model_proof_artifacts(root: Path) -> None:
    def native_cache_health() -> dict[str, object]:
        return {
            "family": "plain_kv",
            "schema": "plain_kv_v1",
            "cache_type": "paged_kv",
            "components": ["attention_kv"],
            "generic_turboquant_kv": {"enabled": True, "reason": "plain_attention"},
            "prefix": True,
            "paged": True,
            "block_disk_l2": True,
        }

    def cache_endpoint_stats() -> dict[str, object]:
        return {
            "before": {
                "scheduler_cache": {"hits": 0, "misses": 0},
            },
            "after": {
                "scheduler_cache": {"hits": 1, "misses": 1},
                "block_disk_cache": {"disk_hits": 1, "disk_writes": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        }

    for row in CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS.values():
        screenshot = root / row["chat_screenshot"]
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        screenshot.write_bytes(_MINIMAL_PNG)
        proof_path = root / row["proof"]
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        proof = {
                    "repoDir": str(root.resolve()),
                    "panelDir": str((root / "panel").resolve()),
                    "script": "panel/scripts/live-real-ui-model-proof.mjs",
                    "modelPath": row["model_path"],
                    "modelName": row["model_name"],
                    "server": {
                        "baseUrl": "http://127.0.0.1:8899",
                        "health": {
                            "status": "healthy",
                            "model_loaded": True,
                            "native_cache": native_cache_health(),
                        },
                        "models": {"data": [{"id": row["model_name"]}]},
                    },
                    "appLogTail": [
                        "dev server running for the electron renderer process at:",
                        "  \u279c  Local:   http://localhost:5173/",
                        "start electron app...",
                    ],
                    "chat": {
                        "turns": [
                            {"role": "user", "content": "Say READY in English."},
                            {"role": "assistant", "content": "READY"},
                            {"role": "user", "content": "Repeat READY once."},
                            {"role": "assistant", "content": "READY"},
                        ],
                        "finalVisibleText": "READY",
                        "rawParserTagLeak": False,
                        "cjkLeakCount": 0,
                        "koreanLeakCount": 0,
                        "reasoningText": "",
                        "reasoningRawParserTagLeak": False,
                        "reasoningCjkLeakCount": 0,
                        "reasoningKoreanLeakCount": 0,
                        "reasoningNumericRunCount": 0,
                    },
                    "cache": cache_endpoint_stats(),
                    "requestContract": {
                        "promptOne": "Say READY in English.",
                        "promptTwo": "Repeat READY once.",
                        "requestMaxTokens": 512,
                        "maxToolIterations": 8,
                        "toolResultMaxChars": 12000,
                        "wireApi": "chat",
                        "builtinToolsEnabled": False,
                        "enableThinking": None,
                        "checkServerCacheControls": False,
                        "checkMedia": False,
                        "checkVideo": False,
                        "expectPagedCacheLocked": False,
                        "imageExpectRegex": None,
                        "videoExpectRegex": None,
                        "cacheExpectRegex": None,
                    },
                    "screenshots": {
                        "chat": str(
                            (
                                root
                                / row["chat_screenshot"]
                            ).resolve()
                        ),
                    },
                }
        if row["proof"].endswith("zaya-text-cachecontrols-20260527-proof.json"):
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "zaya-text-responses-runcommand-tools-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": "$ echo REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt\n\n",
                    },
                ],
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            "$ cat real_ui_tool_probe_1.txt && echo "
                            "REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt\n\n"
                            "REAL_UI_LIVE_TOOL_ONE\n"
                        ),
                    },
                ],
            ]
        if row["proof"].endswith(
            "gemma4-responses-stricttools-max768-visible-20260531-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4, "reasoningDone": 0}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
        if row["proof"].endswith(
            "gemma4-responses-reasoningonly-max768-visible-20260531-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 0, "reasoningDone": 2}
            proof["requestedEnableThinking"] = True
            proof["requestedBuiltinTools"] = False
            proof["chatOverrides"] = {"builtinToolsEnabled": False}
            proof["persistedReasoningCount"] = 2
        if row["proof"].endswith("gemma4-cachecontrols-20260527-proof.json"):
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                ],
            }
        if row["proof"].endswith(
            (
                "zaya-text-responses-stricttools-cachecontrols-20260530-proof.json",
                "zaya-vl-responses-stricttools-cachecontrols-20260530-proof.json",
                "step37-jang2l-responses-tools-image-cachecontrols-after-direct-media-tool-filter-20260531-proof.json",
                "lfm25-moe-a1b-jang2l-stricttools-chat-20260530-proof.json",
                "lfm25-moe-a1b-jang2l-stricttools-responses-filesemantic-20260530-proof.json",
            )
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                ],
            }
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            "$ printf %s REAL_UI_LIVE_TOOL_ONE > "
                            "real_ui_tool_probe_1.txt\nREAL_UI_LIVE_TOOL_ONE"
                        ),
                    },
                    {
                        "phase": "result",
                        "toolName": "write_file",
                        "detail": (
                            "$ cat real_ui_tool_probe_1.txt && printf %s "
                            "REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt && "
                            "cat real_ui_tool_probe_2.txt\n"
                            "real_ui_tool_probe_1.txt\n"
                            "real_ui_tool_probe_2.txt\n"
                            "REAL_UI_LIVE_TOOL_TWO"
                        ),
                    },
                ],
            ]
        if row["proof"].endswith(
            "step37-jang2l-responses-tools-image-cachecontrols-after-direct-media-tool-filter-20260531-proof.json"
        ):
            proof["media"] = {"imageVerified": True}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
        if row["proof"].endswith(
            "step37-jang2l-responses-reasoning-max768-20260531-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 0, "reasoningDone": 1}
            proof["requestedBuiltinTools"] = False
            proof["chatOverrides"] = {"builtinToolsEnabled": False}
            proof["persistedReasoningCount"] = 1
        if row["proof"].endswith("gemma4-cachecontrols-20260527-proof.json"):
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 1, "tool": 4, "reasoningDone": 1}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedReasoningCount"] = 1
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "nemotron-omni-nano-responses-stricttools-cachecontrols-exact-finalizer-20260531-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt\n\n",
                    },
                ],
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            "$ cat real_ui_tool_probe_1.txt && printf %s "
                            "REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt && "
                            "cat real_ui_tool_probe_2.txt\n\n"
                            "REAL_UI_LIVE_TOOL_ONEREAL_UI_LIVE_TOOL_TWO"
                        ),
                    },
                ],
            ]
            proof["toolProbeFiles"] = {
                "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
                "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
            }
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "ling-bailing-jangtq-responses-filesemantic-20260530-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 1, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
            proof["toolProbeFiles"] = {
                "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE\n",
                "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO\n",
            }
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "zaya-vl-responses-tools-cachecontrols-commandargfix-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "run_command"},
                ],
            ]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "minimax-m27-small-responses-tools-cachecontrols-localonly-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4, "reasoningDone": 1}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedReasoningCount"] = 1
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "hy3-jangtq2-responses-tools-filesemantic-20260530-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": '$ echo "REAL_UI_LIVE_TOOL_ONE" > real_ui_tool_probe_1.txt\n\n',
                    },
                ],
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            '$ cat real_ui_tool_probe_1.txt && echo "REAL_UI_LIVE_TOOL_TWO" '
                            "> real_ui_tool_probe_2.txt\n\nREAL_UI_LIVE_TOOL_ONE\n"
                        ),
                    },
                ],
            ]
            proof["toolProbeFiles"] = {
                "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE\n",
                "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO\n",
            }
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith("hy3-jangtq2-responses-reasoning-20260527-proof.json"):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "reasoningDone": 1}
            proof["persistedReasoningCount"] = 1
        if row["proof"].endswith(
            "qwen36-mxfp4-crack-responses-tools-reasoning-image-cachecontrols-localonly-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 3, "tool": 4, "reasoningDone": 1}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedReasoningCount"] = 1
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
            proof["media"] = {
                "requestedImage": True,
                "requestedVideo": False,
                "imageExpectedRegex": "\\bred\\b",
                "videoExpectedRegex": "",
                "imageSemanticVerified": True,
                "videoSemanticVerified": False,
                "imageVerified": True,
                "videoVerified": False,
                "persistedImageAttachment": True,
                "persistedVideoAttachment": False,
            }
        if row["proof"].endswith("qwen36-mxfp4-crack-video-20260527-proof.json"):
            proof["media"] = {
                "requestedImage": False,
                "requestedVideo": True,
                "imageExpectedRegex": "\\bred\\b",
                "videoExpectedRegex": "red",
                "imageSemanticVerified": False,
                "videoSemanticVerified": True,
                "imageVerified": False,
                "videoVerified": True,
                "persistedImageAttachment": False,
                "persistedVideoAttachment": True,
            }
        if row.get("family") == "zaya_vl":
            proof["media"] = {
                "requestedImage": True,
                "requestedVideo": False,
                "imageExpectedRegex": "\\bred\\b",
                "videoExpectedRegex": "",
                "imageSemanticVerified": True,
                "videoSemanticVerified": False,
                "imageVerified": True,
                "videoVerified": False,
                "persistedImageAttachment": True,
                "persistedVideoAttachment": False,
            }
            proof["server"]["health"]["scheduler"] = {
                "batch_generator": {
                    "num_images_processed": 1,
                    "vision_encoding_time": 0.01,
                }
            }
            proof["chat"]["turns"].extend(
                [
                    {
                        "role": "user",
                        "content": json.dumps(
                            [
                                {"type": "text", "text": "What color?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,"
                                    },
                                },
                            ]
                        ),
                    },
                    {"role": "assistant", "content": "Red"},
                ]
            )
            proof["chat"]["finalVisibleText"] = "Red"
        proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")
    _write_expected_issue175_179_release_boundary_audit(root)
    _write_expected_issue179_minimax_k_root_cause_audit(root)
    _write_expected_installed_app_runtime_parity_audit(root)
    _write_expected_staged_app_runtime_parity_audit(root)
    _write_expected_issue175_177_installed_runtime_audit(root)
    _write_expected_issue175_177_live_runtime_audit(root)


def _write_expected_issue175_179_release_boundary_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "issues": {
                    "175": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_app_memory_clear_runtime_live_stress_proven"
                        ),
                    },
                    "176": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_app_promoted_block_cleanup_live_pressure_proven"
                        ),
                    },
                    "177": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_app_cache_selection_live_ttft_proven"
                        ),
                    },
                    "178": {
                        "focused_source_slice": "pass",
                        "release_clearance": "packaged_app_lora_surface_proven",
                    },
                    "179": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_app_reporter_prompt_and_cancel_boundary_proven"
                        ),
                    },
                },
                "open_release_rows": [
                    "Real Electron UI cross-family live model matrix is release-cleared",
                    "DSV4 long-output/code/file-generation quality is release-cleared",
                ],
                "required_live_proof_axes": [
                    "real_app_parameters",
                    "multi_turn_chat",
                    "openai_responses_streaming",
                    "tool_parser_detection_and_execution",
                    "reasoning_parser_display_boundary",
                    "prefix_cache_reuse",
                    "paged_cache_reuse",
                    "l2_disk_cache_reconstruct",
                    "turboquant_encode_decode",
                    "hybrid_ssm_cca_swa_hsa_csa_attention",
                    "metal_matmul_runtime_path",
                    "jang_and_jangtq_metadata",
                    "vl_image_video_paths",
                    "wrong_language_and_parser_tag_leak_check",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_release_regression_manifest_accepts_open_issue175_179_boundary_audit(
    tmp_path,
):
    _write_expected_issue175_179_release_boundary_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "open"
    payload["issues"]["179"]["release_clearance"] = "open_reporter_parity_required"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue175_179_release_boundary_audit(tmp_path)

    assert result["status"] == "open"
    assert "issue175_179_boundary_still_open" not in result["failures"]


def _write_expected_issue179_minimax_k_root_cause_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "proven": {
                    "reporter_log_installed_app_bundled_python_seen": True,
                    "reporter_log_request_shape_and_sampling_kwargs_seen": True,
                    "reporter_log_launch_parser_cache_flags_seen": True,
                    "reporter_log_has_abort_before_visible_content": True,
                    "reporter_log_has_responses_cancel_404": True,
                    "local_real_ui_diagnostics_clean": True,
                    "local_installed_issue179_session_settings_parity": True,
                    "local_installed_bundle_has_responses_cancel_route": True,
                    "public_v1549_tahoe_dmg_has_responses_cancel_route": True,
                    "local_installed_responses_cancel_live_probe": True,
                    "current_source_responses_cancel_contract_proven": True,
                    "current_source_responses_cancel_inactive_404_contract_proven": True,
                    "reporter_cancel_404_after_stream_abort_order_proven": True,
                    "reporter_cancel_404_after_econnreset_same_response_id_proven": True,
                    "local_reporter_prompt_reproduction_clean": True,
                },
                "not_proven": [],
                "local_reporter_prompt_reproduction": {
                    "path": "build/current-issue179-minimax-k-responses-cancel-probe-installed-badtext-20260528.json",
                    "exists": True,
                    "status": "open",
                    "request_matches_reporter": True,
                    "stream_started": True,
                    "response_id_seen": True,
                    "cancel_status": None,
                    "cancel_trigger": None,
                    "bad_text_captured": False,
                    "bad_text_counts": {
                        "raw_parser_tag": False,
                        "cjk": 0,
                        "korean": 0,
                        "numeric": 0,
                    },
                    "clean": True,
                },
                "reporter_parity_artifact": {
                    "path": "build/issue-179/reporter-parity-metadata-20260527.json",
                    "exists": True,
                    "status": "pass",
                    "comparison_status": "ready_for_direct_comparison",
                    "collector_missing_fields": [],
                    "required_fields": [
                        "capture_provenance",
                        "installed_server_sha256",
                        "server_has_responses_cancel_route",
                        "server_cancel_calls_engine_abort",
                        "model_manifest_sha256",
                        "model_file_hashes",
                        "chat_id",
                        "session_settings",
                        "response_id",
                        "response_active_at_cancel",
                        "raw_sse_cancel_lifecycle",
                    ],
                },
                "reporter_parity_comparison": {
                    "status": "pass",
                    "capture_provenance_is_reporter_machine": True,
                    "server_hash_matches_local_installed": True,
                    "server_route_markers_match": True,
                    "model_manifest_sha256_matches_local": True,
                    "model_file_hashes_match_local": True,
                    "chat_id_matches_reporter_log": True,
                    "response_id_matches_reporter_log": True,
                    "response_active_at_cancel_recorded": True,
                    "raw_sse_cancel_lifecycle_present": True,
                    "failures": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_current_proof_sweep_rejects_issue179_missing_live_cancel_probe(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"].pop("local_installed_responses_cancel_live_probe")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:local_installed_responses_cancel_live_probe"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_public_dmg_route_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"].pop("public_v1549_tahoe_dmg_has_responses_cancel_route")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:public_v1549_tahoe_dmg_has_responses_cancel_route"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_econnreset_same_id_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["reporter_cancel_404_after_econnreset_same_response_id_proven"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:reporter_cancel_404_after_econnreset_same_response_id_proven"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_reporter_installed_app_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["reporter_log_installed_app_bundled_python_seen"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:reporter_log_installed_app_bundled_python_seen"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_reporter_settings_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["reporter_log_request_shape_and_sampling_kwargs_seen"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:reporter_log_request_shape_and_sampling_kwargs_seen"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_reporter_launch_config_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["reporter_log_launch_parser_cache_flags_seen"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:reporter_log_launch_parser_cache_flags_seen"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_local_installed_session_parity(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["local_installed_issue179_session_settings_parity"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:local_installed_issue179_session_settings_parity"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_reporter_parity_artifact_contract(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("reporter_parity_artifact")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "missing_reporter_parity_artifact_contract" in result["failures"]


def test_current_proof_sweep_rejects_issue179_missing_reporter_parity_comparison(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("reporter_parity_comparison")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "missing_reporter_parity_comparison" in result["failures"]


def _write_expected_installed_app_runtime_parity_audit(root: Path) -> None:
    path = root / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "checks": {
                    "installed_python_exists": True,
                    "serve_help_runs": True,
                    "responses_cancel_route": True,
                    "image_lora_cli_flags": True,
                    "image_lora_load_signature": True,
                    "image_lora_server_globals": True,
                    "text_lora_flags_rejected": True,
                    "xml_function_tool_parser_cli": True,
                    "plain_attention_kv_native_cache_status": True,
                    "installed_panel_gateway_epipe_guard": True,
                    "installed_panel_gateway_epipe_aggregate_guard": True,
                    "installed_panel_chat_ipc_epipe_aggregate_guard": True,
                    "installed_panel_image_ipc_epipe_aggregate_guard": True,
                    "installed_panel_gateway_guarded_proxy_forwarding": True,
                    "installed_panel_gateway_write_once_behavior_marker": True,
                    "installed_panel_gateway_response_socket_destroyed_guard": True,
                    "installed_panel_gateway_request_socket_destroyed_guard": True,
                    "installed_panel_gateway_single_model_cache_endpoint_routing": True,
                    "installed_panel_gateway_single_model_streaming_routes": True,
                    "installed_panel_max_output_context_settings_wired": True,
                    "installed_panel_model_owned_generation_defaults_wired": True,
                    "installed_panel_request_builders_keep_output_context_separate": True,
                    "installed_panel_ollama_streaming_json_writes_guarded": True,
                    "installed_panel_ollama_proxy_response_errors_guarded": True,
                    "installed_vmlx_user_data_no_raw_disconnect_errors": True,
                    "installed_vmlx_diagnostic_reports_no_raw_disconnect_errors": True,
                },
                "missing_or_stale": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_expected_staged_app_runtime_parity_audit(root: Path) -> None:
    path = root / CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "installed_app": str(
                    root / "panel/release/mac-arm64/vMLX.app"
                ),
                "checks": {
                    "installed_python_exists": True,
                    "serve_help_runs": True,
                    "responses_cancel_route": True,
                    "image_lora_cli_flags": True,
                    "image_lora_load_signature": True,
                    "image_lora_server_globals": True,
                    "text_lora_flags_rejected": True,
                    "xml_function_tool_parser_cli": True,
                    "plain_attention_kv_native_cache_status": True,
                    "installed_panel_gateway_epipe_guard": True,
                    "installed_panel_gateway_epipe_aggregate_guard": True,
                    "installed_panel_chat_ipc_epipe_aggregate_guard": True,
                    "installed_panel_image_ipc_epipe_aggregate_guard": True,
                    "installed_panel_gateway_guarded_proxy_forwarding": True,
                    "installed_panel_gateway_write_once_behavior_marker": True,
                    "installed_panel_gateway_response_socket_destroyed_guard": True,
                    "installed_panel_gateway_request_socket_destroyed_guard": True,
                    "installed_panel_gateway_single_model_cache_endpoint_routing": True,
                    "installed_panel_gateway_single_model_streaming_routes": True,
                    "installed_panel_max_output_context_settings_wired": True,
                    "installed_panel_model_owned_generation_defaults_wired": True,
                    "installed_panel_request_builders_keep_output_context_separate": True,
                    "installed_panel_ollama_streaming_json_writes_guarded": True,
                    "installed_panel_ollama_proxy_response_errors_guarded": True,
                    "installed_vmlx_user_data_no_raw_disconnect_errors": True,
                    "installed_vmlx_diagnostic_reports_no_raw_disconnect_errors": True,
                },
                "missing_or_stale": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_current_proof_sweep_requires_staged_app_runtime_parity_audit(tmp_path):
    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "staged_app_runtime_parity_audit"
    ]

    assert result == {
        "artifact": CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        "status": "missing",
        "missing": [CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT],
        "failures": [],
    }


def test_current_proof_sweep_accepts_staged_app_runtime_parity_pass(tmp_path):
    _write_expected_staged_app_runtime_parity_audit(tmp_path)

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "staged_app_runtime_parity_audit"
    ]

    assert result == {
        "artifact": CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        "status": "pass",
        "missing": [],
        "failures": [],
    }


def test_current_proof_sweep_rejects_installed_app_missing_generation_settings_parity(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_panel_model_owned_generation_defaults_wired")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_model_owned_generation_defaults_wired"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_epipe_aggregate_guard(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_panel_gateway_epipe_aggregate_guard")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_gateway_epipe_aggregate_guard"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_ipc_epipe_aggregate_guard(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_panel_chat_ipc_epipe_aggregate_guard")
    payload["checks"].pop("installed_panel_image_ipc_epipe_aggregate_guard")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_chat_ipc_epipe_aggregate_guard"
        in result["failures"]
    )
    assert (
        "missing_check:installed_panel_image_ipc_epipe_aggregate_guard"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_user_data_epipe_scan(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_vmlx_user_data_no_raw_disconnect_errors")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_vmlx_user_data_no_raw_disconnect_errors"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_diagnostic_epipe_scan(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_vmlx_diagnostic_reports_no_raw_disconnect_errors")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_vmlx_diagnostic_reports_no_raw_disconnect_errors"
        in result["failures"]
    )


def _write_expected_issue175_177_installed_runtime_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "checks": {
                    "installed_python_exists": True,
                    "memory_clear_helper_imports_from_installed_app": True,
                    "memory_clear_uses_available_mlx_api": True,
                    "runtime_paths_avoid_removed_clear_memory_cache": True,
                    "promoted_disk_blocks_drop_parent_mirror": True,
                    "l2_readable_write_through_blocks_drop_parent_mirror": True,
                    "cache_selection_telemetry_installed": True,
                    "cache_execution_timing_telemetry_installed": True,
                },
                "failures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_expected_issue175_177_live_runtime_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "checks": {
                    "installed_app_qwen_live_probe_passed": True,
                    "cache_hit_ttft_improved": True,
                    "l2_block_disk_hits_observed": True,
                    "paged_ssm_cache_hit_observed": True,
                    "visible_content_observed": True,
                    "metal_memory_metrics_observed": True,
                    "prefix_paged_l2_capacity_projection_valid": True,
                    "minimax_installed_live_probe_passed": True,
                    "turboquant_live_decode_observed": True,
                    "scheduler_cache_selection_observed": True,
                    "scheduler_cache_execution_timing_observed": True,
                    "paged_tq_cache_hit_observed": True,
                    "minimax_l2_block_disk_hits_observed": True,
                    "minimax_cache_hit_request_latency_improved": True,
                    "minimax_visible_content_observed": True,
                    "minimax_restart_reader_probe_passed": True,
                    "cold_paged_tq_restart_hit_observed": True,
                    "cold_paged_cache_selection_observed": True,
                    "cold_paged_stream_ttft_observed": True,
                    "restart_visible_content_observed": True,
                    "admin_sleep_lifecycle_probe_passed": True,
                    "admin_sleep_deep_unload_observed": True,
                },
                "remaining_blockers": [],
                "failures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_passing_real_ui_dsv4_memory_preflight_artifact(root: Path) -> None:
    path = root / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_memory",
                "model_path": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "model_size_gb": 80.0,
                "free_plus_speculative_purgeable_gb": 39.15,
                "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                "required_available_gb": 120.0,
                "memory_gap_gb": 80.85,
                "psutil_available_gb": 104.0,
                "psutil_memory_gap_gb": 16.0,
                "inactive_file_cache_gb": 30.0,
                "launch_decision": "do_not_launch",
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
                ],
                "active_heavy_processes": [],
                "active_heavy_process_count": 0,
                "launch_blockers": ["insufficient_memory"],
                "did_not_launch": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_passing_covered_live_tool_smoke_artifacts(root: Path) -> None:
    for row_id, artifact in CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS.items():
        row = dict(CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS[row_id])
        live_labels = (
            _required_live_smoke_request_labels(row_id, {"row": row})
            if artifact in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.values()
            else ["text_cache_repeat_1", "text_cache_repeat_2", "text_multiturn_recall"]
        )
        result = {
            "model": row["name"],
            "row": row,
            "status": "pass",
            "failures": [],
            "command": [
                "panel/bundled-python/python/bin/python3.12",
                "-B",
                "-s",
                "-m",
                "vmlx_engine.cli",
                "serve",
                row["name"],
            ],
            "requests": [
                *[
                    {"label": label, "validation_failures": []}
                    for label in live_labels
                ],
                {
                    "label": "tool_required",
                    "content_head": "",
                    "validation_failures": [],
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "record_fact",
                                "arguments": json.dumps({"value": "blue-cat"}),
                            },
                        }
                    ],
                },
            ],
        }
        for request in result["requests"]:
            if request["label"] == "text_cache_repeat_1":
                request["usage"] = {"prompt_tokens_details": None}
                request["cache_summary"] = {
                    "has_cache_hit": False,
                    "cache_hit_tokens": 0,
                }
            if request["label"] == "text_cache_repeat_2":
                request["usage"] = {
                    "prompt_tokens_details": {
                        "cached_tokens": 10,
                        "cache_detail": "paged",
                    }
                }
                request["cache_summary"] = {
                    "has_cache_hit": True,
                    "cache_hit_tokens": 10,
                }
        if row.get("supports_thinking"):
            result["requests"].insert(
                3,
                {"label": "reasoning_on", "validation_failures": []},
            )
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "completed": 1,
                    "failed": 0,
                    "results": [result],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _write_expected_diagnostic_live_smoke_artifacts(root: Path) -> None:
    for row_id, expectation in CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS.items():
        artifact = expectation["artifact"]
        failures = []
        for label, reason_or_reasons in expectation["expected_label_reasons"].items():
            reasons = (
                reason_or_reasons
                if isinstance(reason_or_reasons, list)
                else [reason_or_reasons]
            )
            failures.extend(
                {"label": label, "reason": reason}
                for reason in reasons
            )
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": expectation["expected_status"],
                    "completed": 1,
                    "failed": 1,
                    "results": [
                        {
                            "model": row_id,
                            "status": expectation["expected_status"],
                            "failures": failures,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _write_passing_mimo_v2_root_cause_artifacts(root: Path) -> None:
    router_path = root / CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT
    router_path.parent.mkdir(parents=True, exist_ok=True)
    router_path.write_text(
        json.dumps(
            {
                "weight_max_abs_diff": 0.0,
                "bias_max_abs_diff": 0.0,
                "topk_weight_max_abs_diff": 0.0,
                "score_max_abs_diff": 0.0,
                "topk_exact_rows": [True, True],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    moe_path = root / CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT
    moe_path.parent.mkdir(parents=True, exist_ok=True)
    moe_path.write_text(
        json.dumps(
            {
                "row_stats": [
                    {"rel_l2": 0.603, "cosine": 0.813},
                    {"rel_l2": 0.437, "cosine": 0.901},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    profile_path = root / CURRENT_MIMO_V2_JANG2L_PROFILE_DIAGNOSTIC_ARTIFACT
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {"profile": "source", "rel_l2_vs_source": 0.0},
                    {"profile": "2L", "rel_l2_vs_source": 0.178},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    nocache_path = root / CURRENT_MIMO_V2_JANG2L_NOCACHE_NO_KVQ_ARTIFACT
    nocache_path.parent.mkdir(parents=True, exist_ok=True)
    nocache_path.write_text(
        json.dumps(
                {
                    "status": "pass",
                    "mode": "disable_prefix_cache_kv_quant_none_after_swa_runtime_sync",
                    "capabilities": {
                    "family": "mimo_v2",
                    "reasoning_parser": "think_xml",
                    "tool_parser": "xml_function",
                },
                "requests": [
                    {
                        "label": "ack",
                        "response": {"code": 200},
                        "content_head": "- - - - -} -} -}- - - - - -",
                    },
                    {
                        "label": "recall_check",
                        "response": {"code": 200},
                        "content_head": "-00 - - -.. - - - - -0 - - -",
                    },
                    {
                        "label": "reasoning",
                        "response": {"code": 200},
                        "content_head": ". - - - - - - -000 - - -",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_open_mimo_v2_sink_ab_artifact(root: Path) -> None:
    path = root / CURRENT_MIMO_V2_JANG2L_SINK_AB_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "open",
                "normal_coherent": False,
                "sink_disabled_coherent": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _expected_passing_covered_live_smoke_summaries() -> dict[str, object]:
    return {
        "status": "pass",
        "non_mimo_status": "pass",
        "artifacts": {
            row_id: {
                "artifact": artifact,
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "failing_results": [],
            }
            for row_id, artifact in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.items()
        },
        "missing": [],
        "not_pass": [],
        "non_mimo_missing": [],
        "non_mimo_not_pass": [],
    }


def _expected_passing_covered_live_tool_smoke_summaries() -> dict[str, object]:
    return {
        "status": "pass",
        "non_mimo_status": "pass",
        "artifacts": {
            row_id: {
                "artifact": artifact,
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "failing_results": [],
            }
            for row_id, artifact in CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS.items()
        },
        "missing": [],
        "not_pass": [],
        "non_mimo_missing": [],
        "non_mimo_not_pass": [],
    }


def _expected_passing_diagnostic_live_smoke_summaries() -> dict[str, object]:
    def expected_failures(expectation: dict[str, object]) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        for label, reason_or_reasons in expectation["expected_label_reasons"].items():
            reasons = (
                reason_or_reasons
                if isinstance(reason_or_reasons, list)
                else [reason_or_reasons]
            )
            failures.extend(
                {"label": str(label), "reason": str(reason)}
                for reason in reasons
            )
        return failures

    return {
        "status": "pass",
        "artifacts": {
            row_id: {
                "artifact": expectation["artifact"],
                "status": expectation["expected_status"],
                "completed": 1,
                "failed": 1,
                "matched_expected_failures": expected_failures(expectation),
                "unexpected_results": [],
            }
            for row_id, expectation in CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS.items()
        },
        "missing": [],
        "not_expected": [],
    }


def _passing_open_requirement_details() -> dict[str, object]:
    return {
        "Cross-family live multi-turn smoke matrix is release-cleared": {
            "details": {
                "missing_required_family_keys": ["mimo_v2"],
                "missing_evidence": [
                    CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["mimo_v2_jang2l"]
                ],
            }
        },
        "Real Electron UI cross-family live model matrix is release-cleared": {
            "details": {
                "release_boundary": "current_real_ui_matrix_is_authoritative_for_unblocked_non_mimo",
                "real_ui_live_model_matrix": {
                    "unblocked_non_mimo_status": "pass",
                    "unblocked_non_mimo_missing_families": [],
                    "unblocked_non_mimo_partial_families": [],
                    "covered_families": {"zaya_text": {"status": "pass"}},
                    "mixed_model_identity_families": [],
                },
                "required_real_ui_surfaces": [
                    "current Electron dev build",
                    "real loaded model per target family",
                    "chat settings persistence and non-sticky defaults",
                    "server cache setting controls",
                    "Responses and Chat Completions paths",
                    "long multi-turn tool calls",
                    "reasoning/tool parser display",
                    "prefix/cache hit telemetry",
                    "image/video follow-up where supported",
                    "no raw parser or reasoning tag leakage",
                ],
            }
        },
        "DSV4 long-output/code/file-generation quality is release-cleared": {
            "details": {
                "direct_off_exactness_boundary": {
                    "direct_off_failure_spans_chat_responses_and_completion": True,
                    "requested_thinking_success_is_diagnostic_only": True,
                    "requested_thinking_success_clears_direct_off": False,
                    "hidden_force_on_would_be_false_clearance": True,
                },
                "exact_code_root_boundary": {
                    "direct_off_route_wide_failure": True,
                    "requested_thinking_is_diagnostic_only": True,
                    "prefix_cache_not_sufficient_root_cause": True,
                    "forced_sampler_controls_not_sufficient_root_cause": True,
                    "bundle_defaults_do_not_clear_direct_off": True,
                    "direct_off_three_dot_identifier_boundary": True,
                    "current_primary_failure": "direct_off_exact_code_generation",
                },
                "current_source_full_output_preflight": {
                    "artifact": CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
                    "artifact_present": True,
                    "status": "skipped",
                    "reason": "insufficient_free_memory",
                    "available_gb": 103.87,
                    "required_available_gb": 120.0,
                    "memory_gap_gb": 16.13,
                    "strict_vm_stat_memory_gap_gb": 16.13,
                    "psutil_available_gap_gb": 16.13,
                    "preflight_memory_source": "psutil_available",
                    "did_not_launch": True,
                    "launch_decision": "do_not_launch",
                    "case_count": 14,
                    "selected_cases": [
                        "chat_off",
                        "chat_off_rep1",
                        "chat_off_no_punct_rep1",
                        "chat_off_bundle_defaults",
                        "chat_on",
                        "chat_on_rep1",
                        "chat_max",
                        "responses_off",
                        "responses_off_rep1",
                        "responses_off_no_punct_rep1",
                        "responses_off_bundle_defaults",
                        "responses_on",
                        "responses_on_rep1",
                        "legacy_completion_raw",
                    ],
                },
            }
        },
    }


def test_release_regression_manifest_covers_required_domains():
    manifest = build_manifest()
    domains = {row["domain"] for row in manifest["rows"]}

    assert REQUIRED_RELEASE_DOMAINS.issubset(domains)


def test_release_regression_manifest_has_stable_unique_ids():
    manifest = build_manifest()
    row_ids = [row["id"] for row in manifest["rows"]]

    assert len(row_ids) == len(set(row_ids))
    assert all(row_id == row_id.lower() for row_id in row_ids)
    assert all(" " not in row_id for row_id in row_ids)


def test_release_regression_manifest_artifacts_are_unique_per_row():
    manifest = build_manifest()

    for row in manifest["rows"]:
        artifacts = row["artifacts"]
        assert len(artifacts) == len(set(artifacts)), row["id"]


def test_release_regression_manifest_artifacts_are_concrete_files():
    manifest = build_manifest()

    for row in manifest["rows"]:
        for artifact in row["artifacts"]:
            assert not artifact.endswith("/"), f"{row['id']} lists directory artifact {artifact}"


def test_release_regression_manifest_tracks_no_fake_sampler_policy():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    assert "generation-defaults-no-hidden-forcing" in rows
    assert rows["generation-defaults-no-hidden-forcing"]["domain"] == "generation_defaults"
    assert "generation_config.json" in " ".join(rows["generation-defaults-no-hidden-forcing"]["proves"])
    assert "jang_config.json" in " ".join(rows["generation-defaults-no-hidden-forcing"]["proves"])


def test_release_regression_manifest_tracks_new_chat_output_cap_inheritance_guard():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "new-chat" in joined
    assert "model-owned maxTokens" in joined
    assert "current-max-output-context-contract-20260522-chat-auto-server-default.json" in joined


def test_release_regression_manifest_tracks_server_chat_max_output_boundary():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "server startup maxTokens and chat maxTokens remain independent" in joined
    assert "Auto chat Max Tokens omits per-request output caps" in joined
    assert "new chat output caps are not inherited or made sticky" in joined
    assert "below or above the server startup default" in joined
    assert "persisted chat maxTokens cannot relaunch server" in joined
    assert "current-max-output-context-contract-20260522-persisted-chat-output-cap.json" in joined
    assert "current-max-output-context-contract-20260522-new-chat-output-cap-nonsticky.json" in joined
    assert "current-max-output-context-contract-20260522-responses-output-boundary.json" in joined
    assert "current-max-output-context-contract-20260522-request-default-mutation.json" in joined
    assert "Streaming Chat/Responses output caps do not mutate the server startup default" in joined
    assert "current-max-output-context-contract-20260523-streaming-mutation.json" in joined
    assert "Responses maxTokens below or above the server startup default remain request scoped" in joined
    assert "Responses Auto does not synthesize max_output_tokens" in joined
    assert "DB-cleared NULL chat maxTokens stays Auto for Chat Completions and Responses" in joined
    assert "current-max-output-context-contract-20260523-null-chat-cap.json" in joined
    assert "do not mutate the server startup default" in joined
    assert "Raw API non-positive max_tokens/max_output_tokens are rejected" in joined
    assert "chat reset does not convert model max_new_tokens" in joined
    assert "string-shaped legacy session maxTokens values are cleared" in joined
    assert "current-max-output-context-contract-20260522-reset-policy-string-legacy.json" in joined
    assert "current-max-output-context-contract-20260522-api-validator-caps.json" in joined
    assert "API gateway output-budget and context-budget paths are source-hashed" in joined
    assert "DSV4 request-budget helper is source-hashed with the max-output boundary gate" in joined
    assert "Casual preset maxTokens is documented as an explicit server output cap" in joined
    assert "current-max-output-context-contract-20260522-casual-server-output-cap.json" in joined


def test_release_regression_manifest_tracks_multifamily_live_workflow_gate():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-live-multiturn-soak"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    for model_row in (
        "gemma4_crack",
        "minimax_m27_tq_k",
        "qwen36_moe_crack",
        "zaya_jangtq2",
        "zaya_vl_jangtq4",
        "ling_flash_tq",
        "nemotron_omni_tq2",
        "dsv4_jang_local",
    ):
        assert model_row in joined

    for contract in (
        "multi-turn",
        "tool",
        "reasoning parser",
        "prefix",
        "cache",
        "memory",
        "wrong-language",
        "generation_config.json",
        "jang_config.json",
        "VL",
        "video",
        "Chat Completions",
        "Responses",
        "Anthropic",
        "Ollama",
        "stop/disconnect",
        "full visible output",
        "full reasoning output",
        "tail review",
    ):
        assert contract in joined

    assert "revalidates visible text for unexpected CJK" in joined
    assert "--include-tools" in joined
    assert "tool_choice=required" in joined
    assert "current-all-local-model-smoke-zaya-text-bundled-20260524/summary.json" in joined
    assert "current-all-local-model-smoke-zaya-text-bundled-toolprobe-20260525/summary.json" in joined
    assert "current-all-local-model-smoke-zaya-vl-bundled-20260524/summary.json" in joined
    assert (
        "current-all-local-model-smoke-zaya-vl-jangtq4-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-zaya-vl-jangtq4-bundled-toolprobe-20260525/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-zaya-vl-jangtq4-true-bundled-toolprobe-media-sentinel-20260525/summary.json"
        in joined
    )
    assert "ZAYA text MXFP4 tool probe remains diagnostic" in joined
    assert "ZAYA-VL JANGTQ4 true-bundled tool probe passes tool_required" in joined
    assert (
        "ZAYA-VL JANGTQ4 true-bundled media-sentinel smoke passes ACK/cache, tool calls, and VL colors"
        in joined
    )
    assert (
        "2026-05-26 reasoning-enabled rerun is diagnostic-fail because reasoning_on produced reasoning-only empty visible output"
        in joined
    )
    assert (
        "prompt-sensitive no-media control returned image"
        in joined
    )
    assert "semantic text-after-image isolation" not in joined
    assert "ZAYA-VL JANGTQ4 bundled smoke reproduces the same exact ACK/template failure" not in joined
    assert "Nemotron Omni Nano failed prompt variants remain diagnostic" in joined
    assert "so nemotron remains attempted-but-not-covered" not in joined
    assert (
        "current-all-local-model-smoke-nemotron-omni-jangtq-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-nemotron-omni-jangtq-explicit-nomedia-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-nemotron-omni-jangtq-video-bundled-20260526-rerun/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-nemotron-omni-jangtq-bundled-toolprobe-20260525/summary.json"
        in joined
    )
    assert "Nemotron Omni Nano bundled tool probe passes tool_required" in joined
    assert "current-nemotron-omni-no-media-prompt-variants-20260524/result.json" in joined
    assert "current-nemotron-omni-no-media-system-prompt-diagnostic-20260524/result.json" in joined
    assert (
        "current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260525-rerun/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-ling-bailing-jangtq-bundled-toolprobe-20260525/summary.json"
        in joined
    )
    assert "Ling/Bailing bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-gemma4-26b-jang4m-crack-video-capfix-bundled-20260526/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-toolprobe-currentmodality-20260526/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-toolprobe-20260525/summary.json"
        not in CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS.values()
    )
    assert "Gemma4 bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260525-rerun/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-toolprobe-20260525/summary.json"
        in joined
    )
    assert "Qwen3.6 MXFP4 bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-hy3-jangtq2-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-hy3-jangtq2-bundled-toolprobe-20260525/summary.json"
        in joined
    )
    assert "Hy3 bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-minimax-small-jangtq-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-minimax-small-jangtq-bundled-20260525-rerun/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-minimax-small-jangtq-bundled-toolprobe-20260525/summary.json"
        in joined
    )
    assert "MiniMax bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-zaya-text-bundled-toolprobe-20260525/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-zaya-vl-jangtq4-bundled-toolprobe-20260525/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-zaya-vl-jangtq4-true-bundled-toolprobe-media-sentinel-20260525/summary.json"
        in joined
    )
    assert "ZAYA text MXFP4 bundled tool probe remains a diagnostic failure" in joined
    assert "--only DeepSeek-V4-Flash-JANGTQ-K" in joined
    assert (
        "current-all-local-model-smoke-dsv4-jangtq-k-bundled-cachehit-20260524/summary.json"
        in joined
    )
    assert "--only ZAYA1-VL-8B-MXFP4" in joined
    assert "--only ZAYA1-VL-8B-JANGTQ4" in joined
    assert "--only Nemotron-Omni-Nano-JANGTQ-CRACK" in joined
    assert "--only Ling-2.6-flash-JANGTQ" in joined
    assert "--only Gemma-4-26B-A4B-it-JANG_4M-CRACK" in joined
    assert "--only Qwen3.6-27B-MXFP4-CRACK" in joined
    assert "--only Hy3-preview-JANGTQ2" in joined
    assert "--only MiniMax-M2.7-Small-JANGTQ" in joined


def test_release_regression_manifest_tracks_covered_live_smoke_artifacts():
    artifacts = set(CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.values())

    assert (
        "build/current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260525-rerun/summary.json"
        in artifacts
    )
    assert (
        "build/current-all-local-model-smoke-gemma4-26b-jang4m-crack-video-capfix-bundled-20260526/summary.json"
        in artifacts
    )
    assert (
        "build/current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260525-rerun/summary.json"
        in artifacts
    )
    assert (
        "build/current-all-local-model-smoke-hy3-jangtq2-bundled-toolprobe-20260525/summary.json"
        in artifacts
    )
    assert (
        "build/current-all-local-model-smoke-minimax-small-jangtq-bundled-20260525-rerun/summary.json"
        in artifacts
    )
    assert (
        "build/current-all-local-model-smoke-dsv4-jangtq-k-bundled-cachehit-20260524/summary.json"
        in artifacts
    )
    assert (
        "build/current-all-local-model-smoke-nemotron-omni-jangtq-video-bundled-20260526-rerun/summary.json"
        in artifacts
    )


def test_release_regression_manifest_tracks_current_post_budget_edge_proof_sweep():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    for row_id, artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.items():
        assert artifact in rows[row_id]["artifacts"], row_id


def test_release_regression_manifest_current_sweep_uses_latest_live_smoke_artifacts():
    current_artifacts = [
        CURRENT_REGRESSION_SUITE_ARTIFACT,
        CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
        *CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values(),
    ]
    manifest = build_manifest()
    row_text = json.dumps(manifest["rows"])
    joined = "\n".join(current_artifacts)

    assert "current-regression-suite-20260528-installed-aggregate-stale.json" not in joined
    assert "current-regression-suite-20260528-epipe-aggregate-guard.json" not in joined
    assert "current-regression-suite-20260528-dsv4-continue-refresh.json" not in joined
    assert "current-regression-suite-20260531-childstream-epipe-guard.json" in joined
    assert "current-regression-suite-20260530-bundled-sync-step37-projector-rerun.json" not in joined
    assert "current-regression-suite-20260530-bundled-sync-step37-projector.json" not in joined
    assert "current-regression-suite-20260529-step37-text-bridge.json" not in joined
    assert "current-regression-suite-20260528-prepackage-gate.json" not in joined
    assert "current-regression-suite-20260528-release-ready-dmg-gate.json" not in joined
    assert "current-regression-suite-20260528-release-gate-wired.json" not in joined
    assert "current-regression-suite-20260528-release-ready-top-level.json" not in joined
    assert "current-regression-suite-20260528-dsv4-memory-refresh.json" not in joined
    assert "current-regression-suite-20260528-signing-detail-ledger.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260531-childstream-epipe-installed-sync.json" in joined
    assert "current-installed-app-runtime-parity-audit-20260531-childstream-epipe-installed-sync.json" in row_text
    assert "current-installed-app-runtime-parity-audit-20260528-epipe-aggregate-guard.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260528-epipe-aggregate-guard.json" not in row_text
    assert "current-staged-app-runtime-parity-audit-20260528-staged-runtime-recheck.json" in joined
    assert "current-staged-app-runtime-parity-audit-20260528-installed-aggregate-stale.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260528-epipe-postrefresh.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260528-epipe-diagnostic-scan.json" not in joined
    assert "current-regression-suite-20260528-epipe-diagnostic-scan.json" not in joined
    assert "current-regression-suite-20260528-named-family-cache-matrix.json" not in joined
    assert "current-regression-suite-20260528-dsv4-blocker-evidence.json" not in joined
    assert "current-regression-suite-20260528-ollama-embedding-timeout.json" not in joined
    assert "current-regression-suite-20260528-local-release-evidence.json" not in joined
    assert "current-regression-suite-20260528-release-clearance-open.json" not in joined
    assert "current-regression-suite-20260528-api-ollama-embeddings-single-model.json" not in joined
    assert "current-regression-suite-20260528-dsv4-legacy-clearance-cleanup.json" not in joined
    assert "current-regression-suite-20260528-signing-root-cause.json" not in joined
    assert "current-regression-suite-20260528-current-cohesive-audit.json" not in joined
    assert "current-regression-suite-20260528-nonmimo-zaya-direct.json" not in joined
    assert "current-regression-suite-20260528-ollama-chat-epipe-dsv4-preflight.json" not in joined
    assert "current-regression-suite-20260528-ollama-chat-epipe-guard.json" not in joined
    assert "current-regression-suite-20260528-dsv4-0625-preflight.json" not in joined
    assert "current-regression-suite-20260528-current-live-smoke-pointers.json" not in joined
    assert "current-regression-suite-20260528-dsv4-0600-preflight.json" not in joined
    assert "current-regression-suite-20260528-userdata-epipe-scan.json" not in joined
    assert "current-regression-suite-20260528-dsv4-preflight-postinstall.json" not in joined
    assert "current-regression-suite-20260528-installed-ipc-epipe-guard.json" not in joined
    assert "current-regression-suite-20260528-ipc-epipe-request-guard.json" not in joined
    assert "current-regression-suite-20260528-dsv4-identifier-candidates.json" not in joined
    assert "current-regression-suite-20260528-issue175-179-boundary-pass.json" not in joined
    assert "current-regression-suite-20260528-epipe-request-end-guard.json" not in joined
    assert "current-regression-suite-20260528-issue179-local-repro-clean.json" not in joined
    assert "current-regression-suite-20260528-issue179-reporter-shape-guard.json" not in joined
    assert "current-regression-suite-20260528-epipe-socket-guard.json" not in joined
    assert "current-regression-suite-20260528-generation-family-matrix.json" not in joined
    assert "current-regression-suite-20260528-cache-swa-hca-csa-row.json" not in joined
    assert "current-regression-suite-20260528-dsv4-memory-gap-preflight.json" not in joined
    assert "current-regression-suite-20260528-ollama-proxy-error-guard.json" not in joined
    assert "current-regression-suite-20260528-reporter-parity-collector.json" not in joined
    assert "current-regression-suite-20260528-reporter-parity-dsv4-preflight-refresh.json" not in joined
    assert "current-regression-suite-20260528-ollama-epipe-multiline-installed.json" not in joined
    assert "current-regression-suite-20260528-issue179-installed-session-proof.json" not in joined
    assert "current-regression-suite-20260528-issue179-launch-config-proof.json" not in joined
    assert "current-regression-suite-20260528-issue179-request-shape-proof.json" not in joined
    assert "current-regression-suite-20260528-issue179-installed-launcher-proof.json" not in joined
    assert "current-regression-suite-20260528-dsv4-preflight-refresh.json" not in joined
    assert "current-regression-suite-20260528-admin-sleep-sourcehash.json" not in joined
    assert "current-regression-suite-20260528-issue179-econnreset-boundary.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260531-live-refresh.json" in joined
    assert "current-real-ui-dsv4-memory-preflight-20260530-local-refresh.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-continue-refresh.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-epipe-install.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-0625-recheck.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-0600-recheck.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-postinstall.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260527.json" not in joined
    assert "current-regression-suite-20260527-local-only-mimo-boundary.json" not in joined
    assert "current-regression-suite-20260527-after-proof-source-hash-coverage.json" not in joined
    assert "current-regression-suite-20260527-after-think-xml-registry-fix.json" not in joined
    assert "current-regression-suite-20260527-issue179-real-ui-diagnostic-manifest.json" not in joined
    assert "current-regression-suite-20260527-live-matrix-audit-final.json" not in joined
    assert "current-regression-suite-20260527-mimo-v2-jang2l-noheavy-proof-rerun.json" not in joined
    assert "current-regression-suite-20260527-issues-175-178-bundled-sync.json" not in joined
    assert "current-regression-suite-20260527-zaya-vl-point-leak-proof-tightening.json" not in joined
    assert "current-regression-suite-20260527-zaya-vl-cachecontrols.json" not in joined
    assert "current-regression-suite-20260527-gemma4-speed-overclaim-fixed.json" not in joined
    assert "current-regression-suite-20260527-real-ui-family-expansion.json" not in joined
    assert "current-regression-suite-20260527-real-ui-zaya-vl-image.json" not in joined
    assert "current-regression-suite-20260526-real-ui-live-model-slice.json" not in joined
    assert "current-regression-suite-20260526-dev-ui-wiring.json" not in joined
    assert "current-regression-suite-20260526-settings-audit.json" not in joined
    assert "current-regression-suite-20260525-smoke-reasoning-loop-guard.json" not in joined
    assert "current-regression-suite-20260525-gemma-short-nocache-boundary.json" not in joined
    assert "current-regression-suite-20260525-gemma-internal-speed-gate.json" not in joined
    assert "current-regression-suite-20260525-dsv4-copy-block-boundary.json" not in joined
    assert "current-regression-suite-20260525-dsv4-prompt-guard-boundary.json" not in joined
    assert "current-regression-suite-20260525-dsv4-nocache-ab-boundary.json" not in joined
    assert "current-regression-suite-20260525-dsv4-default-cache-raw-boundary.json" not in joined
    assert "current-regression-suite-20260525-dsv4-bundle-defaults-cjk-wide.json" not in joined
    assert "current-regression-suite-20260525-dsv4-dryrun-policy-wired.json" not in joined
    assert "current-regression-suite-20260525-cjk-smoke-guard.json" not in joined
    assert "current-regression-suite-20260525-gemma-installed-speed-boundary.json" not in joined
    assert "current-regression-suite-20260524-openai-single-model-streaming-audit.json" not in joined
    assert "current-api-surface-contract-20260531-nested-epipe-childstream-refresh.json" in joined
    assert "current-api-surface-contract-20260529-single-model-transition-lock.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-embedding-timeout.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-embeddings-single-model.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-nested-disconnect.json" not in joined
    assert "current-api-surface-contract-20260528-cache-endpoints-autoswitch-expanded.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-client-close-epipe-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ipc-epipe-request-guard.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-request-end-guard.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-socket-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-proxy-error-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-epipe-multiline-guard.json" not in joined
    assert "current-api-surface-contract-20260527-cache-endpoint-autoswitch-proof.json" not in joined
    assert "current-api-surface-contract-20260526-single-model-auto-switch-review.json" not in joined
    assert "current-api-surface-contract-20260525-single-model-responses-deltas.json" not in joined
    assert "current-packaged-integrity-contract-20260531-childstream-epipe-refresh.json" in joined
    assert "current-packaged-integrity-contract-20260530-bundled-sync-after-step37-projector.json" not in joined
    assert "current-packaged-integrity-contract-20260529-step37-text-bridge.json" not in joined
    assert "current-packaged-integrity-contract-20260528-installed-aggregate-stale.json" not in joined
    assert "current-packaged-integrity-contract-20260528-prepackage-gate.json" not in joined
    assert "current-packaged-integrity-contract-20260528-release-ready-dmg-gate.json" not in joined
    assert "current-packaged-integrity-contract-20260528-release-ready-gate.json" not in joined
    assert "current-packaged-integrity-contract-20260528-signing-keychain-root-cause.json" not in joined
    assert "current-packaged-integrity-contract-20260528-signing-root-cause.json" not in joined
    assert "current-packaged-integrity-contract-20260528-ollama-chat-epipe-guard.json" not in joined
    assert "current-packaged-integrity-contract-20260528-ipc-epipe-request-guard.json" not in joined
    assert "current-packaged-integrity-contract-20260528-epipe-socket-guard-installed-sync.json" not in joined
    assert "current-packaged-integrity-contract-20260527-after-think-xml-registry-fix-rerun.json" not in joined
    assert "current-packaged-integrity-contract-20260527-issues-175-178-bundled-sync-rerun.json" not in joined
    assert "current-packaged-integrity-contract-20260527-mimo-v2-jang2l-bundled-import-rerun.json" not in joined
    assert "current-packaged-integrity-contract-20260526-bundled-release-proof.json" not in joined
    assert "current-packaged-integrity-contract-20260525-additional-args-guard.json" not in joined
    assert "current-regression-suite-20260524-crossfamily-cleared-dsv4-open.json" not in joined
    assert "current-generation-defaults-contract-20260531-post-step-lfm-refresh.json" in joined
    assert "current-generation-defaults-contract-20260526-settings-audit.json" not in joined
    assert "current-max-output-context-contract-20260531-post-step-lfm-refresh.json" in joined
    assert "current-max-output-context-contract-20260526-settings-audit.json" not in joined
    assert "current-reasoning-template-contract-20260526-settings-audit.json" in joined
    assert "current-generation-defaults-contract-20260525-additional-args-guard.json" not in joined
    assert "current-max-output-context-contract-20260524-after-gemma4-telemetry-final.json" not in joined
    assert "current-reasoning-template-contract-20260523-post-budget-edge.json" not in joined
    assert "current-regression-suite-20260524-gemma4-visible-crossfamily-open.json" not in joined
    assert "current-packaged-integrity-contract-20260524-live-smoke-open.json" not in joined
    assert "current-packaged-integrity-contract-20260524-gemma4-smoke-dsv4-open.json" not in joined
    assert "current-packaged-integrity-contract-20260524-after-gemma4-mixed-swa-telemetry.json" not in joined
    assert "current-regression-suite-20260524-gemma4-visible-open.json" not in joined
    assert "current-regression-suite-20260524-after-dsv4-neutral-reppen.json" not in joined
    assert "current-generation-defaults-contract-20260524-after-dsv4-neutral-reppen.json" not in joined
    assert "current-max-output-context-contract-20260524-after-dsv4-neutral-reppen.json" not in joined
    assert "current-regression-suite-20260524-default-cache-live-review.json" not in joined
    assert "current-regression-suite-20260524-zaya-vl-smoke-open.json" not in joined
    assert "current-regression-suite-20260524-nemotron-smoke-open.json" not in joined
    assert "current-packaged-integrity-contract-20260524-default-cache-live-review.json" not in joined
    assert "current-generation-defaults-contract-20260524-after-ling-topk-policy.json" not in joined
    assert "current-max-output-context-contract-20260524-after-chatmax-dsv4-server-edits.json" not in joined


def test_release_regression_manifest_tracks_electron_dev_ui_wiring_proof():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["electron-dev-ui-tools-reasoning-wiring"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["mode"] == "dev-ui"
    assert "VMLINUX_LIVE_PROOF_BASENAME=2026-05-26-live-chat-tools-reasoning" in joined
    assert "panel/scripts/live-chat-tools-reasoning-proof.mjs" in joined
    assert "docs/internal/agent-notes/2026-05-26-live-chat-tools-reasoning-proof.json" in joined
    assert "mock-server UI wiring proof only" in joined
    assert "does not clear live-model language quality" in joined
    assert "reasoning segments" in joined
    assert "read_image/read_video" in joined
    assert "Block Disk Cache L2" in joined


def test_release_regression_manifest_tracks_real_electron_ui_live_model_slice():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["electron-dev-ui-real-model-live-slice"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["mode"] == "dev-ui-live-model"
    assert "panel/scripts/live-real-ui-model-proof.mjs" in joined
    assert "VMLINUX_REAL_UI_MODEL_PATH" in joined
    assert "ZAYA1-8B-MXFP4" in joined
    assert "real vmlx_engine server" in joined
    assert "window.api.chat.sendMessage" in joined
    assert "not enough to clear the full cross-family real-UI blocker" in joined
    assert "current-real-ui-live-model-zaya-text-20260526-proof.json" in joined
    assert "Ling-2.6-flash-JANGTQ" in joined
    assert "current-real-ui-live-model-ling-bailing-jangtq-20260526-proof.json" in joined
    assert "Gemma-4-26B-A4B-it-JANG_4M-CRACK" in joined
    assert "current-real-ui-live-model-gemma4-26b-jang4m-20260526-proof.json" in joined
    assert (
        "diagnostic-real-ui-live-model-gemma4-responses-tools-reasoning-20260527-proof.json"
        in joined
    )
    assert "Responses/tools/reasoning" in joined
    assert "mixed-SWA paged cache" in joined
    assert "affine MLX matmul" in joined
    assert "80 tok/s" in joined
    assert "Hy3-preview-JANGTQ2" in joined
    assert "current-real-ui-live-model-hy3-jangtq2-20260526-proof.json" in joined
    assert "TurboQuantKVCache/paged prefix cache" in joined
    assert "current-lfm25-direct-responses-runcommand-cache-boundary-20260530.json" in joined
    assert "server-clean while the real UI stream path remains malformed" in joined


def test_release_regression_manifest_real_ui_live_model_script_exists_and_uses_real_session_path():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "VMLINUX_REAL_UI_MODEL_PATH" in source
    assert "VMLINUX_REAL_UI_WIRE_API" in source
    assert "VMLINUX_REAL_UI_BUILTIN_TOOLS" in source
    assert "real_ui_tool_probe_1.txt" in source
    assert "real_ui_tool_probe_2.txt" in source
    assert "REAL_UI_LIVE_TOOL_ONE" in source
    assert "REAL_UI_LIVE_TOOL_TWO" in source
    assert "VMLINUX_REAL_UI_WORKING_DIRECTORY" in source
    assert "VMLINUX_REAL_UI_ENABLE_THINKING" in source
    assert "VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS" in source
    assert "VMLINUX_REAL_UI_CHECK_MEDIA" in source
    assert "VMLINUX_REAL_UI_IMAGE_DATA_URL" in source
    assert "VMLINUX_REAL_UI_IMAGE_EXPECT_REGEX" in source
    assert "VMLINUX_REAL_UI_CHECK_VIDEO" in source
    assert "VMLINUX_REAL_UI_VIDEO_DATA_URL" in source
    assert "VMLINUX_REAL_UI_VIDEO_EXPECT_REGEX" in source
    assert "provenSurfaces" in source
    assert "serverCacheControls" in source
    assert "sectionButtons = [...document.querySelectorAll('button')]" in source
    assert "const normalized = text.replace(/\\\\s+/g, ' ').trim()" in source
    assert "titleWithoutDisclosure === title" in source
    assert "new MouseEvent('click'" in source
    assert "sectionClickResults" in source
    assert "media: mediaEvidence" in source
    assert "imageVerified" in source
    assert "videoVerified" in source
    assert "imageSemanticVerified" in source
    assert "videoSemanticVerified" in source
    assert "sendErrors" in source
    assert "rendererFailureStage" in source
    assert "status: rendererResult.rendererFailureStage ? 'fail' : undefined" in source
    assert "failureStage: rendererResult.rendererFailureStage || undefined" in source
    assert "persistedToolsByMessage" in source
    assert "persistedReasoningByMessage" in source
    assert "persistedReasoningText" in source
    assert "reasoningRawParserTagLeak" in source
    assert "reasoningCjkLeakCount" in source
    assert "reasoningKoreanLeakCount" in source
    assert "reasoningNumericRunCount" in source
    assert "numeric/list-like garbage leaked into reasoning segments" in source
    assert "--enable-auto-tool-choice" in source
    assert "--tool-call-parser" in source
    assert "window.api.sessions.create" in source
    assert "window.api.sessions.start" in source
    assert "window.api.chat.sendMessage" in source
    assert "window.api.performance.health" in source
    assert "window.api.cache.stats" in source
    assert "native_cache_status" in source
    assert "cache_endpoint_stats" in source
    assert "health.native_cache" in source
    assert "block_disk_cache" in source
    assert "cache_totals" in source
    assert "VMLINUX_REAL_UI_APP_PATH" in source
    assert "installedAppPath" in source
    assert "uiLaunchMode" in source
    assert "Contents', 'MacOS', 'vMLX'" in source
    assert "installed_app_ui" in source
    assert "terminateProcess" in source
    assert "childProcessTree" in source
    assert "terminateProcessTree" in source
    assert "process.kill(pid, signal)" in source
    assert "removeTemporaryTree(userDataDir)" in source
    assert "removeTemporaryTree(runDir)" in source
    assert "maxRetries" in source
    assert "ENOTEMPTY" in source
    assert "rawParserTagLeak" in source
    assert "<\\\\|point_start\\\\|>" in source
    assert "<\\\\|point_end\\\\|>" in source
    assert "cjkLeakCount" in source
    assert "clearTimeout(timer)" in source
    assert "current-real-ui-live-model-zaya-text-20260526" in source
    assert "server process exited before health" in source
    assert "server.proc.exitCode" in source
    assert "server.logs.slice(-80)" in source


def test_release_regression_manifest_real_ui_cache_checker_is_model_specific():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "VMLINUX_REAL_UI_CACHE_EXPECT_REGEX" in source
    assert "VMLINUX_REAL_UI_EXPECT_PAGED_CACHE_LOCKED" in source
    assert "cacheExpectRegex" in source
    assert "expectPagedCacheLocked" in source
    assert "new RegExp(cacheExpectRegex, 'i').test(bodyText)" in source

    verified_block = source.split("const verified = ", 1)[1].split("return {", 1)[0]
    assert "ZAYA typed CCA cache requires paged cache" not in verified_block
    assert "This architecture requires native/paged cache" not in verified_block


def test_release_regression_manifest_real_ui_script_preserves_tool_probe_file_evidence():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "const toolProbeFiles = {}" in source
    assert "function cacheReconstructionClean(result)" in source
    assert "worker-side paged cache reconstruction failed" in source
    result_block = source.split("const result = {", 2)[2].split("result.provenSurfaces", 1)[0]
    assert result_block.index("...rendererResult") < result_block.index("toolProbeFiles")
    semantics_block = source.split("function namedToolProbeSemanticsOk(result)", 1)[1].split(
        "function countMatches",
        1,
    )[0]
    assert "real_ui_tool_probe_2.txt" in semantics_block
    assert "REAL_UI_LIVE_TOOL_TWO" in semantics_block
    assert "&& fileSemanticsOk" in semantics_block


def test_release_regression_manifest_real_ui_script_records_stream_text_trace():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "streamTraceByMessage" in source
    assert "lastFullContent" in source
    assert "lastReasoningContent" in source
    assert "streamTrace: rendererResult.streamTraceByMessage" in source


def test_release_regression_manifest_real_ui_script_records_request_contract():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    result_block = source.split("const result = {", 2)[2].split(
        "result.visibleAssistantTurnsComplete",
        1,
    )[0]
    assert "requestContract" in result_block
    assert "promptOne" in result_block
    assert "promptTwo" in result_block
    assert "requestMaxTokens" in result_block
    assert "maxToolIterations" in result_block
    assert "toolResultMaxChars" in result_block
    assert "imageExpectRegex" in result_block
    assert "videoExpectRegex" in result_block
    assert "cacheExpectRegex" in result_block


def test_release_regression_manifest_real_ui_default_image_fixture_has_valid_png_crc():
    import base64
    import binascii
    import struct

    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")
    match = re.search(r"\|\| 'data:image/png;base64,([^']+)'", source)
    assert match, "default real UI image data URL not found"

    png = base64.b64decode(match.group(1), validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")

    offset = 8
    seen_iend = False
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        chunk_type = png[offset + 4 : offset + 8]
        chunk_data = png[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(
            ">I",
            png[offset + 8 + length : offset + 12 + length],
        )[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        assert actual_crc == expected_crc, f"invalid PNG CRC for {chunk_type!r}"
        offset += 12 + length
        if chunk_type == b"IEND":
            seen_iend = True
            break

    assert seen_iend
    assert offset == len(png)


def test_release_regression_manifest_real_ui_script_serializes_unset_thinking_contract():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    result_block = source.split("const result = {", 2)[2].split(
        "result.visibleAssistantTurnsComplete",
        1,
    )[0]
    assert "enableThinking: enableThinkingOverride ?? null" in result_block


def test_release_regression_manifest_real_ui_script_rejects_reasoning_only_visible_turns():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "visibleAssistantAfterEachUser" in source
    assert "requested reasoning turn ended with empty visible assistant content" in source
    assert "visibleAssistantTurnsComplete" in source


def test_release_regression_manifest_real_ui_script_rejects_visible_tool_probe_corruption():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    semantics_block = source.split("function namedToolProbeSemanticsOk(result)", 1)[1].split(
        "function countMatches",
        1,
    )[0]
    assert "visibleToolSemanticsOk" in semantics_block
    assert "real_ui_tool_probe_2.txt" in semantics_block
    assert "REAL_UI_LIVE_TOOL_ONE" in semantics_block
    assert "RE:AL_UI_LIVE_TOOL_TWO" in semantics_block
    assert "strictExactReplyOk" in semantics_block
    assert "reply exactly:" in semantics_block
    assert "&& visibleToolSemanticsOk" in semantics_block
    assert "&& strictExactReplyOk" in semantics_block


def test_release_regression_manifest_current_sweep_rejects_missing_dev_ui_proof(tmp_path):
    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["dev_ui_proof"]["status"] == "fail"
    assert (
        "docs/internal/agent-notes/2026-05-26-live-chat-tools-reasoning-proof.json"
        in result["dev_ui_proof"]["missing"]
    )


def test_release_regression_manifest_current_sweep_rejects_fake_dev_ui_screenshots(tmp_path):
    _write_passing_dev_ui_proof_artifacts(tmp_path)
    screenshot = tmp_path / CURRENT_DEV_UI_PROOF_ARTIFACTS["chat_settings_screenshot"]
    screenshot.write_bytes(b"png-placeholder")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["dev_ui_proof"]["status"] == "fail"
    assert "invalid_png:chat_settings_screenshot" in result["dev_ui_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_wrong_dev_ui_worktree(tmp_path):
    _write_passing_dev_ui_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_DEV_UI_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["repoDir"] = "/tmp/wrong-vmlx-worktree"
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["dev_ui_proof"]["status"] == "fail"
    assert "proof_repo_dir_mismatch" in result["dev_ui_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_missing_real_ui_live_model_proof(tmp_path):
    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-20260526-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-cachecontrols-20260527-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-image-20260527-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-20260526-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-20260526-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-qwen36-mxfp4-crack-20260527-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )


def test_release_regression_manifest_real_ui_live_model_rows_include_ling_bailing_slice():
    rows = CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS

    assert rows["zaya_text"]["model_name"] == "ZAYA1-8B-MXFP4"
    assert rows["zaya_text_cachecontrols"]["model_name"] == "ZAYA1-8B-MXFP4"
    assert rows["zaya_text_cachecontrols"]["family"] == "zaya_text"
    assert (
        rows["zaya_text_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-cachecontrols-20260527-proof.json"
    )
    assert rows["zaya_text_responses_tools"]["model_name"] == "ZAYA1-8B-MXFP4"
    assert rows["zaya_text_responses_tools"]["family"] == "zaya_text"
    assert (
        rows["zaya_text_responses_tools"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    assert rows["zaya_vl_image"]["model_path"] == "/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_image"]["model_name"] == "ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_image"]["family"] == "zaya_vl"
    assert (
        rows["zaya_vl_image"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-image-20260527-proof.json"
    )
    assert rows["zaya_vl_cachecontrols"]["model_path"] == "/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_cachecontrols"]["model_name"] == "ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_cachecontrols"]["family"] == "zaya_vl"
    assert (
        rows["zaya_vl_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-cachecontrols-20260527-proof.json"
    )
    assert rows["zaya_vl_responses_tools_cachecontrols"]["model_path"] == "/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_responses_tools_cachecontrols"]["model_name"] == "ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_responses_tools_cachecontrols"]["family"] == "zaya_vl"
    assert (
        rows["zaya_vl_responses_tools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    assert all(
        "point-leak" not in row["proof"]
        for row in rows.values()
    )
    assert all(
        "zaya-vl-responses-tools-cachecontrols-localonly" not in row["proof"]
        for row in rows.values()
    )
    assert rows["ling_bailing_jangtq"]["model_path"] == "/Users/example/models/JANGQ/Ling-2.6-flash-JANGTQ"
    assert rows["ling_bailing_jangtq"]["model_name"] == "Ling-2.6-flash-JANGTQ"
    assert (
        rows["ling_bailing_jangtq"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-20260526-proof.json"
    )
    assert rows["ling_bailing_jangtq_responses_tools_cachecontrols"]["model_path"] == "/Users/example/models/JANGQ/Ling-2.6-flash-JANGTQ"
    assert rows["ling_bailing_jangtq_responses_tools_cachecontrols"]["model_name"] == "Ling-2.6-flash-JANGTQ"
    assert rows["ling_bailing_jangtq_responses_tools_cachecontrols"]["family"] == "ling_bailing"
    assert (
        rows["ling_bailing_jangtq_responses_tools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-responses-filesemantic-20260530-proof.json"
    )
    assert (
        rows["gemma4"]["model_path"]
        == "/Users/example/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK"
    )
    assert rows["gemma4"]["model_name"] == "Gemma-4-26B-A4B-it-JANG_4M-CRACK"
    assert (
        rows["gemma4"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-stricttools-max768-visible-20260531-proof.json"
    )
    assert rows["gemma4_reasoning"]["model_name"] == "Gemma-4-26B-A4B-it-JANG_4M-CRACK"
    assert rows["gemma4_reasoning"]["family"] == "gemma4"
    assert (
        rows["gemma4_reasoning"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-reasoningonly-max768-visible-20260531-proof.json"
    )
    assert rows["gemma4_cachecontrols"]["model_name"] == "Gemma-4-26B-A4B-it-JANG_4M-CRACK"
    assert rows["gemma4_cachecontrols"]["family"] == "gemma4"
    assert (
        rows["gemma4_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-gemma4-cachecontrols-20260527-proof.json"
    )
    assert (
        rows["qwen36_mxfp4_crack"]["model_path"]
        == "/Users/example/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK"
    )
    assert rows["qwen36_mxfp4_crack"]["model_name"] == "Qwen3.6-27B-MXFP4-CRACK"
    assert rows["qwen36_mxfp4_crack"]["family"] == "qwen36"
    assert (
        rows["qwen36_mxfp4_crack"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-qwen36-mxfp4-crack-20260527-proof.json"
    )
    assert (
        rows["qwen36_mxfp4_crack_video"]["model_path"]
        == "/Users/example/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK"
    )
    assert rows["qwen36_mxfp4_crack_video"]["model_name"] == "Qwen3.6-27B-MXFP4-CRACK"
    assert rows["qwen36_mxfp4_crack_video"]["family"] == "qwen36"
    assert (
        rows["qwen36_mxfp4_crack_video"]["proof"]
        == "docs/internal/agent-notes/diagnostic-real-ui-live-model-qwen36-mxfp4-crack-video-20260527-proof.json"
    )
    assert rows["qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols"]["model_path"] == (
        "/Users/example/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK"
    )
    assert (
        rows["qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols"]["model_name"]
        == "Qwen3.6-27B-MXFP4-CRACK"
    )
    assert rows["qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols"]["family"] == "qwen36"
    assert (
        rows["qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-qwen36-mxfp4-crack-responses-tools-reasoning-image-cachecontrols-localonly-20260527-proof.json"
    )
    assert rows["qwen36_mxfp4_crack_responses_stricttools_cachecontrols"]["model_path"] == (
        "/Users/example/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK"
    )
    assert (
        rows["qwen36_mxfp4_crack_responses_stricttools_cachecontrols"]["model_name"]
        == "Qwen3.6-27B-MXFP4-CRACK"
    )
    assert rows["qwen36_mxfp4_crack_responses_stricttools_cachecontrols"]["family"] == "qwen36"
    assert (
        rows["qwen36_mxfp4_crack_responses_stricttools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-qwen36-mxfp4-crack-responses-stricttools-image-cachecontrols-20260530-proof.json"
    )
    assert (
        rows["minimax_m27_small_jangtq"]["model_path"]
        == "/Users/example/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    )
    assert rows["minimax_m27_small_jangtq"]["model_name"] == "MiniMax-M2.7-Small-JANGTQ"
    assert rows["minimax_m27_small_jangtq"]["family"] == "minimax"
    assert (
        rows["minimax_m27_small_jangtq"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-jangtq-20260527-proof.json"
    )
    assert rows["minimax_m27_small_responses_tools_cachecontrols"]["model_path"] == (
        "/Users/example/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    )
    assert (
        rows["minimax_m27_small_responses_tools_cachecontrols"]["model_name"]
        == "MiniMax-M2.7-Small-JANGTQ"
    )
    assert rows["minimax_m27_small_responses_tools_cachecontrols"]["family"] == "minimax"
    assert (
        rows["minimax_m27_small_responses_tools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-tools-cachecontrols-localonly-20260527-proof.json"
    )
    assert rows["minimax_m27_small_responses_stricttools_cachecontrols"]["model_path"] == (
        "/Users/example/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    )
    assert (
        rows["minimax_m27_small_responses_stricttools_cachecontrols"]["model_name"]
        == "MiniMax-M2.7-Small-JANGTQ"
    )
    assert rows["minimax_m27_small_responses_stricttools_cachecontrols"]["family"] == "minimax"
    assert (
        rows["minimax_m27_small_responses_stricttools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    assert rows["minimax_m27_small_installed_responses_tools_cachecontrols_singleid"]["model_path"] == (
        "/Users/example/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    )
    assert (
        rows["minimax_m27_small_installed_responses_tools_cachecontrols_singleid"]["model_name"]
        == "MiniMax-M2.7-Small-JANGTQ"
    )
    assert (
        rows["minimax_m27_small_installed_responses_tools_cachecontrols_singleid"]["family"]
        == "minimax"
    )
    assert (
        rows["minimax_m27_small_installed_responses_tools_cachecontrols_singleid"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-installed-responses-tools-cachecontrols-singleid-20260528-proof.json"
    )
    assert "minimax_m27_jangtq_k_issue179" not in rows
    assert "minimax_m27_jangtq_k_issue179_installed" not in rows
    assert "minimax_m27_jangtq_k_issue179_installed_hi_auto" not in rows
    assert "minimax_m27_jangtq_k_issue179_installed_hi_thinking" not in rows
    assert "minimax_m27_jangtq_k_issue179_installed_512" not in rows
    manifest_rows = {row["id"]: row for row in build_manifest()["rows"]}
    manifest_row = manifest_rows["electron-dev-ui-real-model-live-slice"]
    joined = "\n".join(
        manifest_row["proves"] + manifest_row["commands"] + manifest_row["artifacts"]
    )
    assert CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT in joined
    assert "run_issue175_179_release_boundary_audit.py" in joined
    assert "GitHub issues #175-#179" in joined
    assert CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT in joined
    assert "run_issue179_minimax_k_root_cause_audit.py" in joined
    assert "run_issue179_responses_cancel_probe.py" in joined
    assert "Responses cancel-404 boundary" in joined
    assert CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT in joined
    assert "run_installed_app_runtime_parity_audit.py" in joined
    assert "installed /Applications/vMLX.app runtime parity audit" in joined
    assert CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT in joined
    assert "run_issue175_177_installed_runtime_audit.py" in joined
    assert "installed-app runtime parity on issues #175-#177" in joined
    assert CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT in joined
    assert "run_issue175_177_live_runtime_audit.py" in joined
    assert "installed-app live runtime audit on issues #175-#177" in joined
    assert (
        rows["nemotron_omni_nano_jangtq"]["model_path"]
        == "/Users/example/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK"
    )
    assert rows["nemotron_omni_nano_jangtq"]["model_name"] == "Nemotron-Omni-Nano-JANGTQ-CRACK"
    assert rows["nemotron_omni_nano_jangtq"]["family"] == "nemotron_omni"
    assert (
        rows["nemotron_omni_nano_jangtq"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-jangtq-20260527-proof.json"
    )
    assert rows["nemotron_omni_nano_responses_tools_reasoning_cachecontrols"]["model_path"] == (
        "/Users/example/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK"
    )
    assert (
        rows["nemotron_omni_nano_responses_tools_reasoning_cachecontrols"]["model_name"]
        == "Nemotron-Omni-Nano-JANGTQ-CRACK"
    )
    assert rows["nemotron_omni_nano_responses_tools_reasoning_cachecontrols"]["family"] == "nemotron_omni"
    assert (
        rows["nemotron_omni_nano_responses_tools_reasoning_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260527-proof.json"
    )
    assert rows["hy3_jangtq2"]["model_path"] == "/Users/example/models/JANGQ/Hy3-preview-JANGTQ2"
    assert rows["hy3_jangtq2"]["model_name"] == "Hy3-preview-JANGTQ2"
    assert rows["hy3_jangtq2"]["family"] == "hy3"
    assert (
        rows["hy3_jangtq2"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-20260526-proof.json"
    )
    assert rows["hy3_jangtq2_responses_tools_cachecontrols"]["model_path"] == (
        "/Users/example/models/JANGQ/Hy3-preview-JANGTQ2"
    )
    assert rows["hy3_jangtq2_responses_tools_cachecontrols"]["model_name"] == "Hy3-preview-JANGTQ2"
    assert rows["hy3_jangtq2_responses_tools_cachecontrols"]["family"] == "hy3"
    assert (
        rows["hy3_jangtq2_responses_tools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-filesemantic-20260530-proof.json"
    )
    assert rows["step37_flash_jang2l"]["model_path"] == (
        "/Users/example/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L"
    )
    assert rows["step37_flash_jang2l"]["model_name"] == "Step-3.7-Flash-JANG_2L"
    assert rows["step37_flash_jang2l"]["family"] == "step37"
    assert (
        rows["step37_flash_jang2l"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-tools-image-cachecontrols-after-direct-media-tool-filter-20260531-proof.json"
    )
    assert rows["step37_flash_jang2l_reasoning"]["model_path"] == (
        "/Users/example/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L"
    )
    assert rows["step37_flash_jang2l_reasoning"]["model_name"] == "Step-3.7-Flash-JANG_2L"
    assert rows["step37_flash_jang2l_reasoning"]["family"] == "step37"
    assert (
        rows["step37_flash_jang2l_reasoning"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-reasoning-max768-20260531-proof.json"
    )
    assert rows["lfm25_moe_a1b"]["model_path"] == (
        "/Users/example/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-JANG_2L"
    )
    assert rows["lfm25_moe_a1b"]["model_name"] == "LFM2.5-8B-A1B-JANG_2L"
    assert rows["lfm25_moe_a1b"]["family"] == "lfm25"
    assert (
        rows["lfm25_moe_a1b"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-lfm25-moe-a1b-jang2l-stricttools-chat-20260530-proof.json"
    )
    assert (
        rows["lfm25_moe_a1b_responses"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-lfm25-moe-a1b-jang2l-stricttools-responses-post-epipe-20260531-proof.json"
    )


def test_release_regression_manifest_real_ui_requires_step37_and_lfm25():
    assert "step37" in REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES
    assert "lfm25" in REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES

    assert REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["step37"] == (
        *tuple(
            surface
            for surface in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES
            if surface != "video_where_supported"
        ),
    )
    assert REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["lfm25"] == tuple(
        surface
        for surface in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES
        if surface
        not in {
            "reasoning_display",
            "vl_image",
            "video_where_supported",
        }
    )


def test_release_regression_manifest_real_ui_artifact_inventory_covers_every_row():
    artifacts = set(CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS.values())

    for row_id, row in CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS.items():
        assert row["proof"] in artifacts, row_id
        assert row["chat_screenshot"] in artifacts, row_id


def test_release_regression_manifest_current_sweep_rejects_fake_real_ui_live_model_proof(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["server"]["health"] = {"status": "ok", "model_loaded": False}
    proof["chat"]["rawParserTagLeak"] = True
    proof["chat"]["cjkLeakCount"] = 3
    proof["cache"]["cacheHitTokens"] = 0
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "server_health_not_real_loaded_model" in result["real_ui_live_model_proof"]["failures"]
    assert "raw_parser_or_reasoning_tag_leak" in result["real_ui_live_model_proof"]["failures"]
    assert "visible_language_leak" in result["real_ui_live_model_proof"]["failures"]
    assert "cache_hit_tokens_missing" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_real_ui_proof_rejects_zaya_point_tag_leak(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["chat"]["rawParserTagLeak"] = False
    proof["chat"]["turns"][1]["content"] = "<|point_start|>function leaked into visible UI"
    proof["chat"]["finalVisibleText"] = "<|point_start|>function leaked into visible UI"
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "raw_parser_or_reasoning_tag_leak" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_real_ui_proof_rejects_reasoning_numeric_garbage(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["chat"]["reasoningText"] = (
        "1 2 3 4 5 6 7 8 9 10 11 12 [Generation interrupted]"
    )
    proof["chat"].pop("reasoningNumericRunCount", None)
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "reasoning_language_or_numeric_leak" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_failed_real_ui_live_model_proof(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["status"] = "fail"
    proof["failureStage"] = "renderer_real_ui_chat_send_message"
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "proof_status_failed" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_real_ui_proof_without_request_contract(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.pop("requestContract", None)
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "request_contract_missing" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_accepts_structured_electron_dev_launch_when_log_tail_rotates(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["lfm25_moe_a1b_proof"]
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["appLogTail"] = [
        "[CHAT] Tool execution iteration 1 (1 tool calls)",
        "[CHAT] Builtin tool: run_command",
        "[CHAT] Tool loop completed after 1 iteration(s)",
    ]
    proof["uiLaunchMode"] = "electron-dev"
    proof["uiCommand"] = [
        "npm",
        "run",
        "dev",
        "--",
        "--",
        "--user-data-dir=/tmp/vmlx-real-ui-userdata",
        "--remote-debugging-port=55895",
    ]
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "pass"
    assert "electron_dev_renderer_url_missing" not in result[
        "real_ui_live_model_proof"
    ]["failures"]
    assert "electron_dev_launch_log_missing" not in result[
        "real_ui_live_model_proof"
    ]["failures"]


def test_release_regression_manifest_real_ui_matrix_tracks_mixed_minimax_identity(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)

    result = validate_current_proof_sweep_artifacts(tmp_path)
    matrix = result["real_ui_live_model_matrix"]

    assert matrix["status"] == "open"
    assert matrix["release_blocker"] == "Real Electron UI cross-family live model matrix is release-cleared"
    assert matrix["covered_families"]["zaya_text"]["status"] == "pass"
    assert "chat_completions" in matrix["covered_families"]["zaya_text"]["covered_surfaces"]
    assert "cache_hit_telemetry" in matrix["covered_families"]["zaya_text"]["covered_surfaces"]
    assert matrix["covered_families"]["ling_bailing"]["status"] == "pass"
    assert "responses_api" not in matrix["missing_surfaces"]
    assert "long_tool_loop" not in matrix["missing_surfaces"]
    assert "server_cache_controls" not in matrix["missing_surfaces"]
    assert "vl_image" not in matrix["missing_surfaces"]
    assert matrix["covered_families"]["zaya_vl"]["status"] == "pass"
    assert "vl_image" in matrix["covered_families"]["zaya_vl"]["covered_surfaces"]
    assert "gemma4" in matrix["covered_families"]
    assert matrix["covered_families"]["gemma4"]["status"] == "pass"
    assert "qwen36" not in matrix["missing_families"]
    assert matrix["covered_families"]["qwen36"]["status"] == "pass"
    assert "cache_hit_telemetry" in matrix["covered_families"]["qwen36"]["covered_surfaces"]
    assert matrix["covered_families"]["minimax"]["status"] == "pass"
    assert matrix["covered_families"]["minimax"]["modelNames"] == [
        "MiniMax-M2.7-Small-JANGTQ",
    ]
    assert matrix["covered_families"]["minimax"]["mixed_model_identity_union"] is False
    assert matrix["mixed_model_identity_families"] == []
    assert matrix["covered_families"]["nemotron_omni"]["status"] == "pass"
    assert matrix["covered_families"]["hy3"]["status"] == "pass"
    assert matrix["missing_families"] == ["dsv4"]
    assert matrix["unblocked_non_mimo_status"] == "pass"
    assert matrix["unblocked_non_mimo_missing_families"] == []
    assert matrix["unblocked_non_mimo_partial_families"] == []


def test_release_regression_manifest_current_sweep_requires_dsv4_memory_preflight_when_real_ui_missing(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_dsv4_memory_preflight"]["status"] == "missing"
    assert (
        result["real_ui_dsv4_memory_preflight"]["artifact"]
        == CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    )
    assert result["status"] == "fail"


def test_release_regression_manifest_real_ui_matrix_records_dsv4_memory_blocker(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_memory",
                "model_path": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "model_size_gb": 80.0,
                "free_plus_speculative_purgeable_gb": 39.15,
                "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                "required_available_gb": 120.0,
                "memory_gap_gb": 80.85,
                "psutil_available_gb": 104.0,
                "psutil_memory_gap_gb": 16.0,
                "inactive_file_cache_gb": 30.0,
                "launch_decision": "do_not_launch",
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
                ],
                "active_heavy_processes": [],
                "active_heavy_process_count": 0,
                "launch_blockers": ["insufficient_memory"],
                "did_not_launch": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_dsv4_memory_preflight"]["status"] == "pass"
    assert result["real_ui_live_model_matrix"]["resource_blockers"]["dsv4"] == {
        "artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
        "reason": "insufficient_memory",
        "model_path": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
        "required_available_gb": 120.0,
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
        "free_plus_speculative_purgeable_gb": 39.15,
        "memory_gap_gb": 80.85,
        "psutil_available_gb": 104.0,
        "psutil_memory_gap_gb": 16.0,
        "inactive_file_cache_gb": 30.0,
    }
    assert result["real_ui_live_model_matrix"]["unblocked_non_mimo_status"] == "pass"
    assert (
        result["real_ui_live_model_matrix"][
            "unblocked_non_mimo_missing_families"
        ]
        == []
    )
    assert (
        result["real_ui_live_model_matrix"][
            "unblocked_non_mimo_partial_families"
        ]
        == []
    )
    assert result["real_ui_live_model_matrix"][
        "unblocked_non_mimo_excluded_families"
    ] == ["dsv4", "mimo_v2"]


def test_release_regression_manifest_real_ui_unblocked_non_mimo_status_fails_on_missing_family(
):
    matrix = {
        "missing_families": ["mimo_v2", "dsv4", "qwen36"],
        "partial_families": [],
        "resource_blockers": {"dsv4": {"reason": "insufficient_memory"}},
    }

    _annotate_real_ui_unblocked_non_mimo_status(matrix)

    assert matrix["unblocked_non_mimo_status"] == "open"
    assert matrix["unblocked_non_mimo_missing_families"] == ["qwen36"]
    assert matrix["unblocked_non_mimo_partial_families"] == []


def test_release_regression_manifest_real_ui_unblocked_non_mimo_status_passes_when_only_mimo_and_resource_blocked_dsv4_are_open():
    matrix = {
        "missing_families": ["mimo_v2", "dsv4"],
        "partial_families": [],
        "mixed_model_identity_families": [],
        "resource_blockers": {"dsv4": {"reason": "insufficient_memory"}},
    }

    _annotate_real_ui_unblocked_non_mimo_status(matrix)

    assert matrix["unblocked_non_mimo_status"] == "pass"
    assert matrix["unblocked_non_mimo_missing_families"] == []
    assert matrix["unblocked_non_mimo_partial_families"] == []
    assert matrix["unblocked_non_mimo_excluded_families"] == ["dsv4", "mimo_v2"]


def test_release_regression_manifest_real_ui_unblocked_non_mimo_status_excludes_runtime_blocked_step37():
    matrix = {
        "missing_families": ["mimo_v2", "dsv4", "step37"],
        "partial_families": ["step37"],
        "mixed_model_identity_families": [],
        "resource_blockers": {"dsv4": {"reason": "insufficient_memory"}},
        "runtime_blockers": {
            "step37": {"reason": "missing_mlx_vlm_step3p7_vlm_runtime"}
        },
    }

    _annotate_real_ui_unblocked_non_mimo_status(matrix)

    assert matrix["unblocked_non_mimo_status"] == "pass"
    assert matrix["unblocked_non_mimo_missing_families"] == []
    assert matrix["unblocked_non_mimo_partial_families"] == []
    assert matrix["unblocked_non_mimo_excluded_families"] == [
        "dsv4",
        "mimo_v2",
        "step37",
    ]


def test_release_regression_manifest_real_ui_unblocked_non_mimo_status_rejects_mixed_identity():
    matrix = {
        "missing_families": ["mimo_v2", "dsv4"],
        "partial_families": [],
        "mixed_model_identity_families": ["minimax"],
        "resource_blockers": {"dsv4": {"reason": "insufficient_memory"}},
    }

    _annotate_real_ui_unblocked_non_mimo_status(matrix)

    assert matrix["unblocked_non_mimo_status"] == "open"
    assert matrix["unblocked_non_mimo_missing_families"] == []
    assert matrix["unblocked_non_mimo_partial_families"] == ["minimax"]


def test_release_regression_manifest_rejects_dsv4_preflight_under_model_margin(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_memory",
                "model_path": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "model_size_gb": 100.0,
                "free_plus_speculative_purgeable_gb": 39.15,
                "required_available_gb": 120.0,
                "memory_gap_gb": 80.85,
                "launch_decision": "do_not_launch",
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
                ],
                "active_heavy_processes": [],
                "active_heavy_process_count": 0,
                "launch_blockers": ["insufficient_memory"],
                "did_not_launch": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "real_ui_dsv4_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert "required_available_gb_missing_model_margin" in result["failures"]


def test_release_regression_manifest_rejects_dsv4_preflight_with_active_heavy_process(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_memory",
                "model_path": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "model_size_gb": 80.0,
                "free_plus_speculative_purgeable_gb": 39.15,
                "required_available_gb": 120.0,
                "memory_gap_gb": 80.85,
                "launch_decision": "do_not_launch",
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
                ],
                "active_heavy_processes": [
                    {
                        "pid": 2002,
                        "command": "python tests/cross_matrix/run_runtime_memory_stress_probe.py",
                    }
                ],
                "active_heavy_process_count": 1,
                "launch_blockers": ["insufficient_memory", "active_heavy_process"],
                "did_not_launch": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "real_ui_dsv4_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert "active_heavy_processes_not_clear" in result["failures"]


def test_release_regression_manifest_real_ui_matrix_requires_every_family_surface():
    native_cache = {
        "family": "plain_kv",
        "schema": "plain_kv_v1",
        "cache_type": "paged_kv",
        "components": ["attention_kv"],
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
    }
    cache_stats = {
        "before": {"scheduler_cache": {"hits": 0}},
        "after": {
            "scheduler_cache": {"hits": 1},
            "block_disk_cache": {"disk_hits": 1},
            "cache_totals": {"l2_tokens_on_disk": 12},
        },
        "cacheHitTokens": 12,
    }
    proofs = {
        family: {
            "modelName": f"{family}-model",
            "appLogTail": ["start electron app"],
            "server": {
                "health": {
                    "status": "healthy",
                    "model_loaded": True,
                    "native_cache": native_cache,
                }
            },
            "chat": {
                "turns": [{"role": "assistant", "content": "ok"}],
                "rawParserTagLeak": False,
                "cjkLeakCount": 0,
                "koreanLeakCount": 0,
            },
            "cache": cache_stats,
            "rendererWireApi": "responses",
            "eventCounts": {"complete": 2, "tool": 3, "reasoningDone": 1},
            "persistedToolCount": 1,
            "persistedToolsByMessage": [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ],
            "persistedReasoningCount": 1,
            "requestedBuiltinTools": True,
            "chatOverrides": {"builtinToolsEnabled": True},
            "serverCacheControls": {"verified": True},
            "media": {"imageVerified": True, "videoVerified": True},
        }
        for family in REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES
    }
    proofs["gemma4"]["persistedToolCount"] = 0
    proofs["gemma4"]["persistedToolsByMessage"] = []

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": proofs}
    )

    assert matrix["missing_families"] == []
    assert matrix["missing_surfaces"] == []
    assert matrix["status"] == "open"
    assert matrix["partial_families"] == ["gemma4"]
    assert "long_tool_loop" in matrix["covered_families"]["gemma4"]["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_uses_family_specific_media_requirements():
    def complete_proof(*, image=False, video=False):
        return {
            "modelName": "architecture-aware-model",
            "appLogTail": ["start electron app"],
            "server": {
                "health": {
                    "status": "healthy",
                    "model_loaded": True,
                    "native_cache": {
                        "family": "plain_kv",
                        "schema": "plain_kv_v1",
                        "cache_type": "paged_kv",
                        "components": ["attention_kv"],
                        "prefix": True,
                        "paged": True,
                        "block_disk_l2": True,
                    },
                }
            },
            "chat": {
                "turns": [{"role": "assistant", "content": "ok"}],
                "rawParserTagLeak": False,
                "cjkLeakCount": 0,
                "koreanLeakCount": 0,
            },
            "cache": {
                "before": {"scheduler_cache": {"hits": 0}},
                "after": {
                    "scheduler_cache": {"hits": 1},
                    "block_disk_cache": {"disk_hits": 1},
                    "cache_totals": {"l2_tokens_on_disk": 12},
                },
                "cacheHitTokens": 12,
            },
            "rendererWireApi": "responses",
            "eventCounts": {"complete": 2, "tool": 3, "reasoningDone": 1},
            "persistedToolCount": 1,
            "persistedToolsByMessage": [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ],
            "persistedReasoningCount": 1,
            "requestedBuiltinTools": True,
            "chatOverrides": {"builtinToolsEnabled": True},
            "serverCacheControls": {"verified": True},
            "media": {"imageVerified": image, "videoVerified": video},
        }

    matrix = _validate_current_real_ui_live_model_matrix(
        {
            "status": "pass",
            "proofs": {
                "zaya_text": complete_proof(),
                "zaya_vl_image": complete_proof(image=True),
                "qwen36_mxfp4_crack": complete_proof(image=True),
            },
        }
    )

    assert "vl_image" not in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["zaya_text"]
    assert "video_where_supported" not in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["zaya_text"]
    assert "reasoning_display" not in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["zaya_text"]
    assert "reasoning_display" not in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["ling_bailing"]
    assert matrix["covered_families"]["zaya_text"]["status"] == "pass"
    assert "vl_image" not in matrix["covered_families"]["zaya_text"]["missing_surfaces"]
    assert matrix["covered_families"]["zaya_vl"]["status"] == "pass"
    assert "video_where_supported" not in matrix["covered_families"]["zaya_vl"]["missing_surfaces"]
    assert matrix["covered_families"]["qwen36"]["status"] == "partial"
    assert "video_where_supported" in matrix["covered_families"]["qwen36"]["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_cache_hit_telemetry_when_reconstruction_falls_back_to_miss():
    proof = {
        "modelName": "Nemotron-Omni-Nano-JANGTQ-CRACK",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "nemotron_h",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3, "reasoningDone": 1},
        "persistedToolCount": 1,
        "persistedToolsByMessage": [
            [
                {"phase": "result", "toolName": "run_command"},
                {"phase": "result", "toolName": "write_file"},
            ],
        ],
        "persistedReasoningCount": 1,
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.scheduler:Request resp_a: worker-side paged cache reconstruction failed, treating as cache miss"
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"nemotron_omni_nano_responses_tools_reasoning_cachecontrols": proof}}
    )

    family = matrix["covered_families"]["nemotron_omni"]
    assert "cache_hit_telemetry" not in family["covered_surfaces"]
    assert "cache_hit_telemetry" in family["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_hybrid_ssm_full_prefill_fallback():
    proof = {
        "modelName": "LFM2.5-8B-A1B-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3},
        "persistedToolCount": 1,
        "persistedToolsByMessage": [
            [{"phase": "result", "toolName": "run_command"}],
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.scheduler:Request resp_a: hybrid paged MISS — 448 KV tokens cached but no usable SSM companion, full prefill"
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b": proof}}
    )

    family = matrix["covered_families"]["lfm25"]
    assert "cache_hit_telemetry" not in family["covered_surfaces"]
    assert "cache_hit_telemetry" in family["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_accepts_hybrid_ssm_hit_after_warmup_miss():
    proof = {
        "modelName": "LFM2.5-8B-A1B-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "scheduler": {
                    "last_cache_execution": {
                        "cache_detail": "paged+ssm",
                        "reconstruction_ok": True,
                    }
                },
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 2},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3},
        "persistedToolCount": 1,
        "persistedToolsByMessage": [
            [{"phase": "result", "toolName": "run_command"}],
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.scheduler:Request resp_a: hybrid paged MISS — 448 KV tokens cached but no usable SSM companion, full prefill",
            "INFO:vmlx_engine.scheduler:Request resp_b: hybrid paged HIT — 320 tokens (KV + 18 SSM layers)",
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b": proof}}
    )

    family = matrix["covered_families"]["lfm25"]
    assert "cache_hit_telemetry" in family["covered_surfaces"]


def test_release_regression_manifest_real_ui_named_tool_probe_accepts_split_file_writes():
    proof = {
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_TWO"},
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt",
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_named_tool_probe_rejects_visible_file_semantic_contradiction():
    proof = {
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {
                    "role": "assistant",
                    "content": (
                        "REAL_UI_LIVE_TOOL_TWO\n\n"
                        "The file real_ui_tool_probe_2.txt contains the string "
                        '"Real_UI_Live_Tool_One".'
                    ),
                },
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt",
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert not _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_named_tool_probe_rejects_malformed_visible_tool_sentinel():
    proof = {
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {
                    "role": "assistant",
                    "content": (
                        "The content of real_ui_tool_probe_2.txt is "
                        "REAL_UI_LIVE_TOOL_TWO.\n\nRE:AL_UI_LIVE_TOOL_TWO"
                    ),
                },
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_ONE > "
                        "real_ui_tool_probe_1.txt && cat real_ui_tool_probe_1.txt\n\n"
                        "REAL_UI_LIVE_TOOL_ONE"
                    ),
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_TWO > "
                        "real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert not _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_named_tool_probe_rejects_explanatory_exact_reply():
    proof = {
        "chat": {
            "turns": [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once. "
                        "After the tool result, reply exactly: REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The command has been executed as requested.\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_TWO > "
                        "real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert not _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_named_tool_probe_accepts_exact_reply_contract():
    proof = {
        "chat": {
            "turns": [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once. "
                        "After the tool result, reply exactly: REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_TWO"},
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_TWO > "
                        "real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_matrix_rejects_forged_tool_reasoning_surfaces():
    proof = {
        "modelName": "ZAYA1-8B-MXFP4",
        "provenSurfaces": [
            "responses_api",
            "long_tool_loop",
            "reasoning_display",
        ],
        "rendererWireApi": "chat",
        "eventCounts": {"complete": 2, "tool": 0, "reasoningDone": 0},
        "persistedToolCount": 0,
        "persistedReasoningCount": 0,
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"gemma4": proof}}
    )

    gemma4 = matrix["covered_families"]["gemma4"]
    assert "responses_api" not in gemma4["covered_surfaces"]
    assert "long_tool_loop" not in gemma4["covered_surfaces"]
    assert "reasoning_display" not in gemma4["covered_surfaces"]
    assert "responses_api" in gemma4["missing_surfaces"]
    assert "long_tool_loop" in gemma4["missing_surfaces"]
    assert "reasoning_display" in gemma4["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_reasoning_only_visible_turns():
    proof = {
        "modelName": "Step-3.7-Flash-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "step_vl",
                    "schema": "mllm_rotating_kv_v1",
                    "cache_type": "mllm_rotating_paged_kv",
                    "components": ["attention_kv"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "rawParserTagLeak": False,
            "reasoningRawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
            "reasoningCjkLeakCount": 0,
            "reasoningKoreanLeakCount": 0,
            "reasoningNumericRunCount": 0,
            "turns": [
                {"role": "user", "content": "Say FINAL=STEP37_REASONING_OK."},
                {"role": "assistant", "content": "FINAL=STEP37_REASONING_OK"},
                {"role": "user", "content": "Confirm it again."},
                {"role": "assistant", "content": ""},
            ],
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "requestedEnableThinking": True,
        "eventCounts": {"complete": 2, "reasoningDone": 2},
        "persistedReasoningCount": 2,
        "requestedBuiltinTools": False,
        "chatOverrides": {"builtinToolsEnabled": False},
        "serverCacheControls": {"verified": True},
        "media": {"imageVerified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"step37_flash_jang2l_reasoning": proof}}
    )

    step37 = matrix["covered_families"]["step37"]
    assert "reasoning_display" not in step37["covered_surfaces"]
    assert "reasoning_display" in step37["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_lfm2_raw_tool_markers():
    proof = {
        "modelName": "LFM2.5-8B-A1B-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "model_info": {"model_name": "LFM2.5-8B-A1B-JANG_2L"},
            }
        },
        "chat": {
            "rawParserTagLeak": False,
            "reasoningRawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
            "reasoningCjkLeakCount": 0,
            "reasoningKoreanLeakCount": 0,
            "reasoningNumericRunCount": 0,
            "turns": [
                {
                    "role": "assistant",
                    "content": "<|tool_call_start|>\nvisible leaked tool marker",
                }
            ],
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 2},
        "persistedToolCount": 2,
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "parser_leak_check" not in lfm25["covered_surfaces"]
    assert "parser_leak_check" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_accepts_top_level_parser_leak_flags():
    proof = {
        "modelName": "LFM2.5-8B-A1B-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "model_info": {"model_name": "LFM2.5-8B-A1B-JANG_2L"},
            }
        },
        "chat": {
            "turns": [
                {
                    "role": "assistant",
                    "content": "tool result handled without visible parser tags",
                }
            ],
        },
        "rawParserLeak": False,
        "reasoningRawParserLeak": False,
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3},
        "persistedToolsByMessage": [
            [
                {"phase": "result", "toolName": "write_file"},
                {"phase": "result", "toolName": "read_file"},
            ]
        ],
        "persistedToolCount": 2,
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "parser_leak_check" in lfm25["covered_surfaces"]
    assert "parser_leak_check" not in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_empty_tool_status_spam():
    proof = {
        "modelName": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "plain_kv",
                    "schema": "plain_kv_v1",
                    "cache_type": "paged_kv",
                    "components": ["attention_kv"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 49, "reasoningDone": 1},
        "persistedToolCount": 49,
        "persistedToolsByMessage": [
            [{"phase": "generating", "toolName": "", "iteration": 0} for _ in range(49)]
        ],
        "persistedReasoningCount": 1,
        "chatOverrides": {"builtinToolsEnabled": True},
        "requestedBuiltinTools": True,
        "serverCacheControls": {"verified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"gemma4": proof}}
    )

    gemma4 = matrix["covered_families"]["gemma4"]
    assert "long_tool_loop" not in gemma4["covered_surfaces"]
    assert "long_tool_loop" in gemma4["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_failed_tool_loop():
    proof = {
        "modelName": "LFM2.5-8B-A1B-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "tool failed"}],
            "rawParserTagLeak": False,
            "reasoningRawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
            "reasoningCjkLeakCount": 0,
            "reasoningKoreanLeakCount": 0,
            "reasoningNumericRunCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 8},
        "persistedToolsByMessage": [
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "error",
                    "toolName": "run_command",
                    "detail": "Exit code: 127\n/bin/sh: bad-command: command not found",
                },
                {"phase": "result", "toolName": "run_command"},
            ],
            [
                {"phase": "result", "toolName": "run_command"},
            ],
        ],
        "persistedToolCount": 5,
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "long_tool_loop" not in lfm25["covered_surfaces"]
    assert "long_tool_loop" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_wrong_tool_probe_payload():
    proof = {
        "modelName": "ZAYA1-8B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "zaya",
                    "schema": "zaya_cca_v1",
                    "cache_type": "hybrid_zaya_cca",
                    "components": ["attention_kv", "cca_conv_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [
                {
                    "role": "user",
                    "content": (
                        "Use run_command to create real_ui_tool_probe_1.txt "
                        "with REAL_UI_LIVE_TOOL_ONE"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Use run_command to read real_ui_tool_probe_1.txt, create "
                        "real_ui_tool_probe_2.txt, and write REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
            ],
            "rawParserTagLeak": False,
            "reasoningRawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
            "reasoningCjkLeakCount": 0,
            "reasoningKoreanLeakCount": 0,
            "reasoningNumericRunCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 8},
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_ONE > "
                        "real_ui_tool_probe_1.txt\n\n"
                    ),
                },
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ cat real_ui_tool_probe_1.txt > "
                        "real_ui_tool_probe_2.txt\n\n"
                    ),
                },
            ],
        ],
        "persistedToolCount": 2,
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE\n",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_ONE\n",
        },
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"zaya_text": proof}}
    )

    zaya = matrix["covered_families"]["zaya_text"]
    assert "long_tool_loop" not in zaya["covered_surfaces"]
    assert "long_tool_loop" in zaya["missing_surfaces"]


def test_release_regression_manifest_validates_current_proof_sweep_artifacts(tmp_path):
    model_family_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    model_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    api_cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "noheavy-api-cache-endpoint-runtime"
    ]
    parser_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]
    generation_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]
    api_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    reasoning_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    tool_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    panel_tool_security_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "panel-tool-security-loop-boundary"
    ]
    mtp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    vl_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    mcp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    max_output_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    jang_model_compat_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "jang-model-compat-runtime-boundary"
    ]
    packaged_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
    release_surface_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
    open_requirement_details = _passing_open_requirement_details()
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == model_family_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == max_output_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS
                        },
                        "missing_markers": [],
                        "failed": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == packaged_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS
                        },
                        "failed": [],
                        "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == release_surface_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == model_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == cache_artifact:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_cache_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True
                            for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == parser_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == generation_artifact:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_cache_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True
                            for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == reasoning_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == tool_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == panel_tool_security_artifact:
            path.write_text(
                json.dumps({"status": "pass", "returncode": 0}) + "\n",
                encoding="utf-8",
            )
        elif artifact == mtp_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == vl_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS
                        },
                        "missing_engine_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mcp_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_requirement_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    mimo_live_artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["mimo_v2_jang2l"]
    mimo_live_path = tmp_path / mimo_live_artifact
    mimo_live_payload = json.loads(mimo_live_path.read_text(encoding="utf-8"))
    mimo_live_payload["status"] = "fail"
    mimo_live_payload["failed"] = 1
    mimo_live_payload["results"][0]["status"] = "fail"
    mimo_live_payload["results"][0]["failures"] = ["expected_exact_ack_missing"]
    mimo_live_path.write_text(json.dumps(mimo_live_payload), encoding="utf-8")
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    mimo_tool_artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["mimo_v2_jang2l"]
    mimo_tool_path = tmp_path / mimo_tool_artifact
    mimo_tool_payload = json.loads(mimo_tool_path.read_text(encoding="utf-8"))
    mimo_tool_payload["status"] = "fail"
    mimo_tool_payload["failed"] = 1
    mimo_tool_payload["results"][0]["status"] = "fail"
    mimo_tool_payload["results"][0]["failures"] = ["expected_tool_call_missing"]
    mimo_tool_path.write_text(json.dumps(mimo_tool_payload), encoding="utf-8")
    _write_expected_diagnostic_live_smoke_artifacts(tmp_path)
    _write_current_objective_digest(tmp_path)
    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)
    sink_path = tmp_path / CURRENT_MIMO_V2_JANG2L_SINK_AB_ARTIFACT
    sink_path.parent.mkdir(parents=True, exist_ok=True)
    sink_path.write_text(
        json.dumps(
            {
                "variants": [
                    {"mode": "normal", "output": "- - - 0101"},
                    {"mode": "sink_disabled", "output": "- - - 1010"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_passing_dev_ui_proof_artifacts(tmp_path)
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)

    result = validate_current_proof_sweep_artifacts(tmp_path)

    bad_components = {
        key: value
        for key, value in result.items()
        if isinstance(value, dict)
        and (
            value.get("failures")
            or value.get("missing")
            or value.get("not_pass")
            or value.get("status") not in {"pass", "open"}
        )
    }
    assert result["status"] == "fail", json.dumps(bad_components, indent=2, sort_keys=True)
    assert result["failed_components"] == [
        "no_release_blockers",
        "dsv4_long_output_code_exactness",
        "no_open_objective_requirements",
        "real_ui_dsv4",
    ]
    assert result["non_mimo_component_status"] == "fail"
    assert result["component_ok"]["no_release_blockers"] is False
    assert result["component_ok"]["packaged_app_developer_id_signing"] is True
    assert result["component_ok"]["dsv4_long_output_code_exactness"] is False
    assert result["component_ok"]["no_open_objective_requirements"] is False
    assert result["component_ok"]["real_ui_full_model_matrix"] is True
    assert result["component_ok"]["real_ui_mimo_v2"] is True
    assert result["component_ok"]["real_ui_dsv4"] is False
    assert result["component_ok"]["mimo_bundled_live_smoke"] is True
    assert result["component_ok"]["mimo_bundled_tool_smoke"] is True
    assert result["component_ok"]["mimo_v2_jang2l_sink_ab"] is True
    assert result["component_ok"]["mimo_v2_jang2l_root_cause"] is True
    assert result["component_ok"]["objective_digest"] is True
    assert result["objective_digest"]["status"] == "pass"
    assert result["missing"] == []
    assert result["not_pass"] == []

    assert result["regression_suite"] == {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "pass",
        "failed_steps": [],
        "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "open_requirement_details": open_requirement_details,
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
        "open_requirement_detail_failures": [],
    }
    assert result["model_family_matrix"] == {
        "artifact": model_family_artifact,
        "status": "pass",
        "matched_rows": list(EXPECTED_CURRENT_MODEL_FAMILY_ROWS),
        "missing_rows": [],
        "missing_expected_rows": [],
    }
    assert result["model_artifact_matrix"] == {
        "artifact": model_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["cache_architecture_matrix"] == {
        "artifact": cache_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS},
        "missing_markers": [],
        "missing_api_checks": [],
        "missing_api_command_markers": [],
        "missing_panel_markers": [],
        "missing_family_rows": [],
        "failed_family_rows": [],
        "cache_family_matrix": _passing_cache_family_matrix(),
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["parser_registry_matrix"] == {
        "artifact": parser_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["generation_defaults_matrix"] == {
        "artifact": generation_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS},
        "missing_markers": [],
        "missing_family_rows": [],
        "failed_family_rows": [],
        "generation_defaults_family_matrix": _passing_generation_defaults_family_matrix(),
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["api_surface_matrix"] == {
        "artifact": api_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS},
        "missing_nested_checks": [],
        "missing_nested_markers": [],
        "missing_panel_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["reasoning_template_matrix"] == {
        "artifact": reasoning_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["tool_call_matrix"] == {
        "artifact": tool_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["native_mtp_matrix"] == {
        "artifact": mtp_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["vl_media_matrix"] == {
        "artifact": vl_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS},
        "missing_engine_markers": [],
        "missing_panel_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["mcp_policy_matrix"] == {
        "artifact": mcp_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["max_output_context_matrix"] == {
        "artifact": max_output_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS},
        "missing_markers": [],
        "failed": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["packaged_integrity_matrix"] == {
        "artifact": packaged_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS},
        "failed": [],
        "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "package_signing_preflight": {},
        "release_blockers": [],
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["release_surface_matrix"] == {
        "artifact": release_surface_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS},
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["live_smoke_summaries"]["status"] == "fail"
    assert result["live_smoke_summaries"]["non_mimo_status"] == "pass"
    assert (
        "zaya_text_mxfp4"
        in result["live_smoke_summaries"]["artifacts"]
    )
    assert (
        result["live_smoke_summaries"]["artifacts"]["zaya_text_mxfp4"]["artifact"]
        == "build/current-all-local-model-smoke-zaya-text-bundled-20260524/summary.json"
    )
    assert result["live_smoke_summaries"]["missing"] == []
    assert [
        item["artifact"] for item in result["live_smoke_summaries"]["not_pass"]
    ] == [CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["mimo_v2_jang2l"]]
    assert result["live_tool_smoke_summaries"]["status"] == "fail"
    assert result["live_tool_smoke_summaries"]["non_mimo_status"] == "pass"
    assert result["live_tool_smoke_summaries"]["missing"] == []
    assert [
        item["artifact"]
        for item in result["live_tool_smoke_summaries"]["not_pass"]
    ] == [CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["mimo_v2_jang2l"]]
    assert (
        result["diagnostic_live_smoke_summaries"]
        == _expected_passing_diagnostic_live_smoke_summaries()
    )
    assert result["dev_ui_proof"] == {
        "status": "pass",
        "artifacts": dict(CURRENT_DEV_UI_PROOF_ARTIFACTS),
        "missing": [],
        "failures": [],
    }
    assert result["real_ui_live_model_proof"]["status"] == "pass"
    assert result["real_ui_live_model_proof"]["artifacts"] == dict(
        CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS
    )
    assert result["real_ui_live_model_proof"]["missing"] == []
    assert result["real_ui_live_model_proof"]["failures"] == []
    assert set(result["real_ui_live_model_proof"]["proofs"]) == set(
        CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS
    )
    assert result["real_ui_live_model_matrix"]["unblocked_non_mimo_status"] == "pass"
    assert result["real_ui_live_model_matrix"][
        "unblocked_non_mimo_missing_families"
    ] == []
    assert result["real_ui_live_model_matrix"][
        "unblocked_non_mimo_partial_families"
    ] == []
    assert result["component_ok"]["live_smoke_non_mimo"] is True
    assert result["component_ok"]["live_tool_smoke_non_mimo"] is True
    assert result["component_ok"]["real_ui_unblocked_non_mimo"] is True
    assert result["component_ok"]["jang_model_compat_matrix"] is True
    assert result["component_ok"]["panel_tool_security_matrix"] is True
    assert result["component_ok"]["noheavy_api_cache_matrix"] is True
    assert result["noheavy_api_cache_matrix"] == {
        "artifact": CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
            "noheavy-api-cache-endpoint-runtime"
        ],
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["jang_model_compat_matrix"] == {
        "artifact": CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
            "jang-model-compat-runtime-boundary"
        ],
        "status": "pass",
        "failed": [],
        "missing": [],
    }
    assert result["panel_tool_security_matrix"] == {
        "artifact": CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
            "panel-tool-security-loop-boundary"
        ],
        "status": "pass",
        "returncode": 0,
        "missing": [],
    }
    assert result["installed_app_runtime_parity_audit"] == {
        "artifact": CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        "status": "pass",
        "missing": [],
        "failures": [],
    }
    assert result["issue175_177_installed_runtime_audit"] == {
        "artifact": CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT,
        "status": "pass",
        "missing": [],
        "failures": [],
    }
    assert result["issue175_177_live_runtime_audit"] == {
        "artifact": CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT,
        "status": "pass",
        "missing": [],
        "failures": [],
    }
    assert result["mimo_v2_jang2l_root_cause"] == {
        "status": "pass",
        "artifacts": {
            "router_topk_parity": CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
            "moe_output_parity": CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
            "profile_diagnostic": CURRENT_MIMO_V2_JANG2L_PROFILE_DIAGNOSTIC_ARTIFACT,
            "nocache_no_kvq_probe": CURRENT_MIMO_V2_JANG2L_NOCACHE_NO_KVQ_ARTIFACT,
        },
        "missing": [],
        "failures": [],
        "router_topk_exact": True,
        "moe_expert_distortion_seen": True,
        "profile_diagnostic_seen": True,
        "nocache_no_kvq_incoherent": True,
        "max_moe_rel_l2": 0.603,
        "min_moe_cosine": 0.813,
        "profile_rel_l2": {
            "source": 0.0,
            "2L": 0.178,
        },
        "remote_artifacts": [],
        "remote_evidence_only": False,
        "local_release_clearance": True,
        "root_cause_candidate": "mimo_v2_jang2l_2bit_routed_expert_distortion",
    }
    assert result["release_blocker_ledger"] == {
        "status": "open",
        "blockers": [
            {
                "id": "dsv4_long_output_code_exactness_open",
                "status": "open",
                "evidence": CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
                "next_proof": "Run and pass DSV4 long-output/code exactness with current source/app in a memory-safe local session.",
                "details": {
                    "available_gb": 103.87,
                    "required_available_gb": 120.0,
                    "memory_gap_gb": 16.13,
                    "strict_vm_stat_memory_gap_gb": 16.13,
                    "psutil_available_gap_gb": 16.13,
                    "preflight_memory_source": "psutil_available",
                    "did_not_launch": True,
                    "launch_decision": "do_not_launch",
                    "case_count": 14,
                },
            },
            {
                "id": "real_ui_dsv4_memory_blocked",
                "status": "open",
                "evidence": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
                "next_proof": "Run DSV4 real Electron UI proof on a local machine/session with enough free memory.",
                "details": {
                    "artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
                    "reason": "insufficient_memory",
                    "model_path": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                    "required_available_gb": 120.0,
                    "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                    "free_plus_speculative_purgeable_gb": 39.15,
                    "memory_gap_gb": 80.85,
                    "psutil_available_gb": 104.0,
                    "psutil_memory_gap_gb": 16.0,
                    "inactive_file_cache_gb": 30.0,
                },
            },
        ],
        "deferred_release_families": [
            {"family": "mimo_v2", "reason": "deferred_out_of_release_scope"}
        ],
    }


def test_release_blocker_ledger_defers_mimo_failed_live_smokes_from_release_blockers():
    live_artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["mimo_v2_jang2l"]
    tool_artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["mimo_v2_jang2l"]

    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={
            "status": "fail",
            "missing": [],
            "not_pass": [{"artifact": live_artifact, "status": "fail"}],
        },
        live_tool_smoke_summaries={
            "status": "fail",
            "missing": [],
            "not_pass": [{"artifact": tool_artifact, "status": "fail"}],
        },
        mimo_v2_jang2l_sink_ab={"status": "open"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "open", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "open"},
        real_ui_live_model_matrix={"status": "open", "missing_families": []},
    )

    blocker_ids = [blocker["id"] for blocker in ledger["blockers"]]
    assert "mimo_bundled_live_smoke_failed" not in blocker_ids
    assert "mimo_bundled_tool_smoke_failed" not in blocker_ids
    assert "mimo_bundled_live_smoke_missing" not in blocker_ids
    assert "mimo_bundled_tool_smoke_missing" not in blocker_ids
    assert ledger["deferred_release_families"] == [
        {
            "family": "mimo_v2",
            "reason": "deferred_out_of_release_scope",
        }
    ]


def test_release_blocker_ledger_tracks_public_version_collision():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        release_surface_matrix={
            "status": "pass",
            "failed_checks": ["staged_source_version_not_public"],
            "artifact": "build/current-release-surface-contract.json",
        },
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"status": "pass"},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    blockers = {blocker["id"]: blocker for blocker in ledger["blockers"]}
    assert blockers["source_version_already_public"]["evidence"] == (
        "build/current-release-surface-contract.json"
    )


def test_release_blocker_ledger_does_not_use_remote_max2_artifacts_as_release_evidence():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "open"},
        mimo_v2_jang2l_root_cause={
            "remote_evidence_only": True,
            "remote_artifacts": [
                CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
                CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
            ],
        },
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    assert ledger["status"] == "pass"
    for blocker in ledger["blockers"]:
        assert "remote-max2" not in blocker["evidence"]
    blockers = {blocker["id"]: blocker for blocker in ledger["blockers"]}
    assert blockers == {}
    assert ledger["deferred_release_families"] == [
        {"family": "mimo_v2", "reason": "deferred_out_of_release_scope"}
    ]


def test_release_blocker_ledger_tracks_missing_local_mimo_root_cause_artifacts():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={
            "status": "missing",
            "remote_evidence_only": False,
            "missing": [
                CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
                CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
            ],
        },
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    blockers = {blocker["id"]: blocker for blocker in ledger["blockers"]}
    assert "mimo_root_cause_local_proof_missing" not in blockers
    assert ledger["deferred_release_families"] == [
        {"family": "mimo_v2", "reason": "deferred_out_of_release_scope"}
    ]


def test_release_blocker_ledger_tracks_packaged_signing_blocker():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
        packaged_integrity_matrix={
            "package_signing_preflight": {
                "signing_blocker_reason": "developer_id_keychain_user_interaction_not_allowed",
                "developer_id_signed": False,
                "developer_id_identity_count": 9,
                "signature_is_adhoc": True,
                "team_identifier": "not set",
                "simple_developer_id_sign_rc": 1,
                "codesign_verify_rc": 1,
                "manual_remediation_required": True,
                "remediation_summary": "Developer ID identities are visible, but codesign cannot use the private key from this non-interactive process.",
                "remediation_steps": [
                    "Unlock the signing keychain in an interactive macOS session.",
                    "Grant codesign access to the Developer ID private key, for example with security set-key-partition-list using the keychain password outside Codex logs.",
                    "Rerun the packaged integrity contract and require package_signing_preflight.status=pass before notarization.",
                ],
            },
            "release_blockers": [
                {
                    "id": "packaged_app_developer_id_signing_blocked",
                    "status": "open",
                    "evidence": "package_signing_preflight",
                    "next_proof": "Build and verify a Developer ID signed vMLX.app before notarization.",
                }
            ]
        },
    )

    assert ledger == {
        "status": "open",
        "blockers": [
            {
                "id": "packaged_app_developer_id_signing_blocked",
                "status": "open",
                "evidence": "package_signing_preflight",
                "details": {
                    "signing_blocker_reason": "developer_id_keychain_user_interaction_not_allowed",
                    "developer_id_signed": False,
                    "developer_id_identity_count": 9,
                    "signature_is_adhoc": True,
                    "team_identifier": "not set",
                    "simple_developer_id_sign_rc": 1,
                    "codesign_verify_rc": 1,
                    "manual_remediation_required": True,
                    "remediation_summary": "Developer ID identities are visible, but codesign cannot use the private key from this non-interactive process.",
                    "remediation_steps": [
                        "Unlock the signing keychain in an interactive macOS session.",
                        "Grant codesign access to the Developer ID private key, for example with security set-key-partition-list using the keychain password outside Codex logs.",
                        "Rerun the packaged integrity contract and require package_signing_preflight.status=pass before notarization.",
                    ],
                },
                "next_proof": "Build and verify a Developer ID signed vMLX.app before notarization.",
            }
        ],
        "deferred_release_families": [
            {"family": "mimo_v2", "reason": "deferred_out_of_release_scope"}
        ],
    }


def test_release_blocker_ledger_tracks_dsv4_exactness_open_requirement():
    ledger = _current_release_blocker_ledger(
        regression_suite={
            "open_requirements": [
                "DSV4 long-output/code/file-generation quality is release-cleared"
            ]
        },
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    assert ledger == {
        "status": "open",
        "blockers": [
            {
                "id": "dsv4_long_output_code_exactness_open",
                "status": "open",
                "evidence": CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
                "next_proof": "Run and pass DSV4 long-output/code exactness with current source/app in a memory-safe local session.",
            }
        ],
        "deferred_release_families": [
            {"family": "mimo_v2", "reason": "deferred_out_of_release_scope"}
        ],
    }


def test_release_blocker_ledger_tracks_stale_real_ui_request_contract_proofs():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_proof={
            "status": "fail",
            "failures": [
                "request_contract_missing",
                "request_contract_incomplete:enableThinking",
            ],
        },
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    assert ledger["blockers"] == [
        {
            "id": "real_ui_request_contract_proofs_stale",
            "status": "open",
            "evidence": "current_proof_sweep.real_ui_live_model_proof.failures:request_contract_incomplete:enableThinking,request_contract_missing",
            "next_proof": "Rerun real Electron UI model proofs with current live-real-ui-model-proof.mjs so every artifact records its prompts and request knobs.",
        }
    ]


def test_release_blocker_ledger_splits_real_ui_missing_family_blockers():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["mimo_v2", "dsv4"],
            "resource_blockers": {
                "dsv4": {"artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT}
            },
        },
    )

    blocker_ids = [blocker["id"] for blocker in ledger["blockers"]]
    assert blocker_ids == [
        "real_ui_dsv4_memory_blocked",
    ]

    unblocked_missing_ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["mimo_v2", "dsv4", "step37", "lfm25"],
            "resource_blockers": {
                "dsv4": {"artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT}
            },
        },
    )
    assert [blocker["id"] for blocker in unblocked_missing_ledger["blockers"]] == [
        "real_ui_dsv4_memory_blocked",
        "real_ui_unblocked_non_mimo_missing",
    ]
    assert (
        "missing_families:lfm25,step37"
        in unblocked_missing_ledger["blockers"][-1]["evidence"]
    )

    unblocked_partial_ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["mimo_v2", "dsv4"],
            "partial_families": ["lfm25"],
            "resource_blockers": {
                "dsv4": {"artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT}
            },
        },
    )
    assert [blocker["id"] for blocker in unblocked_partial_ledger["blockers"]] == [
        "real_ui_dsv4_memory_blocked",
        "real_ui_unblocked_non_mimo_partial",
    ]
    assert (
        "partial_families:lfm25"
        in unblocked_partial_ledger["blockers"][-1]["evidence"]
    )

    mixed_ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["mimo_v2", "dsv4"],
            "mixed_model_identity_families": ["minimax"],
            "resource_blockers": {
                "dsv4": {"artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT}
            },
        },
    )
    assert [blocker["id"] for blocker in mixed_ledger["blockers"]] == [
        "real_ui_dsv4_memory_blocked",
        "real_ui_mixed_model_identity_blocked",
    ]


def test_release_blocker_ledger_uses_step37_vlm_runtime_audit_for_missing_ui_family():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        step37_vlm_runtime_audit={
            "status": "open",
            "artifact": CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
            "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
            "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
            "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
            "mlx_vlm_step3p7_importable": False,
            "mlx_vlm_step3p7_runtime_available": False,
            "mlx_vlm_step3p7_runtime": {
                "importable": False,
                "available": False,
            },
            "local_reference_vlm_runtime": {
                "runtime_kind": "torch_reference_not_mlx",
                "has_step3p7_model": True,
                "has_step3_vl_processor": True,
                "has_patch_image_inputs": True,
            },
            "step_jangtq_status": {
                "artifact_format": "jang",
                "quantization_method": "jang-importance",
                "quantization_profile": "JANG_2L",
                "quantization_backend": "mx.quantize",
                "step_jangtq_available": False,
                "step_jangtq_reason": "step_jangtq_not_made_current_artifact_is_jang_2l",
            },
            "mlx_vlm_implementation_contract": {
                "closest_reference_packages": [
                    "mlx_vlm.models.qwen3_vl_moe",
                    "mlx_vlm.models.lfm2_vl",
                    "mlx_vlm.models.gemma4",
                ],
                "required_module_files": [
                    "__init__.py",
                    "config.py",
                    "language.py",
                    "step3p7.py",
                    "vision.py",
                    "processing_step3p7.py",
                ],
                "required_capabilities": [
                    "image_patch_processing",
                    "2d_vision_rope",
                    "full_and_sliding_attention_cache",
                    "head_wise_attention_gate",
                    "multimodal_embedding_merge",
                ],
                "explicitly_rejected_clearance_paths": [
                    "text_only_bridge",
                    "importable_stub",
                ],
            },
            "source_owned_runtime_progress": {
                "module": "vmlx_engine.models.step3p7_mlx_vlm",
                "config_layer_available": True,
                "model_layer_available": True,
                "vision_layer_available": True,
                "projector_layer_available": True,
                "processor_layer_available": True,
                "image_embeds_merge_available": True,
                "pixel_values_processor_available": True,
                "vision_patch_embed_available": True,
                "vision_abs_posemb_resize_available": True,
                "vision_downsamplers_available": True,
                "release_clearance": "source_runtime_surface_present_needs_live_proof",
            },
        },
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["step37"],
            "resource_blockers": {},
        },
    )

    assert ledger == {
        "status": "open",
        "blockers": [
            {
                "id": "real_ui_step37_vlm_runtime_missing",
                "status": "open",
                "evidence": CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
                "next_proof": (
                    "Implement and prove a real Step3p7 VLM runtime path before "
                    "rerunning the Step3.7 real Electron UI image proof."
                ),
                "details": {
                    "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
                    "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
                    "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
                    "mlx_vlm_step3p7_importable": False,
                    "mlx_vlm_step3p7_runtime_available": False,
                    "mlx_vlm_step3p7_runtime": {
                        "importable": False,
                        "available": False,
                    },
                    "local_reference_vlm_runtime": {
                        "runtime_kind": "torch_reference_not_mlx",
                        "has_step3p7_model": True,
                        "has_step3_vl_processor": True,
                        "has_patch_image_inputs": True,
                    },
                    "step_jangtq_status": {
                        "artifact_format": "jang",
                        "quantization_method": "jang-importance",
                        "quantization_profile": "JANG_2L",
                        "quantization_backend": "mx.quantize",
                        "step_jangtq_available": False,
                        "step_jangtq_reason": "step_jangtq_not_made_current_artifact_is_jang_2l",
                    },
                    "mlx_vlm_implementation_contract": {
                        "closest_reference_packages": [
                            "mlx_vlm.models.qwen3_vl_moe",
                            "mlx_vlm.models.lfm2_vl",
                            "mlx_vlm.models.gemma4",
                        ],
                        "required_module_files": [
                            "__init__.py",
                            "config.py",
                            "language.py",
                            "step3p7.py",
                            "vision.py",
                            "processing_step3p7.py",
                        ],
                        "required_capabilities": [
                            "image_patch_processing",
                            "2d_vision_rope",
                            "full_and_sliding_attention_cache",
                            "head_wise_attention_gate",
                            "multimodal_embedding_merge",
                        ],
                        "explicitly_rejected_clearance_paths": [
                            "text_only_bridge",
                            "importable_stub",
                        ],
                    },
                    "source_owned_runtime_progress": {
                        "module": "vmlx_engine.models.step3p7_mlx_vlm",
                        "config_layer_available": True,
                        "model_layer_available": True,
                        "vision_layer_available": True,
                        "projector_layer_available": True,
                        "processor_layer_available": True,
                        "image_embeds_merge_available": True,
                        "pixel_values_processor_available": True,
                        "vision_patch_embed_available": True,
                        "vision_abs_posemb_resize_available": True,
                        "vision_downsamplers_available": True,
                        "release_clearance": "source_runtime_surface_present_needs_live_proof",
                    },
                },
            }
        ],
        "deferred_release_families": [
            {"family": "mimo_v2", "reason": "deferred_out_of_release_scope"}
        ],
    }


def test_release_blocker_ledger_keeps_step37_as_live_ui_gap_when_runtime_audit_passes():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        step37_vlm_runtime_audit={
            "status": "pass",
            "artifact": CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
            "root_cause": "step37_vlm_runtime_present_or_not_applicable",
            "release_clearance": "audit_does_not_block_release",
            "mlx_vlm_step3p7_runtime_available": True,
        },
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["step37"],
            "resource_blockers": {},
        },
    )

    assert ledger == {
        "status": "open",
        "blockers": [
            {
                "id": "real_ui_unblocked_non_mimo_missing",
                "status": "open",
                "evidence": (
                    "current_proof_sweep.real_ui_live_model_matrix."
                    "missing_families:step37"
                ),
                "next_proof": (
                    "Run real Electron UI proofs for the missing "
                    "non-MiMo, non-resource-blocked families on this "
                    "laptop before release."
                ),
            }
        ],
        "deferred_release_families": [
            {"family": "mimo_v2", "reason": "deferred_out_of_release_scope"}
        ],
    }


def test_release_manifest_rejects_step37_audit_without_reference_runtime_markers(tmp_path):
    artifact = tmp_path / CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "open",
                "model_type": "step3p7",
                "text_model_type": "step3p5",
                "has_vision_config": True,
                "jang_has_vision": True,
                "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
                "mlx_vlm_step3p7_runtime_available": False,
                "local_bridge_kind": "text_only",
                "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
                "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
                "local_reference_vlm_runtime": {
                    "runtime_kind": "torch_reference_not_mlx",
                    "has_modeling_step3p7": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = _validate_current_step37_vlm_runtime_audit(tmp_path)

    assert result["status"] == "fail"
    assert (
        "missing_step37_reference_runtime_marker:has_2d_vision_rope"
        in result["failures"]
    )
    assert (
        "missing_step37_reference_runtime_marker:has_head_wise_attention_gate"
        in result["failures"]
    )


def test_release_manifest_rejects_step37_audit_without_jangtq_absence_marker(tmp_path):
    artifact = tmp_path / CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "open",
                "model_type": "step3p7",
                "text_model_type": "step3p5",
                "has_vision_config": True,
                "jang_has_vision": True,
                "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
                "mlx_vlm_step3p7_runtime_available": False,
                "local_bridge_kind": "text_only",
                "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
                "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
                "local_reference_vlm_runtime": {
                    "runtime_kind": "torch_reference_not_mlx",
                    "has_modeling_step3p7": True,
                    "has_processing_step3": True,
                    "has_vision_encoder": True,
                    "has_configuration_step3p7": True,
                    "has_step3p7_model": True,
                    "has_conditional_generation": True,
                    "has_step3_vl_processor": True,
                    "has_multimodal_merge": True,
                    "has_patch_image_inputs": True,
                    "has_2d_vision_rope": True,
                    "has_full_and_sliding_attention_masks": True,
                    "has_qk_norms": True,
                    "has_head_wise_attention_gate": True,
                    "has_step3p7_config": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = _validate_current_step37_vlm_runtime_audit(tmp_path)

    assert result["status"] == "fail"
    assert "missing_step37_jangtq_absence_marker" in result["failures"]


def test_release_manifest_rejects_step37_audit_without_mlx_vlm_port_contract(tmp_path):
    artifact = tmp_path / CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "open",
                "model_type": "step3p7",
                "text_model_type": "step3p5",
                "has_vision_config": True,
                "jang_has_vision": True,
                "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
                "mlx_vlm_step3p7_runtime_available": False,
                "local_bridge_kind": "text_only",
                "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
                "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
                "local_reference_vlm_runtime": {
                    "runtime_kind": "torch_reference_not_mlx",
                    "has_modeling_step3p7": True,
                    "has_processing_step3": True,
                    "has_vision_encoder": True,
                    "has_configuration_step3p7": True,
                    "has_step3p7_model": True,
                    "has_conditional_generation": True,
                    "has_step3_vl_processor": True,
                    "has_multimodal_merge": True,
                    "has_patch_image_inputs": True,
                    "has_2d_vision_rope": True,
                    "has_full_and_sliding_attention_masks": True,
                    "has_qk_norms": True,
                    "has_head_wise_attention_gate": True,
                    "has_step3p7_config": True,
                },
                "step_jangtq_status": {
                    "quantization_profile": "JANG_2L",
                    "step_jangtq_available": False,
                    "step_jangtq_reason": "step_jangtq_not_made_current_artifact_is_jang_2l",
                },
            }
        ),
        encoding="utf-8",
    )

    result = _validate_current_step37_vlm_runtime_audit(tmp_path)

    assert result["status"] == "fail"
    assert "missing_step37_mlx_vlm_implementation_contract" in result["failures"]


def test_current_proof_sweep_accepts_expected_mimo_failed_smoke_as_open_gap():
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["mimo_v2_jang2l"]

    assert _live_smoke_gap_is_expected_mimo_open(
        {
            "status": "fail",
            "missing": [],
            "not_pass": [{"artifact": artifact, "status": "fail"}],
            "non_mimo_status": "pass",
            "non_mimo_missing": [],
            "non_mimo_not_pass": [],
        },
        artifact,
    )


def test_live_smoke_summary_tracks_non_mimo_status_separately(tmp_path):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["mimo_v2_jang2l"]
    payload = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    payload["status"] = "fail"
    payload["failed"] = 1
    payload["results"][0]["status"] = "fail"
    payload["results"][0]["failures"] = ["expected_exact_ack_missing"]
    (tmp_path / artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["non_mimo_status"] == "pass"
    assert result["non_mimo_missing"] == []
    assert result["non_mimo_not_pass"] == []


def test_live_tool_smoke_summary_tracks_non_mimo_status_separately(tmp_path):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_tool_smoke_artifacts,
    )

    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["mimo_v2_jang2l"]
    payload = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    payload["status"] = "fail"
    payload["failed"] = 1
    payload["results"][0]["status"] = "fail"
    payload["results"][0]["failures"] = ["expected_tool_call_missing"]
    (tmp_path / artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_tool_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["non_mimo_status"] == "pass"
    assert result["non_mimo_missing"] == []
    assert result["non_mimo_not_pass"] == []


def test_release_regression_manifest_rejects_stale_current_objective_digest(tmp_path):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_current_objective_digest(
        tmp_path,
        open_requirements=[
            *EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "Unexpected objective row is still open",
        ],
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["component_ok"]["objective_digest"] is False
    assert result["objective_digest"]["unexpected_open_requirements"] == [
        "Unexpected objective row is still open"
    ]


def test_release_regression_manifest_rejects_missing_or_failing_current_artifacts(tmp_path):
    artifacts = list(CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values())
    for artifact in artifacts[1:]:
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        status = "fail" if artifact == artifacts[1] else "pass"
        path.write_text(f'{{"status":"{status}","failed":["example"]}}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["missing"] == [artifacts[0]]
    assert result["not_pass"] == [{"artifact": artifacts[1], "status": "fail"}]


def test_release_regression_manifest_rejects_stale_dsv4_open_requirement_details(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": {
                    "Real Electron UI cross-family live model matrix is release-cleared": _passing_open_requirement_details()[
                        "Real Electron UI cross-family live model matrix is release-cleared"
                    ],
                    "DSV4 long-output/code/file-generation quality is release-cleared": {
                        "details": {
                            "direct_off_exactness_boundary": {
                                "hidden_force_on_would_be_false_clearance": True,
                            },
                            "current_source_full_output_preflight": _passing_open_requirement_details()[
                                "DSV4 long-output/code/file-generation quality is release-cleared"
                            ]["details"]["current_source_full_output_preflight"],
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_exact_code_root_boundary",
        }
    ]


def test_release_regression_manifest_rejects_missing_dsv4_source_preflight_boundary(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    assert isinstance(dsv4_row, dict)
    details = dsv4_row["details"]
    assert isinstance(details, dict)
    details.pop("current_source_full_output_preflight")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_narrow_dsv4_source_preflight_cases(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["selected_cases"] = ["chat_off_rep1"]
    source_preflight["case_count"] = 1
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_accepts_vm_stat_dsv4_source_preflight_reason(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["reason"] = "insufficient_vm_stat_memory"
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["regression_suite"]["open_requirement_detail_failures"] == []


def test_release_regression_manifest_rejects_partial_four_case_dsv4_source_preflight(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["selected_cases"] = [
        "chat_off_rep1",
        "chat_off_no_punct_rep1",
        "responses_off_rep1",
        "responses_off_no_punct_rep1",
    ]
    source_preflight["case_count"] = 4
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_stale_dsv4_source_preflight_artifact(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["artifact"] = (
        "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-0625.json"
    )
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_missing_or_failing_covered_live_smoke_artifacts(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    missing_artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    failing_artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    (tmp_path / missing_artifact).unlink()
    row = dict(CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS["qwen36_moe_crack"])
    failing_result = {
        "model": "Qwen3.6-27B-MXFP4-CRACK",
        "row": row,
        "status": "fail",
        "failures": ["unexpected_cjk_visible_text"],
    }
    failing_result["requests"] = [
        {"label": label, "validation_failures": []}
        for label in _required_live_smoke_request_labels(
            "qwen36_moe_crack",
            failing_result,
        )
    ]
    (tmp_path / failing_artifact).write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 1,
                "results": [failing_result],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["live_smoke_summaries"]["missing"] == [missing_artifact]
    assert result["live_smoke_summaries"]["not_pass"] == [
        {
            "artifact": failing_artifact,
            "status": "pass",
            "completed": 1,
            "failed": 1,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-MXFP4-CRACK",
                    "status": "fail",
                    "failures": ["unexpected_cjk_visible_text"],
                    "missing_request_labels": [],
                    "request_validation_failures": [],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_top_level_fail_status(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    payload = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    payload["status"] = "fail"
    payload["failed"] = 0
    (tmp_path / artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "fail",
            "completed": 1,
            "failed": 0,
            "failing_results": [],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_missing_top_level_status(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    payload = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    payload.pop("status")
    (tmp_path / artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "missing_status",
            "completed": 1,
            "failed": 0,
            "failing_results": [],
        }
    ]


def test_release_regression_manifest_rejects_source_python_live_smoke_surface(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["ling_flash_tq"]
    monkeypatch.setattr(
        manifest,
        "CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS",
        {"ling_flash_tq": artifact},
    )
    row = dict(CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS["ling_flash_tq"])
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "model": row["name"],
                        "row": row,
                        "status": "pass",
                        "failures": [],
                        "command": [
                            ".venv/bin/python",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            row["name"],
                        ],
                        "requests": [
                            {"label": "text_cache_repeat_1", "validation_failures": []},
                            {"label": "text_cache_repeat_2", "validation_failures": []},
                            {"label": "text_multiturn_recall", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"][0]["failing_results"][0]["surface_issues"] == [
        "command_python_not_bundled"
    ]


def test_release_regression_manifest_rejects_missing_or_wrong_tool_smoke_artifacts(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)

    missing_artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["gemma4_crack"]
    failing_artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    (tmp_path / missing_artifact).unlink()
    payload = json.loads((tmp_path / failing_artifact).read_text(encoding="utf-8"))
    tool_request = payload["results"][0]["requests"][-1]
    tool_request["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"value": "example"}
    )
    (tmp_path / failing_artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["live_tool_smoke_summaries"]["missing"] == [missing_artifact]
    assert result["live_tool_smoke_summaries"]["not_pass"] == [
        {
            "artifact": failing_artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-MXFP4-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [
                        "tool_required:expected_tool_argument_missing"
                    ],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_tool_smoke_missing_top_level_status(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_tool_smoke_artifacts,
    )

    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["gemma4_crack"]
    payload = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    payload.pop("status")
    (tmp_path / artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_tool_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "missing_status",
            "completed": 1,
            "failed": 0,
            "failing_results": [],
        }
    ]


def test_release_regression_manifest_rejects_live_tool_smoke_without_repeat_cache_hit(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_tool_smoke_artifacts,
    )

    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["gemma4_crack"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    for request in payload["results"][0]["requests"]:
        if request.get("label") == "text_cache_repeat_2":
            request["usage"] = {"prompt_tokens_details": None}
            request["cache_summary"] = {
                "has_cache_hit": False,
                "cache_hit_tokens": 0,
            }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_tool_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [
                        "text_cache_repeat_2:expected_cache_hit_missing"
                    ],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_missing_or_changed_diagnostic_smoke_artifacts(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_diagnostic_live_smoke_artifacts,
    )

    _write_expected_diagnostic_live_smoke_artifacts(tmp_path)
    missing_artifact = CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS[
        "zaya_text_mxfp4_toolprobe"
    ]["artifact"]
    changed_artifact = CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS[
        "zaya_vl_jangtq4_prefix_toolprobe"
    ]["artifact"]
    (tmp_path / missing_artifact).unlink()
    payload = json.loads((tmp_path / changed_artifact).read_text(encoding="utf-8"))
    payload["status"] = "pass"
    payload["failed"] = 0
    payload["results"][0]["status"] = "pass"
    payload["results"][0]["failures"] = []
    (tmp_path / changed_artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_diagnostic_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["missing"] == [missing_artifact]
    assert result["not_expected"] == [
        {
            "artifact": changed_artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "matched_expected_failures": [],
            "unexpected_results": [
                {
                    "model": "zaya_vl_jangtq4_prefix_toolprobe",
                    "status": "pass",
                    "missing_expected_failure": {
                        "label": "tool_required",
                        "reason": "expected_tool_argument_missing",
                    },
                },
                {
                    "model": "zaya_vl_jangtq4_prefix_toolprobe",
                    "status": "pass",
                    "expected_status": "fail",
                },
                {
                    "model": "zaya_vl_jangtq4_prefix_toolprobe",
                    "status": "pass",
                    "completed": 1,
                    "failed": 0,
                    "reason": "diagnostic_artifact_did_not_record_failure",
                },
            ],
        }
    ]


def test_release_regression_manifest_tracks_zaya_vl_reasoning_media_diagnostic():
    expectation = CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS[
        "zaya_vl_jangtq4_reasoning_media_rerun"
    ]

    assert (
        expectation["artifact"]
        == "build/current-all-local-model-smoke-zaya-vl-jangtq4-tool-reasoning-media-bundled-20260526/summary.json"
    )
    assert expectation["expected_status"] == "fail"
    assert expectation["expected_label_reasons"] == {
        "reasoning_on": "empty_visible",
        "text_no_media_after_image": [
            "expected_no_media_missing",
            "unexpected_media_carryover_claim",
        ],
    }


def test_release_regression_manifest_rejects_live_smoke_missing_required_request_coverage(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Qwen3.6-27B-MXFP4-CRACK",
                            "model_type": "qwen3_5",
                            "is_mllm": True,
                            "supports_video": True,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Qwen3.6-27B-MXFP4-CRACK",
                        ],
                        "requests": [
                            {"label": "text_cache_repeat_1", "validation_failures": []},
                            {"label": "text_cache_repeat_2", "validation_failures": []},
                            {"label": "text_multiturn_recall", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-MXFP4-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [
                        "reasoning_on",
                        "text_no_media_after_image",
                        "text_no_media_after_video",
                        "vl_blue_image",
                        "vl_blue_image_repeat",
                        "vl_blue_video",
                        "vl_red_image_changed",
                    ],
                    "request_validation_failures": [],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_without_repeat_cache_hit(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    for request in payload["results"][0]["requests"]:
        if request.get("label") == "text_cache_repeat_2":
            request["usage"] = {"prompt_tokens_details": None}
            request["cache_summary"] = {
                "has_cache_hit": False,
                "cache_hit_tokens": 0,
            }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [
                        "text_cache_repeat_2:expected_cache_hit_missing"
                    ],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_video_capable_live_smoke_without_video_labels(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    artifact = "build/current-all-local-model-smoke-nemotron-omni-missing-video-test/summary.json"
    monkeypatch.setattr(
        manifest,
        "CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS",
        {"nemotron_omni_tq2_system_nomedia": artifact},
    )
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                            "model_type": "nemotron_h",
                            "is_mllm": True,
                            "supports_video": True,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Nemotron-Omni-Nano-JANGTQ-CRACK",
                        ],
                        "requests": [
                            {
                                "label": "text_cache_repeat_1",
                                "validation_failures": [],
                                "usage": {"prompt_tokens_details": None},
                                "cache_summary": {
                                    "has_cache_hit": False,
                                    "cache_hit_tokens": 0,
                                },
                            },
                            {
                                "label": "text_cache_repeat_2",
                                "validation_failures": [],
                                "usage": {
                                    "prompt_tokens_details": {
                                        "cached_tokens": 10,
                                        "cache_detail": "paged",
                                    }
                                },
                                "cache_summary": {
                                    "has_cache_hit": True,
                                    "cache_hit_tokens": 10,
                                },
                            },
                            {"label": "text_multiturn_recall", "validation_failures": []},
                            {"label": "reasoning_on", "validation_failures": []},
                            {"label": "vl_blue_image", "validation_failures": []},
                            {"label": "text_no_media_after_image", "validation_failures": []},
                            {"label": "vl_blue_image_repeat", "validation_failures": []},
                            {"label": "vl_red_image_changed", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [
                        "text_no_media_after_video",
                        "vl_blue_video",
                    ],
                    "request_validation_failures": [],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_uses_effective_runtime_video_capability(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS[
        "nemotron_omni_tq2_system_nomedia"
    ]
    monkeypatch.setattr(
        manifest,
        "CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS",
        {"nemotron_omni_tq2_system_nomedia": artifact},
    )
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                            "model_type": "nemotron_h",
                            "is_mllm": True,
                            "supports_video": True,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "capabilities": {
                            "body": {
                                "family": "nemotron_h",
                                "tool_parser": "nemotron",
                                "reasoning_parser": "deepseek_r1",
                                "modalities": ["text", "audio", "image"],
                            }
                        },
                        "probe_options": {
                            "include_media": True,
                            "include_video": False,
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Nemotron-Omni-Nano-JANGTQ-CRACK",
                        ],
                        "requests": [
                            {
                                "label": "text_cache_repeat_1",
                                "validation_failures": [],
                                "usage": {"prompt_tokens_details": None},
                                "cache_summary": {
                                    "has_cache_hit": False,
                                    "cache_hit_tokens": 0,
                                },
                            },
                            {
                                "label": "text_cache_repeat_2",
                                "validation_failures": [],
                                "usage": {
                                    "prompt_tokens_details": {
                                        "cached_tokens": 10,
                                        "cache_detail": "paged",
                                    }
                                },
                                "cache_summary": {
                                    "has_cache_hit": True,
                                    "cache_hit_tokens": 10,
                                },
                            },
                            {"label": "text_multiturn_recall", "validation_failures": []},
                            {"label": "reasoning_on", "validation_failures": []},
                            {"label": "vl_blue_image", "validation_failures": []},
                            {"label": "text_no_media_after_image", "validation_failures": []},
                            {"label": "vl_blue_image_repeat", "validation_failures": []},
                            {"label": "vl_red_image_changed", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "pass"
    assert result["not_pass"] == []


def test_release_regression_manifest_does_not_treat_skipped_video_probe_as_capability_proof(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    artifact = "build/current-all-local-model-smoke-video-probe-skipped/summary.json"
    monkeypatch.setattr(
        manifest,
        "CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS",
        {"nemotron_omni_tq2_system_nomedia": artifact},
    )
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                            "model_type": "nemotron_h",
                            "is_mllm": True,
                            "supports_video": True,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "probe_options": {
                            "include_media": True,
                            "include_video": False,
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Qwen3.6-27B-MXFP4-CRACK",
                        ],
                        "requests": [
                            {"label": "text_cache_repeat_1", "validation_failures": []},
                            {"label": "text_cache_repeat_2", "validation_failures": []},
                            {"label": "text_multiturn_recall", "validation_failures": []},
                            {"label": "reasoning_on", "validation_failures": []},
                            {"label": "vl_blue_image", "validation_failures": []},
                            {"label": "text_no_media_after_image", "validation_failures": []},
                            {"label": "vl_blue_image_repeat", "validation_failures": []},
                            {"label": "vl_red_image_changed", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"][0]["failing_results"][0]["missing_request_labels"] == [
        "text_no_media_after_video",
        "vl_blue_video",
    ]


def test_release_regression_manifest_rejects_live_smoke_wrong_model_identity(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Qwen3.6-27B-MXFP4-CRACK",
                            "model_type": "qwen3_5",
                            "is_mllm": True,
                            "supports_video": False,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Qwen3.6-27B-MXFP4-CRACK",
                        ],
                        "requests": [
                            {"label": "text_cache_repeat_1", "validation_failures": []},
                            {"label": "text_cache_repeat_2", "validation_failures": []},
                            {"label": "text_multiturn_recall", "validation_failures": []},
                            {"label": "reasoning_on", "validation_failures": []},
                            {"label": "vl_blue_image", "validation_failures": []},
                            {"label": "text_no_media_after_image", "validation_failures": []},
                            {"label": "vl_blue_image_repeat", "validation_failures": []},
                            {"label": "vl_red_image_changed", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-MXFP4-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [],
                    "identity_mismatches": {
                        "model_type": {
                            "expected": "gemma4",
                            "actual": "qwen3_5",
                        },
                        "name": {
                            "expected": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
                            "actual": "Qwen3.6-27B-MXFP4-CRACK",
                        },
                    },
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_wrong_capability_family(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["capabilities"] = {
        "code": 200,
        "body": {
            "family": "gemma4",
            "tool_parser": "qwen",
            "reasoning_parser": "qwen3",
            "supports_thinking": True,
            "modalities": ["text", "vision"],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-MXFP4-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [],
                    "identity_mismatches": {},
                    "capability_mismatches": {
                        "family": {
                            "expected": "qwen3_5",
                            "actual": "gemma4",
                        },
                    },
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_wrong_runtime_parsers(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["dsv4_jang_local"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["capabilities"] = {
        "code": 200,
        "body": {
            "family": "deepseek_v4",
            "tool_parser": "deepseek",
            "reasoning_parser": "qwen3",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "DeepSeek-V4-Flash-JANGTQ-K",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [],
                    "identity_mismatches": {},
                    "capability_mismatches": {
                        "tool_parser": {
                            "expected": "dsml",
                            "actual": "deepseek",
                        },
                        "reasoning_parser": {
                            "expected": "deepseek_r1",
                            "actual": "qwen3",
                        },
                    },
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_mimo_v2_stale_qwen_reasoning_parser(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["mimo_v2_jang2l"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["capabilities"] = {
        "code": 200,
        "body": {
            "family": "mimo_v2",
            "tool_parser": "xml_function",
            "reasoning_parser": "qwen3",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"][0]["artifact"] == artifact
    assert result["not_pass"][0]["failing_results"][0]["capability_mismatches"] == {
        "reasoning_parser": {
            "expected": "think_xml",
            "actual": "qwen3",
        }
    }


def test_release_regression_manifest_requires_mimo_v2_root_cause_artifacts(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NOCACHE_NO_KVQ_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_PROFILE_DIAGNOSTIC_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    assert "after-swa-runtime-sync" in CURRENT_MIMO_V2_JANG2L_NOCACHE_NO_KVQ_ARTIFACT

    router_path = tmp_path / CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT
    router_path.parent.mkdir(parents=True, exist_ok=True)
    router_path.write_text(
        json.dumps(
            {
                "weight_max_abs_diff": 0.0,
                "bias_max_abs_diff": 0.0,
                "topk_weight_max_abs_diff": 0.0,
                "score_max_abs_diff": 0.0,
                "topk_exact_rows": [True, True],
            }
        ),
        encoding="utf-8",
    )
    moe_path = tmp_path / CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT
    moe_path.parent.mkdir(parents=True, exist_ok=True)
    moe_path.write_text(
        json.dumps(
            {
                "row_stats": [
                    {"rel_l2": 0.603, "cosine": 0.813},
                    {"rel_l2": 0.437, "cosine": 0.901},
                ]
            }
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / CURRENT_MIMO_V2_JANG2L_PROFILE_DIAGNOSTIC_ARTIFACT
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {"profile": "source", "rel_l2_vs_source": 0.0},
                    {"profile": "2L", "rel_l2_vs_source": 0.178},
                ]
            }
        ),
        encoding="utf-8",
    )
    nocache_path = tmp_path / CURRENT_MIMO_V2_JANG2L_NOCACHE_NO_KVQ_ARTIFACT
    nocache_path.parent.mkdir(parents=True, exist_ok=True)
    nocache_path.write_text(
        json.dumps(
            {
                "mode": "disable_prefix_cache_kv_quant_none_after_swa_runtime_sync",
                "capabilities": {
                    "reasoning_parser": "think_xml",
                    "tool_parser": "xml_function",
                },
                "requests": [
                    {"label": "ack", "content_head": "- - - - -"},
                    {"label": "recall_check", "content_head": "-00 - - -"},
                    {"label": "reasoning", "content_head": ". - - -000"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["status"] == "pass"
    assert result["router_topk_exact"] is True
    assert result["moe_expert_distortion_seen"] is True
    assert result["profile_diagnostic_seen"] is True
    assert result["nocache_no_kvq_incoherent"] is True
    assert result["remote_evidence_only"] is False
    assert result["remote_artifacts"] == []
    assert result["local_release_clearance"] is True
    assert result["profile_rel_l2"] == {
        "source": 0.0,
        "2L": 0.178,
    }
    assert result["root_cause_candidate"] == (
        "mimo_v2_jang2l_2bit_routed_expert_distortion"
    )


def test_current_proof_sweep_requires_local_mimo_root_cause_artifacts():
    from tests.cross_matrix.release_regression_manifest import (
        validate_current_proof_sweep_artifacts,
    )

    result = validate_current_proof_sweep_artifacts(Path.cwd())

    assert result["mimo_v2_jang2l_root_cause"]["remote_evidence_only"] is False
    assert result["mimo_v2_jang2l_root_cause"]["remote_artifacts"] == []
    assert result["mimo_v2_jang2l_root_cause"]["missing"] == [
        CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
    ]
    assert result["component_ok"]["mimo_v2_jang2l_root_cause"] is True
    assert result["status"] == "fail"


def test_current_mimo_v2_proof_artifact_constants_are_local_only():
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SINK_AB_ARTIFACT,
    )

    artifacts = [
        CURRENT_MIMO_V2_JANG2L_SINK_AB_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
    ]

    assert all(artifact.startswith("build/current-") for artifact in artifacts)
    assert not any("/remote-" in artifact for artifact in artifacts)


def test_release_regression_manifest_rejects_missing_mimo_v2_root_cause_artifacts(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NOCACHE_NO_KVQ_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_PROFILE_DIAGNOSTIC_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["status"] == "missing"
    assert result["missing"] == [
        CURRENT_MIMO_V2_JANG2L_ROUTER_TOPK_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_MOE_OUTPUT_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_PROFILE_DIAGNOSTIC_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NOCACHE_NO_KVQ_ARTIFACT,
    ]


def test_release_regression_manifest_rejects_unexpected_current_regression_suite_state(tmp_path):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": ["new_failure"],
                "open_requirements": [
                    *EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                    "Server max output/context wiring regressed",
                ],
                "open_requirement_details": _passing_open_requirement_details(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"] == {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "pass",
        "failed_steps": ["new_failure"],
        "open_requirements": [
            *EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "Server max output/context wiring regressed",
        ],
        "open_requirement_details": _passing_open_requirement_details(),
        "unexpected_open_requirements": ["Server max output/context wiring regressed"],
        "missing_expected_open_requirements": [],
        "open_requirement_detail_failures": [],
    }


def test_release_regression_manifest_rejects_stale_current_suite_source_hash(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    source_rel = "tests/cross_matrix/summarize_objective_proof.py"
    source_path = tmp_path / source_rel
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("new source\n", encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "source_hashes": {
                    source_rel: hashlib.sha256(b"old source\n").hexdigest()
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manifest, "CURRENT_SUITE_SOURCE_HASH_FILES", (source_rel,))

    result = manifest._validate_current_regression_suite_artifact(tmp_path)

    assert result["stale_source_hashes"] == [
        {
            "path": source_rel,
            "artifact_sha256": hashlib.sha256(b"old source\n").hexdigest(),
            "current_sha256": hashlib.sha256(b"new source\n").hexdigest(),
        }
    ]
    assert manifest._current_regression_suite_state_is_acceptable(result) is False


def test_release_regression_manifest_rejects_missing_current_suite_source_hashes(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    source_rel = "tests/cross_matrix/summarize_objective_proof.py"
    source_path = tmp_path / source_rel
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("current source\n", encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "source_hashes": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manifest, "CURRENT_SUITE_SOURCE_HASH_FILES", (source_rel,))

    result = manifest._validate_current_regression_suite_artifact(tmp_path)

    assert result["missing_source_hashes"] == [source_rel]
    assert manifest._current_regression_suite_state_is_acceptable(result) is False


def test_release_regression_manifest_requires_dsv4_generation_boundary_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "vmlx_engine/scheduler.py",
        "vmlx_engine/utils/dsv4_batch_generator.py",
        "tests/cross_matrix/run_dsv4_route_mode_code_exactness.py",
        "tests/cross_matrix/run_dsv4_default_cache_tool_loop_gate.py",
        "tests/test_dsv4_route_mode_code_exactness.py",
        "tests/test_dsv4_default_cache_tool_loop_gate.py",
        "tests/test_objective_proof_digest.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))


def test_release_regression_manifest_source_hash_list_matches_current_suite_runner():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_current_regression_suite as suite

    assert manifest.CURRENT_SUITE_SOURCE_HASH_FILES == suite.CURRENT_SUITE_SOURCE_HASH_FILES


def test_release_regression_manifest_runner_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_release_regression_manifest as runner

    assert runner.DEFAULT_OUT == Path(
        "build/current-release-regression-manifest-20260531-childstream-epipe-guard.json"
    )


def test_release_regression_manifest_requires_model_matrix_contract_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "tests/cross_matrix/run_cache_architecture_contract.py",
        "tests/cross_matrix/run_generation_defaults_contract.py",
        "tests/cross_matrix/run_jang_model_compat_contract.py",
        "tests/cross_matrix/run_mcp_policy_contract.py",
        "tests/cross_matrix/run_model_family_detection_contract.py",
        "tests/cross_matrix/run_model_artifact_format_contract.py",
        "tests/cross_matrix/run_native_mtp_contract.py",
        "tests/cross_matrix/run_noheavy_api_cache_contract.py",
        "tests/cross_matrix/run_parser_registry_contract.py",
        "tests/cross_matrix/run_panel_tool_security_contract.py",
        "tests/cross_matrix/run_production_family_audit.py",
        "tests/cross_matrix/run_reasoning_template_contract.py",
        "tests/cross_matrix/run_release_regression_manifest.py",
        "tests/cross_matrix/run_release_surface_contract.py",
        "tests/cross_matrix/run_tool_call_contract.py",
        "tests/cross_matrix/run_vl_media_cache_contract.py",
        "tests/test_all_local_model_smoke.py",
        "vmlx_engine/api/tool_calling.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))


def test_release_regression_manifest_requires_release_blocker_boundary_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "tests/cross_matrix/run_installed_app_runtime_parity_audit.py",
        "tests/cross_matrix/run_issue175_177_installed_runtime_audit.py",
        "tests/cross_matrix/run_issue175_177_live_runtime_audit.py",
        "tests/cross_matrix/run_issue175_179_release_boundary_audit.py",
        "tests/cross_matrix/run_issue179_minimax_k_model_manifest.py",
        "tests/cross_matrix/run_issue179_reporter_parity_metadata.py",
        "tests/cross_matrix/run_issue179_minimax_k_root_cause_audit.py",
        "tests/cross_matrix/run_issue179_responses_cancel_probe.py",
        "tests/cross_matrix/run_real_ui_dsv4_memory_preflight.py",
        "tests/test_installed_app_runtime_parity_audit.py",
        "tests/test_issue175_177_installed_runtime_audit.py",
        "tests/test_issue175_177_live_runtime_audit.py",
        "tests/test_issue175_179_release_boundary_audit.py",
        "tests/test_issue179_minimax_k_model_manifest.py",
        "tests/test_issue179_reporter_parity_metadata.py",
        "tests/test_issue179_minimax_k_root_cause_audit.py",
        "tests/test_issue179_responses_cancel_probe.py",
        "tests/test_cancellation.py",
        "tests/test_real_ui_dsv4_memory_preflight.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_source_hashes_all_referenced_code_files():
    from tests.cross_matrix import release_regression_manifest as manifest

    text_parts: list[str] = []
    for row in manifest.build_manifest()["rows"]:
        for key in ("commands", "artifacts", "proves"):
            values = row.get(key, [])
            if isinstance(values, list):
                text_parts.extend(str(value) for value in values)
    joined = "\n".join(text_parts)
    referenced_paths = {
        match.rstrip(".,;)")
        for match in re.findall(
            r"((?:tests|panel|bench|vmlx_engine)/[^\s`\"']+)",
            joined,
        )
    }
    referenced_code = {
        path
        for path in referenced_paths
        if path.endswith((".py", ".mjs", ".ts", ".tsx", ".sh"))
        and Path(path).is_file()
    }

    missing = sorted(referenced_code - set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))

    assert missing == []


def test_release_regression_manifest_requires_runtime_launch_and_mllm_cache_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "bench/native_mtp_speed_ab.py",
        "tests/cross_matrix/run_decode_speed_gate.py",
        "vmlx_engine/block_disk_store.py",
        "vmlx_engine/cli.py",
        "vmlx_engine/engine/batched.py",
        "vmlx_engine/engine/simple.py",
        "vmlx_engine/mllm_scheduler.py",
        "vmlx_engine/models/mllm.py",
        "vmlx_engine/paged_cache.py",
        "vmlx_engine/reranker.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_requires_panel_api_settings_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "panel/src/main/api-gateway.ts",
        "panel/src/main/engine-manager.ts",
        "panel/scripts/live-chat-tools-reasoning-proof.mjs",
        "panel/src/main/index.ts",
        "panel/src/main/server.ts",
        "panel/src/main/sessions.ts",
        "panel/src/main/ipc/chat.ts",
        "panel/src/main/ipc/developer.ts",
        "panel/src/main/ipc/image.ts",
        "panel/src/main/ipc/imageGenerationState.ts",
        "panel/src/main/ipc/models.ts",
        "panel/src/main/model-config-registry.ts",
        "panel/src/main/process-manager.ts",
        "panel/src/main/tools/executor.ts",
        "panel/src/renderer/src/components/chat/MessageBubble.tsx",
        "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
        "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
        "panel/src/shared/reasoningParserAliases.ts",
        "panel/tests/api-gateway-ollama-behavior.test.ts",
        "panel/tests/api-gateway-ollama.test.ts",
        "panel/tests/api-gateway-single-model.behavior.test.ts",
        "panel/tests/generation-defaults.test.ts",
        "panel/tests/image-system.test.ts",
        "panel/tests/interleaved-reasoning-render.test.ts",
        "panel/tests/model-config-registry.test.ts",
        "panel/tests/settings-flow.test.ts",
        "tests/cross_matrix/run_api_surface_contract.py",
        "tests/cross_matrix/run_max_output_context_contract.py",
        "tests/cross_matrix/run_noheavy_panel_settings_contract.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_requires_focused_pytest_gate_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "tests/test_objective_proof_digest.py",
        "tests/test_dsv4_default_cache_tool_loop_gate.py",
        "tests/test_release_gate_python_app.py",
        "tests/test_current_regression_suite.py",
        "tests/test_release_regression_manifest.py",
        "tests/test_model_family_detection_contract.py",
        "tests/test_mcp_policy_contract.py",
        "tests/test_vl_media_cache_contract.py",
        "tests/test_batching.py",
        "tests/test_dsv4_batch_generator_speed.py",
        "tests/test_scheduler_repetition_context.py",
        "tests/test_dsv4_paged_cache.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_requires_dirty_contract_unit_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "tests/test_api_surface_contract.py",
        "tests/test_cache_architecture_contract.py",
        "tests/test_dsml_tool_parser.py",
        "tests/test_engine_audit.py",
        "tests/test_generation_defaults_contract.py",
        "tests/test_image_api.py",
        "tests/test_image_gen.py",
        "tests/test_local_generation_metadata_audit.py",
        "tests/test_mllm_continuous_batching.py",
        "tests/test_mllm_scheduler_cache.py",
        "tests/test_model_config_registry.py",
        "tests/test_mlx_memory_cleanup.py",
        "tests/test_packaged_integrity_contract.py",
        "tests/test_paged_cache_unit.py",
        "tests/test_panel_cli_flag_contract.py",
        "tests/test_reasoning_modes.py",
        "tests/test_runtime_memory_stress_probe.py",
        "tests/test_server.py",
        "tests/test_tool_format.py",
        "tests/test_tool_parsers.py",
        "vmlx_engine/image_gen.py",
        "vmlx_engine/model_config_registry.py",
        "vmlx_engine/model_configs.py",
        "vmlx_engine/mlx_memory.py",
        "vmlx_engine/reasoning/__init__.py",
        "vmlx_engine/reasoning/think_xml_parser.py",
        "vmlx_engine/tool_parsers/__init__.py",
        "vmlx_engine/tool_parsers/xml_function_tool_parser.py",
        "vmlx_engine/tool_parsers/zaya_tool_parser.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_allows_current_suite_bootstrap_self_reference(tmp_path):
    result = {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "open",
        "failed_steps": ["release_regression_manifest"],
        "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
    }

    assert _current_regression_suite_state_is_acceptable(result)


def test_release_regression_manifest_allows_current_suite_open_with_only_expected_requirements(
    tmp_path,
):
    result = {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "open",
        "failed_steps": [],
        "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
        "open_requirement_detail_failures": [],
    }

    assert _current_regression_suite_state_is_acceptable(result)


def test_release_regression_manifest_rejects_incomplete_current_max_output_context_matrix(tmp_path):
    max_output_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == max_output_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "server startup maxTokens and chat maxTokens remain independent when both are set"
                        ],
                        "failed": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["max_output_context_matrix"]["artifact"] == max_output_artifact
    assert result["max_output_context_matrix"]["status"] == "pass"
    assert result["max_output_context_matrix"]["missing_markers"] == [
        "server startup maxTokens and chat maxTokens remain independent when both are set"
    ]
    assert result["max_output_context_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_packaged_integrity_matrix(tmp_path):
    packaged_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == packaged_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "failed": ["bundled_python_verifier"],
                        "known_expected_release_gate_open_requirements": [
                            *EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                            "Unexpected release gate blocker",
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["packaged_integrity_matrix"]["artifact"] == packaged_artifact
    assert result["packaged_integrity_matrix"]["status"] == "pass"
    assert result["packaged_integrity_matrix"]["failed"] == ["bundled_python_verifier"]
    assert result["packaged_integrity_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS[-1]
    ]
    assert result["packaged_integrity_matrix"]["unexpected_open_requirements"] == [
        "Unexpected release gate blocker"
    ]
    assert result["packaged_integrity_matrix"]["missing_expected_open_requirements"] == []


def test_release_regression_manifest_rejects_incomplete_current_release_surface_matrix(tmp_path):
    assert "staged_source_version_not_public" in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS

    release_surface_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == release_surface_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["release_surface_matrix"]["artifact"] == release_surface_artifact
    assert result["release_surface_matrix"]["status"] == "pass"
    assert result["release_surface_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_model_family_matrix(tmp_path):
    model_family_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == model_family_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS[:-1],
                        "missing_rows": [EXPECTED_CURRENT_MODEL_FAMILY_ROWS[-1]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["model_family_matrix"]["artifact"] == model_family_artifact
    assert result["model_family_matrix"]["status"] == "pass"
    assert result["model_family_matrix"]["missing_rows"] == [EXPECTED_CURRENT_MODEL_FAMILY_ROWS[-1]]
    assert result["model_family_matrix"]["missing_expected_rows"] == [EXPECTED_CURRENT_MODEL_FAMILY_ROWS[-1]]


def test_release_regression_manifest_rejects_incomplete_current_model_artifact_matrix(tmp_path):
    model_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == model_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": ["test_native_mtp_detection_uses_weights_not_path_name"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["model_artifact_matrix"]["artifact"] == model_artifact
    assert result["model_artifact_matrix"]["status"] == "pass"
    assert result["model_artifact_matrix"]["missing_markers"] == [
        "test_native_mtp_detection_uses_weights_not_path_name"
    ]
    assert result["model_artifact_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_cache_architecture_matrix(tmp_path):
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == cache_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": ["test_mllm_stats_include_cache_fields"],
                        "missing_api_checks": ["turboquant_disk_roundtrip"],
                        "missing_api_command_markers": ["generic_turboquant_patcher_skips_hybrid_ssm"],
                        "missing_panel_markers": ["deepseek-v4 disables composite prefix cache by default even with stale cache config"],
                        "cache_family_matrix": _passing_cache_family_matrix(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["cache_architecture_matrix"]["artifact"] == cache_artifact
    assert result["cache_architecture_matrix"]["status"] == "pass"
    assert result["cache_architecture_matrix"]["missing_markers"] == [
        "test_mllm_stats_include_cache_fields"
    ]
    assert result["cache_architecture_matrix"]["missing_api_checks"] == [
        "turboquant_disk_roundtrip"
    ]
    assert result["cache_architecture_matrix"]["missing_api_command_markers"] == [
        "generic_turboquant_patcher_skips_hybrid_ssm"
    ]
    assert result["cache_architecture_matrix"]["missing_panel_markers"] == [
        "deepseek-v4 disables composite prefix cache by default even with stale cache config"
    ]
    assert result["cache_architecture_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_missing_current_cache_family_matrix(tmp_path):
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == cache_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS
                        },
                        "missing_markers": [],
                        "missing_api_checks": [],
                        "missing_api_command_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["cache_architecture_matrix"]["missing_family_rows"] == list(
        EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
    )
    assert result["cache_architecture_matrix"]["failed_family_rows"] == list(
        EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
    )


def test_release_regression_manifest_rejects_incomplete_current_parser_registry_matrix(tmp_path):
    parser_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == parser_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": ["passes MiniMax through the registered minimax_m2 reasoning parser"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["parser_registry_matrix"]["artifact"] == parser_artifact
    assert result["parser_registry_matrix"]["status"] == "pass"
    assert result["parser_registry_matrix"]["missing_markers"] == [
        "passes MiniMax through the registered minimax_m2 reasoning parser"
    ]
    assert result["parser_registry_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_generation_defaults_matrix(tmp_path):
    generation_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == generation_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "test_request_output_caps_override_server_default_without_touching_context_cap"
                        ],
                        "generation_defaults_family_matrix": _passing_generation_defaults_family_matrix(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["generation_defaults_matrix"]["artifact"] == generation_artifact
    assert result["generation_defaults_matrix"]["status"] == "pass"
    assert result["generation_defaults_matrix"]["missing_markers"] == [
        "test_request_output_caps_override_server_default_without_touching_context_cap"
    ]
    assert result["generation_defaults_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_api_surface_matrix(tmp_path):
    api_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_API_SURFACE_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_nested_checks": ["request_output_caps_override_server_default"],
                        "missing_nested_markers": ["streaming_cache_detail_usage"],
                        "missing_panel_markers": [
                            "keeps Responses maxTokens as output budget only, never prompt context"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["api_surface_matrix"]["artifact"] == api_artifact
    assert result["api_surface_matrix"]["status"] == "pass"
    assert result["api_surface_matrix"]["missing_nested_checks"] == [
        "request_output_caps_override_server_default"
    ]
    assert result["api_surface_matrix"]["missing_nested_markers"] == [
        "streaming_cache_detail_usage"
    ]
    assert result["api_surface_matrix"]["missing_panel_markers"] == [
        "keeps Responses maxTokens as output budget only, never prompt context"
    ]
    assert result["api_surface_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_API_SURFACE_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_reasoning_template_matrix(tmp_path):
    reasoning_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == reasoning_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "test_dsv4_reasoning_effort_preserves_requested_rails"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["reasoning_template_matrix"]["artifact"] == reasoning_artifact
    assert result["reasoning_template_matrix"]["status"] == "pass"
    assert result["reasoning_template_matrix"]["missing_markers"] == [
        "test_dsv4_reasoning_effort_preserves_requested_rails"
    ]
    assert result["reasoning_template_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_tool_call_matrix(tmp_path):
    tool_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == tool_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_TOOL_CALL_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "panel max tool iterations caps tool loops"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["tool_call_matrix"]["artifact"] == tool_artifact
    assert result["tool_call_matrix"]["status"] == "pass"
    assert result["tool_call_matrix"]["missing_markers"] == [
        "panel max tool iterations caps tool loops"
    ]
    assert result["tool_call_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_TOOL_CALL_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_native_mtp_matrix(tmp_path):
    mtp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mtp_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_NATIVE_MTP_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "DSV4 additional args cannot reenable native MTP or deterministic sampling policy"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["native_mtp_matrix"]["artifact"] == mtp_artifact
    assert result["native_mtp_matrix"]["status"] == "pass"
    assert result["native_mtp_matrix"]["missing_markers"] == [
        "DSV4 additional args cannot reenable native MTP or deterministic sampling policy"
    ]
    assert result["native_mtp_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_NATIVE_MTP_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_vl_media_matrix(tmp_path):
    vl_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == vl_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_VL_MEDIA_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_engine_markers": ["test_qwen36_jangtq_is_detected_as_vl"],
                        "missing_panel_markers": [
                            "keeps mxfp8 Qwen hybrid VLM multimodal"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["vl_media_matrix"]["artifact"] == vl_artifact
    assert result["vl_media_matrix"]["status"] == "pass"
    assert result["vl_media_matrix"]["missing_engine_markers"] == [
        "test_qwen36_jangtq_is_detected_as_vl"
    ]
    assert result["vl_media_matrix"]["missing_panel_markers"] == [
        "keeps mxfp8 Qwen hybrid VLM multimodal"
    ]
    assert result["vl_media_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_VL_MEDIA_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_mcp_policy_matrix(tmp_path):
    mcp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS
                        },
                        "missing_engine_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mcp_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_MCP_POLICY_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "panel MCP import redacts managed metadata"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["mcp_policy_matrix"]["artifact"] == mcp_artifact
    assert result["mcp_policy_matrix"]["status"] == "pass"
    assert result["mcp_policy_matrix"]["missing_markers"] == [
        "panel MCP import redacts managed metadata"
    ]
    assert result["mcp_policy_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_MCP_POLICY_CHECKS[-1]
    ]


def test_release_regression_manifest_runner_embeds_current_proof_validation(tmp_path):
    model_family_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    model_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    parser_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]
    generation_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]
    api_cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "noheavy-api-cache-endpoint-runtime"
    ]
    api_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    reasoning_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    tool_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    panel_tool_security_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "panel-tool-security-loop-boundary"
    ]
    mtp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    vl_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    mcp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    max_output_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    jang_model_compat_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "jang-model-compat-runtime-boundary"
    ]
    packaged_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
    release_surface_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == model_family_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == max_output_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS
                        },
                        "missing_markers": [],
                        "failed": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == packaged_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS
                        },
                        "failed": [],
                        "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == panel_tool_security_artifact:
            path.write_text(
                json.dumps({"status": "pass", "returncode": 0}) + "\n",
                encoding="utf-8",
            )
        elif artifact == jang_model_compat_artifact:
            path.write_text(
                json.dumps({"status": "pass", "failed": []}) + "\n",
                encoding="utf-8",
            )
        elif artifact == release_surface_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == model_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == cache_artifact:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == parser_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == generation_artifact:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_cache_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True
                            for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == reasoning_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == tool_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mtp_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == vl_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS
                        },
                        "missing_engine_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mcp_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    _write_expected_diagnostic_live_smoke_artifacts(tmp_path)
    _write_current_objective_digest(tmp_path)
    _write_open_mimo_v2_sink_ab_artifact(tmp_path)
    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)
    _write_passing_dev_ui_proof_artifacts(tmp_path)
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)

    expected_real_ui_live_model_proof = validate_current_proof_sweep_artifacts(
        tmp_path
    )["real_ui_live_model_proof"]
    expected_real_ui_live_model_matrix = validate_current_proof_sweep_artifacts(
        tmp_path
    )["real_ui_live_model_matrix"]
    expected_issue175_179_release_boundary_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue175_179_release_boundary_audit"]
    expected_issue179_minimax_k_root_cause_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue179_minimax_k_root_cause_audit"]
    expected_mimo_v2_jang2l_sink_ab = validate_current_proof_sweep_artifacts(
        tmp_path
    )["mimo_v2_jang2l_sink_ab"]
    expected_mimo_v2_jang2l_root_cause = validate_current_proof_sweep_artifacts(
        tmp_path
    )["mimo_v2_jang2l_root_cause"]
    expected_installed_app_runtime_parity_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["installed_app_runtime_parity_audit"]
    expected_staged_app_runtime_parity_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["staged_app_runtime_parity_audit"]
    expected_issue175_177_installed_runtime_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue175_177_installed_runtime_audit"]
    expected_issue175_177_live_runtime_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue175_177_live_runtime_audit"]
    expected_real_ui_dsv4_memory_preflight = validate_current_proof_sweep_artifacts(
        tmp_path
    )["real_ui_dsv4_memory_preflight"]
    expected_step37_vlm_runtime_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["step37_vlm_runtime_audit"]
    expected_objective_digest = validate_current_proof_sweep_artifacts(
        tmp_path
    )["objective_digest"]
    expected_release_blocker_ledger = validate_current_proof_sweep_artifacts(
        tmp_path
    )["release_blocker_ledger"]
    expected_component_ok = validate_current_proof_sweep_artifacts(
        tmp_path
    )["component_ok"]
    artifact = build_manifest_artifact(tmp_path)

    assert artifact["current_proof_sweep"] == {
        "status": "fail",
        "component_ok": expected_component_ok,
        "failed_components": [
            name for name, ok in expected_component_ok.items() if not ok
        ],
        "non_mimo_component_status": (
            "pass"
            if all(
                ok
                for name, ok in expected_component_ok.items()
                if name
                not in {
                    "mimo_v2_jang2l_sink_ab",
                    "mimo_v2_jang2l_root_cause",
                }
            )
            else "fail"
        ),
        "missing": [],
        "not_pass": [],
        "regression_suite": {
            "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
            "status": "pass",
            "failed_steps": [],
            "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "open_requirement_details": _passing_open_requirement_details(),
            "unexpected_open_requirements": [],
            "missing_expected_open_requirements": [],
            "open_requirement_detail_failures": [],
        },
        "objective_digest": expected_objective_digest,
        "model_family_matrix": {
            "artifact": model_family_artifact,
            "status": "pass",
            "matched_rows": list(EXPECTED_CURRENT_MODEL_FAMILY_ROWS),
            "missing_rows": [],
            "missing_expected_rows": [],
        },
        "model_artifact_matrix": {
            "artifact": model_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "cache_architecture_matrix": {
            "artifact": cache_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS},
            "missing_markers": [],
            "missing_api_checks": [],
            "missing_api_command_markers": [],
            "missing_panel_markers": [],
            "missing_family_rows": [],
            "failed_family_rows": [],
            "cache_family_matrix": _passing_cache_family_matrix(),
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "parser_registry_matrix": {
            "artifact": parser_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "generation_defaults_matrix": {
            "artifact": generation_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS},
            "missing_markers": [],
            "missing_family_rows": [],
            "failed_family_rows": [],
            "generation_defaults_family_matrix": _passing_generation_defaults_family_matrix(),
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "noheavy_api_cache_matrix": {
            "artifact": api_cache_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "api_surface_matrix": {
            "artifact": api_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS},
            "missing_nested_checks": [],
            "missing_nested_markers": [],
            "missing_panel_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "reasoning_template_matrix": {
            "artifact": reasoning_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "tool_call_matrix": {
            "artifact": tool_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "panel_tool_security_matrix": {
            "artifact": panel_tool_security_artifact,
            "status": "pass",
            "returncode": 0,
            "missing": [],
        },
        "native_mtp_matrix": {
            "artifact": mtp_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "vl_media_matrix": {
            "artifact": vl_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS},
            "missing_engine_markers": [],
            "missing_panel_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "mcp_policy_matrix": {
            "artifact": mcp_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "max_output_context_matrix": {
            "artifact": max_output_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS},
            "missing_markers": [],
            "failed": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "jang_model_compat_matrix": {
            "artifact": jang_model_compat_artifact,
            "status": "pass",
            "failed": [],
            "missing": [],
        },
        "packaged_integrity_matrix": {
            "artifact": packaged_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS},
            "failed": [],
            "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "package_signing_preflight": {},
            "release_blockers": [],
            "unexpected_open_requirements": [],
            "missing_expected_open_requirements": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "release_surface_matrix": {
            "artifact": release_surface_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS},
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "live_smoke_summaries": _expected_passing_covered_live_smoke_summaries(),
        "live_tool_smoke_summaries": (
            _expected_passing_covered_live_tool_smoke_summaries()
        ),
        "diagnostic_live_smoke_summaries": (
            _expected_passing_diagnostic_live_smoke_summaries()
        ),
        "dev_ui_proof": {
            "status": "pass",
            "artifacts": dict(CURRENT_DEV_UI_PROOF_ARTIFACTS),
            "missing": [],
            "failures": [],
        },
        "real_ui_live_model_proof": expected_real_ui_live_model_proof,
        "issue175_179_release_boundary_audit": expected_issue175_179_release_boundary_audit,
        "issue179_minimax_k_root_cause_audit": expected_issue179_minimax_k_root_cause_audit,
        "mimo_v2_jang2l_sink_ab": expected_mimo_v2_jang2l_sink_ab,
        "mimo_v2_jang2l_root_cause": expected_mimo_v2_jang2l_root_cause,
        "installed_app_runtime_parity_audit": expected_installed_app_runtime_parity_audit,
        "staged_app_runtime_parity_audit": expected_staged_app_runtime_parity_audit,
        "issue175_177_installed_runtime_audit": expected_issue175_177_installed_runtime_audit,
        "issue175_177_live_runtime_audit": expected_issue175_177_live_runtime_audit,
        "real_ui_live_model_matrix": expected_real_ui_live_model_matrix,
        "real_ui_dsv4_memory_preflight": expected_real_ui_dsv4_memory_preflight,
        "step37_vlm_runtime_audit": expected_step37_vlm_runtime_audit,
        "release_blocker_ledger": expected_release_blocker_ledger,
    }


def test_release_regression_manifest_runner_can_require_current_proof_sweep(tmp_path):
    artifact = build_manifest_artifact(tmp_path, require_current_proof_sweep=True)

    assert artifact["status"] == "fail"
    assert artifact["current_proof_sweep"]["missing"]


def test_release_regression_manifest_runner_fails_when_current_proof_sweep_fails_by_default(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "fail",
            "regression_suite": {
                "status": "open",
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "mimo_root_cause_local_proof_missing",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["current_proof_sweep"]["status"] == "fail"
    assert artifact["release_ready"] is False


def test_release_regression_manifest_artifact_exposes_open_release_clearance(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "pass",
            "regression_suite": {
                "status": "pass",
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "dsv4_long_output_code_exactness_open",
                        "status": "open",
                    },
                    {
                        "id": "real_ui_dsv4_memory_blocked",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path)

    assert artifact["status"] == "pass"
    assert artifact["release_ready"] is False
    assert artifact["current_proof_sweep"]["status"] == "pass"
    assert artifact["release_clearance"]["status"] == "open"
    assert artifact["release_clearance"]["proof_sweep_status"] == "pass"
    assert artifact["release_clearance"]["open_requirements"] == (
        EXPECTED_CURRENT_OPEN_REQUIREMENTS
    )
    blocker_ids = [
        blocker["id"] for blocker in artifact["release_clearance"]["blockers"]
    ]
    assert "dsv4_long_output_code_exactness_open" in blocker_ids
    assert "real_ui_dsv4_memory_blocked" in blocker_ids
    assert artifact["release_clearance"]["release_ready"] is False


def test_release_regression_manifest_artifact_exposes_top_level_release_blockers(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "fail",
            "regression_suite": {
                "status": "open",
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "mimo_root_cause_local_proof_missing",
                        "status": "open",
                    },
                    {
                        "id": "real_ui_dsv4_memory_blocked",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["release_ready"] is False
    assert [
        blocker["id"] for blocker in artifact["release_blockers"]
    ] == [
        "mimo_root_cause_local_proof_missing",
        "real_ui_dsv4_memory_blocked",
    ]
    assert artifact["release_blockers"] == artifact["release_clearance"]["blockers"]


def test_release_regression_manifest_runner_can_require_release_ready(monkeypatch, tmp_path):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "pass",
            "regression_suite": {
                "status": "pass",
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "dsv4_long_output_code_exactness_open",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path, require_release_ready=True)

    assert artifact["status"] == "fail"
    assert artifact["release_ready"] is False
    assert artifact["release_clearance"]["status"] == "open"


def test_release_regression_manifest_runner_prepackage_allows_only_packaging_blockers(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "pass",
            "regression_suite": {
                "status": "pass",
                "open_requirements": [],
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "packaged_app_developer_id_signing_blocked",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(
        tmp_path,
        require_prepackage_ready=True,
    )

    assert artifact["status"] == "pass"
    assert artifact["prepackage_ready"] is True
    assert artifact["release_ready"] is False


def test_release_regression_manifest_runner_prepackage_allows_real_signing_only_sweep_failure(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "fail",
            "failed_components": ["no_release_blockers"],
            "component_ok": {
                "no_release_blockers": False,
                "regression_suite": True,
                "packaged_integrity_matrix": True,
            },
            "regression_suite": {
                "status": "pass",
                "open_requirements": [],
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "packaged_app_developer_id_signing_blocked",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(
        tmp_path,
        require_prepackage_ready=True,
    )

    assert artifact["status"] == "pass"
    assert artifact["prepackage_ready"] is True
    assert artifact["release_ready"] is False
    assert artifact["prepackage_clearance"]["proof_sweep_failed_components"] == [
        "no_release_blockers"
    ]


def test_release_regression_manifest_runner_prepackage_rejects_model_blockers(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "pass",
            "regression_suite": {
                "status": "pass",
                "open_requirements": [],
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "dsv4_long_output_code_exactness_open",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(
        tmp_path,
        require_prepackage_ready=True,
    )

    assert artifact["status"] == "fail"
    assert artifact["prepackage_ready"] is False
    assert artifact["release_ready"] is False


def test_release_regression_manifest_tracks_legacy_completions_output_boundary():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "Legacy /v1/completions max_tokens" in joined
    assert "per-request output cap" in joined
    assert "non-streaming and streaming" in joined
    assert "current-max-output-context-contract-20260522-chat-auto-server-default.json" in joined


def test_release_regression_manifest_tracks_generation_defaults_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["generation-defaults-no-hidden-forcing"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_generation_defaults_contract.py" in joined
    assert "current-generation-defaults-contract-20260531-post-step-lfm-refresh.json" in joined
    assert "current-generation-defaults-contract-20260526-settings-audit.json" not in joined
    assert "current-generation-defaults-contract-20260521.json" in joined
    assert "generation_config.json" in joined
    assert "jang_config.json" in joined
    assert "hidden sampler forcing" in joined
    assert "Additional Args cannot override app-owned reasoning, parser, cache, MTP" in joined


def test_release_regression_manifest_tracks_reasoning_template_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["reasoning-template-no-think-tag-leak"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_reasoning_template_contract.py" in joined
    assert "current-reasoning-template-contract-20260526-settings-audit.json" in joined
    assert "current-reasoning-template-contract-20260521.json" in joined
    assert "Reasoning on/off" in joined
    assert "think tags" in joined
    assert "Interleaved reasoning" in joined


def test_release_regression_manifest_tracks_api_surface_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["api-chat-responses-anthropic-ollama-parity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_api_surface_contract.py" in joined
    assert "current-api-surface-contract-20260531-nested-epipe-childstream-refresh.json" in joined
    assert "current-api-surface-contract-20260529-single-model-transition-lock.json" in joined
    assert "current-api-surface-contract-20260528-ollama-embeddings-single-model.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-nested-disconnect.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-request-end-guard.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-socket-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-proxy-error-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-epipe-multiline-guard.json" not in joined
    assert "current-api-surface-contract-20260527-cache-endpoint-autoswitch-proof.json" not in joined
    assert "current-api-surface-contract-20260526-single-model-auto-switch-review.json" not in joined
    assert "current-api-surface-contract-20260525-single-model-responses-deltas.json" not in joined
    assert "current-api-surface-contract-20260525-single-model-ollama-chat-deltas.json" in joined
    assert "current-api-surface-contract-20260522-stream-cache-detail.json" in joined
    assert "destroyed client/backend socket" in joined
    assert "top-level request handler disconnect failures" in joined
    assert "EPIPE/ECONNRESET" in joined
    assert "Ollama-specific backend response streams guard response errors" in joined
    assert "OpenAI Chat Completions" in joined
    assert "OpenAI Responses" in joined
    assert "non-streaming and streaming" in joined
    assert "OpenAI legacy Completions" in joined
    assert "Anthropic" in joined
    assert "streaming max_tokens override" in joined
    assert "Ollama" in joined
    assert "Auto chat Max Tokens omits per-request output caps" in joined
    assert "streaming num_predict" in joined
    assert "malformed num_predict" in joined
    assert "malformed context" in joined
    assert "cache_detail" in joined


def test_release_regression_manifest_tracks_current_defaults_reasoning_api_rechecks():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    generation = rows["generation-defaults-no-hidden-forcing"]
    generation_joined = " ".join(generation["commands"] + generation["artifacts"] + generation["proves"])
    assert "current-generation-defaults-contract-20260531-post-step-lfm-refresh.json" in generation_joined
    assert "current-generation-defaults-contract-20260526-settings-audit.json" not in generation_joined
    assert "current-generation-defaults-contract-20260522-recheck-no-hidden-forcing.json" in generation_joined
    assert "bundle max_new_tokens" in generation_joined
    assert "without hidden sampler or repetition floors" in generation_joined
    assert "server default output cap is not a request ceiling" in generation_joined

    reasoning = rows["reasoning-template-no-think-tag-leak"]
    reasoning_joined = " ".join(reasoning["commands"] + reasoning["artifacts"] + reasoning["proves"])
    assert "current-reasoning-template-contract-20260526-settings-audit.json" in reasoning_joined
    assert "DSV4 requested reasoning-on/effort rails are preserved" in reasoning_joined
    assert "MiniMax and Ling family-specific reasoning boundaries" in reasoning_joined
    assert "tool follow-up reasoning state resets" in reasoning_joined

    api = rows["api-chat-responses-anthropic-ollama-parity"]
    api_joined = " ".join(api["commands"] + api["artifacts"] + api["proves"])
    assert "current-api-surface-contract-20260522-recheck-endpoint-assembly.json" in api_joined
    assert "server cache and DSV4 tool/parser surfaces stay named" in api_joined
    assert "panel request builders omit invalid persisted maxTokens" in api_joined


def test_release_regression_manifest_tracks_tool_calls_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["tool-call-loop-parser-cleanup"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_tool_call_contract.py" in joined
    assert "current-tool-call-contract-20260528-tool-parser-loop-matrix.json" in joined
    assert "Tool parser residue" in joined
    assert "DSV4" in joined
    assert "maxToolIterations" in joined
    assert "live DSV4 write_file DSML degradation is repaired schema-safely" in joined
    assert "current-tool-call-contract-20260522-dsv4-live-write-file-repair.json" in joined
    assert "current-dsv4-default-cache-tool-loop-thinking-on-20260525.json" in joined
    assert "explicit-thinking DSV4 default-cache tool-loop diagnostic remains review" in joined
    assert "current-dsv4-default-cache-tool-loop-poolon-materialized-parserfix-20260522/result.json" in joined
    assert "default-cache live loop reaches three tools but still fails exact WebGLRenderer fidelity" in joined


def test_release_regression_manifest_tracks_panel_cache_family_gating():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["panel-session-cache-settings-family-gating"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "cache_architecture"
    assert "run_noheavy_panel_settings_contract.py" in joined
    assert "current-panel-settings-contract-proof-20260528-cache-ui-matrix.json" in joined
    assert "current-panel-settings-contract-proof-20260524-text-additional-args-sanitizer.json" in joined
    assert "current-panel-settings-contract-proof-20260522-launch-memory-warning.json" in joined
    assert "DSV4 pool quant" in joined
    assert "JANG/JANGTQ/MXFP" in joined
    assert "stale Additional Args cannot override app-owned server, template, model-name, MCP, max output/context" in joined
    assert "warning-only for lazy-mmap" in joined


def test_release_regression_manifest_tracks_mcp_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["mcp-policy-ui-gateway"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_mcp_policy_contract.py" in joined
    assert "current-mcp-policy-contract-20260531-post-step-lfm-refresh.json" in joined
    assert "MCP autodiscovery" in joined
    assert "redaction" in joined
    assert "gateway routing" in joined
    assert "panel config source" in joined
    assert "required marker" in joined


def test_release_regression_manifest_tracks_current_mcp_and_vl_media_rechecks():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    mcp = rows["mcp-policy-ui-gateway"]
    mcp_joined = " ".join(mcp["commands"] + mcp["artifacts"] + mcp["proves"])
    assert "current-mcp-policy-contract-20260522-recheck-ui-gateway.json" in mcp_joined
    assert "mcp.json/jsonc import" in mcp_joined
    assert "built-in Electron tools remain separate from MCP execution" in mcp_joined
    assert "ambiguous multi-session MCP requests" in mcp_joined

    vl_media = rows["vl-media-cache-tool-followup"]
    vl_joined = " ".join(vl_media["commands"] + vl_media["artifacts"] + vl_media["proves"])
    assert "current-vl-media-cache-contract-20260522-recheck-qwen-zaya-media.json" in vl_joined
    assert "Qwen3.6 VL JANG indexed-MTP" in vl_joined
    assert "MXTQ/JANGTQ and MXFP4/MXFP8 Qwen VLM" in vl_joined
    assert "hybrid SSM companion cache" in vl_joined
    assert "read_video" in vl_joined
    assert "image/video tool-result follow-up" in vl_joined


def test_release_regression_manifest_tracks_qwen_jang_live_speed_review():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    row = rows["qwen-jang-mx-live-speed-review"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "performance"
    assert row["mode"] == "live"
    assert "current-decode-speed-live-qwen27-jang4m-20260522-hybrid-tq-review.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-source-keepalloc-20260522.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-packaged-keepalloc-20260522.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-packaged-tahoe-dmg-20260522.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-text-baseline-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-source-bypass-fix-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace3-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-default-after-norm-shift-20260523.json" in joined
    assert "MLX and MLX-metal wheel tags" in joined
    assert "selective live TurboQuant for Qwen attention KV layers" in joined
    assert "Post norm-format loader fix live Qwen MTP/VLM default-cache row clears coherency" in joined
    assert "SingleBatchGenerator honors the explicit prefill keep-alloc CLI/env path" in joined
    assert "Compat bundled MTP prefill is retained as the user-facing packaged regression diagnostic" in joined
    assert "native-MTP MLLM/VL prefill remains below the floor" in joined
    assert "prompt-processing floor compares isolated native-MTP/VL prefill against the isolated text-loader baseline" in joined


def test_release_regression_manifest_tracks_packaged_integrity_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["packaged-release-integrity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_packaged_integrity_contract.py" in joined
    assert "current-packaged-integrity-contract-20260531-childstream-epipe-refresh.json" in joined
    assert "current-packaged-integrity-contract-20260530-bundled-sync-after-step37-projector.json" not in joined
    assert "current-packaged-integrity-contract-20260521.json" not in joined
    assert "Version triples" in joined
    assert "bundled Python hash parity" in joined
    assert "objective proof digest" in joined
    assert "current-objective-proof-audit-20260531-nemotron-exact-finalizer-ledger.json" in joined
    assert "objective-gate-enforced" in joined
    assert "verify-bundled" in joined


def test_release_regression_manifest_tracks_current_packaged_integrity_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["packaged-release-integrity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-packaged-integrity-contract-20260522-recheck-bundled-release-gate.json" in joined
    assert "current-packaged-integrity-contract-20260524-text-additional-args-sanitizer.json" in joined
    assert "clean JANG source path" in joined
    assert "bundled critical jang_tools files match source content" in joined
    assert "console-script shebangs are relocatable" in joined
    assert "fails only on the known objective rows" in joined


def test_release_regression_manifest_tracks_public_release_surface_preflight():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["public-release-surface-preflight"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "release_surface"
    assert "run_release_surface_contract.py" in joined
    assert (
        CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
        == "build/current-release-surface-contract-20260528-release-surface-matrix.json"
    )
    assert "current-release-surface-contract-20260528-release-surface-matrix.json" in joined
    assert "latest.json" in joined
    assert "PyPI" in joined
    assert "GitHub release" in joined
    assert "updater" in joined.lower()
    assert "post-release-updater" in joined
    assert "published updater state is complete" in joined
    assert "--live-public" in joined
    assert "mlx.studio/update/latest.json" in joined
    assert "PyPI files" in joined
    assert "no-store/no-cache headers" in joined
    assert "source repo release tag" in joined
    assert "current-release-surface-contract-20260524-live-source-tag-parity.json" in joined


def test_release_regression_manifest_tracks_current_updater_and_i18n_rechecks():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    panel = rows["panel-session-cache-settings-family-gating"]
    panel_joined = " ".join(panel["commands"] + panel["artifacts"] + panel["proves"])
    assert "current-panel-settings-contract-proof-20260522-recheck-update-notice-i18n.json" in panel_joined
    assert "What's New update notice" in panel_joined
    assert "MCP, MTP, latest.json, Developer ID, L2 disk cache, and max_tokens" in panel_joined
    assert "all five locales" in panel_joined

    release = rows["public-release-surface-preflight"]
    release_joined = " ".join(release["commands"] + release["artifacts"] + release["proves"])
    assert "current-release-surface-contract-20260522-recheck-updater-i18n.json" in release_joined
    assert "source version consistent across pyproject.toml and panel/package.json" in release_joined
    assert "complete post-release updater state" in release_joined
    assert "incomplete bumped latest.json" in release_joined

    ling = rows["ling-bailing-multilingual-quality-live"]
    ling_joined = " ".join(ling["commands"] + ling["artifacts"] + ling["proves"])
    assert "current-objective-proof-audit-20260531-nemotron-exact-finalizer-ledger.json" in ling_joined


def test_release_regression_manifest_tracks_live_only_boundaries():
    manifest = build_manifest()
    live_rows = [row for row in manifest["rows"] if row["mode"] == "live"]
    live_ids = {row["id"] for row in live_rows}

    assert "dsv4-long-output-quality-live" in live_ids
    assert "model-family-live-multiturn-soak" in live_ids
    assert all(row["heavy"] for row in live_rows)


def test_release_regression_manifest_tracks_fresh_dsv4_live_failure_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["dsv4-long-output-quality-live"]
    joined = " ".join(row["artifacts"] + row["proves"])

    assert "current-dsv4-jangtq-k-route-mode-code-exactness-20260524.json" in joined
    assert "current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json" in joined
    assert "current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json" in joined
    assert "current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json" in joined
    assert "current-dsv4-route-mode-code-exactness-current-rep1-controls-20260524-2104.json" in joined
    assert "current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json" in joined
    assert "current-dsv4-route-mode-code-exactness-existing-server-chatmax-20260524.json" in joined
    assert "current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json" in joined
    assert "current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json" in joined
    assert "current-dsv4-prompt-boundary-bisection-live-20260524-1317.json" in joined
    assert "current-dsv4-colon-vs-period-logprob-trace-live-20260524-1320.json" in joined
    assert "current-dsv4-colon-vs-period-visible-logprob-trace-live-20260524-1322.json" in joined
    assert "current-dsv4-scene-token-rank-contrast-live-20260524-1324.json" in joined
    assert "current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json" in joined
    assert "current-dsv4-hidden-reasoning-control-live-20260524-1335.json" in joined
    assert "current-dsv4-template-parity-diagnostic-20260524-1343.json" in joined
    assert "current-dsv4-jang-prefill-execution-variant-logits-20260524.json" in joined
    assert "current-dsv4-jang-prompt-variant-logit-probe-20260524.json" in joined
    assert "current-dsv4-reasoning-policy-live-20260524-1408.json" in joined
    assert "current-dsv4-jang-cache-vs-full-logit-isolation-threejs-20260524.json" in joined
    assert "current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json" in joined
    assert "current-dsv4-jang-thinking-off-logit-probe-20260524.json" in joined
    assert "current-dsv4-jang-live-api-copy-framing-canary-20260524.json" in joined
    assert "current-dsv4-route-mode-code-exactness-live-memory-preflight-20260531.json" in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-post-install-sync.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-continue-refresh.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-0625.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-0600.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260526.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260525.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json" in joined
    assert "current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json" in joined
    assert "current-dsv4-route-mode-code-exactness-dryrun-20260528-identifier-candidates.json" in joined
    assert "current-dsv4-route-mode-code-exactness-dryrun-20260528-current-cohesive-audit.json" in joined
    assert "current-dsv4-route-mode-code-exactness-direct-off-no-punct-dryrun-20260525.json" in joined
    assert "current-dsv4-route-mode-code-exactness-direct-off-no-punct-preflight-20260525.json" in joined
    assert "identifier integrity" in joined
    assert "explicit thinking-off still fails exact code" in joined
    assert "repetition_penalty=1.0 live route controls still fail direct/off" in joined
    assert "true-bundled JANGTQ-K direct/off recheck still fails" in joined
    assert "chat_max remains open" in joined
    assert "budget, stops, requested thinking, Responses, and completion route" in joined
    assert "explicit DSV4 thinking-off produces visible output" in joined
    assert "prompt-boundary bisection" in joined
    assert "colon-vs-period logprob trace" in joined
    assert "scene token rank contrast" in joined
    assert "hidden reasoning" in joined
    assert "template parity" in joined
    assert "prefill execution variants" in joined
    assert "prompt wording changes identifier logits" in joined
    assert "batch generator logit divergence" in joined
    assert "source full-output exactness is still missing because local memory preflight skips" in joined
    assert "no-punctuation direct/off variant is queued under thinking_closed" in joined


def test_release_regression_manifest_tracks_ling_multilingual_quality_clearance_boundary():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    row = rows["ling-bailing-multilingual-quality-live"]
    joined = " ".join(row["proves"] + row["commands"] + row["artifacts"])

    assert row["domain"] == "reasoning_template"
    assert row["mode"] == "live"
    assert row["heavy"] is True
    assert "Ling/Bailing multilingual output quality is release-cleared" in joined
    assert "current-ling-jangtq-strict-russian-nocache-bundled-4850c9c2-20260524.json" in joined
    assert "current-ling-mxfp4-crack-strict-russian-nocache-bundled-4850c9c2-20260524.json" in joined
    assert "current-ling-jangtq-russian-prompt-variant-probe-20260524.json" in joined
    assert "current-ling-jangtq-server-repeat-russian-source-prefill-stream-20260524.json" in joined
    assert "current-ling-jangtq-server-repeat-russian-bundled-prefill-stream-20260524.json" in joined
    assert "current-ling-jangtq-server-repeat-russian-bundled-native-prefill-stream-20260524.json" in joined
    assert "current-production-family-live-ling-bundled-native-rerun-20260524.json" in joined
    assert "current-ling-jangtq-cold-skipcache-repeat-bundled-native-20260524.json" in joined
    assert "current-ling-jangtq-batchgen-temp0-repeat-bundled-native-20260524.json" in joined
    assert "current-ling-jangtq-simple-engine-control-bundled-after-mpp-fix-20260524.json" in joined
    assert "current-ling-jangtq-continuous-control-bundled-after-mpp-fix-20260524.json" in joined
    assert "current-production-family-live-ling-bundled-after-mpp-fix-20260524.json" in joined
    assert "current-production-family-live-ling-bundled-after-topk-policy-20260524.json" in joined
    assert "clearance artifacts are zero-CJK" in joined
    assert "Older failing artifacts remain tracked as regression evidence" in joined
    assert "current open row" not in joined
    assert "CJK leakage remains release-visible" not in joined


def test_release_regression_manifest_tracks_gemma4_crack_language_visible_quality_boundary():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    row = rows["gemma4-crack-live-language-visible-quality"]
    joined = " ".join(row["proves"] + row["commands"] + row["artifacts"])

    assert row["domain"] == "reasoning_template"
    assert row["mode"] == "live"
    assert row["heavy"] is True
    assert "Gemma-4-26B-A4B JANG_4M CRACK" in joined
    assert "switch_mlp experts" in joined
    assert "dense 31B JANG_4M MTP comparison row" in joined
    assert "Bundled Python must import jang_tools" in joined
    assert "visible English output with preserved identifiers" in joined
    assert "max_thinking_tokens=16 artifact remains diagnostic" in joined
    assert "CJK/Korean/Chinese leakage" in joined
    assert "affine/JANG matmul dispatch" in joined
    assert "Gemma4 switch_mlp remapping" in joined
    assert "mixed_swa_kv_v1" in joined
    assert "80 tok/s decode floor" in joined
    assert "generic live TurboQuant KV" in joined
    assert "generic TurboQuant KV off" in joined
    assert "source now proves native mixed-SWA cache telemetry as paged+mixed_swa" in joined
    assert "clears the sustained 512-token speed-floor prompt" in joined
    assert "Current installed app proves source-hash parity and paged+mixed_swa cache telemetry" in joined
    assert "installed app speed remains a release risk" in joined
    assert "wall decode samples are below 80 tok/s" in joined
    assert "decode_tok_s_stream" in joined
    assert "installed repeat still remains below or unstable" not in joined
    assert "compat MLX wheels remain a separate installed-app speed risk" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-contract-20260524.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-source-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-streaming-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-native-schema-v5-20260525.json" not in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json" not in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-speedram-945c84ba-20260524b.json" not in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-text-installed149.json" not in joined
    assert "current-production-family-live-gemma4-crack-parser-tool-config-latest-jang-20260524.json" in joined


def test_release_regression_manifest_does_not_overclaim_gemma4_installed_speed_floor():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    row = rows["gemma4-crack-live-language-visible-quality"]
    joined = " ".join(row["proves"] + row["commands"] + row["artifacts"])
    installed_speed_artifacts = [
        artifact
        for artifact in row["artifacts"]
        if "installed-app" in artifact and "speed-floor" in artifact
    ]

    below_floor: list[float] = []
    for artifact in installed_speed_artifacts:
        payload = json.loads(Path(artifact).read_text())
        for result in payload.get("results", []):
            speed = result.get("speed", {})
            decode_tok_s_wall = speed.get("decode_tok_s_wall")
            if decode_tok_s_wall is not None and decode_tok_s_wall < 80.0:
                below_floor.append(decode_tok_s_wall)

    assert below_floor
    assert "installed app speed remains a release risk" in joined
    assert "installed-app chat streaming proof clears the 80 tok/s decode floor" not in joined
    assert "every required chat/Responses/cold/cache-hit sample meets the 80 tok/s decode floor" not in joined


def test_release_regression_manifest_live_soak_does_not_overclaim_qwen_mtp():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-live-multiturn-soak"]
    command_text = " ".join(row["commands"]).lower()
    proves_text = " ".join(row["proves"]).lower()

    if "qwen mtp" in proves_text and "no-heavy" not in proves_text:
        assert "qwen" in command_text and "mtp" in command_text


def test_release_regression_manifest_live_soak_targets_present_qwen_hybrid_row():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-live-multiturn-soak"]
    command_text = " ".join(row["commands"])

    assert "qwen36_moe_crack" in command_text
    assert "qwen36_moe_tq4" not in command_text


def test_release_regression_manifest_tracks_family_parser_cli_choice_guard():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "CLI-accepted parser choice" in joined
    assert "current-model-family-detection-contract-20260522-plain-kv-cache-health.json" in joined


def test_release_regression_manifest_tracks_decode_speed_artifact_format_matrix():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "DSV4 native composite" in joined
    assert "generic JANGTQ/MXTQ" in joined
    assert "current-model-family-detection-contract-20260522-plain-kv-cache-health.json" in joined
    assert "row parser/modality policy" in joined
    assert "forced JANGTQ acceleration" in joined
    assert "JANG-only MX matmul rows stay text-only launch rows" in joined
    assert "current-model-family-detection-contract-20260522-jang-only-mx-matmul-policy.json" in joined
    assert "Mistral JANGTQ" in joined
    assert "Mistral MXFP4" in joined
    assert "GPT-OSS" in joined
    assert "Nemotron 3 JANGTQ2" in joined


def test_release_regression_manifest_tracks_qwen_nemotron_hybrid_cache_rows():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "Qwen dense linear-attention wrappers stay hybrid cache" in joined
    assert "Qwen MoE text linear-attention wrappers stay hybrid cache" in joined
    assert "base Nemotron-H registry rows stay hybrid cache" in joined
    assert "current-model-family-detection-contract-20260522-qwen-nemotron-hybrid-cache.json" in joined
    assert "Nemotron parser/reasoning" in joined
    assert "MXFP4" in joined
    assert "plain KV JANG/JANGTQ/MXFP rows" in joined
    assert "local high-risk DSV4" in joined
    assert "Nemotron Omni/Nano" in joined
    assert "current-model-family-detection-contract-20260522-local-artifact-registry.json" in joined
    assert "Panel detection matches the same current local high-risk paths" in joined
    assert "native-MTP Qwen affine-JANG VL routing" in joined
    assert "current-model-family-detection-contract-20260522-panel-local-paths.json" in joined
    assert "Qwen 3.6 release rows intentionally keep qwen3_5/qwen3.5 family aliases" in joined
    assert "current-model-family-detection-contract-20260522-qwen36-alias.json" in joined


def test_release_regression_manifest_tracks_zaya_stale_stamp_policy():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "stale ZAYA converter stamps" in joined
    assert "cannot disable reasoning" in joined
    assert "current-model-family-detection-contract-20260522-zaya-stale-stamp.json" in joined


def test_release_regression_manifest_tracks_panel_family_launch_wiring():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-model-family-detection-contract-20260522-panel-launch-wiring.json" in joined
    assert "current-model-family-detection-contract-20260522-zaya-hy3-qwen-vl-profile-rows.json" in joined
    assert "Panel session launch builder" in joined
    assert "MiniMax minimax_m2" in joined
    assert "Qwen3.6 hybrid cache forces paged cache" in joined
    assert "ZAYA qwen3 reasoning parser" in joined
    assert "DSV4 stale cache/additionalArgs suppression" in joined
    assert "native-MTP D3 launch policy" in joined
    assert "ZAYA1-VL JANGTQ_K/JANGTQ2/JANGTQ4 qwen3 reasoning rails" in joined
    assert "Hy3 JANGTQ_K Low/High reasoning contract" in joined
    assert "affine-JANG Qwen native-MTP VL/video" in joined


def test_release_regression_manifest_tracks_mxfp_vlm_loader_quant_mode():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "MXFP4/MXFP8 VLM loader" in joined
    assert "declared quantization mode" in joined
    assert "family-specific loader wrappers" in joined
    assert "current-model-artifact-format-contract-20260522-mxfp-vlm-loader.json" in joined
    assert "current-model-artifact-format-contract-20260522-loader-wrapper-hashes.json" in joined


def test_release_regression_manifest_tracks_affine_jang_loader_acceptance():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "generic affine JANG loader" in joined
    assert "weight_format=affine" in joined
    assert "current-model-artifact-format-contract-20260522-affine-jang-loader.json" in joined


def test_release_regression_manifest_commands_are_declared_for_noheavy_rows():
    manifest = build_manifest()
    noheavy_rows = [row for row in manifest["rows"] if row["mode"] == "noheavy"]

    assert noheavy_rows
    for row in noheavy_rows:
        assert row["commands"], row["id"]


def test_release_regression_manifest_runner_commands_exist():
    manifest = build_manifest()

    for row in manifest["rows"]:
        for command in row["commands"]:
            if "tests/cross_matrix/" not in command:
                continue
            runner = command.split("tests/cross_matrix/", 1)[1].split()[0]
            path = Path("tests/cross_matrix") / runner
            assert path.exists(), f"{row['id']} references missing runner {path}"


def test_release_regression_manifest_python_entrypoints_are_tracked():
    manifest = build_manifest()

    for row in manifest["rows"]:
        for command in row["commands"]:
            for token in shlex.split(command):
                if not token.endswith(".py"):
                    continue
                path = Path(token)
                if path.is_absolute():
                    continue
                assert path.exists(), f"{row['id']} references missing Python entrypoint {path}"


def test_release_regression_manifest_python_runner_commands_are_executable_invocations():
    manifest = build_manifest()

    for row in manifest["rows"]:
        for command in row["commands"]:
            parts = shlex.split(command)
            for index, token in enumerate(parts):
                if not token.startswith("tests/cross_matrix/") or not token.endswith(".py"):
                    continue
                launcher = parts[:index]
                assert launcher, f"{row['id']} references {token} without a Python launcher"
                assert (
                    "python" in launcher
                    or "python3" in launcher
                    or any(part.endswith("/python") or part.endswith("/python3") for part in launcher)
                ), f"{row['id']} references {token} without an executable Python command"


def test_release_regression_manifest_tracks_model_artifact_detection_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_model_artifact_format_contract.py" in joined
    assert "current-model-artifact-format-contract-20260522-mxfp-vlm-loader.json" in joined
    assert "JANGTQ" in joined
    assert "MXFP4" in joined
    assert "MXFP8" in joined
    assert "MTP" in joined


def test_release_regression_manifest_tracks_current_jang_mxfp_mtp_loader_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-model-artifact-format-contract-20260522-recheck-jang-mxfp-mtp-loaders.json" in joined
    assert "JANGTQ_K mixed routed bits use the down-projection bit width for gather_dn" in joined
    assert "Ling/Bailing flat 2D switch_mlp repair and correct 3D no-op stay covered" in joined
    assert "Qwen plain MLX 4bit stays distinct from JANG, JANGTQ/MXTQ, MXFP4, and MXFP8" in joined
    assert "native-MTP detection uses real indexed weights rather than path names" in joined


def test_release_regression_manifest_tracks_named_model_family_detection_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "model_family_detection"
    assert "run_model_family_detection_contract.py" in joined
    assert "current-model-family-detection-contract-20260522-plain-kv-cache-health.json" in joined
    assert "DSV4" in joined
    assert "ZAYA" in joined
    assert "Ling" in joined
    assert "Nemotron" in joined
    assert "Qwen 3.6" in joined
    assert "MXFP4" in joined
    assert "MXFP8" in joined
    assert "JANG-only" in joined
    assert "Mistral JANGTQ" in joined
    assert "Mistral MXFP4" in joined
    assert "Nemotron 3 JANGTQ2" in joined
    assert "GPT-OSS" in joined
    assert "JANGTQ/MXTQ" in joined
    assert "registered engine parser" in joined
    assert "CLI-accepted parser choice" in joined
    assert "MiniMax" in joined
    assert "Hy3" in joined


def test_release_regression_manifest_tracks_current_family_launch_policy_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-model-family-detection-contract-20260522-recheck-launch-policy-matrix.json" in joined
    assert "engine, panel, and session launch wiring all run in the same family launch-policy matrix" in joined
    assert "ZAYA, ZAYA1-VL, Ling/Bailing, Nemotron-H, Qwen 3.6, MiniMax, Hy3, and DSV4 rows keep aligned parser/cache/modality policy" in joined
    assert "session launch strips stale DSV4 additionalArgs and preserves native-MTP D3 only where supported" in joined


def test_release_regression_manifest_tracks_parser_parity_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["parser-registry-tool-reasoning-parity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_parser_registry_contract.py" in joined
    assert "current-parser-registry-contract-20260522-reasoning-dropdown.json" in joined
    assert "current-parser-registry-contract-20260522-non-reasoning-boundaries.json" in joined
    assert "MiniMax" in joined
    assert "reasoning parser" in joined
    assert "Reasoning parser dropdown covers every parser" in joined
    assert "CLI accepts every registered reasoning parser" in joined
    assert "tool parser" in joined
    assert "Qwen2/Qwen2-VL, Gemma 3, and GLM base stay off reasoning rails" in joined


def test_release_regression_manifest_tracks_max_output_context_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_max_output_context_contract.py" in joined
    assert "current-max-output-context-contract-20260522-chat-auto-server-default.json" in joined
    assert "Server Default Max Output Tokens" in joined
    assert "Chat Max Output Tokens" in joined
    assert "non-streaming and streaming" in joined
    assert "Legacy /v1/completions max_tokens" in joined
    assert "non-streaming and streaming" in joined
    assert "Prompt/context aliases clamp" in joined
    assert "Ollama gateway non-stream num_predict remains request-scoped" in joined
    assert "current-max-output-context-contract-20260523-ollama-nonstream-caps.json" in joined
    assert "malformed num_predict" in joined
    assert "malformed context" in joined
    assert "Max Context Tokens" in joined
    assert "--max-tokens" in joined
    assert "--max-prompt-tokens" in joined
    assert "coding-tool configs" in joined
    assert "Auto chat Max Tokens omits per-request output caps" in joined


def test_release_regression_manifest_tracks_current_server_chat_output_compat_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-max-output-context-contract-20260522-recheck-server-chat-output-compat.json" in joined
    assert "explicit per-request Chat/Responses output caps can be below or above Server Default Max Output Tokens" in joined
    assert "later Auto Chat/Responses requests still use the original server startup default" in joined
    assert "server startup maxTokens explicitness survives wake/reload and CLI implicit startup paths" in joined


def test_release_regression_manifest_tracks_vl_media_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["vl-media-cache-tool-followup"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_vl_media_cache_contract.py" in joined
    assert "current-vl-media-cache-contract-20260522-panel-family.json" in joined
    assert "VLM media request serialization" in joined
    assert "media cache salting" in joined
    assert "tool follow-up" in joined
    assert "Qwen VL/video/hybrid" in joined
    assert "affine-JANG Qwen" in joined
    assert "Nemotron stale-Omni" in joined
    assert "video" in joined.lower()


def test_release_regression_manifest_tracks_cache_architecture_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["cache-architecture-family-classification"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_cache_architecture_contract.py" in joined
    assert "current-cache-architecture-contract-20260530-lfm2-tool-parser-local.json" in joined
    assert "current-cache-architecture-contract-20260528-gemma4-mixed-swa-row.json" in joined
    assert "current-cache-architecture-contract-20260527-cache-family-matrix.json" in joined
    assert "current-cache-architecture-contract-20260524-openai-single-model-streaming-audit.json" in joined
    assert "current-cache-architecture-contract-20260521.json" in joined
    assert "DSV4 composite" in joined
    assert "DSV4 SWA/HCA/CSA native composite cache components" in joined
    assert "Ling/Bailing" in joined
    assert "Qwen3.6" in joined
    assert "ZAYA CCA" in joined
    assert "hybrid SSM" in joined
    assert "TurboQuant KV" in joined
    assert "MLA" in joined
    assert "current-cache-architecture-contract-20260522-panel-cache-launch.json" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-quant-append.json" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-timing.json" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-env-gate.json" in joined
    assert "DSV4 pool quant reads reuse a materialized pool view" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-materialized-cache.json" in joined
    assert "pool quant codec appends only newly generated CSA/HCA pool rows" in joined
    assert "DSV4 panel env mapping enables pool quant by default only under DSV4 native composite cache" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json" in joined
    assert "DSV4 timing probe covers prefix-cache replay and cold-store boundaries" in joined
    assert "Panel session launch builder preserves DSV4 default-on native prefix-cache policy" in joined
    assert "Qwen3.6 hybrid and Mamba paged-cache forcing" in joined
    assert "regular KV stale saved false semantics" in joined
    assert "Gemma4" in joined
    assert "mixed-SWA" in joined


def test_release_regression_manifest_tracks_native_mtp_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["native-mtp-d3-effect-policy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_native_mtp_contract.py" in joined
    assert "current-native-mtp-contract-20260522-config-only.json" in joined
    assert "current-native-mtp-contract-20260522-dsv4-additional-args.json" in joined
    assert "Native MTP D3" in joined
    assert "DSV4" in joined
    assert "stale additionalArgs" in joined
    assert "deterministic MTP sampling" in joined
    assert "Config-only MTP bundles" in joined
    assert "indexed mtp.* tensors" in joined
    assert "decode speed and output equivalence" in joined
    assert "prompt-processing floor" in joined
    assert "current-native-mtp-speed-ab-qwen27-jang4m-mtp-20260523/result.json" in joined
    assert "same-artifact AR-vs-MTP" in joined
    assert "MTP decode is not the MLLM prefill bottleneck" in joined
    assert "prefill" in joined
