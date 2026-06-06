#!/usr/bin/env python3
"""Summarize the current DSV4/cache/tool objective proof state.

This is a no-heavy helper: it reads existing JSON proof artifacts and produces
a requirement-by-requirement digest. It deliberately keeps broad DSV4
long-output/code-quality open unless a dedicated clearance artifact exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.cross_matrix.run_max_output_context_contract import (
    SOURCE_HASH_FILES as MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_cache_architecture_contract import (
    DEFAULT_OUT as CACHE_ARCHITECTURE_CONTRACT_DEFAULT_OUT,
    SOURCE_HASH_FILES as CACHE_ARCHITECTURE_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_tool_call_contract import (
    DEFAULT_OUT as TOOL_CALL_CONTRACT_DEFAULT_OUT,
    SOURCE_HASH_FILES as TOOL_CALL_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_model_artifact_format_contract import (
    SOURCE_HASH_FILES as MODEL_ARTIFACT_FORMAT_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_model_family_detection_contract import (
    REQUIRED_ROWS as MODEL_FAMILY_CONTRACT_CHECKS,
    SOURCE_HASH_FILES as MODEL_FAMILY_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_generation_defaults_contract import (
    REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX,
    SOURCE_HASH_FILES as GENERATION_DEFAULTS_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_native_mtp_contract import (
    SOURCE_HASH_FILES as NATIVE_MTP_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_parser_registry_contract import (
    SOURCE_HASH_FILES as PARSER_REGISTRY_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_vl_media_cache_contract import (
    SOURCE_HASH_FILES as VL_MEDIA_SOURCE_HASH_FILES,
)
from tests.cross_matrix.release_regression_manifest import (
    CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
    CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT,
    CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
    EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS,
    EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS,
)


DEFAULT_OUT = Path("build/current-objective-proof-audit-20260602-cache-detail-zero-cached.json")
CURRENT_RELEASE_REGRESSION_MANIFEST_REL = (
    "build/current-release-regression-manifest-after-mimo-sink-falsification-20260606.json"
)
DSV4_QUALITY_CLEARANCE_REL = "build/current-dsv4-long-output-quality-clearance-20260521.json"
DSV4_CURRENT_IDENTIFIER_CANARY_REL = (
    "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524.json"
)
DSV4_CURRENT_IDENTIFIER_CANARY_FALLBACK_STRICT_REL = (
    "build/current-dsv4-jangtq-k-identifier-canary-strict-nocache-bundled-b3345c29-rerun2-20260524.json"
)
DSV4_CURRENT_IDENTIFIER_CANARY_FALLBACK_REL = "build/current-dsv4-live-identifier-canary-20260523.json"
DSV4_CURRENT_IDENTIFIER_MATRIX_REL = "build/current-dsv4-live-identifier-matrix-20260523.json"
DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL = "build/current-dsv4-installed-tokenizer-roundtrip-20260523.json"
DSV4_LIVE_LOGPROBS_COPY_REL = "build/current-dsv4-live-logprobs-copy-20260523.json"
DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL = "build/current-dsv4-live-logprob-context-matrix-20260523.json"
DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL = (
    "build/current-dsv4-live-cache-context-identifier-probe-20260523.json"
)
DSV4_SOURCE_NOCACHE_IDENTIFIER_REL = (
    "build/current-dsv4-live-identifier-list-nocache-source-20260523.json"
)
DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL = (
    "build/current-dsv4-live-identifier-sameprompt-nocache-source-20260523.json"
)
DSV4_SOURCE_CACHE_COMPARISON_REL = (
    "build/current-dsv4-live-identifier-cache-source-comparison-20260523.json"
)
DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_REL = (
    "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json"
)
DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_FALLBACK_PRE_FORCE_OFF_REL = (
    "build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json"
)
DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_FALLBACK_REL = (
    "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524-thinking-on-responses-controls.json"
)
DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_LEGACY_FALLBACK_REL = (
    "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524-promptdiag-bb5cfe0c.json"
)
DSV4_CURRENT_GENERATED_ONLY_DIRECT_RAIL_EXACTNESS_REL = (
    "build/current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json"
)
DSV4_CURRENT_REQUESTED_THINKING_EXACTNESS_REL = (
    "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json"
)
DSV4_CURRENT_REP1_RAIL_EXACTNESS_REL = (
    "build/current-dsv4-route-mode-code-exactness-current-rep1-controls-20260524-2104.json"
)
DSV4_CURRENT_SOURCE_REP1_RAIL_EXACTNESS_REL = (
    "build/current-dsv4-route-mode-code-exactness-source-thinking-ab-prefill-logits-eval-20260525.json"
)
DSV4_CURRENT_SOURCE_TOKEN_TAIL_AB_EXACTNESS_REL = (
    "build/current-dsv4-route-mode-code-exactness-token-tail-source-ab-20260525.json"
)
DSV4_CURRENT_SOURCE_REP1_DIRECT_ONLY_REL = (
    "build/current-dsv4-route-mode-code-exactness-source-rep1-prefill-logits-eval-20260525.json"
)
DSV4_CURRENT_SOURCE_BUNDLE_DEFAULTS_EXACTNESS_REL = (
    "build/current-dsv4-route-mode-code-exactness-bundle-defaults-source-20260525.json"
)
DSV4_CURRENT_SOURCE_BUNDLE_DEFAULTS_DRYRUN_REL = (
    "build/current-dsv4-route-mode-code-exactness-bundle-defaults-dryrun-20260525.json"
)
DSV4_CURRENT_SOURCE_MEMORY_PREFLIGHT_REL = (
    CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT
)
DSV4_CURRENT_JANGTQK_DIRECT_OFF_RECHECK_REL = (
    "build/current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json"
)
DSV4_CURRENT_ROUTE_MODE_DRYRUN_REL = (
    "build/current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json"
)
DSV4_CURRENT_ROUTE_MODE_DRYRUN_FALLBACK_REL = (
    "build/current-dsv4-route-mode-code-exactness-dryrun-20260524-thinking-on-responses-controls.json"
)
DSV4_CURRENT_ROUTE_MODE_DRYRUN_IDENTIFIER_CANDIDATES_REL = (
    "build/current-dsv4-route-mode-code-exactness-dryrun-20260528-identifier-candidates.json"
)
DSV4_CURRENT_ROUTE_MODE_DRYRUN_COHESIVE_AUDIT_REL = (
    "build/current-dsv4-route-mode-code-exactness-dryrun-20260528-current-cohesive-audit.json"
)
DSV4_CHATMAX_PROMPT_TRIGGER_REL = (
    "build/current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json"
)
DSV4_CHATMAX_BUDGET_STOP_RAIL_REL = (
    "build/current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json"
)
DSV4_PROMPT_BOUNDARY_BISECTION_REL = (
    "build/current-dsv4-prompt-boundary-bisection-live-20260524-1317.json"
)
DSV4_COLON_PERIOD_LOGPROB_TRACE_REL = (
    "build/current-dsv4-colon-vs-period-logprob-trace-live-20260524-1320.json"
)
DSV4_COLON_PERIOD_VISIBLE_LOGPROB_TRACE_REL = (
    "build/current-dsv4-colon-vs-period-visible-logprob-trace-live-20260524-1322.json"
)
DSV4_SCENE_TOKEN_RANK_CONTRAST_REL = (
    "build/current-dsv4-scene-token-rank-contrast-live-20260524-1324.json"
)
DSV4_DIRECT_VS_THINKING_WEBGL_LOGIT_PROBE_REL = (
    "build/current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json"
)
DSV4_HIDDEN_REASONING_CONTROL_REL = (
    "build/current-dsv4-hidden-reasoning-control-live-20260524-1335.json"
)
DSV4_TEMPLATE_PARITY_DIAGNOSTIC_REL = (
    "build/current-dsv4-template-parity-diagnostic-20260524-1343.json"
)
DSV4_PREFILL_EXECUTION_VARIANT_LOGITS_REL = (
    "build/current-dsv4-jang-prefill-execution-variant-logits-20260524.json"
)
DSV4_PROMPT_VARIANT_LOGIT_PROBE_REL = (
    "build/current-dsv4-jang-prompt-variant-logit-probe-20260524.json"
)
DSV4_REASONING_POLICY_LIVE_REL = (
    "build/current-dsv4-reasoning-policy-live-20260524-1408.json"
)
DSV4_CACHE_VS_FULL_LOGIT_ISOLATION_REL = (
    "build/current-dsv4-jang-cache-vs-full-logit-isolation-threejs-20260524.json"
)
DSV4_BATCH_GENERATOR_LOGIT_TRACE_REL = (
    "build/current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json"
)
DSV4_BATCH_GENERATOR_WARMUP_ABLATION_REL = (
    "build/current-dsv4-jang-batch-generator-warmup-ablation-20260524.json"
)
API_CACHE_CONTRACT_REL = "build/current-noheavy-api-cache-contract-after-mimo-modality-truth-20260606.json"
CACHE_ARCHITECTURE_CONTRACT_REL = "build/current-cache-architecture-contract-after-mimo-modality-truth-20260606.json"
TOOL_CALL_CONTRACT_REL = "build/current-tool-call-contract-after-mimo-modality-truth-20260606.json"
PANEL_SETTINGS_CONTRACT_REL = "build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json"
MAX_OUTPUT_CONTEXT_CONTRACT_REL = "build/current-max-output-context-contract-after-mimo-modality-truth-20260606.json"
MAX_OUTPUT_CONTEXT_CONTRACT_FALLBACK_REL = "build/current-max-output-context-contract-20260521.json"
MODEL_FAMILY_CONTRACT_REL = "build/current-model-family-detection-contract-after-mimo-modality-truth-20260606.json"
PARSER_REGISTRY_CONTRACT_REL = "build/current-parser-registry-contract-after-mimo-modality-truth-20260606.json"
MODEL_ARTIFACT_FORMAT_CONTRACT_REL = "build/current-model-artifact-format-contract-after-mimo-modality-truth-20260606.json"
GENERATION_DEFAULTS_CONTRACT_REL = "build/current-generation-defaults-contract-after-mimo-modality-truth-20260606.json"
NATIVE_MTP_CONTRACT_REL = "build/current-native-mtp-contract-after-mimo-modality-truth-20260606.json"
VL_MEDIA_CONTRACT_REL = "build/current-vl-media-cache-contract-after-bundled-refresh-continuation-20260606.json"
QWEN_JANG_SOURCE_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-source-20260606.json"
QWEN_JANG_PACKAGED_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-installed-app-deterministic-pp-20260606.json"
QWEN_NATIVE_MTP_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-20260523.json"
QWEN_NATIVE_MTP_PREFILL_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json"
QWEN_NATIVE_MTP_PACKAGED_PREFILL_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json"
QWEN_NATIVE_MTP_PREFILL_TRACE_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json"
QWEN_JANG_TEXT_BASELINE_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-source-20260606.json"
QWEN_NATIVE_MTP_NO_PREFIX_LOGITS_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-no-prefix-logits-20260523.json"
QWEN_NATIVE_MTP_HYBRID_LONG_PREFIX_SPLIT_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-hybrid-long-prefix-split-20260523.json"
QWEN_NATIVE_MTP_KVNONE_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-kvnone-20260523.json"
QWEN_NATIVE_MTP_ROUTE_TRACE_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-route-trace-20260523.json"
QWEN_RAW_FORWARD_AB_1024_REL = "build/current-qwen-forward-path-ab-1024-vlm-loader-20260523.json"
QWEN_RAW_FORWARD_AB_4096_REL = "build/current-qwen-forward-path-ab-4096-vlm-loader-20260523.json"
QWEN_NATIVE_MTP_NORM_SHIFT_CLEARANCE_REL = (
    "build/current-decode-speed-live-qwen27-jang4m-mtp-installed-app-deterministic-pp-20260606.json"
)
QWEN_NATIVE_MTP_AB_REL = (
    "build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-installed-app-20260606/result.json"
)
DSV4_DEFAULT_CACHE_TOOL_LOOP_REL = "build/current-dsv4-default-cache-tool-loop/result.json"
DSV4_RESPONSES_CACHE_GATE_REL = "build/current-dsv4-responses-cache-gate-20260606.json"
DSV4_RESPONSES_RESTART_L2_GATE_REL = (
    "build/current-dsv4-responses-restart-l2-gate-20260606.json"
)
DSV4_RESPONSES_ONE_TOOL_STOP_REL = (
    "build/current-dsv4-responses-one-tool-stop-20260606.json"
)
DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_ON_REL = (
    "build/current-dsv4-default-cache-tool-loop-thinking-on-20260525.json"
)
DSV4_DEFAULT_CACHE_TOOL_LOOP_NOCACHE_AB_REL = (
    "build/current-dsv4-default-cache-tool-loop-nocache-ab-20260525.json"
)
DSV4_DEFAULT_CACHE_TOOL_LOOP_PROMPT_GUARD_REL = (
    "build/current-dsv4-default-cache-tool-loop-after-invoke-prompt-guard-20260525.json"
)
DSV4_DEFAULT_CACHE_TOOL_LOOP_COPY_BLOCK_REL = (
    "build/current-dsv4-default-cache-tool-loop-copy-block-prompt-20260525.json"
)
DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_COPY_BLOCK_REL = (
    "build/current-dsv4-default-cache-tool-loop-thinking-on-copy-block-768-20260525.json"
)
DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_COPY_BLOCK_DRYRUN_REL = (
    "build/current-dsv4-default-cache-tool-loop-thinking-on-copy-block-768-dryrun-20260525.json"
)
DSV4_DEFAULT_CACHE_TOOL_LOOP_DRYRUN_CONTROLS_REL = (
    "build/current-dsv4-default-cache-tool-loop-dryrun-controls-20260525.json"
)
LING_INSTALLED_LIVE_AUDIT_REL = (
    "build/current-production-family-live-ling-bundled-current-20260606.json"
)
LING_JANGTQ_STRICT_RUSSIAN_NOCACHE_REL = (
    "build/current-ling-jangtq-strict-russian-nocache-bundled-4850c9c2-20260524.json"
)
LING_MXFP4_STRICT_RUSSIAN_NOCACHE_REL = (
    "build/current-ling-mxfp4-crack-strict-russian-nocache-bundled-4850c9c2-20260524.json"
)
LING_JANGTQ_RUSSIAN_PROMPT_VARIANT_REL = (
    "build/current-ling-jangtq-russian-prompt-variant-probe-20260524.json"
)
LING_JANGTQ_SOURCE_PREFILL_STREAM_REL = (
    "build/current-ling-jangtq-server-repeat-russian-source-prefill-stream-20260524.json"
)
LING_JANGTQ_BUNDLED_COMPAT_PREFILL_STREAM_REL = (
    "build/current-ling-jangtq-server-repeat-russian-bundled-prefill-stream-20260524.json"
)
LING_JANGTQ_BUNDLED_NATIVE_PREFILL_STREAM_REL = (
    "build/current-ling-jangtq-server-repeat-russian-bundled-native-prefill-stream-20260524.json"
)
LING_BUNDLED_NATIVE_LIVE_RERUN_REL = (
    "build/current-production-family-live-ling-bundled-native-rerun-20260524.json"
)
LING_JANGTQ_COLD_SKIPCACHE_REPEAT_REL = (
    "build/current-ling-jangtq-cold-skipcache-repeat-bundled-native-20260524.json"
)
LING_JANGTQ_BATCHGEN_TEMP0_REPEAT_REL = (
    "build/current-ling-jangtq-batchgen-temp0-repeat-bundled-native-20260524.json"
)
LING_SIMPLE_AFTER_MPP_FIX_REL = (
    "build/current-ling-jangtq-simple-engine-control-bundled-after-mpp-fix-20260524.json"
)
LING_CONTINUOUS_AFTER_MPP_FIX_REL = (
    "build/current-ling-jangtq-continuous-control-bundled-after-mpp-fix-20260524.json"
)
LING_BUNDLED_AFTER_MPP_FIX_LIVE_REL = (
    "build/current-production-family-live-ling-bundled-after-mpp-fix-20260524.json"
)
LING_BUNDLED_AFTER_TOPK_POLICY_LIVE_REL = (
    "build/current-production-family-live-ling-bundled-after-topk-policy-20260524.json"
)
GEMMA4_RESPONSES_VISIBLE_CONTRACT_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json"
)
GEMMA4_RESPONSES_VISIBLE_CURRENT_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260606.json"
)
GEMMA4_RESPONSES_UNSUPPORTED_THINKING_BUDGET_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-contract-20260524.json"
)
GEMMA4_RESPONSES_VISIBLE_NOCACHE_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-nocache-20260524.json"
)
GEMMA4_RESPONSES_VISIBLE_512_NOCACHE_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-512-nocache-20260524.json"
)
GEMMA4_RESPONSES_THINKING_OFF_NOCACHE_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingoff-visible-nocache-20260524.json"
)
GEMMA4_CHAT_VISIBLE_NOCACHE_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingbudget16-visible-nocache-20260524.json"
)
GEMMA4_LOCAL_METADATA_AUDIT_REL = (
    "build/current-local-generation-metadata-audit-20260524-gemma4-visible-budget.json"
)
ALL_LOCAL_MODEL_SMOKE_ZAYA_TEXT_REL = (
    "build/current-all-local-model-smoke-zaya-text-bundled-20260524/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_ZAYA_TEXT_VL_CURRENT_REL = (
    "build/current-all-local-model-smoke-zaya-text-vl-tools-media-after-reasoning-budget-20260606/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_REL = (
    "build/current-all-local-model-smoke-zaya-vl-bundled-20260524/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_JANGTQ4_REL = (
    "build/current-all-local-model-smoke-zaya-vl-jangtq4-true-bundled-toolprobe-media-sentinel-20260525/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_CURRENT_REL = (
    "build/current-all-local-model-smoke-zaya-vl-mxfp4-after-no-media-contract-20260606/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_NEMOTRON_OMNI_JANGTQ_REL = (
    "build/current-all-local-model-smoke-nemotron-omni-jangtq-video-bundled-20260526-rerun/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_LING_HY3_NEMOTRON_CURRENT_REL = (
    "build/current-all-local-model-smoke-ling-hy3-nemotron-tools-media-20260606/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_GEMMA4_26B_CRACK_REL = (
    "build/current-all-local-model-smoke-gemma26-jang4m-tools-media-continuation-20260606/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_LING_BAILING_JANGTQ_REL = (
    "build/current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260525-rerun/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_QWEN36_MXFP4_CRACK_REL = (
    "build/current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260525-rerun/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_QWEN35_MXFP8_MTP_CURRENT_REL = (
    "build/current-all-local-model-smoke-qwen35-mxfp8-mtp-tools-media-20260606/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_HY3_JANGTQ2_REL = (
    "build/current-all-local-model-smoke-hy3-jangtq2-bundled-toolprobe-20260525/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_MINIMAX_SMALL_JANGTQ_REL = (
    "build/current-all-local-model-smoke-minimaxk-tools-continuation-20260606/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_LIVE_SLICE_CURRENT_REL = (
    "build/current-all-local-model-smoke-live-slice-tools-media-continuation-20260606/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_MIMO_V2_JANG2L_REL = (
    "build/current-all-local-model-smoke-mimo-v25-jang2l-tools-media-rerun-20260606/summary.json"
)
MIMO_V2_JANG2L_STRUCTURAL_VERIFY_REL = (
    "build/current-mimo-jang2l-local-structural-verify-20260606.json"
)
MIMO_V2_JANG2L_TEXT_CACHE_REL = (
    "build/current-mimo-jang2l-live-text-cache-smoke-20260606.json"
)
MIMO_V2_JANG2L_SWITCHGLU_PARITY_REL = (
    "build/current-mimo-v2-jang2l-quantized-switchglu-parity-20260606.json"
)
MIMO_V2_JANG2L_LENGTH_SWEEP_REL = (
    "build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json"
)
MIMO_V2_JANG2L_TOOL_DIALECT_REL = (
    "build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json"
)
MIMO_V2_JANG2L_CURRENT_AUDIT_REL = (
    "build/current-mimo-v2-jang2l-current-audit-20260606.json"
)
MIMO_V2_JANG2L_CONSERVATIVE_DIAGNOSTIC_REL = (
    "build/current-mimo-conservative-diagnostic-20260606/summary.json"
)
ALL_LOCAL_MODEL_SMOKE_DSV4_JANGTQ_K_REL = (
    "build/current-all-local-model-smoke-dsv4-jangtq-k-bundled-cachehit-20260524/summary.json"
)
NEMOTRON_OMNI_NO_MEDIA_DIAGNOSTIC_REL = (
    "build/current-nemotron-omni-no-media-carryover-diagnostic-20260524b/result.json"
)
NEMOTRON_OMNI_NO_MEDIA_PROMPT_VARIANTS_REL = (
    "build/current-nemotron-omni-no-media-prompt-variants-20260524/result.json"
)
NEMOTRON_OMNI_NO_MEDIA_SYSTEM_PROMPT_REL = (
    "build/current-nemotron-omni-no-media-system-prompt-diagnostic-20260524/result.json"
)
NEMOTRON_OMNI_NO_MEDIA_SYSTEM_NEGATIVE_REL = (
    "build/current-nemotron-omni-no-media-system-negative-diagnostic-20260524/result.json"
)
ZAYA_VL_JANGTQ4_ACK_DIAGNOSTIC_REL = (
    "build/current-zaya-vl-jangtq4-ack-diagnostic-20260524/external_probe.json"
)
ZAYA_VL_JANGTQ4_RENDERED_PROMPT_COMPARE_REL = (
    "build/current-zaya-vl-jangtq4-ack-diagnostic-20260524/rendered_prompt_compare.json"
)
ALL_LOCAL_MODEL_SMOKE_REQUIRED_FAMILIES = (
    "dsv4",
    "gemma4",
    "hy3",
    "lfm",
    "ling_bailing",
    "minimax",
    "mimo_v2",
    "nemotron",
    "qwen36",
    "step3p7",
    "zaya_text",
    "zaya_vl",
)
ALL_LOCAL_MODEL_SMOKE_ARTIFACTS_BY_FAMILY = {
    "dsv4": [ALL_LOCAL_MODEL_SMOKE_DSV4_JANGTQ_K_REL],
    "gemma4": [ALL_LOCAL_MODEL_SMOKE_LIVE_SLICE_CURRENT_REL, ALL_LOCAL_MODEL_SMOKE_GEMMA4_26B_CRACK_REL],
    "hy3": [ALL_LOCAL_MODEL_SMOKE_LING_HY3_NEMOTRON_CURRENT_REL, ALL_LOCAL_MODEL_SMOKE_HY3_JANGTQ2_REL],
    "lfm": [ALL_LOCAL_MODEL_SMOKE_LIVE_SLICE_CURRENT_REL],
    "ling_bailing": [ALL_LOCAL_MODEL_SMOKE_LING_HY3_NEMOTRON_CURRENT_REL, ALL_LOCAL_MODEL_SMOKE_LING_BAILING_JANGTQ_REL],
    "minimax": [ALL_LOCAL_MODEL_SMOKE_LIVE_SLICE_CURRENT_REL, ALL_LOCAL_MODEL_SMOKE_MINIMAX_SMALL_JANGTQ_REL],
    "mimo_v2": [ALL_LOCAL_MODEL_SMOKE_MIMO_V2_JANG2L_REL],
    "nemotron": [ALL_LOCAL_MODEL_SMOKE_LING_HY3_NEMOTRON_CURRENT_REL, ALL_LOCAL_MODEL_SMOKE_NEMOTRON_OMNI_JANGTQ_REL],
    "qwen36": [ALL_LOCAL_MODEL_SMOKE_QWEN35_MXFP8_MTP_CURRENT_REL, ALL_LOCAL_MODEL_SMOKE_LIVE_SLICE_CURRENT_REL, ALL_LOCAL_MODEL_SMOKE_QWEN36_MXFP4_CRACK_REL],
    "step3p7": [ALL_LOCAL_MODEL_SMOKE_LIVE_SLICE_CURRENT_REL],
    "zaya_text": [ALL_LOCAL_MODEL_SMOKE_ZAYA_TEXT_VL_CURRENT_REL, ALL_LOCAL_MODEL_SMOKE_ZAYA_TEXT_REL],
    "zaya_vl": [
        ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_CURRENT_REL,
        ALL_LOCAL_MODEL_SMOKE_ZAYA_TEXT_VL_CURRENT_REL,
        ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_JANGTQ4_REL,
    ],
}
GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S = 80.0
GEMMA4_MIXED_SWA_STREAMING_MIN_COMPLETION_TOKENS = 256
GEMMA4_MIXED_SWA_STREAMING_MIN_TURNS = 2
GEMMA4_MIXED_SWA_SPEED_ARTIFACT_RELS = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
)
GEMMA4_MIXED_SWA_SPEED_CURRENT_RELS = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-20260606.json",
)
GEMMA4_MIXED_SWA_SUSTAINED_CACHEHIT_DIAGNOSTIC_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-cachehit-trace-20260525.json"
)
GEMMA4_MIXED_SWA_SHORT_NOCACHE_REPEAT_DIAGNOSTIC_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-trace-20260525.json"
)
GEMMA4_MIXED_SWA_SHORT_NOCACHE_SCHEDULER_TRACE_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-scheduler-trace-20260525.json"
)
GEMMA4_MIXED_SWA_SHORT_NOCACHE_SYNC_EVAL_AB_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-source-triple-nocache-256-sync-eval-ab-20260525.json"
)
GEMMA4_MIXED_SWA_SHORT_NOCACHE_STREAMING_REL = (
    "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-streaming-20260525.json"
)
DSV4_QUALITY_CLEARANCE_CHECKS = (
    "identifier_integrity",
    "threejs_single_file",
    "no_markdown_fence",
    "no_corrupt_identifiers",
    "non_length_stop",
    "source_or_rebuilt_body_clearance",
    "cached_vs_no_cache_semantic_equivalence",
)
DSV4_LEGACY_CLEARANCE_ARTIFACT_PATHS = frozenset(
    {
        "build/dsv4-source-identifier/result.json",
        "build/dsv4-source-full-output/result.json",
        "build/dsv4-chat-prompt-ablation-20260520101331/result.json",
    }
)
DSV4_THREEJS_IDENTIFIERS = (
    "THREE.Scene",
    "THREE.WebGLRenderer",
    "THREE.PerspectiveCamera",
    "THREE.Mesh",
    "THREE.BoxGeometry",
    "THREE.MeshBasicMaterial",
)
DSV4_THREEJS_CORRUPT_PATTERNS = (
    "THREE.WebRenderer",
    "THREE.WebWebGLRenderer",
    "THREE.WebScriptRenderer",
    "THREE.PPerspectiveCamera",
    "THREE.PersPerspectiveCamera",
    "THREE.BBoxGeometry",
    "THREE.MMeshBasicMaterial",
    "Three.MeshBasicMaterial",
)
API_CACHE_CONTRACT_CHECKS = (
    "openai_chat_sampling_kwargs",
    "responses_sampling_kwargs",
    "anthropic_bundle_defaults",
    "ollama_adapter_surface",
    "responses_previous_response_history",
    "dsv4_native_cache_status",
    "dsv4_dsml_parser_residue_rejection",
    "dsv4_dsml_valid_tool_call_preserved",
    "dsv4_suppressed_tool_markup_not_stored",
    "zaya_typed_cca_status",
    "hybrid_ssm_partial_reuse",
    "turboquant_kv_runtime_contract",
    "turboquant_disk_roundtrip",
    "no_generic_tq_on_hybrid_ssm",
)
API_CACHE_SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/tool_parsers/dsml_tool_parser.py",
    "tests/test_engine_audit.py",
    "tests/test_batching.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_tq_disk_cache.py",
    "tests/test_dsml_tool_parser.py",
    "tests/test_responses_history.py",
    "tests/test_tool_format.py",
)
PANEL_SETTINGS_CONTRACT_CHECKS = (
    "dsv4_default_native_prefix_on",
    "dsv4_explicit_prefix_off_disables_native_flags",
    "dsv4_l2_explicit_off_preserves_prefix",
    "dsv4_generic_kv_flags_suppressed",
    "max_output_context_cli_split",
    "chat_max_output_is_per_chat_override",
    "non_dsv4_cache_toggles_preserved",
    "i18n_max_output_context_copy",
    "panel_typecheck",
)
MAX_OUTPUT_CONTEXT_CONTRACT_CHECKS = (
    "server_default_output_cap_uses_max_tokens",
    "startup_output_cap_is_default_not_request_ceiling",
    "request_output_caps_can_go_below_or_above_startup_default",
    "request_output_caps_do_not_mutate_server_default",
    "legacy_completions_output_cap_overrides_server_default",
    "chat_max_tokens_overrides_server_default_per_request",
    "responses_max_output_tokens_overrides_server_default_per_request",
    "anthropic_messages_preserves_bundle_and_explicit_output_caps",
    "ollama_num_predict_maps_only_positive_output_caps",
    "prompt_context_caps_do_not_rewrite_output_cap",
    "panel_server_default_output_maps_to_max_tokens",
    "panel_max_context_maps_to_max_prompt_tokens",
    "stale_32768_session_output_caps_are_migrated",
    "chat_output_cap_remains_per_chat_override",
    "request_builders_omit_auto_output_cap",
    "new_chat_output_caps_are_not_inherited_or_made_sticky",
    "all_family_max_token_precedence_stays_uniform",
    "wake_reload_and_cli_preserve_explicitness",
    "all_required_max_output_context_markers_present",
)
PARSER_REGISTRY_CONTRACT_CHECKS = (
    "engine_accepts_registered_reasoning_parsers",
    "engine_accepts_registered_tool_parsers",
    "panel_emitted_reasoning_parsers_are_engine_valid",
    "panel_emitted_tool_parsers_are_engine_valid",
    "minimax_m2_reasoning_parser_regression",
    "parser_aliases_are_canonical_before_cli",
    "zaya_hy3_ling_dsv4_parser_rows_are_present",
    "non_reasoning_family_boundaries_are_present",
    "all_required_parser_markers_present",
)
MODEL_ARTIFACT_FORMAT_CONTRACT_CHECKS = (
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
GENERATION_DEFAULTS_CONTRACT_CHECKS = (
    "generation_config_defaults_are_surfaced",
    "jang_config_sampling_defaults_override_generation_config",
    "disabled_top_k_sentinels_normalize_to_off",
    "mode_specific_jang_repetition_penalty_is_metadata_owned",
    "request_api_overrides_win_over_startup_defaults",
    "bundle_max_new_tokens_preserved_when_omitted",
    "omitted_max_tokens_without_bundle_default_is_bounded",
    "server_default_output_cap_is_not_request_ceiling",
    "no_hidden_sampler_forcing_or_repetition_floor",
    "additional_args_cannot_override_app_owned_cli_flags",
    "local_high_risk_model_metadata_audit",
    "panel_does_not_emit_default_sampler_cli_flags",
    "legacy_count_floor_still_nontrivial",
    "generation_defaults_family_matrix_complete",
)
NATIVE_MTP_CONTRACT_CHECKS = (
    "native_mtp_d3_default_policy",
    "model_tuning_depth_policy",
    "dropped_and_preserved_mtp_detection",
    "config_only_mtp_never_activates",
    "mxfp4_mxfp8_mtp_artifact_detection",
    "mllm_native_mtp_decode_loop",
    "native_mtp_telemetry_edge_cases",
    "panel_native_mtp_controls_visible_when_supported",
    "panel_native_mtp_suppressed_for_dsv4_or_unsupported",
    "live_speed_equivalence_not_claimed",
)
VL_MEDIA_CONTRACT_CHECKS = (
    "video_url_request_schema",
    "video_fallback_processing",
    "qwen36_vl_video_detection",
    "media_cache_salt_separates_modal_inputs",
    "hybrid_ssm_vlm_cache_contracts",
    "mllm_tool_replay_preserves_effective_tools",
    "panel_read_video_builtin_tool",
    "panel_media_tool_followup_content_parts",
    "panel_image_display_consistency",
    "panel_vlm_launch_settings",
    "all_required_panel_markers_present",
)
PANEL_SETTINGS_SOURCE_HASH_FILES = (
    "panel/src/main/sessions.ts",
    "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/chat/ChatSettings.tsx",
    "panel/src/shared/dsv4Env.ts",
    "panel/src/shared/cacheControlPolicy.ts",
    "panel/tests/settings-flow.test.ts",
    "panel/tests/dsv4-env.test.ts",
    "panel/tests/cache-control-policy.test.ts",
)


def _load(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic digest should continue
        return {"_load_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def _status(ok: bool, *, partial: bool = False) -> str:
    if ok:
        return "pass"
    return "partial" if partial else "open"


def _add(
    requirements: list[dict[str, Any]],
    requirement: str,
    status: str,
    evidence: list[str],
    *,
    caveat: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
    }
    if caveat:
        item["caveat"] = caveat
    if details is not None:
        item["details"] = details
    requirements.append(item)


def _static_rows(static: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in static.get("rows") or []:
        if not isinstance(row, dict):
            continue
        payload = row.get("static") or {}
        row_id = payload.get("id")
        if isinstance(row_id, str):
            rows[row_id] = payload
    return rows


def _function_calls(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in response_body.get("output") or []
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def _usage_details(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    return details if isinstance(details, dict) else {}


def _cached_tokens_from_payload(payload: dict[str, Any]) -> int:
    details = _usage_details(payload)
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _cache_detail_from_payload(payload: dict[str, Any]) -> str:
    return str(_usage_details(payload).get("cache_detail") or "")


def _present_patterns(text: str | None, patterns: tuple[str, ...]) -> list[str]:
    if not isinstance(text, str):
        return []
    return [pattern for pattern in patterns if pattern in text]


def _known_corrupt_identifier_patterns(value: Any, text: str | None = None) -> list[str]:
    known = set(DSV4_THREEJS_CORRUPT_PATTERNS)
    patterns: list[str] = []
    if isinstance(value, list):
        patterns.extend(str(item) for item in value if item in known)
    patterns.extend(_present_patterns(text, DSV4_THREEJS_CORRUPT_PATTERNS))
    return list(dict.fromkeys(patterns))


def _command_tokens(payload: dict[str, Any]) -> list[str]:
    cmd = payload.get("cmd") or payload.get("command") or payload.get("server_args") or []
    if isinstance(cmd, str):
        return cmd.split()
    if isinstance(cmd, list):
        return [str(item) for item in cmd]
    return []


def _resolve_artifact_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _path_present(root: Path, value: str) -> bool:
    path = _resolve_artifact_path(root, value)
    return path.is_file() and path.stat().st_size > 0


def _load_first_present(root: Path, values: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    for value in values:
        if _path_present(root, value):
            return value, _load(root, value)
    first = values[0] if values else ""
    return first, _load(root, first) if first else {}


def _current_real_ui_live_model_matrix(root: Path) -> dict[str, Any]:
    manifest = _load(root, CURRENT_RELEASE_REGRESSION_MANIFEST_REL)
    sweep = manifest.get("current_proof_sweep")
    if not isinstance(sweep, dict):
        return {}
    matrix = sweep.get("real_ui_live_model_matrix")
    if not isinstance(matrix, dict):
        return {}
    preflight = _load(root, CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT)
    if (
        "dsv4" in {str(item) for item in matrix.get("missing_families", [])}
        and preflight.get("status") == "skipped_insufficient_memory"
        and preflight.get("launch_decision") == "do_not_launch"
    ):
        resource_blockers = dict(matrix.get("resource_blockers") or {})
        resource_blockers["dsv4"] = {
            "artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
            "reason": "insufficient_memory",
            "model_path": preflight.get("model_path"),
            "required_available_gb": preflight.get("required_available_gb"),
            "free_plus_speculative_purgeable_gb": preflight.get(
                "free_plus_speculative_purgeable_gb"
            ),
            "memory_gap_gb": preflight.get("memory_gap_gb"),
        }
        matrix["resource_blockers"] = resource_blockers
    return matrix


def _real_ui_blocking_family_artifacts(matrix: dict[str, Any]) -> dict[str, list[str]]:
    missing = [str(item) for item in matrix.get("missing_families", [])]
    resource_blockers = matrix.get("resource_blockers")
    if not isinstance(resource_blockers, dict):
        resource_blockers = {}
    blocking: dict[str, list[str]] = {}
    for family in missing:
        artifacts: list[str] = []
        blocker = resource_blockers.get(family)
        if isinstance(blocker, dict) and blocker.get("artifact"):
            artifacts.append(str(blocker["artifact"]))
        else:
            artifacts.extend(ALL_LOCAL_MODEL_SMOKE_ARTIFACTS_BY_FAMILY.get(family, []))
        blocking[family] = sorted(set(artifacts))
    return blocking


def _attach_evidence_file_status(requirements: list[dict[str, Any]], root: Path) -> None:
    for item in requirements:
        evidence = item.get("evidence") or []
        if not isinstance(evidence, list):
            continue
        evidence_files_present: dict[str, bool] = {}
        missing_evidence: list[str] = []
        for value in evidence:
            if not isinstance(value, str) or not value:
                continue
            present = _path_present(root, value)
            evidence_files_present[value] = present
            if not present:
                missing_evidence.append(value)
        details = item.setdefault("details", {})
        if isinstance(details, dict):
            details["evidence_files_present"] = evidence_files_present
            details["missing_evidence"] = missing_evidence
        if item.get("status") == "pass" and missing_evidence:
            item["status"] = "open"
            caveat = item.get("caveat")
            missing_note = "Listed evidence files are missing or empty."
            item["caveat"] = f"{caveat} {missing_note}" if caveat else missing_note


def _dsv4_quality_clearance(clearance: dict[str, Any], root: Path) -> tuple[bool, dict[str, Any]]:
    checks = clearance.get("checks") or {}
    required = {key: checks.get(key) is True for key in DSV4_QUALITY_CLEARANCE_CHECKS}
    artifacts = clearance.get("artifacts") or {}
    required_artifacts = ("identifier_gate", "full_output_gate")
    artifact_paths_present: dict[str, bool] = {}
    legacy_artifacts: dict[str, str] = {}
    missing_artifacts: list[str] = []
    non_string_artifacts: list[str] = []

    for key in required_artifacts:
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            non_string_artifacts.append(key)
            artifact_paths_present[key] = False
            missing_artifacts.append(str(value) if value is not None else key)

    for key, value in artifacts.items():
        if not isinstance(value, str) or not value:
            if key not in non_string_artifacts:
                non_string_artifacts.append(str(key))
            artifact_paths_present[str(key)] = False
            continue
        if value in DSV4_LEGACY_CLEARANCE_ARTIFACT_PATHS:
            legacy_artifacts[str(key)] = value
            artifact_paths_present[str(key)] = False
            continue
        present = _path_present(root, value)
        artifact_paths_present[str(key)] = present
        if not present:
            missing_artifacts.append(value)

    required_artifact_paths_present = all(
        artifact_paths_present.get(key) is True for key in required_artifacts
    )
    ok = (
        clearance.get("status") == "pass"
        and all(required.values())
        and required_artifact_paths_present
        and not missing_artifacts
        and not non_string_artifacts
    )
    return ok, {
        "clearance_status": clearance.get("status"),
        "clearance_checks": required,
        "clearance_artifacts": artifacts,
        "clearance_artifact_paths_present": artifact_paths_present,
        "legacy_clearance_artifacts": legacy_artifacts,
        "non_current_clearance_reason": (
            "Legacy DSV4 clearance artifacts are not accepted as current release evidence."
            if legacy_artifacts
            else None
        ),
        "missing_clearance_artifacts": missing_artifacts,
        "non_string_clearance_artifacts": non_string_artifacts,
    }


def _cjk_count_from_text(value: str) -> int:
    return sum(
        1
        for char in value
        if (
            "\u3400" <= char <= "\u4dbf"
            or "\u4e00" <= char <= "\u9fff"
            or "\uf900" <= char <= "\ufaff"
        )
    )


def _ling_request_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "content" in payload or "counts" in payload or "quality" in payload:
        rows.append(payload)
    for request in payload.get("requests") or []:
        if isinstance(request, dict):
            rows.append(request)
    for row in payload.get("rows") or []:
        if isinstance(row, dict) and (
            "content" in row or "counts" in row or "quality" in row
        ):
            rows.append(row)
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        live = row.get("live") or {}
        if not isinstance(live, dict):
            continue
        for request in live.get("requests") or []:
            if isinstance(request, dict):
                rows.append(request)
        for check in live.get("checks") or []:
            if not isinstance(check, dict):
                continue
            detail = check.get("detail")
            if isinstance(detail, dict):
                row = dict(detail)
                row.setdefault("name", check.get("name"))
                rows.append(row)
    return rows


def _ling_multilingual_quality_detail(
    root: Path,
    installed_live: dict[str, Any],
    jangtq_strict: dict[str, Any],
    mxfp4_strict: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    artifacts = {
        LING_INSTALLED_LIVE_AUDIT_REL: installed_live,
        LING_JANGTQ_STRICT_RUSSIAN_NOCACHE_REL: jangtq_strict,
        LING_MXFP4_STRICT_RUSSIAN_NOCACHE_REL: mxfp4_strict,
        LING_JANGTQ_RUSSIAN_PROMPT_VARIANT_REL: _load(
            root,
            LING_JANGTQ_RUSSIAN_PROMPT_VARIANT_REL,
        ),
        LING_JANGTQ_SOURCE_PREFILL_STREAM_REL: _load(
            root,
            LING_JANGTQ_SOURCE_PREFILL_STREAM_REL,
        ),
        LING_JANGTQ_BUNDLED_COMPAT_PREFILL_STREAM_REL: _load(
            root,
            LING_JANGTQ_BUNDLED_COMPAT_PREFILL_STREAM_REL,
        ),
        LING_JANGTQ_BUNDLED_NATIVE_PREFILL_STREAM_REL: _load(
            root,
            LING_JANGTQ_BUNDLED_NATIVE_PREFILL_STREAM_REL,
        ),
        LING_BUNDLED_NATIVE_LIVE_RERUN_REL: _load(
            root,
            LING_BUNDLED_NATIVE_LIVE_RERUN_REL,
        ),
        LING_JANGTQ_COLD_SKIPCACHE_REPEAT_REL: _load(
            root,
            LING_JANGTQ_COLD_SKIPCACHE_REPEAT_REL,
        ),
        LING_JANGTQ_BATCHGEN_TEMP0_REPEAT_REL: _load(
            root,
            LING_JANGTQ_BATCHGEN_TEMP0_REPEAT_REL,
        ),
        LING_SIMPLE_AFTER_MPP_FIX_REL: _load(
            root,
            LING_SIMPLE_AFTER_MPP_FIX_REL,
        ),
        LING_CONTINUOUS_AFTER_MPP_FIX_REL: _load(
            root,
            LING_CONTINUOUS_AFTER_MPP_FIX_REL,
        ),
        LING_BUNDLED_AFTER_MPP_FIX_LIVE_REL: _load(
            root,
            LING_BUNDLED_AFTER_MPP_FIX_LIVE_REL,
        ),
        LING_BUNDLED_AFTER_TOPK_POLICY_LIVE_REL: _load(
            root,
            LING_BUNDLED_AFTER_TOPK_POLICY_LIVE_REL,
        ),
    }
    clearance_artifacts = {
        LING_INSTALLED_LIVE_AUDIT_REL,
    }
    artifact_statuses: dict[str, Any] = {}
    artifacts_with_cjk: list[str] = []
    clearance_artifacts_with_cjk: list[str] = []
    request_summaries: list[dict[str, Any]] = []
    max_cjk_chars = 0

    for rel, payload in artifacts.items():
        artifact_statuses[rel] = payload.get("status") or (
            (payload.get("summary") or {}).get("live_status_counts")
        )
        artifact_has_cjk = False
        for index, request in enumerate(_ling_request_rows(payload)):
            counts = request.get("counts") or request.get("quality") or {}
            content = str(request.get("content") or request.get("text") or "")
            try:
                cjk_chars = int(counts.get("cjk_chars"))
            except (AttributeError, TypeError, ValueError):
                cjk_chars = _cjk_count_from_text(content)
            max_cjk_chars = max(max_cjk_chars, cjk_chars)
            if cjk_chars:
                artifact_has_cjk = True
            request_summaries.append(
                {
                    "artifact": rel,
                    "index": index,
                    "cjk_chars": cjk_chars,
                    "content_preview": content[:180],
                }
            )
        if artifact_has_cjk:
            artifacts_with_cjk.append(rel)
            if rel in clearance_artifacts:
                clearance_artifacts_with_cjk.append(rel)

    missing = [
        rel
        for rel in artifacts
        if not _path_present(root, rel)
    ]
    missing_clearance = [
        rel
        for rel in clearance_artifacts
        if not _path_present(root, rel)
    ]
    installed_failures = []
    for row in installed_live.get("rows") or []:
        if not isinstance(row, dict):
            continue
        live = row.get("live") or {}
        if isinstance(live, dict):
            installed_failures.extend(str(item) for item in live.get("failures") or [])

    failing_statuses = [
        rel
        for rel, payload in artifacts.items()
        if str(payload.get("status") or "").lower() == "fail"
    ]
    failing_clearance_statuses = [
        rel
        for rel in clearance_artifacts
        if str((artifacts.get(rel) or {}).get("status") or "").lower() == "fail"
    ]
    live_failed = any(
        isinstance(row, dict)
        and str(((row.get("live") or {}).get("status") or "")).upper() == "FAIL"
        for row in installed_live.get("rows") or []
    )
    ok = (
        not missing_clearance
        and not clearance_artifacts_with_cjk
        and not failing_clearance_statuses
        and all(_ling_request_rows(artifacts[rel]) for rel in clearance_artifacts)
    )
    return ok, {
        "artifacts": list(artifacts),
        "missing_artifacts": missing,
        "clearance_artifacts": sorted(clearance_artifacts),
        "missing_clearance_artifacts": missing_clearance,
        "artifact_statuses": artifact_statuses,
        "artifacts_with_cjk": artifacts_with_cjk,
        "clearance_artifacts_with_cjk": clearance_artifacts_with_cjk,
        "max_cjk_chars": max_cjk_chars,
        "request_summaries": request_summaries,
        "installed_live_failed": live_failed,
        "installed_live_failures": installed_failures,
        "failing_status_artifacts": failing_statuses,
        "failing_clearance_status_artifacts": failing_clearance_statuses,
    }


def _dsv4_identifier_canary_detail(canary: dict[str, Any], root: Path, rel: str) -> dict[str, Any]:
    path_present = _path_present(root, rel)
    if isinstance(canary.get("cases"), list):
        case_summaries: list[dict[str, Any]] = []
        failed_cases: list[str] = []
        for case in canary.get("cases") or []:
            if not isinstance(case, dict):
                continue
            name = str(case.get("name") or "")
            exact = case.get("exact") is True
            if not exact and name:
                failed_cases.append(name)
            content = case.get("content")
            if not isinstance(content, str):
                content = ""
            case_summaries.append(
                {
                    "name": name,
                    "route": case.get("route"),
                    "http_code": case.get("http_code"),
                    "finish_reason": case.get("finish"),
                    "exact": exact,
                    "corrupt_patterns": case.get("corrupt_patterns") or [],
                    "missing_identifiers": case.get("missing") or [],
                    "decode_tok_s_wall": case.get("decode_tps_wall"),
                    "completion_tokens": case.get("completion_tokens"),
                    "prompt_tokens": case.get("prompt_tokens"),
                    "content": content[:1000],
                }
            )
        return {
            "artifact": rel,
            "present": path_present,
            "status": canary.get("status") if path_present else "missing",
            "model_path": canary.get("model"),
            "env": canary.get("env_overrides") or {},
            "case_count": len(case_summaries),
            "failed_cases": failed_cases,
            "case_summaries": case_summaries,
        }
    if isinstance(canary.get("probes"), list):
        probe_summaries: list[dict[str, Any]] = []
        for probe in canary.get("probes") or []:
            if not isinstance(probe, dict):
                continue
            content = probe.get("content")
            if not isinstance(content, str):
                content = ""
            probe_summaries.append(
                {
                    "name": probe.get("name"),
                    "code": probe.get("code"),
                    "finish_reason": probe.get("finish"),
                    "elapsed_sec": probe.get("elapsed_sec"),
                    "usage": probe.get("usage"),
                    "perf": probe.get("perf"),
                    "analysis": probe.get("analysis") or {},
                    "content": content[:1000],
                }
            )
        return {
            "artifact": rel,
            "present": path_present,
            "status": canary.get("status") if path_present else "missing",
            "health_model_name": (canary.get("health_after_load") or {}).get("model_name")
            if isinstance(canary.get("health_after_load"), dict)
            else None,
            "served_model_name": canary.get("served_model_name"),
            "env": canary.get("env") or {},
            "failures": canary.get("failures") or [],
            "probe_summaries": probe_summaries,
        }
    response = canary.get("response") if isinstance(canary.get("response"), dict) else {}
    choices = response.get("choices") if isinstance(response, dict) else []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    content = canary.get("content")
    if not isinstance(content, str):
        content = ""
    return {
        "artifact": rel,
        "present": path_present,
        "status": canary.get("status") if path_present else "missing",
        "elapsed_sec": canary.get("elapsed_sec"),
        "required_identifier_counts": canary.get("required_identifier_counts") or {},
        "bad_patterns": canary.get("bad_patterns") or [],
        "has_markdown_fence": canary.get("has_markdown_fence"),
        "finish_reason": first_choice.get("finish_reason"),
        "usage": response.get("usage") if isinstance(response, dict) else None,
        "content": content[:1000],
    }


def _dsv4_prompt_rail_exactness_detail(
    artifact: dict[str, Any],
    dry_run: dict[str, Any],
    root: Path,
    artifact_rel: str,
    dry_run_rel: str,
) -> dict[str, Any]:
    path_present = _path_present(root, artifact_rel)
    dry_run_present = _path_present(root, dry_run_rel)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    dry_run_cases = dry_run.get("cases") if isinstance(dry_run.get("cases"), list) else []
    case_summaries: list[dict[str, Any]] = []
    failed_cases: list[str] = []
    failed_thinking_closed_cases: list[str] = []
    thinking_open_cases_without_identifier_corruption: list[str] = []
    requested_thinking_closed_but_effective_open_cases: list[str] = []
    rep1_still_corrupt_patterns: dict[str, list[str]] = {}

    for case in cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or "")
        requested_prompt_diagnostics = (
            case.get("requested_prompt_diagnostics")
            if isinstance(case.get("requested_prompt_diagnostics"), dict)
            else case.get("prompt_diagnostics")
            if isinstance(case.get("prompt_diagnostics"), dict)
            else {}
        )
        effective_prompt_diagnostics = (
            case.get("effective_prompt_diagnostics")
            if isinstance(case.get("effective_prompt_diagnostics"), dict)
            else case.get("prompt_diagnostics")
            if isinstance(case.get("prompt_diagnostics"), dict)
            else {}
        )
        requested_suffix = str(
            requested_prompt_diagnostics.get("assistant_suffix_kind") or ""
        )
        suffix = str(effective_prompt_diagnostics.get("assistant_suffix_kind") or "")
        corrupt_patterns = [
            str(pattern) for pattern in (case.get("corrupt_patterns") or [])
        ]
        missing = [str(identifier) for identifier in (case.get("missing") or [])]
        exact = case.get("exact") is True
        if not exact and name:
            failed_cases.append(name)
        if suffix == "thinking_closed" and not exact and name:
            failed_thinking_closed_cases.append(name)
        if (
            requested_suffix == "thinking_closed"
            and suffix == "thinking_open"
            and name
        ):
            requested_thinking_closed_but_effective_open_cases.append(name)
        if suffix == "thinking_open" and not corrupt_patterns and not missing and name:
            thinking_open_cases_without_identifier_corruption.append(name)
        if name.endswith("_rep1") and corrupt_patterns:
            rep1_still_corrupt_patterns[name] = corrupt_patterns
        prompt_tail = effective_prompt_diagnostics.get("prompt_tail")
        if not isinstance(prompt_tail, str):
            prompt_tail = ""
        content = case.get("content")
        if not isinstance(content, str):
            content = ""
        token_diagnostics = (
            effective_prompt_diagnostics.get("token_diagnostics")
            if isinstance(effective_prompt_diagnostics.get("token_diagnostics"), dict)
            else {}
        )
        case_summaries.append(
            {
                "name": name,
                "route": case.get("route"),
                "exact": exact,
                "normalized_exact": case.get("normalized_exact"),
                "assistant_suffix_kind": suffix,
                "requested_assistant_suffix_kind": requested_suffix,
                "dsv4_policy_reason": effective_prompt_diagnostics.get(
                    "dsv4_policy_reason"
                ),
                "prompt_endswith_assistant_think_open": (
                    effective_prompt_diagnostics.get(
                        "prompt_endswith_assistant_think_open"
                    )
                ),
                "prompt_endswith_assistant_think_close": (
                    effective_prompt_diagnostics.get(
                        "prompt_endswith_assistant_think_close"
                    )
                ),
                "missing_identifiers": missing,
                "corrupt_patterns": corrupt_patterns,
                "has_markdown_fence": case.get("has_markdown_fence"),
                "finish_reason": case.get("finish") or case.get("finish_reason"),
                "prompt_tail": prompt_tail[-500:],
                "content": content[:1000],
                "token_tail_ids": token_diagnostics.get("tail_ids"),
                "token_count": token_diagnostics.get("token_count"),
            }
        )

    dry_run_suffixes: dict[str, str] = {}
    dry_run_requested_suffixes: dict[str, str] = {}
    dry_run_effective_suffixes: dict[str, str] = {}
    dry_run_overrides: dict[str, Any] = {}
    for case in dry_run_cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or "")
        if not name:
            continue
        requested_diagnostics = (
            case.get("requested_prompt_diagnostics")
            if isinstance(case.get("requested_prompt_diagnostics"), dict)
            else case.get("prompt_diagnostics")
            if isinstance(case.get("prompt_diagnostics"), dict)
            else {}
        )
        effective_diagnostics = (
            case.get("effective_prompt_diagnostics")
            if isinstance(case.get("effective_prompt_diagnostics"), dict)
            else case.get("prompt_diagnostics")
            if isinstance(case.get("prompt_diagnostics"), dict)
            else {}
        )
        requested_suffix = str(
            requested_diagnostics.get("assistant_suffix_kind") or ""
        )
        effective_suffix = str(
            effective_diagnostics.get("assistant_suffix_kind") or ""
        )
        dry_run_suffixes[name] = effective_suffix
        dry_run_requested_suffixes[name] = requested_suffix
        dry_run_effective_suffixes[name] = effective_suffix
        dry_run_overrides[name] = case.get("request_overrides") or {}

    return {
        "artifact": artifact_rel,
        "present": path_present,
        "status": artifact.get("status") if path_present else "missing",
        "case_count": len(case_summaries),
        "failed_cases": failed_cases,
        "failed_thinking_closed_cases": failed_thinking_closed_cases,
        "requested_thinking_closed_but_effective_open_cases": (
            requested_thinking_closed_but_effective_open_cases
        ),
        "thinking_open_cases_without_identifier_corruption": (
            thinking_open_cases_without_identifier_corruption
        ),
        "rep1_still_corrupt_patterns": rep1_still_corrupt_patterns,
        "case_summaries": case_summaries,
        "dry_run_artifact": dry_run_rel,
        "dry_run_present": dry_run_present,
        "dry_run_status": dry_run.get("status") if dry_run_present else "missing",
        "dry_run_case_count": dry_run.get("case_count") if dry_run_present else None,
        "dry_run_suffixes": dry_run_suffixes,
        "dry_run_requested_suffixes": dry_run_requested_suffixes,
        "dry_run_effective_suffixes": dry_run_effective_suffixes,
        "dry_run_request_overrides": dry_run_overrides,
        "env_overrides": (
            artifact.get("env_overrides")
            if isinstance(artifact.get("env_overrides"), dict)
            else {}
        ),
    }


def _dsv4_failed_quality_gates(details: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for gate, required in (
        ("current_installed_prompt_rail_exactness_probe", True),
        ("current_generated_only_direct_rail_exactness_subset", True),
        ("current_rep1_direct_vs_requested_thinking_exactness_subset", True),
        ("current_source_rep1_direct_vs_requested_thinking_exactness_subset", True),
        (
            "current_source_token_tail_direct_vs_requested_thinking_exactness_subset",
            True,
        ),
        ("current_source_rep1_direct_only_exactness_subset", True),
        ("current_source_bundle_defaults_exactness_subset", True),
        ("current_jangtqk_direct_off_recheck", False),
    ):
        detail = details.get(gate)
        if not isinstance(detail, dict):
            continue
        status = detail.get("status")
        if not required and status == "missing" and detail.get("present") is False:
            continue
        if status == "pass":
            continue
        if status is None and detail.get("present") is True:
            continue
        failed_cases = [
            str(case) for case in detail.get("failed_cases", []) if str(case)
        ]
        item: dict[str, Any] = {
            "gate": gate,
            "artifact": detail.get("artifact"),
            "status": status,
        }
        if failed_cases:
            item["failed_cases"] = failed_cases
        failed.append(item)
    long_context_status = details.get("long_context_status")
    if long_context_status not in (None, "pass"):
        failed.append(
            {
                "gate": "long_context",
                "status": long_context_status,
            }
        )
    batch_divergence = details.get("current_batch_generator_logit_divergence")
    if isinstance(batch_divergence, dict) and (
        batch_divergence.get("direct_vs_batch_THREE_P_diverges") is True
        or batch_divergence.get("batch_decoded_contains_corrupt_pertive") is True
    ):
        failed.append(
            {
                "gate": "current_batch_generator_logit_divergence",
                "artifact": batch_divergence.get("batch_artifact"),
                "status": "fail",
            }
        )
    return failed


def _dsv4_direct_off_exactness_boundary(details: dict[str, Any]) -> dict[str, Any]:
    token_tail = details.get(
        "current_source_token_tail_direct_vs_requested_thinking_exactness_subset"
    )
    if not isinstance(token_tail, dict):
        return {
            "present": False,
            "root_boundary": (
                "Current direct/off versus requested-thinking DSV4 exactness "
                "boundary is missing."
            ),
        }

    direct_off_gate_names = (
        "current_installed_prompt_rail_exactness_probe",
        "current_generated_only_direct_rail_exactness_subset",
        "current_rep1_direct_vs_requested_thinking_exactness_subset",
        "current_source_rep1_direct_vs_requested_thinking_exactness_subset",
        "current_source_token_tail_direct_vs_requested_thinking_exactness_subset",
        "current_source_rep1_direct_only_exactness_subset",
        "current_source_bundle_defaults_exactness_subset",
        "current_jangtqk_direct_off_recheck",
    )
    requested_thinking_gate_names = (
        "current_requested_thinking_exactness_subset",
        "current_rep1_direct_vs_requested_thinking_exactness_subset",
        "current_source_rep1_direct_vs_requested_thinking_exactness_subset",
        "current_source_token_tail_direct_vs_requested_thinking_exactness_subset",
    )
    direct_off_failed_routes: set[str] = set()
    direct_off_failed_cases_all: list[str] = []
    direct_off_failure_artifacts: list[str] = []
    requested_thinking_exact_routes: set[str] = set()
    requested_thinking_exact_cases_all: list[str] = []

    for gate_name in direct_off_gate_names:
        gate = details.get(gate_name)
        if not isinstance(gate, dict):
            continue
        gate_failed = False
        for item in gate.get("case_summaries", []):
            if not isinstance(item, dict):
                continue
            suffix = item.get("assistant_suffix_kind")
            exact = item.get("exact") is True
            route = item.get("route")
            name = item.get("name")
            if suffix == "thinking_closed" and not exact:
                if isinstance(route, str) and route:
                    direct_off_failed_routes.add(route)
                if isinstance(name, str) and name:
                    direct_off_failed_cases_all.append(name)
                gate_failed = True
        artifact = gate.get("artifact")
        if gate_failed and isinstance(artifact, str) and artifact:
            direct_off_failure_artifacts.append(artifact)

    for gate_name in requested_thinking_gate_names:
        gate = details.get(gate_name)
        if not isinstance(gate, dict):
            continue
        for item in gate.get("case_summaries", []):
            if not isinstance(item, dict):
                continue
            suffix = item.get("assistant_suffix_kind")
            exact = item.get("exact") is True
            route = item.get("route")
            name = item.get("name")
            missing = item.get("missing_identifiers") or []
            corrupt = item.get("corrupt_patterns") or []
            if suffix == "thinking_open" and exact and not missing and not corrupt:
                if isinstance(route, str) and route:
                    requested_thinking_exact_routes.add(route)
                if isinstance(name, str) and name:
                    requested_thinking_exact_cases_all.append(name)

    case_summaries = [
        item for item in token_tail.get("case_summaries", []) if isinstance(item, dict)
    ]
    direct_off_failing = [
        str(item.get("name"))
        for item in case_summaries
        if item.get("assistant_suffix_kind") == "thinking_closed"
        and item.get("exact") is False
    ]
    requested_thinking_exact = [
        str(item.get("name"))
        for item in case_summaries
        if item.get("assistant_suffix_kind") == "thinking_open"
        and item.get("exact") is True
        and not item.get("corrupt_patterns")
        and not item.get("missing_identifiers")
    ]
    direct_suffixes = {
        str(item.get("assistant_suffix_kind"))
        for item in case_summaries
        if str(item.get("name", "")).endswith("_off_rep1")
    }
    thinking_suffixes = {
        str(item.get("assistant_suffix_kind"))
        for item in case_summaries
        if str(item.get("name", "")).endswith("_on_rep1")
    }
    direct_off_blocker_active = bool(direct_off_failed_routes)
    requested_thinking_success_seen = bool(requested_thinking_exact_routes)

    return {
        "present": bool(case_summaries),
        "artifact": token_tail.get("artifact"),
        "direct_off_failing_cases": direct_off_failing,
        "requested_thinking_exact_cases": requested_thinking_exact,
        "direct_off_suffix_kind": (
            next(iter(direct_suffixes)) if len(direct_suffixes) == 1 else None
        ),
        "requested_thinking_suffix_kind": (
            next(iter(thinking_suffixes)) if len(thinking_suffixes) == 1 else None
        ),
        "all_direct_off_failed_cases": sorted(set(direct_off_failed_cases_all)),
        "all_direct_off_failed_routes": sorted(direct_off_failed_routes),
        "direct_off_failure_artifacts": sorted(set(direct_off_failure_artifacts)),
        "all_requested_thinking_exact_cases": sorted(
            set(requested_thinking_exact_cases_all)
        ),
        "requested_thinking_exact_routes": sorted(requested_thinking_exact_routes),
        "direct_off_failure_spans_chat_responses_and_completion": {
            "chat",
            "responses",
            "completion",
        }.issubset(direct_off_failed_routes),
        "requested_thinking_exact_spans_chat_and_responses": {
            "chat",
            "responses",
        }.issubset(requested_thinking_exact_routes),
        "direct_off_release_blocker_active": direct_off_blocker_active,
        "requested_thinking_success_clears_direct_off": False,
        "requested_thinking_success_is_diagnostic_only": (
            direct_off_blocker_active and requested_thinking_success_seen
        ),
        "hidden_force_on_would_be_false_clearance": bool(
            direct_off_failing and requested_thinking_exact
        ),
        "root_boundary": (
            "Direct/off DSV4 exact-code generation fails under thinking_closed while "
            "requested-thinking exactness passes under thinking_open; hidden force-on "
            "does not clear explicit direct/off quality."
        ),
    }


def _dsv4_source_memory_preflight_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    telemetry = artifact.get("telemetry")
    if not isinstance(telemetry, list):
        telemetry = []
    memory: dict[str, Any] = {}
    for item in telemetry:
        if not isinstance(item, dict):
            continue
        if item.get("name") != "preflight":
            continue
        system_memory = item.get("system_memory")
        if isinstance(system_memory, dict):
            memory = system_memory
        break
    available_gb = artifact.get("available_gb", memory.get("available_gb"))
    required_available_gb = artifact.get("required_available_gb")
    memory_gap_gb = artifact.get("memory_gap_gb")
    if (
        memory_gap_gb is None
        and isinstance(available_gb, (int, float))
        and isinstance(required_available_gb, (int, float))
    ):
        memory_gap_gb = round(max(0.0, required_available_gb - available_gb), 2)
    did_not_launch = artifact.get("did_not_launch")
    if did_not_launch is None and artifact.get("status") == "skipped":
        did_not_launch = artifact.get("reason") in {
            "insufficient_free_memory",
            "insufficient_vm_stat_memory",
        }
    launch_decision = artifact.get("launch_decision")
    if launch_decision is None and did_not_launch is True:
        launch_decision = "do_not_launch"
    return {
        "artifact": DSV4_CURRENT_SOURCE_MEMORY_PREFLIGHT_REL,
        "artifact_present": _path_present(
            root,
            DSV4_CURRENT_SOURCE_MEMORY_PREFLIGHT_REL,
        ),
        "created_at": artifact.get("created_at"),
        "status": artifact.get("status"),
        "reason": artifact.get("reason"),
        "model": artifact.get("model"),
        "selected_cases": artifact.get("selected_cases"),
        "case_count": artifact.get("case_count"),
        "available_gb": available_gb,
        "total_gb": memory.get("total_gb"),
        "required_available_gb": required_available_gb,
        "required_free_gb": artifact.get("required_free_gb"),
        "min_free_gb": artifact.get("min_free_gb"),
        "available_for_gate_gb": artifact.get("available_for_gate_gb"),
        "required_model_margin_gb": artifact.get("required_model_margin_gb"),
        "model_size_gb": artifact.get("model_size_gb"),
        "safety_margin_gb": artifact.get("safety_margin_gb"),
        "floor_valid": artifact.get("floor_valid"),
        "memory_gap_gb": memory_gap_gb,
        "strict_vm_stat_memory_gap_gb": artifact.get(
            "strict_vm_stat_memory_gap_gb"
        ),
        "psutil_available_gap_gb": artifact.get("psutil_available_gap_gb"),
        "memory_pressure_free_percent": artifact.get(
            "memory_pressure_free_percent"
        ),
        "memory_pressure_error": artifact.get("memory_pressure_error"),
        "preflight_memory_source": artifact.get("preflight_memory_source"),
        "free_plus_speculative_purgeable_gb": artifact.get(
            "free_plus_speculative_purgeable_gb"
        ),
        "launch_blockers": artifact.get("launch_blockers"),
        "active_heavy_processes": artifact.get("active_heavy_processes"),
        "active_heavy_process_count": artifact.get("active_heavy_process_count"),
        "top_memory_processes": artifact.get("top_memory_processes"),
        "commands": artifact.get("commands"),
        "did_not_launch": did_not_launch,
        "launch_decision": launch_decision,
        "launch_allowed": artifact.get("launch_allowed"),
    }


def _dsv4_exact_code_root_boundary(details: dict[str, Any]) -> dict[str, Any]:
    direct = details.get("direct_off_exactness_boundary")
    if not isinstance(direct, dict):
        direct = {}
    template = details.get("current_template_parity_diagnostic")
    if not isinstance(template, dict):
        template = {}
    hidden = details.get("current_hidden_reasoning_control_probe")
    if not isinstance(hidden, dict):
        hidden = {}
    prefill = details.get("current_prefill_execution_variant_logits")
    if not isinstance(prefill, dict):
        prefill = {}
    prompt_variant = details.get("current_prompt_variant_logit_probe")
    if not isinstance(prompt_variant, dict):
        prompt_variant = {}
    prompt_boundary = details.get("current_prompt_boundary_bisection_probe")
    if not isinstance(prompt_boundary, dict):
        prompt_boundary = {}
    webgl_logit = details.get("current_direct_vs_thinking_webgl_logit_probe")
    if not isinstance(webgl_logit, dict):
        webgl_logit = {}
    batch = details.get("current_batch_generator_logit_divergence")
    if not isinstance(batch, dict):
        batch = {}
    source_rep1 = details.get("current_source_rep1_direct_only_exactness_subset")
    if not isinstance(source_rep1, dict):
        source_rep1 = {}
    bundle_defaults = details.get("current_source_bundle_defaults_exactness_subset")
    if not isinstance(bundle_defaults, dict):
        bundle_defaults = {}
    source_preflight = details.get("current_source_full_output_preflight")
    if not isinstance(source_preflight, dict):
        source_preflight = {}
    jangtqk_direct_off = details.get("current_jangtqk_direct_off_recheck")
    if not isinstance(jangtqk_direct_off, dict):
        jangtqk_direct_off = {}

    failed_rep1_cases = [
        str(name)
        for name in source_rep1.get("failed_thinking_closed_cases", [])
        if isinstance(name, str)
    ]
    dry_run_overrides = source_rep1.get("dry_run_request_overrides")
    if not isinstance(dry_run_overrides, dict):
        dry_run_overrides = {}

    def _failed_rep1_overrides_match(expected: dict[str, Any]) -> bool:
        if not failed_rep1_cases:
            return False
        for name in failed_rep1_cases:
            overrides = dry_run_overrides.get(name)
            if not isinstance(overrides, dict):
                return False
            for key, value in expected.items():
                if overrides.get(key) != value:
                    return False
        return True

    direct_off_route_wide_failure = (
        direct.get("direct_off_failure_spans_chat_responses_and_completion") is True
    )
    requested_thinking_is_diagnostic_only = (
        direct.get("requested_thinking_success_is_diagnostic_only") is True
        and direct.get("requested_thinking_success_clears_direct_off") is False
    )
    template_mismatch_ruled_out = template.get("template_mismatch_ruled_out") is True
    hidden_reasoning_not_sufficient = (
        hidden.get("hidden_reasoning_not_sufficient_root_cause") is True
    )
    prefix_cache_not_sufficient = _failed_rep1_overrides_match(
        {"skip_prefix_cache": True}
    )
    forced_sampler_not_sufficient = _failed_rep1_overrides_match(
        {
            "temperature": 0,
            "top_p": 1,
            "repetition_penalty": 1.0,
        }
    )
    bundle_defaults_do_not_clear = (
        bundle_defaults.get("status") == "fail"
        and bool(bundle_defaults.get("failed_thinking_closed_cases"))
    )
    jangtqk_direct_off_still_fails = (
        jangtqk_direct_off.get("status") == "fail"
        and sorted(jangtqk_direct_off.get("failed_thinking_closed_cases") or [])
        == ["chat_off_rep1", "responses_off_rep1"]
    )
    batch_intrinsic_failure_not_proven = (
        batch.get("batch_generator_intrinsic_failure_not_proven") is True
    )
    stream_warmup_ruled_out = (
        prefill.get("stream_warmup_state_ruled_out_for_isolated_prefix") is True
    )
    prompt_wording_changes_logits = (
        prompt_variant.get("prompt_wording_changes_identifier_logits") is True
    )
    direct_after_three_dot_web_rank = webgl_logit.get(
        "direct_after_three_dot_web_rank"
    )
    thinking_after_three_dot_web_rank = webgl_logit.get(
        "thinking_after_three_dot_web_rank"
    )
    direct_off_three_dot_identifier_boundary = (
        isinstance(direct_after_three_dot_web_rank, int)
        and isinstance(thinking_after_three_dot_web_rank, int)
        and direct_after_three_dot_web_rank != 1
        and thinking_after_three_dot_web_rank == 1
    )
    direct_off_identifier_logit_divergence = {
        "webweb_not_explained_by_after_web_rank": webgl_logit.get(
            "webweb_not_explained_by_after_web_rank"
        ),
        "direct_after_three_dot_web_rank": direct_after_three_dot_web_rank,
        "thinking_after_three_dot_web_rank": thinking_after_three_dot_web_rank,
        "direct_after_three_dot_top_token": webgl_logit.get(
            "direct_after_three_dot_top_token"
        ),
        "thinking_after_three_dot_top_token": webgl_logit.get(
            "thinking_after_three_dot_top_token"
        ),
        "direct_after_camera_p_top_token": webgl_logit.get(
            "direct_after_camera_p_top_token"
        ),
        "thinking_after_camera_p_top_token": webgl_logit.get(
            "thinking_after_camera_p_top_token"
        ),
    }
    direct_off_prompt_boundary_is_wording_sensitive = (
        prompt_boundary.get("canonical_colon_passes") is True
        and prompt_boundary.get("no_punct_after_fences_still_passes") is True
        and prompt_boundary.get("period_after_fences_breaks_exactness") is True
        and prompt_boundary.get("same_effective_rail") is True
    )
    prompt_boundary_bisection = {
        "canonical_colon_passes": prompt_boundary.get("canonical_colon_passes"),
        "no_punct_after_fences_still_passes": prompt_boundary.get(
            "no_punct_after_fences_still_passes"
        ),
        "period_after_fences_breaks_exactness": prompt_boundary.get(
            "period_after_fences_breaks_exactness"
        ),
        "same_effective_rail": prompt_boundary.get("same_effective_rail"),
        "passing_cases": prompt_boundary.get("passing_cases"),
        "failed_cases": prompt_boundary.get("failed_cases"),
        "effective_rails": prompt_boundary.get("effective_rails"),
        "all_effective_policy_reasons": prompt_boundary.get(
            "all_effective_policy_reasons"
        ),
    }
    prompt_boundary_passes_are_effective_thinking_open = (
        prompt_boundary.get("passing_cases_effective_thinking_open") is True
    )
    prompt_boundary_requested_off_but_effective_force_on = (
        prompt_boundary.get("passing_cases_requested_off_but_effective_force_on")
        is True
    )
    prompt_boundary_passes_clear_direct_off = (
        prompt_boundary_passes_are_effective_thinking_open is False
        and direct_off_route_wide_failure is False
    )

    ruled_out = []
    if template_mismatch_ruled_out:
        ruled_out.append("template_mismatch")
    if hidden_reasoning_not_sufficient:
        ruled_out.append("hidden_reasoning_corruption")
    if prefix_cache_not_sufficient:
        ruled_out.append("prefix_cache")
    if forced_sampler_not_sufficient:
        ruled_out.append("forced_sampling_controls")
    if bundle_defaults_do_not_clear:
        ruled_out.append("bundle_defaults")
    if jangtqk_direct_off_still_fails:
        ruled_out.append("true_bundled_jangtqk_direct_off_recheck")
    if stream_warmup_ruled_out:
        ruled_out.append("stream_warmup_state")
    if batch_intrinsic_failure_not_proven:
        ruled_out.append("batch_generator_warmup_intrinsic")

    current_primary_failure = (
        "direct_off_exact_code_generation"
        if direct_off_route_wide_failure
        else "insufficient_evidence"
    )

    return {
        "present": bool(direct) or bool(template) or bool(hidden),
        "direct_off_route_wide_failure": direct_off_route_wide_failure,
        "requested_thinking_is_diagnostic_only": requested_thinking_is_diagnostic_only,
        "template_mismatch_ruled_out": template_mismatch_ruled_out,
        "hidden_reasoning_not_sufficient_root_cause": (
            hidden_reasoning_not_sufficient
        ),
        "prefix_cache_not_sufficient_root_cause": prefix_cache_not_sufficient,
        "forced_sampler_controls_not_sufficient_root_cause": (
            forced_sampler_not_sufficient
        ),
        "bundle_defaults_do_not_clear_direct_off": bundle_defaults_do_not_clear,
        "true_bundled_jangtqk_direct_off_still_fails": (
            jangtqk_direct_off_still_fails
        ),
        "true_bundled_jangtqk_direct_off_recheck": jangtqk_direct_off,
        "batch_generator_intrinsic_failure_not_proven": (
            batch_intrinsic_failure_not_proven
        ),
        "stream_warmup_state_ruled_out_for_isolated_prefix": stream_warmup_ruled_out,
        "prompt_wording_changes_identifier_logits": prompt_wording_changes_logits,
        "direct_off_identifier_logit_divergence": (
            direct_off_identifier_logit_divergence
        ),
        "direct_off_three_dot_identifier_boundary": (
            direct_off_three_dot_identifier_boundary
        ),
        "prompt_boundary_bisection": prompt_boundary_bisection,
        "direct_off_prompt_boundary_is_wording_sensitive": (
            direct_off_prompt_boundary_is_wording_sensitive
        ),
        "prompt_boundary_passes_are_effective_thinking_open": (
            prompt_boundary_passes_are_effective_thinking_open
        ),
        "prompt_boundary_requested_off_but_effective_force_on": (
            prompt_boundary_requested_off_but_effective_force_on
        ),
        "prompt_boundary_passes_clear_direct_off": (
            prompt_boundary_passes_clear_direct_off
        ),
        "ruled_out_root_causes": ruled_out,
        "current_primary_failure": current_primary_failure,
        "source_full_output_preflight": source_preflight,
        "source_full_output_clearance_missing": (
            source_preflight.get("status") != "pass"
        ),
        "root_boundary": (
            "Current DSV4 exact-code evidence points at explicit direct/off "
            "rail exact-code generation, not template mismatch, hidden "
            "reasoning leakage, prefix cache, forced sampling controls, or "
            "bundle defaults. Requested-thinking success is diagnostic only; "
            "the current logit boundary is at THREE. / identifier continuation, "
            "and live prompt-boundary bisection is same-rail wording-sensitive "
            "but its passing rows are not direct/off clearance when they run "
            "on the effective thinking-open rail."
        ),
    }


def _dsv4_chatmax_prompt_trigger_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_CHATMAX_PROMPT_TRIGGER_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    case_summaries: list[dict[str, Any]] = []
    canonical_exact_cases: list[str] = []
    failed_cases: list[str] = []
    blank_visible_cases: list[str] = []
    effective_suffixes: set[str] = set()
    effective_reasons: set[str] = set()
    corrupt_patterns_by_case: dict[str, list[str]] = {}

    for case in cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or "")
        content = case.get("content")
        if not isinstance(content, str):
            content = ""
        effective_prompt_diagnostics = (
            case.get("effective_prompt_diagnostics")
            if isinstance(case.get("effective_prompt_diagnostics"), dict)
            else {}
        )
        suffix = str(effective_prompt_diagnostics.get("assistant_suffix_kind") or "")
        reason = str(effective_prompt_diagnostics.get("dsv4_policy_reason") or "")
        if suffix:
            effective_suffixes.add(suffix)
        if reason:
            effective_reasons.add(reason)

        missing = [str(item) for item in (case.get("missing") or [])]
        corrupt_patterns = [
            str(item) for item in (case.get("corrupt_patterns") or [])
        ]
        exact = case.get("exact") is True
        if exact and name:
            canonical_exact_cases.append(name)
        elif name:
            failed_cases.append(name)
        if name and not content:
            blank_visible_cases.append(name)
        if name and corrupt_patterns:
            corrupt_patterns_by_case[name] = corrupt_patterns

        prompt_tail = effective_prompt_diagnostics.get("prompt_tail")
        if not isinstance(prompt_tail, str):
            prompt_tail = ""
        case_summaries.append(
            {
                "name": name,
                "exact": exact,
                "normalized_exact": case.get("normalized_exact"),
                "assistant_suffix_kind": suffix,
                "dsv4_policy_reason": reason,
                "missing_identifiers": missing,
                "corrupt_patterns": corrupt_patterns,
                "has_markdown_fence": case.get("has_markdown_fence"),
                "prompt_tokens": case.get("prompt_tokens"),
                "completion_tokens": case.get("completion_tokens"),
                "blank_visible_content": not content,
                "content": content[:500],
                "prompt_tail": prompt_tail[-500:],
            }
        )

    return {
        "artifact": DSV4_CHATMAX_PROMPT_TRIGGER_REL,
        "present": path_present,
        "status": artifact.get("status") if path_present else "missing",
        "case_count": len(case_summaries),
        "canonical_exact_cases": canonical_exact_cases,
        "failed_cases": failed_cases,
        "blank_visible_cases": blank_visible_cases,
        "same_effective_rail": len(effective_suffixes) == 1 and bool(effective_suffixes),
        "effective_rails": sorted(effective_suffixes),
        "all_effective_policy_reasons": sorted(effective_reasons),
        "corrupt_patterns_by_case": corrupt_patterns_by_case,
        "case_summaries": case_summaries,
    }


def _dsv4_chatmax_budget_stop_rail_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_CHATMAX_BUDGET_STOP_RAIL_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    case_summaries: list[dict[str, Any]] = []
    failed_cases: list[str] = []
    completion_tokens_for_failed: set[int] = set()
    effective_rails: set[str] = set()
    effective_reasons: set[str] = set()

    for case in cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or "")
        route = str(case.get("route") or "")
        exact = case.get("exact") is True
        completion_tokens = case.get("completion_tokens")
        if not exact and name:
            failed_cases.append(name)
            if isinstance(completion_tokens, int):
                completion_tokens_for_failed.add(completion_tokens)

        diagnostics = (
            case.get("effective_prompt_diagnostics")
            if isinstance(case.get("effective_prompt_diagnostics"), dict)
            else {}
        )
        suffix = str(diagnostics.get("assistant_suffix_kind") or "")
        reason = str(diagnostics.get("dsv4_policy_reason") or "")
        if suffix:
            effective_rails.add(suffix)
        if reason:
            effective_reasons.add(reason)

        content = case.get("content")
        if not isinstance(content, str):
            content = ""
        case_summaries.append(
            {
                "name": name,
                "route": route,
                "exact": exact,
                "normalized_exact": case.get("normalized_exact"),
                "finish_reason": case.get("finish_reason") or case.get("finish"),
                "completion_tokens": completion_tokens,
                "missing_identifiers": case.get("missing") or [],
                "corrupt_patterns": case.get("corrupt_patterns") or [],
                "assistant_suffix_kind": suffix,
                "dsv4_policy_reason": reason,
                "content": content[:500],
            }
        )

    return {
        "artifact": DSV4_CHATMAX_BUDGET_STOP_RAIL_REL,
        "present": path_present,
        "status": artifact.get("status") if path_present else "missing",
        "case_count": len(case_summaries),
        "failed_cases": failed_cases,
        "all_failed_cases_same_completion_tokens": (
            len(completion_tokens_for_failed) == 1
            and bool(completion_tokens_for_failed)
        ),
        "failed_completion_tokens": sorted(completion_tokens_for_failed),
        "larger_budget_ruled_out": "chatmax_1024_budget" in failed_cases,
        "role_stops_ruled_out": "chatmax_1024_stop_roles" in failed_cases,
        "thinking_on_ruled_out": "chatmax_1024_thinking_on" in failed_cases,
        "effort_low_ruled_out": "chatmax_1024_effort_low" in failed_cases,
        "responses_route_also_fails": "responses_chatmax_1024" in failed_cases,
        "completion_route_also_fails": "completion_chatmax_512" in failed_cases,
        "effective_rails": sorted(effective_rails),
        "all_effective_policy_reasons": sorted(effective_reasons),
        "case_summaries": case_summaries,
    }


def _dsv4_prompt_boundary_bisection_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_PROMPT_BOUNDARY_BISECTION_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    passing_cases: list[str] = []
    failed_cases: list[str] = []
    length_failed_cases: list[str] = []
    blank_visible_cases: list[str] = []
    effective_rails: set[str] = set()
    effective_reasons: set[str] = set()
    passing_effective_rails: set[str] = set()
    passing_effective_reasons: set[str] = set()
    passing_requested_enable_thinking: set[bool] = set()
    case_summaries: list[dict[str, Any]] = []

    for case in cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or "")
        exact = case.get("exact") is True
        if exact and name:
            passing_cases.append(name)
        elif name:
            failed_cases.append(name)

        finish_reason = str(case.get("finish_reason") or case.get("finish") or "")
        if name and not exact and finish_reason == "length":
            length_failed_cases.append(name)

        diagnostics = (
            case.get("effective_prompt_diagnostics")
            if isinstance(case.get("effective_prompt_diagnostics"), dict)
            else {}
        )
        requested_diagnostics = (
            case.get("requested_prompt_diagnostics")
            if isinstance(case.get("requested_prompt_diagnostics"), dict)
            else {}
        )
        suffix = str(diagnostics.get("assistant_suffix_kind") or "")
        reason = str(diagnostics.get("dsv4_policy_reason") or "")
        if suffix:
            effective_rails.add(suffix)
        if reason:
            effective_reasons.add(reason)
        if exact and suffix:
            passing_effective_rails.add(suffix)
        if exact and reason:
            passing_effective_reasons.add(reason)
        requested_enable = requested_diagnostics.get("enable_thinking")
        if exact and isinstance(requested_enable, bool):
            passing_requested_enable_thinking.add(requested_enable)

        content = case.get("content")
        if not isinstance(content, str):
            content = ""
        if name and not content:
            blank_visible_cases.append(name)

        case_summaries.append(
            {
                "name": name,
                "exact": exact,
                "normalized_exact": case.get("normalized_exact"),
                "finish_reason": finish_reason,
                "completion_tokens": case.get("completion_tokens"),
                "missing_identifiers": case.get("missing") or [],
                "corrupt_patterns": case.get("corrupt_patterns") or [],
                "assistant_suffix_kind": suffix,
                "dsv4_policy_reason": reason,
                "requested_enable_thinking": requested_enable,
                "requested_assistant_suffix_kind": requested_diagnostics.get(
                    "assistant_suffix_kind"
                ),
                "blank_visible_content": not content,
                "content": content[:500],
            }
        )

    passing_cases_effective_thinking_open = bool(passing_cases) and (
        passing_effective_rails == {"thinking_open"}
    )
    passing_cases_requested_thinking_disabled = bool(passing_cases) and (
        passing_requested_enable_thinking == {False}
    )
    return {
        "artifact": DSV4_PROMPT_BOUNDARY_BISECTION_REL,
        "present": path_present,
        "status": artifact.get("status") if path_present else "missing",
        "case_count": len(case_summaries),
        "passing_cases": passing_cases,
        "failed_cases": failed_cases,
        "length_failed_cases": length_failed_cases,
        "blank_visible_cases": blank_visible_cases,
        "period_after_fences_breaks_exactness": "return_fences_period" in failed_cases,
        "no_punct_after_fences_still_passes": "return_fences_no_punct" in passing_cases,
        "canonical_colon_passes": "canonical_return_fences_colon" in passing_cases,
        "same_effective_rail": len(effective_rails) == 1 and bool(effective_rails),
        "effective_rails": sorted(effective_rails),
        "all_effective_policy_reasons": sorted(effective_reasons),
        "passing_effective_rails": sorted(passing_effective_rails),
        "passing_effective_policy_reasons": sorted(passing_effective_reasons),
        "passing_requested_enable_thinking": sorted(
            passing_requested_enable_thinking
        ),
        "passing_cases_effective_thinking_open": (
            passing_cases_effective_thinking_open
        ),
        "passing_cases_requested_thinking_disabled": (
            passing_cases_requested_thinking_disabled
        ),
        "passing_cases_requested_off_but_effective_force_on": (
            passing_cases_effective_thinking_open
            and passing_cases_requested_thinking_disabled
            and "dsv4_direct_rail_identifier_unsafe" in passing_effective_reasons
        ),
        "case_summaries": case_summaries,
    }


def _dsv4_colon_period_logprob_trace_detail(
    raw_artifact: dict[str, Any],
    visible_artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    raw_present = _path_present(root, DSV4_COLON_PERIOD_LOGPROB_TRACE_REL)
    visible_present = _path_present(
        root, DSV4_COLON_PERIOD_VISIBLE_LOGPROB_TRACE_REL
    )

    def _cases_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        cases = payload.get("cases") if isinstance(payload.get("cases"), list) else []
        return {
            str(case.get("name") or ""): case
            for case in cases
            if isinstance(case, dict) and case.get("name")
        }

    raw_cases = _cases_by_name(raw_artifact)
    visible_cases = _cases_by_name(visible_artifact)
    colon_visible = visible_cases.get("colon_pass", {})
    period_visible = visible_cases.get("period_fail", {})
    colon_raw = raw_cases.get("colon_pass", {})
    period_raw = raw_cases.get("period_fail", {})

    prompt_tokens = [
        case.get("prompt_tokens")
        for case in (colon_visible, period_visible)
        if isinstance(case.get("prompt_tokens"), int)
    ]
    completion_tokens = [
        case.get("completion_tokens")
        for case in (colon_visible, period_visible)
        if isinstance(case.get("completion_tokens"), int)
    ]
    logprob_entry_counts = [
        case.get("logprob_entry_count")
        for case in (colon_visible, period_visible)
        if isinstance(case.get("logprob_entry_count"), int)
    ]

    cache_hit_values: list[int] = []
    for payload in (raw_artifact, visible_artifact):
        for key in ("health_before", "health_after"):
            health = payload.get(key) if isinstance(payload.get(key), dict) else {}
            scheduler = (
                health.get("scheduler") if isinstance(health.get("scheduler"), dict) else {}
            )
            value = scheduler.get("cache_hit_tokens")
            if isinstance(value, int):
                cache_hit_values.append(value)

    colon_joined = str(colon_visible.get("joined_logprob_text_prefix") or "")
    period_context = (
        period_visible.get("visible_diff_token_context")
        if isinstance(period_visible.get("visible_diff_token_context"), list)
        else []
    )
    period_context_summary = []
    for entry in period_context[:5]:
        if not isinstance(entry, dict):
            continue
        top_logprobs = (
            entry.get("top_logprobs")
            if isinstance(entry.get("top_logprobs"), list)
            else []
        )
        period_context_summary.append(
            {
                "index": entry.get("index"),
                "token": entry.get("token"),
                "logprob": entry.get("logprob"),
                "top_tokens": [
                    item.get("token")
                    for item in top_logprobs[:5]
                    if isinstance(item, dict)
                ],
            }
        )

    return {
        "raw_artifact": DSV4_COLON_PERIOD_LOGPROB_TRACE_REL,
        "raw_present": raw_present,
        "raw_status": raw_artifact.get("status") if raw_present else "missing",
        "visible_artifact": DSV4_COLON_PERIOD_VISIBLE_LOGPROB_TRACE_REL,
        "visible_present": visible_present,
        "visible_status": (
            visible_artifact.get("status") if visible_present else "missing"
        ),
        "same_prompt_tokens": len(set(prompt_tokens)) == 1 and bool(prompt_tokens),
        "same_completion_tokens": (
            len(set(completion_tokens)) == 1 and bool(completion_tokens)
        ),
        "same_logprob_entry_count": (
            len(set(logprob_entry_counts)) == 1 and bool(logprob_entry_counts)
        ),
        "prompt_tokens": sorted(set(prompt_tokens)),
        "completion_tokens": sorted(set(completion_tokens)),
        "logprob_entry_counts": sorted(set(logprob_entry_counts)),
        "cache_hit_tokens_zero": bool(cache_hit_values)
        and all(value == 0 for value in cache_hit_values),
        "colon_exact": colon_visible.get("exact") is True,
        "period_exact": period_visible.get("exact") is True,
        "period_first_char_diff_index": period_raw.get("first_char_diff_index"),
        "period_first_visible_diff_index": period_visible.get(
            "first_visible_diff_index"
        ),
        "period_visible_diff_token_index": period_visible.get(
            "visible_diff_token_index"
        ),
        "period_corrupt_patterns": [
            str(item) for item in (period_visible.get("corrupt_patterns") or [])
        ],
        "period_missing_identifiers": [
            str(item) for item in (period_visible.get("missing") or [])
        ],
        "colon_hidden_thinking_contains_corruption": any(
            marker in colon_joined
            for marker in (
                "WebWebGLRenderer",
                "PPerspectiveCamera",
                "BBoxGeometry",
                "MMeshBasicMaterial",
            )
        ),
        "period_visible_diff_token_context": period_context_summary,
    }


def _rank_of_token(top_logprobs: list[Any], token: str) -> int | None:
    for index, item in enumerate(top_logprobs, start=1):
        if isinstance(item, dict) and item.get("token") == token:
            return index
    return None


def _scene_token_entry(case: dict[str, Any], offset: int = 0) -> dict[str, Any]:
    context = (
        case.get("scene_token_context")
        if isinstance(case.get("scene_token_context"), list)
        else []
    )
    target_index = case.get("scene_token_index")
    if isinstance(target_index, int):
        target_index += offset
    for entry in context:
        if isinstance(entry, dict) and entry.get("index") == target_index:
            return entry
    return {}


def _dsv4_scene_token_rank_contrast_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_SCENE_TOKEN_RANK_CONTRAST_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    by_name = {
        str(case.get("name") or ""): case
        for case in cases
        if isinstance(case, dict) and case.get("name")
    }
    colon = by_name.get("colon_pass", {})
    period = by_name.get("period_fail", {})
    colon_entry = _scene_token_entry(colon, offset=1)
    period_entry = _scene_token_entry(period, offset=1)
    colon_top = (
        colon_entry.get("top_logprobs")
        if isinstance(colon_entry.get("top_logprobs"), list)
        else []
    )
    period_top = (
        period_entry.get("top_logprobs")
        if isinstance(period_entry.get("top_logprobs"), list)
        else []
    )

    prompt_tokens = [
        case.get("prompt_tokens")
        for case in (colon, period)
        if isinstance(case.get("prompt_tokens"), int)
    ]
    completion_tokens = [
        case.get("completion_tokens")
        for case in (colon, period)
        if isinstance(case.get("completion_tokens"), int)
    ]
    cache_hit_values: list[int] = []
    for key in ("health_before", "health_after"):
        health = artifact.get(key) if isinstance(artifact.get(key), dict) else {}
        scheduler = (
            health.get("scheduler") if isinstance(health.get("scheduler"), dict) else {}
        )
        value = scheduler.get("cache_hit_tokens")
        if isinstance(value, int):
            cache_hit_values.append(value)

    return {
        "artifact": DSV4_SCENE_TOKEN_RANK_CONTRAST_REL,
        "present": path_present,
        "status": artifact.get("status") if path_present else "missing",
        "same_prompt_tokens": len(set(prompt_tokens)) == 1 and bool(prompt_tokens),
        "same_completion_tokens": (
            len(set(completion_tokens)) == 1 and bool(completion_tokens)
        ),
        "prompt_tokens": sorted(set(prompt_tokens)),
        "completion_tokens": sorted(set(completion_tokens)),
        "cache_hit_tokens_zero": bool(cache_hit_values)
        and all(value == 0 for value in cache_hit_values),
        "scene_token_index": colon.get("scene_token_index")
        if colon.get("scene_token_index") == period.get("scene_token_index")
        else None,
        "rank_contrast_token_index": colon_entry.get("index")
        if colon_entry.get("index") == period_entry.get("index")
        else None,
        "scene_token_abs_index": colon.get("scene_token_abs_index")
        if colon.get("scene_token_abs_index") == period.get("scene_token_abs_index")
        else None,
        "colon_exact": colon.get("exact") is True,
        "period_exact": period.get("exact") is True,
        "colon_selected_token": colon_entry.get("token"),
        "period_selected_token": period_entry.get("token"),
        "colon_selected_logprob": colon_entry.get("logprob"),
        "period_selected_logprob": period_entry.get("logprob"),
        "colon_correct_ene_rank": _rank_of_token(colon_top, "ene"),
        "colon_wrong_close_rank": _rank_of_token(colon_top, "();\n"),
        "period_correct_ene_rank": _rank_of_token(period_top, "ene"),
        "period_wrong_close_rank": _rank_of_token(period_top, "();\n"),
        "colon_top_tokens": [
            item.get("token") for item in colon_top[:5] if isinstance(item, dict)
        ],
        "period_top_tokens": [
            item.get("token") for item in period_top[:5] if isinstance(item, dict)
        ],
        "period_corrupt_patterns": [
            str(item) for item in (period.get("corrupt_patterns") or [])
        ],
        "period_missing_identifiers": [
            str(item) for item in (period.get("missing") or [])
        ],
    }


def _dsv4_direct_vs_thinking_webgl_logit_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_DIRECT_VS_THINKING_WEBGL_LOGIT_PROBE_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    by_key = {
        (str(case.get("rail") or ""), str(case.get("prefix_name") or "")): case
        for case in cases
        if isinstance(case, dict)
    }

    def rank(rail: str, prefix: str, token: str) -> int | None:
        ranks = by_key.get((rail, prefix), {}).get("ranks")
        if not isinstance(ranks, dict):
            return None
        value = ranks.get(token)
        return value if isinstance(value, int) else None

    def top_token(rail: str, prefix: str) -> str | None:
        top = by_key.get((rail, prefix), {}).get("top")
        if not isinstance(top, list) or not top or not isinstance(top[0], dict):
            return None
        value = top[0].get("token")
        return str(value) if value is not None else None

    direct_after_web_gl = rank("direct_off", "after_const_renderer_web", "GL")
    thinking_after_web_gl = rank(
        "requested_thinking", "after_const_renderer_web", "GL"
    )
    direct_after_three_dot_web = rank(
        "direct_off", "after_const_renderer_three_dot", "Web"
    )
    thinking_after_three_dot_web = rank(
        "requested_thinking", "after_const_renderer_three_dot", "Web"
    )
    direct_after_camera_p_pers = rank("direct_off", "after_camera_p", "Pers")
    direct_after_camera_p_ers = rank("direct_off", "after_camera_p", "ers")
    thinking_after_camera_p_pers = rank(
        "requested_thinking", "after_camera_p", "Pers"
    )
    thinking_after_camera_p_ers = rank(
        "requested_thinking", "after_camera_p", "ers"
    )

    return {
        "artifact": DSV4_DIRECT_VS_THINKING_WEBGL_LOGIT_PROBE_REL,
        "present": path_present,
        "schema": artifact.get("schema") if path_present else None,
        "case_count": len(by_key),
        "direct_after_web_gl_rank": direct_after_web_gl,
        "direct_after_web_web_rank": rank(
            "direct_off", "after_const_renderer_web", "Web"
        ),
        "thinking_after_web_gl_rank": thinking_after_web_gl,
        "thinking_after_web_web_rank": rank(
            "requested_thinking", "after_const_renderer_web", "Web"
        ),
        "webweb_not_explained_by_after_web_rank": (
            direct_after_web_gl == 1 and thinking_after_web_gl == 1
        ),
        "direct_after_three_dot_web_rank": direct_after_three_dot_web,
        "thinking_after_three_dot_web_rank": thinking_after_three_dot_web,
        "direct_after_three_dot_top_token": top_token(
            "direct_off", "after_const_renderer_three_dot"
        ),
        "thinking_after_three_dot_top_token": top_token(
            "requested_thinking", "after_const_renderer_three_dot"
        ),
        "direct_after_camera_p_pers_rank": direct_after_camera_p_pers,
        "direct_after_camera_p_ers_rank": direct_after_camera_p_ers,
        "thinking_after_camera_p_pers_rank": thinking_after_camera_p_pers,
        "thinking_after_camera_p_ers_rank": thinking_after_camera_p_ers,
        "direct_after_camera_p_top_token": top_token(
            "direct_off", "after_camera_p"
        ),
        "thinking_after_camera_p_top_token": top_token(
            "requested_thinking", "after_camera_p"
        ),
        "direct_camera_p_prefers_suffix_over_whole_token": (
            direct_after_camera_p_ers == 1 and direct_after_camera_p_pers == 2
        ),
        "thinking_camera_p_prefers_whole_token": (
            thinking_after_camera_p_pers == 1 and thinking_after_camera_p_ers == 2
        ),
    }


def _dsv4_hidden_reasoning_control_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_HIDDEN_REASONING_CONTROL_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    case_rows = [case for case in cases if isinstance(case, dict)]
    by_name = {
        str(case.get("name") or ""): case
        for case in case_rows
        if case.get("name")
    }
    summary = (
        artifact.get("diagnostic_summary")
        if isinstance(artifact.get("diagnostic_summary"), dict)
        else {}
    )
    failed_cases = [
        str(case.get("name"))
        for case in case_rows
        if case.get("exact") is not True and isinstance(case.get("name"), str)
    ]
    no_draft = by_name.get("period_no_reasoning_code_draft_system", {})
    system_controls = [
        by_name.get("period_no_reasoning_code_draft_system", {}),
        by_name.get("period_verify_identifiers_system", {}),
    ]
    cache_hit_values: list[int] = []
    for key in ("health_before", "health_after"):
        health = artifact.get(key) if isinstance(artifact.get(key), dict) else {}
        scheduler = (
            health.get("scheduler") if isinstance(health.get("scheduler"), dict) else {}
        )
        value = scheduler.get("cache_hit_tokens")
        if isinstance(value, int):
            cache_hit_values.append(value)

    return {
        "artifact": DSV4_HIDDEN_REASONING_CONTROL_REL,
        "present": path_present,
        "status": artifact.get("status") if path_present else "missing",
        "colon_control_exact": summary.get("colon_control_exact")
        if "colon_control_exact" in summary
        else by_name.get("colon_known_pass_control", {}).get("exact") is True,
        "period_control_exact": summary.get("period_control_exact")
        if "period_control_exact" in summary
        else by_name.get("period_known_fail_control", {}).get("exact") is True,
        "system_no_draft_exact": summary.get("system_no_draft_exact")
        if "system_no_draft_exact" in summary
        else no_draft.get("exact") is True,
        "system_verify_exact": summary.get("system_verify_exact")
        if "system_verify_exact" in summary
        else by_name.get("period_verify_identifiers_system", {}).get("exact") is True,
        "failed_cases": failed_cases,
        "system_controls_still_fail": bool(system_controls)
        and all(case.get("exact") is not True for case in system_controls),
        "no_draft_hidden_corruption_removed_but_visible_still_failed": (
            no_draft.get("hidden_contains_corruption") is False
            and no_draft.get("exact") is not True
        ),
        "hidden_reasoning_not_sufficient_root_cause": (
            no_draft.get("hidden_contains_corruption") is False
            and no_draft.get("exact") is not True
        ),
        "cache_hit_tokens_zero": bool(cache_hit_values)
        and all(value == 0 for value in cache_hit_values),
        "case_summaries": [
            {
                "name": case.get("name"),
                "exact": case.get("exact"),
                "finish_reason": case.get("finish_reason"),
                "prompt_tokens": case.get("prompt_tokens"),
                "completion_tokens": case.get("completion_tokens"),
                "hidden_contains_corruption": case.get("hidden_contains_corruption"),
                "corrupt_patterns": [
                    str(item) for item in (case.get("corrupt_patterns") or [])
                ],
                "missing": [str(item) for item in (case.get("missing") or [])],
            }
            for case in case_rows
        ],
    }

def _template_parity_boundary_token(case: dict[str, Any]) -> str | None:
    tokens = case.get("boundary_tokens_tail")
    if not isinstance(tokens, list):
        return None
    for token in reversed(tokens):
        if isinstance(token, str) and token.endswith("Ċ"):
            return token
    return None


def _dsv4_template_parity_diagnostic_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_TEMPLATE_PARITY_DIAGNOSTIC_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    case_rows = [case for case in cases if isinstance(case, dict)]
    mismatches = [
        case
        for case in case_rows
        if case.get("sidecar_equals_tokenizer_template") is not True
    ]
    normal_cases = [
        case
        for case in case_rows
        if case.get("enable_thinking") is False
        and case.get("reasoning_effort") is None
        and case.get("assistant_suffix") == "thinking_closed"
    ]
    period = next(
        (case for case in normal_cases if case.get("name") == "period_exact_code"),
        {},
    )
    colon = next(
        (case for case in normal_cases if case.get("name") == "colon_exact_code"),
        {},
    )
    period_tokens = period.get("prompt_tokens")
    colon_tokens = colon.get("prompt_tokens")

    return {
        "artifact": DSV4_TEMPLATE_PARITY_DIAGNOSTIC_REL,
        "present": path_present,
        "all_sidecar_equals_tokenizer_template": artifact.get(
            "all_sidecar_equals_tokenizer_template"
        )
        is True,
        "case_count": len(case_rows),
        "mismatch_count": len(mismatches),
        "mismatched_cases": [
            str(case.get("name"))
            for case in mismatches
            if isinstance(case.get("name"), str)
        ],
        "normal_colon_period_prompt_tokens_equal": (
            isinstance(period_tokens, int)
            and isinstance(colon_tokens, int)
            and period_tokens == colon_tokens
        ),
        "period_prompt_tokens": period_tokens,
        "colon_prompt_tokens": colon_tokens,
        "period_boundary_token": _template_parity_boundary_token(period),
        "colon_boundary_token": _template_parity_boundary_token(colon),
        "template_mismatch_ruled_out": (
            path_present
            and artifact.get("all_sidecar_equals_tokenizer_template") is True
            and len(mismatches) == 0
            and isinstance(period_tokens, int)
            and period_tokens == colon_tokens
            and _template_parity_boundary_token(period) == ".Ċ"
            and _template_parity_boundary_token(colon) == ":Ċ"
        ),
    }


def _dsv4_prefill_execution_variant_logits_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_PREFILL_EXECUTION_VARIANT_LOGITS_REL)
    variants = artifact.get("variants") if isinstance(artifact.get("variants"), list) else []
    variant_rows = [variant for variant in variants if isinstance(variant, dict)]
    top_tokens_by_variant: dict[str, list[str]] = {}
    selected_tokens: list[str | None] = []
    pers_ranks: dict[str, int | None] = {}
    for variant in variant_rows:
        name = str(variant.get("name") or "")
        top = variant.get("top") if isinstance(variant.get("top"), list) else []
        tokens = [
            str(item.get("token"))
            for item in top
            if isinstance(item, dict) and item.get("token") is not None
        ]
        top_tokens_by_variant[name] = tokens[:8]
        selected_tokens.append(tokens[0] if tokens else None)
        pers_ranks[name] = _rank_of_token(top, "Pers")
    non_empty_sequences = [tuple(tokens) for tokens in top_tokens_by_variant.values() if tokens]
    all_same_top_tokens = bool(non_empty_sequences) and len(set(non_empty_sequences)) == 1

    return {
        "artifact": DSV4_PREFILL_EXECUTION_VARIANT_LOGITS_REL,
        "present": path_present,
        "schema": artifact.get("schema") if path_present else None,
        "target_context": artifact.get("target_context"),
        "variant_count": len(variant_rows),
        "variant_names": [
            str(variant.get("name"))
            for variant in variant_rows
            if variant.get("name") is not None
        ],
        "top_tokens_by_variant": top_tokens_by_variant,
        "selected_tokens": selected_tokens,
        "all_variants_same_top_tokens": all_same_top_tokens,
        "all_variants_select_correct_ers": bool(selected_tokens)
        and all(token == "ers" for token in selected_tokens),
        "whole_pers_ranks": pers_ranks,
        "whole_pers_never_top": bool(pers_ranks)
        and all(rank is None or rank > 1 for rank in pers_ranks.values()),
        "stream_warmup_state_ruled_out_for_isolated_prefix": (
            path_present
            and all_same_top_tokens
            and bool(selected_tokens)
            and all(token == "ers" for token in selected_tokens)
            and bool(pers_ranks)
            and all(rank is None or rank > 1 for rank in pers_ranks.values())
        ),
    }


def _candidate_logprob(case: dict[str, Any], label: str) -> float | None:
    candidates = case.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("label") == label:
            value = candidate.get("first_token_logprob")
            return value if isinstance(value, int | float) else None
    return None


def _dsv4_prompt_variant_logit_probe_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_PROMPT_VARIANT_LOGIT_PROBE_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    case_rows = [case for case in cases if isinstance(case, dict)]
    by_name = {
        str(case.get("name")): case
        for case in case_rows
        if case.get("name") is not None
    }
    original_list = by_name.get("original_list_fourth_after_TH", {})
    copy_list = by_name.get("copy_block_list_fourth_after_TH", {})
    original_camera = by_name.get("original_snippet_camera_after_THREE_dot_P", {})
    copy_camera = by_name.get("copy_block_snippet_camera_after_THREE_dot_P", {})
    original_box = by_name.get("original_snippet_box_after_THREE_dot_B", {})

    original_list_corrupt = _candidate_logprob(original_list, "corrupt_IVE")
    copy_list_corrupt = _candidate_logprob(copy_list, "corrupt_IVE")
    original_list_top = original_list.get("top") if isinstance(original_list.get("top"), list) else []
    copy_list_top = copy_list.get("top") if isinstance(copy_list.get("top"), list) else []
    original_camera_top = (
        original_camera.get("top") if isinstance(original_camera.get("top"), list) else []
    )
    copy_camera_top = (
        copy_camera.get("top") if isinstance(copy_camera.get("top"), list) else []
    )
    original_box_top = original_box.get("top") if isinstance(original_box.get("top"), list) else []
    original_camera_ers_rank = _rank_of_token(original_camera_top, "ers")
    copy_camera_ers_rank = _rank_of_token(copy_camera_top, "ers")
    original_camera_pers_rank = _rank_of_token(original_camera_top, "Pers")
    copy_camera_pers_rank = _rank_of_token(copy_camera_top, "Pers")
    original_camera_ers = _candidate_logprob(original_camera, "correct_ers")
    copy_camera_ers = _candidate_logprob(copy_camera, "correct_ers")
    original_camera_pers = _candidate_logprob(original_camera, "whole_Pers")
    copy_camera_pers = _candidate_logprob(copy_camera, "whole_Pers")
    camera_ers_improves = (
        isinstance(original_camera_ers, int | float)
        and isinstance(copy_camera_ers, int | float)
        and copy_camera_ers > original_camera_ers
    )
    camera_pers_reduces = (
        isinstance(original_camera_pers, int | float)
        and isinstance(copy_camera_pers, int | float)
        and copy_camera_pers < original_camera_pers
    )

    return {
        "artifact": DSV4_PROMPT_VARIANT_LOGIT_PROBE_REL,
        "present": path_present,
        "schema": artifact.get("schema") if path_present else None,
        "case_count": len(case_rows),
        "case_names": [
            str(case.get("name"))
            for case in case_rows
            if case.get("name") is not None
        ],
        "original_list_correct_REE_rank": _rank_of_token(original_list_top, "REE"),
        "copy_block_list_correct_REE_rank": _rank_of_token(copy_list_top, "REE"),
        "original_list_corrupt_IVE_logprob": original_list_corrupt,
        "copy_block_list_corrupt_IVE_logprob": copy_list_corrupt,
        "copy_block_reduces_list_corrupt_IVE_logprob": (
            isinstance(original_list_corrupt, int | float)
            and isinstance(copy_list_corrupt, int | float)
            and copy_list_corrupt < original_list_corrupt
        ),
        "original_camera_whole_Pers_rank": original_camera_pers_rank,
        "original_camera_correct_ers_rank": original_camera_ers_rank,
        "original_camera_correct_ers_logprob": original_camera_ers,
        "copy_block_camera_correct_ers_logprob": copy_camera_ers,
        "copy_block_improves_camera_correct_ers_logprob": camera_ers_improves,
        "original_camera_whole_Pers_logprob": original_camera_pers,
        "copy_block_camera_whole_Pers_logprob": copy_camera_pers,
        "copy_block_reduces_camera_whole_Pers_logprob": camera_pers_reduces,
        "copy_block_camera_correct_ers_rank": copy_camera_ers_rank,
        "copy_block_camera_whole_Pers_rank": copy_camera_pers_rank,
        "original_box_whole_Box_rank": _rank_of_token(original_box_top, "Box"),
        "original_box_correct_ox_rank": _rank_of_token(original_box_top, "ox"),
        "prompt_wording_changes_identifier_logits": (
            path_present
            and camera_ers_improves
            and camera_pers_reduces
            and original_camera_pers_rank == 1
            ),
    }


def _dsv4_reasoning_policy_live_detail(
    artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_REASONING_POLICY_LIVE_REL)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    case_rows = [case for case in cases if isinstance(case, dict)]
    by_name = {
        str(case.get("name")): case
        for case in case_rows
        if case.get("name") is not None
    }
    explicit_off = by_name.get("short_chat_explicit_off", {})
    one_token_off = by_name.get("one_token_chat_explicit_off", {})
    requested_on = by_name.get("short_chat_requested_on", {})

    def _visible_no_reasoning(case: dict[str, Any]) -> bool:
        return (
            case.get("content_is_null") is False
            and int(case.get("reasoning_chars") or 0) == 0
        )

    return {
        "artifact": DSV4_REASONING_POLICY_LIVE_REL,
        "present": path_present,
        "status": artifact.get("status") if path_present else "missing",
        "case_count": len(case_rows),
        "case_names": [
            str(case.get("name"))
            for case in case_rows
            if case.get("name") is not None
        ],
        "explicit_off_visible_no_reasoning": _visible_no_reasoning(explicit_off),
        "one_token_off_visible_no_reasoning": _visible_no_reasoning(one_token_off),
        "requested_on_reasoning_observed": int(
            requested_on.get("reasoning_chars") or 0
        )
        > 0,
        "requested_on_content_is_null": requested_on.get("content_is_null") is True,
        "health_after": artifact.get("health_after") if path_present else None,
    }


def _selected_token(top: Any) -> str | None:
    if not isinstance(top, list) or not top:
        return None
    first = top[0]
    if not isinstance(first, dict):
        return None
    token = first.get("token")
    return str(token) if token is not None else None


def _dsv4_batch_generator_logit_divergence_detail(
    direct_artifact: dict[str, Any],
    batch_artifact: dict[str, Any],
    warmup_ablation_artifact: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    direct_present = _path_present(root, DSV4_CACHE_VS_FULL_LOGIT_ISOLATION_REL)
    batch_present = _path_present(root, DSV4_BATCH_GENERATOR_LOGIT_TRACE_REL)
    warmup_ablation_present = _path_present(
        root, DSV4_BATCH_GENERATOR_WARMUP_ABLATION_REL
    )
    checks = (
        direct_artifact.get("checks")
        if isinstance(direct_artifact.get("checks"), list)
        else []
    )
    direct_case = next(
        (
            check
            for check in checks
            if isinstance(check, dict)
            and check.get("name") == "isolated_after_THREE_dot_P"
        ),
        {},
    )
    events = (
        batch_artifact.get("events")
        if isinstance(batch_artifact.get("events"), list)
        else []
    )
    batch_event = next(
        (
            event
            for event in events
            if isinstance(event, dict)
            and event.get("generated_so_far_text") == "THREE.P"
        ),
        {},
    )
    direct_full_selected = _selected_token(direct_case.get("full_top"))
    direct_incremental_selected = _selected_token(direct_case.get("incremental_top"))
    batch_selected = _selected_token(batch_event.get("top"))
    decoded = str(batch_artifact.get("decoded") or "")
    warmup_runs = (
        warmup_ablation_artifact.get("runs")
        if isinstance(warmup_ablation_artifact.get("runs"), list)
        else []
    )
    exact_warmup_labels = [
        str(run.get("label"))
        for run in warmup_runs
        if isinstance(run, dict)
        and str(run.get("decoded") or "").startswith("THREE.PerspectiveCamera")
    ]

    return {
        "direct_artifact": DSV4_CACHE_VS_FULL_LOGIT_ISOLATION_REL,
        "batch_artifact": DSV4_BATCH_GENERATOR_LOGIT_TRACE_REL,
        "warmup_ablation_artifact": DSV4_BATCH_GENERATOR_WARMUP_ABLATION_REL,
        "direct_artifact_present": direct_present,
        "batch_artifact_present": batch_present,
        "warmup_ablation_artifact_present": warmup_ablation_present,
        "direct_full_THREE_P_selected": direct_full_selected,
        "direct_incremental_THREE_P_selected": direct_incremental_selected,
        "batch_THREE_P_selected": batch_selected,
        "batch_decoded": decoded,
        "batch_decoded_contains_corrupt_pertive": "PertiveCamera" in decoded,
        "warmup_ablation_exact_runs": exact_warmup_labels,
        "batch_generator_intrinsic_failure_not_proven": (
            warmup_ablation_present and bool(exact_warmup_labels)
        ),
        "direct_vs_batch_THREE_P_diverges": (
            direct_present
            and batch_present
            and direct_full_selected == "ers"
            and direct_incremental_selected == "ers"
            and batch_selected == "ert"
        ),
    }


def _dsv4_identifier_matrix_detail(matrix: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_CURRENT_IDENTIFIER_MATRIX_REL)
    probes = matrix.get("probes") if isinstance(matrix.get("probes"), list) else []
    probe_summaries: list[dict[str, Any]] = []
    failed_probe_ids: list[str] = []
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        probe_id = probe.get("id")
        if probe.get("status") != "pass" and isinstance(probe_id, str):
            failed_probe_ids.append(probe_id)
        content = probe.get("content")
        if not isinstance(content, str):
            content = ""
        probe_summaries.append(
            {
                "id": probe_id,
                "status": probe.get("status"),
                "elapsed_sec": probe.get("elapsed_sec"),
                "required_identifier_counts": probe.get("required_identifier_counts") or {},
                "has_markdown_fence": probe.get("has_markdown_fence"),
                "content": content[:1000],
            }
        )
    return {
        "artifact": DSV4_CURRENT_IDENTIFIER_MATRIX_REL,
        "present": path_present,
        "status": matrix.get("status") if path_present else "missing",
        "probe_count": len(probe_summaries),
        "failed_probe_ids": failed_probe_ids,
        "copy_task_failed": "copy_identifiers_only" in failed_probe_ids,
        "code_task_failed": "single_line_threejs_code" in failed_probe_ids,
        "probe_summaries": probe_summaries,
    }


def _dsv4_tokenizer_roundtrip_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    row_summaries: list[dict[str, Any]] = []
    failed_inputs: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("input")
        if row.get("roundtrip_exact") is not True and isinstance(text, str):
            failed_inputs.append(text)
        row_summaries.append(
            {
                "input": text,
                "ids": row.get("ids") or [],
                "tokens": row.get("tokens") or [],
                "decoded": row.get("decoded"),
                "roundtrip_exact": row.get("roundtrip_exact") is True,
            }
        )
    status = "missing"
    if path_present:
        status = "pass" if payload.get("all_roundtrip_exact") is True and not failed_inputs else "review"
    return {
        "artifact": DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL,
        "present": path_present,
        "status": status,
        "load_method": payload.get("load_method"),
        "tokenizer_class": payload.get("tokenizer_class"),
        "all_roundtrip_exact": payload.get("all_roundtrip_exact") is True,
        "row_count": len(row_summaries),
        "failed_inputs": failed_inputs,
        "rows": row_summaries,
    }


def _dsv4_logprob_copy_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_LIVE_LOGPROBS_COPY_REL)
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    logprobs = first_choice.get("logprobs") if isinstance(first_choice.get("logprobs"), dict) else {}
    entries = logprobs.get("content") if isinstance(logprobs.get("content"), list) else []
    tokens = [entry.get("token") for entry in entries if isinstance(entry, dict)]
    wrong_entry: dict[str, Any] = {}
    wrong_index: int | None = None
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if entry.get("token") == "c":
            prev_tokens = tokens[max(0, idx - 4) : idx]
            next_token = tokens[idx + 1] if idx + 1 < len(tokens) else None
            if prev_tokens[-2:] == [".P", "ers"] and next_token == "pective":
                wrong_entry = entry
                wrong_index = idx
                break
    top_logprobs = (
        wrong_entry.get("top_logprobs")
        if isinstance(wrong_entry.get("top_logprobs"), list)
        else []
    )
    top_tokens = [
        item.get("token")
        for item in top_logprobs
        if isinstance(item, dict) and isinstance(item.get("token"), str)
    ]
    wrong_rank = top_tokens.index("c") + 1 if "c" in top_tokens else None
    correct_rank = top_tokens.index("pective") + 1 if "pective" in top_tokens else None
    return {
        "artifact": DSV4_LIVE_LOGPROBS_COPY_REL,
        "present": path_present,
        "status": (
            "review"
            if path_present and wrong_entry
            else ("pass" if path_present and payload.get("status") == "pass" else "missing")
        ),
        "content": (payload.get("content") if isinstance(payload.get("content"), str) else "")[:1000],
        "identifier_counts": payload.get("identifier_counts") or {},
        "logprob_entry_count": payload.get("logprob_entry_count"),
        "tokens": tokens,
        "wrong_token_index": wrong_index,
        "wrong_token": wrong_entry.get("token") if wrong_entry else None,
        "wrong_token_logprob": wrong_entry.get("logprob") if wrong_entry else None,
        "correct_token": "pective" if "pective" in top_tokens else None,
        "wrong_token_rank": wrong_rank,
        "correct_token_rank": correct_rank,
        "model_preferred_wrong_token": (
            wrong_rank == 1 and correct_rank is not None and correct_rank > wrong_rank
        ),
        "top_logprobs_at_wrong_token": top_logprobs,
    }


def _dsv4_logprob_context_matrix_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL)
    probes = payload.get("probes") if isinstance(payload.get("probes"), list) else []
    probe_summaries: dict[str, dict[str, Any]] = {}
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        probe_id = probe.get("id")
        if not isinstance(probe_id, str):
            continue
        content = probe.get("content") if isinstance(probe.get("content"), str) else ""
        probe_summaries[probe_id] = {
            "content": content[:1000],
            "target_count": probe.get("target_count"),
            "has_wrong_perscpective": probe.get("has_wrong_perscpective") is True,
            "tokens": probe.get("tokens") or [],
            "wrong_c_after_pers_events": probe.get("wrong_c_after_pers_events") or [],
        }
    isolated = probe_summaries.get("isolated_identifier", {})
    list_copy = probe_summaries.get("list_copy", {})
    constructor = probe_summaries.get("constructor_sentence", {})
    isolated_passed = isolated.get("target_count") == 1 and not isolated.get("has_wrong_perscpective")
    list_failed = list_copy.get("target_count") == 0 and bool(list_copy.get("wrong_c_after_pers_events"))
    constructor_content = constructor.get("content") if isinstance(constructor.get("content"), str) else ""
    constructor_failed = constructor.get("target_count") == 0 and "PerspectiveCamera" not in constructor_content
    return {
        "artifact": DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL,
        "present": path_present,
        "status": payload.get("status") if path_present else "missing",
        "target": payload.get("target"),
        "isolated_identifier_passed": isolated_passed,
        "list_copy_failed": list_failed,
        "constructor_sentence_failed": constructor_failed,
        "context_sensitive_identifier_failure": isolated_passed
        and (list_failed or constructor_failed),
        "probe_summaries": probe_summaries,
    }


def _dsv4_cache_context_identifier_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL)
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    probe_summaries: dict[str, dict[str, Any]] = {}
    corrupt_identifiers_by_probe: dict[str, list[str]] = {}
    missing_identifiers_by_probe: dict[str, list[str]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        name = result.get("name")
        if not isinstance(name, str):
            continue
        corrupt = [
            item
            for item in (result.get("corrupt_identifier_tokens") or [])
            if isinstance(item, str)
        ]
        missing = [
            item
            for item in (result.get("expected_identifiers_missing") or [])
            if isinstance(item, str)
        ]
        content = result.get("content") if isinstance(result.get("content"), str) else ""
        corrupt_identifiers_by_probe[name] = corrupt
        missing_identifiers_by_probe[name] = missing
        probe_summaries[name] = {
            "status": result.get("status"),
            "elapsed_sec": result.get("elapsed_sec"),
            "content": content[:1000],
            "expected_identifiers_missing": missing,
            "corrupt_identifier_tokens": corrupt,
        }
    health_before = payload.get("health_before") if isinstance(payload.get("health_before"), dict) else {}
    native_cache = (
        health_before.get("native_cache") if isinstance(health_before.get("native_cache"), dict) else {}
    )
    pool_quant = (
        native_cache.get("pool_quant") if isinstance(native_cache.get("pool_quant"), dict) else {}
    )
    generic_tq = (
        native_cache.get("generic_turboquant_kv")
        if isinstance(native_cache.get("generic_turboquant_kv"), dict)
        else {}
    )
    return {
        "artifact": DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL,
        "present": path_present,
        "status": payload.get("status") if path_present else "missing",
        "plain_list_failed": bool(
            corrupt_identifiers_by_probe.get("list_plain")
            or missing_identifiers_by_probe.get("list_plain")
        ),
        "unique_prefix_failed": bool(
            corrupt_identifiers_by_probe.get("list_unique_prefix")
            or missing_identifiers_by_probe.get("list_unique_prefix")
        ),
        "constructor_unique_prefix_passed": (
            probe_summaries.get("constructor_unique_prefix", {}).get("status") == "pass"
        ),
        "native_cache_pool_quant_enabled": pool_quant.get("enabled") is True,
        "generic_turboquant_kv_enabled": generic_tq.get("enabled") is True,
        "corrupt_identifiers_by_probe": corrupt_identifiers_by_probe,
        "missing_identifiers_by_probe": missing_identifiers_by_probe,
        "probe_summaries": probe_summaries,
    }


def _dsv4_source_nocache_identifier_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_SOURCE_NOCACHE_IDENTIFIER_REL)
    probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
    counts = probe.get("identifier_counts") if isinstance(probe.get("identifier_counts"), dict) else {}
    health = payload.get("health_before") if isinstance(payload.get("health_before"), dict) else {}
    native_cache = health.get("native_cache") if isinstance(health.get("native_cache"), dict) else {}
    pool_quant = (
        native_cache.get("pool_quant") if isinstance(native_cache.get("pool_quant"), dict) else {}
    )
    generic_tq = (
        native_cache.get("generic_turboquant_kv")
        if isinstance(native_cache.get("generic_turboquant_kv"), dict)
        else {}
    )
    content = probe.get("content") if isinstance(probe.get("content"), str) else ""
    return {
        "artifact": DSV4_SOURCE_NOCACHE_IDENTIFIER_REL,
        "present": path_present,
        "status": payload.get("status") if path_present else "missing",
        "probe_elapsed_sec": payload.get("probe_elapsed_sec"),
        "usage": probe.get("usage"),
        "identifier_counts": counts,
        "all_identifiers_present": bool(counts) and all(
            isinstance(value, int) and value > 0 for value in counts.values()
        ),
        "content": content[:1000],
        "native_cache_prefix": native_cache.get("prefix") is True,
        "native_cache_paged": native_cache.get("paged") is True,
        "native_cache_block_disk_l2": native_cache.get("block_disk_l2") is True,
        "native_cache_pool_quant_enabled": pool_quant.get("enabled") is True,
        "generic_turboquant_kv_enabled": generic_tq.get("enabled") is True,
    }


def _dsv4_source_same_prompt_cache_boundary_detail(
    same_prompt_payload: dict[str, Any],
    cache_payload: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    same_present = _path_present(root, DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL)
    cache_present = _path_present(root, DSV4_SOURCE_CACHE_COMPARISON_REL)
    probe = same_prompt_payload.get("probe") if isinstance(same_prompt_payload.get("probe"), dict) else {}
    analysis = probe.get("analysis") if isinstance(probe.get("analysis"), dict) else {}
    same_content = analysis.get("content") if isinstance(analysis.get("content"), str) else ""
    missing = [
        item
        for item in (analysis.get("missing_identifiers") or [])
        if isinstance(item, str)
    ]
    same_nocache_failed = same_present and (
        same_prompt_payload.get("status") != "pass"
        or bool(missing)
        or analysis.get("has_common_corruptions") is True
    )

    cases = cache_payload.get("cases") if isinstance(cache_payload.get("cases"), list) else []
    case_summaries: dict[str, dict[str, Any]] = {}
    cache_failures: dict[str, list[dict[str, Any]]] = {}
    cache_hit_proven_by_case: dict[str, bool] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        name = case.get("name")
        if not isinstance(name, str):
            continue
        failures = [
            item
            for item in (case.get("failures") or [])
            if isinstance(item, dict)
        ]
        cache_failures[name] = failures
        cache_hit_proven_by_case[name] = False
        case_summaries[name] = {
            "status": case.get("status"),
            "artifact": case.get("artifact"),
            "failure_count": len(failures),
            "failures": failures,
        }

    pooloff_failed = bool(cache_failures.get("pooloff")) or (
        case_summaries.get("pooloff", {}).get("status") not in {None, "pass"}
    )
    poolon_failed = bool(cache_failures.get("poolon")) or (
        case_summaries.get("poolon", {}).get("status") not in {None, "pass"}
    )
    return {
        "artifact": DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL,
        "cache_comparison_artifact": DSV4_SOURCE_CACHE_COMPARISON_REL,
        "present": same_present and cache_present,
        "same_prompt_nocache": {
            "status": same_prompt_payload.get("status") if same_present else "missing",
            "content": same_content[:1000],
            "missing_identifiers": missing,
            "has_common_corruptions": analysis.get("has_common_corruptions") is True,
            "usage": probe.get("usage"),
        },
        "same_prompt_nocache_failed": same_nocache_failed,
        "cache_enabled_pooloff_failed": pooloff_failed,
        "cache_enabled_poolon_failed": poolon_failed,
        "pool_quant_is_not_differentiator": pooloff_failed and poolon_failed,
        "cache_hit_restore_not_proven_by_short_prompt": (
            same_nocache_failed and pooloff_failed and poolon_failed
        ),
        "cache_case_summaries": case_summaries,
        "cache_hit_proven_by_case": cache_hit_proven_by_case,
    }


def _contract_checks(payload: dict[str, Any], required: tuple[str, ...]) -> tuple[bool, dict[str, bool]]:
    checks = payload.get("checks") or {}
    required_checks = {key: checks.get(key) is True for key in required}
    return payload.get("status") == "pass" and all(required_checks.values()), required_checks


def _source_hash_status(
    root: Path, payload: dict[str, Any], required_files: tuple[str, ...]
) -> tuple[bool, dict[str, Any]]:
    recorded = payload.get("source_hashes") or {}
    current_hashes: dict[str, str | None] = {}
    missing_source_hashes: list[str] = []
    stale_source_hashes: list[str] = []
    missing_source_files: list[str] = []

    for rel in required_files:
        expected = recorded.get(rel)
        if not isinstance(expected, str) or not expected:
            missing_source_hashes.append(rel)
        path = root / rel
        if not path.is_file():
            current_hashes[rel] = None
            missing_source_files.append(rel)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        current_hashes[rel] = actual
        if isinstance(expected, str) and expected and actual != expected:
            stale_source_hashes.append(rel)

    ok = not missing_source_hashes and not stale_source_hashes and not missing_source_files
    return ok, {
        "source_hashes_recorded": recorded,
        "source_hashes_current": current_hashes,
        "missing_source_hashes": missing_source_hashes,
        "stale_source_hashes": stale_source_hashes,
        "missing_source_files": missing_source_files,
    }


def _contract_detail(
    root: Path,
    payload: dict[str, Any],
    required_checks: tuple[str, ...],
    required_files: tuple[str, ...],
) -> tuple[bool, dict[str, Any]]:
    checks_ok, required = _contract_checks(payload, required_checks)
    hashes_ok, hash_details = _source_hash_status(root, payload, required_files)
    missing_rows = payload.get("missing_rows") or []
    family_matrix_required = "generation_defaults_family_matrix_complete" in required_checks
    family_matrix = payload.get("generation_defaults_family_matrix") or {}
    missing_family_rows = (
        [
            row
            for row in REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX
            if row not in family_matrix
        ]
        if family_matrix_required
        else []
    )
    failed_family_rows = (
        [
            row
            for row in REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX
            if family_matrix.get(row, {}).get("status") != "pass"
        ]
        if family_matrix_required
        else []
    )
    missing_markers = payload.get("missing_markers") or []
    failed = payload.get("failed") or []
    ok = (
        checks_ok
        and hashes_ok
        and not missing_rows
        and not missing_family_rows
        and not failed_family_rows
        and not missing_markers
        and not failed
    )
    return ok, {
        "status": payload.get("status"),
        "failed": failed,
        "missing_rows": missing_rows,
        "missing_family_rows": missing_family_rows,
        "failed_family_rows": failed_family_rows,
        "missing_markers": missing_markers,
        "contract_checks": required,
        **hash_details,
    }


def _speed_artifact_detail(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    results = payload.get("results") or []
    row = results[0] if results and isinstance(results[0], dict) else {}
    pp_rows = row.get("pp_rows") or []
    pp_values: list[float] = []
    loopish_values: list[bool] = []
    for item in pp_rows:
        if not isinstance(item, dict):
            continue
        try:
            pp_values.append(float(item.get("pp_wall_tok_s")))
        except (TypeError, ValueError):
            pass
        loopish_values.append(bool(item.get("loopish")))
    min_pp = min(pp_values) if pp_values else None
    status = row.get("status")
    notes = row.get("notes") or []
    ok = (
        status == "pass"
        and min_pp is not None
        and min_pp >= 600.0
        and not any(loopish_values)
        and not notes
    )
    return ok, {
        "status": status,
        "notes": notes,
        "runtime_wheels": row.get("runtime_wheels"),
        "pp_wall_tok_s": pp_values,
        "min_pp_wall_tok_s": min_pp,
        "loopish_values": loopish_values,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _native_mtp_ab_detail(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    rows = payload.get("rows") or []
    by_label = {
        row.get("label"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    }
    baseline = by_label.get("baseline_no_mtp") or {}
    native = by_label.get("native_mtp") or {}
    baseline_tps = _float_or_none((baseline.get("summary") or {}).get("mean_wall_tok_s"))
    native_tps = _float_or_none((native.get("summary") or {}).get("mean_wall_tok_s"))
    speedup = _float_or_none(payload.get("speedup_vs_baseline"))
    output_equivalence = payload.get("output_equivalence") or {}
    mtp_totals = ((native.get("mtp_stats") or {}).get("totals") or {})
    acceptance_rate = _float_or_none(mtp_totals.get("acceptance_rate"))
    ok = (
        bool(output_equivalence.get("all_content_equal"))
        and bool(output_equivalence.get("all_full_text_equal"))
        and speedup is not None
        and speedup >= 1.1
        and native_tps is not None
        and native_tps >= 25.0
        and acceptance_rate is not None
        and acceptance_rate >= 0.5
    )
    return ok, {
        "speedup_vs_baseline": speedup,
        "baseline_decode_tps_wall": baseline_tps,
        "native_mtp_decode_tps_wall": native_tps,
        "output_equivalence": {
            "all_content_equal": output_equivalence.get("all_content_equal"),
            "all_full_text_equal": output_equivalence.get("all_full_text_equal"),
        },
        "native_mtp_acceptance_rate": acceptance_rate,
        "native_mtp_accepted_tokens": mtp_totals.get("accepted_tokens"),
        "native_mtp_drafted_tokens": mtp_totals.get("drafted_tokens"),
        "best_native_mtp_depth": payload.get("best_native_mtp_depth"),
    }


def _prefill_trace_detail(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    row = results[0] if results and isinstance(results[0], dict) else {}
    health = row.get("health_after") if isinstance(row.get("health_after"), dict) else {}
    scheduler = health.get("scheduler") if isinstance(health.get("scheduler"), dict) else {}
    batch_generator = (
        scheduler.get("batch_generator")
        if isinstance(scheduler.get("batch_generator"), dict)
        else {}
    )
    trace = batch_generator.get("last_prefill_trace")
    diagnosis: dict[str, Any] = {}
    if isinstance(trace, dict):
        forward_ms = _float_or_none(trace.get("forward_ms"))
        logits_eval_ms = _float_or_none(trace.get("logits_eval_ms"))
        sample_ms = _float_or_none(trace.get("sample_ms"))
        total_ms = _float_or_none(trace.get("total_ms"))
        preprocess_ms = _float_or_none(trace.get("preprocess_ms"))
        diagnosis = {
            "text_only_mllm_path": trace.get("has_images") is False,
            "native_mtp": trace.get("native_mtp") is True,
            "is_hybrid": trace.get("is_hybrid") is True,
            "uses_vlm_language_copy": str(trace.get("language_model_class") or "").startswith(
                "mlx_vlm."
            ),
            "force_text_rope_1d": trace.get("force_text_rope_1d") is True,
            "supports_return_logits": trace.get("supports_return_logits") is True,
            "preprocess_is_not_bottleneck": (
                preprocess_ms is not None
                and total_ms is not None
                and total_ms > 0
                and preprocess_ms / total_ms < 0.01
            ),
            "logits_eval_dominates_forward": (
                logits_eval_ms is not None
                and forward_ms is not None
                and forward_ms > 0
                and logits_eval_ms / forward_ms >= 5.0
            ),
            "sample_dominates_total": (
                sample_ms is not None
                and total_ms is not None
                and total_ms > 0
                and sample_ms / total_ms >= 0.5
            ),
            "logits_eval_to_forward_ratio": (
                round(logits_eval_ms / forward_ms, 3)
                if logits_eval_ms is not None and forward_ms is not None and forward_ms > 0
                else None
            ),
            "sample_to_total_ratio": (
                round(sample_ms / total_ms, 3)
                if sample_ms is not None and total_ms is not None and total_ms > 0
                else None
            ),
            "suspected_bottleneck": "unknown",
        }
        if (
            diagnosis["text_only_mllm_path"]
            and diagnosis["logits_eval_dominates_forward"]
            and diagnosis["sample_dominates_total"]
        ):
            diagnosis["suspected_bottleneck"] = "logits/sample materialization"
    return {
        "status": row.get("status"),
        "notes": row.get("notes") or [],
        "last_prefill_trace": trace,
        "diagnosis": diagnosis,
    }


def _no_prefix_logits_trial_detail(payload: dict[str, Any]) -> dict[str, Any]:
    ok, details = _speed_artifact_detail(payload)
    details["clears_prompt_processing_floor"] = ok
    return details


def _qwen_pp_trial_detail(payload: dict[str, Any]) -> dict[str, Any]:
    ok, details = _speed_artifact_detail(payload)
    details["clears_prompt_processing_floor"] = ok
    return details


def _qwen_raw_forward_ab_detail(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    raw_forward_only = True
    for payload in payloads:
        policy = str(payload.get("generation_defaults_policy") or "")
        raw_forward_only = raw_forward_only and "raw-forward-only" in policy
        for item in payload.get("comparison") or []:
            if not isinstance(item, dict):
                continue
            comparisons.append(
                {
                    "prompt_tokens": item.get("prompt_tokens"),
                    "text_pp_tok_s": _float_or_none(item.get("text_pp_tok_s")),
                    "vlm_pp_tok_s": _float_or_none(item.get("vlm_pp_tok_s")),
                    "text_over_vlm": _float_or_none(item.get("text_over_vlm")),
                }
            )
    ratios = [
        float(item["text_over_vlm"])
        for item in comparisons
        if item.get("text_over_vlm") is not None
    ]
    return {
        "raw_forward_only": raw_forward_only,
        "comparisons": comparisons,
        "min_text_over_vlm": min(ratios) if ratios else None,
        "max_text_over_vlm": max(ratios) if ratios else None,
    }


def _first_result_rails(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    for item in results:
        if not isinstance(item, dict):
            continue
        rails = item.get("response_rails")
        if isinstance(rails, dict):
            return rails
    return {}


def _gemma4_visible_diag(payload: dict[str, Any]) -> dict[str, Any]:
    rails = _first_result_rails(payload)
    return {
        "status": payload.get("status"),
        "reason": payload.get("reason"),
        "request_route": payload.get("request_route"),
        "max_tokens": payload.get("max_tokens"),
        "max_thinking_tokens": payload.get("max_thinking_tokens"),
        "visible_chars": rails.get("visible_chars"),
        "reasoning_chars": rails.get("reasoning_chars"),
    }


def _gemma4_metadata_budget_unsupported(payload: dict[str, Any]) -> bool:
    notes = payload.get("review_notes")
    if not isinstance(notes, dict):
        return False
    row_notes = notes.get("Gemma-4-26B-A4B-it-JANG_4M-CRACK")
    if not isinstance(row_notes, list):
        return False
    return "thinking_budget_override_forwarded_but_template_does_not_enforce" in row_notes


def _gemma4_visible_content_detail(
    payload: dict[str, Any],
    root: Path,
    artifact_rel: str = GEMMA4_RESPONSES_VISIBLE_CONTRACT_REL,
    responses_nocache_payload: dict[str, Any] | None = None,
    responses_512_payload: dict[str, Any] | None = None,
    thinking_off_payload: dict[str, Any] | None = None,
    chat_nocache_payload: dict[str, Any] | None = None,
    metadata_payload: dict[str, Any] | None = None,
    unsupported_budget_payload: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    path_present = _path_present(root, artifact_rel)
    contract = payload.get("response_contract")
    if not isinstance(contract, dict):
        contract = {}
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    stage_statuses = [
        item.get("status")
        for item in results
        if isinstance(item, dict) and item.get("status") is not None
    ]
    responses_nocache_payload = responses_nocache_payload or {}
    responses_512_payload = responses_512_payload or {}
    thinking_off_payload = thinking_off_payload or {}
    chat_nocache_payload = chat_nocache_payload or {}
    metadata_payload = metadata_payload or {}
    unsupported_budget_payload = unsupported_budget_payload or {}
    primary_visible = _gemma4_visible_diag(payload)
    unsupported_budget = _gemma4_visible_diag(unsupported_budget_payload)
    responses_nocache = _gemma4_visible_diag(responses_nocache_payload)
    responses_512 = _gemma4_visible_diag(responses_512_payload)
    thinking_off = _gemma4_visible_diag(thinking_off_payload)
    chat_nocache = _gemma4_visible_diag(chat_nocache_payload)
    metadata_budget_unsupported = _gemma4_metadata_budget_unsupported(metadata_payload)
    primary_visible_ok = (
        payload.get("status") == "pass"
        and int(primary_visible.get("visible_chars") or 0) > 0
        and int(primary_visible.get("reasoning_chars") or 0) > 0
    )
    cache_bypass_reproduces_missing_visible = (
        responses_nocache.get("status") == "fail"
        and int(responses_nocache.get("visible_chars") or 0) == 0
        and int(responses_nocache.get("reasoning_chars") or 0) > 0
    )
    higher_output_cap_restores_visible = (
        responses_512.get("status") == "pass"
        and int(responses_512.get("visible_chars") or 0) > 0
        and int(responses_512.get("reasoning_chars") or 0) > 0
    )
    thinking_off_restores_visible = (
        thinking_off.get("status") == "pass"
        and int(thinking_off.get("visible_chars") or 0) > 0
        and int(thinking_off.get("reasoning_chars") or 0) == 0
    )
    chat_route_reproduces_missing_visible = (
        chat_nocache.get("status") == "fail"
        and int(chat_nocache.get("visible_chars") or 0) == 0
        and int(chat_nocache.get("reasoning_chars") or 0) > 0
    )
    root_cause_summary = None
    if (
        cache_bypass_reproduces_missing_visible
        and higher_output_cap_restores_visible
        and thinking_off_restores_visible
        and chat_route_reproduces_missing_visible
        and metadata_budget_unsupported
    ):
        root_cause_summary = (
            "Gemma4 thinking_budget is forwarded but the local template does not "
            "consume it; thinking-on can exhaust a low max_tokens cap before "
            "visible content."
        )
    ok = (
        path_present
        and payload.get("status") == "pass"
        and contract.get("expect_visible_content") is True
        and all(status == "ok" for status in stage_statuses)
        and primary_visible_ok
    )
    return ok, {
        "artifact": artifact_rel,
        "artifact_present": path_present,
        "artifact_status": payload.get("status"),
        "reason": payload.get("reason"),
        "row": payload.get("row"),
        "request_route": payload.get("request_route"),
        "expect_visible_content": contract.get("expect_visible_content"),
        "stage_statuses": stage_statuses,
        "max_thinking_tokens": payload.get("max_thinking_tokens"),
        "model_path": payload.get("model_path"),
        "root_cause_summary": root_cause_summary,
        "cache_bypass_reproduces_missing_visible": cache_bypass_reproduces_missing_visible,
        "higher_output_cap_restores_visible": higher_output_cap_restores_visible,
        "thinking_off_restores_visible": thinking_off_restores_visible,
        "chat_route_reproduces_missing_visible": chat_route_reproduces_missing_visible,
        "metadata_budget_unsupported": metadata_budget_unsupported,
        "primary_visible_ok": primary_visible_ok,
        "diagnostic_artifacts": {
            "primary_app_visible": primary_visible,
            "unsupported_budget_contract": unsupported_budget,
            "responses_128_nocache": responses_nocache,
            "responses_512_nocache": responses_512,
            "responses_thinking_off_nocache": thinking_off,
            "chat_128_nocache": chat_nocache,
            "local_metadata": {
                "status": metadata_payload.get("status"),
                "budget_unsupported": metadata_budget_unsupported,
            },
        },
    }


def _find_native_cache_schema(value: Any) -> str | None:
    if isinstance(value, dict):
        native = value.get("native_cache")
        if isinstance(native, dict):
            schema = native.get("schema") or native.get("cache_type")
            if isinstance(schema, str):
                return schema
        for key in ("body", "health", "after", "before", "health_ready"):
            found = _find_native_cache_schema(value.get(key))
            if found:
                return found
        for child in value.values():
            found = _find_native_cache_schema(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_native_cache_schema(child)
            if found:
                return found
    return None


def _find_native_cache_generic_tq_enabled(value: Any) -> bool | None:
    if isinstance(value, dict):
        native = value.get("native_cache")
        if isinstance(native, dict):
            generic = native.get("generic_turboquant_kv")
            if isinstance(generic, dict) and isinstance(generic.get("enabled"), bool):
                return bool(generic["enabled"])
        for key in ("body", "health", "after", "before", "health_ready"):
            found = _find_native_cache_generic_tq_enabled(value.get(key))
            if found is not None:
                return found
        for child in value.values():
            found = _find_native_cache_generic_tq_enabled(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_native_cache_generic_tq_enabled(child)
            if found is not None:
                return found
    return None


def _collect_decode_speeds(value: Any) -> list[float]:
    speeds: list[float] = []
    if isinstance(value, dict):
        speed = value.get("speed")
        if isinstance(speed, dict):
            decoded = _float_or_none(speed.get("decode_tok_s_wall"))
            if decoded is not None:
                speeds.append(decoded)
        decoded = _float_or_none(value.get("decode_tok_s_wall"))
        if decoded is not None:
            speeds.append(decoded)
        log_tail = value.get("log_tail")
        if isinstance(log_tail, list):
            for line in log_tail:
                if not isinstance(line, str):
                    continue
                for match in re.finditer(r"\(([0-9]+(?:\.[0-9]+)?) tok/s\)", line):
                    speeds.append(float(match.group(1)))
        for child in value.values():
            speeds.extend(_collect_decode_speeds(child))
    elif isinstance(value, list):
        for child in value:
            speeds.extend(_collect_decode_speeds(child))
    return speeds


def _collect_structured_decode_speeds(value: Any) -> list[float]:
    speeds: list[float] = []
    if isinstance(value, dict):
        speed = value.get("speed")
        if isinstance(speed, dict):
            decoded = _float_or_none(speed.get("decode_tok_s_wall"))
            if decoded is not None:
                speeds.append(decoded)
        decoded = _float_or_none(value.get("decode_tok_s_wall"))
        if decoded is not None:
            speeds.append(decoded)
        for child in value.values():
            speeds.extend(_collect_structured_decode_speeds(child))
    elif isinstance(value, list):
        for child in value:
            speeds.extend(_collect_structured_decode_speeds(child))
    return speeds


def _collect_internal_generation_tps(value: Any) -> list[float]:
    speeds: list[float] = []
    if isinstance(value, dict):
        batch_generator = value.get("batch_generator")
        if isinstance(batch_generator, dict):
            decoded = _float_or_none(batch_generator.get("generation_tps"))
            if decoded is not None:
                speeds.append(decoded)
        decoded = _float_or_none(value.get("generation_tps"))
        if decoded is not None and "generation_tps" in value:
            speeds.append(decoded)
        for child in value.values():
            speeds.extend(_collect_internal_generation_tps(child))
    elif isinstance(value, list):
        for child in value:
            speeds.extend(_collect_internal_generation_tps(child))
    return list(dict.fromkeys(speeds))


def _batch_generation_counters(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, dict):
        return None, None
    after = value.get("after")
    if isinstance(after, dict):
        health = after.get("health")
        if isinstance(health, dict):
            body = health.get("body")
            if isinstance(body, dict):
                scheduler = body.get("scheduler")
                if isinstance(scheduler, dict):
                    batch_generator = scheduler.get("batch_generator")
                    if isinstance(batch_generator, dict):
                        tokens = _float_or_none(batch_generator.get("generation_tokens"))
                        seconds = _float_or_none(batch_generator.get("generation_time"))
                        if tokens is not None and seconds is not None:
                            return tokens, seconds
    return None, None


def _batch_prompt_seconds(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    after = value.get("after")
    if isinstance(after, dict):
        health = after.get("health")
        if isinstance(health, dict):
            body = health.get("body")
            if isinstance(body, dict):
                scheduler = body.get("scheduler")
                if isinstance(scheduler, dict):
                    batch_generator = scheduler.get("batch_generator")
                    if isinstance(batch_generator, dict):
                        return _float_or_none(batch_generator.get("prompt_time"))
    return None


def _collect_per_request_internal_generation_tps(payload: Any) -> list[float]:
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    speeds: list[float] = []
    previous_tokens = 0.0
    previous_seconds = 0.0
    for result in results:
        tokens, seconds = _batch_generation_counters(result)
        if tokens is None or seconds is None:
            continue
        delta_tokens = tokens - previous_tokens
        delta_seconds = seconds - previous_seconds
        if delta_tokens > 0 and delta_seconds > 0:
            speeds.append(round(delta_tokens / delta_seconds, 3))
        previous_tokens = tokens
        previous_seconds = seconds
    return speeds


def _collect_wall_vs_generation_time(payload: Any) -> list[dict[str, float]]:
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    details: list[dict[str, float]] = []
    previous_tokens = 0.0
    previous_seconds = 0.0
    previous_prompt_seconds = 0.0
    for result in results:
        if not isinstance(result, dict):
            continue
        tokens, seconds = _batch_generation_counters(result)
        if tokens is None or seconds is None:
            continue
        delta_tokens = tokens - previous_tokens
        delta_seconds = seconds - previous_seconds
        previous_tokens = tokens
        previous_seconds = seconds
        prompt_seconds = _batch_prompt_seconds(result)
        delta_prompt_seconds = None
        if prompt_seconds is not None:
            delta_prompt_seconds = prompt_seconds - previous_prompt_seconds
            previous_prompt_seconds = prompt_seconds
        if delta_tokens <= 0 or delta_seconds <= 0:
            continue
        speed = result.get("speed")
        if not isinstance(speed, dict):
            continue
        wall_seconds = _float_or_none(speed.get("wall_seconds"))
        if wall_seconds is None or wall_seconds <= 0:
            continue
        wall_decode = _float_or_none(speed.get("decode_tok_s_wall"))
        internal_tps = delta_tokens / delta_seconds
        overhead_seconds = max(0.0, wall_seconds - delta_seconds)
        item = {
            "completion_tokens": int(delta_tokens),
            "wall_seconds": round(wall_seconds, 3),
            "internal_generation_seconds": round(delta_seconds, 3),
            "overhead_seconds": round(overhead_seconds, 3),
            "wall_decode_tok_s": round(wall_decode, 3) if wall_decode is not None else 0.0,
            "internal_generation_tok_s": round(internal_tps, 3),
        }
        if delta_prompt_seconds is not None:
            scheduler_prompt_seconds = max(0.0, delta_prompt_seconds)
            item["scheduler_prompt_seconds"] = round(scheduler_prompt_seconds, 3)
            item["residual_overhead_seconds"] = round(
                max(0.0, overhead_seconds - scheduler_prompt_seconds),
                3,
            )
        details.append(item)
    return details


def _collect_cache_details(value: Any) -> list[str]:
    details: list[str] = []
    if isinstance(value, dict):
        speed = value.get("speed")
        if isinstance(speed, dict):
            detail = speed.get("cache_detail")
            if isinstance(detail, str) and detail:
                details.append(detail)
        usage_summary = value.get("usage_summary")
        if isinstance(usage_summary, dict):
            detail = usage_summary.get("cache_detail")
            if isinstance(detail, str) and detail:
                details.append(detail)
        usage = value.get("usage")
        if isinstance(usage, dict):
            prompt_details = usage.get("prompt_tokens_details") or usage.get(
                "input_tokens_details"
            )
            if isinstance(prompt_details, dict):
                detail = prompt_details.get("cache_detail")
                if isinstance(detail, str) and detail:
                    details.append(detail)
        for child in value.values():
            details.extend(_collect_cache_details(child))
    elif isinstance(value, list):
        for child in value:
            details.extend(_collect_cache_details(child))
    return list(dict.fromkeys(details))


def _gemma4_sustained_cachehit_diagnostic(
    payload: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    cold_speeds: list[float] = []
    cachehit_speeds: list[float] = []
    cachehit_details: list[str] = []
    cachehit_cached_tokens: list[int] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        speed = item.get("speed") if isinstance(item.get("speed"), dict) else {}
        usage = (
            item.get("usage_summary")
            if isinstance(item.get("usage_summary"), dict)
            else {}
        )
        decoded = _float_or_none(speed.get("decode_tok_s_wall"))
        cached_tokens = _float_or_none(
            speed.get("cached_tokens", usage.get("cached_tokens"))
        )
        cache_detail = speed.get("cache_detail") or usage.get("cache_detail")
        if decoded is None:
            continue
        if cached_tokens is not None and cached_tokens > 0:
            cachehit_speeds.append(decoded)
            cachehit_cached_tokens.append(int(cached_tokens))
            if isinstance(cache_detail, str) and cache_detail:
                cachehit_details.append(cache_detail)
        else:
            cold_speeds.append(decoded)

    min_cachehit = min(cachehit_speeds) if cachehit_speeds else None
    cold = cold_speeds[0] if cold_speeds else None
    per_request_internal_generation_tps = _collect_per_request_internal_generation_tps(
        payload
    )
    wall_vs_generation_time = _collect_wall_vs_generation_time(payload)
    return {
        "artifact": GEMMA4_MIXED_SWA_SUSTAINED_CACHEHIT_DIAGNOSTIC_REL,
        "artifact_present": _path_present(
            root, GEMMA4_MIXED_SWA_SUSTAINED_CACHEHIT_DIAGNOSTIC_REL
        ),
        "artifact_status": payload.get("status"),
        "python": payload.get("python"),
        "native_cache_schema": _find_native_cache_schema(payload),
        "native_cache_generic_tq_enabled": _find_native_cache_generic_tq_enabled(
            payload
        ),
        "cold_decode_tok_s_wall": cold,
        "cold_turn_clears_floor": (
            cold is not None and cold >= GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S
        ),
        "cachehit_decode_tok_s_wall": cachehit_speeds,
        "cachehit_cached_tokens": cachehit_cached_tokens,
        "cachehit_cache_details": list(dict.fromkeys(cachehit_details)),
        "per_request_internal_generation_tps": per_request_internal_generation_tps,
        "min_internal_generation_tok_s": (
            min(per_request_internal_generation_tps)
            if per_request_internal_generation_tps
            else None
        ),
        "wall_vs_generation_time": wall_vs_generation_time,
        "max_overhead_seconds": (
            max(item["overhead_seconds"] for item in wall_vs_generation_time)
            if wall_vs_generation_time
            else None
        ),
        "min_cachehit_decode_tok_s_wall": min_cachehit,
        "cachehit_turns_clear_floor": bool(
            cachehit_speeds
            and min_cachehit is not None
            and min_cachehit >= GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S
        ),
    }


def _gemma4_short_nocache_repeat_diagnostic(
    payload: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    speeds: list[float] = []
    completion_tokens: list[int] = []
    cached_tokens_seen: list[int] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        speed = item.get("speed") if isinstance(item.get("speed"), dict) else {}
        usage = (
            item.get("usage_summary")
            if isinstance(item.get("usage_summary"), dict)
            else {}
        )
        decoded = _float_or_none(speed.get("decode_tok_s_wall"))
        cached_tokens = _float_or_none(
            speed.get("cached_tokens", usage.get("cached_tokens"))
        )
        if decoded is None:
            continue
        if cached_tokens is not None and cached_tokens > 0:
            continue
        speeds.append(decoded)
        cached_tokens_seen.append(int(cached_tokens or 0))
        token_count = _float_or_none(usage.get("completion_tokens"))
        if token_count is not None:
            completion_tokens.append(int(token_count))
    minimum = min(speeds) if speeds else None
    per_request_internal_generation_tps = _collect_per_request_internal_generation_tps(
        payload
    )
    wall_vs_generation_time = _collect_wall_vs_generation_time(payload)
    return {
        "artifact": GEMMA4_MIXED_SWA_SHORT_NOCACHE_REPEAT_DIAGNOSTIC_REL,
        "artifact_present": _path_present(
            root, GEMMA4_MIXED_SWA_SHORT_NOCACHE_REPEAT_DIAGNOSTIC_REL
        ),
        "artifact_status": payload.get("status"),
        "python": payload.get("python"),
        "decode_tok_s_wall": speeds,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens_seen,
        "per_request_internal_generation_tps": per_request_internal_generation_tps,
        "min_internal_generation_tok_s": (
            min(per_request_internal_generation_tps)
            if per_request_internal_generation_tps
            else None
        ),
        "wall_vs_generation_time": wall_vs_generation_time,
        "max_overhead_seconds": (
            max(item["overhead_seconds"] for item in wall_vs_generation_time)
            if wall_vs_generation_time
            else None
        ),
        "min_decode_tok_s_wall": minimum,
        "turns_clear_floor": bool(
            speeds
            and minimum is not None
            and minimum >= GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S
        ),
    }


def _gemma4_short_nocache_scheduler_trace_diagnostic(
    payload: dict[str, Any],
    root: Path,
    artifact_rel: str = GEMMA4_MIXED_SWA_SHORT_NOCACHE_SCHEDULER_TRACE_REL,
) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    speeds: list[float] = []
    completion_tokens: list[int] = []
    cached_tokens_seen: list[int] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        speed = item.get("speed") if isinstance(item.get("speed"), dict) else {}
        usage = (
            item.get("usage_summary")
            if isinstance(item.get("usage_summary"), dict)
            else {}
        )
        decoded = _float_or_none(speed.get("decode_tok_s_wall"))
        cached_tokens = _float_or_none(
            speed.get("cached_tokens", usage.get("cached_tokens"))
        )
        if decoded is None or (cached_tokens is not None and cached_tokens > 0):
            continue
        speeds.append(decoded)
        cached_tokens_seen.append(int(cached_tokens or 0))
        token_count = _float_or_none(usage.get("completion_tokens"))
        if token_count is not None:
            completion_tokens.append(int(token_count))

    log_tail = payload.get("log_tail") if isinstance(payload.get("log_tail"), list) else []
    prefill_total_ms: list[float] = []
    scheduler_batch_next_ms: list[float] = []
    scheduler_process_cleanup_ms: list[float] = []
    scheduler_cleanup_finished_ms: list[float] = []
    decode_next_last_total_ms: list[float] = []
    decode_next_last_step_ms: list[float] = []
    decode_next_last_async_ms: list[float] = []
    decode_next_last_materialize_ms: list[float] = []
    for line_obj in log_tail:
        line = str(line_obj)
        if "VMLINUX_MLLM_PREFILL_TRACE" in line:
            match = re.search(r"\btotal_ms=([0-9.]+)", line)
            if match:
                prefill_total_ms.append(round(float(match.group(1)), 3))
        if "VMLINUX_MLLM_SCHEDULER_TRACE" in line:
            for key, target in (
                ("batch_next_ms", scheduler_batch_next_ms),
                ("process_cleanup_ms", scheduler_process_cleanup_ms),
                ("cleanup_finished_ms", scheduler_cleanup_finished_ms),
            ):
                match = re.search(rf"\b{key}=([0-9.]+)", line)
                if match:
                    target.append(round(float(match.group(1)), 3))
        if "VMLINUX_DECODE_TRACE_NEXT" in line:
            for key, target in (
                ("last_total_ms", decode_next_last_total_ms),
                ("last_step_ms", decode_next_last_step_ms),
                ("last_async_ms", decode_next_last_async_ms),
                ("last_materialize_ms", decode_next_last_materialize_ms),
            ):
                match = re.search(rf"\b{key}=([0-9.]+)", line)
                if match:
                    target.append(round(float(match.group(1)), 3))

    max_async_ms = (
        max(decode_next_last_async_ms) if decode_next_last_async_ms else None
    )
    max_total_ms = (
        max(decode_next_last_total_ms) if decode_next_last_total_ms else None
    )
    async_decode_wait_dominates = bool(
        max_async_ms is not None
        and max_total_ms is not None
        and max_total_ms > 0
        and (max_async_ms / max_total_ms) >= 0.5
    )
    return {
        "artifact": artifact_rel,
        "artifact_present": _path_present(root, artifact_rel),
        "artifact_status": payload.get("status"),
        "python": payload.get("python"),
        "decode_tok_s_wall": speeds,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens_seen,
        "min_decode_tok_s_wall": min(speeds) if speeds else None,
        "turns_clear_floor": bool(
            speeds
            and min(speeds) >= GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S
        ),
        "prefill_total_ms": prefill_total_ms,
        "scheduler_batch_next_ms": scheduler_batch_next_ms,
        "scheduler_process_cleanup_ms": scheduler_process_cleanup_ms,
        "scheduler_cleanup_finished_ms": scheduler_cleanup_finished_ms,
        "decode_next_last_total_ms": decode_next_last_total_ms,
        "decode_next_last_step_ms": decode_next_last_step_ms,
        "decode_next_last_async_ms": decode_next_last_async_ms,
        "decode_next_last_materialize_ms": decode_next_last_materialize_ms,
        "max_decode_next_last_async_ms": max_async_ms,
        "async_decode_wait_dominates": async_decode_wait_dominates,
    }


def _gemma4_short_nocache_streaming_diagnostic(
    payload: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    wall_speeds: list[float] = []
    stream_speeds: list[float] = []
    ttft_seconds: list[float] = []
    post_first_token_seconds: list[float] = []
    cached_tokens_seen: list[int] = []
    completion_tokens_seen: list[int] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        speed = item.get("speed") if isinstance(item.get("speed"), dict) else {}
        stream_speed = (
            item.get("stream_speed")
            if isinstance(item.get("stream_speed"), dict)
            else {}
        )
        usage = (
            item.get("usage_summary")
            if isinstance(item.get("usage_summary"), dict)
            else {}
        )
        cached_tokens = _float_or_none(
            stream_speed.get(
                "cached_tokens",
                speed.get("cached_tokens", usage.get("cached_tokens")),
            )
        )
        if cached_tokens is not None and cached_tokens > 0:
            continue
        completion_tokens = _float_or_none(
            stream_speed.get("completion_tokens", usage.get("completion_tokens"))
        )
        wall = _float_or_none(speed.get("decode_tok_s_wall"))
        stream = _float_or_none(stream_speed.get("decode_tok_s_stream"))
        ttft = _float_or_none(stream_speed.get("ttft_seconds"))
        post_first = _float_or_none(stream_speed.get("post_first_token_seconds"))
        if wall is not None:
            wall_speeds.append(wall)
        if stream is not None:
            stream_speeds.append(stream)
        if ttft is not None:
            ttft_seconds.append(ttft)
        if post_first is not None:
            post_first_token_seconds.append(post_first)
        cached_tokens_seen.append(int(cached_tokens or 0))
        if completion_tokens is not None:
            completion_tokens_seen.append(int(completion_tokens))
    min_wall = min(wall_speeds) if wall_speeds else None
    min_stream = min(stream_speeds) if stream_speeds else None
    min_completion_tokens = (
        min(completion_tokens_seen) if completion_tokens_seen else None
    )
    installed_app_python = "/Applications/vMLX.app/" in str(payload.get("python") or "")
    all_uncached = bool(cached_tokens_seen) and all(value == 0 for value in cached_tokens_seen)
    sustained_completion = bool(
        len(completion_tokens_seen) >= GEMMA4_MIXED_SWA_STREAMING_MIN_TURNS
        and min_completion_tokens is not None
        and min_completion_tokens >= GEMMA4_MIXED_SWA_STREAMING_MIN_COMPLETION_TOKENS
    )
    return {
        "artifact": GEMMA4_MIXED_SWA_SHORT_NOCACHE_STREAMING_REL,
        "artifact_present": _path_present(
            root, GEMMA4_MIXED_SWA_SHORT_NOCACHE_STREAMING_REL
        ),
        "artifact_status": payload.get("status"),
        "python": payload.get("python"),
        "request_route": payload.get("request_route"),
        "installed_app_python": installed_app_python,
        "chat_route": payload.get("request_route") == "chat",
        "decode_tok_s_wall": wall_speeds,
        "decode_tok_s_stream": stream_speeds,
        "ttft_seconds": ttft_seconds,
        "post_first_token_seconds": post_first_token_seconds,
        "cached_tokens": cached_tokens_seen,
        "completion_tokens": completion_tokens_seen,
        "all_uncached": all_uncached,
        "min_completion_tokens": min_completion_tokens,
        "min_required_completion_tokens": (
            GEMMA4_MIXED_SWA_STREAMING_MIN_COMPLETION_TOKENS
        ),
        "min_required_turns": GEMMA4_MIXED_SWA_STREAMING_MIN_TURNS,
        "sustained_completion": sustained_completion,
        "min_decode_tok_s_wall": min_wall,
        "min_decode_tok_s_stream": min_stream,
        "wall_turns_clear_floor": bool(
            wall_speeds
            and min_wall is not None
            and min_wall >= GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S
        ),
        "stream_turns_clear_floor": bool(
            stream_speeds
            and min_stream is not None
            and min_stream >= GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S
            and sustained_completion
        ),
    }


def _gemma4_mixed_swa_speed_floor_detail(
    payloads: list[tuple[str, dict[str, Any]]],
    root: Path,
    sustained_cachehit_payload: dict[str, Any] | None = None,
    short_nocache_repeat_payload: dict[str, Any] | None = None,
    short_nocache_scheduler_trace_payload: dict[str, Any] | None = None,
    short_nocache_sync_eval_ab_payload: dict[str, Any] | None = None,
    short_nocache_streaming_payload: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    artifact_details: list[dict[str, Any]] = []
    all_speeds: list[float] = []
    all_structured_speeds: list[float] = []
    for rel, payload in payloads:
        present = _path_present(root, rel)
        speeds = _collect_decode_speeds(payload)
        structured_speeds = _collect_structured_decode_speeds(payload)
        internal_generation_tps = _collect_internal_generation_tps(payload)
        per_request_internal_generation_tps = (
            _collect_per_request_internal_generation_tps(payload)
        )
        wall_vs_generation_time = _collect_wall_vs_generation_time(payload)
        cache_details = _collect_cache_details(payload)
        all_speeds.extend(speeds)
        all_structured_speeds.extend(structured_speeds)
        schema = _find_native_cache_schema(payload)
        generic_tq_enabled = _find_native_cache_generic_tq_enabled(payload)
        mixed_swa_runtime_contract = (
            schema == "mixed_swa_kv_v1" and generic_tq_enabled is not True
        )
        min_decode_tok_s = min(speeds) if speeds else None
        min_internal_generation_tok_s = (
            min(per_request_internal_generation_tps)
            if per_request_internal_generation_tps
            else None
        )
        clears_speed_floor = bool(
            speeds and min_decode_tok_s >= GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S
        )
        clears_internal_generation_floor = bool(
            per_request_internal_generation_tps
            and min_internal_generation_tok_s >= GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S
        )
        artifact_details.append(
            {
                "artifact": rel,
                "artifact_present": present,
                "artifact_status": payload.get("status"),
                "row": payload.get("row"),
                "request_route": payload.get("request_route"),
                "native_cache_schema": schema,
                "native_cache_generic_tq_enabled": generic_tq_enabled,
                "mixed_swa_runtime_contract": mixed_swa_runtime_contract,
                "cache_details": cache_details,
                "decode_tok_s_wall": speeds,
                "structured_decode_tok_s_wall": structured_speeds,
                "internal_generation_tps": internal_generation_tps,
                "per_request_internal_generation_tps": per_request_internal_generation_tps,
                "wall_vs_generation_time": wall_vs_generation_time,
                "max_overhead_seconds": (
                    max(
                        item["overhead_seconds"]
                        for item in wall_vs_generation_time
                    )
                    if wall_vs_generation_time
                    else None
                ),
                "min_decode_tok_s": min_decode_tok_s,
                "min_internal_generation_tok_s": min_internal_generation_tok_s,
                "clears_speed_floor": clears_speed_floor,
                "clears_internal_generation_floor": clears_internal_generation_floor,
                "has_structured_speed": bool(structured_speeds),
            }
        )
    all_required_artifacts_present = all(
        item["artifact_present"] for item in artifact_details
    )
    all_required_artifacts_pass = all(
        item["artifact_status"] == "pass" for item in artifact_details
    )
    all_required_artifacts_mixed_swa = all(
        item["mixed_swa_runtime_contract"]
        for item in artifact_details
    )
    all_decode_speeds_clear_floor = all(
        bool(item["decode_tok_s_wall"]) and item["clears_speed_floor"]
        for item in artifact_details
    )
    all_required_artifacts_have_structured_speed = all(
        item["has_structured_speed"] for item in artifact_details
    )
    all_internal_generation_speeds_clear_floor = all(
        bool(item["internal_generation_tps"])
        and item["clears_internal_generation_floor"]
        for item in artifact_details
    )
    sustained_cachehit_diagnostic = _gemma4_sustained_cachehit_diagnostic(
        sustained_cachehit_payload or {},
        root,
    )
    sustained_cachehit_blocks_clearance = bool(
        sustained_cachehit_diagnostic["artifact_present"]
        and sustained_cachehit_diagnostic["artifact_status"] == "pass"
        and (
            sustained_cachehit_diagnostic["cold_turn_clears_floor"] is False
            or sustained_cachehit_diagnostic["cachehit_turns_clear_floor"] is False
        )
    )
    short_nocache_repeat_diagnostic = _gemma4_short_nocache_repeat_diagnostic(
        short_nocache_repeat_payload or {},
        root,
    )
    short_nocache_repeat_blocks_clearance = bool(
        short_nocache_repeat_diagnostic["artifact_present"]
        and short_nocache_repeat_diagnostic["artifact_status"] == "pass"
        and short_nocache_repeat_diagnostic["turns_clear_floor"] is False
    )
    short_nocache_scheduler_trace = (
        _gemma4_short_nocache_scheduler_trace_diagnostic(
            short_nocache_scheduler_trace_payload or {},
            root,
        )
    )
    short_nocache_sync_eval_ab = (
        _gemma4_short_nocache_scheduler_trace_diagnostic(
            short_nocache_sync_eval_ab_payload or {},
            root,
            GEMMA4_MIXED_SWA_SHORT_NOCACHE_SYNC_EVAL_AB_REL,
        )
    )
    short_nocache_streaming_diagnostic = (
        _gemma4_short_nocache_streaming_diagnostic(
            short_nocache_streaming_payload or {},
            root,
        )
    )
    legacy_wall_speed_ok = (
        all_required_artifacts_present
        and all_required_artifacts_pass
        and all_required_artifacts_mixed_swa
        and all_decode_speeds_clear_floor
        and all_internal_generation_speeds_clear_floor
        and all_required_artifacts_have_structured_speed
        and not sustained_cachehit_blocks_clearance
        and not short_nocache_repeat_blocks_clearance
    )
    app_streaming_speed_floor_clears = bool(
        short_nocache_streaming_diagnostic["artifact_present"]
        and short_nocache_streaming_diagnostic["artifact_status"] == "pass"
        and short_nocache_streaming_diagnostic["installed_app_python"] is True
        and short_nocache_streaming_diagnostic["chat_route"] is True
        and short_nocache_streaming_diagnostic["all_uncached"] is True
        and short_nocache_streaming_diagnostic["stream_turns_clear_floor"] is True
    )
    ok = (
        all_required_artifacts_present
        and all_required_artifacts_pass
        and all_required_artifacts_mixed_swa
        and all_required_artifacts_have_structured_speed
        and (legacy_wall_speed_ok or app_streaming_speed_floor_clears)
    )
    failed_speed_floor_artifacts: list[dict[str, Any]] = []
    for item in artifact_details:
        failed_metrics: list[str] = []
        if not item["clears_speed_floor"]:
            failed_metrics.append("wall_decode")
        if item["per_request_internal_generation_tps"] and not item[
            "clears_internal_generation_floor"
        ]:
            failed_metrics.append("internal_generation")
        if failed_metrics:
            failed_speed_floor_artifacts.append(
                {
                    "artifact": item["artifact"],
                    "min_decode_tok_s": item["min_decode_tok_s"],
                    "min_internal_generation_tok_s": item[
                        "min_internal_generation_tok_s"
                    ],
                    "failed_metrics": failed_metrics,
                }
            )
    return ok, {
        "speed_floor_tok_s": GEMMA4_MIXED_SWA_SPEED_FLOOR_TOK_S,
        "artifacts": artifact_details,
        "sustained_cachehit_diagnostic": sustained_cachehit_diagnostic,
        "sustained_cachehit_blocks_clearance": sustained_cachehit_blocks_clearance,
        "short_nocache_repeat_diagnostic": short_nocache_repeat_diagnostic,
        "short_nocache_repeat_blocks_clearance": short_nocache_repeat_blocks_clearance,
        "short_nocache_scheduler_trace": short_nocache_scheduler_trace,
        "short_nocache_sync_eval_ab": short_nocache_sync_eval_ab,
        "short_nocache_streaming_diagnostic": short_nocache_streaming_diagnostic,
        "legacy_wall_speed_ok": legacy_wall_speed_ok,
        "app_streaming_speed_floor_clears": app_streaming_speed_floor_clears,
        "all_required_artifacts_present": all_required_artifacts_present,
        "all_required_artifacts_pass": all_required_artifacts_pass,
        "all_required_artifacts_mixed_swa": all_required_artifacts_mixed_swa,
        "all_decode_speeds_clear_floor": all_decode_speeds_clear_floor,
        "all_internal_generation_speeds_clear_floor": all_internal_generation_speeds_clear_floor,
        "all_required_artifacts_have_structured_speed": all_required_artifacts_have_structured_speed,
        "failed_speed_floor_artifacts": failed_speed_floor_artifacts,
        "max_wall_generation_overhead_seconds": (
            max(
                item["max_overhead_seconds"]
                for item in artifact_details
                if item["max_overhead_seconds"] is not None
            )
            if any(
                item["max_overhead_seconds"] is not None
                for item in artifact_details
            )
            else None
        ),
        "min_decode_tok_s": min(all_speeds) if all_speeds else None,
        "max_decode_tok_s": max(all_speeds) if all_speeds else None,
        "min_structured_decode_tok_s": (
            min(all_structured_speeds) if all_structured_speeds else None
        ),
    }


def _smoke_family_key(row: dict[str, Any]) -> str | None:
    name = str(row.get("name") or "").lower()
    model_type = str(row.get("model_type") or "").lower()
    path = str(row.get("path") or "").lower()
    blob = " ".join((name, model_type, path))
    if "deepseek-v4" in blob or "deepseek_v4" in blob or "dsv4" in blob:
        return "dsv4"
    if "gemma" in blob:
        return "gemma4"
    if "hy3" in blob or "hy_v3" in blob:
        return "hy3"
    if "ling" in blob or "bailing" in blob:
        return "ling_bailing"
    if "minimax" in blob:
        return "minimax"
    if "mimo" in blob or model_type == "mimo_v2":
        return "mimo_v2"
    if "nemotron" in blob:
        return "nemotron"
    if "qwen3.6" in blob or "qwen3_5" in blob or "qwen36" in blob:
        return "qwen36"
    if "lfm2" in blob or "lfm2.5" in blob:
        return "lfm"
    if "step-3.7" in blob or "step3p7" in blob or model_type == "step3p7":
        return "step3p7"
    if "zaya1-vl" in blob or model_type == "zaya1_vl":
        return "zaya_vl"
    if "zaya" in blob:
        return "zaya_text"
    return None


def _effective_smoke_supports_video(result: dict[str, Any]) -> bool:
    capabilities = result.get("capabilities")
    if isinstance(capabilities, dict):
        body = capabilities.get("body")
        if isinstance(body, dict):
            modalities = body.get("modalities")
            if isinstance(modalities, list):
                return any(str(item).lower() == "video" for item in modalities)

    row = result.get("row")
    return isinstance(row, dict) and row.get("supports_video") is True


SMOKE_REQUIRED_REQUEST_LABELS_BY_FAMILY = {
    "dsv4": ("reasoning_on",),
    "gemma4": ("reasoning_on",),
    "hy3": ("reasoning_on",),
    "minimax": ("reasoning_on",),
    "mimo_v2": ("reasoning_on",),
    "qwen36": ("reasoning_on",),
    "step3p7": ("reasoning_on",),
}
SMOKE_REQUIRED_CACHE_HIT_FAMILIES = set(ALL_LOCAL_MODEL_SMOKE_REQUIRED_FAMILIES)
SMOKE_UNEXPECTED_CJK_RE = re.compile(
    r"[\u1100-\u11ff\u3040-\u309f\u30a0-\u30ff\u3130-\u318f"
    r"\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7af"
    r"\ud7b0-\ud7ff\uf900-\ufaff\uff66-\uff9f"
    r"\U00020000-\U0002a6df\U0002a700-\U0002b73f"
    r"\U0002b740-\U0002b81f\U0002b820-\U0002ceaf"
    r"\U0002ceb0-\U0002ebef\U0002f800-\U0002fa1f"
    r"\U00030000-\U0003134f]"
)
SMOKE_MAX_REASONING_ON_CHARS = 4096


def _smoke_semantic_failures_from_request(request: dict[str, Any]) -> list[dict[str, Any]]:
    label = str(request.get("label") or "unknown")
    content = str(request.get("content") or "")
    failures: list[dict[str, Any]] = []
    cjk_chars = len(SMOKE_UNEXPECTED_CJK_RE.findall(content))
    if cjk_chars:
        failures.append(
            {
                "label": label,
                "reason": "unexpected_cjk_visible_text",
                "cjk_chars": cjk_chars,
            }
        )
    if label == "reasoning_on":
        try:
            reasoning_chars = int(request.get("reasoning_chars") or 0)
        except (TypeError, ValueError):
            reasoning_chars = 0
        if reasoning_chars > SMOKE_MAX_REASONING_ON_CHARS:
            failures.append(
                {
                    "label": label,
                    "reason": "reasoning_loop_too_long",
                    "reasoning_chars": reasoning_chars,
                    "max_reasoning_chars": SMOKE_MAX_REASONING_ON_CHARS,
                }
            )
    return failures


def _all_local_model_smoke_detail(
    payloads: list[tuple[str, dict[str, Any]]],
    root: Path,
    diagnostics: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, dict[str, Any]]:
    diagnostics = diagnostics or {}
    artifact_details: list[dict[str, Any]] = []
    covered: set[str] = set()
    for rel, payload in payloads:
        present = _path_present(root, rel)
        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        statuses = [
            str(item.get("status"))
            for item in results
            if isinstance(item, dict) and item.get("status") is not None
        ]
        validation_failures: list[dict[str, Any]] = []
        labels: list[str] = []
        cache_hit_observed = False
        family_keys: set[str] = set()
        family_statuses: dict[str, list[str]] = {}
        family_labels: dict[str, set[str]] = {}
        family_validation_failures: dict[str, list[dict[str, Any]]] = {}
        family_cache_hit_observed: dict[str, bool] = {}
        required_labels_by_family: dict[str, set[str]] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            row = item.get("row")
            key = None
            if isinstance(row, dict):
                key = _smoke_family_key(row)
                if key:
                    family_keys.add(key)
                    family_statuses.setdefault(key, []).append(
                        str(item.get("status") or "open")
                    )
                    family_labels.setdefault(key, set())
                    family_validation_failures.setdefault(key, [])
                    family_cache_hit_observed.setdefault(key, False)
                    required = required_labels_by_family.setdefault(
                        key,
                        set(SMOKE_REQUIRED_REQUEST_LABELS_BY_FAMILY.get(key, ())),
                    )
                    if _effective_smoke_supports_video(item):
                        required.update(("text_no_media_after_video", "vl_blue_video"))
            for request in item.get("requests") or []:
                if not isinstance(request, dict):
                    continue
                label = request.get("label")
                if isinstance(label, str):
                    labels.append(label)
                    if key:
                        family_labels.setdefault(key, set()).add(label)
                failures = request.get("validation_failures")
                if isinstance(failures, list):
                    typed_failures = [
                        failure for failure in failures if isinstance(failure, dict)
                    ]
                    validation_failures.extend(typed_failures)
                    if key:
                        family_validation_failures.setdefault(key, []).extend(
                            typed_failures
                        )
                semantic_failures = _smoke_semantic_failures_from_request(request)
                for failure in semantic_failures:
                    if failure not in validation_failures:
                        validation_failures.append(failure)
                    if key and failure not in family_validation_failures.setdefault(
                        key, []
                    ):
                        family_validation_failures[key].append(failure)
                cache_summary = request.get("cache_summary")
                if isinstance(cache_summary, dict) and cache_summary.get("has_cache_hit") is True:
                    cache_hit_observed = True
                    if key:
                        family_cache_hit_observed[key] = True
        missing_required_labels = sorted(
            {
                label
                for family, required_labels in required_labels_by_family.items()
                for label in required_labels
                if label not in labels
            }
        )
        missing_required_labels_by_family = {
            family: sorted(
                label
                for label in required_labels
                if label not in family_labels.get(family, set())
            )
            for family, required_labels in required_labels_by_family.items()
        }
        missing_required_labels_by_family = {
            family: labels
            for family, labels in missing_required_labels_by_family.items()
            if labels
        }
        missing_cache_hit_family_keys = sorted(
            family
            for family in family_keys
            if family in SMOKE_REQUIRED_CACHE_HIT_FAMILIES
            and not family_cache_hit_observed.get(family, False)
        )
        passing_family_keys = sorted(
            family
            for family in family_keys
            if present
            and family_statuses.get(family)
            and all(status == "pass" for status in family_statuses[family])
            and not family_validation_failures.get(family)
            and family not in missing_required_labels_by_family
            and family not in missing_cache_hit_family_keys
        )
        failing_family_keys = sorted(family_keys.difference(passing_family_keys))
        artifact_pass = (
            present
            and payload.get("completed") == payload.get("row_count")
            and statuses
            and all(status == "pass" for status in statuses)
            and not validation_failures
            and not missing_required_labels
            and not missing_cache_hit_family_keys
        )
        covered.update(passing_family_keys)
        detail = {
            "artifact": rel,
            "artifact_present": present,
            "status": "pass" if artifact_pass else payload.get("status") or "open",
            "completed": payload.get("completed"),
            "row_count": payload.get("row_count"),
            "family_keys": sorted(family_keys),
            "passing_family_keys": passing_family_keys,
            "failing_family_keys": failing_family_keys,
            "request_labels": sorted(set(labels)),
            "missing_required_labels": missing_required_labels,
            "missing_required_labels_by_family": missing_required_labels_by_family,
            "missing_cache_hit_family_keys": missing_cache_hit_family_keys,
            "cache_hit_observed": cache_hit_observed,
            "validation_failures": validation_failures,
        }
        if rel in diagnostics:
            detail["diagnostics"] = diagnostics[rel]
        artifact_details.append(detail)
    missing = [
        family
        for family in ALL_LOCAL_MODEL_SMOKE_REQUIRED_FAMILIES
        if family not in covered
    ]
    missing_cache_hit = sorted(
        {
            family
            for item in artifact_details
            for family in item["missing_cache_hit_family_keys"]
        }
    )
    non_mimo_missing = [family for family in missing if family != "mimo_v2"]
    non_mimo_not_pass = [
        item["artifact"]
        for item in artifact_details
        if any(family != "mimo_v2" for family in item["failing_family_keys"])
    ]
    not_pass_required_family_artifacts: dict[str, list[str]] = {}
    for item in artifact_details:
        artifact = str(item["artifact"])
        for family in item["failing_family_keys"]:
            if family in ALL_LOCAL_MODEL_SMOKE_REQUIRED_FAMILIES:
                not_pass_required_family_artifacts.setdefault(family, []).append(artifact)
    expected_artifact_by_family: dict[str, list[str]] = {}
    for item in artifact_details:
        artifact = str(item["artifact"])
        for family in item["family_keys"]:
            if family in ALL_LOCAL_MODEL_SMOKE_REQUIRED_FAMILIES:
                expected_artifact_by_family.setdefault(family, []).append(artifact)
    ok = not missing and all(item["status"] == "pass" for item in artifact_details)
    only_mimo_open = (
        not ok
        and not non_mimo_missing
        and not non_mimo_not_pass
        and set(missing).issubset({"mimo_v2"})
        and set(not_pass_required_family_artifacts).issubset({"mimo_v2"})
    )
    release_boundary = (
        "global_matrix_all_families_pass"
        if ok
        else (
            "non_mimo_live_smoke_clear_mimo_v2_deferred"
            if only_mimo_open
            else "global_matrix_requires_all_families"
        )
    )
    return ok, {
        "required_family_keys": list(ALL_LOCAL_MODEL_SMOKE_REQUIRED_FAMILIES),
        "required_cache_hit_family_keys": sorted(SMOKE_REQUIRED_CACHE_HIT_FAMILIES),
        "covered_family_keys": sorted(covered),
        "missing_required_family_keys": missing,
        "missing_cache_hit_family_keys": missing_cache_hit,
        "non_mimo_status": (
            "pass" if not non_mimo_missing and not non_mimo_not_pass else "open"
        ),
        "non_mimo_missing_required_family_keys": non_mimo_missing,
        "non_mimo_not_pass_artifacts": non_mimo_not_pass,
        "not_pass_required_family_artifacts": {
            family: sorted(set(artifacts))
            for family, artifacts in sorted(
                not_pass_required_family_artifacts.items()
            )
        },
        "blocking_required_family_artifacts": {
            family: sorted(
                set(
                    expected_artifact_by_family.get(family)
                    or ALL_LOCAL_MODEL_SMOKE_ARTIFACTS_BY_FAMILY.get(family, [])
                )
            )
            for family in missing
        },
        "release_boundary": release_boundary,
        "mimo_v2_deferred": only_mimo_open,
        "artifacts": artifact_details,
    }


def _mimo_v2_jang2l_quality_detail(root: Path) -> tuple[bool, dict[str, Any]]:
    artifacts = {
        "structural_verify": MIMO_V2_JANG2L_STRUCTURAL_VERIFY_REL,
        "text_cache": MIMO_V2_JANG2L_TEXT_CACHE_REL,
        "switchglu_parity": MIMO_V2_JANG2L_SWITCHGLU_PARITY_REL,
        "length_sweep": MIMO_V2_JANG2L_LENGTH_SWEEP_REL,
        "tool_dialect": MIMO_V2_JANG2L_TOOL_DIALECT_REL,
        "current_audit": MIMO_V2_JANG2L_CURRENT_AUDIT_REL,
        "conservative_diagnostic": MIMO_V2_JANG2L_CONSERVATIVE_DIAGNOSTIC_REL,
    }
    payloads = {key: _load(root, rel) for key, rel in artifacts.items()}
    missing = [
        rel
        for rel in artifacts.values()
        if not (root / rel).exists()
    ]
    current_audit = payloads["current_audit"]
    current_audit_component_ok = current_audit.get("component_ok")
    if not isinstance(current_audit_component_ok, dict):
        current_audit_component_ok = {}
    manifest_integrity_passed = current_audit_component_ok.get("manifest_integrity") is True
    stale_local_state_absent = (
        current_audit_component_ok.get("stale_local_state_absent") is True
    )

    structural_pass = payloads["structural_verify"].get("status") == "pass"

    text_requests = payloads["text_cache"].get("requests")
    if not isinstance(text_requests, list):
        text_requests = []
    text_outputs = [
        str(item.get("content") or "")
        for item in text_requests
        if isinstance(item, dict)
    ]
    cached_tokens = []
    for item in text_requests:
        if not isinstance(item, dict):
            continue
        usage = item.get("usage")
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        value = details.get("cached_tokens") if isinstance(details, dict) else None
        if isinstance(value, int):
            cached_tokens.append(value)
    text_cache_narrow_pass = (
        len(text_outputs) >= 2
        and all(output == "cache ok" for output in text_outputs[:2])
        and any(value > 0 for value in cached_tokens)
    )

    switchglu_max_abs_diff = payloads["switchglu_parity"].get("max_abs_diff")
    switchglu_mean_abs_diff = payloads["switchglu_parity"].get("mean_abs_diff")
    switchglu_parity_pass = (
        isinstance(switchglu_max_abs_diff, (int, float))
        and float(switchglu_max_abs_diff) <= 0.002
        and isinstance(switchglu_mean_abs_diff, (int, float))
        and float(switchglu_mean_abs_diff) <= 0.001
    )

    length_cases = payloads["length_sweep"].get("cases")
    if not isinstance(length_cases, list):
        length_cases = []
    corrupt_length_cases = [
        {
            "prompt_tokens": item.get("prompt_tokens"),
            "status": item.get("status"),
            "output": item.get("output"),
        }
        for item in length_cases
        if isinstance(item, dict)
        and isinstance(item.get("prompt_tokens"), int)
        and item.get("prompt_tokens") >= 148
        and "fail" in str(item.get("status") or "")
    ]
    prompt_length_coherence_blocked = (
        payloads["length_sweep"].get("status") == "fail"
        and bool(corrupt_length_cases)
    )

    tool_observations = payloads["tool_dialect"].get("runtime_observations")
    if not isinstance(tool_observations, list):
        tool_observations = []
    tool_protocol_blocked = (
        payloads["tool_dialect"].get("status") == "fail"
        and any(
            isinstance(item, dict)
            and item.get("http_status") == 200
            and item.get("tool_calls") is None
            for item in tool_observations
        )
        and any(
            isinstance(item, dict)
            and item.get("http_status") == 400
            and "did not produce any tool calls" in str(item.get("error") or "")
            for item in tool_observations
        )
    )
    conservative_diagnostic = payloads["conservative_diagnostic"]
    conservative_rows = conservative_diagnostic.get("rows")
    if not isinstance(conservative_rows, list):
        conservative_rows = []
    conservative_tool_failures = [
        {
            "name": item.get("name"),
            "code": item.get("code"),
            "elapsed_sec": item.get("elapsed_sec"),
            "error": item.get("error"),
        }
        for item in conservative_rows
        if isinstance(item, dict)
        and str(item.get("name") or "").startswith("tool_required")
        and item.get("tool_calls") in (None, [])
    ]
    if conservative_tool_failures:
        tool_protocol_blocked = True

    audit_long_prompt_open = current_audit_component_ok.get("long_prompt_coherence") is False
    audit_tool_protocol_open = current_audit_component_ok.get("tool_protocol") is False
    if audit_long_prompt_open:
        prompt_length_coherence_blocked = True
    if audit_tool_protocol_open:
        tool_protocol_blocked = True

    ok = (
        not missing
        and manifest_integrity_passed
        and stale_local_state_absent
        and structural_pass
        and text_cache_narrow_pass
        and switchglu_parity_pass
        and not prompt_length_coherence_blocked
        and not tool_protocol_blocked
    )
    return ok, {
        "artifacts": artifacts,
        "missing": missing,
        "current_audit_status": current_audit.get("status"),
        "manifest_integrity_passed": manifest_integrity_passed,
        "stale_local_state_absent": stale_local_state_absent,
        "structural_verify_passed": structural_pass,
        "text_cache_narrow_pass": text_cache_narrow_pass,
        "switchglu_selected_expert_parity_passed": switchglu_parity_pass,
        "switchglu_max_abs_diff": switchglu_max_abs_diff,
        "switchglu_mean_abs_diff": switchglu_mean_abs_diff,
        "prompt_length_coherence_blocked": prompt_length_coherence_blocked,
        "prompt_length_corrupt_cases": corrupt_length_cases,
        "tool_protocol_blocked": tool_protocol_blocked,
        "conservative_tool_failures": conservative_tool_failures,
        "status": "pass" if ok else "open",
        "release_boundary": (
            "mimo_v2_jang2l_current_local_runtime_cleared"
            if ok
            else "mimo_v2_jang2l_current_local_runtime_quality_open"
        ),
    }


def _zaya_vl_jangtq4_diagnostics(
    root: Path,
    external_ack_probe: dict[str, Any],
    rendered_prompt_compare: dict[str, Any],
) -> dict[str, Any]:
    ack_results = {
        str(item.get("label")): item
        for item in external_ack_probe.get("results") or []
        if isinstance(item, dict) and item.get("label")
    }
    rendered_rows = {
        str(item.get("name")): item
        for item in rendered_prompt_compare.get("rows") or []
        if isinstance(item, dict) and item.get("name")
    }

    def _rendered_detail(name: str) -> dict[str, Any]:
        row = rendered_rows.get(name) or {}
        rendered = row.get("rendered")
        return {
            "rendered_head": rendered[:160] if isinstance(rendered, str) else None,
            "token_count": row.get("token_count"),
        }

    return {
        "external_ack_probe": {
            "artifact_present": _path_present(root, ZAYA_VL_JANGTQ4_ACK_DIAGNOSTIC_REL),
            "short_exact_ack_content": (ack_results.get("short_exact_ack") or {}).get("content"),
            "json_ack_content": (ack_results.get("json_ack") or {}).get("content"),
            "system_cache_repeat_1_content": (
                ack_results.get("system_cache_repeat_1") or {}
            ).get("content"),
            "system_cache_repeat_2_content": (
                ack_results.get("system_cache_repeat_2") or {}
            ).get("content"),
            "system_cache_repeat_2_usage": (
                ack_results.get("system_cache_repeat_2") or {}
            ).get("usage"),
            "user_strong_cache_repeat_2_content": (
                ack_results.get("user_strong_cache_repeat_2") or {}
            ).get("content"),
        },
        "rendered_prompt_compare": {
            "artifact_present": _path_present(root, ZAYA_VL_JANGTQ4_RENDERED_PROMPT_COMPARE_REL),
            "zaya_vl_jangtq4": _rendered_detail("zaya_vl_jangtq4"),
            "zaya_text_mxfp4": _rendered_detail("zaya_text_mxfp4"),
        },
    }


def _nemotron_omni_no_media_diagnostics(
    root: Path,
    carryover_probe: dict[str, Any],
    prompt_variants_probe: dict[str, Any],
    system_prompt_probe: dict[str, Any],
    system_negative_probe: dict[str, Any],
) -> dict[str, Any]:
    results = {
        str(item.get("label")): item
        for item in carryover_probe.get("results") or []
        if isinstance(item, dict) and item.get("label")
    }
    variant_results = {
        str(item.get("label")): item
        for item in prompt_variants_probe.get("results") or []
        if isinstance(item, dict) and item.get("label")
    }
    system_results = {
        str(item.get("label")): item
        for item in system_prompt_probe.get("results") or []
        if isinstance(item, dict) and item.get("label")
    }
    system_negative_results = {
        str(item.get("label")): item
        for item in system_negative_probe.get("variants") or []
        if isinstance(item, dict) and item.get("label")
    }

    def _content(label: str) -> str | None:
        value = (results.get(label) or {}).get("content")
        return value if isinstance(value, str) else None

    def _variant_content(label: str) -> str | None:
        value = (variant_results.get(label) or {}).get("content")
        return value if isinstance(value, str) else None

    def _system_content(label: str) -> str | None:
        value = (system_results.get(label) or {}).get("content")
        return value if isinstance(value, str) else None

    def _system_negative_content(label: str) -> str | None:
        value = (system_negative_results.get(label) or {}).get("content")
        return value if isinstance(value, str) else None

    return {
        "artifact_present": _path_present(root, NEMOTRON_OMNI_NO_MEDIA_DIAGNOSTIC_REL),
        "text_no_media_before_any_image": _content("text_no_media_before_any_image"),
        "text_no_media_before_any_image_repeat": _content(
            "text_no_media_before_any_image_repeat"
        ),
        "text_no_media_after_image": _content("text_no_media_after_image"),
        "text_no_media_after_red_image": _content("text_no_media_after_red_image"),
        "vl_blue_image": _content("vl_blue_image"),
        "vl_red_image": _content("vl_red_image"),
        "prompt_variants_artifact_present": _path_present(
            root, NEMOTRON_OMNI_NO_MEDIA_PROMPT_VARIANTS_REL
        ),
        "prompt_variant_ambiguous_before": _variant_content("ambiguous_before"),
        "prompt_variant_negative_control_before": _variant_content(
            "negative_control_before"
        ),
        "prompt_variant_attachment_none_before": _variant_content(
            "attachment_none_before"
        ),
        "prompt_variant_count_before": _variant_content("count_before"),
        "prompt_variant_attachment_none_after": _variant_content(
            "attachment_none_after"
        ),
        "prompt_variant_count_after": _variant_content("count_after"),
        "prompt_variant_attachment_none_after_red": _variant_content(
            "attachment_none_after_red"
        ),
        "system_prompt_artifact_present": _path_present(
            root, NEMOTRON_OMNI_NO_MEDIA_SYSTEM_PROMPT_REL
        ),
        "system_prompt_user_only_none_before": _system_content(
            "user_only_none_before"
        ),
        "system_prompt_fact_none_before": _system_content(
            "system_fact_none_before"
        ),
        "system_prompt_exact_none_before": _system_content(
            "system_exact_none_before"
        ),
        "system_prompt_fact_none_after": _system_content("system_fact_none_after"),
        "system_prompt_exact_none_after": _system_content(
            "system_exact_none_after"
        ),
        "system_negative_artifact_present": _path_present(
            root, NEMOTRON_OMNI_NO_MEDIA_SYSTEM_NEGATIVE_REL
        ),
        "system_negative_before_any_image": _system_negative_content(
            "system_no_media_before_any_image"
        ),
        "system_negative_count_before_any_image": _system_negative_content(
            "count_zero_before_any_image"
        ),
        "system_negative_after_blue_image": _system_negative_content(
            "system_no_media_after_blue_image"
        ),
        "system_negative_count_after_blue_image": _system_negative_content(
            "count_zero_after_blue_image"
        ),
        "system_negative_after_red_image": _system_negative_content(
            "system_no_media_after_red_image"
        ),
        "system_negative_count_after_red_image": _system_negative_content(
            "count_zero_after_red_image"
        ),
    }


def _prompt_processing_speed_detail(
    text_payload: dict[str, Any],
    mtp_prefill_payload: dict[str, Any],
    mtp_packaged_payload: dict[str, Any],
    trace_payload: dict[str, Any],
    no_prefix_logits_payload: dict[str, Any],
    hybrid_long_prefix_split_payload: dict[str, Any],
    kvnone_payload: dict[str, Any],
    route_trace_payload: dict[str, Any],
    raw_forward_ab_payloads: list[dict[str, Any]],
    norm_shift_clearance_payload: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    text_ok, text_details = _speed_artifact_detail(text_payload)
    mtp_ok, mtp_details = _speed_artifact_detail(mtp_prefill_payload)
    mtp_packaged_ok, mtp_packaged_details = _speed_artifact_detail(mtp_packaged_payload)
    norm_shift_ok, norm_shift_details = _speed_artifact_detail(norm_shift_clearance_payload)
    trace_details = _prefill_trace_detail(trace_payload)
    no_prefix_logits_details = _no_prefix_logits_trial_detail(no_prefix_logits_payload)
    hybrid_long_prefix_split_details = _qwen_pp_trial_detail(hybrid_long_prefix_split_payload)
    kvnone_details = _qwen_pp_trial_detail(kvnone_payload)
    route_trace_details = _prefill_trace_detail(route_trace_payload)
    raw_forward_ab_details = _qwen_raw_forward_ab_detail(raw_forward_ab_payloads)
    return text_ok and norm_shift_ok, {
        "text_loader": text_details,
        "native_mtp_prefill_source_native_wheels": mtp_details,
        "native_mtp_prefill_packaged_compat_wheels": mtp_packaged_details,
        "native_mtp_after_norm_shift_default_cache": norm_shift_details,
        "prefill_trace": trace_details,
        "native_mtp_no_prefix_logits_trial": no_prefix_logits_details,
        "native_mtp_hybrid_long_prefix_split_trial": hybrid_long_prefix_split_details,
        "native_mtp_kvnone_trial": kvnone_details,
        "native_mtp_vlm_route_trace": route_trace_details,
        "raw_forward_path_ab": raw_forward_ab_details,
    }


def build_digest(root: Path | str = Path(".")) -> dict[str, Any]:
    root = Path(root)
    release_manifest = _load(root, CURRENT_RELEASE_REGRESSION_MANIFEST_REL)
    cache = _load(root, "build/current-dsv4-cache-proof-digest-20260521.json")
    dev_cache = _load(root, "build/dev-ui-dsv4-live-cache-proof-20260521/result.json")
    dev_tool = _load(root, "build/dev-ui-dsv4-live-tool-proof-20260521/result.json")
    two_tool = _load(
        root, "build/v1546-current-bundled-dsv4-two-tool-proof-20260521001426/result.json"
    )
    default_cache_tool_loop = _load(root, DSV4_DEFAULT_CACHE_TOOL_LOOP_REL)
    dsv4_responses_cache_gate = _load(root, DSV4_RESPONSES_CACHE_GATE_REL)
    dsv4_responses_restart_l2_gate = _load(
        root, DSV4_RESPONSES_RESTART_L2_GATE_REL
    )
    dsv4_responses_one_tool_stop = _load(root, DSV4_RESPONSES_ONE_TOOL_STOP_REL)
    default_cache_tool_loop_thinking_on = _load(
        root, DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_ON_REL
    )
    default_cache_tool_loop_nocache_ab = _load(
        root, DSV4_DEFAULT_CACHE_TOOL_LOOP_NOCACHE_AB_REL
    )
    default_cache_tool_loop_prompt_guard = _load(
        root, DSV4_DEFAULT_CACHE_TOOL_LOOP_PROMPT_GUARD_REL
    )
    default_cache_tool_loop_copy_block = _load(
        root, DSV4_DEFAULT_CACHE_TOOL_LOOP_COPY_BLOCK_REL
    )
    default_cache_tool_loop_thinking_copy_block = _load(
        root, DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_COPY_BLOCK_REL
    )
    default_cache_tool_loop_thinking_copy_block_dryrun = _load(
        root, DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_COPY_BLOCK_DRYRUN_REL
    )
    default_cache_tool_loop_dryrun_controls = _load(
        root, DSV4_DEFAULT_CACHE_TOOL_LOOP_DRYRUN_CONTROLS_REL
    )
    tool_call_contract = _load(root, TOOL_CALL_CONTRACT_REL)
    ui = _load(root, "build/dev-ui-smoke-20260521/summary.json")
    cache_architecture_contract = _load(root, CACHE_ARCHITECTURE_CONTRACT_REL)
    longctx = _load(root, "build/current-dsv4-long-context-proof-digest-20260521.json")
    quality_clearance = _load(root, DSV4_QUALITY_CLEARANCE_REL)
    dsv4_current_identifier_canary_rel, dsv4_current_identifier_canary = _load_first_present(
        root,
        (
            DSV4_CURRENT_IDENTIFIER_CANARY_REL,
            DSV4_CURRENT_IDENTIFIER_CANARY_FALLBACK_STRICT_REL,
            DSV4_CURRENT_IDENTIFIER_CANARY_FALLBACK_REL,
        ),
    )
    dsv4_current_identifier_matrix = _load(root, DSV4_CURRENT_IDENTIFIER_MATRIX_REL)
    dsv4_installed_tokenizer_roundtrip = _load(root, DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL)
    dsv4_live_logprobs_copy = _load(root, DSV4_LIVE_LOGPROBS_COPY_REL)
    dsv4_live_logprob_context_matrix = _load(root, DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL)
    dsv4_live_cache_context_identifier = _load(root, DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL)
    dsv4_source_nocache_identifier = _load(root, DSV4_SOURCE_NOCACHE_IDENTIFIER_REL)
    dsv4_source_same_prompt_nocache = _load(root, DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL)
    dsv4_source_cache_comparison = _load(root, DSV4_SOURCE_CACHE_COMPARISON_REL)
    dsv4_prompt_rail_exactness_rel, dsv4_prompt_rail_exactness = _load_first_present(
        root,
        (
            DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_REL,
            DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_FALLBACK_PRE_FORCE_OFF_REL,
            DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_FALLBACK_REL,
            DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_LEGACY_FALLBACK_REL,
        ),
    )
    dsv4_route_mode_dryrun_rel, dsv4_route_mode_dryrun = _load_first_present(
        root,
        (
            DSV4_CURRENT_ROUTE_MODE_DRYRUN_REL,
            DSV4_CURRENT_ROUTE_MODE_DRYRUN_FALLBACK_REL,
        ),
    )
    dsv4_current_generated_only_direct_rail_exactness = _load(
        root, DSV4_CURRENT_GENERATED_ONLY_DIRECT_RAIL_EXACTNESS_REL
    )
    dsv4_current_requested_thinking_exactness = _load(
        root, DSV4_CURRENT_REQUESTED_THINKING_EXACTNESS_REL
    )
    dsv4_current_rep1_rail_exactness = _load(
        root, DSV4_CURRENT_REP1_RAIL_EXACTNESS_REL
    )
    dsv4_current_source_rep1_rail_exactness = _load(
        root, DSV4_CURRENT_SOURCE_REP1_RAIL_EXACTNESS_REL
    )
    dsv4_current_source_token_tail_ab_exactness = _load(
        root, DSV4_CURRENT_SOURCE_TOKEN_TAIL_AB_EXACTNESS_REL
    )
    dsv4_current_source_rep1_direct_only = _load(
        root, DSV4_CURRENT_SOURCE_REP1_DIRECT_ONLY_REL
    )
    dsv4_current_source_bundle_defaults_exactness = _load(
        root, DSV4_CURRENT_SOURCE_BUNDLE_DEFAULTS_EXACTNESS_REL
    )
    dsv4_current_source_bundle_defaults_dryrun = _load(
        root, DSV4_CURRENT_SOURCE_BUNDLE_DEFAULTS_DRYRUN_REL
    )
    dsv4_current_source_memory_preflight = _load(
        root, DSV4_CURRENT_SOURCE_MEMORY_PREFLIGHT_REL
    )
    dsv4_current_jangtqk_direct_off_recheck = _load(
        root, DSV4_CURRENT_JANGTQK_DIRECT_OFF_RECHECK_REL
    )
    dsv4_chatmax_prompt_trigger = _load(root, DSV4_CHATMAX_PROMPT_TRIGGER_REL)
    dsv4_chatmax_budget_stop_rail = _load(
        root, DSV4_CHATMAX_BUDGET_STOP_RAIL_REL
    )
    dsv4_prompt_boundary_bisection = _load(
        root, DSV4_PROMPT_BOUNDARY_BISECTION_REL
    )
    dsv4_colon_period_logprob_trace = _load(
        root, DSV4_COLON_PERIOD_LOGPROB_TRACE_REL
    )
    dsv4_colon_period_visible_logprob_trace = _load(
        root, DSV4_COLON_PERIOD_VISIBLE_LOGPROB_TRACE_REL
    )
    dsv4_scene_token_rank_contrast = _load(
        root, DSV4_SCENE_TOKEN_RANK_CONTRAST_REL
    )
    dsv4_direct_vs_thinking_webgl_logit = _load(
        root, DSV4_DIRECT_VS_THINKING_WEBGL_LOGIT_PROBE_REL
    )
    dsv4_hidden_reasoning_control = _load(
        root, DSV4_HIDDEN_REASONING_CONTROL_REL
    )
    dsv4_template_parity_diagnostic = _load(
        root, DSV4_TEMPLATE_PARITY_DIAGNOSTIC_REL
    )
    dsv4_prefill_execution_variant_logits = _load(
        root, DSV4_PREFILL_EXECUTION_VARIANT_LOGITS_REL
    )
    dsv4_prompt_variant_logit_probe = _load(
        root, DSV4_PROMPT_VARIANT_LOGIT_PROBE_REL
    )
    dsv4_reasoning_policy_live = _load(root, DSV4_REASONING_POLICY_LIVE_REL)
    dsv4_cache_vs_full_logit_isolation = _load(
        root, DSV4_CACHE_VS_FULL_LOGIT_ISOLATION_REL
    )
    dsv4_batch_generator_logit_trace = _load(
        root, DSV4_BATCH_GENERATOR_LOGIT_TRACE_REL
    )
    dsv4_batch_generator_warmup_ablation = _load(
        root, DSV4_BATCH_GENERATOR_WARMUP_ABLATION_REL
    )
    api_cache_contract = _load(root, API_CACHE_CONTRACT_REL)
    panel_settings_contract = _load(root, PANEL_SETTINGS_CONTRACT_REL)
    max_output_context_contract_rel, max_output_context_contract = _load_first_present(
        root,
        (
            MAX_OUTPUT_CONTEXT_CONTRACT_REL,
            MAX_OUTPUT_CONTEXT_CONTRACT_FALLBACK_REL,
        ),
    )
    model_family_contract = _load(root, MODEL_FAMILY_CONTRACT_REL)
    parser_registry_contract = _load(root, PARSER_REGISTRY_CONTRACT_REL)
    model_artifact_format_contract = _load(root, MODEL_ARTIFACT_FORMAT_CONTRACT_REL)
    generation_defaults_contract = _load(root, GENERATION_DEFAULTS_CONTRACT_REL)
    native_mtp_contract = _load(root, NATIVE_MTP_CONTRACT_REL)
    vl_media_contract = _load(root, VL_MEDIA_CONTRACT_REL)
    qwen_jang_source_speed = _load(root, QWEN_JANG_SOURCE_SPEED_REL)
    qwen_jang_packaged_speed = _load(root, QWEN_JANG_PACKAGED_SPEED_REL)
    qwen_jang_text_baseline_speed = _load(root, QWEN_JANG_TEXT_BASELINE_SPEED_REL)
    qwen_native_mtp_prefill_speed = _load(root, QWEN_NATIVE_MTP_PREFILL_SPEED_REL)
    qwen_native_mtp_packaged_prefill_speed = _load(root, QWEN_NATIVE_MTP_PACKAGED_PREFILL_SPEED_REL)
    qwen_native_mtp_prefill_trace = _load(root, QWEN_NATIVE_MTP_PREFILL_TRACE_REL)
    qwen_native_mtp_no_prefix_logits = _load(root, QWEN_NATIVE_MTP_NO_PREFIX_LOGITS_REL)
    qwen_native_mtp_hybrid_long_prefix_split = _load(root, QWEN_NATIVE_MTP_HYBRID_LONG_PREFIX_SPLIT_REL)
    qwen_native_mtp_kvnone = _load(root, QWEN_NATIVE_MTP_KVNONE_REL)
    qwen_native_mtp_route_trace = _load(root, QWEN_NATIVE_MTP_ROUTE_TRACE_REL)
    qwen_raw_forward_ab_1024 = _load(root, QWEN_RAW_FORWARD_AB_1024_REL)
    qwen_raw_forward_ab_4096 = _load(root, QWEN_RAW_FORWARD_AB_4096_REL)
    qwen_native_mtp_norm_shift_clearance = _load(root, QWEN_NATIVE_MTP_NORM_SHIFT_CLEARANCE_REL)
    qwen_native_mtp_ab = _load(root, QWEN_NATIVE_MTP_AB_REL)
    real_ui_live_model_matrix = _current_real_ui_live_model_matrix(root)
    ling_installed_live = _load(root, LING_INSTALLED_LIVE_AUDIT_REL)
    ling_jangtq_strict_russian = _load(root, LING_JANGTQ_STRICT_RUSSIAN_NOCACHE_REL)
    ling_mxfp4_strict_russian = _load(root, LING_MXFP4_STRICT_RUSSIAN_NOCACHE_REL)
    (
        gemma4_responses_visible_contract_rel,
        gemma4_responses_visible_contract,
    ) = _load_first_present(
        root,
        (
            GEMMA4_RESPONSES_VISIBLE_CURRENT_REL,
            GEMMA4_RESPONSES_VISIBLE_CONTRACT_REL,
        ),
    )
    gemma4_responses_unsupported_thinking_budget = _load(
        root, GEMMA4_RESPONSES_UNSUPPORTED_THINKING_BUDGET_REL
    )
    gemma4_responses_visible_nocache = _load(root, GEMMA4_RESPONSES_VISIBLE_NOCACHE_REL)
    gemma4_responses_visible_512_nocache = _load(root, GEMMA4_RESPONSES_VISIBLE_512_NOCACHE_REL)
    gemma4_responses_thinking_off_nocache = _load(root, GEMMA4_RESPONSES_THINKING_OFF_NOCACHE_REL)
    gemma4_chat_visible_nocache = _load(root, GEMMA4_CHAT_VISIBLE_NOCACHE_REL)
    gemma4_local_metadata_audit = _load(root, GEMMA4_LOCAL_METADATA_AUDIT_REL)
    all_local_model_smoke_payloads = [
        (
            ALL_LOCAL_MODEL_SMOKE_LIVE_SLICE_CURRENT_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_LIVE_SLICE_CURRENT_REL),
        ),
        (
            ALL_LOCAL_MODEL_SMOKE_ZAYA_TEXT_VL_CURRENT_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_ZAYA_TEXT_VL_CURRENT_REL),
        ),
        (
            ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_CURRENT_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_CURRENT_REL),
        ),
        (
            ALL_LOCAL_MODEL_SMOKE_LING_HY3_NEMOTRON_CURRENT_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_LING_HY3_NEMOTRON_CURRENT_REL),
        ),
        (
            ALL_LOCAL_MODEL_SMOKE_QWEN35_MXFP8_MTP_CURRENT_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_QWEN35_MXFP8_MTP_CURRENT_REL),
        ),
        (
            ALL_LOCAL_MODEL_SMOKE_GEMMA4_26B_CRACK_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_GEMMA4_26B_CRACK_REL),
        ),
        (
            ALL_LOCAL_MODEL_SMOKE_MINIMAX_SMALL_JANGTQ_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_MINIMAX_SMALL_JANGTQ_REL),
        ),
        (
            ALL_LOCAL_MODEL_SMOKE_MIMO_V2_JANG2L_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_MIMO_V2_JANG2L_REL),
        ),
        (
            ALL_LOCAL_MODEL_SMOKE_DSV4_JANGTQ_K_REL,
            _load(root, ALL_LOCAL_MODEL_SMOKE_DSV4_JANGTQ_K_REL),
        ),
    ]
    zaya_vl_jangtq4_ack_diagnostic = _load(root, ZAYA_VL_JANGTQ4_ACK_DIAGNOSTIC_REL)
    zaya_vl_jangtq4_rendered_prompt_compare = _load(
        root, ZAYA_VL_JANGTQ4_RENDERED_PROMPT_COMPARE_REL
    )
    nemotron_omni_no_media_diagnostic = _load(
        root, NEMOTRON_OMNI_NO_MEDIA_DIAGNOSTIC_REL
    )
    nemotron_omni_no_media_prompt_variants = _load(
        root, NEMOTRON_OMNI_NO_MEDIA_PROMPT_VARIANTS_REL
    )
    nemotron_omni_no_media_system_prompt = _load(
        root, NEMOTRON_OMNI_NO_MEDIA_SYSTEM_PROMPT_REL
    )
    nemotron_omni_no_media_system_negative = _load(
        root, NEMOTRON_OMNI_NO_MEDIA_SYSTEM_NEGATIVE_REL
    )
    gemma4_mixed_swa_speed_rels = (
        GEMMA4_MIXED_SWA_SPEED_CURRENT_RELS
        if all(_path_present(root, rel) for rel in GEMMA4_MIXED_SWA_SPEED_CURRENT_RELS)
        else GEMMA4_MIXED_SWA_SPEED_ARTIFACT_RELS
    )
    gemma4_mixed_swa_speed_payloads = [
        (rel, _load(root, rel))
        for rel in gemma4_mixed_swa_speed_rels
    ]
    gemma4_mixed_swa_sustained_cachehit = _load(
        root,
        (
            GEMMA4_MIXED_SWA_SPEED_CURRENT_RELS[0]
            if gemma4_mixed_swa_speed_rels == GEMMA4_MIXED_SWA_SPEED_CURRENT_RELS
            else GEMMA4_MIXED_SWA_SUSTAINED_CACHEHIT_DIAGNOSTIC_REL
        ),
    )
    gemma4_mixed_swa_short_nocache_repeat = _load(
        root,
        (
            GEMMA4_MIXED_SWA_SPEED_CURRENT_RELS[0]
            if gemma4_mixed_swa_speed_rels == GEMMA4_MIXED_SWA_SPEED_CURRENT_RELS
            else GEMMA4_MIXED_SWA_SHORT_NOCACHE_REPEAT_DIAGNOSTIC_REL
        ),
    )
    gemma4_mixed_swa_short_nocache_scheduler_trace = _load(
        root,
        GEMMA4_MIXED_SWA_SHORT_NOCACHE_SCHEDULER_TRACE_REL,
    )
    gemma4_mixed_swa_short_nocache_sync_eval_ab = _load(
        root,
        GEMMA4_MIXED_SWA_SHORT_NOCACHE_SYNC_EVAL_AB_REL,
    )
    gemma4_mixed_swa_short_nocache_streaming = _load(
        root,
        GEMMA4_MIXED_SWA_SHORT_NOCACHE_STREAMING_REL,
    )

    requirements: list[dict[str, Any]] = []
    cache_checks = cache.get("checks") or {}
    native = (
        ((dev_cache.get("before") or {}).get("body") or {}).get("native_cache")
        or (dev_tool.get("health") or {}).get("native_cache")
        or {}
    )
    turn1 = dev_cache.get("turn1") or {}
    turn2 = dev_cache.get("turn2") or {}
    round1_body = ((dev_tool.get("round1") or {}).get("body") or {})
    round2_body = ((dev_tool.get("round2") or {}).get("body") or {})
    round1_calls = _function_calls(round1_body)
    round2_calls = _function_calls(round2_body)
    one_tool_stop_checks = (
        dsv4_responses_one_tool_stop.get("checks")
        if isinstance(dsv4_responses_one_tool_stop.get("checks"), dict)
        else {}
    )
    one_tool_round1 = (
        dsv4_responses_one_tool_stop.get("round1")
        if isinstance(dsv4_responses_one_tool_stop.get("round1"), dict)
        else {}
    )
    one_tool_round2 = (
        dsv4_responses_one_tool_stop.get("round2")
        if isinstance(dsv4_responses_one_tool_stop.get("round2"), dict)
        else {}
    )
    one_tool_round1_calls = (
        one_tool_round1.get("function_calls")
        if isinstance(one_tool_round1.get("function_calls"), list)
        else []
    )
    one_tool_round2_calls = (
        one_tool_round2.get("function_calls")
        if isinstance(one_tool_round2.get("function_calls"), list)
        else []
    )
    current_one_tool_stop_ok = (
        dsv4_responses_one_tool_stop.get("status") == "pass"
        and one_tool_stop_checks.get("round1_exactly_one_tool") is True
        and one_tool_stop_checks.get("round1_tool_is_list_directory") is True
        and one_tool_stop_checks.get("round2_tools_still_available") is True
        and one_tool_stop_checks.get("round2_no_function_calls") is True
        and one_tool_stop_checks.get("round2_final_done") is True
        and one_tool_stop_checks.get("previous_response_id_used") is True
    )
    two_tool_rounds = two_tool.get("rounds") or []
    executed_tools = [
        tool
        for round_item in two_tool_rounds
        for tool in (round_item.get("executed_tools") or [])
        if isinstance(tool, dict)
    ]
    final_text = two_tool_rounds[-1].get("output_text") if two_tool_rounds else None
    two_tool_names = [tool.get("name") for tool in executed_tools]
    default_tool_rounds = default_cache_tool_loop.get("rounds") or []
    default_tool_executed = [
        tool
        for round_item in default_tool_rounds
        for tool in (round_item.get("executed_tools") or [])
        if isinstance(tool, dict)
    ]
    default_tool_names = [tool.get("name") for tool in default_tool_executed]
    default_tool_final_text = (
        default_tool_rounds[-1].get("output_text") if default_tool_rounds else None
    )
    default_tool_cmd = _command_tokens(default_cache_tool_loop)
    default_tool_health = default_cache_tool_loop.get("health") or {}
    default_tool_native = default_tool_health.get("native_cache") or {}
    default_tool_cached_tokens = sum(
        _cached_tokens_from_payload(row) for row in default_tool_rounds
    )
    default_tool_cache_details = [
        _cache_detail_from_payload(row)
        for row in default_tool_rounds
        if _cache_detail_from_payload(row)
    ]
    default_tool_cache_detail_has_dsv4 = any(
        "dsv4" in detail for detail in default_tool_cache_details
    )
    code_tool_probe = default_cache_tool_loop.get("code_tool_probe") or {}
    code_tool_expected = code_tool_probe.get("expected_content")
    code_tool_actual = code_tool_probe.get("actual_content")
    code_tool_missing_identifiers = code_tool_probe.get("missing_expected_fragments")
    if not isinstance(code_tool_missing_identifiers, list):
        code_tool_missing_identifiers = [
            item
            for item in _present_patterns(
                code_tool_expected,
                DSV4_THREEJS_IDENTIFIERS,
            )
            if item not in _present_patterns(
                code_tool_actual,
                DSV4_THREEJS_IDENTIFIERS,
            )
        ]
    code_tool_corrupt_identifier_patterns = code_tool_probe.get(
        "corrupt_identifier_patterns"
    )
    if isinstance(code_tool_corrupt_identifier_patterns, list):
        code_tool_corrupt_identifier_patterns = _known_corrupt_identifier_patterns(
            code_tool_corrupt_identifier_patterns,
            code_tool_actual,
        )
    else:
        code_tool_corrupt_identifier_patterns = _present_patterns(
            code_tool_actual,
            DSV4_THREEJS_CORRUPT_PATTERNS,
        )
    default_tool_checks = (
        default_cache_tool_loop.get("checks")
        if isinstance(default_cache_tool_loop.get("checks"), dict)
        else {}
    )
    required_default_tool_checks = (
        "tool_sequence_ordered",
        "final_done",
        "file_written",
        "native_cache",
        "native_prefix",
        "native_paged",
        "native_l2",
        "generic_tq_kv_off",
        "cached_tokens_seen",
        "dsv4_cache_detail_seen",
    )
    default_tool_detail_checks = (
        *required_default_tool_checks,
        "code_file_written_exact",
    )
    failed_required_default_tool_checks = [
        key
        for key in required_default_tool_checks
        if default_tool_checks.get(key) is not True
    ]
    default_tool_required_checks_ok = all(
        default_tool_checks.get(key) is True for key in required_default_tool_checks
    )
    default_tool_parser_ok = "--tool-call-parser" in default_tool_cmd and "dsml" in default_tool_cmd
    default_reasoning_parser_ok = (
        "--reasoning-parser" in default_tool_cmd and "deepseek_r1" in default_tool_cmd
    )
    default_tool_cache_ok = (
        "--disable-prefix-cache" not in default_tool_cmd
        and "--dsv4-enable-prefix-cache" in default_tool_cmd
        and "--use-paged-cache" in default_tool_cmd
        and "--enable-block-disk-cache" in default_tool_cmd
        and default_tool_native.get("cache_type") == "native_composite"
        and default_tool_native.get("prefix") is True
        and default_tool_native.get("paged") is True
        and default_tool_native.get("block_disk_l2") is True
        and (default_tool_native.get("generic_turboquant_kv") or {}).get("enabled")
        is False
        and default_tool_cached_tokens > 0
        and default_tool_cache_detail_has_dsv4
        and default_tool_parser_ok
        and default_reasoning_parser_ok
        and default_tool_required_checks_ok
    )
    current_default_native_cache_ok = (
        "--disable-prefix-cache" not in default_tool_cmd
        and "--dsv4-enable-prefix-cache" in default_tool_cmd
        and "--use-paged-cache" in default_tool_cmd
        and "--enable-block-disk-cache" in default_tool_cmd
        and default_tool_native.get("cache_type") == "native_composite"
        and default_tool_native.get("prefix") is True
        and default_tool_native.get("paged") is True
        and default_tool_native.get("block_disk_l2") is True
        and (default_tool_native.get("generic_turboquant_kv") or {}).get("enabled")
        is False
    )
    panel_settings_raw_checks = (
        panel_settings_contract.get("checks")
        if isinstance(panel_settings_contract.get("checks"), dict)
        else {}
    )
    current_app_launch_default_cache_ok = (
        panel_settings_contract.get("status") == "pass"
        and panel_settings_raw_checks.get("dsv4_default_native_prefix_on") is True
        and panel_settings_raw_checks.get("dsv4_generic_kv_flags_suppressed") is True
        and current_default_native_cache_ok
    )
    thinking_tool_rounds = default_cache_tool_loop_thinking_on.get("rounds") or []
    thinking_tool_executed = [
        tool
        for round_item in thinking_tool_rounds
        for tool in (round_item.get("executed_tools") or [])
        if isinstance(tool, dict)
    ]
    thinking_tool_checks = (
        default_cache_tool_loop_thinking_on.get("checks")
        if isinstance(default_cache_tool_loop_thinking_on.get("checks"), dict)
        else {}
    )
    thinking_tool_probe = default_cache_tool_loop_thinking_on.get("code_tool_probe") or {}
    thinking_tool_diagnostic = {
        "artifact": DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_ON_REL,
        "artifact_status": default_cache_tool_loop_thinking_on.get("status"),
        "request_thinking_mode": default_cache_tool_loop_thinking_on.get(
            "request_thinking_mode"
        ),
        "executed_tool_names": [
            tool.get("name") for tool in thinking_tool_executed
        ],
        "final_text": (
            thinking_tool_rounds[-1].get("output_text") if thinking_tool_rounds else None
        ),
        "tool_loop_checks": {
            key: thinking_tool_checks.get(key)
            for key in default_tool_detail_checks
        },
        "tool_loop_cached_tokens": default_cache_tool_loop_thinking_on.get(
            "tool_loop_cached_tokens"
        ),
        "tool_loop_cache_details": default_cache_tool_loop_thinking_on.get(
            "tool_loop_cache_details"
        ),
        "code_tool_expected_content": thinking_tool_probe.get("expected_content"),
        "code_tool_actual_content": thinking_tool_probe.get("actual_content"),
    }
    nocache_tool_checks = (
        default_cache_tool_loop_nocache_ab.get("checks")
        if isinstance(default_cache_tool_loop_nocache_ab.get("checks"), dict)
        else {}
    )
    nocache_tool_probe = default_cache_tool_loop_nocache_ab.get("code_tool_probe") or {}
    no_cache_tool_diagnostic = {
        "artifact": DSV4_DEFAULT_CACHE_TOOL_LOOP_NOCACHE_AB_REL,
        "artifact_present": _path_present(
            root, DSV4_DEFAULT_CACHE_TOOL_LOOP_NOCACHE_AB_REL
        ),
        "artifact_status": default_cache_tool_loop_nocache_ab.get("status"),
        "diagnostic_cache_mode": default_cache_tool_loop_nocache_ab.get(
            "diagnostic_cache_mode"
        ),
        "code_round_request_controls": default_cache_tool_loop_nocache_ab.get(
            "code_round_request_controls"
        ),
        "tool_loop_checks": {
            key: nocache_tool_checks.get(key)
            for key in default_tool_detail_checks
        },
        "tool_loop_cached_tokens": default_cache_tool_loop_nocache_ab.get(
            "tool_loop_cached_tokens"
        ),
        "tool_loop_cache_details": default_cache_tool_loop_nocache_ab.get(
            "tool_loop_cache_details"
        ),
        "code_tool_first_difference": nocache_tool_probe.get("first_difference"),
        "code_tool_missing_identifiers": nocache_tool_probe.get(
            "missing_expected_fragments"
        ),
        "code_tool_corrupt_identifier_patterns": _known_corrupt_identifier_patterns(
            nocache_tool_probe.get("corrupt_identifier_patterns"),
            nocache_tool_probe.get("actual_content"),
        ),
    }
    prompt_guard_tool_checks = (
        default_cache_tool_loop_prompt_guard.get("checks")
        if isinstance(default_cache_tool_loop_prompt_guard.get("checks"), dict)
        else {}
    )
    prompt_guard_tool_probe = (
        default_cache_tool_loop_prompt_guard.get("code_tool_probe") or {}
    )
    prompt_guard_tool_diagnostic = {
        "artifact": DSV4_DEFAULT_CACHE_TOOL_LOOP_PROMPT_GUARD_REL,
        "artifact_present": _path_present(
            root, DSV4_DEFAULT_CACHE_TOOL_LOOP_PROMPT_GUARD_REL
        ),
        "artifact_status": default_cache_tool_loop_prompt_guard.get("status"),
        "tool_loop_checks": {
            key: prompt_guard_tool_checks.get(key)
            for key in default_tool_detail_checks
        },
        "tool_loop_cached_tokens": default_cache_tool_loop_prompt_guard.get(
            "tool_loop_cached_tokens"
        ),
        "tool_loop_cache_details": default_cache_tool_loop_prompt_guard.get(
            "tool_loop_cache_details"
        ),
        "code_tool_first_difference": prompt_guard_tool_probe.get("first_difference"),
        "code_tool_missing_identifiers": prompt_guard_tool_probe.get(
            "missing_expected_fragments"
        ),
        "code_tool_corrupt_identifier_patterns": _known_corrupt_identifier_patterns(
            prompt_guard_tool_probe.get("corrupt_identifier_patterns"),
            prompt_guard_tool_probe.get("actual_content"),
        ),
    }
    copy_block_tool_checks = (
        default_cache_tool_loop_copy_block.get("checks")
        if isinstance(default_cache_tool_loop_copy_block.get("checks"), dict)
        else {}
    )
    copy_block_tool_probe = (
        default_cache_tool_loop_copy_block.get("code_tool_probe") or {}
    )
    copy_block_tool_diagnostic = {
        "artifact": DSV4_DEFAULT_CACHE_TOOL_LOOP_COPY_BLOCK_REL,
        "artifact_present": _path_present(
            root, DSV4_DEFAULT_CACHE_TOOL_LOOP_COPY_BLOCK_REL
        ),
        "artifact_status": default_cache_tool_loop_copy_block.get("status"),
        "diagnostic_code_prompt_variant": default_cache_tool_loop_copy_block.get(
            "diagnostic_code_prompt_variant"
        ),
        "tool_loop_checks": {
            key: copy_block_tool_checks.get(key)
            for key in default_tool_detail_checks
        },
        "tool_loop_cached_tokens": default_cache_tool_loop_copy_block.get(
            "tool_loop_cached_tokens"
        ),
        "tool_loop_cache_details": default_cache_tool_loop_copy_block.get(
            "tool_loop_cache_details"
        ),
        "code_tool_first_difference": copy_block_tool_probe.get("first_difference"),
        "code_tool_missing_identifiers": copy_block_tool_probe.get(
            "missing_expected_fragments"
        ),
        "code_tool_corrupt_identifier_patterns": _known_corrupt_identifier_patterns(
            copy_block_tool_probe.get("corrupt_identifier_patterns"),
            copy_block_tool_probe.get("actual_content"),
        ),
    }
    thinking_copy_block_telemetry = (
        default_cache_tool_loop_thinking_copy_block.get("telemetry")
        if isinstance(
            default_cache_tool_loop_thinking_copy_block.get("telemetry"), list
        )
        else []
    )
    thinking_copy_block_memory = (
        thinking_copy_block_telemetry[0].get("system_memory")
        if thinking_copy_block_telemetry
        and isinstance(thinking_copy_block_telemetry[0], dict)
        and isinstance(thinking_copy_block_telemetry[0].get("system_memory"), dict)
        else {}
    )
    thinking_copy_block_diagnostic = {
        "artifact": DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_COPY_BLOCK_REL,
        "artifact_present": _path_present(
            root, DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_COPY_BLOCK_REL
        ),
        "artifact_status": default_cache_tool_loop_thinking_copy_block.get(
            "status"
        ),
        "artifact_reason": default_cache_tool_loop_thinking_copy_block.get(
            "reason"
        ),
        "required_available_gb": default_cache_tool_loop_thinking_copy_block.get(
            "required_available_gb"
        ),
        "available_gb": thinking_copy_block_memory.get("available_gb"),
        "dry_run_artifact": (
            DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_COPY_BLOCK_DRYRUN_REL
        ),
        "dry_run_present": _path_present(
            root, DSV4_DEFAULT_CACHE_TOOL_LOOP_THINKING_COPY_BLOCK_DRYRUN_REL
        ),
        "dry_run_status": default_cache_tool_loop_thinking_copy_block_dryrun.get(
            "status"
        ),
        "dry_run_cache_mode": (
            default_cache_tool_loop_thinking_copy_block_dryrun.get(
                "diagnostic_cache_mode"
            )
        ),
        "dry_run_code_prompt_variant": (
            default_cache_tool_loop_thinking_copy_block_dryrun.get(
                "diagnostic_code_prompt_variant"
            )
        ),
        "dry_run_request_max_output_tokens": (
            default_cache_tool_loop_thinking_copy_block_dryrun.get(
                "request_max_output_tokens"
            )
        ),
        "dry_run_request_thinking_mode": (
            default_cache_tool_loop_thinking_copy_block_dryrun.get(
                "request_thinking_mode"
            )
        ),
        "dry_run_code_round_request_controls": (
            default_cache_tool_loop_thinking_copy_block_dryrun.get(
                "code_round_request_controls"
            )
        ),
    }
    dryrun_controls_probe = default_cache_tool_loop_dryrun_controls.get(
        "code_tool_probe"
    )
    if not isinstance(dryrun_controls_probe, dict):
        dryrun_controls_probe = {}
    dryrun_controls_diagnostic = {
        "artifact": DSV4_DEFAULT_CACHE_TOOL_LOOP_DRYRUN_CONTROLS_REL,
        "artifact_present": _path_present(
            root, DSV4_DEFAULT_CACHE_TOOL_LOOP_DRYRUN_CONTROLS_REL
        ),
        "artifact_status": default_cache_tool_loop_dryrun_controls.get("status"),
        "code_round_request_controls": default_cache_tool_loop_dryrun_controls.get(
            "code_round_request_controls"
        ),
        "code_tool_probe": {
            "exact": dryrun_controls_probe.get("exact"),
            "first_difference": dryrun_controls_probe.get("first_difference"),
            "missing_expected_fragments": dryrun_controls_probe.get(
                "missing_expected_fragments"
            ),
            "corrupt_identifier_patterns": _known_corrupt_identifier_patterns(
                dryrun_controls_probe.get("corrupt_identifier_patterns"),
                dryrun_controls_probe.get("actual_content"),
            ),
        },
    }
    code_round_controls = default_cache_tool_loop.get("code_round_request_controls")
    if not isinstance(code_round_controls, dict):
        code_round_controls = {}
    no_cache_code_round_controls = no_cache_tool_diagnostic.get(
        "code_round_request_controls"
    )
    if not isinstance(no_cache_code_round_controls, dict):
        no_cache_code_round_controls = {}
    effective_code_round_controls = code_round_controls or no_cache_code_round_controls
    no_cache_exact_code_also_failed = (
        no_cache_tool_diagnostic.get("artifact_present") is True
        and no_cache_tool_diagnostic.get("tool_loop_checks", {}).get(
            "code_file_written_exact"
        )
        is False
        and no_cache_tool_diagnostic.get("diagnostic_cache_mode") == "disabled"
    )
    default_cache_tool_loop_root_cause = {
        "tool_parser_and_cache_path_proven": (
            default_tool_names == ["list_directory", "write_file", "write_file"]
            and default_tool_final_text == "DONE"
            and default_tool_parser_ok
            and default_reasoning_parser_ok
            and default_tool_cache_detail_has_dsv4
            and default_tool_cached_tokens > 0
        ),
        "default_cache_exact_code_failed": (
            default_tool_checks.get("code_file_written_exact") is False
        ),
        "no_cache_exact_code_also_failed": no_cache_exact_code_also_failed,
        "cache_not_sufficient_root_cause": no_cache_exact_code_also_failed,
        "request_controls_rule_out_forced_sampling_fix": (
            effective_code_round_controls.get("temperature") == 0.0
            and effective_code_round_controls.get("top_p") == 1.0
            and effective_code_round_controls.get("top_k") == 0
            and effective_code_round_controls.get("repetition_penalty") == 1.0
        ),
        "current_primary_failure": (
            "exact_code_generation"
            if default_tool_checks.get("code_file_written_exact") is False
            else None
        ),
    }
    ui_visible = ui.get("visible_assertions") or {}
    ui_cli = ui.get("cli_preview_assertions") or {}
    _add(
        requirements,
        "DSV4 Flash prefix/paged/L2 cache is enabled by default from app launch",
        _status(
            current_app_launch_default_cache_ok
            or all(
                cache_checks.get(key)
                for key in (
                    "persistedDefaultOn",
                    "launchHasDsv4EnablePrefix",
                    "launchHasUsePagedCache",
                    "launchHasBlockDisk",
                    "launchNoDisablePrefix",
                )
            )
        ),
        (
            [PANEL_SETTINGS_CONTRACT_REL, DSV4_DEFAULT_CACHE_TOOL_LOOP_REL]
            if current_app_launch_default_cache_ok
            else [
                "build/current-dsv4-cache-proof-digest-20260521.json",
                "build/dev-ui-smoke-20260521/summary.json",
            ]
        ),
        details={
            **{
                key: cache_checks.get(key)
                for key in (
                    "persistedDefaultOn",
                    "launchHasDsv4EnablePrefix",
                    "launchHasUsePagedCache",
                    "launchHasBlockDisk",
                    "launchNoDisablePrefix",
                )
            },
            "current_panel_settings_status": panel_settings_contract.get("status"),
            "current_panel_settings_dsv4_default_native_prefix_on": (
                panel_settings_raw_checks.get("dsv4_default_native_prefix_on")
            ),
            "current_panel_settings_dsv4_generic_kv_flags_suppressed": (
                panel_settings_raw_checks.get("dsv4_generic_kv_flags_suppressed")
            ),
            "current_default_cache_launch_has_dsv4_enable_prefix_cache": (
                "--dsv4-enable-prefix-cache" in default_tool_cmd
            ),
            "current_default_cache_launch_has_use_paged_cache": (
                "--use-paged-cache" in default_tool_cmd
            ),
            "current_default_cache_launch_has_block_disk_cache": (
                "--enable-block-disk-cache" in default_tool_cmd
            ),
            "current_default_cache_launch_has_disable_prefix_cache": (
                "--disable-prefix-cache" in default_tool_cmd
            ),
            "current_default_cache_native_cache_type": default_tool_native.get(
                "cache_type"
            ),
            "current_default_cache_native_prefix": default_tool_native.get("prefix"),
            "current_default_cache_native_paged": default_tool_native.get("paged"),
            "current_default_cache_native_block_disk_l2": default_tool_native.get(
                "block_disk_l2"
            ),
            "current_default_cache_generic_turboquant_kv": (
                default_tool_native.get("generic_turboquant_kv")
            ),
            "current_default_native_cache_ok": current_default_native_cache_ok,
            "current_app_launch_default_cache_ok": current_app_launch_default_cache_ok,
        },
    )
    _add(
        requirements,
        "DSV4 cache is native SWA+CSA/HCA composite, not generic KV/TurboQuant KV",
        _status(
            (
                native.get("cache_type") == "native_composite"
                and native.get("prefix") is True
                and native.get("paged") is True
                and native.get("block_disk_l2") is True
                and (native.get("generic_turboquant_kv") or {}).get("enabled") is False
            )
            or current_default_native_cache_ok
        ),
        (
            ["build/dev-ui-dsv4-live-cache-proof-20260521/result.json"]
            if (
                native.get("cache_type") == "native_composite"
                and native.get("prefix") is True
                and native.get("paged") is True
                and native.get("block_disk_l2") is True
                and (native.get("generic_turboquant_kv") or {}).get("enabled")
                is False
            )
            else [DSV4_DEFAULT_CACHE_TOOL_LOOP_REL]
        ),
        details={
            "cache_type": native.get("cache_type"),
            "components": native.get("components"),
            "generic_turboquant_kv": native.get("generic_turboquant_kv"),
            "cache_store_policy": native.get("cache_store_policy"),
            "current_default_cache_native_cache_type": default_tool_native.get(
                "cache_type"
            ),
            "current_default_cache_components": default_tool_native.get(
                "components"
            ),
            "current_default_cache_generic_turboquant_kv": (
                default_tool_native.get("generic_turboquant_kv")
            ),
            "current_default_cache_store_policy": (
                default_tool_native.get("cache_store_policy")
            ),
            "current_default_cache_native_cache_ok": current_default_native_cache_ok,
        },
    )
    cached_details = (((turn2.get("body") or {}).get("usage") or {}).get("prompt_tokens_details") or {})
    responses_cache_cases = (
        dsv4_responses_cache_gate.get("cases")
        if isinstance(dsv4_responses_cache_gate.get("cases"), dict)
        else {}
    )
    responses_cached_follow = (
        responses_cache_cases.get("previous_response_follow")
        if isinstance(responses_cache_cases.get("previous_response_follow"), dict)
        else {}
    )
    responses_stream_follow = (
        responses_cache_cases.get("stream_previous_response_follow")
        if isinstance(responses_cache_cases.get("stream_previous_response_follow"), dict)
        else {}
    )
    responses_no_cache = (
        responses_cache_cases.get("explicit_no_cache_full_prompt")
        if isinstance(responses_cache_cases.get("explicit_no_cache_full_prompt"), dict)
        else {}
    )
    responses_cached_follow_details = (
        (responses_cached_follow.get("usage") or {}).get("input_tokens_details")
        or (responses_cached_follow.get("usage") or {}).get("prompt_tokens_details")
        or {}
    )
    responses_stream_follow_details = (
        (responses_stream_follow.get("usage") or {}).get("input_tokens_details")
        or (responses_stream_follow.get("usage") or {}).get("prompt_tokens_details")
        or {}
    )
    responses_no_cache_details = (
        (responses_no_cache.get("usage") or {}).get("input_tokens_details")
        or (responses_no_cache.get("usage") or {}).get("prompt_tokens_details")
        or {}
    )
    responses_cached_tokens = int(
        responses_cached_follow_details.get("cached_tokens") or 0
    )
    responses_stream_cached_tokens = int(
        responses_stream_follow_details.get("cached_tokens") or 0
    )
    responses_no_cache_tokens = int(
        responses_no_cache_details.get("cached_tokens") or 0
    )
    responses_cached_wall = responses_cached_follow.get("wall_seconds")
    responses_no_cache_wall = responses_no_cache.get("wall_seconds")
    responses_stream_ttft = responses_stream_follow.get("ttft_seconds")
    current_responses_cache_hit_ok = (
        dsv4_responses_cache_gate.get("status") == "pass"
        and responses_cached_tokens > 0
        and "dsv4" in str(responses_cached_follow_details.get("cache_detail") or "")
        and responses_stream_cached_tokens > 0
        and "dsv4" in str(responses_stream_follow_details.get("cache_detail") or "")
        and isinstance(responses_stream_ttft, (int, float))
        and responses_stream_ttft > 0
        and responses_no_cache_tokens == 0
        and isinstance(responses_cached_wall, (int, float))
        and isinstance(responses_no_cache_wall, (int, float))
        and responses_cached_wall < responses_no_cache_wall
    )
    restart_l2_checks = (
        dsv4_responses_restart_l2_gate.get("checks")
        if isinstance(dsv4_responses_restart_l2_gate.get("checks"), dict)
        else {}
    )
    restart_l2_before = (
        dsv4_responses_restart_l2_gate.get("before_restart")
        if isinstance(dsv4_responses_restart_l2_gate.get("before_restart"), dict)
        else {}
    )
    restart_l2_after = (
        dsv4_responses_restart_l2_gate.get("after_restart")
        if isinstance(dsv4_responses_restart_l2_gate.get("after_restart"), dict)
        else {}
    )
    restart_l2_before_block = (
        (restart_l2_before.get("cache_stats") or {}).get("block_disk_cache")
        if isinstance(restart_l2_before.get("cache_stats"), dict)
        else {}
    )
    if not isinstance(restart_l2_before_block, dict):
        restart_l2_before_block = {}
    restart_l2_after_block = (
        (restart_l2_after.get("cache_stats") or {}).get("block_disk_cache")
        if isinstance(restart_l2_after.get("cache_stats"), dict)
        else {}
    )
    if not isinstance(restart_l2_after_block, dict):
        restart_l2_after_block = {}
    restart_l2_after_response = (
        restart_l2_after.get("response")
        if isinstance(restart_l2_after.get("response"), dict)
        else {}
    )
    restart_l2_after_usage_details = (
        (restart_l2_after_response.get("usage") or {}).get("input_tokens_details")
        or (restart_l2_after_response.get("usage") or {}).get("prompt_tokens_details")
        or {}
    )
    if not isinstance(restart_l2_after_usage_details, dict):
        restart_l2_after_usage_details = {}
    restart_l2_cached_tokens = int(
        restart_l2_after_usage_details.get("cached_tokens") or 0
    )
    restart_l2_cache_detail = str(
        restart_l2_after_usage_details.get("cache_detail") or ""
    )
    current_restart_l2_ok = (
        dsv4_responses_restart_l2_gate.get("status") == "pass"
        and restart_l2_checks.get("native_cache") is True
        and restart_l2_checks.get("native_prefix") is True
        and restart_l2_checks.get("native_paged") is True
        and restart_l2_checks.get("native_l2") is True
        and restart_l2_checks.get("generic_tq_kv_off") is True
        and restart_l2_checks.get("disk_write_before_restart") is True
        and restart_l2_checks.get("restart_l2_disk_hit") is True
        and restart_l2_checks.get("restart_dsv4_cache_hit") is True
        and restart_l2_checks.get("same_block_disk_cache_dir") is True
        and restart_l2_checks.get("fresh_run_nonce") is True
        and restart_l2_cached_tokens > 0
        and "dsv4" in restart_l2_cache_detail
        and int(restart_l2_before_block.get("disk_writes") or 0) > 0
        and int(restart_l2_after_block.get("disk_hits") or 0) > 0
    )
    _add(
        requirements,
        "DSV4 same-process cache hit improves latency/TTFT and records paged+dsv4 hit",
        _status(
            (
                cache_checks.get("hotFasterTtft")
                and cache_checks.get("sameProcessHitDsv4")
            )
            or current_responses_cache_hit_ok
        ),
        (
            [
                "build/current-dsv4-cache-proof-digest-20260521.json",
                "build/dev-ui-dsv4-live-cache-proof-20260521/result.json",
            ]
            if cache_checks.get("hotFasterTtft")
            and cache_checks.get("sameProcessHitDsv4")
            else [DSV4_RESPONSES_CACHE_GATE_REL]
        ),
        details={
            "cold_ttft_sec": (cache.get("timings") or {}).get("cold_ttft_sec"),
            "hot_ttft_sec": (cache.get("timings") or {}).get("hot_ttft_sec"),
            "dev_cold_elapsed_sec": turn1.get("elapsed_sec"),
            "dev_hot_elapsed_sec": turn2.get("elapsed_sec"),
            "dev_cached_tokens": cached_details.get("cached_tokens"),
            "dev_cache_detail": cached_details.get("cache_detail"),
            "current_responses_cache_gate_status": (
                dsv4_responses_cache_gate.get("status")
            ),
            "current_responses_cached_tokens": responses_cached_tokens,
            "current_responses_cache_detail": (
                responses_cached_follow_details.get("cache_detail")
            ),
            "current_responses_stream_cached_tokens": responses_stream_cached_tokens,
            "current_responses_stream_cache_detail": (
                responses_stream_follow_details.get("cache_detail")
            ),
            "current_responses_stream_ttft_sec": responses_stream_ttft,
            "current_responses_cached_wall_sec": responses_cached_wall,
            "current_responses_no_cache_wall_sec": responses_no_cache_wall,
            "current_responses_no_cache_tokens": responses_no_cache_tokens,
            "current_responses_cache_hit_ok": current_responses_cache_hit_ok,
        },
    )
    _add(
        requirements,
        "DSV4 block disk L2 stores and hits after restart",
        _status(
            (
                cache_checks.get("blockDiskWrite")
                and cache_checks.get("restartL2DiskHit")
                and cache_checks.get("restartDsv4CacheHit")
            )
            or current_restart_l2_ok
        ),
        (
            ["build/current-dsv4-cache-proof-digest-20260521.json"]
            if cache_checks.get("blockDiskWrite")
            and cache_checks.get("restartL2DiskHit")
            and cache_checks.get("restartDsv4CacheHit")
            else [DSV4_RESPONSES_RESTART_L2_GATE_REL]
        ),
        details={
            "after_hot_block_disk": (cache.get("stats_after_hot") or {}).get("block_disk_cache"),
            "after_restart_block_disk": (cache.get("stats_after_restart") or {}).get("block_disk_cache"),
            "current_restart_l2_gate_status": (
                dsv4_responses_restart_l2_gate.get("status")
            ),
            "current_restart_l2_checks": {
                key: restart_l2_checks.get(key)
                for key in (
                    "native_cache",
                    "native_prefix",
                    "native_paged",
                    "native_l2",
                    "generic_tq_kv_off",
                    "disk_write_before_restart",
                    "restart_l2_disk_hit",
                    "restart_dsv4_cache_hit",
                    "same_block_disk_cache_dir",
                    "fresh_run_nonce",
                    "server_restarted",
                    "store_turn_fresh",
                )
            },
            "current_restart_cache_dir": (
                dsv4_responses_restart_l2_gate.get("cache_dir")
            ),
            "current_restart_before_block_disk": restart_l2_before_block,
            "current_restart_after_block_disk": restart_l2_after_block,
            "current_restart_cached_tokens": restart_l2_cached_tokens,
            "current_restart_cache_detail": restart_l2_cache_detail,
            "current_restart_l2_disk_hits": restart_l2_after_block.get("disk_hits"),
            "current_restart_l2_disk_writes_before": restart_l2_before_block.get(
                "disk_writes"
            ),
            "current_restart_l2_ok": current_restart_l2_ok,
        },
    )
    _add(
        requirements,
        "DSV4 Responses one-tool call stops after tool result",
        _status(
            (
                len(round1_calls) == 1
                and round1_calls[0].get("name") == "list_directory"
                and not round2_calls
                and round2_body.get("output_text") == "DONE"
            )
            or current_one_tool_stop_ok
        ),
        (
            ["build/dev-ui-dsv4-live-tool-proof-20260521/result.json"]
            if (
                len(round1_calls) == 1
                and round1_calls[0].get("name") == "list_directory"
                and not round2_calls
                and round2_body.get("output_text") == "DONE"
            )
            else [DSV4_RESPONSES_ONE_TOOL_STOP_REL]
        ),
        details={
            "round1_calls": round1_calls,
            "round2_output_text": round2_body.get("output_text"),
            "round2_function_calls": round2_calls,
            "current_one_tool_gate_status": (
                dsv4_responses_one_tool_stop.get("status")
            ),
            "current_one_tool_checks": one_tool_stop_checks,
            "current_one_tool_round1_calls": one_tool_round1_calls,
            "current_one_tool_round2_function_calls": one_tool_round2_calls,
            "current_one_tool_round2_output_text": one_tool_round2.get(
                "output_text"
            ),
            "current_one_tool_tools_still_available": one_tool_round2.get(
                "tools_still_available"
            ),
        },
    )
    _add(
        requirements,
        "DSV4 can perform multiple tool iterations then final answer",
        _status(
            (two_tool_names == ["list_directory", "write_file"] and final_text == "DONE")
            or (
                default_tool_names == [
                    "list_directory",
                    "write_file",
                    "write_file",
                ]
                and default_tool_final_text == "DONE"
                and default_tool_required_checks_ok
            )
        ),
        (
            [
                "build/v1546-current-bundled-dsv4-two-tool-proof-20260521001426/result.json"
            ]
            if two_tool_names == ["list_directory", "write_file"]
            and final_text == "DONE"
            else [DSV4_DEFAULT_CACHE_TOOL_LOOP_REL]
        ),
        caveat=(
            "This proof used --disable-prefix-cache, so it proves DSML multi-tool "
            "loop behavior separately from default-on cache."
            if two_tool_names == ["list_directory", "write_file"]
            and final_text == "DONE"
            and default_tool_names
            != ["list_directory", "write_file", "write_file"]
            else None
        ),
        details={
            "executed_tools": executed_tools,
            "final_text": final_text,
            "current_default_cache_executed_tool_names": default_tool_names,
            "current_default_cache_final_text": default_tool_final_text,
            "current_default_cache_required_checks_ok": (
                default_tool_required_checks_ok
            ),
        },
    )
    _add(
        requirements,
            "DSV4 default-cache multi-tool agent loop is proven",
            _status(
                default_tool_names == ["list_directory", "write_file", "write_file"]
                and default_tool_final_text == "DONE"
                and default_tool_cache_ok
            ),
        [DSV4_DEFAULT_CACHE_TOOL_LOOP_REL],
        caveat=(
            None
            if default_tool_cache_ok
            else (
                "Existing multi-tool proof failed required check(s): "
                + ", ".join(failed_required_default_tool_checks)
            )
            if failed_required_default_tool_checks
            else "Existing multi-tool proof does not prove the default DSV4 native prefix/paged/L2 cache path."
        ),
        details={
            "artifact_status": default_cache_tool_loop.get("status"),
            "artifact_reason": default_cache_tool_loop.get("reason"),
            "executed_tool_names": default_tool_names,
            "final_text": default_tool_final_text,
            "launch_has_disable_prefix_cache": "--disable-prefix-cache" in default_tool_cmd,
            "launch_has_dsv4_enable_prefix_cache": "--dsv4-enable-prefix-cache"
            in default_tool_cmd,
            "launch_has_use_paged_cache": "--use-paged-cache" in default_tool_cmd,
            "launch_has_block_disk_cache": "--enable-block-disk-cache"
            in default_tool_cmd,
            "native_cache_type": default_tool_native.get("cache_type"),
            "native_cache_prefix": default_tool_native.get("prefix"),
            "native_cache_paged": default_tool_native.get("paged"),
            "native_cache_block_disk_l2": default_tool_native.get("block_disk_l2"),
            "generic_turboquant_kv": default_tool_native.get("generic_turboquant_kv"),
            "tool_parser_dsml": default_tool_parser_ok,
            "reasoning_parser_deepseek_r1": default_reasoning_parser_ok,
            "tool_loop_checks": {
                key: default_tool_checks.get(key)
                for key in default_tool_detail_checks
            },
            "failed_required_tool_loop_checks": failed_required_default_tool_checks,
            "tool_loop_cached_tokens": default_tool_cached_tokens,
            "tool_loop_cache_details": default_tool_cache_details,
            "tool_loop_cache_detail_has_dsv4": default_tool_cache_detail_has_dsv4,
            "tool_loop_round_outputs": [
                row.get("output_text") for row in default_tool_rounds
            ],
            "tool_loop_round_response_diagnostics": [
                row.get("response_diagnostics") for row in default_tool_rounds
            ],
            "explicit_thinking_tool_loop_diagnostic": thinking_tool_diagnostic,
            "no_cache_tool_loop_diagnostic": no_cache_tool_diagnostic,
            "prompt_guard_tool_loop_diagnostic": prompt_guard_tool_diagnostic,
            "copy_block_tool_loop_diagnostic": copy_block_tool_diagnostic,
            "thinking_copy_block_tool_loop_diagnostic": (
                thinking_copy_block_diagnostic
            ),
            "default_cache_tool_loop_dry_run_controls": dryrun_controls_diagnostic,
            "default_cache_tool_loop_root_cause": (
                default_cache_tool_loop_root_cause
            ),
            "code_round_request_controls": default_cache_tool_loop.get(
                "code_round_request_controls"
            ),
            "code_tool_expected_content": code_tool_expected,
            "code_tool_actual_content": code_tool_actual,
            "code_tool_first_difference": code_tool_probe.get("first_difference"),
            "code_tool_missing_identifiers": code_tool_missing_identifiers,
            "code_tool_corrupt_identifier_patterns": code_tool_corrupt_identifier_patterns,
        },
    )
    tool_call_checks = tool_call_contract.get("checks") or {}
    tool_call_cap_hash_ok, tool_call_cap_hash_details = _source_hash_status(
        root,
        tool_call_contract,
        TOOL_CALL_SOURCE_HASH_FILES,
    )
    _add(
        requirements,
        "App maxToolIterations cap is enforced for DSV4 tool loop",
        _status(
            tool_call_checks.get("panel_max_tool_iterations_caps_tool_loops") is True
            and not tool_call_contract.get("failed")
            and not tool_call_contract.get("missing_markers")
            and tool_call_cap_hash_ok
        ),
        [TOOL_CALL_CONTRACT_REL],
        details={
            "contract_status": tool_call_contract.get("status"),
            "contract_checks": {
                "panel_max_tool_iterations_caps_tool_loops": (
                    tool_call_checks.get("panel_max_tool_iterations_caps_tool_loops") is True
                ),
                "panel_tool_executor_blocks_unsafe_paths_and_commands": (
                    tool_call_checks.get("panel_tool_executor_blocks_unsafe_paths_and_commands") is True
                ),
                "all_required_tool_call_markers_present": (
                    tool_call_checks.get("all_required_tool_call_markers_present") is True
                ),
                "live_default_cache_dsv4_tool_loop_artifact_passed": (
                    tool_call_checks.get("live_default_cache_dsv4_tool_loop_artifact_passed") is True
                ),
            },
            "failed": tool_call_contract.get("failed") or [],
            "missing_markers": tool_call_contract.get("missing_markers") or [],
            "open_proof_gaps": tool_call_contract.get("open_proof_gaps") or [],
            **tool_call_cap_hash_details,
        },
    )
    max_output_context_ok, max_output_context_checks = _contract_checks(
        max_output_context_contract, MAX_OUTPUT_CONTEXT_CONTRACT_CHECKS
    )
    max_output_context_hash_ok, max_output_context_hash_details = _source_hash_status(
        root,
        max_output_context_contract,
        MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES,
    )
    _add(
        requirements,
        "Server default max output and max context are distinct and map to correct CLI flags",
        _status(
            max_output_context_ok and max_output_context_hash_ok
        ),
        [
            max_output_context_contract_rel,
        ],
        details={
            "contract_status": max_output_context_contract.get("status"),
            "contract_checks": max_output_context_checks,
            **max_output_context_hash_details,
        },
    )
    panel_settings_ok, panel_settings_checks = _contract_checks(
        panel_settings_contract, PANEL_SETTINGS_CONTRACT_CHECKS
    )
    panel_settings_hash_ok, panel_settings_hash_details = _source_hash_status(
        root, panel_settings_contract, PANEL_SETTINGS_SOURCE_HASH_FILES
    )
    _add(
        requirements,
        "Panel settings keep DSV4 cache, max output, and max context controls unambiguous",
        _status(panel_settings_ok and panel_settings_hash_ok),
        [PANEL_SETTINGS_CONTRACT_REL],
        caveat=(
            None
            if panel_settings_ok and panel_settings_hash_ok
            else "Run the no-heavy panel settings contract before claiming settings/UI coverage."
        ),
        details={
            "contract_status": panel_settings_contract.get("status"),
            "contract_checks": panel_settings_checks,
            "commands": panel_settings_contract.get("commands"),
            **panel_settings_hash_details,
        },
    )
    cache_architecture_checks = cache_architecture_contract.get("checks") or {}
    cache_architecture_check_status = {
        key: cache_architecture_checks.get(key) is True
        for key in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS
    }
    cache_architecture_matrix = cache_architecture_contract.get("cache_family_matrix") or {}
    missing_cache_family_rows = [
        row_id
        for row_id in EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
        if row_id not in cache_architecture_matrix
    ]
    failed_cache_family_rows = [
        row_id
        for row_id in EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
        if (cache_architecture_matrix.get(row_id) or {}).get("status") != "pass"
    ]
    cache_architecture_hash_ok, cache_architecture_hash_details = _source_hash_status(
        root,
        cache_architecture_contract,
        CACHE_ARCHITECTURE_SOURCE_HASH_FILES,
    )
    _add(
        requirements,
        "Cross-family cache architecture is classified per family",
        _status(
            cache_architecture_contract.get("status") == "pass"
            and all(cache_architecture_check_status.values())
            and not missing_cache_family_rows
            and not failed_cache_family_rows
            and not cache_architecture_contract.get("failed")
            and not cache_architecture_contract.get("missing_markers")
            and not cache_architecture_contract.get("missing_api_checks")
            and not cache_architecture_contract.get("missing_api_command_markers")
            and not cache_architecture_contract.get("missing_panel_markers")
            and cache_architecture_hash_ok
        ),
        [CACHE_ARCHITECTURE_CONTRACT_REL],
        caveat="Static architecture proof only; broad live generation still requires per-family live rows.",
        details={
            "contract_status": cache_architecture_contract.get("status"),
            "contract_checks": cache_architecture_check_status,
            "missing_family_rows": missing_cache_family_rows,
            "failed_family_rows": failed_cache_family_rows,
            "failed": cache_architecture_contract.get("failed") or [],
            "missing_markers": cache_architecture_contract.get("missing_markers") or [],
            "missing_api_checks": cache_architecture_contract.get("missing_api_checks") or [],
            "missing_api_command_markers": cache_architecture_contract.get("missing_api_command_markers") or [],
            "missing_panel_markers": cache_architecture_contract.get("missing_panel_markers") or [],
            "cache_family_matrix": {
                row_id: cache_architecture_matrix.get(row_id)
                for row_id in EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
            },
            **cache_architecture_hash_details,
        },
    )
    model_family_ok, model_family_details = _contract_detail(
        root,
        model_family_contract,
        MODEL_FAMILY_CONTRACT_CHECKS,
        MODEL_FAMILY_SOURCE_HASH_FILES,
    )
    parser_registry_ok, parser_registry_details = _contract_detail(
        root,
        parser_registry_contract,
        PARSER_REGISTRY_CONTRACT_CHECKS,
        PARSER_REGISTRY_SOURCE_HASH_FILES,
    )
    model_artifact_format_ok, model_artifact_format_details = _contract_detail(
        root,
        model_artifact_format_contract,
        MODEL_ARTIFACT_FORMAT_CONTRACT_CHECKS,
        MODEL_ARTIFACT_FORMAT_SOURCE_HASH_FILES,
    )
    _add(
        requirements,
        "High-risk model family parser, artifact, and launch policy gates are current",
        _status(model_family_ok and parser_registry_ok and model_artifact_format_ok),
        [
            MODEL_FAMILY_CONTRACT_REL,
            PARSER_REGISTRY_CONTRACT_REL,
            MODEL_ARTIFACT_FORMAT_CONTRACT_REL,
        ],
        caveat=(
            "This is no-heavy source/static compatibility proof; live multi-turn "
            "output quality and speed rows remain separate."
        ),
        details={
            "model_family": model_family_details,
            "parser_registry": parser_registry_details,
            "model_artifact_format": model_artifact_format_details,
        },
    )
    generation_defaults_ok, generation_defaults_details = _contract_detail(
        root,
        generation_defaults_contract,
        GENERATION_DEFAULTS_CONTRACT_CHECKS,
        GENERATION_DEFAULTS_SOURCE_HASH_FILES,
    )
    native_mtp_ok, native_mtp_details = _contract_detail(
        root,
        native_mtp_contract,
        NATIVE_MTP_CONTRACT_CHECKS,
        NATIVE_MTP_SOURCE_HASH_FILES,
    )
    vl_media_ok, vl_media_details = _contract_detail(
        root,
        vl_media_contract,
        VL_MEDIA_CONTRACT_CHECKS,
        VL_MEDIA_SOURCE_HASH_FILES,
    )
    _add(
        requirements,
        "Generation defaults, Native MTP, and VL media gates are current",
        _status(generation_defaults_ok and native_mtp_ok and vl_media_ok),
        [
            GENERATION_DEFAULTS_CONTRACT_REL,
            NATIVE_MTP_CONTRACT_REL,
            VL_MEDIA_CONTRACT_REL,
        ],
        caveat=(
            "This proves no-heavy wiring, metadata ownership, and media/cache "
            "compatibility. Live MTP speed/equivalence and video/Omni quality "
            "remain separate live rows."
        ),
        details={
            "generation_defaults": generation_defaults_details,
            "native_mtp": native_mtp_details,
            "vl_media": vl_media_details,
        },
    )
    qwen_source_speed_ok, qwen_source_speed_details = _speed_artifact_detail(
        qwen_jang_source_speed
    )
    qwen_packaged_speed_ok, qwen_packaged_speed_details = _speed_artifact_detail(
        qwen_jang_packaged_speed
    )
    _add(
        requirements,
        "Qwen/JANG packaged MX matmul speed is release-cleared",
        _status(qwen_source_speed_ok and qwen_packaged_speed_ok),
        [QWEN_JANG_SOURCE_SPEED_REL, QWEN_JANG_PACKAGED_SPEED_REL],
        caveat=(
            "Source/native-wheel speed and packaged-app speed are separate. "
            "A source pass does not clear a packaged compat-wheel PP review."
        ),
        details={
            "source": qwen_source_speed_details,
            "packaged": qwen_packaged_speed_details,
        },
    )
    qwen_native_mtp_decode_ok, qwen_native_mtp_decode_details = (
        _native_mtp_ab_detail(qwen_native_mtp_ab)
    )
    _add(
        requirements,
        "Qwen native MTP live decode speed and output equivalence are release-cleared",
        _status(qwen_native_mtp_decode_ok),
        [QWEN_NATIVE_MTP_AB_REL],
        caveat=(
            "This row covers decode speed/equivalence only. Prompt-processing "
            "throughput is tracked separately so MTP decode is not blamed for "
            "a broader Qwen/JANG PP floor failure."
        ),
        details=qwen_native_mtp_decode_details,
    )
    qwen_prompt_ok, qwen_prompt_details = _prompt_processing_speed_detail(
        qwen_jang_text_baseline_speed,
        qwen_native_mtp_prefill_speed,
        qwen_native_mtp_packaged_prefill_speed,
        qwen_native_mtp_prefill_trace,
        qwen_native_mtp_no_prefix_logits,
        qwen_native_mtp_hybrid_long_prefix_split,
        qwen_native_mtp_kvnone,
        qwen_native_mtp_route_trace,
        [qwen_raw_forward_ab_1024, qwen_raw_forward_ab_4096],
        qwen_native_mtp_norm_shift_clearance,
    )
    _add(
        requirements,
        "Qwen 27B JANG_4M prompt-processing speed floor is release-cleared",
        _status(qwen_prompt_ok),
        [
            QWEN_JANG_TEXT_BASELINE_SPEED_REL,
            QWEN_NATIVE_MTP_NORM_SHIFT_CLEARANCE_REL,
        ],
        caveat=(
            "The current clearance row is the post norm-format loader fix live "
            "default MTP/VLM artifact. Older native-MTP/VL prefill diagnostics "
            "remain in the details because they identify the prior slow/corrupt "
            "route; packaged-app parity is still covered by the packaged "
            "integrity gate and separate Qwen/JANG packaged speed row."
        ),
        details=qwen_prompt_details,
    )
    api_cache_ok, api_cache_checks = _contract_checks(
        api_cache_contract, API_CACHE_CONTRACT_CHECKS
    )
    api_cache_hash_ok, api_cache_hash_details = _source_hash_status(
        root, api_cache_contract, API_CACHE_SOURCE_HASH_FILES
    )
    _add(
        requirements,
        "Current-source API adapters and non-DSV4 cache contracts are no-heavy covered",
        _status(api_cache_ok and api_cache_hash_ok),
        [API_CACHE_CONTRACT_REL],
        caveat=(
            None
            if api_cache_ok and api_cache_hash_ok
            else "Run the current-source no-heavy API/cache contract proof before claiming API/cache coverage."
        ),
        details={
            "contract_status": api_cache_contract.get("status"),
            "contract_checks": api_cache_checks,
            "commands": api_cache_contract.get("commands"),
            **api_cache_hash_details,
        },
    )
    ling_quality_ok, ling_quality_details = _ling_multilingual_quality_detail(
        root,
        ling_installed_live,
        ling_jangtq_strict_russian,
        ling_mxfp4_strict_russian,
    )
    _add(
        requirements,
        "Ling/Bailing multilingual output quality is release-cleared",
        _status(ling_quality_ok),
        [
            LING_INSTALLED_LIVE_AUDIT_REL,
        ],
        caveat=(
            None
            if ling_quality_ok
            else "Current live artifacts show CJK leakage in Russian/non-CJK prompts or missing clearance evidence. Do not release-claim Ling/Bailing multilingual output quality yet."
        ),
        details=ling_quality_details,
    )
    gemma4_visible_ok, gemma4_visible_details = _gemma4_visible_content_detail(
        gemma4_responses_visible_contract,
        root,
        gemma4_responses_visible_contract_rel,
        gemma4_responses_visible_nocache,
        gemma4_responses_visible_512_nocache,
        gemma4_responses_thinking_off_nocache,
        gemma4_chat_visible_nocache,
        gemma4_local_metadata_audit,
        gemma4_responses_unsupported_thinking_budget,
    )
    gemma4_visible_evidence = (
        [gemma4_responses_visible_contract_rel]
        if gemma4_responses_visible_contract_rel == GEMMA4_RESPONSES_VISIBLE_CURRENT_REL
        else [
            GEMMA4_RESPONSES_VISIBLE_CONTRACT_REL,
            GEMMA4_RESPONSES_UNSUPPORTED_THINKING_BUDGET_REL,
            GEMMA4_RESPONSES_VISIBLE_NOCACHE_REL,
            GEMMA4_RESPONSES_VISIBLE_512_NOCACHE_REL,
            GEMMA4_RESPONSES_THINKING_OFF_NOCACHE_REL,
            GEMMA4_CHAT_VISIBLE_NOCACHE_REL,
            GEMMA4_LOCAL_METADATA_AUDIT_REL,
        ]
    )
    _add(
        requirements,
        "Gemma4 26B CRACK Responses visible-content and language quality is release-cleared",
        _status(gemma4_visible_ok),
        gemma4_visible_evidence,
        caveat=(
            None
            if gemma4_visible_ok
            else "Current Gemma4 Responses thinking-budget artifact lacks visible assistant content or is missing. Do not treat HTTP 200, import success, or production-family PASS as Gemma4 visible quality clearance."
        ),
        details=gemma4_visible_details,
    )
    gemma4_speed_ok, gemma4_speed_details = _gemma4_mixed_swa_speed_floor_detail(
        gemma4_mixed_swa_speed_payloads,
        root,
        gemma4_mixed_swa_sustained_cachehit,
        gemma4_mixed_swa_short_nocache_repeat,
        gemma4_mixed_swa_short_nocache_scheduler_trace,
        gemma4_mixed_swa_short_nocache_sync_eval_ab,
        gemma4_mixed_swa_short_nocache_streaming,
    )
    _add(
        requirements,
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared",
        _status(gemma4_speed_ok),
        list(gemma4_mixed_swa_speed_rels),
        caveat=(
            None
            if gemma4_speed_ok
            else "Current Gemma4 mixed-SWA bundled/app-engine artifacts do not all meet the 80 tok/s decode floor. Do not claim Gemma4 heterogeneous SWA speed clearance from one cold sample."
        ),
        details=gemma4_speed_details,
    )
    all_local_smoke_ok, all_local_smoke_details = _all_local_model_smoke_detail(
        all_local_model_smoke_payloads,
        root,
        diagnostics={
            ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_JANGTQ4_REL: _zaya_vl_jangtq4_diagnostics(
                root,
                zaya_vl_jangtq4_ack_diagnostic,
                zaya_vl_jangtq4_rendered_prompt_compare,
            ),
            ALL_LOCAL_MODEL_SMOKE_NEMOTRON_OMNI_JANGTQ_REL: _nemotron_omni_no_media_diagnostics(
                root,
                nemotron_omni_no_media_diagnostic,
                nemotron_omni_no_media_prompt_variants,
                nemotron_omni_no_media_system_prompt,
                nemotron_omni_no_media_system_negative,
            ),
        },
    )
    all_local_smoke_release_ok = all_local_smoke_ok
    _add(
        requirements,
        "Cross-family live multi-turn smoke matrix is release-cleared",
        _status(all_local_smoke_release_ok),
        [rel for rel, _payload in all_local_model_smoke_payloads],
        caveat=(
            None
            if all_local_smoke_ok
            else "Current live all-local smoke coverage is incomplete. Do not claim broad family support until DSV4, Gemma4, Hy3, Ling/Bailing, MiniMax, MiMo-V2, Nemotron, Qwen3.6, ZAYA text, and ZAYA-VL live rows pass cache, recall, visible-output, and gibberish checks."
        ),
        details=all_local_smoke_details,
    )
    mimo_quality_ok, mimo_quality_details = _mimo_v2_jang2l_quality_detail(root)
    _add(
        requirements,
        "MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is release-cleared",
        _status(mimo_quality_ok),
        [
            MIMO_V2_JANG2L_STRUCTURAL_VERIFY_REL,
            MIMO_V2_JANG2L_TEXT_CACHE_REL,
            MIMO_V2_JANG2L_SWITCHGLU_PARITY_REL,
            MIMO_V2_JANG2L_LENGTH_SWEEP_REL,
            MIMO_V2_JANG2L_TOOL_DIALECT_REL,
            MIMO_V2_JANG2L_CURRENT_AUDIT_REL,
        ],
        caveat=(
            None
            if mimo_quality_ok
            else "MiMo V2.5 JANG_2L has current structural, narrow text/cache, and selected-expert parity evidence, but current local artifacts still show long-prompt corruption and tool protocol failure. Do not release-clear MiMo from short smokes."
        ),
        details=mimo_quality_details,
    )
    release_blockers = release_manifest.get("release_blockers")
    if not isinstance(release_blockers, list):
        release_blockers = []
    issue179_blocker = next(
        (
            blocker
            for blocker in release_blockers
            if isinstance(blocker, dict)
            and blocker.get("id") == "issue179_minimax_k_root_cause_audit"
        ),
        None,
    )
    current_sweep = release_manifest.get("current_proof_sweep")
    if not isinstance(current_sweep, dict):
        current_sweep = {}
    direct_issue179_audit = _load(root, CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT)
    direct_issue179_not_proven = [
        str(item) for item in direct_issue179_audit.get("not_proven", []) if str(item)
    ]
    if direct_issue179_audit.get("status") in {"open", "pass"}:
        issue179_audit = direct_issue179_audit
        if (
            direct_issue179_audit.get("status") == "pass"
            and not direct_issue179_not_proven
        ):
            issue179_blocker = None
    else:
        issue179_audit = current_sweep.get("issue179_minimax_k_root_cause_audit")
        if not isinstance(issue179_audit, dict):
            issue179_audit = {}
    issue179_evidence = (
        str(issue179_blocker.get("evidence"))
        if isinstance(issue179_blocker, dict) and issue179_blocker.get("evidence")
        else CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    )
    if not issue179_audit:
        issue179_audit = direct_issue179_audit or _load(root, issue179_evidence)
    issue179_not_proven = [
        str(item) for item in issue179_audit.get("not_proven", []) if str(item)
    ]
    issue179_open = isinstance(issue179_blocker, dict) or issue179_audit.get(
        "status"
    ) == "open"
    issue179_pass = (
        not issue179_open
        and issue179_audit.get("status") == "pass"
        and _path_present(root, CURRENT_RELEASE_REGRESSION_MANIFEST_REL)
    )
    _add(
        requirements,
        "MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared",
        "pass" if issue179_pass else "open",
        [CURRENT_RELEASE_REGRESSION_MANIFEST_REL, issue179_evidence],
        caveat=(
            None
            if issue179_pass
            else (
                "MiniMax #179 remains open while "
                + (
                    "; ".join(issue179_not_proven)
                    if issue179_not_proven
                    else "reporter bundle/session parity and screenshot-shaped output reproduction remain unproven"
                )
                + "."
            )
        ),
        details={
            "release_blocker_id": "issue179_minimax_k_root_cause_audit",
            "release_blocker": issue179_blocker,
            "audit_status": issue179_audit.get("status") or "missing",
            "not_proven": issue179_not_proven,
            "release_boundary": issue179_audit.get("release_boundary"),
        },
    )
    if real_ui_live_model_matrix:
        real_ui_unblocked_non_mimo_ok = (
            real_ui_live_model_matrix.get("unblocked_non_mimo_status") == "pass"
            and not real_ui_live_model_matrix.get(
                "unblocked_non_mimo_missing_families"
            )
            and not real_ui_live_model_matrix.get(
                "unblocked_non_mimo_partial_families"
            )
        )
        real_ui_covered_families = real_ui_live_model_matrix.get("covered_families")
        if not isinstance(real_ui_covered_families, dict):
            real_ui_covered_families = {}
        real_ui_excluded_families = [
            str(item)
            for item in real_ui_live_model_matrix.get(
                "unblocked_non_mimo_excluded_families", []
            )
        ]
        real_ui_excluded_family_set = set(real_ui_excluded_families)
        real_ui_unblocked_non_mimo_evidence = [
            str(row.get("artifact"))
            for family, row in sorted(real_ui_covered_families.items())
            if family not in real_ui_excluded_family_set
            and isinstance(row, dict)
            and row.get("artifact")
        ]
        _add(
            requirements,
            "Real Electron UI unblocked non-MiMo live model matrix is proven",
            _status(real_ui_unblocked_non_mimo_ok),
            [
                CURRENT_RELEASE_REGRESSION_MANIFEST_REL,
                *real_ui_unblocked_non_mimo_evidence,
            ],
            caveat=(
                None
                if real_ui_unblocked_non_mimo_ok
                else "Current real-UI proof is still missing or partial for at least one unblocked non-MiMo family."
            ),
            details={
                "status": real_ui_live_model_matrix.get("unblocked_non_mimo_status")
                or "open",
                "release_boundary": (
                    "unblocked_non_mimo_real_ui_matrix_proven_with_explicit_exclusions"
                ),
                "missing_family_keys": [
                    str(item)
                    for item in real_ui_live_model_matrix.get(
                        "unblocked_non_mimo_missing_families", []
                    )
                ],
                "partial_family_keys": [
                    str(item)
                    for item in real_ui_live_model_matrix.get(
                        "unblocked_non_mimo_partial_families", []
                    )
                ],
                "excluded_families": sorted(real_ui_excluded_families),
                "covered_family_keys": sorted(
                    str(family)
                    for family, row in real_ui_covered_families.items()
                    if family not in real_ui_excluded_family_set
                    and isinstance(row, dict)
                    and row.get("status") == "pass"
                ),
                "real_ui_live_model_matrix": real_ui_live_model_matrix,
            },
        )
    _add(
        requirements,
        "Real Electron UI cross-family live model matrix is release-cleared",
        "open",
        [
            CURRENT_RELEASE_REGRESSION_MANIFEST_REL,
            "docs/internal/agent-notes/2026-05-26-live-chat-tools-reasoning-proof.json",
            "docs/internal/agent-notes/2026-05-26-live-chat-tools-reasoning-chat-settings.png",
            "docs/internal/agent-notes/2026-05-26-live-chat-tools-reasoning-server-cache-settings.png",
            ALL_LOCAL_MODEL_SMOKE_ZAYA_TEXT_REL,
            ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_CURRENT_REL,
            ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_JANGTQ4_REL,
            ALL_LOCAL_MODEL_SMOKE_NEMOTRON_OMNI_JANGTQ_REL,
            ALL_LOCAL_MODEL_SMOKE_GEMMA4_26B_CRACK_REL,
            ALL_LOCAL_MODEL_SMOKE_LING_BAILING_JANGTQ_REL,
            ALL_LOCAL_MODEL_SMOKE_QWEN36_MXFP4_CRACK_REL,
            ALL_LOCAL_MODEL_SMOKE_HY3_JANGTQ2_REL,
            ALL_LOCAL_MODEL_SMOKE_MINIMAX_SMALL_JANGTQ_REL,
            ALL_LOCAL_MODEL_SMOKE_MIMO_V2_JANG2L_REL,
            ALL_LOCAL_MODEL_SMOKE_DSV4_JANGTQ_K_REL,
        ],
        caveat=(
            "Current proof separates mock Electron-dev UI wiring from bundled/API "
            "live-model smokes. Do not claim the real UI cross-family matrix is "
            "release-cleared until each target family is exercised through the "
            "current dev app UI with a real loaded model, real chat settings, "
            "tool/reasoning display, cache reuse, and no parser leakage."
        ),
        details={
            "status": "open",
            "real_ui_live_model_matrix": real_ui_live_model_matrix,
            "mock_ui_wiring_artifacts": [
                "docs/internal/agent-notes/2026-05-26-live-chat-tools-reasoning-proof.json",
                "docs/internal/agent-notes/2026-05-26-live-chat-tools-reasoning-chat-settings.png",
                "docs/internal/agent-notes/2026-05-26-live-chat-tools-reasoning-server-cache-settings.png",
            ],
            "server_live_smoke_matrix_status": (
                "pass" if all_local_smoke_ok else "open"
            ),
            "required_family_keys": all_local_smoke_details.get(
                "required_family_keys", []
            ),
            "covered_family_keys": all_local_smoke_details.get(
                "covered_family_keys", []
            ),
            "missing_required_family_keys": all_local_smoke_details.get(
                "missing_required_family_keys", []
            ),
            "real_ui_missing_required_family_keys": [
                str(item)
                for item in real_ui_live_model_matrix.get("missing_families", [])
            ],
            "real_ui_partial_family_keys": [
                str(item)
                for item in real_ui_live_model_matrix.get("partial_families", [])
            ],
            "real_ui_blocking_required_family_artifacts": (
                _real_ui_blocking_family_artifacts(real_ui_live_model_matrix)
            ),
            "release_boundary": (
                "current_real_ui_matrix_is_authoritative_for_unblocked_non_mimo"
                if real_ui_live_model_matrix
                else "mock_ui_plus_server_smoke_is_not_real_ui_live_model_clearance"
            ),
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
        },
    )
    quality_ok, quality_details = _dsv4_quality_clearance(quality_clearance, root)
    dsv4_prompt_rail_exactness_detail = _dsv4_prompt_rail_exactness_detail(
        dsv4_prompt_rail_exactness,
        dsv4_route_mode_dryrun,
        root,
        dsv4_prompt_rail_exactness_rel,
        dsv4_route_mode_dryrun_rel,
    )
    quality_ok = (
        quality_ok
        and dsv4_prompt_rail_exactness_detail.get("status") == "pass"
    )
    dsv4_source_bundle_defaults_detail = _dsv4_prompt_rail_exactness_detail(
        dsv4_current_source_bundle_defaults_exactness,
        dsv4_current_source_bundle_defaults_dryrun,
        root,
        DSV4_CURRENT_SOURCE_BUNDLE_DEFAULTS_EXACTNESS_REL,
        DSV4_CURRENT_SOURCE_BUNDLE_DEFAULTS_DRYRUN_REL,
    )
    quality_ok = (
        quality_ok
        and dsv4_source_bundle_defaults_detail.get("status") == "pass"
    )
    dsv4_current_jangtqk_direct_off_detail = _dsv4_prompt_rail_exactness_detail(
        dsv4_current_jangtqk_direct_off_recheck,
        {},
        root,
        DSV4_CURRENT_JANGTQK_DIRECT_OFF_RECHECK_REL,
        "",
    )
    if dsv4_current_jangtqk_direct_off_detail.get("present") is True:
        quality_ok = (
            quality_ok
            and dsv4_current_jangtqk_direct_off_detail.get("status") == "pass"
        )
    quality_details.update(
        {
            "long_context_status": longctx.get("status"),
            "long_context_notes": longctx.get("notes"),
            "current_installed_identifier_canary": _dsv4_identifier_canary_detail(
                dsv4_current_identifier_canary, root, dsv4_current_identifier_canary_rel
            ),
            "current_installed_identifier_matrix": _dsv4_identifier_matrix_detail(
                dsv4_current_identifier_matrix, root
            ),
            "current_installed_tokenizer_roundtrip": _dsv4_tokenizer_roundtrip_detail(
                dsv4_installed_tokenizer_roundtrip, root
            ),
            "current_installed_logprob_copy_probe": _dsv4_logprob_copy_detail(
                dsv4_live_logprobs_copy, root
            ),
            "current_installed_logprob_context_matrix": _dsv4_logprob_context_matrix_detail(
                dsv4_live_logprob_context_matrix, root
            ),
            "current_installed_unique_prefix_identifier_probe": _dsv4_cache_context_identifier_detail(
                dsv4_live_cache_context_identifier, root
            ),
            "current_source_nocache_identifier_probe": _dsv4_source_nocache_identifier_detail(
                dsv4_source_nocache_identifier, root
            ),
            "current_source_same_prompt_cache_boundary": (
                _dsv4_source_same_prompt_cache_boundary_detail(
                    dsv4_source_same_prompt_nocache,
                    dsv4_source_cache_comparison,
                    root,
                )
            ),
            "current_installed_prompt_rail_exactness_probe": (
                dsv4_prompt_rail_exactness_detail
            ),
            "current_generated_only_direct_rail_exactness_subset": (
                _dsv4_prompt_rail_exactness_detail(
                    dsv4_current_generated_only_direct_rail_exactness,
                    {},
                    root,
                    DSV4_CURRENT_GENERATED_ONLY_DIRECT_RAIL_EXACTNESS_REL,
                    "",
                )
            ),
            "current_requested_thinking_exactness_subset": (
                _dsv4_prompt_rail_exactness_detail(
                    dsv4_current_requested_thinking_exactness,
                    {},
                    root,
                    DSV4_CURRENT_REQUESTED_THINKING_EXACTNESS_REL,
                    "",
                )
            ),
            "current_rep1_direct_vs_requested_thinking_exactness_subset": (
                _dsv4_prompt_rail_exactness_detail(
                    dsv4_current_rep1_rail_exactness,
                    {},
                    root,
                    DSV4_CURRENT_REP1_RAIL_EXACTNESS_REL,
                    "",
                )
            ),
            "current_source_rep1_direct_vs_requested_thinking_exactness_subset": (
                _dsv4_prompt_rail_exactness_detail(
                    dsv4_current_source_rep1_rail_exactness,
                    dsv4_route_mode_dryrun,
                    root,
                    DSV4_CURRENT_SOURCE_REP1_RAIL_EXACTNESS_REL,
                    dsv4_route_mode_dryrun_rel,
                )
            ),
            "current_source_token_tail_direct_vs_requested_thinking_exactness_subset": (
                _dsv4_prompt_rail_exactness_detail(
                    dsv4_current_source_token_tail_ab_exactness,
                    {},
                    root,
                    DSV4_CURRENT_SOURCE_TOKEN_TAIL_AB_EXACTNESS_REL,
                    "",
                )
            ),
            "current_source_rep1_direct_only_exactness_subset": (
                _dsv4_prompt_rail_exactness_detail(
                    dsv4_current_source_rep1_direct_only,
                    dsv4_route_mode_dryrun,
                    root,
                    DSV4_CURRENT_SOURCE_REP1_DIRECT_ONLY_REL,
                    dsv4_route_mode_dryrun_rel,
                )
            ),
            "current_source_bundle_defaults_exactness_subset": (
                dsv4_source_bundle_defaults_detail
            ),
            "current_source_full_output_preflight": (
                _dsv4_source_memory_preflight_detail(
                    dsv4_current_source_memory_preflight,
                    root,
                )
            ),
            "current_jangtqk_direct_off_recheck": (
                dsv4_current_jangtqk_direct_off_detail
            ),
            "current_chatmax_prompt_trigger_probe": (
                _dsv4_chatmax_prompt_trigger_detail(
                    dsv4_chatmax_prompt_trigger,
                    root,
                )
            ),
            "current_chatmax_budget_stop_rail_probe": (
                _dsv4_chatmax_budget_stop_rail_detail(
                    dsv4_chatmax_budget_stop_rail,
                    root,
                )
            ),
            "current_prompt_boundary_bisection_probe": (
                _dsv4_prompt_boundary_bisection_detail(
                    dsv4_prompt_boundary_bisection,
                    root,
                )
            ),
            "current_colon_period_logprob_trace": (
                _dsv4_colon_period_logprob_trace_detail(
                    dsv4_colon_period_logprob_trace,
                    dsv4_colon_period_visible_logprob_trace,
                    root,
                )
            ),
            "current_scene_token_rank_contrast": (
                _dsv4_scene_token_rank_contrast_detail(
                    dsv4_scene_token_rank_contrast,
                    root,
                )
            ),
            "current_direct_vs_thinking_webgl_logit_probe": (
                _dsv4_direct_vs_thinking_webgl_logit_detail(
                    dsv4_direct_vs_thinking_webgl_logit,
                    root,
                )
            ),
            "current_hidden_reasoning_control_probe": (
                _dsv4_hidden_reasoning_control_detail(
                    dsv4_hidden_reasoning_control,
                    root,
                )
            ),
            "current_template_parity_diagnostic": (
                _dsv4_template_parity_diagnostic_detail(
                    dsv4_template_parity_diagnostic,
                    root,
                )
            ),
            "current_prefill_execution_variant_logits": (
                _dsv4_prefill_execution_variant_logits_detail(
                    dsv4_prefill_execution_variant_logits,
                    root,
                )
            ),
            "current_prompt_variant_logit_probe": (
                _dsv4_prompt_variant_logit_probe_detail(
                    dsv4_prompt_variant_logit_probe,
                    root,
                )
            ),
            "current_reasoning_policy_live": (
                _dsv4_reasoning_policy_live_detail(
                    dsv4_reasoning_policy_live,
                    root,
                )
            ),
            "current_batch_generator_logit_divergence": (
                _dsv4_batch_generator_logit_divergence_detail(
                    dsv4_cache_vs_full_logit_isolation,
                    dsv4_batch_generator_logit_trace,
                    dsv4_batch_generator_warmup_ablation,
                    root,
                )
            ),
        }
    )
    quality_details["direct_off_exactness_boundary"] = (
        _dsv4_direct_off_exactness_boundary(quality_details)
    )
    quality_details["exact_code_root_boundary"] = (
        _dsv4_exact_code_root_boundary(quality_details)
    )
    quality_details["failed_quality_gates"] = _dsv4_failed_quality_gates(
        quality_details
    )
    _add(
        requirements,
        "DSV4 long-output/code/file-generation quality is release-cleared",
        _status(quality_ok),
        [
            DSV4_QUALITY_CLEARANCE_REL,
            "build/current-dsv4-long-context-proof-digest-20260521.json",
            "build/current-dsv4-identifier-count-ablation-20260521/result.json",
            DSV4_CURRENT_IDENTIFIER_MATRIX_REL,
            DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL,
            DSV4_LIVE_LOGPROBS_COPY_REL,
            DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL,
            DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL,
            DSV4_SOURCE_NOCACHE_IDENTIFIER_REL,
            DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL,
            DSV4_SOURCE_CACHE_COMPARISON_REL,
            DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_REL,
            DSV4_CURRENT_GENERATED_ONLY_DIRECT_RAIL_EXACTNESS_REL,
            DSV4_CURRENT_REQUESTED_THINKING_EXACTNESS_REL,
            DSV4_CURRENT_REP1_RAIL_EXACTNESS_REL,
            DSV4_CURRENT_SOURCE_REP1_RAIL_EXACTNESS_REL,
            DSV4_CURRENT_SOURCE_TOKEN_TAIL_AB_EXACTNESS_REL,
            DSV4_CURRENT_SOURCE_REP1_DIRECT_ONLY_REL,
            DSV4_CURRENT_SOURCE_MEMORY_PREFLIGHT_REL,
            DSV4_CURRENT_ROUTE_MODE_DRYRUN_REL,
            DSV4_CURRENT_ROUTE_MODE_DRYRUN_IDENTIFIER_CANDIDATES_REL,
            DSV4_CURRENT_ROUTE_MODE_DRYRUN_COHESIVE_AUDIT_REL,
            DSV4_CHATMAX_PROMPT_TRIGGER_REL,
            DSV4_CHATMAX_BUDGET_STOP_RAIL_REL,
            DSV4_PROMPT_BOUNDARY_BISECTION_REL,
            DSV4_COLON_PERIOD_LOGPROB_TRACE_REL,
            DSV4_COLON_PERIOD_VISIBLE_LOGPROB_TRACE_REL,
            DSV4_SCENE_TOKEN_RANK_CONTRAST_REL,
            DSV4_DIRECT_VS_THINKING_WEBGL_LOGIT_PROBE_REL,
            DSV4_HIDDEN_REASONING_CONTROL_REL,
            DSV4_TEMPLATE_PARITY_DIAGNOSTIC_REL,
            DSV4_PREFILL_EXECUTION_VARIANT_LOGITS_REL,
            DSV4_PROMPT_VARIANT_LOGIT_PROBE_REL,
            DSV4_REASONING_POLICY_LIVE_REL,
            DSV4_CACHE_VS_FULL_LOGIT_ISOLATION_REL,
            DSV4_BATCH_GENERATOR_LOGIT_TRACE_REL,
            DSV4_BATCH_GENERATOR_WARMUP_ABLATION_REL,
        ],
        caveat=(
            None
            if quality_ok
            else "Current long-context digest is status=review with caveats; "
            "identifier ablations still show exact-code/identifier corruption. "
            "Do not release-claim this yet."
        ),
        details=quality_details,
    )
    _attach_evidence_file_status(requirements, root)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirements": requirements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    digest = build_digest(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(digest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out)
    for item in digest["requirements"]:
        print(f"{item['status'].upper():7} {item['requirement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
