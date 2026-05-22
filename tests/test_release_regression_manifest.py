from pathlib import Path
import shlex

from tests.cross_matrix.release_regression_manifest import (
    REQUIRED_RELEASE_DOMAINS,
    build_manifest,
)


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
    assert "Responses maxTokens below or above the server startup default remain request scoped" in joined
    assert "Responses Auto does not synthesize max_output_tokens" in joined
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
    assert "current-generation-defaults-contract-20260521.json" in joined
    assert "generation_config.json" in joined
    assert "jang_config.json" in joined
    assert "hidden sampler forcing" in joined


def test_release_regression_manifest_tracks_reasoning_template_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["reasoning-template-no-think-tag-leak"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_reasoning_template_contract.py" in joined
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
    assert "current-api-surface-contract-20260522-stream-cache-detail.json" in joined
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


def test_release_regression_manifest_tracks_tool_calls_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["tool-call-loop-parser-cleanup"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_tool_call_contract.py" in joined
    assert "current-tool-call-contract-20260521.json" in joined
    assert "Tool parser residue" in joined
    assert "DSV4" in joined
    assert "maxToolIterations" in joined
    assert "live DSV4 write_file DSML degradation is repaired schema-safely" in joined
    assert "current-tool-call-contract-20260522-dsv4-live-write-file-repair.json" in joined
    assert "current-dsv4-default-cache-tool-loop-poolon-materialized-parserfix-20260522/result.json" in joined


def test_release_regression_manifest_tracks_panel_cache_family_gating():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["panel-session-cache-settings-family-gating"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "cache_architecture"
    assert "run_noheavy_panel_settings_contract.py" in joined
    assert "current-panel-settings-contract-proof-20260522-launch-memory-warning.json" in joined
    assert "DSV4 pool quant" in joined
    assert "JANG/JANGTQ/MXFP" in joined
    assert "warning-only for lazy-mmap" in joined


def test_release_regression_manifest_tracks_mcp_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["mcp-policy-ui-gateway"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_mcp_policy_contract.py" in joined
    assert "current-mcp-policy-contract-20260521.json" in joined
    assert "MCP autodiscovery" in joined
    assert "redaction" in joined
    assert "gateway routing" in joined
    assert "panel config source" in joined
    assert "required marker" in joined


def test_release_regression_manifest_tracks_packaged_integrity_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["packaged-release-integrity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_packaged_integrity_contract.py" in joined
    assert "current-packaged-integrity-contract-20260521.json" in joined
    assert "Version triples" in joined
    assert "bundled Python hash parity" in joined
    assert "objective proof digest" in joined
    assert "objective-gate-enforced" in joined
    assert "verify-bundled" in joined


def test_release_regression_manifest_tracks_public_release_surface_preflight():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["public-release-surface-preflight"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "release_surface"
    assert "run_release_surface_contract.py" in joined
    assert "latest.json" in joined
    assert "PyPI" in joined
    assert "GitHub release" in joined
    assert "updater" in joined.lower()
    assert "post-release-updater" in joined
    assert "published updater state is complete" in joined


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

    assert "current-production-family-audit-live-dsv4-jang-local-20260522-after-stream-cache-detail.json" in joined
    assert "identifier integrity" in joined


def test_release_regression_manifest_live_soak_does_not_overclaim_qwen_mtp():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-live-multiturn-soak"]
    command_text = " ".join(row["commands"]).lower()
    proves_text = " ".join(row["proves"]).lower()

    if "qwen mtp" in proves_text and "no-heavy" not in proves_text:
        assert "qwen" in command_text and "mtp" in command_text


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
    assert "malformed num_predict" in joined
    assert "malformed context" in joined
    assert "Max Context Tokens" in joined
    assert "--max-tokens" in joined
    assert "--max-prompt-tokens" in joined
    assert "coding-tool configs" in joined
    assert "Auto chat Max Tokens omits per-request output caps" in joined


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
    assert "current-cache-architecture-contract-20260521.json" in joined
    assert "DSV4 composite" in joined
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
    assert "DSV4 panel env mapping enables pool quant only from explicit DSV4 config" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json" in joined
    assert "DSV4 timing probe covers prefix-cache replay and cold-store boundaries" in joined
    assert "Panel session launch builder preserves DSV4 default and diagnostic prefix-cache policy" in joined
    assert "Qwen3.6 hybrid and Mamba paged-cache forcing" in joined
    assert "regular KV stale saved false semantics" in joined


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
    assert "live equivalence/speed" in joined
