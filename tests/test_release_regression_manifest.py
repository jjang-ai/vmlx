from pathlib import Path

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


def test_release_regression_manifest_tracks_no_fake_sampler_policy():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    assert "generation-defaults-no-hidden-forcing" in rows
    assert rows["generation-defaults-no-hidden-forcing"]["domain"] == "generation_defaults"
    assert "generation_config.json" in " ".join(rows["generation-defaults-no-hidden-forcing"]["proves"])
    assert "jang_config.json" in " ".join(rows["generation-defaults-no-hidden-forcing"]["proves"])


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
    assert "current-api-surface-contract-20260521.json" in joined
    assert "OpenAI Chat Completions" in joined
    assert "OpenAI Responses" in joined
    assert "Anthropic" in joined
    assert "Ollama" in joined


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


def test_release_regression_manifest_tracks_live_only_boundaries():
    manifest = build_manifest()
    live_rows = [row for row in manifest["rows"] if row["mode"] == "live"]
    live_ids = {row["id"] for row in live_rows}

    assert "dsv4-long-output-quality-live" in live_ids
    assert "model-family-live-multiturn-soak" in live_ids
    assert all(row["heavy"] for row in live_rows)


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


def test_release_regression_manifest_tracks_model_artifact_detection_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_model_artifact_format_contract.py" in joined
    assert "current-model-artifact-format-contract-20260521.json" in joined
    assert "JANGTQ" in joined
    assert "MXFP4" in joined
    assert "MXFP8" in joined
    assert "MTP" in joined


def test_release_regression_manifest_tracks_parser_parity_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["parser-registry-tool-reasoning-parity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_parser_registry_contract.py" in joined
    assert "current-parser-registry-contract-20260521.json" in joined
    assert "MiniMax" in joined
    assert "reasoning parser" in joined
    assert "tool parser" in joined


def test_release_regression_manifest_tracks_max_output_context_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_max_output_context_contract.py" in joined
    assert "current-max-output-context-contract-20260521.json" in joined
    assert "Server Default Max Output Tokens" in joined
    assert "Chat Max Output Tokens" in joined
    assert "Max Context Tokens" in joined
    assert "--max-tokens" in joined
    assert "--max-prompt-tokens" in joined


def test_release_regression_manifest_tracks_vl_media_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["vl-media-cache-tool-followup"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_vl_media_cache_contract.py" in joined
    assert "current-vl-media-cache-contract-20260521.json" in joined
    assert "VLM media request serialization" in joined
    assert "media cache salting" in joined
    assert "tool follow-up" in joined
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


def test_release_regression_manifest_tracks_native_mtp_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["native-mtp-d3-effect-policy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_native_mtp_contract.py" in joined
    assert "current-native-mtp-contract-20260521.json" in joined
    assert "Native MTP D3" in joined
    assert "DSV4" in joined
    assert "live equivalence/speed" in joined
