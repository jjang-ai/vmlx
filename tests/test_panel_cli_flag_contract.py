from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_NON_SERVE_DIAGNOSTIC_FLAGS = {
    # sessions.ts still parses old log lines from `python -m vmlx_engine.server
    # --model <path>` so the UI can recover a model path from historical output.
    # It is not emitted by the current `vmlx_engine.cli serve` launcher.
    "--model",
}


def _serve_cli_flags() -> set[str]:
    source = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")
    serve_block = source[
        source.index('serve_parser = subparsers.add_parser("serve"'):
        source.index('bench_parser = subparsers.add_parser("bench"')
    ]
    return set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', serve_block))


def _panel_source_flags() -> dict[str, set[str]]:
    rels = (
        "panel/src/main/sessions.ts",
        "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
        "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
        "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
        "panel/src/renderer/src/components/sessions/CreateSession.tsx",
    )
    return {
        rel: set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', (ROOT / rel).read_text(encoding="utf-8")))
        for rel in rels
    }


def _constant_flag_set(rel: str, const_name: str, _seen: set[str] | None = None) -> set[str]:
    _seen = set(_seen or ())
    if const_name in _seen:
        return set()
    _seen.add(const_name)
    source = (ROOT / rel).read_text(encoding="utf-8")
    start = source.index(f"const {const_name}")
    end = source.index("])", start)
    block = source[start:end]
    flags = set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', block))
    for spread in re.findall(r"\.\.\.([A-Z0-9_]+)", block):
        flags.update(_constant_flag_set(rel, spread, _seen))
    return flags


def _serve_cli_value_flags() -> set[str]:
    """Approximate argparse serve flags that consume a following value.

    The additional-args sanitizer must skip the next token for these flags when
    it strips stale DSV4 launch overrides. Otherwise a blocked flag such as
    ``--max-tokens 32768`` would leave ``32768`` behind as a positional-looking
    argument and break the launch command.
    """

    source = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")
    serve_block = source[
        source.index('serve_parser = subparsers.add_parser("serve"'):
        source.index('bench_parser = subparsers.add_parser("bench"')
    ]
    value_flags: set[str] = set()
    for match in re.finditer(r"serve_parser\.add_argument\((?P<body>.*?)\n    \)", serve_block, re.S):
        body = match.group("body")
        flag_match = re.search(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', body)
        if not flag_match:
            continue
        flag = flag_match.group(1)
        if "action=\"store_true\"" in body or "action='store_true'" in body:
            continue
        if "action=\"store_false\"" in body or "action='store_false'" in body:
            continue
        value_flags.add(flag)
    return value_flags


def test_panel_serve_flags_are_registered_engine_cli_flags() -> None:
    """The app must not emit or preview serve flags argparse cannot accept."""

    serve_flags = _serve_cli_flags()
    panel_flags = _panel_source_flags()
    missing = {}
    for rel, flags in panel_flags.items():
        unsupported = sorted(flags - serve_flags - LEGACY_NON_SERVE_DIAGNOSTIC_FLAGS)
        if unsupported:
            missing[rel] = unsupported
    assert missing == {}


def test_dsv4_advanced_args_blocklist_names_real_serve_flags() -> None:
    """DSV4 stale-arg sanitizer should track real serve flags, not dead names."""

    source = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(encoding="utf-8")
    start = source.index("const DSV4_ADDITIONAL_ARG_BLOCKLIST")
    end = source.index("])", start)
    block = source[start:end]
    blocked = set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', block))
    assert blocked, "DSV4 blocklist unexpectedly empty"
    assert sorted(blocked - _serve_cli_flags()) == []


def test_runtime_and_preview_additional_arg_filters_share_blocklists() -> None:
    """Runtime launch and UI command preview must strip the same stale flags."""

    names = (
        "ADDITIONAL_ARG_VALUE_FLAGS",
        "IMAGE_ADDITIONAL_ARG_BLOCKLIST",
        "TEXT_ADDITIONAL_ARG_BLOCKLIST",
        "DSV4_ADDITIONAL_ARG_BLOCKLIST",
    )
    runtime_rel = "panel/src/main/sessions.ts"
    preview_rel = "panel/src/renderer/src/components/sessions/SessionSettings.tsx"
    for name in names:
        assert _constant_flag_set(preview_rel, name) == _constant_flag_set(runtime_rel, name)


def test_dsv4_stale_value_flags_strip_their_values_in_preview_and_runtime() -> None:
    """Blocked DSV4 Advanced Args must not leave orphan values behind."""

    dsv4_blocked = _constant_flag_set(
        "panel/src/main/sessions.ts",
        "DSV4_ADDITIONAL_ARG_BLOCKLIST",
    )
    serve_value_flags = _serve_cli_value_flags()
    blocked_value_flags = dsv4_blocked & serve_value_flags
    expected_value_skip_flags = _constant_flag_set(
        "panel/src/main/sessions.ts",
        "ADDITIONAL_ARG_VALUE_FLAGS",
    )

    assert sorted(blocked_value_flags - expected_value_skip_flags) == []


def test_text_stale_value_flags_strip_their_values_in_preview_and_runtime() -> None:
    """Non-DSV4 sessions must not let stale Advanced Args override UI/autodetect."""

    text_blocked = _constant_flag_set(
        "panel/src/main/sessions.ts",
        "TEXT_ADDITIONAL_ARG_BLOCKLIST",
    )
    serve_value_flags = _serve_cli_value_flags()
    blocked_value_flags = text_blocked & serve_value_flags
    expected_value_skip_flags = _constant_flag_set(
        "panel/src/main/sessions.ts",
        "ADDITIONAL_ARG_VALUE_FLAGS",
    )

    assert sorted(blocked_value_flags - expected_value_skip_flags) == []
    for flag in (
        "--max-tokens",
        "--max-prompt-tokens",
        "--default-enable-thinking",
        "--default-repetition-penalty",
        "--reasoning-parser",
        "--tool-call-parser",
        "--enable-auto-tool-choice",
        "--host",
        "--port",
        "--timeout",
        "--rate-limit",
        "--log-level",
        "--allowed-origins",
        "--served-model-name",
        "--chat-template",
        "--chat-template-kwargs",
        "--mcp-config",
        "--mcp-enabled-servers",
        "--mcp-disabled-tools",
        "--api-key",
        "--uds",
        "--wake-timeout",
        "--inference-endpoints",
        "--native-mtp-depth",
        "--native-mtp-sampling-policy",
        "--use-paged-cache",
        "--kv-cache-quantization",
    ):
        assert flag in text_blocked


def test_panel_cli_flag_contract_covers_dsv4_cache_and_output_boundaries() -> None:
    """Pin the risky rows Eric called out so this file stays purpose-built."""

    sessions = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(encoding="utf-8")
    assert "--dsv4-enable-prefix-cache" in sessions
    assert "--use-paged-cache" in sessions
    assert "--enable-block-disk-cache" in sessions
    assert "--kv-cache-quantization" in sessions
    assert "--max-tokens" in sessions
    assert "--max-prompt-tokens" in sessions
    assert "--native-mtp-depth" in sessions
    assert "--native-mtp-sampling-policy" in sessions


def test_serve_cli_exposes_image_lora_flags_and_startup_passes_them_through() -> None:
    """vmlx serve must make image LoRA lower-stack support reachable."""

    cli_source = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")
    serve_flags = _serve_cli_flags()

    assert "--lora-paths" in serve_flags
    assert "--lora-scales" in serve_flags
    assert "server._image_lora_paths = _split_cli_values(getattr(args, \"lora_paths\", None))" in cli_source
    assert "server._image_lora_scales = _parse_lora_scales(" in cli_source
    assert "lora_paths=server._image_lora_paths" in cli_source
    assert "lora_scales=server._image_lora_scales" in cli_source


def test_command_preview_uses_runtime_numeric_sanitizers_for_advanced_modes() -> None:
    """The visible CLI preview must not round/surface values runtime floors away."""

    preview = (
        ROOT / "panel" / "src" / "renderer" / "src" / "components" / "sessions" / "SessionSettings.tsx"
    ).read_text(encoding="utf-8")
    sessions = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(encoding="utf-8")

    preview_native = preview[
        preview.index("// Native in-model MTP mirrors sessions.ts"):
        preview.index("// Generation defaults", preview.index("// Native in-model MTP mirrors sessions.ts"))
    ]
    runtime_native = sessions[
        sessions.index("// Native in-model MTP."):
        sessions.index("// Generation defaults", sessions.index("// Native in-model MTP."))
    ]
    preview_advanced = preview[
        preview.index("// Smelt mode"):
        preview.index("// Generation defaults", preview.index("// Smelt mode"))
    ]
    runtime_advanced = sessions[
        sessions.index("// Smelt mode"):
        sessions.index("// Generation defaults", sessions.index("// Smelt mode"))
    ]

    assert "finitePositiveInteger(configuredDepth)" in runtime_native
    assert "finitePositiveInteger(configuredDepth)" in preview_native
    assert "Math.round(Number(configuredDepth" not in preview_native
    for expression in (
        "finitePositiveInteger((config as any).smeltExperts)",
        "finitePositiveInteger((config as any).flashMoeSlotBank)",
        "finitePositiveInteger((config as any).flashMoeIoSplit)",
        "finitePositiveInteger(config.numDraftTokens)",
    ):
        assert expression in runtime_advanced
        assert expression in preview_advanced


def test_command_preview_uses_runtime_numeric_sanitizers_for_core_flags() -> None:
    """Core launch knobs must not show CLI values runtime would omit or floor."""

    preview = (
        ROOT / "panel" / "src" / "renderer" / "src" / "components" / "sessions" / "SessionSettings.tsx"
    ).read_text(encoding="utf-8")
    sessions = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(encoding="utf-8")

    preview_body = preview[
        preview.index("function buildCommandPreview("):
        preview.index("return parts.join", preview.index("function buildCommandPreview("))
    ]
    runtime_start = sessions.index("buildArgs(config: ServerConfig): string[]")
    runtime_body = sessions[
        runtime_start:
        sessions.index("findEnginePath()", runtime_start)
    ]

    for expression in (
        "finitePositiveInteger(config.rateLimit)",
        "finitePositiveInteger(config.maxNumSeqs)",
        "finitePositiveInteger(config.prefillBatchSize)",
        "finitePositiveInteger(config.prefillStepSize)",
        "finitePositiveInteger(config.completionBatchSize)",
        "finitePositiveInteger(config.prefixCacheSize)",
        "finitePositiveInteger(config.prefixCacheMaxBytes)",
        "finitePositiveInteger(config.cacheMemoryMb)",
        "finitePositiveNumber(config.cacheMemoryPercent)",
        "finitePositiveNumber(config.cacheTtlMinutes)",
        "finitePositiveInteger(effectivePagedCacheBlockSize)",
        "finitePositiveInteger(config.maxCacheBlocks)",
        "finitePositiveInteger(config.kvCacheGroupSize)",
        "finiteNonNegativeNumber(config.diskCacheMaxGb)",
        "finiteNonNegativeNumber(config.blockDiskCacheMaxGb)",
    ):
        assert expression in runtime_body
        assert expression in preview_body


def test_paged_cache_capacity_is_visible_and_not_memory_mb_driven() -> None:
    """Paged cache UI/CLI must surface the real token capacity knob."""

    panel = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(encoding="utf-8")
    cli = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")

    assert "pagedCacheCapacityLogLine" in panel
    assert "tokens/block x" in panel
    assert "--cache-memory-mb/--cache-memory-percent are ignored while paged cache is active" in panel
    assert "Max Cache Blocks" in panel
    assert "capacity={capacity} tokens" in cli
    assert "--cache-memory-mb ignored for paged cache" in cli


def test_live_metal_headroom_ui_proof_checks_paged_capacity_log() -> None:
    source = (ROOT / "panel" / "scripts" / "live-metal-headroom-ui-proof.mjs").read_text(
        encoding="utf-8"
    )

    assert "Paged cache capacity: 64 tokens/block x 64 blocks = 4096 tokens." in source
    assert "--cache-memory-mb/--cache-memory-percent are ignored while paged cache is active" in source
    assert "window.api.sessions.create" in source
    assert "window.api.sessions.getLogs" in source


def test_live_metal_headroom_ui_proof_checks_minimax_m3_native_cache_controls() -> None:
    source = (ROOT / "panel" / "scripts" / "live-metal-headroom-ui-proof.mjs").read_text(
        encoding="utf-8"
    )

    assert "collectMinimaxM3SettingsUi" in source
    assert "minimax_m3_vl" in source
    assert "MiniMax-M3 uses paged-off SSD prefix cache with native MSA idx_keys" in source
    assert "MiniMax-M3 keeps generic KV q4/q8 disabled" in source
    assert "generic stored-KV codecs cannot preserve that cache format" in source
    assert "Stored Cache Quantization" in source


def test_live_metal_headroom_chat_ui_proof_checks_visible_safety_block() -> None:
    source = (
        ROOT / "panel" / "scripts" / "live-metal-headroom-chat-ui-proof.mjs"
    ).read_text(encoding="utf-8")

    assert "Requested max output tokens exceed projected safe Metal headroom" in source
    assert "Generation blocked" in source
    assert "window.api.sessions.createRemote" in source
    assert "window.api.chat.sendMessage" in source
    assert "window.api.chat.isStreaming" in source


def test_metal_oom_startup_errors_surface_wired_limit_guidance() -> None:
    shared = (ROOT / "panel" / "src" / "shared" / "metalWiredLimit.ts").read_text(
        encoding="utf-8"
    )
    sessions = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(
        encoding="utf-8"
    )
    form = (
        ROOT
        / "panel"
        / "src"
        / "renderer"
        / "src"
        / "components"
        / "sessions"
        / "SessionConfigForm.tsx"
    ).read_text(encoding="utf-8")

    assert "sudo sysctl iogpu.wired_limit_mb=120000" in shared
    assert "115000-120000 MB" in shared
    assert "Do not set it equal to physical RAM" in shared
    assert "SIGKILL" in shared
    assert "kIOGPUCommandBufferCallbackErrorOutOfMemory" in shared
    assert "Command buffer execution failed" in shared
    assert "Insufficient Memory" in shared
    assert "appendMetalWiredLimitGuidance(reason)" in sessions
    assert "Process exited before becoming ready" in sessions
    assert "metalWiredLimitHelpText" in form


def test_jang_loader_wired_limit_keeps_physical_ram_headroom() -> None:
    source = (ROOT / "vmlx_engine" / "utils" / "jang_loader.py").read_text(
        encoding="utf-8"
    )

    assert "VMLINUX_METAL_WIRED_RESERVE_FRACTION" in source
    assert "VMLX_METAL_WIRED_RESERVE_FRACTION" in source
    assert "VMLINUX_METAL_WIRED_RESERVE_GB" in source
    assert "ram_capped_target" in source
    assert "capped to %.0f GB to leave %.0f GB system/Metal headroom" in source


def test_jang_loader_wired_limit_caps_rich_128gb_case(monkeypatch) -> None:
    """A 128GiB Mac with sysctl at 128000MiB must not pass the full cap to MLX."""

    from vmlx_engine.utils import jang_loader

    class FakeStat:
        st_size = 113_000_000_000

    class FakeWeight:
        def stat(self):
            return FakeStat()

    class FakeMx:
        def __init__(self) -> None:
            self.targets: list[int] = []

        def set_wired_limit(self, target: int) -> None:
            self.targets.append(target)

    fake_mx = FakeMx()
    total_ram = 128 * 1024**3
    page_size = 4096
    phys_pages = total_ram // page_size

    monkeypatch.delenv("VMLX_METAL_WIRED_RESERVE_FRACTION", raising=False)
    monkeypatch.delenv("VMLINUX_METAL_WIRED_RESERVE_FRACTION", raising=False)
    monkeypatch.delenv("VMLINUX_METAL_WIRED_RESERVE_PCT", raising=False)
    monkeypatch.delenv("VMLX_METAL_WIRED_RESERVE_GB", raising=False)
    monkeypatch.delenv("VMLINUX_METAL_WIRED_RESERVE_GB", raising=False)
    monkeypatch.setattr(jang_loader, "mx", fake_mx)
    monkeypatch.setattr(
        jang_loader,
        "get_effective_metal_working_set_bytes",
        lambda _mx: (0, 134_000_000_000),
    )
    monkeypatch.setattr(
        jang_loader.os,
        "sysconf",
        lambda name: page_size if name == "SC_PAGE_SIZE" else phys_pages,
    )

    jang_loader._set_wired_limit_for_model([FakeWeight()])

    assert fake_mx.targets, "expected MLX wired limit to be set"
    expected_cap = total_ram - int(total_ram * 0.16)
    assert fake_mx.targets[0] <= expected_cap
    assert fake_mx.targets[0] < 120_000_000_000
    assert fake_mx.targets[0] > FakeStat.st_size


def test_cli_minimax_m3_vl_autoroutes_to_text_msa_runtime() -> None:
    """Direct CLI must match the panel's MiniMax-M3 text-routed VL path."""

    source = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")
    utils = (ROOT / "vmlx_engine" / "api" / "utils.py").read_text(encoding="utf-8")

    assert '_m3_mt == "minimax_m3_vl"' in source
    assert 'os.environ["VMLX_M3_VL"] = "1"' in source
    assert "ignoring --is-mllm" in source
    assert source.index('os.environ["VMLX_M3_VL"] = "1"') < source.index(
        "MiniMax-M3 AUTODETECTED"
    )
    assert "MiniMax-M3 overrides force_mllm" in utils
    assert "mlx_vlm has no minimax_m3_vl runtime" in utils


def test_live_clean_start_proof_checks_mm3_actual_launch_argv() -> None:
    """The UI clean-start proof must validate the spawned MM3 CLI, not config alone."""

    source = (
        ROOT / "panel" / "scripts" / "live-clean-start-autodetect-proof.mjs"
    ).read_text(encoding="utf-8")

    assert "extractLaunchCommand" in source
    assert "result.launchCommand" in source
    assert "MM3 launch argv missing --enable-disk-cache" in source
    assert "MM3 launch argv incorrectly disabled prefix cache" in source
    assert "MM3 launch argv incorrectly enabled generic paged KV cache" in source
    assert "MM3 launch argv incorrectly enabled generic block disk cache" in source
    assert "MM3 launch argv incorrectly passed generic --kv-cache-quantization" in source
    assert "MM3 launch argv incorrectly passed --enable-jit" in source
    assert "MM3 launch argv incorrectly passed generic --is-mllm" in source
    assert "MM3 launch argv missing --tool-call-parser minimax_m3" in source
    assert "MM3 launch argv missing --reasoning-parser minimax_m3" in source
    assert "MM3 launch argv missing --enable-auto-tool-choice" in source
    assert "MM3 launch argv missing --timeout 900 long-generation default" in source
    assert (
        "MM3 launch argv incorrectly forced --max-tokens; default must remain model-owned"
        in source
    )


def test_mm3_and_gemma_live_stress_harnesses_cover_api_auth_matrix() -> None:
    """Release live proof must exercise missing/wrong/right bearer auth."""

    for rel in (
        "panel/scripts/live-mm3-stress-proof.mjs",
        "panel/scripts/live-gemma4-media-stress-proof.mjs",
    ):
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert "apiKey:" in source, rel
        assert "runApiAuthMatrix" in source, rel
        assert "Authorization: `Bearer ${apiKey}`" in source, rel
        assert "auth missing request did not return 401" in source, rel
        assert "auth wrong request did not return 401" in source, rel
        assert "auth correct request did not return 200" in source, rel
        assert "gatewayAuth" in source, rel
        assert "gateway auth missing request did not return 401" in source, rel
        assert "gateway auth wrong request did not return 401" in source, rel
        assert "gateway auth correct request did not return 200" in source, rel


def test_mm3_and_gemma_live_stress_harnesses_gate_actual_launch_argv() -> None:
    """Stress proof must validate the UI-spawned engine argv, not just settings state."""

    mm3 = (ROOT / "panel" / "scripts" / "live-mm3-stress-proof.mjs").read_text(
        encoding="utf-8"
    )
    gemma = (
        ROOT / "panel" / "scripts" / "live-gemma4-media-stress-proof.mjs"
    ).read_text(encoding="utf-8")

    for rel, source in (
        ("live-mm3-stress-proof.mjs", mm3),
        ("live-gemma4-media-stress-proof.mjs", gemma),
    ):
        assert "extractLaunchCommand" in source, rel
        assert "result.launchCommand" in source, rel
        assert "stress launch command missing from UI logs" in source, rel
        assert "--enable-disk-cache" in source, rel
        assert "--disable-prefix-cache" in source, rel
        assert "--use-paged-cache" in source, rel
        assert "--tool-call-parser" in source, rel
        assert "--reasoning-parser" in source, rel
        assert "--enable-auto-tool-choice" in source, rel

    assert "MM3 stress launch argv missing --enable-disk-cache" in mm3
    assert "MM3 stress launch argv incorrectly disabled prefix cache" in mm3
    assert "MM3 stress launch argv incorrectly enabled generic paged KV cache" in mm3
    assert "MM3 stress launch argv incorrectly enabled generic block disk cache" in mm3
    assert "MM3 stress launch argv incorrectly passed generic --kv-cache-quantization" in mm3
    assert "MM3 stress launch argv incorrectly passed --enable-jit" in mm3
    assert "MM3 stress launch argv incorrectly passed generic --is-mllm" in mm3
    assert "MM3 stress launch argv missing --tool-call-parser minimax_m3" in mm3
    assert "MM3 stress launch argv missing --reasoning-parser minimax_m3" in mm3
    assert "MM3 stress launch argv missing --enable-auto-tool-choice" in mm3
    assert "MM3 stress launch argv missing --timeout 900 long-generation default" in mm3
    assert (
        "MM3 stress launch argv incorrectly forced --max-tokens; default must remain model-owned"
        in mm3
    )

    assert "Gemma4 stress launch argv missing --enable-disk-cache" in gemma
    assert "Gemma4 stress launch argv incorrectly disabled prefix cache" in gemma
    assert "Gemma4 stress launch argv enabled generic paged KV despite usePagedCache=false" in gemma
    assert "Gemma4 stress launch argv unexpectedly enabled block disk cache for default row" in gemma
    assert "Gemma4 stress launch argv passed explicit --kv-cache-quantization despite auto defaults" in gemma
    assert "Gemma4 stress launch argv missing --tool-call-parser gemma4" in gemma
    assert "Gemma4 stress launch argv missing --reasoning-parser gemma4" in gemma
    assert "Gemma4 stress launch argv missing --enable-auto-tool-choice" in gemma
    assert "cfg.usePagedCache === false" in gemma
    assert "cfg.kvCacheQuantization === 'auto'" in gemma
