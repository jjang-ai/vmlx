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


# Constants that used to be copied into every call site and now have ONE owner
# under src/shared/. A consumer no longer names the constant at all — it calls
# the shared helper that consults it — so the check below is that the consumer
# gets the rule from the owner rather than keeping a private copy.
SHARED_CONSTANT_OWNERS = {
    "ADDITIONAL_ARG_VALUE_FLAGS": "panel/src/shared/launchArgValues.ts",
}


def _resolve_constant_owner(rel: str, const_name: str) -> str:
    """Return the file that DEFINES ``const_name`` for ``rel``.

    This test was written when each of these constants was duplicated per file,
    and it compared the copies. That is the right check for the ones still
    declared locally, but reading only ``rel`` raises ValueError once a constant
    moves to a shared owner — a green-to-red flip caused by the duplication
    being REMOVED, which is backwards.

    So: a local declaration is still compared as before; a shared one resolves
    to its owner after asserting ``rel`` actually imports from that module. If a
    private copy is ever reintroduced, the local branch takes over and the two
    sides get compared against each other again.
    """
    source = (ROOT / rel).read_text(encoding="utf-8")
    if f"const {const_name}" in source:
        return rel
    owner = SHARED_CONSTANT_OWNERS.get(const_name)
    assert owner, f"{rel} does not define {const_name} and it has no shared owner"
    module_stem = Path(owner).stem
    assert re.search(rf"from\s*['\"][^'\"]*/{re.escape(module_stem)}['\"]", source), (
        f"{rel} neither declares {const_name} nor imports from {owner} — "
        "it must not carry a private copy of this rule"
    )
    return owner


def _constant_flag_set(rel: str, const_name: str, _seen: set[str] | None = None) -> set[str]:
    _seen = set(_seen or ())
    if const_name in _seen:
        return set()
    _seen.add(const_name)
    rel = _resolve_constant_owner(rel, const_name)
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
        "--native-mtp-depth-policy",
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
    assert "--native-mtp-depth-policy" in sessions
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
    mtp_helper = (ROOT / "panel" / "src" / "shared" / "nativeMtpLaunchArgs.ts").read_text(encoding="utf-8")

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

    assert "buildNativeMtpLaunchArgs" in runtime_native
    assert "buildNativeMtpLaunchArgs" in preview_native
    # Adaptive mode emits NO explicit depth (an explicit --native-mtp-depth
    # becomes the engine's VMLINUX_NATIVE_MTP_DEPTH override, pinning the
    # start depth and bypassing tuning sidecars and the session profile);
    # only a user Fixed override sends a sanitized depth.
    assert "input.depthOverride !== true" in mtp_helper
    assert "input.configuredDepth" in mtp_helper
    assert "resolveFixedNativeMtpDepth(input.configuredDepth, input.detectedDepth)" in mtp_helper
    assert "finitePositiveInteger(configuredDepth)" in mtp_helper
    assert "'--native-mtp-depth-policy', 'adaptive'" in mtp_helper
    assert "'fixed'" in mtp_helper
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
    cache_helper = (ROOT / "panel" / "src" / "shared" / "cacheLaunchArgs.ts").read_text(encoding="utf-8")

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
        # kvCacheGroupSize left this list when generic --kv-cache-quantization
        # emission was retired (stored/live cache is exact everywhere); the
        # panel no longer sends a group size at all.
    ):
        assert expression in runtime_body
        assert expression in preview_body

    assert "buildCacheLaunchArgs" in runtime_body
    assert "buildCacheLaunchArgs" in preview_body
    for expression in (
        "finitePositiveInteger(input.prefixCacheSize)",
        "finitePositiveInteger(input.prefixCacheMaxBytes)",
        "finitePositiveInteger(input.cacheMemoryMb)",
        "finitePositiveNumber(input.cacheMemoryPercent)",
        "finitePositiveNumber(input.cacheTtlMinutes)",
        "finitePositiveInteger(input.effectivePagedCacheBlockSize)",
        "finitePositiveInteger(input.maxCacheBlocks)",
        "finiteNonNegativeNumber(input.diskCacheMaxGb)",
        "finiteNonNegativeNumber(input.blockDiskCacheMaxGb)",
    ):
        assert expression in cache_helper


def test_paged_cache_capacity_is_visible_and_not_memory_mb_driven() -> None:
    """Paged cache UI/CLI must surface the real token capacity knob."""

    panel = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(encoding="utf-8")
    cli = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")

    assert "pagedCacheCapacityLogLine" in panel
    assert "tokens/block x" in panel
    # In-memory paged cache is now locked off product-wide. In the remaining
    # RAM-enabled path these controls bound unified-memory eviction, while the
    # SSD-only default reports token capacity separately.
    assert "set the Apple unified-memory ceiling that evicts free blocks" in panel
    assert "evicts free blocks" in panel
    assert "Max Cache Blocks" in panel
    assert "capacity={capacity} tokens" in cli
    assert "set the L1 RAM byte " in cli


def test_live_metal_headroom_ui_proof_checks_paged_capacity_log() -> None:
    source = (ROOT / "panel" / "scripts" / "live-metal-headroom-ui-proof.mjs").read_text(
        encoding="utf-8"
    )

    assert "Paged cache capacity: 64 tokens/block x 64 blocks = 4096 tokens." in source
    assert "--cache-memory-mb/--cache-memory-percent set the L1 RAM byte ceiling that evicts free blocks" in source
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
    # The form shows this guidance through t() so it localizes (the shared module
    # above still owns the command and the detection strings, because the MAIN
    # process appends them to error messages that are not localized). Assert the
    # form is wired to the key AND that en.json carries the guidance, so the note
    # cannot silently lose either half.
    assert "metalWiredLimitCommand" in form
    assert "t('sessions.config.metalWiredLimitHelp'" in form
    catalog = (
        ROOT / "panel" / "src" / "renderer" / "src" / "i18n" / "locales" / "en.json"
    ).read_text(encoding="utf-8")
    assert "Metal wired-memory limit" in catalog
    assert "115000-120000 MB" in catalog


def test_large_model_low_memory_preflight_never_blocks_engine_spawn() -> None:
    """2026-08-17: this test used to REQUIRE the block. It now forbids it.

    The refusal it pinned shipped in v1.6.28-v1.6.33 and made dots3-note
    (101.6 GB, estimated 103.6 GB because an unrecognised model_type falls back
    to a 1.0x residency ratio) unloadable for EVERY 128 GB user, along with any
    other bundle over ~76 GB the ratio table did not recognise. Eric never
    asked for a RAM limit: "U CANNOT MAKE UP SOME FUCKING RAM SAFETY RULE".

    The preflight may advise. It may not refuse. Loading big models on unified
    memory is the product.
    """

    shared = (ROOT / "panel" / "src" / "shared" / "metalWiredLimit.ts").read_text(
        encoding="utf-8"
    )
    sessions = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(
        encoding="utf-8"
    )

    assert "classifyLargeModelMemoryPreflight" in shared
    assert "effectivelyNoFreeRam" in shared
    # `block` was deleted from the union so no call site can reintroduce it.
    assert "action: 'block'" not in shared
    assert "Refusing to start" not in shared
    assert "classifyLargeModelMemoryPreflight" in sessions
    assert "memoryPreflight.action === 'block'" not in sessions
    assert "Refusing to start this model" not in sessions
    assert "this.emit('session:error', { sessionId, error: memoryPreflight.message })" not in sessions
    assert "throw new Error(memoryPreflight.message)" not in sessions
    # An override env is not consent -- it is a wall with a secret door.
    assert "unsafeModelLaunchOverrideHint()" not in sessions
    # It must still SAY something: advice is the whole remaining job.
    assert "unsafeModelLaunchReason(" in sessions

    preflight_index = sessions.index("const memoryPreflight = classifyLargeModelMemoryPreflight")
    owned_port_index = sessions.index("await this.ensureOwnedSessionPortAvailable(session)")
    bundled_spawn_index = sessions.index("proc = spawn(engineResult.pythonPath")
    system_spawn_index = sessions.index("proc = spawn(engineResult.binaryPath")
    assert preflight_index < owned_port_index < bundled_spawn_index
    assert preflight_index < owned_port_index < system_spawn_index
    assert "await this.killByPort(session.port)" not in sessions
    assert "terminateDetectedEngineForSession(session)" in sessions


def test_jang_loader_wired_limit_honors_sysctl_by_default() -> None:
    """Per Eric directive 2026-06-27: default = honor OS sysctl iogpu.wired_limit_mb
    fully, no artificial reserve cap. Reserve is opt-in via VMLX_METAL_WIRED_RESERVE_GB."""

    source = (ROOT / "vmlx_engine" / "utils" / "jang_loader.py").read_text(
        encoding="utf-8"
    )

    # Env-var reserve remains available as opt-in
    assert "VMLX_METAL_WIRED_RESERVE_GB" in source
    assert "VMLINUX_METAL_WIRED_RESERVE_GB" in source
    # Legacy fraction env still honored (deprecated but not removed)
    assert "VMLX_METAL_WIRED_RESERVE_FRACTION" in source
    assert "VMLINUX_METAL_WIRED_RESERVE_FRACTION" in source
    # Ram cap logic conditionally applied only when reserve_gb > 0 or fraction env set
    assert "ram_capped_target" in source
    # New wording after Eric directive removed the mandatory reserve
    assert "opt-in reserve" in source
    assert "Authoritative upper bound: the OS sysctl iogpu.wired_limit_mb" in source


def test_jang_loader_wired_limit_default_honors_full_sysctl(monkeypatch) -> None:
    """Default (no reserve env set): wired limit = full OS max_ws value from sysctl,
    not clipped by a hidden reserve."""

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
    os_max_ws = 134_000_000_000  # what sysctl iogpu.wired_limit_mb reports via MLX

    monkeypatch.delenv("VMLX_METAL_WIRED_RESERVE_FRACTION", raising=False)
    monkeypatch.delenv("VMLINUX_METAL_WIRED_RESERVE_FRACTION", raising=False)
    monkeypatch.delenv("VMLINUX_METAL_WIRED_RESERVE_PCT", raising=False)
    monkeypatch.delenv("VMLX_METAL_WIRED_RESERVE_GB", raising=False)
    monkeypatch.delenv("VMLINUX_METAL_WIRED_RESERVE_GB", raising=False)
    monkeypatch.setattr(jang_loader, "mx", fake_mx)
    monkeypatch.setattr(
        jang_loader,
        "get_effective_metal_working_set_bytes",
        lambda _mx: (0, os_max_ws),
    )
    monkeypatch.setattr(
        jang_loader.os,
        "sysconf",
        lambda name: page_size if name == "SC_PAGE_SIZE" else phys_pages,
    )

    jang_loader._set_wired_limit_for_model([FakeWeight()])

    assert fake_mx.targets, "expected MLX wired limit to be set"
    # Default behavior: no reserve → target = min(model+headroom, os_max_ws)
    # model=113 GB + max(16 GB, 30% * 113 GB) = 113 + 33.9 = 146.9 GB target
    # OS cap = 134 GB → clipped to 134 GB (sysctl-honoring, no additional reserve)
    assert fake_mx.targets[0] == os_max_ws, (
        f"default should honor full OS sysctl {os_max_ws}, got {fake_mx.targets[0]}"
    )
    assert fake_mx.targets[0] > FakeStat.st_size


def test_jang_loader_wired_limit_opt_in_reserve_still_works(monkeypatch) -> None:
    """When user sets VMLX_METAL_WIRED_RESERVE_GB explicitly, reserve applies."""

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
    monkeypatch.delenv("VMLINUX_METAL_WIRED_RESERVE_GB", raising=False)
    # OPT IN to 16 GB reserve (the previously-mandatory floor)
    monkeypatch.setenv("VMLX_METAL_WIRED_RESERVE_GB", "16")
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
    expected_cap = total_ram - 16 * 1024**3
    assert fake_mx.targets[0] <= expected_cap
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

def test_native_mtp_auto_pins_greedy_startup_defaults_and_deterministic_is_hard() -> None:
    """Enabled native MTP pins greedy STARTUP DEFAULTS for every mode.

    2026-09-02: 7972082c had switched Auto to compatible-only, which kept the
    bundle's sampled temperature as the default — so 27B/Flash-Next app
    sessions silently ran stochastic MTP at nonzero temperature. Restored
    contract: Auto launches deterministic-defaults (greedy defaults, explicit
    request kwargs still win) and Deterministic launches hard greedy-only. Off
    disables the runtime outright.
    """

    sessions = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(
        encoding="utf-8"
    )
    shared = (
        ROOT / "panel" / "src" / "shared" / "nativeMtpLaunchArgs.ts"
    ).read_text(encoding="utf-8")

    assert "buildNativeMtpLaunchArgs" in sessions
    assert "--native-mtp-sampling-policy" in shared
    assert "const mode = resolveNativeMtpMode(input)" in shared
    assert (
        "mode === 'deterministic' ? 'greedy-only' : 'deterministic-defaults'"
        in shared
    )
    assert "'greedy-only'" in shared
    assert "'deterministic-defaults'" in shared
    # Auto must NOT emit compatible-only any more (that was the regression).
    assert "'compatible-only'" not in shared
    # off must still disable the runtime outright
    assert "return ['--disable-native-mtp']" in shared

def test_native_mtp_depth_derives_from_the_bundle_not_a_hardcoded_three() -> None:
    """2026-08-17: depth fell through to a hardcoded 3 and tanked acceptance.

    Qwen3.8-27B-JANG_4D-CRACK declares `recommended_num_drafts: 1` and
    `upstream_num_speculative_tokens: 2` in jang_config, and its
    vmlx_mtp_tuning.json says `best_depth: 1` -- yet the running engine used
    depth 3 (a stale session nativeMtpDepthOverride). Asking a head trained for
    1-2 tokens to draft 3 ahead measured 58.6% acceptance, under the ~0.68
    breakeven, so MTP paid draft+verify and then fell back to AR.

    Derivation order must be: measured tuning > what the bundle declares for
    itself > layer count > 1. The final fallback must be 1, never 3: an
    uncharacterised head has to draft conservatively because an over-deep
    draft is pure loss.
    """

    registry = (ROOT / "panel" / "src" / "main" / "model-config-registry.ts").read_text(
        encoding="utf-8"
    )
    sessions = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(
        encoding="utf-8"
    )
    settings = (
        ROOT / "panel" / "src" / "renderer" / "src" / "components" /
        "sessions" / "SessionSettings.tsx"
    ).read_text(encoding="utf-8")
    shared = (
        ROOT / "panel" / "src" / "shared" / "nativeMtpLaunchArgs.ts"
    ).read_text(encoding="utf-8")

    # the bundle's own recommendation is actually read
    assert "jang_config.mtp.recommended_num_drafts" in registry
    assert "declaredNativeMtpDrafts" in registry

    # upstream_num_speculative_tokens must NEVER be used as a depth source.
    # vLLM's field counts DRAFTS while the kit's D-notation counts
    # drafts+verified, so upstream's "2" is the kit's forbidden D3 -- the exact
    # config the Nemotron sweep measured at 0.48x. The engine documents this
    # trap in vmlx_engine/native_mtp.py and the panel must not diverge.
    # forbid the ACCESS, not the word -- the warning comment names the field
    assert "mtp?.upstream_num_speculative_tokens" not in registry
    assert "mtp.upstream_num_speculative_tokens]" not in registry
    assert "NEVER" in registry  # the trap stays documented at the call site

    # and neither call site may fall back to 3
    assert "?? 3," not in registry
    assert "buildNativeMtpLaunchArgs" in sessions
    assert "buildNativeMtpLaunchArgs" in settings
    assert "resolveFixedNativeMtpDepth(input.configuredDepth, input.detectedDepth)" in shared
    assert "finitePositiveInteger(detectedDepth)" in shared
    assert "|| 1" in shared
    assert "|| 3" not in shared
