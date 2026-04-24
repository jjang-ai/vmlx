import SwiftUI
import AppKit
import vMLXEngine
import vMLXTheme

/// Per-session configuration form. Reads/writes directly to
/// `SettingsStore.session(id)` — every field change routes through
/// `engine.settings.setSession(id, updated)` which debounces the write for
/// 500ms so rapid slider scrubs coalesce into a single SQLite commit.
///
/// Sections are organized into collapsible `DisclosureGroup`s matching the
/// CLI flag buckets documented in `SettingsTypes.swift`:
///   Engine / Cache / Inference defaults / Lifecycle / Server / Advanced / Logging
///
/// Electron parity: `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx`.
struct SessionConfigForm: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale
    let sessionId: UUID

    // Local mirror of the persisted SessionSettings. Updated when the user
    // types / moves a slider, then pushed into SettingsStore via
    // `commit()`. Initialized from the store on first appear.
    @State private var loaded: Bool = false
    @State private var modelPath: URL = URL(fileURLWithPath: "/")
    @State private var s: SessionSettings = SessionSettings(modelPath: URL(fileURLWithPath: "/"))

    // Disclosure group state
    @State private var openEngine      = true
    @State private var openCache       = true
    @State private var openInference   = true
    @State private var openLifecycle   = true
    @State private var openServer      = true
    @State private var openAuth        = false
    @State private var openRemote      = false
    @State private var openAdvanced    = false
    @State private var openLogging     = false

    /// Set to `true` when the user toggles "Allow LAN access" ON — gates
    /// the `.confirmationDialog(...)` at the view root that asks the user
    /// to confirm exposing the HTTP listener to every device on the
    /// Wi-Fi. Flipping the toggle OFF is always safe (local-only) and
    /// bypasses this prompt.
    @State private var pendingLANEnable: Bool = false

    // Global defaults snapshot — shown as placeholder text behind each field
    // so the user sees what value will be inherited if they leave it empty.
    @State private var globalDefaults: GlobalSettings = GlobalSettings()

    // §368 — model-supplied sampling defaults from generation_config.json,
    // if any. Populated on load() and after a successful engine load.
    // Surfaced as an informational caption at the top of Inference
    // defaults so users see "Qwen recommends temp=0.6, top_p=0.95" vs
    // "Gemma recommends temp=1.0". Not a binding — read-only hint.
    @State private var modelDefaults: Engine.ModelGenerationDefaults = .init()

    /// True when this session is configured to proxy to a remote HTTP
    /// endpoint. In remote mode, all the local-only settings (Engine,
    /// Cache, Lifecycle, Advanced) are inert — the local MLX loader
    /// never runs — so we hide those sections entirely instead of
    /// showing settings that silently do nothing. Inference defaults
    /// still apply because we forward them as request parameters to
    /// the remote.
    private var isRemote: Bool {
        (s.remoteURL?.isEmpty == false)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                Text(L10n.SessionConfig.sectionSessionConfig.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)

                modelSection
                divider
                // Remote-endpoint toggle sits first after the model row
                // so users immediately see the fork: local engine vs
                // proxy-to-remote. Below sections flip based on this.
                disclosure("Remote endpoint (proxy mode)", isOn: $openRemote) { remoteSection }
                divider

                if isRemote {
                    // Remote mode: only Inference + Server (the
                    // listener that proxy clients hit) + Auth + Logging
                    // make sense. Local engine/cache/lifecycle/advanced
                    // don't run, so suppress them rather than show
                    // dead controls.
                    disclosure("Inference defaults", isOn: $openInference) { inferenceSection }
                    divider
                    disclosure("Server (local listener)", isOn: $openServer) { serverSection }
                    divider
                    disclosure("Auth", isOn: $openAuth) { authSection }
                    divider
                    disclosure("Logging", isOn: $openLogging) { loggingSection }
                } else {
                    // Local engine mode: full stack.
                    disclosure("Engine", isOn: $openEngine, loadTimeOnly: true) { engineSection }
                    divider
                    disclosure("Cache", isOn: $openCache, loadTimeOnly: true) { cacheSection }
                    divider
                    disclosure("Inference defaults", isOn: $openInference) { inferenceSection }
                    divider
                    disclosure("Lifecycle", isOn: $openLifecycle) { lifecycleSection }
                    divider
                    disclosure("Server", isOn: $openServer, loadTimeOnly: true) { serverSection }
                    divider
                    disclosure("Auth", isOn: $openAuth) { authSection }
                    divider
                    disclosure("Advanced", isOn: $openAdvanced, loadTimeOnly: true) { advancedSection }
                    divider
                    disclosure("Logging", isOn: $openLogging) { loggingSection }
                }
            }
            .font(Theme.Typography.body)
            .foregroundStyle(Theme.Colors.textHigh)
            .padding(.bottom, Theme.Spacing.lg)
        }
        .frame(maxHeight: 520)
        .task { await load() }
        // LAN-bind confirmation dialog. Exposing the HTTP listener to
        // `0.0.0.0` surfaces the engine to every device on the same
        // network — users should explicitly opt in, not silently toggle
        // a switch and discover later. Canceling keeps the per-session
        // lan flag off (set to false rather than leaving nil so the
        // absence of a flag doesn't defeat the prompt on next toggle).
        .confirmationDialog(
            "Bind server to 0.0.0.0?",
            isPresented: $pendingLANEnable,
            titleVisibility: .visible
        ) {
            Button(L10n.SessionConfig.enableLANAccess.render(appLocale), role: .destructive) {
                s.lan = true
                commit()
            }
            Button(L10n.Common.cancel.render(appLocale), role: .cancel) {
                s.lan = false
                commit()
            }
        } message: {
            Text(L10n.SessionConfig.lanWarning.render(appLocale))
        }
    }

    private var divider: some View {
        Divider()
            .overlay(Theme.Colors.border)
            .padding(.vertical, Theme.Spacing.xs)
    }

    // MARK: - Sections

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Text(L10n.SessionConfig.sectionModel.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            HStack {
                Text(modelPath.lastPathComponent)
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textHigh)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
            }
            .padding(Theme.Spacing.sm)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.sm)
                    .fill(Theme.Colors.surfaceHi)
            )
            // Model alias lives here (not in serverSection) because it's
            // a model-identity concern — what name the model advertises
            // in /v1/models and chat-completion responses — not part of
            // the HTTP listener config. It also applies in remote mode
            // as a local display override.
            textFieldRow("Served as",
                         placeholder: modelPath.lastPathComponent,
                         value: Binding(
                            get: { s.modelAlias ?? "" },
                            set: { s.modelAlias = $0.isEmpty ? nil : $0; commit() }
                         ))
            // §359 — alias is read at session-start time and used to
            // register the engine in the gateway + stamped into
            // OpenAI/Anthropic responses. Changing it mid-session
            // persists but does not re-register; `/v1/chat/completions`
            // with the new alias will 404 until the session restarts.
            HStack(alignment: .top, spacing: 4) {
                Image(systemName: "arrow.triangle.2.circlepath")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Theme.Colors.warning)
                Text(L10n.SessionConfig.restartRequired.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 4)
        }
    }

    @ViewBuilder
    private var engineSection: some View {
        Picker("Engine kind", selection: Binding(
            get: { s.engineKind ?? globalDefaults.engineKind },
            set: { s.engineKind = $0; commit() }
        )) {
            Text("Batched").tag(EngineKindCodable.batched)
            Text("Simple").tag(EngineKindCodable.simple)
        }
        .pickerStyle(.segmented)

        ValidatedField(title: "Max concurrent sequences",
                       value: intBinding(\.maxNumSeqs, default: globalDefaults.maxNumSeqs),
                       range: 1...128, step: 1)
        ValidatedField(title: "Prefill step size",
                       value: intBinding(\.prefillStepSize, default: globalDefaults.prefillStepSize),
                       range: 64...8192, step: 64)
    }

    @ViewBuilder
    private var cacheSection: some View {
        ValidatedField(title: "Max cache blocks",
                       value: intBinding(\.maxCacheBlocks, default: globalDefaults.maxCacheBlocks),
                       range: 16...10000, step: 16)
        // iter-46 fix: pagedCacheBlockSize gates whether short prompts
        // get any paged-cache benefit. Default 64 means sub-64-token
        // chat prompts return nil from fetchPrefix without counting
        // hits or misses. Lowering to 32 halves the short-prompt
        // dead-zone at the cost of a ~2x hash-map size. Exposed here
        // so power users running short chat workloads can tune.
        // Only a CLI flag existed before this iter.
        ValidatedField(title: "Paged block size",
                       value: intBinding(\.pagedCacheBlockSize,
                                         default: globalDefaults.pagedCacheBlockSize),
                       range: 8...256, step: 8)
        // Memory cache (L1.5) — byte-budgeted LRU between paged L1 and disk L2.
        // Sub-controls (% and TTL) are gated on the parent toggle so the
        // user doesn't think they're tweaking something live when the
        // feature is off. Audit 2026-04-15 finding #3.
        toggleRow("Memory cache (L1.5)",
                  boolBinding(\.enableMemoryCache,
                              default: globalDefaults.enableMemoryCache))
        if (s.enableMemoryCache ?? globalDefaults.enableMemoryCache) {
            ValidatedField(title: "Memory cache %",
                           value: doubleBinding(\.memoryCachePercent,
                                                default: globalDefaults.memoryCachePercent,
                                                transformGet: { $0 * 100 },
                                                transformSet: { $0 / 100 }),
                           range: 0...100, step: 1, format: "%.0f%%")
            ValidatedField(title: "Memory cache TTL (min, 0=off)",
                           value: doubleBinding(\.memoryCacheTTLMinutes,
                                                default: globalDefaults.memoryCacheTTLMinutes),
                           range: 0...1440, step: 1, format: "%.0f min")
        }

        // §354 — "KV cache quantization" picker below is the SINGLE
        // source of truth. The old separate `enableTurboQuant` Bool
        // toggle was redundant (SettingsStore.resolved() derives the
        // Bool from the picker string at line ~410), and showing both
        // let users set them to disagree. Toggle removed.
        //
        // TurboQuant bits stepper still renders — gated on the picker
        // currently reading "turboquant". The orphan `enableBlockDiskCache`
        // toggle was also removed: it's a Python-parity field that has
        // never had a Swift consumer. The L2 disk cache below IS the
        // disk cache on Swift. If a future block-level store lands it
        // will have its own field + clear semantics.
        if (s.kvCacheQuantization ?? globalDefaults.kvCacheQuantization)
            .lowercased() == "turboquant"
        {
            Stepper(value: Binding(
                get: { s.turboQuantBits ?? globalDefaults.turboQuantBits },
                set: { s.turboQuantBits = $0; commit() }
            ), in: 3...8, step: 1) {
                Text(L10n.SessionConfig.turboquantBitsFormat.format(locale: appLocale, Int64(s.turboQuantBits ?? globalDefaults.turboQuantBits)))
            }
            .padding(.leading, 20)
        }
        toggleRow("Prefix cache",     boolBinding(\.enablePrefixCache,     default: globalDefaults.enablePrefixCache))
        toggleRow("SSM companion",    boolBinding(\.enableSSMCompanion,    default: globalDefaults.enableSSMCompanion))
        toggleRow("L2 disk cache",    boolBinding(\.enableDiskCache,       default: globalDefaults.enableDiskCache))
        // Directory + GB cap: previously only in Tray, not per-session.
        // Audit 2026-04-16 — disk cache dir was unconfigurable anywhere
        // per-session; users with external SSDs couldn't redirect the
        // cache off the boot volume.
        if s.enableDiskCache ?? globalDefaults.enableDiskCache {
            // NOTE: disk cache dir + max-GB are GLOBAL-only knobs because
            // `setupCacheCoordinator` runs at load-time and reads
            // `settings.global()`. Per-session overrides for these cannot
            // take effect without a coordinator re-init, so the UI shows
            // the global defaults as read-only info. Edit them via Tray →
            // Server section. Audit 2026-04-16 cleanup.
            HStack {
                Text(L10n.SessionConfig.diskCacheDirFormat.format(
                    locale: appLocale,
                    (globalDefaults.diskCacheDir.isEmpty
                        ? L10n.SessionConfig.defaultLabel.render(appLocale)
                        : globalDefaults.diskCacheDir) as NSString))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .padding(.leading, 20)
            HStack {
                Text(L10n.SessionConfig.diskCacheMaxFormat.format(
                    locale: appLocale, Int64(globalDefaults.diskCacheMaxGB)))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            .padding(.leading, 20)
        }
        // JANG repack is NOT a user setting — JangLoader auto-detects
        // jang_config.json at load time and activates itself. Showing a
        // toggle here would let users disable it on a JANG model (which
        // would silently fail to load the MXTQ weights) or enable it on
        // a non-JANG model (no-op but confusing). Removed 2026-04-15.

        Picker("KV cache quantization", selection: Binding(
            get: { s.kvCacheQuantization ?? globalDefaults.kvCacheQuantization },
            set: { s.kvCacheQuantization = $0; commit() }
        )) {
            // Audit 2026-04-16: lead with TurboQuant (default + recommended).
            // "None" removed — users shouldn't accidentally downgrade from
            // TQ to raw fp16 KV cache; the advanced fixed-bit options stay
            // available for comparison/testing.
            Text("TurboQuant").tag("turboquant")
            Text("Q8").tag("q8")
            Text("Q4").tag("q4")
        }
        .pickerStyle(.segmented)
        Text(L10n.SessionConfig.cacheKindHelp.render(appLocale))
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private var inferenceSection: some View {
        // §368 — surface the model's own sampling recommendations when
        // generation_config.json exists. Makes Qwen-recommended temp=0.6
        // vs Gemma-recommended temp=1.0 vs Nemotron default visible in
        // the UI instead of silently using the global 0.7 default.
        modelDefaultsCaption
        toggleRow("Enable thinking by default",
                  triStateBinding(\.defaultEnableThinking, default: globalDefaults.defaultEnableThinking ?? false))

        ValidatedField(title: "Temperature",
                       value: doubleBinding(\.defaultTemperature, default: globalDefaults.defaultTemperature),
                       range: 0.0...2.0, step: 0.05, format: "%.2f")
        ValidatedField(title: "Top-p",
                       value: doubleBinding(\.defaultTopP, default: globalDefaults.defaultTopP),
                       range: 0.0...1.0, step: 0.01, format: "%.2f")
        ValidatedField(title: "Top-k",
                       value: intBinding(\.defaultTopK, default: globalDefaults.defaultTopK),
                       range: 0...1000, step: 1)
        ValidatedField(title: "Min-p",
                       value: doubleBinding(\.defaultMinP, default: globalDefaults.defaultMinP),
                       range: 0.0...1.0, step: 0.01, format: "%.2f")
        ValidatedField(title: "Repetition penalty",
                       value: doubleBinding(\.defaultRepetitionPenalty, default: globalDefaults.defaultRepetitionPenalty),
                       range: 1.0...2.0, step: 0.01, format: "%.2f")
        ValidatedField(title: "Max tokens",
                       value: intBinding(\.defaultMaxTokens, default: globalDefaults.defaultMaxTokens),
                       range: 256...131072, step: 256)

        textFieldRow("System prompt",
                     placeholder: globalDefaults.defaultSystemPrompt ?? "(none)",
                     value: Binding(
                        get: { s.defaultSystemPrompt ?? "" },
                        set: { s.defaultSystemPrompt = $0.isEmpty ? nil : $0; commit() }
                     ))
    }

    @ViewBuilder
    private var lifecycleSection: some View {
        toggleRow("Enable idle sleep",
                  boolBinding(\.idleEnabled, default: globalDefaults.idleEnabled))

        // Stored in seconds (60..86400 per spec); UI displays minutes
        // (1..1440). Hidden when idle sleep is off so the user doesn't
        // think they're configuring a feature that won't fire. Audit
        // 2026-04-15 finding #5.
        if (s.idleEnabled ?? globalDefaults.idleEnabled) {
            ValidatedField(title: "Soft sleep after (min)",
                           value: doubleBinding(
                            \.idleSoftSec,
                            default: globalDefaults.idleSoftSec,
                            transformGet: { $0 / 60.0 },
                            transformSet: { $0 * 60.0 }
                           ),
                           range: 1...1440, step: 1, format: "%.0f")
            ValidatedField(title: "Deep sleep after (min)",
                           value: doubleBinding(
                            \.idleDeepSec,
                            default: globalDefaults.idleDeepSec,
                            transformGet: { $0 / 60.0 },
                            transformSet: { $0 * 60.0 }
                           ),
                           range: 1...1440, step: 1, format: "%.0f",
                           invariant: { deepMin in
                               let softMin = (s.idleSoftSec ?? globalDefaults.idleSoftSec) / 60.0
                               if deepMin < softMin {
                                   return "Deep sleep must be ≥ soft sleep (\(Int(softMin)) min)"
                               }
                               return nil
                           })
        }
    }

    @ViewBuilder
    // Per-session HTTP listener binding ONLY. Authentication and CORS
    // live in `authSection` below — those are security-policy concerns,
    // not HTTP-listener concerns, and the two got confused when they
    // shared a section. Model alias (how the model identifies itself in
    // API responses) moved to `modelSection` since it's model-identity,
    // not network plumbing. See project_vmlx_session_sections.md.
    private var serverSection: some View {
        Text(L10n.SessionConfig.httpListenerHelp.render(appLocale))
            .font(Theme.Typography.caption)
            .foregroundStyle(Theme.Colors.textLow)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.bottom, Theme.Spacing.xs)
        textFieldRow("Host",
                     placeholder: globalDefaults.defaultHost,
                     value: Binding(
                        get: { s.host ?? "" },
                        set: { s.host = $0.isEmpty ? nil : $0; commit() }
                     ))
        ValidatedField(title: "Port",
                       value: Binding(
                        get: { Double(s.port ?? globalDefaults.defaultPort) },
                        set: { s.port = Int($0); commit() }
                       ),
                       range: 1024...65535, step: 1, format: "%.0f")
        // LAN bind intercepts the write: flipping ON exposes the engine's
        // HTTP listener to every device on the Wi-Fi, so we surface a
        // confirmation dialog before persisting. Flipping OFF is always
        // safe (local-only).
        toggleRow("Allow LAN access (bind 0.0.0.0)",
                  Binding(
                    get: { s.lan ?? globalDefaults.defaultLAN },
                    set: { newValue in
                        if newValue {
                            pendingLANEnable = true
                        } else {
                            s.lan = false
                            commit()
                        }
                    }
                  ))
    }

    /// Authentication + CORS policy. Was tangled into `serverSection`.
    /// Split out because you can (and often want to) change auth
    /// without restarting the HTTP listener, and users looking for
    /// "where do I set the API key" should NOT have to read past
    /// host/port/LAN fields.
    @ViewBuilder
    private var authSection: some View {
        textFieldRow("API key",
                     placeholder: "(inherit global)",
                     value: Binding(
                        get: { s.apiKey ?? "" },
                        set: { s.apiKey = $0.isEmpty ? nil : $0; commit() }
                     ))
        textFieldRow("Admin token",
                     placeholder: "(inherit global)",
                     value: Binding(
                        get: { s.adminToken ?? "" },
                        set: { s.adminToken = $0.isEmpty ? nil : $0; commit() }
                     ))
        textFieldRow("CORS origins (comma-sep)",
                     placeholder: globalDefaults.corsOrigins.joined(separator: ","),
                     value: Binding(
                        get: { s.corsOrigins?.joined(separator: ",") ?? "" },
                        set: {
                            let parts = $0.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
                            s.corsOrigins = parts.isEmpty ? nil : parts
                            commit()
                        }
                     ))
        // §358: CORS is NOT live-swappable — the Hummingbird middleware
        // captures the allowlist at Server.init() time. API key + admin
        // token above ARE live (applyAuthCredentials), but CORS requires
        // a session restart. Previously this lied by omission: no badge,
        // no caption. Users would edit CORS mid-session and silently
        // get the old policy until next Stop + Start. Surface it.
        HStack(alignment: .top, spacing: 4) {
            Image(systemName: "arrow.triangle.2.circlepath")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(Theme.Colors.warning)
            Text(L10n.SessionConfig.restartRequired.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.warning)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 4)
    }

    /// Remote-endpoint section: turns this session into a thin proxy to
    /// an OpenAI / Ollama / Anthropic-compatible server. When enabled the
    /// local engine's load() is skipped — Chat / Terminal dispatch goes
    /// straight to the remote server via RemoteEngineClient. The model
    /// path on disk is ignored (kept only as a display name).
    @ViewBuilder
    private var remoteSection: some View {
        Toggle(L10n.SessionConfig.useRemoteEndpoint.render(appLocale),
               isOn: remoteEnabledBinding)
            .toggleStyle(.switch)

        if s.remoteURL != nil {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                remoteEndpointRow
                remoteProtocolRow
                remoteModelNameRow
                remoteAPIKeyRow
                Text(L10n.SessionConfig.remoteEndpointHelp.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.leading, Theme.Spacing.md)
        }
    }

    private var remoteEnabledBinding: Binding<Bool> {
        Binding(
            get: { s.remoteURL != nil && !(s.remoteURL ?? "").isEmpty },
            set: { on in
                if on {
                    if (s.remoteURL ?? "").isEmpty { s.remoteURL = "http://" }
                } else {
                    s.remoteURL = nil
                    s.remoteAPIKey = nil
                    s.remoteModelName = nil
                    s.remoteProtocol = nil
                }
                commit()
            }
        )
    }

    private var remoteEndpointRow: some View {
        HStack {
            Text(L10n.SessionConfig.endpointURL.render(appLocale)).frame(width: 140, alignment: .leading)
            TextField("https://api.openai.com or http://192.168.1.50:8000",
                      text: Binding(
                        get: { s.remoteURL ?? "" },
                        set: { s.remoteURL = $0; commit() }
                      ))
            .textFieldStyle(.roundedBorder)
            .font(Theme.Typography.mono)
        }
    }

    private var remoteProtocolRow: some View {
        HStack {
            Text(L10n.SessionConfig.proto.render(appLocale)).frame(width: 140, alignment: .leading)
            Picker("", selection: Binding(
                get: { s.remoteProtocol ?? "openai" },
                set: { s.remoteProtocol = $0; commit() }
            )) {
                Text("OpenAI-compatible").tag("openai")
                Text("Ollama").tag("ollama")
                Text("Anthropic").tag("anthropic")
            }
            .pickerStyle(.menu)
        }
    }

    private var remoteModelNameRow: some View {
        HStack {
            Text(L10n.SessionConfig.modelName.render(appLocale)).frame(width: 140, alignment: .leading)
            TextField("e.g. gpt-4o-mini, llama3.1, claude-sonnet-4-5",
                      text: Binding(
                        get: { s.remoteModelName ?? "" },
                        set: { s.remoteModelName = $0; commit() }
                      ))
            .textFieldStyle(.roundedBorder)
            .font(Theme.Typography.mono)
        }
    }

    private var remoteAPIKeyRow: some View {
        HStack {
            Text(L10n.SessionConfig.apiKey.render(appLocale)).frame(width: 140, alignment: .leading)
            SecureField("sk-... (left blank for unauthenticated remotes)",
                        text: Binding(
                          get: { s.remoteAPIKey ?? "" },
                          set: { s.remoteAPIKey = $0.isEmpty ? nil : $0; commit() }
                        ))
            .textFieldStyle(.roundedBorder)
            .font(Theme.Typography.mono)
        }
    }

    @ViewBuilder
    private var advancedSection: some View {
        // Smelt — sub-controls hidden when off. Audit 2026-04-15 finding #4.
        // NOTE: Smelt is not yet wired in the Swift engine (Python-only).
        // The engine logs a one-shot warning per request when the flag is
        // on so users aren't silently no-opped. See `EngineDFlash.swift`
        // / `StreamDFlash.swift` for the integrated DFlash replacement.
        // iter-53: added explicit caption — the log warning only
        // surfaces in the logs tab, and users flipping the toggle
        // need to see the gap up-front.
        toggleRow("Smelt mode (Python engine only)",
                  boolBinding(\.smelt, default: globalDefaults.smelt))
        Text(L10n.SessionConfig.smeltHelp.render(appLocale))
            .font(Theme.Typography.caption)
            .foregroundStyle(Theme.Colors.textLow)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.leading, 4)
            .padding(.bottom, 2)
        if (s.smelt ?? globalDefaults.smelt) {
            // iter-62: globalDefaults.smeltMode defaults to "" which
            // renders as a blank placeholder that looks like the field
            // is broken. Fall back to "(inherit global)" so users know
            // the field accepts input but currently has no default.
            textFieldRow("Smelt variant",
                         placeholder: globalDefaults.smeltMode.isEmpty
                            ? "(inherit global — e.g. routed/default)"
                            : globalDefaults.smeltMode,
                         value: Binding(
                            get: { s.smeltMode ?? "" },
                            set: { s.smeltMode = $0.isEmpty ? nil : $0; commit() }
                         ))
        }

        // JANG-DFlash speculative decoding — block diffusion drafter +
        // DDTree beam. Targets: MiniMax, Mistral 4, DeepSeek V3.
        // `dflashDrafterPath` must point at a compatible safetensors
        // checkpoint; when unset the engine falls back to the standard
        // token iterator silently (and logs why in the stream log).
        toggleRow("DFlash speculative decoding",
                  boolBinding(\.dflash, default: globalDefaults.dflash))
        if (s.dflash ?? globalDefaults.dflash) {
            // iter-62: drafter path defaults to "" globally — show a
            // helpful placeholder so users know what to put there.
            // Engine logs a skip + falls back to standard token iter
            // when empty (see EngineDFlash.swift:92).
            textFieldRow("DFlash drafter path",
                         placeholder: globalDefaults.dflashDrafterPath.isEmpty
                            ? "/path/to/drafter.safetensors (inherit global = none)"
                            : globalDefaults.dflashDrafterPath,
                         value: Binding(
                            get: { s.dflashDrafterPath ?? "" },
                            set: { s.dflashDrafterPath = $0.isEmpty ? nil : $0; commit() }
                         ))
            ValidatedField(title: "DFlash block size",
                           value: intBinding(\.dflashBlockSize, default: globalDefaults.dflashBlockSize),
                           range: 1...64, step: 1)
            ValidatedField(title: "DFlash top-K (per slot)",
                           value: intBinding(\.dflashTopK, default: globalDefaults.dflashTopK),
                           range: 1...16, step: 1)
            ValidatedField(title: "DFlash num paths",
                           value: intBinding(\.dflashNumPaths, default: globalDefaults.dflashNumPaths),
                           range: 1...256, step: 1)
            textFieldRow("DFlash tap layers (csv)",
                         placeholder: globalDefaults.dflashTapLayers.isEmpty
                            ? "e.g. 10,22,34,46,58 (inherit global = none)"
                            : globalDefaults.dflashTapLayers,
                         value: Binding(
                            get: { s.dflashTapLayers ?? "" },
                            set: { s.dflashTapLayers = $0.isEmpty ? nil : $0; commit() }
                         ))
        }

        // Flash MoE — slot bank slider hidden when off. When user flips
        // the toggle ON, auto-populate slotBank from the global default
        // if still nil, so the slider shows a real number instead of 0.
        // Audit finding B5: toggling flashMoe with slotBank still nil
        // produced a silently-broken (bank=0) load.
        toggleRow("Flash MoE",
                  Binding<Bool>(
                    get: { s.flashMoe ?? globalDefaults.flashMoe },
                    set: { newValue in
                        s.flashMoe = newValue
                        if newValue, s.flashMoeSlotBank == nil {
                            s.flashMoeSlotBank = globalDefaults.flashMoeSlotBank
                        }
                        commit()
                    }
                  ))
        if (s.flashMoe ?? globalDefaults.flashMoe) {
            ValidatedField(title: "Flash MoE slot bank",
                           value: intBinding(\.flashMoeSlotBank, default: globalDefaults.flashMoeSlotBank),
                           range: 32...4096, step: 32)
        }

        // §384 — Distributed toggle removed from UI. The engine has no
        // consumer for it (GlobalSettings.distributed is doc-only; the
        // resolver threads it through but nothing else reads it). Per
        // Eric's "no BS placeholders" rule: don't ship a dead toggle.
        // GlobalSettings.distributed stays in the struct for forward-
        // compatibility when RDMA/tensor-parallel lands — the UI comes
        // back then, wired through a real engine consumer.
    }

    @ViewBuilder
    private var loggingSection: some View {
        Picker("Default log level", selection: Binding(
            get: { globalDefaults.defaultLogLevel },
            set: { newVal in
                Task {
                    var g = await app.engine.settings.global()
                    g.defaultLogLevel = newVal
                    await app.engine.applySettings(g)
                    globalDefaults = g
                }
            }
        )) {
            ForEach(["trace","debug","info","warn","error"], id: \.self) { lvl in
                Text(lvl.uppercased()).tag(lvl)
            }
        }
        .pickerStyle(.segmented)
    }

    /// §368 — compact caption showing model-recommended sampling params.
    /// Hidden when the model ships no generation_config.json (no fields
    /// populated). Parts are conditionally joined so users only see what's
    /// actually recommended. Rendered above inference fields so users
    /// understand why their values might differ from the global 0.7 default.
    @ViewBuilder
    private var modelDefaultsCaption: some View {
        let parts: [String] = {
            var p: [String] = []
            if let t = modelDefaults.temperature { p.append("temp=\(String(format: "%.2f", t))") }
            if let v = modelDefaults.topP       { p.append("top_p=\(String(format: "%.2f", v))") }
            if let v = modelDefaults.topK       { p.append("top_k=\(v)") }
            if let v = modelDefaults.repetitionPenalty { p.append("rep=\(String(format: "%.2f", v))") }
            if let v = modelDefaults.maxTokens  { p.append("max=\(v)") }
            return p
        }()
        if !parts.isEmpty {
            HStack(alignment: .top, spacing: 4) {
                Image(systemName: "sparkles")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Colors.accent)
                Text("Model recommends: " + parts.joined(separator: ", "))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Load + commit

    private func load() async {
        let store = app.engine.settings
        globalDefaults = await store.global()
        if let existing = await store.session(sessionId) {
            s = existing
            modelPath = existing.modelPath
        } else if let cardModel = app.sessions.first(where: { $0.id == sessionId }) {
            s = SessionSettings(modelPath: cardModel.modelPath)
            modelPath = cardModel.modelPath
        }
        // §368 — read generation_config.json for the session's model
        // path (off the main thread, since it's just a JSON file read).
        // We can't ask the engine actor for its live `loadedModelDefaults`
        // because THIS session may not be the one currently loaded in the
        // actor, but the file read is cheap + authoritative.
        let path = modelPath
        modelDefaults = await Task.detached(priority: .utility) {
            Engine.readGenerationConfig(at: path)
        }.value
        loaded = true
    }

    private func commit() {
        guard loaded else { return }
        let snap = s
        let id = sessionId
        Task { await app.engine.settings.setSession(id, snap) }
    }

    // MARK: - Binding helpers
    //
    // These wrap a nullable SessionSettings keypath (e.g. `\.maxNumSeqs: Int?`)
    // into a non-optional Binding that reads the global default when the
    // session field is nil. Writing `nil`-equivalent (the default value) will
    // CLEAR the override — but for slider ergonomics we always write the
    // selected value, which means the override "sticks" even when matching
    // the default. That matches Electron behavior.

    private func intBinding(
        _ path: WritableKeyPath<SessionSettings, Int?>,
        default def: Int
    ) -> Binding<Double> {
        Binding(
            get: { Double(s[keyPath: path] ?? def) },
            set: { s[keyPath: path] = Int($0); commit() }
        )
    }

    private func doubleBinding(
        _ path: WritableKeyPath<SessionSettings, Double?>,
        default def: Double,
        transformGet: @escaping (Double) -> Double = { $0 },
        transformSet: @escaping (Double) -> Double = { $0 }
    ) -> Binding<Double> {
        Binding(
            get: { transformGet(s[keyPath: path] ?? def) },
            set: { s[keyPath: path] = transformSet($0); commit() }
        )
    }

    private func boolBinding(
        _ path: WritableKeyPath<SessionSettings, Bool?>,
        default def: Bool
    ) -> Binding<Bool> {
        Binding(
            get: { s[keyPath: path] ?? def },
            set: { s[keyPath: path] = $0; commit() }
        )
    }

    /// Tri-state (Bool?) — UI renders as a 2-state toggle but stores the
    /// override. Matches `defaultEnableThinking` semantics.
    private func triStateBinding(
        _ path: WritableKeyPath<SessionSettings, Bool?>,
        default def: Bool
    ) -> Binding<Bool> {
        Binding(
            get: { s[keyPath: path] ?? def },
            set: { s[keyPath: path] = $0; commit() }
        )
    }

    // MARK: - Row UI

    @ViewBuilder
    private func disclosure<Content: View>(
        _ title: String,
        isOn: Binding<Bool>,
        loadTimeOnly: Bool = false,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        DisclosureGroup(isExpanded: isOn) {
            VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                if loadTimeOnly {
                    // R5 §306 — prominent yellow badge + caption so
                    // users notice mid-session that fields under this
                    // disclosure are baked into the model container at
                    // load time (engine kind, cache block count,
                    // TurboQuant bits, etc.). Changing them on a
                    // running session silently persists the new value
                    // but the engine keeps the OLD one until
                    // Stop+Start — the yellow ⟳ is the universal
                    // "restart required" signal.
                    HStack(alignment: .top, spacing: Theme.Spacing.sm) {
                        Image(systemName: "arrow.triangle.2.circlepath")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(Theme.Colors.warning)
                        Text(L10n.SessionConfig.restartRequired.render(appLocale))
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.vertical, 4)
                    .padding(.horizontal, Theme.Spacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.sm)
                            .fill(Theme.Colors.warning.opacity(0.10))
                    )
                    .padding(.bottom, 2)
                }
                content()
            }
            .padding(.top, Theme.Spacing.sm)
        } label: {
            HStack(spacing: 4) {
                Text(title.uppercased())
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                if loadTimeOnly {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(Theme.Colors.warning)
                        .help(L10n.Tooltip.sectionRestart.render(appLocale))
                }
            }
        }
        .tint(Theme.Colors.textMid)
    }

    @ViewBuilder
    private func toggleRow(_ label: String, _ binding: Binding<Bool>) -> some View {
        Toggle(label, isOn: binding)
            .toggleStyle(.switch)
            .tint(Theme.Colors.accent)
    }

    @ViewBuilder
    private func sliderRow(
        _ label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        format: String = "%.0f",
        scale: Double = 1.0
    ) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack {
                Text(label)
                    .font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textMid)
                Spacer()
                Text(String(format: format, value.wrappedValue * scale))
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textHigh)
            }
            Slider(value: value, in: range, step: step)
                .tint(Theme.Colors.accent)
        }
    }

    @ViewBuilder
    private func textFieldRow(
        _ label: String,
        placeholder: String,
        value: Binding<String>
    ) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text(label)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
            TextField(placeholder, text: value)
                .textFieldStyle(.plain)
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textHigh)
                .padding(Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.sm)
                        .fill(Theme.Colors.surfaceHi)
                )
        }
    }
}

// MARK: - ModelPickerRow (still used by SessionDashboard's New-Session popover)

struct ModelPickerRow: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale
    @Binding var path: URL?
    @State private var showImporter = false
    @State private var entries: [ModelLibrary.ModelEntry] = []
    @State private var scanning = false

    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Menu {
                if entries.isEmpty {
                    Text(L10n.SessionConfig.noModelsFound.render(appLocale)).foregroundStyle(Theme.Colors.textLow)
                }
                ForEach(entries) { entry in
                    Button {
                        path = entry.canonicalPath
                    } label: {
                        entryLabel(entry)
                    }
                }
                Divider()
                Button(L10n.SessionConfig.browse.render(appLocale)) { showImporter = true }
                Button(L10n.SessionConfig.addCustomDir.render(appLocale)) { pickUserDir() }
            } label: {
                HStack {
                    Text(displayForPath())
                        .font(Theme.Typography.body)
                        .foregroundStyle(path == nil ? Theme.Colors.textLow : Theme.Colors.textHigh)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Image(systemName: "chevron.down")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                }
                .padding(Theme.Spacing.md)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surface)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
            }
            .buttonStyle(.plain)
            .menuStyle(.borderlessButton)

            Button {
                Task { await refresh(force: true) }
            } label: {
                Image(systemName: "arrow.clockwise")
                    .foregroundStyle(Theme.Colors.textMid)
                    .padding(Theme.Spacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .fill(Theme.Colors.surface)
                            .overlay(
                                RoundedRectangle(cornerRadius: Theme.Radius.md)
                                    .stroke(Theme.Colors.border, lineWidth: 1)
                            )
                    )
            }
            .buttonStyle(.plain)
            .disabled(scanning)
        }
        .task { await refresh(force: false) }
        // Auto-refresh the picker list when a download completes.
        // AppState.downloadedModelCount is incremented on every .completed
        // DownloadManager event — observing it here means users never have
        // to click the refresh button after a HuggingFace pull finishes.
        .onChange(of: app.downloadedModelCount) { _, _ in
            Task { await refresh(force: true) }
        }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.folder]
        ) { result in
            if case .success(let url) = result { path = url }
        }
    }

    private func displayForPath() -> String {
        if let path {
            if let match = entries.first(where: { $0.canonicalPath == path }) {
                return match.displayName
            }
            return path.lastPathComponent
        }
        return "Select a model"
    }

    private func entryLabel(_ entry: ModelLibrary.ModelEntry) -> Label<Text, Image> {
        let icon: String
        switch entry.modality {
        case .text:      icon = "text.bubble"
        case .vision:    icon = "eye"
        case .embedding: icon = "square.grid.3x3"
        case .image:     icon = "photo"
        case .rerank:    icon = "arrow.up.arrow.down"
        case .unknown:   icon = "questionmark"
        }
        var badges = ""
        if entry.isJANG { badges += " · JANG" }
        if entry.isMXTQ { badges += " · MXTQ" }
        if let b = entry.quantBits { badges += " · \(b)bit" }
        let size = byteFormatter(entry.totalSizeBytes)
        return Label {
            Text("\(entry.displayName)  ·  \(entry.family)  ·  \(size)\(badges)")
        } icon: {
            Image(systemName: icon)
        }
    }

    private func byteFormatter(_ bytes: Int64) -> String {
        let f = ByteCountFormatter()
        f.countStyle = .binary
        return f.string(fromByteCount: bytes)
    }

    private func refresh(force: Bool) async {
        scanning = true
        defer { scanning = false }
        // A2: image models (Flux, Z-Image, SDXL, Schnell, Qwen-Image) are
        // classified modality=.image by ModelLibrary and handled by the
        // Image tab's dedicated FluxBackend pipeline, not the chat/server
        // Engine.load() path. Surfacing them here lets users pick a Flux
        // model for a chat session which fails at load with a confusing
        // `notImplemented` error. Filter them out of the server picker;
        // the Image tab has its own catalog.
        entries = await app.engine.scanModels(force: force)
            .filter { $0.modality != .image }
    }

    private func pickUserDir() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Add"
        if panel.runModal() == .OK, let url = panel.url {
            Task {
                await app.engine.modelLibrary.addUserDir(url)
                await refresh(force: true)
            }
        }
    }
}
