import SwiftUI
import vMLXEngine
import vMLXTheme

/// Menu-bar server control center. Renders inside a `MenuBarExtra(style: .window)`
/// popover so it can host the full set of macOS controls — sliders, steppers,
/// buttons, live stats. The menu-style MenuBarExtra would have limited us to
/// plain buttons; `.window` gives us a ~420 pt-wide SwiftUI surface.
///
/// Sections, top → bottom:
///
///   1. Header        — engine state pill, model name, live tok/s, uptime
///   2. Lifecycle     — Start / Stop / Restart / Soft Sleep / Deep Sleep / Wake
///   3. Model         — current path + "Open picker"
///   4. Server        — host, port, LAN toggle, base URL + copy
///   5. Performance   — GPU mem bar, CPU %, queue depth
///   6. Runtime       — temperature / top-p / top-k / min-p / rep penalty / max tokens
///   7. Cache         — turbo-quant bits, disk cache toggle + GB, memory cache
///   8. MoE           — Flash MoE toggle + slot bank slider
///   9. Adapter       — active LoRA + unload
///  10. Logs          — last 5 log lines peek
///  11. Footer        — Appearance / Open app / Quit
///
/// All widgets read + write live from `engine.settings.global()` /
/// `engine.settings.setGlobal(...)` so changes survive app restart. Lifecycle
/// buttons call `Engine.load` / `stop` / `softSleep` / `deepSleep` / `wake`
/// directly — no HTTP roundtrip, no blocking main thread.
struct TrayItem: View {
    @Environment(AppState.self) private var app
    @Environment(\.openWindow) private var openWindow
    @Environment(\.appLocale) private var appLocale
    @AppStorage("vmlx.appearance") private var appearanceRaw: String =
        AppearanceMode.dark.rawValue

    // Live metrics driven by a polling task started in `.task`. Separate from
    // `app.engineState` because metrics flip at ~1 Hz whereas state is event.
    @State private var snapshot: MetricsCollector.Snapshot? = nil
    @State private var metricsTask: Task<Void, Never>? = nil
    // Live-editable GlobalSettings snapshot. Fetched on appear, pushed back
    // after each slider/toggle debounce window (200 ms).
    @State private var draft: GlobalSettings = GlobalSettings()
    @State private var loadedDraft = false
    @State private var pushDebounce: Task<Void, Never>? = nil
    // Logs tail — last 5 lines from engine.logs.
    @State private var logTail: [String] = []
    // Live cache architecture flags for the header pills. Refreshed at the
    // same 1 Hz cadence as `snapshot` so users get visual confirmation
    // whether the loaded model actually uses hybrid SSM / SWA / TQ
    // without cracking open the Server → Cache panel.
    @State private var archFlags = ArchPillFlags()
    // Disclosure group open/closed state, persisted across popover re-opens.
    @State private var showLifecycle = true
    @State private var showServer = false
    @State private var showRuntime = false
    @State private var showCache = false
    @State private var showMoE = false
    @State private var showAdapter = false
    @State private var showLogging = false
    @State private var showLogs = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                if app.sessions.count > 1 { sessionPickerRow }
                Divider()
                lifecycleSection
                modelRow
                Divider()
                statsRow
                Divider()
                disclosureGroup(
                    title: L10n.TrayPanel.serverBinding.render(appLocale), open: $showServer, icon: "network"
                ) { serverSection }
                disclosureGroup(
                    title: L10n.TrayPanel.samplingGlobals.render(appLocale), open: $showRuntime, icon: "slider.horizontal.3"
                ) { runtimeSection }
                disclosureGroup(
                    title: L10n.TrayPanel.cache.render(appLocale), open: $showCache, icon: "externaldrive.fill"
                ) { cacheSection }
                disclosureGroup(
                    title: L10n.TrayPanel.flashMoE.render(appLocale), open: $showMoE, icon: "bolt.fill"
                ) { moeSection }
                disclosureGroup(
                    title: L10n.TrayPanel.adapter.render(appLocale), open: $showAdapter, icon: "puzzlepiece.fill"
                ) { adapterSection }
                disclosureGroup(
                    title: L10n.TrayPanel.logging.render(appLocale), open: $showLogging, icon: "text.alignleft"
                ) { loggingSection }
                disclosureGroup(
                    title: L10n.TrayPanel.recentLogs.render(appLocale), open: $showLogs, icon: "doc.text"
                ) { logsSection }
                Divider()
                footerRow
            }
            .padding(14)
        }
        .frame(width: 420)
        .frame(maxHeight: 600)
        .task { await startMetricsPoll() }
        .onDisappear { metricsTask?.cancel(); metricsTask = nil }
        .onAppear {
            Task { await loadDraftIfNeeded() }
            // H4 §273 — opening the tray popover acknowledges the
            // error-count badge. Mirrors the Logs panel behaviour:
            // user saw there's been an issue, count clears until the
            // next post-ack `.error` lands.
            app.acknowledgeErrors()
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            ZStack(alignment: .topTrailing) {
                Image(systemName: Self.icon(for: app.engineState))
                    .foregroundColor(stateColor)
                    .font(.system(size: 18, weight: .semibold))
                // H4 §273 — red dot for unacknowledged .error log entries.
                // Non-zero → dot is visible. Hover/click the tray clears
                // it via `app.acknowledgeErrors()` (wired in onAppear).
                if app.errorLogCount > 0 {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 8, height: 8)
                        .overlay(Circle().stroke(Color.white, lineWidth: 1))
                        .offset(x: 4, y: -4)
                        .help(L10n.Tooltip.unreadErrorLogs.format(
                            locale: appLocale, "\(app.errorLogCount)" as NSString))
                }
            }
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Text(stateLabel)
                        .font(.system(size: 13, weight: .semibold))
                    if archFlags.hybrid { trayArchPill("hybrid", tint: .accentColor) }
                    if archFlags.swa    { trayArchPill("SWA",    tint: .green) }
                    if archFlags.tq     { trayArchPill("TQ",     tint: .accentColor) }
                    // H1 §271 — thermal throttle pill.  Only shown when
                    // the OS is actually pressuring.  `.fair` is mild;
                    // `.serious`/`.critical` deserve a prominent warning
                    // because decode tok/s will drop ~40% until the Mac
                    // cools off. `thermalPill` is a ViewBuilder that
                    // emits EmptyView on `.nominal`.
                    thermalPill
                }
                Text(modelName)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                if let s = snapshot, s.tokensPerSecondRolling > 0 {
                    Text(String(format: "%.0f tok/s", s.tokensPerSecondRolling))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundColor(.accentColor)
                }
                if let p = app.loadProgress, let frac = p.fraction {
                    Text("\(Int(frac * 100))%")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                } else if case .loading = app.engineState {
                    Text(app.loadProgress?.label ?? "…")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
        }
    }

    // MARK: - Session picker (multi-engine)

    /// Compact picker that flips `app.selectedServerSessionId` when the
    /// user has more than one server session open. Mirrors the Electron
    /// tray's session switcher — previously the macOS tray showed only
    /// the selected session with no way to switch without opening the
    /// main window. Hidden when there's 0 or 1 session (no point in a
    /// picker with a single choice).
    private var sessionPickerRow: some View {
        HStack(spacing: 6) {
            Image(systemName: "rectangle.on.rectangle")
                .foregroundStyle(.secondary)
                .font(.system(size: 11))
            Picker(L10n.Tray.session.render(appLocale), selection: Binding(
                get: { app.selectedServerSessionId ?? app.sessions.first?.id ?? UUID() },
                set: { newId in
                    app.selectedServerSessionId = newId
                    app.rebindEngineObserver()
                }
            )) {
                ForEach(app.sessions) { s in
                    Text("\(s.displayName) · \(Self.sessionShortState(s))")
                        .tag(s.id)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
        }
        .padding(.top, 4)
    }

    /// One-word state label for the picker row so the menu stays scannable
    /// even with long model names.
    private static func sessionShortState(_ s: Session) -> String {
        // iter-131 §157: remote sessions stay at local-engine .stopped
        // forever but are functionally ready.
        if s.isRemote { return "remote" }
        switch s.state {
        case .stopped:        return "stopped"
        case .loading:        return "loading"
        case .running:        return "running"
        case .standby(.soft): return "soft-sleep"
        case .standby(.deep): return "deep-sleep"
        case .error:          return "error"
        }
    }

    // MARK: - Lifecycle row

    private var lifecycleSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                lifecycleButton(L10n.Tray.start.render(appLocale), icon: "play.fill",
                                enabled: canStart, tint: .green) {
                    app.onTrayStartServer()
                }
                lifecycleButton(L10n.Tray.stop.render(appLocale), icon: "stop.fill",
                                enabled: canStop, tint: .red) {
                    app.onTrayStopServer()
                }
                lifecycleButton(L10n.Tray.restart.render(appLocale), icon: "arrow.clockwise",
                                enabled: canRestart, tint: .orange) {
                    app.onTrayRestartServer()
                }
            }
            HStack(spacing: 6) {
                lifecycleButton(L10n.Tray.softSleep.render(appLocale), icon: "moon",
                                enabled: canSoftSleep, tint: .blue) {
                    app.onTraySoftSleepServer()
                }
                lifecycleButton(L10n.Tray.deepSleep.render(appLocale), icon: "moon.zzz.fill",
                                enabled: canDeepSleep, tint: .purple) {
                    app.onTrayDeepSleepServer()
                }
                lifecycleButton(L10n.Tray.wake.render(appLocale), icon: "sun.max.fill",
                                enabled: canWake, tint: .yellow) {
                    app.onTrayWakeServer()
                }
            }
        }
    }

    private func lifecycleButton(
        _ title: String, icon: String,
        enabled: Bool, tint: Color, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                Text(title)
                    .font(.system(size: 11, weight: .medium))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .foregroundColor(enabled ? tint : .secondary)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(enabled ? tint.opacity(0.12) : Color.secondary.opacity(0.05))
            )
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        // VoiceOver reads the title (e.g. "Start", "Soft Sleep") + the
        // engine-state hint so the user knows why a button is disabled.
        .accessibilityLabel(title)
        .accessibilityHint(enabled
            ? "Sends the \(title.lowercased()) command to the server"
            : "Disabled in the current engine state")
    }

    // MARK: - Model row

    private var modelRow: some View {
        HStack {
            Image(systemName: "doc.fill")
                .foregroundStyle(.secondary)
                .font(.system(size: 11))
            Text(modelName)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button(L10n.Tray.pick.render(appLocale)) {
                openAppWindow()
                app.mode = .server
            }
            .buttonStyle(.plain)
            .font(.system(size: 11, weight: .medium))
            .foregroundColor(.accentColor)
        }
    }

    // MARK: - Stats row

    private var statsRow: some View {
        HStack(spacing: 12) {
            stat(
                "GPU",
                value: gpuLabel,
                icon: "cpu"
            )
            stat(
                "CPU",
                value: snapshot.map { String(format: "%.0f%%", $0.cpuPercent) } ?? "—",
                icon: "gauge"
            )
            stat(
                "Queue",
                value: snapshot.map { "\($0.queueDepth)" } ?? "—",
                icon: "list.bullet"
            )
            stat(
                "Active",
                value: snapshot.map { "\($0.activeRequests)" } ?? "—",
                icon: "bolt.horizontal"
            )
        }
    }

    private func stat(_ label: String, value: String, icon: String) -> some View {
        VStack(spacing: 2) {
            HStack(spacing: 3) {
                Image(systemName: icon).font(.system(size: 9))
                Text(label).font(.system(size: 9, weight: .medium))
            }
            .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
        }
        .frame(maxWidth: .infinity)
    }

    private var gpuLabel: String {
        guard let s = snapshot, s.gpuMemBytesUsed > 0 else { return "—" }
        let gb = Double(s.gpuMemBytesUsed) / 1_073_741_824
        return String(format: "%.1f GB", gb)
    }

    // MARK: - Disclosure group helper

    private func disclosureGroup<Content: View>(
        title: String,
        open: Binding<Bool>,
        icon: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: { open.wrappedValue.toggle() }) {
                HStack(spacing: 6) {
                    Image(systemName: icon)
                        .foregroundStyle(.secondary)
                        .font(.system(size: 10))
                    Text(title)
                        .font(.system(size: 11, weight: .medium))
                    Spacer()
                    Image(systemName: open.wrappedValue ? "chevron.down" : "chevron.right")
                        .foregroundStyle(.secondary)
                        .font(.system(size: 9))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.vertical, 4)
            if open.wrappedValue {
                content()
                    .padding(.leading, 4)
                    .padding(.top, 4)
                    .padding(.bottom, 6)
            }
        }
    }

    // MARK: - Server section

    private var serverSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledField(L10n.TrayUI.host.render(appLocale)) {
                TextField(
                    "127.0.0.1",
                    text: Binding(
                        get: { draft.defaultHost },
                        set: { draft.defaultHost = $0; schedulePush() }))
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11))
            }
            LabeledField(L10n.TrayUI.port.render(appLocale)) {
                Stepper(
                    value: Binding(
                        get: { draft.defaultPort },
                        set: { draft.defaultPort = $0; schedulePush() }),
                    in: 1024...65535) {
                    Text("\(draft.defaultPort)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            Toggle(
                "Allow LAN (bind 0.0.0.0)",
                isOn: Binding(
                    get: { draft.defaultLAN },
                    set: { draft.defaultLAN = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            HStack {
                Text(L10n.TrayUI.url.render(appLocale))
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                Text(serverURL)
                    .font(.system(size: 10, design: .monospaced))
                    .textSelection(.enabled)
                Spacer()
                Button(L10n.TrayUI.copy.render(appLocale)) {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(serverURL, forType: .string)
                }
                .buttonStyle(.plain)
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(.accentColor)
            }

            // UI-9: Gateway — single base URL that fans /v1/chat/completions
            // out to every loaded session by `model` field. Disabled by
            // default; flip on + pick a port and SDK clients get one URL
            // regardless of how many sessions you're running.
            Divider().padding(.vertical, 4)
            Toggle(
                "Gateway (one URL, many models)",
                isOn: Binding(
                    get: { draft.gatewayEnabled },
                    set: { on in
                        draft.gatewayEnabled = on
                        schedulePush()
                        Task {
                            if on { await app.ensureGatewayRunning() }
                            else  { await app.stopGateway() }
                        }
                    }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            if draft.gatewayEnabled {
                LabeledField(L10n.TrayUI.gatewayPort.render(appLocale)) {
                    Stepper(
                        value: Binding(
                            get: { draft.gatewayPort },
                            set: { draft.gatewayPort = $0; schedulePush() }),
                        in: 1024...65535) {
                        Text("\(draft.gatewayPort)")
                            .font(.system(size: 11, design: .monospaced))
                    }
                }
                Toggle(
                    "Gateway LAN (bind 0.0.0.0)",
                    isOn: Binding(
                        get: { draft.gatewayLAN },
                        set: { draft.gatewayLAN = $0; schedulePush() }))
                    .font(.system(size: 11))
                    .toggleStyle(.switch)
                Text("SDK base URL: http://\(draft.gatewayLAN ? "0.0.0.0" : "127.0.0.1"):\(draft.gatewayPort)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                // §358 — persistent gateway status pill. Replaces the
                // 3-sec flashBanner-only signal. Green = bound at
                // configured port. Orange = auto-bumped (port was taken,
                // we moved to the next free one — requests to the
                // configured port will miss). Red = failed to bind at
                // all (check if Ollama/another service holds the port).
                gatewayStatusPill
            }

            // UI-11: Rate limit. 0 = unlimited. When > 0, RateLimitMiddleware
            // enforces per-IP requests/minute on every bound listener
            // (per-session servers AND the gateway). Change takes effect
            // at the next listener start.
            Divider().padding(.vertical, 4)
            LabeledField(L10n.TrayUI.rateLimit.render(appLocale)) {
                Stepper(
                    value: Binding(
                        get: { draft.rateLimit },
                        set: { draft.rateLimit = max(0, $0); schedulePush() }),
                    in: 0...10000, step: 10) {
                    Text(draft.rateLimit == 0 ? L10n.TrayUI.unlimited.render(appLocale) : "\(draft.rateLimit)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            Text(L10n.TrayUI.rateLimitHelp.render(appLocale))
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
    }

    private var serverURL: String {
        let host = draft.defaultLAN ? "0.0.0.0" : draft.defaultHost
        return "http://\(host):\(draft.defaultPort)"
    }

    // MARK: - Runtime section

    private var runtimeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            slider(
                label: "Temperature",
                value: Binding(
                    get: { draft.defaultTemperature },
                    set: { draft.defaultTemperature = $0; schedulePush() }),
                range: 0.0...2.0, step: 0.05,
                format: "%.2f")
            slider(
                label: "Top-P",
                value: Binding(
                    get: { draft.defaultTopP },
                    set: { draft.defaultTopP = $0; schedulePush() }),
                range: 0.0...1.0, step: 0.01,
                format: "%.2f")
            slider(
                label: "Min-P",
                value: Binding(
                    get: { draft.defaultMinP },
                    set: { draft.defaultMinP = $0; schedulePush() }),
                range: 0.0...1.0, step: 0.01,
                format: "%.2f")
            slider(
                label: "Rep penalty",
                value: Binding(
                    get: { draft.defaultRepetitionPenalty },
                    set: { draft.defaultRepetitionPenalty = $0; schedulePush() }),
                range: 0.5...2.0, step: 0.05,
                format: "%.2f")
            LabeledField(L10n.TrayUI.topK.render(appLocale)) {
                Stepper(
                    value: Binding(
                        get: { draft.defaultTopK },
                        set: { draft.defaultTopK = $0; schedulePush() }),
                    in: 0...200) {
                    Text(draft.defaultTopK == 0 ? "off" : "\(draft.defaultTopK)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            LabeledField(L10n.TrayUI.maxTokens.render(appLocale)) {
                Stepper(
                    value: Binding(
                        get: { draft.defaultMaxTokens },
                        set: { draft.defaultMaxTokens = $0; schedulePush() }),
                    in: 16...131072, step: 256) {
                    Text("\(draft.defaultMaxTokens)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            Toggle(
                "Enable thinking (reasoning)",
                isOn: Binding(
                    get: { draft.defaultEnableThinking ?? false },
                    set: { draft.defaultEnableThinking = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
        }
    }

    // MARK: - Cache section

    private var cacheSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            // §354 — single source of truth for KV cache kind. The old
            // separate "TurboQuant KV cache" Bool toggle was redundant
            // with the picker below (SettingsStore derived the Bool
            // from the picker anyway). Removed. TurboQuant is the
            // default. Bits stepper renders only when picker selects
            // turboquant.
            LabeledField(L10n.TrayUI.kvCache.render(appLocale)) {
                Picker("", selection: Binding(
                    get: { draft.kvCacheQuantization },
                    set: { draft.kvCacheQuantization = $0; schedulePush() })) {
                    Text("TurboQuant").tag("turboquant")
                    Text("Q8").tag("q8")
                    Text("Q4").tag("q4")
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .font(.system(size: 11))
            }
            if draft.kvCacheQuantization.lowercased() == "turboquant" {
                LabeledField(L10n.TrayUI.tqBits.render(appLocale)) {
                    Stepper(
                        value: Binding(
                            get: { draft.turboQuantBits },
                            set: { draft.turboQuantBits = $0; schedulePush() }),
                        in: 3...8) {
                        Text("\(draft.turboQuantBits)")
                            .font(.system(size: 11, design: .monospaced))
                    }
                }
            }
            // §403 — sliding-window override picker. Auto = honor model
            // config and keep compile-first SWA families on the fast path;
            // Long = force full-context and may slow SWA models; Bounded =
            // hard cap of `slidingWindowSize` for memory.
            LabeledField("Sliding window") {
                Picker("", selection: Binding(
                    get: { draft.slidingWindowMode },
                    set: { draft.slidingWindowMode = $0; schedulePush() })) {
                    Text("Auto").tag("auto")
                    Text("Long").tag("long")
                    Text("Bounded").tag("bounded")
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .font(.system(size: 11))
            }
            if draft.slidingWindowMode.lowercased() == "bounded" {
                LabeledField("SW size") {
                    Stepper(
                        value: Binding(
                            get: { draft.slidingWindowSize },
                            set: { draft.slidingWindowSize = $0; schedulePush() }),
                        in: 256...262144, step: 1024
                    ) {
                        Text("\(draft.slidingWindowSize)")
                            .font(.system(size: 11, design: .monospaced))
                    }
                }
            }
            Toggle(
                L10n.TrayPanel.l2DiskCache.render(appLocale),
                isOn: Binding(
                    get: { draft.enableDiskCache },
                    set: { draft.enableDiskCache = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            if draft.enableDiskCache {
                slider(
                    label: L10n.TrayPanel.diskBudgetGB.render(appLocale),
                    value: Binding(
                        get: { draft.diskCacheMaxGB },
                        set: { draft.diskCacheMaxGB = $0; schedulePush() }),
                    range: 1.0...100.0, step: 1.0,
                    format: "%.0f")
            }
            Toggle(
                "Block disk cache",
                isOn: Binding(
                    get: { draft.enableBlockDiskCache },
                    set: { draft.enableBlockDiskCache = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            if draft.enableBlockDiskCache {
                slider(
                    label: "Block budget GB",
                    value: Binding(
                        get: { draft.blockDiskCacheMaxGB },
                        set: { draft.blockDiskCacheMaxGB = $0; schedulePush() }),
                    range: 1.0...100.0, step: 1.0,
                    format: "%.0f")
            }
            Toggle(
                L10n.TrayPanel.prefixCache.render(appLocale),
                isOn: Binding(
                    get: { draft.enablePrefixCache },
                    set: { draft.enablePrefixCache = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            Toggle(
                L10n.TrayPanel.memoryAwarePrefixCache.render(appLocale),
                isOn: Binding(
                    get: { draft.enableMemoryCache },
                    set: { draft.enableMemoryCache = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            Toggle(
                L10n.TrayPanel.ssmReDerive.render(appLocale),
                isOn: Binding(
                    get: { draft.enableSSMReDerive },
                    set: { draft.enableSSMReDerive = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            slider(
                label: L10n.TrayPanel.cacheMemoryPercent.render(appLocale),
                value: Binding(
                    get: { draft.memoryCachePercent * 100 },
                    set: { draft.memoryCachePercent = $0 / 100; schedulePush() }),
                range: 5...80, step: 1,
                format: "%.0f%%")
            LabeledField(L10n.TrayUI.maxCacheBlocks.render(appLocale)) {
                Stepper(
                    value: Binding(
                        get: { draft.maxCacheBlocks },
                        set: { draft.maxCacheBlocks = $0; schedulePush() }),
                    in: 100...10000, step: 100) {
                    Text("\(draft.maxCacheBlocks)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            LabeledField(L10n.TrayUI.prefillStepSize.render(appLocale)) {
                Stepper(
                    value: Binding(
                        get: { draft.prefillStepSize },
                        set: { draft.prefillStepSize = $0; schedulePush() }),
                    in: 128...8192, step: 128) {
                    Text("\(draft.prefillStepSize)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
        }
    }

    // MARK: - Flash MoE section

    private var moeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(
                L10n.TrayPanel.enableFlashMoE.render(appLocale),
                isOn: Binding(
                    get: { draft.flashMoe },
                    set: { draft.flashMoe = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            if draft.flashMoe {
                slider(
                    label: L10n.TrayPanel.slotBankSize.render(appLocale),
                    value: Binding(
                        get: { Double(draft.flashMoeSlotBank) },
                        set: { draft.flashMoeSlotBank = Int($0); schedulePush() }),
                    range: 16...2048, step: 16,
                    format: "%.0f")
                // §384 — FlashMoE Prefetch picker removed from UI per
                // "no BS placeholders" rule. Only "none" was functional;
                // "temporal" persisted to flashMoePrefetch but
                // applyFlashMoEIfEnabled never consumed it. Field stays
                // in GlobalSettings for forward-compat when warm-up lands.
            }
        }
    }

    // MARK: - Adapter section

    private var adapterSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(L10n.TrayPanel.adapterBlurb.render(appLocale))
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Button(L10n.TrayPanel.openAdapterPanel.render(appLocale)) {
                openAppWindow()
                app.mode = .server
            }
            .buttonStyle(.plain)
            .font(.system(size: 11, weight: .medium))
            .foregroundColor(.accentColor)
        }
    }

    // MARK: - Logging settings (UI-7)

    /// Global default verbosity for the Hummingbird HTTP server's
    /// per-request access log AND the inline "Recent logs" tail. Writes
    /// `defaultLogLevel` into GlobalSettings; HTTPServerActor reads the
    /// session-resolved value at start time so the next session bind
    /// inherits the new ceiling without restarting the running server.
    /// Categories map 1:1 to the strings emitted by Engine.log().
    private var loggingSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(L10n.TrayPanel.defaultLevel.render(appLocale))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("", selection: Binding(
                    get: { draft.defaultLogLevel },
                    set: { draft.defaultLogLevel = $0; schedulePush() }
                )) {
                    Text("TRACE").tag("trace")
                    Text("DEBUG").tag("debug")
                    Text("INFO").tag("info")
                    Text("WARN").tag("warn")
                    Text("ERROR").tag("error")
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 110)
            }
            Text("Sets the floor for new HTTP server bindings + the recent-logs tail. The Server tab's LogsPanel already sees TRACE and filters client-side, so changing this won't drop existing scrollback.")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Logs tail

    private var logsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(logTail, id: \.self) { line in
                Text(line)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            if logTail.isEmpty {
                Text(L10n.Tray.noRecentLogs.render(appLocale))
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
            Button(L10n.Tray.openLogs.render(appLocale)) {
                openAppWindow()
                app.mode = .server
            }
            .buttonStyle(.plain)
            .font(.system(size: 11, weight: .medium))
            .foregroundColor(.accentColor)
        }
    }

    // MARK: - Footer

    private var footerRow: some View {
        HStack(spacing: 10) {
            Menu(L10n.Settings.appearance.render(appLocale)) {
                ForEach(AppearanceMode.allCases) { mode in
                    Button(action: { appearanceRaw = mode.rawValue }) {
                        HStack {
                            Text(mode.label)
                            if appearanceRaw == mode.rawValue {
                                Spacer()
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            }
            .font(.system(size: 11))

            // §349 — language picker is part of the tray footer
            // (top-level visible) so users can flip UI language
            // without digging into Settings. Compact variant renders
            // the current language's endonym ("日本語") so the active
            // pick is visible before opening the menu.
            LanguagePickerCompact()

            Spacer()
            Button(L10n.Menu.openVMLX.render(appLocale)) { openAppWindow() }
                .buttonStyle(.plain)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.accentColor)
            Button(L10n.Menu.quit.render(appLocale)) { NSApp.terminate(nil) }
                .buttonStyle(.plain)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.red)
                .keyboardShortcut("q")
        }
    }

    // MARK: - Helpers

    private func openAppWindow() {
        NSApp.activate(ignoringOtherApps: true)
        // Iter 144 — prefer the main app window (no "downloads" /
        // "settings" identifier) so the tray "Open vMLX" doesn't
        // accidentally surface the Downloads window when it's first
        // in NSApp.windows order. Falls back to any canBecomeMain
        // window if no untagged window is found.
        let candidates = NSApp.windows.filter { $0.canBecomeMain }
        let primary = candidates.first { w in
            let id = w.identifier?.rawValue ?? ""
            return !id.contains("downloads") && !id.contains("settings")
        }
        let target = primary ?? candidates.first
        target?.makeKeyAndOrderFront(nil)
    }

    private func slider(
        label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        format: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.system(size: 10, design: .monospaced))
            }
            Slider(value: value, in: range, step: step)
        }
    }

    // MARK: - Live data + settings plumbing

    private func startMetricsPoll() async {
        metricsTask?.cancel()
        let eng = app.engine
        metricsTask = Task { @MainActor [weak app] in
            let stream = await eng.metrics.subscribe()
            var tick = 0
            for await snap in stream {
                if Task.isCancelled { break }
                self.snapshot = snap
                // Refresh the log tail every other tick. `snapshot()`
                // returns newest-last; take the final 5 so the rendered
                // tail shows the most recent activity.
                if let all = await app?.engine.logs.snapshot() {
                    self.logTail = all.suffix(5).map { $0.message }
                }
                // Refresh arch flags every ~4 ticks (~4 s) since model
                // architecture doesn't change between loads — polling
                // faster than the metrics stream would waste cache-stat
                // round-trips.
                if tick % 4 == 0, let e = app?.engine {
                    let stats = (try? await e.cacheStats()) ?? [:]
                    self.archFlags = ArchPillFlags(stats: stats)
                }
                tick &+= 1
            }
        }
    }

    private func loadDraftIfNeeded() async {
        guard !loadedDraft else { return }
        let g = await app.engine.settings.global()
        self.draft = g
        self.loadedDraft = true
    }

    private func schedulePush() {
        pushDebounce?.cancel()
        pushDebounce = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 200_000_000)
            if Task.isCancelled { return }
            let snapshot = self.draft
            await app.engine.settings.setGlobal(snapshot)
        }
    }

    // MARK: - State-derived flags / strings

    private var modelName: String {
        app.selectedModelPath?.lastPathComponent ?? "(no model)"
    }

    /// iter-131 §157: selected session is bound to a remote endpoint.
    /// Tray renders "Remote" instead of the always-.stopped local state,
    /// and hides Start/Stop/Wake/Sleep controls since there's no local
    /// lifecycle to manage.
    private var isRemoteSession: Bool {
        guard let sid = app.selectedServerSessionId else { return false }
        return app.sessions.first(where: { $0.id == sid })?.isRemote ?? false
    }

    private var stateLabel: String {
        // iter-131 §157: see isRemoteSession. The tray is the most
        // "at-a-glance" surface in the app — it must not lie about
        // a remote session's availability by showing "Stopped".
        if isRemoteSession { return "Remote" }
        switch app.engineState {
        case .stopped:        return "Stopped"
        case .loading:        return "Loading…"
        case .running:
            // A7 §259 — both soft and deep remainders shown
            // simultaneously so the user doesn't wonder which clock is
            // ticking. Falls through to the legacy single-kind label
            // when the new dual fields aren't populated (gateway path
            // before the poller has run its first tick).
            let soft = app.idleCountdownSoftSeconds
            let deep = app.idleCountdownDeepSeconds
            if let s = soft, s > 0, let d = deep, d > 0 {
                return String(format: "Running · soft %@ · deep %@", mmss(s), mmss(d))
            }
            if let d = deep, d > 0 {
                return String(format: "Running · deep %@", mmss(d))
            }
            if let s = app.idleCountdownSeconds, s > 0 {
                let kind = app.idleCountdownKindIsDeep ? "Deep sleep" : "Sleeps"
                return String(format: "Running · %@ in %@", kind, mmss(s))
            }
            return "Running"
        case .standby(.soft):
            // Deep countdown is still meaningful after soft has fired —
            // users can see exactly when the model will be unloaded.
            if let d = app.idleCountdownDeepSeconds, d > 0 {
                return String(format: "Light sleep · deep %@", mmss(d))
            }
            return "Light sleep"
        case .standby(.deep): return "Deep sleep"
        case .error:          return "Error"
        }
    }

    /// A7 §259 — format a TimeInterval as M:SS. Handles large sleep
    /// windows (hours) by widening the minute field rather than
    /// silently wrapping.
    private func mmss(_ s: TimeInterval) -> String {
        let total = Int(max(0, s))
        let m = total / 60, r = total % 60
        return String(format: "%d:%02d", m, r)
    }

    /// H1 §271 — thermal state pill. Returns nil for `.nominal` so the
    /// header stays clean when the system isn't throttling.
    @ViewBuilder
    private var thermalPill: some View {
        switch app.thermalLevel {
        case .nominal:
            EmptyView()
        case .fair:
            trayArchPill("THERMAL FAIR", tint: .yellow)
        case .serious:
            trayArchPill("THROTTLING", tint: .orange)
        case .critical:
            trayArchPill("CRITICAL TEMP", tint: .red)
        }
    }

    private var stateColor: Color {
        if isRemoteSession { return .accentColor }
        switch app.engineState {
        case .running:        return .green
        case .loading:        return .orange
        case .stopped:        return .secondary
        case .standby:        return .blue
        case .error:          return .red
        }
    }

    private var canStart: Bool {
        // iter-131 §157: remote sessions have no local "start" — the
        // endpoint is either reachable or not, there's nothing to
        // boot here.
        if isRemoteSession { return false }
        if case .stopped = app.engineState { return true }
        if case .error = app.engineState { return true }
        return false
    }
    private var canStop: Bool {
        if isRemoteSession { return false }
        if case .running = app.engineState { return true }
        if case .standby = app.engineState { return true }
        return false
    }
    private var canRestart: Bool {
        if isRemoteSession { return false }
        if case .running = app.engineState { return true }
        if case .standby = app.engineState { return true }
        return false
    }
    private var canSoftSleep: Bool {
        if isRemoteSession { return false }
        if case .running = app.engineState { return true }
        return false
    }
    private var canDeepSleep: Bool {
        if isRemoteSession { return false }
        if case .running = app.engineState { return true }
        if case .standby(.soft) = app.engineState { return true }
        return false
    }
    private var canWake: Bool {
        if isRemoteSession { return false }
        if case .standby = app.engineState { return true }
        return false
    }

    // MARK: - MenuBarExtra icon

    /// Maps an `EngineState` to an SF symbol. Static so the scene closure in
    /// `vMLXApp` can call it at Scene-construction time without instantiating
    /// the view.
    /// Pill for the tray header — compact, outline-style, matches the
    /// CachePanel pill visual so users see the same signal in two places.
    @ViewBuilder
    private func trayArchPill(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold))
            .foregroundStyle(tint)
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(Capsule().fill(tint.opacity(0.12)))
            .overlay(Capsule().stroke(tint.opacity(0.35), lineWidth: 0.5))
            .accessibilityLabel(text)
            .accessibilityHint("Active cache architecture flag")
            // R2 §303 — plain-English tooltip on hover instead of the
            // bare label. Longpress on trackpad also surfaces this.
            .help(Self.archPillDescription(for: text))
    }

    /// §358 — Gateway status pill. Shows the ACTUAL bound port (green
    /// if matches configured, orange if auto-bumped because the
    /// configured port was taken, red if the bind failed). Replaces
    /// the 3-second flashBanner-only signal, which vanished too fast
    /// for the user to read when another app (Ollama on 8080) was
    /// already holding the port.
    @ViewBuilder
    private var gatewayStatusPill: some View {
        HStack(spacing: 6) {
            switch app.gatewayStatus {
            case .disabled:
                EmptyView()
            case .running(let bound, let requested):
                if bound == requested {
                    Circle().fill(Color.green).frame(width: 6, height: 6)
                    Text("Gateway on :\(bound)")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                } else {
                    Circle().fill(Color.orange).frame(width: 6, height: 6)
                    Text("Gateway bumped :\(requested)→:\(bound)")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(Color.orange)
                        .help(L10n.Tooltip.gatewayPortBumped.format(
                            locale: appLocale,
                            "\(requested)" as NSString,
                            "\(bound)" as NSString,
                            "\(requested)" as NSString))
                }
            case .failed(let port, let msg):
                Circle().fill(Color.red).frame(width: 6, height: 6)
                Text("Gateway offline :\(port)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(Color.red)
                    .help(L10n.Tooltip.gatewayBindFailed.format(
                        locale: appLocale,
                        "\(port)" as NSString,
                        msg as NSString))
            }
        }
    }

    /// R2 §303 — plain-English text for tray arch pill tooltip.
    /// Keep lines short — macOS `.help(_:)` renders a narrow bubble.
    fileprivate static func archPillDescription(for label: String) -> String {
        switch label.lowercased() {
        case "hybrid":
            return "Hybrid SSM: this model alternates state-space (SSM) layers with attention layers. SSM state is rolled into a separate companion cache so prefix cache can still hit across turns."
        case "swa":
            return "Sliding-Window Attention: attention is bounded to a fixed window (e.g. 4096 tokens) so KV memory stays constant regardless of prompt length. Older tokens drop off once the window fills."
        case "tq":
            return "TurboQuant cache: keys/values are quantized on the fly (typically 2-4 bit with calibrated codebooks) so long-context KV fits in a fraction of the memory of fp16. Decode stays on the MXTQ fast path."
        default:
            return "Active cache architecture flag"
        }
    }

    static func icon(for state: EngineState) -> String {
        switch state {
        case .stopped:        return "circle"
        case .loading:        return "circle.dotted"
        case .running:        return "circle.fill"
        // iter-42: distinguish soft vs deep sleep visually. Previously
        // both sleep depths collapsed to `moon`, so a user glancing at
        // the menu bar couldn't tell whether weights were still in
        // memory (soft — quick wake) or dropped (deep — reload needed).
        // `moon` = soft, `moon.zzz.fill` = deep, matching the
        // lifecycle-button icons already used in the popover.
        case .standby(.soft): return "moon"
        case .standby(.deep): return "moon.zzz.fill"
        case .error:          return "exclamationmark.circle"
        }
    }
}

/// Cache architecture pill flags derived from `Engine.cacheStats().architecture`.
/// Populated on the metrics poll loop; empty flags when the model isn't
/// loaded or the architecture dict is missing.
struct ArchPillFlags: Equatable, Sendable {
    var hybrid: Bool = false
    var swa: Bool = false
    var tq: Bool = false

    init() { }

    init(stats: [String: Any]) {
        let arch = (stats["architecture"] as? [String: Any]) ?? [:]
        self.hybrid = (arch["hybridSSMActive"] as? Bool) ?? false
        self.swa = (arch["slidingWindowActive"] as? Bool) ?? false
        self.tq = (arch["turboQuantActive"] as? Bool) ?? false
    }
}

/// Row helper — single-line label on the left, arbitrary control on the right.
private struct LabeledField<Content: View>: View {
    let label: String
    @ViewBuilder let content: () -> Content

    init(_ label: String, @ViewBuilder content: @escaping () -> Content) {
        self.label = label
        self.content = content
    }

    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
            content()
        }
    }
}
