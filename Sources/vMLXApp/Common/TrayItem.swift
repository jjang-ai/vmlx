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
                    title: "Server binding", open: $showServer, icon: "network"
                ) { serverSection }
                disclosureGroup(
                    title: "Sampling (global defaults)", open: $showRuntime, icon: "slider.horizontal.3"
                ) { runtimeSection }
                disclosureGroup(
                    title: "Cache", open: $showCache, icon: "externaldrive.fill"
                ) { cacheSection }
                disclosureGroup(
                    title: "Flash MoE", open: $showMoE, icon: "bolt.fill"
                ) { moeSection }
                disclosureGroup(
                    title: "Adapter", open: $showAdapter, icon: "puzzlepiece.fill"
                ) { adapterSection }
                disclosureGroup(
                    title: "Logging", open: $showLogging, icon: "text.alignleft"
                ) { loggingSection }
                disclosureGroup(
                    title: "Recent logs", open: $showLogs, icon: "doc.text"
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
        .onAppear { Task { await loadDraftIfNeeded() } }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: Self.icon(for: app.engineState))
                .foregroundColor(stateColor)
                .font(.system(size: 18, weight: .semibold))
            VStack(alignment: .leading, spacing: 2) {
                Text(stateLabel)
                    .font(.system(size: 13, weight: .semibold))
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
            Picker("Session", selection: Binding(
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
                lifecycleButton("Start", icon: "play.fill",
                                enabled: canStart, tint: .green) {
                    app.onTrayStartServer()
                }
                lifecycleButton("Stop", icon: "stop.fill",
                                enabled: canStop, tint: .red) {
                    app.onTrayStopServer()
                }
                lifecycleButton("Restart", icon: "arrow.clockwise",
                                enabled: canRestart, tint: .orange) {
                    app.onTrayRestartServer()
                }
            }
            HStack(spacing: 6) {
                lifecycleButton("Soft Sleep", icon: "moon",
                                enabled: canSoftSleep, tint: .blue) {
                    app.onTraySoftSleepServer()
                }
                lifecycleButton("Deep Sleep", icon: "moon.zzz.fill",
                                enabled: canDeepSleep, tint: .purple) {
                    app.onTrayDeepSleepServer()
                }
                lifecycleButton("Wake", icon: "sun.max.fill",
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
            Button("Pick…") {
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
            LabeledField("Host") {
                TextField(
                    "127.0.0.1",
                    text: Binding(
                        get: { draft.defaultHost },
                        set: { draft.defaultHost = $0; schedulePush() }))
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11))
            }
            LabeledField("Port") {
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
                Text("URL")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                Text(serverURL)
                    .font(.system(size: 10, design: .monospaced))
                    .textSelection(.enabled)
                Spacer()
                Button("Copy") {
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
                LabeledField("Gateway port") {
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
            }

            // UI-11: Rate limit. 0 = unlimited. When > 0, RateLimitMiddleware
            // enforces per-IP requests/minute on every bound listener
            // (per-session servers AND the gateway). Change takes effect
            // at the next listener start.
            Divider().padding(.vertical, 4)
            LabeledField("Rate limit (req/min/IP)") {
                Stepper(
                    value: Binding(
                        get: { draft.rateLimit },
                        set: { draft.rateLimit = max(0, $0); schedulePush() }),
                    in: 0...10000, step: 10) {
                    Text(draft.rateLimit == 0 ? "unlimited" : "\(draft.rateLimit)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            Text("0 = unlimited. Applies to every new HTTP listener. Active listeners pick up on next (re)start.")
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
            LabeledField("Top-K") {
                Stepper(
                    value: Binding(
                        get: { draft.defaultTopK },
                        set: { draft.defaultTopK = $0; schedulePush() }),
                    in: 0...200) {
                    Text(draft.defaultTopK == 0 ? "off" : "\(draft.defaultTopK)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            LabeledField("Max tokens") {
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
            Toggle(
                "TurboQuant KV cache",
                isOn: Binding(
                    get: { draft.enableTurboQuant },
                    set: { draft.enableTurboQuant = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            if draft.enableTurboQuant {
                LabeledField("TQ bits") {
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
            Toggle(
                "L2 disk cache",
                isOn: Binding(
                    get: { draft.enableDiskCache },
                    set: { draft.enableDiskCache = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            if draft.enableDiskCache {
                slider(
                    label: "Disk budget (GB)",
                    value: Binding(
                        get: { draft.diskCacheMaxGB },
                        set: { draft.diskCacheMaxGB = $0; schedulePush() }),
                    range: 1.0...100.0, step: 1.0,
                    format: "%.0f")
            }
            Toggle(
                "Prefix cache",
                isOn: Binding(
                    get: { draft.enablePrefixCache },
                    set: { draft.enablePrefixCache = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            Toggle(
                "Memory-aware prefix cache",
                isOn: Binding(
                    get: { draft.enableMemoryCache },
                    set: { draft.enableMemoryCache = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            Toggle(
                "SSM re-derive (hybrid+thinking)",
                isOn: Binding(
                    get: { draft.enableSSMReDerive },
                    set: { draft.enableSSMReDerive = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            slider(
                label: "Cache memory %",
                value: Binding(
                    get: { draft.memoryCachePercent * 100 },
                    set: { draft.memoryCachePercent = $0 / 100; schedulePush() }),
                range: 5...80, step: 1,
                format: "%.0f%%")
            LabeledField("Max cache blocks") {
                Stepper(
                    value: Binding(
                        get: { draft.maxCacheBlocks },
                        set: { draft.maxCacheBlocks = $0; schedulePush() }),
                    in: 100...10000, step: 100) {
                    Text("\(draft.maxCacheBlocks)")
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            LabeledField("Prefill step size") {
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
                "Enable Flash MoE (stream experts from SSD)",
                isOn: Binding(
                    get: { draft.flashMoe },
                    set: { draft.flashMoe = $0; schedulePush() }))
                .font(.system(size: 11))
                .toggleStyle(.switch)
            if draft.flashMoe {
                slider(
                    label: "Slot bank size",
                    value: Binding(
                        get: { Double(draft.flashMoeSlotBank) },
                        set: { draft.flashMoeSlotBank = Int($0); schedulePush() }),
                    range: 16...2048, step: 16,
                    format: "%.0f")
                Picker(
                    "Prefetch",
                    selection: Binding(
                        get: { draft.flashMoePrefetch },
                        set: { draft.flashMoePrefetch = $0; schedulePush() })
                ) {
                    Text("None").tag("none")
                    Text("Temporal").tag("temporal")
                }
                .pickerStyle(.segmented)
                .font(.system(size: 10))
            }
        }
    }

    // MARK: - Adapter section

    private var adapterSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Active LoRA adapter is managed via Server → Adapter panel.")
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Button("Open Adapter panel") {
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
                Text("Default level")
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
                Text("No recent logs.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
            Button("Open Logs") {
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
        HStack {
            Menu("Appearance") {
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
            Spacer()
            Button("Open vMLX") { openAppWindow() }
                .buttonStyle(.plain)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.accentColor)
            Button("Quit") { NSApp.terminate(nil) }
                .buttonStyle(.plain)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.red)
                .keyboardShortcut("q")
        }
    }

    // MARK: - Helpers

    private func openAppWindow() {
        NSApp.activate(ignoringOtherApps: true)
        for w in NSApp.windows where w.canBecomeMain {
            w.makeKeyAndOrderFront(nil)
            break
        }
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
            for await snap in stream {
                if Task.isCancelled { break }
                self.snapshot = snap
                // Refresh the log tail every other tick. `snapshot()`
                // returns newest-last; take the final 5 so the rendered
                // tail shows the most recent activity.
                if let all = await app?.engine.logs.snapshot() {
                    self.logTail = all.suffix(5).map { $0.message }
                }
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

    private var stateLabel: String {
        switch app.engineState {
        case .stopped:        return "Stopped"
        case .loading:        return "Loading…"
        case .running:        return "Running"
        case .standby(.soft): return "Light sleep"
        case .standby(.deep): return "Deep sleep"
        case .error:          return "Error"
        }
    }

    private var stateColor: Color {
        switch app.engineState {
        case .running:        return .green
        case .loading:        return .orange
        case .stopped:        return .secondary
        case .standby:        return .blue
        case .error:          return .red
        }
    }

    private var canStart: Bool {
        if case .stopped = app.engineState { return true }
        if case .error = app.engineState { return true }
        return false
    }
    private var canStop: Bool {
        if case .running = app.engineState { return true }
        if case .standby = app.engineState { return true }
        return false
    }
    private var canRestart: Bool {
        if case .running = app.engineState { return true }
        if case .standby = app.engineState { return true }
        return false
    }
    private var canSoftSleep: Bool {
        if case .running = app.engineState { return true }
        return false
    }
    private var canDeepSleep: Bool {
        if case .running = app.engineState { return true }
        if case .standby(.soft) = app.engineState { return true }
        return false
    }
    private var canWake: Bool {
        if case .standby = app.engineState { return true }
        return false
    }

    // MARK: - MenuBarExtra icon

    /// Maps an `EngineState` to an SF symbol. Static so the scene closure in
    /// `vMLXApp` can call it at Scene-construction time without instantiating
    /// the view.
    static func icon(for state: EngineState) -> String {
        switch state {
        case .stopped:        return "circle"
        case .loading:        return "circle.dotted"
        case .running:        return "circle.fill"
        case .standby:        return "moon"
        case .error:          return "exclamationmark.circle"
        }
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
