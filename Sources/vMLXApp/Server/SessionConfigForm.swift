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
    @State private var openAdvanced    = false
    @State private var openLogging     = false

    // Global defaults snapshot — shown as placeholder text behind each field
    // so the user sees what value will be inherited if they leave it empty.
    @State private var globalDefaults: GlobalSettings = GlobalSettings()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                Text("SESSION CONFIG")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)

                modelSection
                divider
                disclosure("Engine", isOn: $openEngine) { engineSection }
                divider
                disclosure("Cache", isOn: $openCache) { cacheSection }
                divider
                disclosure("Inference defaults", isOn: $openInference) { inferenceSection }
                divider
                disclosure("Lifecycle", isOn: $openLifecycle) { lifecycleSection }
                divider
                disclosure("Server", isOn: $openServer) { serverSection }
                divider
                disclosure("Advanced", isOn: $openAdvanced) { advancedSection }
                divider
                disclosure("Logging", isOn: $openLogging) { loggingSection }
            }
            .font(Theme.Typography.body)
            .foregroundStyle(Theme.Colors.textHigh)
            .padding(.bottom, Theme.Spacing.lg)
        }
        .frame(maxHeight: 520)
        .task { await load() }
    }

    private var divider: some View {
        Divider()
            .overlay(Theme.Colors.border)
            .padding(.vertical, Theme.Spacing.xs)
    }

    // MARK: - Sections

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Text("MODEL")
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
        ValidatedField(title: "Cache memory %",
                       value: doubleBinding(\.cacheMemoryPercent,
                                            default: globalDefaults.cacheMemoryPercent,
                                            transformGet: { $0 * 100 },
                                            transformSet: { $0 / 100 }),
                       range: 0...100, step: 1, format: "%.0f%%")
        ValidatedField(title: "Max cache blocks",
                       value: intBinding(\.maxCacheBlocks, default: globalDefaults.maxCacheBlocks),
                       range: 16...10000, step: 16)
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

        toggleRow("TurboQuant",       boolBinding(\.enableTurboQuant,      default: globalDefaults.enableTurboQuant))
        toggleRow("Prefix cache",     boolBinding(\.enablePrefixCache,     default: globalDefaults.enablePrefixCache))
        toggleRow("SSM companion",    boolBinding(\.enableSSMCompanion,    default: globalDefaults.enableSSMCompanion))
        toggleRow("Block disk cache", boolBinding(\.enableBlockDiskCache,  default: globalDefaults.enableBlockDiskCache))
        toggleRow("L2 disk cache",    boolBinding(\.enableDiskCache,       default: globalDefaults.enableDiskCache))
        toggleRow("JANG repack",      boolBinding(\.enableJANG,            default: globalDefaults.enableJANG))

        Picker("KV cache quantization", selection: Binding(
            get: { s.kvCacheQuantization ?? globalDefaults.kvCacheQuantization },
            set: { s.kvCacheQuantization = $0; commit() }
        )) {
            Text("None").tag("none")
            Text("Q4").tag("q4")
            Text("Q8").tag("q8")
        }
        .pickerStyle(.segmented)
    }

    @ViewBuilder
    private var inferenceSection: some View {
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
    private var serverSection: some View {
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
        toggleRow("LAN (bind 0.0.0.0)",
                  Binding(
                    get: { s.lan ?? globalDefaults.defaultLAN },
                    set: { s.lan = $0; commit() }
                  ))
        textFieldRow("Model alias",
                     placeholder: modelPath.lastPathComponent,
                     value: Binding(
                        get: { s.modelAlias ?? "" },
                        set: { s.modelAlias = $0.isEmpty ? nil : $0; commit() }
                     ))
        textFieldRow("API key",
                     placeholder: "(inherit)",
                     value: Binding(
                        get: { s.apiKey ?? "" },
                        set: { s.apiKey = $0.isEmpty ? nil : $0; commit() }
                     ))
        textFieldRow("Admin token",
                     placeholder: "(inherit)",
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
    }

    @ViewBuilder
    private var advancedSection: some View {
        // Smelt — sub-controls hidden when off. Audit 2026-04-15 finding #4.
        toggleRow("Smelt mode",
                  boolBinding(\.smelt, default: globalDefaults.smelt))
        if (s.smelt ?? globalDefaults.smelt) {
            textFieldRow("Smelt variant",
                         placeholder: globalDefaults.smeltMode,
                         value: Binding(
                            get: { s.smeltMode ?? "" },
                            set: { s.smeltMode = $0.isEmpty ? nil : $0; commit() }
                         ))
        }

        // Flash MoE — slot bank slider hidden when off. Audit finding #1.
        toggleRow("Flash MoE",
                  boolBinding(\.flashMoe, default: globalDefaults.flashMoe))
        if (s.flashMoe ?? globalDefaults.flashMoe) {
            ValidatedField(title: "Flash MoE slot bank",
                           value: intBinding(\.flashMoeSlotBank, default: globalDefaults.flashMoeSlotBank),
                           range: 32...4096, step: 32)
        }

        // Distributed — host / port hidden when off. Audit finding #2.
        toggleRow("Distributed",
                  boolBinding(\.distributed, default: globalDefaults.distributed))
        if (s.distributed ?? globalDefaults.distributed) {
            textFieldRow("Distributed host",
                         placeholder: globalDefaults.distributedHost,
                         value: Binding(
                            get: { s.distributedHost ?? "" },
                            set: { s.distributedHost = $0.isEmpty ? nil : $0; commit() }
                         ))
            ValidatedField(title: "Distributed port",
                           value: intBinding(\.distributedPort, default: globalDefaults.distributedPort),
                           range: 1024...65535, step: 1, format: "%.0f")
        }
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
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        DisclosureGroup(isExpanded: isOn) {
            VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                content()
            }
            .padding(.top, Theme.Spacing.sm)
        } label: {
            Text(title.uppercased())
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
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
    @Binding var path: URL?
    @State private var showImporter = false
    @State private var entries: [ModelLibrary.ModelEntry] = []
    @State private var scanning = false

    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Menu {
                if entries.isEmpty {
                    Text("No models found").foregroundStyle(Theme.Colors.textLow)
                }
                ForEach(entries) { entry in
                    Button {
                        path = entry.canonicalPath
                    } label: {
                        entryLabel(entry)
                    }
                }
                Divider()
                Button("Browse…") { showImporter = true }
                Button("Add custom dir…") { pickUserDir() }
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
        entries = await app.engine.scanModels(force: force)
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
