import SwiftUI
import vMLXEngine
import vMLXTheme

struct ChatScreen: View {
    @Environment(AppState.self) private var app
    @State private var vm = ChatViewModel()

    var body: some View {
        HStack(spacing: 0) {
            SessionsSidebar(vm: vm)
                .frame(width: 260)
                .background(Theme.Colors.surface)
                .overlay(alignment: .trailing) {
                    Rectangle()
                        .fill(Theme.Colors.border)
                        .frame(width: 1)
                }

            VStack(spacing: 0) {
                ChatTopBar(vm: vm)
                Divider().background(Theme.Colors.border)

                EngineStateBanner(state: app.engineState,
                                  loadProgress: app.loadProgress,
                                  isStreaming: vm.isGenerating,
                                  hasContent: !(vm.messages.last?.content.isEmpty ?? true)) {
                    app.mode = .server
                } onRetry: {
                    // Caller hint — ChatViewModel can re-send when the engine
                    // state flips to .running. For now we just clear the error
                    // banner; the user can hit send again.
                    vm.bannerMessage = nil
                }

                MessageList(vm: vm)
                    .frame(maxHeight: .infinity)
                InputBar(vm: vm)
            }
            .frame(maxWidth: .infinity)
            .background(Theme.Colors.background)
        }
        .onAppear {
            vm.attach(app)
            // Publish the vm to AppState so global Cmd-N / Cmd-Shift-T /
            // Cmd-K shortcuts and the command bar can drive it.
            app.chatViewModelRef = vm
        }
        .onDisappear {
            if app.chatViewModelRef === vm { app.chatViewModelRef = nil }
        }
        // Esc stops the current generation. We intentionally attach at
        // the ChatScreen level so any focus state inside the pane (other
        // than the TextField in InputBar, which handles Esc itself first)
        // still sees the keypress. When `isGenerating` is false we return
        // `.ignored` so Esc falls through to default SwiftUI handling
        // (close popovers, etc).
        .onKeyPress(.escape) {
            if vm.isGenerating {
                vm.stop()
                return .handled
            }
            return .ignored
        }
        // Up arrow in an empty input recalls the previous user message.
        .onKeyPress(.upArrow) {
            if vm.inputText.isEmpty && vm.recallPreviousInput() {
                return .handled
            }
            return .ignored
        }
        .background(
            // Cmd-W close current chat. Attached here (not globally) so it
            // only fires while the Chat screen is mounted.
            Button("") {
                if let id = vm.activeSessionId {
                    vm.closeSession(id)
                }
            }
            .keyboardShortcut("w", modifiers: .command)
            .hidden()
            .frame(width: 0, height: 0)
        )
    }
}

/// State-aware banner stack. Renders zero or one banner depending on
/// `EngineState`. Mirrors the React `ChatInterface.tsx` switch on
/// `sessionStatus` (loading / soft-sleep / deep-sleep / stopped / error /
/// running+evaluating).
struct EngineStateBanner: View {
    let state: EngineState
    let loadProgress: LoadProgress?
    let isStreaming: Bool
    let hasContent: Bool
    let onLoadModel: () -> Void
    let onRetry: () -> Void

    var body: some View {
        Group {
            switch state {
            case .loading:
                ColoredBanner(
                    icon: "arrow.down.circle",
                    tint: Theme.Colors.accent,
                    title: "Loading model…",
                    detail: loadProgress?.label,
                    showSpinner: true,
                    extra: AnyView(
                        LoadingBar(fraction: loadProgress?.fraction)
                            .padding(.top, Theme.Spacing.xs)
                    )
                )
            case .standby(.soft):
                ColoredBanner(
                    icon: "moon.zzz",
                    tint: Theme.Colors.warning,
                    title: "Waking up model and restoring caches…",
                    showSpinner: true
                )
            case .standby(.deep):
                ColoredBanner(
                    icon: "moon.stars",
                    tint: Theme.Colors.warning,
                    title: "Model sleeping — will auto-wake on your next message"
                )
            case .stopped:
                ColoredBanner(
                    icon: "exclamationmark.circle",
                    tint: Theme.Colors.textMid,
                    title: "Model is not running",
                    cta: ("Load Model", onLoadModel)
                )
            case .error(let msg):
                ColoredBanner(
                    icon: "xmark.octagon",
                    tint: Theme.Colors.danger,
                    title: msg,
                    cta: ("Retry", onRetry)
                )
            case .running:
                if isStreaming && !hasContent {
                    EvaluatingStrip()
                } else {
                    EmptyView()
                }
            }
        }
        .animation(.easeOut(duration: 0.18), value: stateKey)
    }

    /// Cheap key for `.animation(_, value:)` — we can't use `state` directly
    /// because LoadProgress isn't Hashable.
    private var stateKey: String {
        switch state {
        case .stopped: return "stopped"
        case .loading(let p): return "loading-\(p.phase.rawValue)-\(p.fraction ?? -1)"
        case .running: return isStreaming && !hasContent ? "evaluating" : "running"
        case .standby(let d): return "standby-\(d.rawValue)"
        case .error(let m): return "error-\(m)"
        }
    }
}

/// Tiny "Evaluating prompt…" header strip — NOT a full banner. Shown only
/// while the engine is mid-prefill on a new turn (no tokens yet).
private struct EvaluatingStrip: View {
    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            ProgressView()
                .controlSize(.mini)
            Text("Evaluating prompt…")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
            Spacer()
        }
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.vertical, Theme.Spacing.xs)
        .background(Theme.Colors.surface.opacity(0.6))
    }
}

/// Colored banner with optional spinner, optional CTA button, and an
/// optional extra view slot (used to embed the loading progress bar).
struct ColoredBanner: View {
    let icon: String
    let tint: Color
    let title: String
    var detail: String? = nil
    var showSpinner: Bool = false
    var cta: (String, () -> Void)? = nil
    var extra: AnyView? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack(spacing: Theme.Spacing.sm) {
                Image(systemName: icon)
                    .foregroundStyle(tint)
                Text(title)
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textHigh)
                if showSpinner {
                    ProgressView().controlSize(.mini)
                }
                Spacer()
                if let cta {
                    Button(cta.0, action: cta.1)
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .tint(tint)
                }
            }
            if let detail, !detail.isEmpty {
                Text(detail)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .lineLimit(2)
            }
            if let extra { extra }
        }
        .padding(Theme.Spacing.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(tint.opacity(0.4), lineWidth: 1)
                )
        )
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.top, Theme.Spacing.sm)
    }
}

private struct ChatTopBar: View {
    @Environment(AppState.self) private var app
    @Bindable var vm: ChatViewModel
    @State private var showSettings = false
    // showQuickSliders removed in UX-5 dedup pass — the QuickSlidersPopover
    // duplicated controls already in ChatSettingsPopover. See header HStack.
    /// Persisted appearance — cycled by the top-bar theme button.
    /// Mirror of the same @AppStorage key used in `vMLXApp` and
    /// `TrayItem` so all three surfaces stay in sync.
    @AppStorage("vmlx.appearance") private var appearanceRaw: String = AppearanceMode.dark.rawValue

    var body: some View {
        HStack(spacing: Theme.Spacing.md) {
            Text(currentTitle)
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()

            // In-chat model picker — fixes the "no way to pick a model
            // from Chat" gap the audit flagged as #2 priority. Lists
            // every ModelLibrary entry; writes the selection to the
            // per-chat `modelAlias` so `ChatViewModel.send()` picks it
            // up on the next turn via `engine.settings.chat(chatId)`.
            ChatModelPicker(vm: vm)

            Toggle(isOn: $vm.reasoningEnabled) {
                Text("Reasoning")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            .toggleStyle(.switch)
            .tint(Theme.Colors.accent)

            // Appearance cycle button — click to rotate
            // system → light → dark → system. Hover tooltip shows the
            // current mode's name. Matches the Appearance submenu in
            // the tray (TrayItem.swift) so the two surfaces agree,
            // but surfacing it in the main window removes the
            // tray-hunting step the audit flagged as #2 UX gap.
            Button {
                appearanceRaw = nextAppearance.rawValue
            } label: {
                Image(systemName: currentAppearance.iconName)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            .buttonStyle(.borderless)
            .help("Appearance: \(currentAppearance.label) — click to cycle")

            // Chat settings gear — opens ChatSettingsPopover with
            // inheritance preview. Disabled when no active chat.
            // The QuickSlidersPopover thermometer button used to live
            // here as a "fast access to temperature/top-p" shortcut,
            // but it duplicated the same controls from
            // `ChatSettingsPopover` (which already has every knob).
            // Removed in UX-5 dedup pass — the chat header is now a
            // single-source-of-truth gear button to keep users from
            // wondering why the same slider exists in two places. The
            // global defaults still live in the menu bar TrayItem.
            Button {
                if vm.activeSessionId != nil { showSettings.toggle() }
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .foregroundStyle(Theme.Colors.textMid)
            }
            .buttonStyle(.borderless)
            .disabled(vm.activeSessionId == nil)
            .popover(isPresented: $showSettings, arrowEdge: .bottom) {
                if let cid = vm.activeSessionId {
                    ChatSettingsPopover(chatId: cid, sessionId: nil)
                }
            }
        }
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.vertical, Theme.Spacing.md)
        .background(Theme.Colors.background)
    }

    private var currentAppearance: AppearanceMode {
        AppearanceMode(rawValue: appearanceRaw) ?? .dark
    }

    /// Cycle order: system → light → dark → system.
    /// Matches the `AppearanceMode.allCases` order defined in
    /// `vMLXApp.swift` so the button and tray menu stay in sync.
    private var nextAppearance: AppearanceMode {
        switch currentAppearance {
        case .auto:  return .light
        case .light: return .dark
        case .dark:  return .auto
        }
    }

    private var currentTitle: String {
        if let id = vm.activeSessionId,
           let s = vm.sessions.first(where: { $0.id == id }) {
            return s.title
        }
        return "Chat"
    }
}

/// Compact model picker for the ChatTopBar.
///
/// Lists every entry returned by `engine.modelLibrary.entries()` and
/// writes the selection to `engine.settings.chat(chatId).modelAlias`.
/// `ChatViewModel.send()` already reads this field at request-build
/// time (`ChatViewModel.swift:331`), so the picker immediately takes
/// effect on the next turn without any additional plumbing.
///
/// If no chat is active (empty state) the picker still lets the user
/// pick a default that gets written as a global alias. If the model
/// isn't loaded yet, clicking a different model triggers the load
/// via the existing Server-tab flow — we just flash a banner and nudge
/// the user to the Server tab (same pattern as the current
/// `ChatViewModel.send()` guard).
private struct ChatModelPicker: View {
    @Environment(AppState.self) private var app
    @Bindable var vm: ChatViewModel

    @State private var entries: [ModelLibrary.ModelEntry] = []
    @State private var currentAlias: String = ""
    /// Substring filter applied to the picker menu. Empty string =
    /// show everything. Case-insensitive. Populated by a text field
    /// that lives at the top of the menu; SwiftUI `Menu` supports
    /// arbitrary view contents in a custom label/body, so we embed
    /// a `TextField` row that keeps focus while the user types.
    @State private var filterQuery: String = ""

    /// Entries passing the current filter. When the query is blank
    /// this is a reference to `entries`; when non-blank it matches
    /// against `displayName` case-insensitively plus `family` so a
    /// user typing "qwen" finds Qwen3MoE, Qwen3.5-VL, etc.
    private var filteredEntries: [ModelLibrary.ModelEntry] {
        let q = filterQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return entries }
        let qLower = q.lowercased()
        return entries.filter { e in
            e.displayName.lowercased().contains(qLower)
                || e.family.lowercased().contains(qLower)
        }
    }

    /// Repeated user ask: "there needs to be a way to directly easily
    /// start / stop models from the chat page and list without needing
    /// to go to the server page". Each menu row now carries a load-state
    /// dot + an inline ▶ / ■ button that calls `app.startSession` /
    /// `app.stopSession`. The picker label also shows the state of the
    /// CURRENT chat's model so you can tell at a glance whether the
    /// next send will hit a warm model or block on load.
    private enum LoadState { case running, loading, standby, stopped, absent }

    private func loadState(for entry: ModelLibrary.ModelEntry) -> LoadState {
        let canonical = entry.canonicalPath.standardizedFileURL.resolvingSymlinksInPath()
        guard let session = app.sessions.first(where: {
            $0.modelPath.standardizedFileURL.resolvingSymlinksInPath() == canonical
        }) else {
            return .absent
        }
        switch session.state {
        case .running: return .running
        case .loading: return .loading
        case .standby: return .standby
        case .stopped, .error: return .stopped
        }
    }

    private func stateColor(_ s: LoadState) -> Color {
        switch s {
        case .running: return .green
        case .loading: return .yellow
        case .standby: return .orange
        case .stopped: return Theme.Colors.textLow
        case .absent:  return Theme.Colors.textLow
        }
    }

    private func stateLabel(_ s: LoadState) -> String {
        switch s {
        case .running: return "In RAM"
        case .loading: return "Loading…"
        case .standby: return "Standby"
        case .stopped: return "Not loaded"
        case .absent:  return "No session"
        }
    }

    var body: some View {
        HStack(spacing: Theme.Spacing.xs) {
            // Primary picker menu — filter + select + per-row ▶/■
            Menu {
                if entries.isEmpty {
                    Text("No models discovered")
                        .foregroundStyle(Theme.Colors.textLow)
                } else {
                    TextField("Filter models…", text: $filterQuery)
                        .textFieldStyle(.plain)
                        .font(Theme.Typography.caption)
                    Divider()

                    let shown = filteredEntries
                    if shown.isEmpty {
                        Text("No models match `\(filterQuery)`")
                            .foregroundStyle(Theme.Colors.textLow)
                    } else {
                        ForEach(shown, id: \.id) { e in
                            Menu {
                                Button("Select for this chat") {
                                    Task { await select(e.displayName) }
                                }
                                let s = loadState(for: e)
                                switch s {
                                case .running, .loading, .standby:
                                    Button("Stop / unload from RAM") {
                                        Task { await stopModel(for: e) }
                                    }
                                case .stopped, .absent:
                                    Button("Start / load into RAM") {
                                        Task { await startModel(for: e) }
                                    }
                                }
                                Divider()
                                Button("Show in Server tab") {
                                    if let sid = app.sessionId(forModelPath: e.canonicalPath) {
                                        app.selectedServerSessionId = sid
                                    }
                                    app.mode = .server
                                }
                            } label: {
                                HStack(spacing: Theme.Spacing.sm) {
                                    Circle()
                                        .fill(stateColor(loadState(for: e)))
                                        .frame(width: 8, height: 8)
                                    Text(e.displayName)
                                    Spacer()
                                    if e.displayName == currentAlias {
                                        Image(systemName: "checkmark")
                                    }
                                    Text(stateLabel(loadState(for: e)))
                                        .font(Theme.Typography.caption)
                                        .foregroundStyle(Theme.Colors.textLow)
                                }
                            }
                        }
                    }
                    Divider()
                    Button("Manage in Server tab") {
                        app.mode = .server
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Circle()
                        .fill(stateColor(currentEntryState()))
                        .frame(width: 8, height: 8)
                    Text(currentAlias.isEmpty ? "Select model" : currentAlias)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .frame(maxWidth: 180, alignment: .leading)
                }
                .foregroundStyle(Theme.Colors.textMid)
                .font(Theme.Typography.caption)
                .padding(.horizontal, Theme.Spacing.sm)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Theme.Colors.surface)
                )
            }
            .menuStyle(.borderlessButton)
            .help("Pick a model · colored dot = load state. Right side ▶ / ■ toggles the model in/out of RAM without leaving the Chat tab.")

            // Inline ▶ / ■ for the CURRENT chat's selected model. One
            // click = start if stopped, stop if running. Matches the
            // repeated user ask.
            if let current = currentEntry() {
                let s = loadState(for: current)
                Button {
                    Task {
                        switch s {
                        case .running, .loading, .standby:
                            await stopModel(for: current)
                        case .stopped, .absent:
                            await startModel(for: current)
                        }
                    }
                } label: {
                    Image(systemName: iconName(for: s))
                        .font(.system(size: 11))
                        .foregroundStyle(iconColor(for: s))
                        .padding(4)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Theme.Colors.surface)
                        )
                }
                .buttonStyle(.plain)
                .help(buttonTooltip(for: s))
            }
        }
        .task {
            await refresh()
        }
        .task(id: vm.activeSessionId) {
            await loadCurrentAlias()
        }
    }

    private func iconName(for s: LoadState) -> String {
        switch s {
        case .running: return "stop.fill"
        case .loading: return "hourglass"
        case .standby: return "play.fill"
        case .stopped, .absent: return "play.fill"
        }
    }

    private func iconColor(for s: LoadState) -> Color {
        switch s {
        case .running: return .red
        case .loading: return .yellow
        case .standby: return .orange
        case .stopped, .absent: return .green
        }
    }

    private func buttonTooltip(for s: LoadState) -> String {
        switch s {
        case .running: return "Unload this model from RAM"
        case .loading: return "Loading in progress — click to cancel and unload"
        case .standby: return "Wake / reload from standby"
        case .stopped: return "Load this model into RAM"
        case .absent:  return "Create a session and load this model"
        }
    }

    private func currentEntry() -> ModelLibrary.ModelEntry? {
        guard !currentAlias.isEmpty else { return nil }
        return entries.first(where: { $0.displayName == currentAlias })
    }

    private func currentEntryState() -> LoadState {
        currentEntry().map(loadState(for:)) ?? .absent
    }

    @MainActor
    private func refresh() async {
        entries = await app.engine.modelLibrary.entries()
        if entries.isEmpty {
            // First-run: force a scan so fresh downloads show up.
            _ = await app.engine.modelLibrary.scan(force: false)
            entries = await app.engine.modelLibrary.entries()
        }
        await loadCurrentAlias()
    }

    @MainActor
    private func loadCurrentAlias() async {
        guard let chatId = vm.activeSessionId else {
            currentAlias = app.selectedModelPath?.lastPathComponent ?? ""
            return
        }
        if let existing = await app.engine.settings.chat(chatId)?.modelAlias {
            currentAlias = existing
        } else {
            currentAlias = app.selectedModelPath?.lastPathComponent ?? ""
        }
    }

    @MainActor
    private func select(_ alias: String) async {
        currentAlias = alias
        guard let chatId = vm.activeSessionId else {
            vm.bannerMessage = "No active chat — open or create a chat first."
            return
        }
        // Persist the alias on the chat-level settings row so future
        // turns pick it up. ChatViewModel.send() already reads this.
        var chat = await app.engine.settings.chat(chatId) ?? .init()
        chat.modelAlias = alias
        await app.engine.settings.setChat(chatId, chat)
    }

    /// Start / load the model associated with `entry`. Creates a
    /// session row on the fly if the user picked a model that hasn't
    /// been added to the Server tab yet. No-op if the session is
    /// already running (AppState.startSession is idempotent).
    @MainActor
    private func startModel(for entry: ModelLibrary.ModelEntry) async {
        let sid: UUID
        if let existing = app.sessionId(forModelPath: entry.canonicalPath) {
            sid = existing
        } else {
            sid = await app.createSession(forModel: entry.canonicalPath)
        }
        await app.startSession(sid)
    }

    /// Stop / unload the model associated with `entry`. No-op if no
    /// session exists for the model.
    @MainActor
    private func stopModel(for entry: ModelLibrary.ModelEntry) async {
        guard let sid = app.sessionId(forModelPath: entry.canonicalPath) else { return }
        await app.stopSession(sid)
    }
}
