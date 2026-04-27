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
                                  hasContent: !(vm.messages.last?.content.isEmpty ?? true),
                                  // iter-129 §155: pass through the
                                  // selected session's isRemote so the
                                  // banner suppresses "Load Model" on
                                  // remote chats. Resolves via
                                  // selectedServerSessionId → session
                                  // row → isRemote. False for no-
                                  // session or local sessions.
                                  isRemote: app.sessions.first(where: {
                                      $0.id == app.selectedServerSessionId
                                  })?.isRemote ?? false) {
                    // User complaint: the "Load Model" banner button used to
                    // bounce to `app.mode = .server` — a pointless detour
                    // when the chat already knows which model it wants. Try
                    // to DIRECTLY start the chat's selected model first:
                    //   1. find ModelEntry matching chat's `modelAlias`, OR
                    //      fall back to the last-used `selectedModelPath`
                    //   2. find an existing session for that modelPath, OR
                    //      auto-create one with default settings
                    //   3. `startSession` — loads weights + brings up HTTP
                    //      listener in the same step (observer fires so the
                    //      banner flips to Loading → Running live).
                    // Only bounce to Server tab as a LAST resort when the
                    // chat has zero model selected yet (cold first-run).
                    Task { @MainActor in
                        await loadChatModelInline(app: app, vm: vm)
                    }
                } onRetry: {
                    // iter-133 §159: the Retry button on the .error banner
                    // used to just clear `vm.bannerMessage` — but the
                    // banner renders from `app.engineState` (.error(msg)),
                    // NOT from vm.bannerMessage. So clicking Retry
                    // visually did nothing — the engine stayed stuck in
                    // .error, the banner stayed on screen, and the user
                    // had no way to recover except bouncing to Server tab
                    // and hitting Start manually.
                    //
                    // Fix: Retry now attempts a re-load via the same path
                    // the Load Model CTA uses. The new load() transitions
                    // state out of .error through its normal .loading →
                    // .running flow, so the banner flips live. Also clear
                    // any stale vm.bannerMessage so unrelated banners
                    // don't stick around.
                    vm.bannerMessage = nil
                    Task { @MainActor in
                        await loadChatModelInline(app: app, vm: vm)
                    }
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
        // iter-112 §138: screen-level Up-arrow handler REMOVED.
        // InputBar owns history recall when TextField is focused
        // (per-chat `vm.messages`-derived, cycles, escapes to draft).
        // The prior screen-level fallback pointed at a stale cross-
        // session `inputHistory` dict, only recalled once (couldn't
        // cycle — second press hit the `!inputText.isEmpty` guard),
        // and caused messages from chat B to appear when pressing Up
        // in chat A while the TextField wasn't first-responder.
        // The fix is to own history recall exclusively in InputBar,
        // so the two paths can't diverge. User impact: the user now
        // has to click into the input field before Up-arrow recalls —
        // a minor change that's already what every terminal/chat app
        // requires (recall only fires in the actual input surface).
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
    /// iter-129 §155: chat is bound to a remote endpoint. Local engine
    /// state is always `.stopped` for remote sessions — rendering the
    /// "Model is not running / Load Model" banner would lie to the
    /// user. When true, the banner suppresses the `.stopped` and
    /// `.standby` branches since there's no local lifecycle to expose.
    /// Errors still render (remote HTTP failures surface via the
    /// RemoteEngineClient error path → vm.bannerMessage → vm shows
    /// error via MessageList's error bubble rather than this banner).
    var isRemote: Bool = false
    let onLoadModel: () -> Void
    let onRetry: () -> Void

    var body: some View {
        Group {
            // iter-129 §155: remote chats have no local engine state to
            // render — dispatch goes to RemoteEngineClient, not
            // app.engine. Short-circuit to EmptyView so the banner
            // stays out of the way. The `isStreaming` evaluating strip
            // is handled by MessageList for remote-streaming chats.
            if isRemote {
                EmptyView()
            } else {
            switch state {
            case .loading:
                // iter-68 (§97): chat-side banner now surfaces the text
                // percent alongside the visual bar. Before, the banner
                // showed only "Loading model…" + label + bar while the
                // server-tab SessionLoadBar and the tray item both
                // rendered "NN%" explicitly. Banner parity matters
                // because the chat is where users spend most of their
                // load-wait time (they flip to chat once the server
                // session is Starting…) and a determinate % beats a
                // visual-only progress indicator on a long MoE load.
                ColoredBanner(
                    icon: "arrow.down.circle",
                    tint: Theme.Colors.accent,
                    title: loadProgress?.fraction.map {
                        "Loading model… \(Int($0 * 100))%"
                    } ?? "Loading model…",
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
                    title: "Model sleeping — will auto-wake on your next message",
                    cta: ("Wake now", onLoadModel)
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
            } // close else branch of `if isRemote`
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
    @Environment(\.appLocale) private var appLocale: AppLocale
    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            ProgressView()
                .controlSize(.mini)
            // §349 — localized via L10n catalog.
            Text(L10n.Chat.evaluatingPrompt.render(appLocale))
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
            HStack(alignment: .top, spacing: Theme.Spacing.sm) {
                Image(systemName: icon)
                    .foregroundStyle(tint)
                    .padding(.top, 2)
                // iter-138 §213: title was single-line-truncating long
                // `.error(msg)` payloads like "Model directory not
                // found at /very/long/path/... — re-download or point
                // to a different model". HStack without a `maxWidth`
                // constraint + the trailing CTA button squeezed the
                // title to whatever fit on one line; the rest was
                // dropped with no ellipsis (SwiftUI default clips
                // silently when no lineLimit is set but layout is
                // constrained). Fix: `.fixedSize(horizontal: false,
                // vertical: true)` forces the Text to expand
                // vertically rather than clip horizontally, so the
                // banner grows to fit multi-line messages. Plus
                // textSelection so users can copy the message into
                // a support issue — same rationale as §212
                // SessionView banner.
                Text(title)
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textHigh)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
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
    @Environment(\.appLocale) private var appLocale: AppLocale
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
                // §349 — localized "Reasoning" label.
                Text(L10n.Chat.reasoning.render(appLocale))
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
    @Environment(\.appLocale) private var appLocale: AppLocale
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

    /// §245 — disambiguate when two discovered entries share a
    /// `displayName`. Two copies of the same HF repo in different
    /// org/ directories (common when users have both HF cache and a
    /// mlxstudio custom dir) would render as identical rows in the
    /// picker and the alias string (which is displayName) would pick
    /// whichever sorted first. Append the parent-directory in
    /// parentheses for the duplicate cohort so the user can tell
    /// them apart; the alias persisted to DB uses this
    /// disambiguated string too so selection is deterministic.
    private func labelForEntry(_ e: ModelLibrary.ModelEntry) -> String {
        let dupeCount = entries.filter { $0.displayName == e.displayName }.count
        guard dupeCount > 1 else { return e.displayName }
        let parent = e.canonicalPath.deletingLastPathComponent().lastPathComponent
        return "\(e.displayName) (\(parent))"
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
                    // §349 — localized empty-state label.
                    Text(L10n.Chat.noModelsDiscovered.render(appLocale))
                        .foregroundStyle(Theme.Colors.textLow)
                } else {
                    TextField(L10n.Misc.filterModels.render(appLocale), text: $filterQuery)
                        .textFieldStyle(.plain)
                        .font(Theme.Typography.caption)
                    Divider()

                    let shown = filteredEntries
                    if shown.isEmpty {
                        Text(L10n.ChatUI.noModelsMatch.format(locale: appLocale, filterQuery as NSString))
                            .foregroundStyle(Theme.Colors.textLow)
                    } else {
                        ForEach(shown, id: \.id) { e in
                            Menu {
                                Button(L10n.Misc.selectForChat.render(appLocale)) {
                                    Task { await select(labelForEntry(e)) }
                                }
                                let s = loadState(for: e)
                                switch s {
                                case .running, .loading, .standby:
                                    Button(L10n.Misc.stopUnload.render(appLocale)) {
                                        Task { await stopModel(for: e) }
                                    }
                                case .stopped, .absent:
                                    Button(L10n.Misc.startLoad.render(appLocale)) {
                                        Task { await startModel(for: e) }
                                    }
                                }
                                Divider()
                                Button(L10n.Misc.showInServer.render(appLocale)) {
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
                                    Text(labelForEntry(e))
                                    Spacer()
                                    if labelForEntry(e) == currentAlias {
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
                    Button(L10n.Misc.manageInServer.render(appLocale)) {
                        app.mode = .server
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    // §425 — when no model is picked, hide the state dot
                    // (it's confusing to show a grey dot for "absent")
                    // and show a folder/picker icon instead so the user
                    // sees this is a chooser.
                    if currentAlias.isEmpty {
                        Image(systemName: "rectangle.and.text.magnifyingglass")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.Colors.accent)
                    } else {
                        Circle()
                            .fill(stateColor(currentEntryState()))
                            .frame(width: 8, height: 8)
                    }
                    Text(currentAlias.isEmpty ? "Select model →" : currentAlias)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .frame(maxWidth: 180, alignment: .leading)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 9))
                        .foregroundStyle(Theme.Colors.textLow)
                }
                // §425 — when no model is picked, render in accent color
                // + bold caption so the picker feels like a primary call-
                // to-action rather than an inert label. Once a model is
                // selected the standard textMid/caption styling kicks in.
                .foregroundStyle(currentAlias.isEmpty ? Theme.Colors.accent : Theme.Colors.textMid)
                .font(Theme.Typography.caption)
                .fontWeight(currentAlias.isEmpty ? .semibold : .regular)
                .padding(.horizontal, Theme.Spacing.sm)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(currentAlias.isEmpty
                              ? Theme.Colors.accent.opacity(0.10)
                              : Theme.Colors.surface)
                )
                .overlay(
                    // Dashed accent stroke when empty, signalling
                    // "click here to pick" instead of a flat surface.
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(
                            currentAlias.isEmpty
                                ? Theme.Colors.accent.opacity(0.6)
                                : Color.clear,
                            style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
                )
            }
            .menuStyle(.borderlessButton)
            .help(currentAlias.isEmpty
                  ? "Pick a model — then click Load to start the engine."
                  : L10n.Tooltip.modelPicker.render(appLocale))

            // Always-visible Load/Unload button. Previously this was only
            // rendered when `currentEntry()` was non-nil, which meant fresh
            // installs (no chat alias + no selectedModelPath) had no
            // visible Load Model control in the top bar — the user had to
            // wait for the banner to render in `.stopped` state and there
            // was no control at all for `.standby(.deep)`. Now we render
            // the button unconditionally; when no model is picked yet it
            // stays disabled with a hint tooltip, nudging the user to the
            // picker on its left.
            let currentState = currentEntryState()
            Button {
                guard let current = currentEntry() else { return }
                Task {
                    switch currentState {
                    case .running, .loading, .standby:
                        await stopModel(for: current)
                    case .stopped, .absent:
                        await startModel(for: current)
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: iconName(for: currentState))
                        .font(.system(size: 11))
                    Text(actionLabel(for: currentState))
                        .font(Theme.Typography.caption)
                }
                .foregroundStyle(iconColor(for: currentState))
                .padding(.horizontal, Theme.Spacing.sm)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Theme.Colors.surface)
                )
            }
            .buttonStyle(.plain)
            .disabled(currentEntry() == nil)
            .help(currentEntry() == nil
                  ? "Pick a model in the menu on the left first"
                  : buttonTooltip(for: currentState))
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

    private func actionLabel(for s: LoadState) -> String {
        switch s {
        case .running: return "Unload"
        case .loading: return "Loading…"
        case .standby: return "Wake"
        case .stopped, .absent: return "Load Model"
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

/// File-private helper used by the chat-screen banner's "Load Model"
/// CTA. Previously the CTA just called `app.mode = .server` — a
/// pointless tab bounce when the chat already knows which model it
/// wants. Now it tries to:
///
///   1. Resolve the chat's `modelAlias` → matching `ModelLibrary.ModelEntry`.
///   2. Reuse an existing session for that model path, OR auto-create
///      a new session with default settings if it's the first time.
///   3. Call `AppState.startSession(id)` which loads weights + starts
///      the HTTP listener. The per-session observer flips the banner
///      through `.loading(…)` → `.running` live, so no manual refresh
///      is needed.
///
/// Only bounces to Server tab in the truly cold case where the user
/// has neither a chat-level alias nor a last-used `selectedModelPath`
/// and no entries have been discovered yet.
@MainActor
private func loadChatModelInline(app: AppState, vm: ChatViewModel) async {
    // Prefer the chat's currently-configured model. `currentAlias` is
    // set by the ChatModelPicker when the user picks a model row;
    // falling back to `selectedModelPath` covers the "just loaded via
    // cmd+k" + "fresh first-run" cases.
    var targetAlias: String? = nil
    if let chatId = vm.activeSessionId,
       let chat = await app.engine.settings.chat(chatId)
    {
        targetAlias = chat.modelAlias
    }
    let entries = await app.engine.modelLibrary.entries()
    var target: ModelLibrary.ModelEntry? = nil
    if let alias = targetAlias, !alias.isEmpty {
        target = entries.first { $0.displayName == alias }
    }
    if target == nil, let path = app.selectedModelPath {
        let canonical = path.standardizedFileURL.resolvingSymlinksInPath()
        target = entries.first {
            $0.canonicalPath.standardizedFileURL.resolvingSymlinksInPath() == canonical
        }
    }
    guard let entry = target else {
        // No chat alias, no selectedModelPath — genuinely first-run.
        // Bounce to Server tab so the user can pick + configure.
        app.flashBanner("Pick a model from the chat's model menu above, or add one in the Server tab")
        app.mode = .server
        return
    }
    // Reuse existing session if we already have one for this path; the
    // session's saved settings (port, host, quant, prefill step, etc.)
    // carry forward. Auto-create a fresh one (default settings +
    // CapabilityDetector auto-detection) on first use — user's repeated
    // ask: "user should be able to load model from the last saved
    // session of that model's settings, or just start new with auto
    // detect if new model first time".
    let sid: UUID
    if let existing = app.sessionId(forModelPath: entry.canonicalPath) {
        sid = existing
    } else {
        sid = await app.createSession(forModel: entry.canonicalPath)
    }
    // Fast-path: if the engine is ALREADY in standby (idle-fired or
    // manually slept), calling `wakeFromStandby` is much cheaper than
    // `startSession` — no re-registration with the gateway, no HTTP
    // listener respin, and for soft-standby no weight reload. The
    // banner's "Wake now" CTA routes through this inline helper, so
    // picking the cheap path matters for perceived responsiveness.
    // Fall back to full startSession for `.stopped` / `.error` / fresh
    // sessions, which need the end-to-end init.
    let eng = app.engine(for: sid)
    if case .standby = eng.state {
        await eng.wakeFromStandby()
        // `rebindEngineObserver` so the global state tracker picks up
        // the transition immediately (normally startSession does this).
        app.selectedServerSessionId = sid
        app.rebindEngineObserver()
    } else {
        await app.startSession(sid)
    }
}
