import SwiftUI
import vMLXEngine
import vMLXTheme

/// Cmd-K command palette. Pure-SwiftUI popover sheet with fuzzy search over
/// actions (switch chat, switch model, switch mode, toggle reasoning). Driven
/// by `AppState.showCommandBar` which RootView binds to a `.sheet`.
struct CommandBar: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale
    @Environment(\.dismiss) private var dismiss

    @State private var query: String = ""
    @State private var models: [ModelLibrary.ModelEntry] = []
    @State private var selectedIndex: Int = 0
    @FocusState private var focused: Bool

    /// Optional chat view-model so "Switch chat" actions can work. Set by
    /// the caller in `ChatScreen`; when nil, chat actions are omitted.
    var chatVM: ChatViewModel?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.Colors.border)
            list
        }
        .frame(width: 540, height: 420)
        .background(Theme.Colors.surface)
        .task { await loadModels() }
        .onKeyPress(.downArrow) {
            selectedIndex = min(selectedIndex + 1, max(0, actions.count - 1))
            return .handled
        }
        .onKeyPress(.upArrow) {
            selectedIndex = max(0, selectedIndex - 1)
            return .handled
        }
        .onKeyPress(.return) {
            runSelected()
            return .handled
        }
        .onKeyPress(.escape) {
            dismiss()
            return .handled
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(Theme.Colors.textMid)
            TextField(L10n.Misc.typeCommand.render(appLocale), text: $query)
                .textFieldStyle(.plain)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
                .focused($focused)
                .onAppear { focused = true }
                .onChange(of: query) { _, _ in selectedIndex = 0 }
        }
        .padding(Theme.Spacing.md)
    }

    // MARK: - List

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(Array(actions.enumerated()), id: \.offset) { idx, action in
                    row(for: action, idx: idx)
                }
                if actions.isEmpty {
                    Text(L10n.Misc.noMatches.render(appLocale))
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                        .padding(Theme.Spacing.md)
                }
            }
        }
    }

    private func row(for action: Action, idx: Int) -> some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: action.icon)
                .foregroundStyle(Theme.Colors.textMid)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 1) {
                Text(action.title)
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textHigh)
                if let sub = action.subtitle {
                    Text(sub)
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                }
            }
            Spacer()
        }
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, Theme.Spacing.sm)
        .background(
            idx == selectedIndex ? Theme.Colors.surfaceHi : Color.clear
        )
        .contentShape(Rectangle())
        .onTapGesture {
            selectedIndex = idx
            runSelected()
        }
    }

    // MARK: - Actions model

    struct Action: Identifiable {
        let id = UUID()
        let title: String
        let subtitle: String?
        let icon: String
        let run: @MainActor () -> Void
    }

    /// Full unfiltered action list, built from current app state.
    private var allActions: [Action] {
        var out: [Action] = []

        // Modes
        for m in AppState.Mode.allCases {
            let name = m.rawValue
            out.append(.init(
                title: "Switch mode: \(name)",
                subtitle: nil,
                icon: "square.grid.2x2",
                run: { app.mode = m }
            ))
        }

        // Toggle reasoning
        if let vm = chatVM {
            out.append(.init(
                title: vm.reasoningEnabled ? "Disable reasoning" : "Enable reasoning",
                subtitle: nil,
                icon: "brain",
                run: { vm.reasoningEnabled.toggle() }
            ))
            // Chats
            for s in vm.sessions {
                out.append(.init(
                    title: "Switch chat: \(s.title)",
                    subtitle: nil,
                    icon: "bubble.left",
                    run: { vm.selectSession(s.id) }
                ))
            }
            out.append(.init(
                title: "New chat",
                subtitle: nil,
                icon: "plus.bubble",
                run: { vm.newSession() }
            ))
        }

        // Models — BLOCKER #4. Previously this only set
        // `app.selectedModelPath`, which is a UI cursor, not a load
        // command. Cmd+K → "Load model: X" looked successful (palette
        // closed, tray pill flipped) but no engine.load fired, so the
        // user's next chat message hit the "Model not running" banner
        // with no breadcrumb back to the failed action. Now we route
        // through `startSession` (creating a session row if needed)
        // exactly like the chat dropdown's auto-load (BLOCKER #1) and
        // the Server tab's Load Model button. Failures land on the
        // session's `.error(msg)` state, which both `SessionView`
        // (top of the Server tab) and `ChatScreen.ChatStateBanner`
        // already render with retry CTAs, plus `startSession` calls
        // `flashBanner("Engine load failed: …")` for global visibility
        // when the user is on neither tab. Single error pipeline,
        // three surfaces.
        for entry in models {
            out.append(.init(
                title: "Load model: \(entry.displayName)",
                subtitle: entry.family,
                icon: "cpu",
                run: {
                    Task { @MainActor in
                        let sid: UUID
                        if let existing = app.sessionId(forModelPath: entry.canonicalPath) {
                            sid = existing
                        } else {
                            sid = await app.createSession(forModel: entry.canonicalPath)
                        }
                        await app.startSession(sid)
                    }
                }
            ))
        }

        return out
    }

    private var actions: [Action] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return allActions }
        return allActions.filter { action in
            Self.fuzzyMatch(needle: q, haystack: action.title.lowercased())
                || (action.subtitle?.lowercased().contains(q) ?? false)
        }
    }

    /// Order-preserving fuzzy match — every char in `needle` must appear in
    /// `haystack` in order (substring hit also matches trivially).
    static func fuzzyMatch(needle: String, haystack: String) -> Bool {
        var it = haystack.makeIterator()
        outer: for ch in needle {
            while let h = it.next() {
                if h == ch { continue outer }
            }
            return false
        }
        return true
    }

    private func runSelected() {
        let list = actions
        guard selectedIndex >= 0, selectedIndex < list.count else { return }
        list[selectedIndex].run()
        dismiss()
    }

    private func loadModels() async {
        let entries = await app.engine.scanModels(force: false)
        await MainActor.run { self.models = entries }
    }
}
