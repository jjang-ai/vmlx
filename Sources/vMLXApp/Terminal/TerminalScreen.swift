import SwiftUI
import vMLXEngine
import vMLXTheme

/// Terminal mode — chat with full shell access.
///
/// The model runs with the `bash` tool injected. When it emits a `bash` tool
/// call, the engine executes the command via `BashTool` and feeds stdout/
/// stderr back into the conversation. No allowlist, no sandbox.
///
/// First-launch warning is gated on `UserDefaults.terminalWarningAcknowledged`.
struct TerminalScreen: View {
    @Environment(AppState.self) private var state
    @Environment(\.appLocale) private var appLocale: AppLocale
    @State private var input: String = ""
    @State private var transcript: [TerminalTurn] = []
    @State private var streaming: Bool = false
    @State private var cwd: URL = FileManager.default.homeDirectoryForCurrentUser
    @State private var showFirstRunWarning: Bool = !UserDefaults.standard.bool(
        forKey: "terminalWarningAcknowledged")
    @State private var currentStreamTask: Task<Void, Never>? = nil

    /// Command history for Up/Down arrow recall. Persisted for the life
    /// of the view; on Send a new entry is pushed.
    @State private var commandHistory: [String] = []
    /// Cursor into `commandHistory` during an active recall session;
    /// `nil` means we're on a fresh empty line (no active recall).
    @State private var historyIndex: Int? = nil
    /// Input text before recall started, so Down at the end restores it.
    @State private var preRecallInput: String = ""

    // §424 — Terminal toolbar state. Settings/logs sheets, plus the
    // per-session knobs that flow into the agentic loop.
    @State private var showSettings: Bool = false
    @State private var showLogs: Bool = false
    /// Reasoning effort sent as `reasoning_effort` chat-template kwarg.
    /// "none" maps to enable_thinking=false; everything else implies
    /// thinking on. Persisted in UserDefaults so the choice survives
    /// across app restarts.
    @AppStorage("terminal.reasoningEffort") private var reasoningEffort: String = "medium"
    /// Verbose surface — when ON, the bash tool's full command + stdout/
    /// stderr is rendered in the transcript (vs. a one-liner summary).
    /// Mirrors `vmlxctl chat --verbose`.
    @AppStorage("terminal.verbose") private var verbose: Bool = true
    /// Safety toggles — passed into the system prompt so the model knows
    /// the constraints. Defaults match `vmlxctl chat` defaults: no-
    /// destructive ON, the others OFF.
    @AppStorage("terminal.readOnly") private var readOnly: Bool = false
    @AppStorage("terminal.noNetwork") private var noNetwork: Bool = false
    @AppStorage("terminal.noDestructive") private var noDestructive: Bool = true
    @AppStorage("terminal.sandboxCwd") private var sandboxCwd: Bool = false
    /// Max tool-call iterations per user turn. Bumps the agentic loop
    /// ceiling for long multi-step tasks. Mirrors `vmlxctl chat
    /// --max-tool-calls`.
    @AppStorage("terminal.maxToolIterations") private var maxToolIterations: Int = 16
    /// Optional system-prompt override appended to the default. Empty
    /// string disables.
    @AppStorage("terminal.systemPromptOverride") private var systemPromptOverride: String = ""

    /// Locally-cached ModelLibrary entries — loaded on appear and
    /// refreshed when the user opens the picker. The library is an
    /// actor so we can't read it inline from a SwiftUI body; instead
    /// we kick a Task on appear and store the snapshot in @State.
    @State private var modelEntries: [ModelLibrary.ModelEntry] = []

    /// §427 — capabilities of the currently-loaded model, fetched
    /// from `engine.currentCapabilities()`. Drives the reasoning-effort
    /// picker (visible only when supportsThinking) AND the settings
    /// sheet's "detected parsers" line. Refreshes on engine state
    /// change so it tracks load/unload events.
    @State private var loadedCaps: ModelCapabilities? = nil

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.Colors.border)
            ScrollViewReader { proxy in
                ScrollView {
                    if transcript.isEmpty {
                        emptyState
                            .padding(Theme.Spacing.xxl)
                    } else {
                        LazyVStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                            ForEach(transcript) { turn in
                                TerminalTurnView(turn: turn, verbose: verbose)
                                    .id(turn.id)
                            }
                        }
                        .padding(Theme.Spacing.lg)
                    }
                }
                .onChange(of: transcript.count) { _, _ in
                    if let last = transcript.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
            Divider().background(Theme.Colors.border)
            inputBar
        }
        .background(Theme.Colors.background)
        .sheet(isPresented: $showFirstRunWarning) { firstRunWarning }
        .sheet(isPresented: $showSettings) {
            terminalSettingsSheet
        }
        .sheet(isPresented: $showLogs) {
            terminalLogsSheet
        }
        .task {
            await refreshModelEntries()
            await refreshLoadedCaps()
        }
        .onChange(of: state.sessions.count) { _, _ in
            // A session was added/removed — refresh state-tagged entries
            // so the picker reflects load/unload events triggered from
            // the Server tab too.
            Task {
                await refreshModelEntries()
                await refreshLoadedCaps()
            }
        }
        .onChange(of: state.engineState) { _, _ in
            // §427 — when engine transitions stopped→running or vice
            // versa, re-fetch capabilities so the reasoning picker
            // hides/shows + level options refresh per the loaded model.
            Task { await refreshLoadedCaps() }
        }
        // Up/Down arrow command history recall in the terminal input.
        // Matches the existing chat-input recall in ChatScreen so muscle
        // memory carries over. Only fires when the input is empty or the
        // user is already mid-recall — Up on a non-empty fresh line is
        // left to the TextEditor for normal caret movement.
        .onKeyPress(.upArrow) {
            recallPrevious() ? .handled : .ignored
        }
        .onKeyPress(.downArrow) {
            recallNext() ? .handled : .ignored
        }
    }

    // MARK: - Command history recall

    /// Move one step back in `commandHistory`. Returns `true` when the
    /// input was updated so the caller can swallow the keypress.
    private func recallPrevious() -> Bool {
        guard !commandHistory.isEmpty else { return false }
        // Only start a recall session on an empty line — otherwise the
        // user is typing something and Up should pass through.
        if historyIndex == nil {
            guard input.isEmpty else { return false }
            preRecallInput = input
            historyIndex = commandHistory.count - 1
        } else if let idx = historyIndex, idx > 0 {
            historyIndex = idx - 1
        } else {
            return true   // already at the oldest entry
        }
        if let idx = historyIndex {
            input = commandHistory[idx]
        }
        return true
    }

    /// Move one step forward in `commandHistory`. Down past the newest
    /// entry restores the pre-recall draft (typically empty).
    private func recallNext() -> Bool {
        guard let idx = historyIndex else { return false }
        let next = idx + 1
        if next >= commandHistory.count {
            historyIndex = nil
            input = preRecallInput
        } else {
            historyIndex = next
            input = commandHistory[next]
        }
        return true
    }

    // MARK: header

    /// Engine state → colored dot for the header indicator. Mirrors the
    /// EngineStatusFooter color scheme so the user sees the same signal
    /// in both surfaces.
    private var engineStateColor: Color {
        switch state.engineState {
        case .running: return .green
        case .loading: return .yellow
        case .standby: return .orange
        case .stopped: return Theme.Colors.textLow
        case .error:   return .red
        }
    }

    private var engineStateLabel: String {
        switch state.engineState {
        case .running: return "running"
        case .loading: return "loading"
        case .standby(let kind): return kind == .deep ? "deep sleep" : "light sleep"
        case .stopped: return "stopped"
        case .error:   return "error"
        }
    }

    /// Currently-active model display name. Two distinct things:
    ///   - `selectedModelName`: the user's pick from the picker (may be
    ///      .stopped, .loading, .running, or absent)
    ///   - `runningModelName`: the model that's actually live — only
    ///      non-nil when engine is `.running` so callers can switch on
    ///      "is the agentic loop available right now"
    /// The empty-state copy + chip suggestions key off `runningModelName`
    /// so the user never sees "Agentic Terminal" + chips while the
    /// engine is .stopped/.loading and Send would actually fall back to
    /// raw shell.
    private var selectedModelName: String? {
        state.selectedModelPath?.lastPathComponent
    }

    private var runningModelName: String? {
        guard case .running = state.engineState else { return nil }
        if let p = state.selectedModelPath { return p.lastPathComponent }
        if let s = state.sessions.first(where: {
            if case .running = $0.state { return true }; return false
        }) { return s.modelPath.lastPathComponent }
        return nil
    }

    private var header: some View {
        VStack(spacing: 0) {
            // ─── Row 1: title + model picker + load/stop + cwd ──────
            HStack(spacing: Theme.Spacing.md) {
                Image(systemName: "terminal")
                    .foregroundStyle(Theme.Colors.textHigh)
                Text(L10n.Terminal.terminal.render(appLocale))
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)

                // Model picker — Menu listing every entry in the
                // ModelLibrary. Each row shows a state dot + label +
                // running/loading/standby/stopped tag. Picking a row
                // sets selectedModelPath; the Load/Stop button below
                // calls app.startSession / app.stopSession.
                modelPickerMenu

                // Load/Stop button — toggle for the currently-selected
                // model. Disabled when no model is selected.
                loadStopButton

                Spacer()

                // CWD pill — click to pick a new directory.
                Button { pickCwd() } label: {
                    HStack(spacing: Theme.Spacing.xs) {
                        Image(systemName: "folder")
                            .foregroundStyle(Theme.Colors.textMid)
                        Text(cwd.path)
                            .font(Theme.Typography.mono)
                            .foregroundStyle(Theme.Colors.textLow)
                            .lineLimit(1)
                            .truncationMode(.head)
                            .frame(maxWidth: 280, alignment: .trailing)
                    }
                    .padding(.horizontal, Theme.Spacing.md)
                    .padding(.vertical, Theme.Spacing.xs)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .fill(Theme.Colors.surfaceHi)
                    )
                }
                .buttonStyle(.plain)
                .help("Change working directory")
            }

            // ─── Row 2: reasoning + verbose + action buttons ────────
            // §425b — switched from segmented Picker (320 px fixed
            // width, overflows at <850 px window widths) to a
            // dropdown Menu so the row gracefully handles narrow
            // windows. Verbose toggle + action buttons keep their
            // compact footprint; Spacer() between the two clusters
            // pushes the action group to the right edge.
            HStack(spacing: Theme.Spacing.md) {
                // §427 — Reasoning effort dropdown driven by the loaded
                // model's capabilities. The level set varies per family:
                // mistral4 = ["none","high"]; deepseek_v4 = full 5-tier
                // including "max"; everyone else = 4-tier (no max).
                // Models that don't support thinking at all (modality
                // .image, .embedding, or supportsThinking=false) hide
                // the picker entirely so we don't lie to the user about
                // toggles that have no effect.
                let levels = availableReasoningLevels
                if !levels.isEmpty {
                    Menu {
                        Picker("Reasoning effort", selection: $reasoningEffort) {
                            ForEach(levels, id: \.self) { lvl in
                                Text(reasoningLevelLabel(lvl)).tag(lvl)
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "brain")
                                .font(.system(size: 11))
                            Text("Reasoning: \(reasoningEffort)")
                                .font(Theme.Typography.caption)
                            Image(systemName: "chevron.down")
                                .font(.system(size: 9))
                        }
                        .foregroundStyle(Theme.Colors.textMid)
                        .padding(.horizontal, Theme.Spacing.sm)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Theme.Colors.surfaceHi)
                        )
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .help("Reasoning effort options come from the loaded model's family. \"none\" disables thinking entirely.")
                }

                // Verbose toggle — show full bash commands + output in
                // the transcript when ON. §425b — wired to
                // TerminalTurnView so the foldThreshold is bypassed
                // when ON (vs. the prior commit where this toggle
                // was a dead control).
                Toggle(isOn: $verbose) {
                    Text("Verbose")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textMid)
                }
                .toggleStyle(.switch)
                .tint(Theme.Colors.accent)
                .fixedSize()
                .help("Show full bash commands + stdout/stderr in the transcript (no auto-fold for long outputs)")

                Spacer()

                // Settings gear — opens TerminalSettingsSheet with
                // safety toggles, max iterations, system prompt
                // override.
                Button { showSettings = true } label: {
                    Image(systemName: "slider.horizontal.3")
                        .foregroundStyle(Theme.Colors.textMid)
                }
                .buttonStyle(.borderless)
                .help("Terminal settings: safety toggles, max iterations, system prompt")

                // Logs button — opens LogsPanel in a sheet so the user
                // can tail engine + model + tool events without
                // jumping to the Server tab.
                Button { showLogs = true } label: {
                    Image(systemName: "doc.text.below.ecg")
                        .foregroundStyle(Theme.Colors.textMid)
                }
                .buttonStyle(.borderless)
                .help("Live engine logs")

                // Clear-transcript button — drops the conversation
                // history but keeps command history + cwd. Use when
                // switching topic to avoid context bloat.
                Button { transcript.removeAll() } label: {
                    Image(systemName: "eraser")
                        .foregroundStyle(transcript.isEmpty ? Theme.Colors.textLow : Theme.Colors.textMid)
                }
                .buttonStyle(.borderless)
                .help("Clear transcript (keeps command history)")
                .disabled(transcript.isEmpty)
            }
            .padding(.top, Theme.Spacing.sm)
        }
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.vertical, Theme.Spacing.md)
        .background(Theme.Colors.surface)
    }

    // MARK: - Model picker

    /// Lookup-state for a library entry — same five states the
    /// ChatModelPicker exposes so the visual is consistent.
    private enum ModelLoadState { case running, loading, standby, stopped, absent }

    private func modelLoadState(for entry: ModelLibrary.ModelEntry) -> ModelLoadState {
        let canonical = entry.canonicalPath.standardizedFileURL.resolvingSymlinksInPath()
        guard let session = state.sessions.first(where: {
            $0.modelPath.standardizedFileURL.resolvingSymlinksInPath() == canonical
        }) else { return .absent }
        switch session.state {
        case .running: return .running
        case .loading: return .loading
        case .standby: return .standby
        case .stopped, .error: return .stopped
        }
    }

    private func modelStateColor(_ s: ModelLoadState) -> Color {
        switch s {
        case .running: return .green
        case .loading: return .yellow
        case .standby: return .orange
        case .stopped, .absent: return Theme.Colors.textLow
        }
    }

    private var currentEntry: ModelLibrary.ModelEntry? {
        guard let path = state.selectedModelPath else { return nil }
        let canonical = path.standardizedFileURL.resolvingSymlinksInPath()
        return modelEntries.first {
            $0.canonicalPath.standardizedFileURL.resolvingSymlinksInPath() == canonical
        }
    }

    private var modelPickerMenu: some View {
        Menu {
            if modelEntries.isEmpty {
                Text("No models found — add a directory in Server tab")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            } else {
                ForEach(modelEntries) { e in
                    Button {
                        state.selectedModelPath = e.canonicalPath
                    } label: {
                        HStack(spacing: Theme.Spacing.sm) {
                            Circle()
                                .fill(modelStateColor(modelLoadState(for: e)))
                                .frame(width: 8, height: 8)
                            Text(e.displayName)
                            Spacer()
                            if e.canonicalPath == state.selectedModelPath {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
                Divider()
                Button("Manage in Server tab…") { state.mode = .server }
            }
        } label: {
            HStack(spacing: 4) {
                Circle()
                    .fill(currentEntry.map { modelStateColor(modelLoadState(for: $0)) } ?? Theme.Colors.textLow)
                    .frame(width: 8, height: 8)
                Text(currentEntry?.displayName ?? "Select model")
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 220, alignment: .leading)
                Image(systemName: "chevron.down")
                    .font(.system(size: 10))
            }
            .foregroundStyle(Theme.Colors.textMid)
            .font(Theme.Typography.bodyHi)
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.surfaceHi)
            )
        }
        .menuStyle(.borderlessButton)
        .help("Pick a model — green = running, yellow = loading, gray = stopped")
    }

    private var loadStopButton: some View {
        let s = currentEntry.map { modelLoadState(for: $0) } ?? .absent
        return Button {
            guard let entry = currentEntry else { return }
            Task {
                let canonical = entry.canonicalPath.standardizedFileURL.resolvingSymlinksInPath()
                let existing = state.sessions.first(where: {
                    $0.modelPath.standardizedFileURL.resolvingSymlinksInPath() == canonical
                })
                switch s {
                case .running, .loading, .standby:
                    if let sid = existing?.id { await state.stopSession(sid) }
                case .stopped, .absent:
                    let sid: UUID
                    if let existing { sid = existing.id }
                    else { sid = await state.createSession(forModel: entry.canonicalPath) }
                    await state.startSession(sid)
                }
            }
        } label: {
            HStack(spacing: 4) {
                switch s {
                case .running, .loading, .standby:
                    Image(systemName: "stop.circle")
                    Text("Stop")
                case .stopped, .absent:
                    Image(systemName: "play.circle")
                    Text("Load")
                }
            }
            .font(Theme.Typography.caption)
            .foregroundStyle(currentEntry == nil ? Theme.Colors.textLow : Theme.Colors.accent)
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.surfaceHi)
            )
        }
        .buttonStyle(.plain)
        .disabled(currentEntry == nil)
        .help(currentEntry == nil ? "Pick a model first" : "Toggle model load/unload")
    }

    // MARK: - Settings sheet

    private var terminalSettingsSheet: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            HStack {
                Text("Terminal Settings")
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)
                Spacer()
                Button("Done") { showSettings = false }
                    .keyboardShortcut(.defaultAction)
            }

            Divider().background(Theme.Colors.border)

            VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                Text("Safety constraints")
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textMid)

                Toggle("Read-only — model instructed not to modify files",
                       isOn: $readOnly)
                Toggle("No network — model instructed not to make HTTP/git fetch requests",
                       isOn: $noNetwork)
                Toggle("Block destructive commands (rm -rf, dd, mkfs, force-push, etc.)",
                       isOn: $noDestructive)
                Toggle("Sandbox cwd — forbid `cd` outside the working directory",
                       isOn: $sandboxCwd)
            }

            Divider().background(Theme.Colors.border)

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                Text("Max tool iterations: \(maxToolIterations)")
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textMid)
                Slider(
                    value: Binding(
                        get: { Double(maxToolIterations) },
                        set: { maxToolIterations = Int($0) }
                    ),
                    in: 1...64, step: 1
                )
                Text("How many tool calls the model can chain before giving up. 16 fits most multi-step tasks; bump higher for long scripted runs.")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }

            Divider().background(Theme.Colors.border)

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                Text("System prompt override (optional)")
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textMid)
                TextEditor(text: $systemPromptOverride)
                    .font(Theme.Typography.mono)
                    .frame(minHeight: 80)
                    .scrollContentBackground(.hidden)
                    .background(Theme.Colors.surfaceHi)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .stroke(Theme.Colors.border, lineWidth: 1)
                    )
                Text("Replaces the default terminal-assistant system prompt. Leave blank to use the default.")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        }
        .padding(Theme.Spacing.xxl)
        .frame(width: 540)
        .background(Theme.Colors.surface)
    }

    // MARK: - Logs sheet

    /// Embeds LogsPanel in a sheet. The user can dismiss with Done or
    /// the macOS standard close button.
    private var terminalLogsSheet: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Engine Logs")
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)
                Spacer()
                Button("Done") { showLogs = false }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(Theme.Spacing.lg)
            Divider().background(Theme.Colors.border)
            LogsPanel()
                .frame(minWidth: 720, minHeight: 480)
        }
        .frame(width: 880, height: 600)
        .background(Theme.Colors.background)
    }

    // MARK: empty state

    /// Shown when the transcript is empty. Two layouts:
    ///   • Model loaded → welcome + 4 quick-action chips that each
    ///     pre-fill the input with a sample prompt (model can call
    ///     bash via the agentic loop). User clicks Send to dispatch.
    ///   • No model loaded → fallback hint explaining raw-shell mode +
    ///     button that jumps to the Server tab so the user can pick a
    ///     model.
    private var emptyState: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            HStack(spacing: Theme.Spacing.md) {
                Image(systemName: "terminal.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(Theme.Colors.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text(runningModelName != nil
                         ? "Agentic Terminal"
                         : "Raw Shell Mode")
                        .font(Theme.Typography.title)
                        .foregroundStyle(Theme.Colors.textHigh)
                    Text(runningModelName != nil
                         ? "Ask the model to run shell commands. It uses the `bash` tool to execute, reads the output, and decides what to do next."
                         : "No model loaded — input is dispatched directly to /bin/bash. Load a model from the Server tab to enable the agentic loop.")
                        .font(Theme.Typography.body)
                        .foregroundStyle(Theme.Colors.textMid)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if runningModelName != nil {
                // Quick-action chips — sample prompts to get the user
                // started. Each fills the input; the user reviews and
                // clicks Send.
                Text("Try one of these:")
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textMid)
                    .padding(.top, Theme.Spacing.md)
                let suggestions: [(String, String)] = [
                    ("doc.text.magnifyingglass", "Summarize the README in this directory"),
                    ("chevron.left.forwardslash.chevron.right", "Show me the git log for the last week"),
                    ("ladybug", "Find any TODOs or FIXMEs in the source tree"),
                    ("hammer", "Run the tests and tell me what failed"),
                ]
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), spacing: Theme.Spacing.md)],
                          spacing: Theme.Spacing.md) {
                    ForEach(suggestions, id: \.1) { (icon, text) in
                        Button {
                            input = text
                        } label: {
                            HStack(spacing: Theme.Spacing.sm) {
                                Image(systemName: icon)
                                    .frame(width: 18)
                                    .foregroundStyle(Theme.Colors.accent)
                                Text(text)
                                    .font(Theme.Typography.body)
                                    .foregroundStyle(Theme.Colors.textHigh)
                                    .multilineTextAlignment(.leading)
                                Spacer(minLength: 0)
                            }
                            .padding(Theme.Spacing.md)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: Theme.Radius.md)
                                    .fill(Theme.Colors.surfaceHi)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: Theme.Radius.md)
                                    .stroke(Theme.Colors.border, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }
            } else {
                Button {
                    state.mode = .server
                } label: {
                    HStack(spacing: Theme.Spacing.sm) {
                        Image(systemName: "server.rack")
                        Text("Go to Server tab")
                    }
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(.white)
                    .padding(.horizontal, Theme.Spacing.lg)
                    .padding(.vertical, Theme.Spacing.md)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .fill(Theme.Colors.accent)
                    )
                }
                .buttonStyle(.plain)
                .padding(.top, Theme.Spacing.md)
            }

            // Footer hint — reasoning surface + history recall keys so
            // users discover the affordances without trial-and-error.
            HStack(spacing: Theme.Spacing.lg) {
                Label("↑/↓ recalls history", systemImage: "arrow.up.arrow.down")
                Label("⌘4 jumps here", systemImage: "command")
                Label("Working dir: \(cwd.lastPathComponent)", systemImage: "folder")
            }
            .font(Theme.Typography.caption)
            .foregroundStyle(Theme.Colors.textLow)
            .padding(.top, Theme.Spacing.lg)
        }
        .frame(maxWidth: 720, alignment: .leading)
    }

    // MARK: input bar

    private var inputBar: some View {
        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
            TextEditor(text: $input)
                .font(Theme.Typography.mono)
                .scrollContentBackground(.hidden)
                .background(Theme.Colors.surfaceHi)
                .foregroundStyle(Theme.Colors.textHigh)
                .frame(minHeight: 60, maxHeight: 140)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.md))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
                .onSubmit { send() }
            Button(action: send) {
                Image(systemName: streaming ? "stop.circle.fill" : "arrow.up.circle.fill")
                    .resizable()
                    .frame(width: 28, height: 28)
                    .foregroundStyle(streaming ? Theme.Colors.danger : Theme.Colors.accent)
            }
            .buttonStyle(.plain)
            .disabled(input.isEmpty && !streaming)
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.background)
    }

    // MARK: first-run warning

    private var firstRunWarning: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            HStack(spacing: Theme.Spacing.md) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(Theme.Colors.warning)
                    .font(.system(size: 28))
                Text(L10n.Terminal.terminalModeHelp.render(appLocale))
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)
            }
            Text("""
                The model can run any command on your Mac with your user permissions: \
                read and write files, install packages, push to git, send network requests, \
                anything you can do in Terminal.app.

                There is no sandbox, no allow-list, and no confirmation prompt for individual \
                commands. Treat this like giving someone SSH access to your machine.

                Use only with models you trust, in directories you can afford to lose.
                """)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button(L10n.TerminalUI.understandEnable.render(appLocale)) {
                    UserDefaults.standard.set(true, forKey: "terminalWarningAcknowledged")
                    showFirstRunWarning = false
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(Theme.Spacing.xxl)
        .frame(width: 520)
        .background(Theme.Colors.surface)
    }

    // MARK: actions

    @MainActor
    private func pickCwd() {
        // Iter 129 (vmlx#121 / #133): macOS 26 ad-hoc XPC failure
        // mitigation. NSOpenPanelSafe falls back to manual-path entry
        // when the picker XPC link is dead.
        let result = NSOpenPanelSafe.pick(configure: { panel in
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            panel.directoryURL = cwd
        }, fallbackTitle: L10n.PickerFallback.cwdTitle.render(appLocale),
           fallbackMessage: L10n.PickerFallback.cwdMessage.render(appLocale),
           canChooseFiles: false)
        if let url = result.url { cwd = url }
    }

    /// Refresh the cached `modelEntries` snapshot. Called on appear and
    /// whenever the global session list changes so the picker reflects
    /// load/unload events from the Server tab.
    private func refreshModelEntries() async {
        let lib = await state.engine.modelLibrary
        let snapshot = await lib.entries()
        await MainActor.run { modelEntries = snapshot }
    }

    /// §427 — Refresh `loadedCaps` from the engine. nil unless the
    /// engine reached `.running` AND the loader populated capabilities
    /// (jang_config-stamped > silver-table > template-sniff > fallback).
    private func refreshLoadedCaps() async {
        let caps = await state.engine.currentCapabilities()
        await MainActor.run {
            loadedCaps = caps
            clampReasoningEffortToModel()
        }
    }

    /// §427b — Reasoning-effort levels supported by the loaded model.
    /// What the model's chat template actually CONSUMES, not what
    /// generic OpenAI clients can send.
    ///
    /// Most families implement reasoning as a BINARY switch via
    /// `enable_thinking: bool` in the chat template — Qwen3, Gemma4,
    /// Nemotron, MiniMax, GLM-MoE all read just `enable_thinking`
    /// and ignore graded `reasoning_effort` strings. Showing
    /// low/medium/high for these is misleading — those values don't
    /// change the prompt at all.
    ///
    /// Only three families actually consume graded levels:
    ///   - mistral4: template branches on `reasoning_effort=='high'`
    ///     vs everything else → ["none", "high"]
    ///   - deepseek_v4: template branches on `none/low/medium/high/max`
    ///     with a special max-tier system prefix → 5-tier
    ///   - gpt_oss (Harmony): channel header carries
    ///     `<|channel|>analysis effort=...` with minimal/low/medium/high
    ///
    /// Everyone else → binary [off, on] surfaced as [none, thinking].
    /// Models that don't support thinking return [] → picker hides.
    private var availableReasoningLevels: [String] {
        guard let caps = loadedCaps, caps.supportsThinking else { return [] }
        switch caps.family {
        case "mistral4":
            return ["none", "high"]
        case "deepseek_v4":
            return ["none", "low", "medium", "high", "max"]
        case "gpt_oss", "gpt_oss_v2":
            return ["minimal", "low", "medium", "high"]
        default:
            // Binary family: enable_thinking on/off. Show as
            // [none, thinking] so the persisted value reads the same
            // ("none" everywhere = thinking off).
            return ["none", "thinking"]
        }
    }

    /// Display label for a reasoning level + family-specific hint.
    /// Hints reflect what the level actually does on the active
    /// family — e.g. "Low" on DSV4 vs "Low" on GPT-OSS are different.
    private func reasoningLevelLabel(_ level: String) -> String {
        let family = loadedCaps?.family ?? ""
        switch level {
        case "none":
            return "None — chat mode (no thinking)"
        case "thinking":
            return "Thinking — `<think>` blocks enabled"
        case "minimal":
            return "Minimal — Harmony minimal effort"
        case "low":
            return family == "deepseek_v4" ? "Low" : "Low — Harmony low effort"
        case "medium":
            return family == "deepseek_v4" ? "Medium" : "Medium — Harmony medium effort"
        case "high":
            return family == "mistral4"
                ? "High — Mistral 4 thinking on"
                : (family == "deepseek_v4" ? "High" : "High — Harmony high effort")
        case "max":
            return "Max — DSV4 absolute-maximum prefix"
        default:
            return level
        }
    }

    /// §427 — clamp the persisted `reasoningEffort` to what the loaded
    /// model supports. Called after `loadedCaps` refreshes so a user
    /// who had effort=max persisted from a prior DSV4 session doesn't
    /// silently get an invalid level passed to the next model's chat
    /// template. Falls back to the highest supported level.
    private func clampReasoningEffortToModel() {
        let levels = availableReasoningLevels
        guard !levels.isEmpty else { return }
        if !levels.contains(reasoningEffort) {
            reasoningEffort = levels.last ?? "none"
        }
    }

    /// Build the system prompt — default terminal-assistant message plus
    /// any safety constraint text plus the user's optional override.
    /// §428 — explicit retry-on-failure instructions. Models were
    /// giving up after a single failed `bash` call; the new prompt
    /// tells them to inspect stderr/exit_code, plan a correction in
    /// reasoning, and retry. Interleaved reasoning between attempts
    /// is encouraged.
    private func buildSystemPrompt() -> String {
        // §428c — modality-aware preamble. Vision/web-agent models
        // (Holo3, Qwen3.5-VL, Gemma4-VL) were trained on browser
        // automation + screenshot reasoning and tend to hallucinate
        // Selenium / Playwright / browser tools that don't exist in
        // Terminal mode. Tell them up-front: in this mode you have
        // BASH ONLY, here's how to do screen / browser tasks via shell.
        let isVL = (loadedCaps?.modality == .vision)
        let modalityNote: String
        if isVL {
            modalityNote = """

                ✓ VISION-MODEL TOOLBELT: You have THREE tools — `bash`, `screenshot`, AND `browser`. Every screenshot/browser action returns a PNG that is attached to your NEXT input as a real image you can SEE.

                  ► `browser` — Headless web browser (preferred for web tasks)
                     The `browser` tool is a stateful, hidden Chromium-class browser that lives across calls. Cookies and DOM persist. ALWAYS use `browser` for web tasks instead of `bash`+`open` — `open` just spawns the user's Safari and returns; you cannot click, type, or see the result.
                     Actions:
                       - {"action":"open", "url":"https://..."}     → loads URL, returns screenshot
                       - {"action":"click", "selector":"button.go"} → CSS-selector click (preferred)
                       - {"action":"click", "x":640, "y":420}        → click at viewport pixel (fallback when no selector)
                       - {"action":"type", "selector":"input#q", "text":"hello"}
                       - {"action":"scroll", "delta_y":600}          → scroll down 600px
                       - {"action":"eval", "script":"document.title"} → JS eval, returns text
                       - {"action":"screenshot"}                     → re-snap without action
                       - {"action":"back" / "forward" / "reload" / "close"}

                  ► `screenshot` — Capture the user's actual desktop screen (NOT the browser).
                     Use for OS-level UI tasks: reading what's open in their app, debugging visible state, etc.

                  ► `bash` — Shell access for files, processes, system queries, installs.
                     Useful adjuncts: `pbpaste` (read clipboard), `osascript` (drive native macOS apps), `curl -sL` (raw HTTP without rendering).

                Browser-task playbook:
                  1. {"action":"open", "url":"…"}              → see landing page
                  2. analyze the screenshot → identify next interaction
                  3. {"action":"click"…} or {"action":"type"…} → see result
                  4. repeat 2-3 until task done

                NEVER substitute `bash open URL` for `browser open URL` — `open` is fire-and-forget; `browser` keeps a session you can drive.

                """
        } else {
            modalityNote = ""
        }

        var lines: [String] = [
            """
            You are an autonomous terminal agent on the user's Mac. You have ONE tool: `bash`. Your job is to USE the tool, not describe it.

            Current working directory: \(cwd.path).
            \(modalityNote)

            ╔══════════════════════════════════════════════════════════╗
            ║  CRITICAL RULES — VIOLATING THESE BREAKS THE INTERFACE   ║
            ╠══════════════════════════════════════════════════════════╣
            ║                                                          ║
            ║  1. NEVER write ```bash code blocks in your reply text. ║
            ║     The user CANNOT execute markdown code. Markdown      ║
            ║     means you've failed to use the tool.                 ║
            ║                                                          ║
            ║  2. EVERY shell command MUST go through the `bash` tool ║
            ║     call mechanism — NOT prose, NOT markdown.            ║
            ║                                                          ║
            ║  3. NEVER say "Let me check…" or "I'll run…" without    ║
            ║     ALSO emitting the tool call in the same response.   ║
            ║     Talking about what you would do = task failed.      ║
            ║                                                          ║
            ║  4. When in doubt: call the tool. Empty output is fine. ║
            ║     Wrong-command output is fine. The tool is cheap.    ║
            ║                                                          ║
            ╚══════════════════════════════════════════════════════════╝

            AGENTIC LOOP:
            • User asks a task → call `bash` with the best first command
            • Read the tool result: stdout, stderr, exit_code
            • exit_code != 0 → reason about WHY, call `bash` again with a fix
              (up to \(maxToolIterations) attempts per user turn)
            • Common recovery patterns:
              - Command not found → `which X` / `command -v X` / install via brew
              - Permission denied → `ls -la` to inspect ownership
              - No such file → `ls` the parent directory
              - Wrong syntax → check `man X` or try alternatives
            • Task verified complete → write a concise one-paragraph summary

            REASONING:
            Use reasoning blocks (`<think>...</think>`) to plan BEFORE each tool call. After seeing the result, reason again before the next call. Interleaved reasoning + tool calls is expected and encouraged. Do NOT put commands inside reasoning — reasoning is for thought only.

            DO NOT FORGET RULE #1. NO MARKDOWN CODE BLOCKS. CALL THE TOOL.
            """,
        ]
        if readOnly {
            lines.append("READ-ONLY MODE: do not modify, create, or delete files. Do not start long-running processes. Read-only commands only.")
        }
        if noNetwork {
            lines.append("NO-NETWORK MODE: do not run commands that make network requests (curl, wget, git fetch/push/pull, npm install, pip install, etc.).")
        }
        if noDestructive {
            lines.append("NO-DESTRUCTIVE: never run `rm -rf`, `dd`, `mkfs`, `git push --force`, `git reset --hard`, or any command that destroys data without explicit user confirmation.")
        }
        if sandboxCwd {
            lines.append("SANDBOX-CWD: do not `cd` outside \(cwd.path). All commands must operate inside this directory.")
        }
        let trimmed = systemPromptOverride.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            lines.append("Additional user instructions: \(trimmed)")
        }
        return lines.joined(separator: "\n\n")
    }

    private func send() {
        // Stop button: cancel in-flight generation.
        if streaming {
            currentStreamTask?.cancel()
            currentStreamTask = nil
            streaming = false
            return
        }
        guard !input.isEmpty else { return }
        let userText = input
        // Push to command history (dedup against most-recent) and reset
        // the recall cursor so Up next time starts at the new tail.
        if commandHistory.last != userText {
            commandHistory.append(userText)
        }
        historyIndex = nil
        preRecallInput = ""
        input = ""
        transcript.append(TerminalTurn(role: .user, text: userText))
        streaming = true
        currentStreamTask = Task {
            await runViaEngine(userText)
            await MainActor.run { streaming = false }
        }
    }

    /// Run the message through `Engine.stream` with the `bash` tool
    /// injected into the request. The engine's tool-call loop executes
    /// `BashTool` server-side between generation passes and surfaces
    /// progress via `ToolStatus` events so the UI can render live
    /// "running $ ls -la" cards. If the engine is stopped or no model is
    /// loaded, we fall back to direct shell exec so the Terminal still
    /// works as a raw REPL.
    private func runViaEngine(_ userText: String) async {
        guard let engine = state.activeEngine,
              case .running = state.engineState
        else {
            // No engine available — direct shell exec fallback so the
            // Terminal always works even without a loaded model.
            await runShellDirect(userText)
            return
        }

        // Build the conversation history from the transcript. Every
        // previous user/assistant/tool turn goes in so the model has
        // context for multi-turn shell sessions. System prompt is
        // assembled by `buildSystemPrompt()` so user safety toggles +
        // optional override apply.
        var messages: [ChatRequest.Message] = [
            ChatRequest.Message(role: "system", content: .string(buildSystemPrompt()))
        ]
        for t in transcript {
            let role: String
            switch t.role {
            case .user: role = "user"
            case .assistant: role = "assistant"
            case .tool: role = "tool"
            case .reasoning: continue
            }
            messages.append(ChatRequest.Message(
                role: role,
                content: .string(t.text)
            ))
        }

        // JIT wake: same idle-sleep recovery as ChatViewModel. Terminal
        // sessions share the engine with Chat, so an idle-sleep during
        // a long terminal conversation used to leave the next command
        // stranded. wakeFromStandby is a no-op when .running.
        await engine.wakeFromStandby()

        // §429/§431 — Tool list. Bash always. VL models additionally
        // get `screenshot` (capture the user's actual screen) and
        // `browser` (drive a headless WKWebView for autonomous web
        // tasks — open / click / type / scroll, with each action
        // returning a fresh PNG the model SEES on the next turn).
        // Non-VL models don't see screenshot/browser since they
        // couldn't act on the resulting image.
        var tools: [ChatRequest.Tool] = [BashTool.openAISchema]
        if loadedCaps?.modality == .vision {
            tools.append(ScreenshotTool.openAISchema)
            #if canImport(WebKit) && canImport(AppKit)
            tools.append(BrowserTool.openAISchema)
            #endif
        }

        // §424 — wire the toolbar settings into the request. Reasoning
        // effort flows through `reasoning_effort` chat-template kwarg
        // (Stream.swift:792 picks it up). max_tool_iterations bumps
        // the agentic-loop ceiling per the ChatRequest field.
        //
        // §427b — reasoningEffort=="thinking" is the binary-family
        // sentinel. Don't pass it as a graded `reasoning_effort` kwarg
        // (no template recognizes "thinking" as a value); just flip
        // enable_thinking=true and leave effort nil. Same for "none":
        // enable_thinking=false, effort nil. Graded values pass
        // through unchanged.
        let enableThinking: Bool = (reasoningEffort != "none")
        let effort: String?
        switch reasoningEffort {
        case "none", "thinking":
            effort = nil   // binary families — only enable_thinking matters
        default:
            effort = reasoningEffort   // graded families (deepseek_v4 / gpt_oss / mistral4)
        }
        var request = ChatRequest(
            model: state.selectedModelPath?.lastPathComponent ?? "default",
            messages: messages,
            stream: true,
            enableThinking: enableThinking,
            reasoningEffort: effort,
            tools: tools,
            toolChoice: .auto
        )
        // maxToolIterations is set as a settable property after init
        // (matches the ChatViewModel pattern at ChatViewModel.swift:1015 —
        // the public init predates the field).
        if maxToolIterations > 0 {
            request.maxToolIterations = maxToolIterations
        }

        // Track the current assistant turn so we can append streaming
        // content in place. We create a new turn per final pass.
        var activeAssistantId = UUID()
        transcript.append(TerminalTurn(id: activeAssistantId, role: .assistant, text: ""))

        // §371 — active reasoning turn tracker. Created on first
        // reasoning chunk of each "block" and auto-collapsed when the
        // model stops reasoning (first content or tool_call chunk
        // after a reasoning run). Modern agentic models interleave
        // thought with tool calls: reason → call bash → read output
        // → reason again → call bash again → content. Each reasoning
        // block gets its own collapsible turn so the user can see the
        // full decision loop without scrollback drowning.
        var activeReasoningId: UUID? = nil

        // Pull the stream from the actor in one `await` first, then
        // iterate locally. `Engine.stream(request:)` is actor-isolated,
        // so it cannot be called directly inside a `for try await in`
        // expression under Swift-6 strict concurrency — matches the
        // ChatViewModel pattern at `ChatViewModel.swift:350`.
        let upstream = await engine.stream(request: request)
        do {
            for try await chunk in upstream {
                if Task.isCancelled { break }

                // §371 — append reasoning deltas into the active
                // reasoning turn (or start a new one). Interleaved
                // models can reason mid-tool-loop, so we allow a new
                // reasoning block after every tool_status.done.
                if let think = chunk.reasoning, !think.isEmpty {
                    if let rid = activeReasoningId,
                       let idx = transcript.firstIndex(where: { $0.id == rid }) {
                        transcript[idx].text += think
                    } else {
                        let rid = UUID()
                        activeReasoningId = rid
                        transcript.append(TerminalTurn(
                            id: rid, role: .reasoning, text: think,
                            isExpanded: true))
                    }
                }

                // Append content into the active assistant turn. Close
                // any open reasoning block since content = model is done
                // thinking (for now).
                if let content = chunk.content, !content.isEmpty {
                    if let rid = activeReasoningId,
                       let idx = transcript.firstIndex(where: { $0.id == rid }) {
                        transcript[idx].isExpanded = false
                        activeReasoningId = nil
                    }
                    if let idx = transcript.firstIndex(where: { $0.id == activeAssistantId }) {
                        transcript[idx].text += content
                    }
                }

                // Tool-call lifecycle. `.started` creates a pending tool
                // turn, `.done` fills in the exit code + output.
                if let status = chunk.toolStatus {
                    // §371 — collapse any active reasoning block when
                    // a tool call fires so the user sees the thought
                    // block that led up to the call, then the call.
                    if let rid = activeReasoningId,
                       let idx = transcript.firstIndex(where: { $0.id == rid }) {
                        transcript[idx].isExpanded = false
                        activeReasoningId = nil
                    }
                    switch status.phase {
                    case .started:
                        transcript.append(TerminalTurn(
                            role: .tool,
                            text: "running: \(status.name)",
                            exitCode: nil
                        ))
                    case .done:
                        // Replace the last `.tool` turn with the real
                        // result. The engine already packed stdout/stderr/
                        // exit_code into status.message as JSON.
                        // §428 — when exit_code != 0, prefix the visible
                        // turn with a clear failure marker so the user
                        // (and the model on its next reasoning pass)
                        // knows this attempt failed and another should
                        // follow. The engine ALREADY feeds the tool
                        // result back to the model as a tool message;
                        // this is purely a UI-side affordance so the
                        // failure → retry pattern is visible in
                        // scrollback.
                        if let idx = transcript.lastIndex(where: { $0.role == .tool && $0.exitCode == nil }) {
                            let (text, code) = decodeBashResult(status.message)
                            transcript[idx].text = text
                            transcript[idx].exitCode = code
                        }
                        // iter-92 §119: sync UI-side `cwd` from the
                        // engine's authoritative `terminalCwd`.
                        // `ToolDispatcher.executeBashTool` already calls
                        // `engine.updateTerminalCwd(result.newCwd)` so
                        // the engine tracks the post-exec pwd, but the
                        // UI's `@State cwd` was only updated from the
                        // raw-shell fallback (§87). Model-mode bash
                        // `cd foo` updated the engine but the header
                        // path at the top of TerminalScreen silently
                        // stayed at the old value, and the next raw-
                        // shell invocation used the stale UI cwd —
                        // divergence between what the user sees and
                        // what the next command runs in.
                        if let engineCwd = await engine.terminalCwd,
                           engineCwd != cwd {
                            cwd = engineCwd
                        }
                        // Start a fresh assistant turn for the next pass.
                        activeAssistantId = UUID()
                        transcript.append(TerminalTurn(
                            id: activeAssistantId, role: .assistant, text: ""))
                    case .error:
                        if let idx = transcript.lastIndex(where: { $0.role == .tool && $0.exitCode == nil }) {
                            transcript[idx].text = status.message ?? "tool error"
                            transcript[idx].exitCode = -1
                        }
                    case .running:
                        break  // no UI change
                    }
                }
            }

            // Drop any trailing empty assistant turn (happens after the
            // final tool result when the model finishes without more text).
            if let last = transcript.last, last.role == .assistant, last.text.isEmpty {
                transcript.removeLast()
            }

            // §429 — VL screenshot rendezvous. If the model called the
            // `screenshot` tool during this stream, the engine has
            // staged the captured PNG path(s) on its actor. Drain them
            // and AUTO-CONTINUE the conversation with a synthetic user
            // message that attaches the image as `image_url` so the VL
            // model actually SEES the pixels on the next forward pass.
            //
            // We bound the auto-continue to one round per user turn to
            // avoid runaway loops where the model keeps screenshotting
            // without converging. A future refinement can lift this to
            // N rounds and respect maxToolIterations.
            let captures = await engine.consumeLatestScreenshots()
            if !captures.isEmpty {
                await autoContinueWithScreenshots(
                    captures, engine: engine, baseMessages: messages, tools: tools)
            }
        } catch {
            // §428 — engine error mid-stream. Drop the empty assistant
            // turn (so the failure isn't followed by a blank bubble),
            // then surface a clear `.tool` turn explaining what
            // happened. Common shapes:
            //   • CancellationError — user hit Stop, no need to
            //     yelp (suppress)
            //   • EngineError.invalidRequest — model rejected the
            //     prompt (e.g. too-long)
            //   • Other — usually MLX or scheduler hiccup; suggest
            //     reload via the Server tab
            //
            // Note: an MLX Metal command-buffer abort (SIGABRT from
            // `MTLReportFailure`) is NOT catchable as a Swift error —
            // it bypasses Swift's try/catch and tears the process
            // down. Real defense would be running the model in a
            // subprocess. The diagnostic on next launch is the
            // crash log; we can't catch it here.
            if let idx = transcript.lastIndex(where: { $0.role == .assistant && $0.text.isEmpty }) {
                transcript.remove(at: idx)
            }
            if error is CancellationError {
                // User hit Stop — silent stop is fine, don't lecture.
                return
            }
            transcript.append(TerminalTurn(
                role: .tool,
                text: "Engine error: \(error)\n\nIf this keeps happening: open Server tab → Stop → Load again to refresh the model. Long agentic loops on smaller models can cause Metal OOM.",
                exitCode: -1
            ))
        }
    }

    /// §429 — auto-continue path for VL screenshot rendezvous. After
    /// the engine's tool-call loop concludes, if the model captured
    /// screenshots during the turn, drain them, append a synthetic
    /// user message that attaches each PNG as an `image_url` content
    /// part, and re-stream so the model can SEE its captures on the
    /// next forward pass.
    ///
    /// This is the critical piece that turns a vision model into a
    /// real screen-aware agent: the engine's tool-call loop only
    /// flows TEXT through tool messages (per OpenAI spec), but VL
    /// inference needs IMAGE content blocks in user messages. We
    /// bridge those two by exiting the inner agentic loop, attaching
    /// pixels, and entering a fresh stream.
    private func autoContinueWithScreenshots(
        _ paths: [URL],
        engine: Engine,
        baseMessages: [ChatRequest.Message],
        tools: [ChatRequest.Tool]
    ) async {
        // Build the multipart user message: [text, image_url, image_url, …]
        var parts: [ChatRequest.ContentPart] = [
            ChatRequest.ContentPart(
                type: "text",
                text: "Here is the screenshot you captured. Look at it carefully and tell me what you see, then continue with the original task.",
                imageUrl: nil,
                videoUrl: nil
            )
        ]
        for url in paths {
            // Inline as a `file://` ImageURL — ChatRequest.ImageURL
            // already handles file:// inputs through its loader.
            parts.append(ChatRequest.ContentPart(
                type: "image_url",
                text: nil,
                imageUrl: ChatRequest.ContentPart.ImageURL(url: "file://\(url.path)"),
                videoUrl: nil
            ))
        }

        // Walk the in-flight transcript to rebuild the chat history so
        // the assistant turns the model produced THIS turn (including
        // any tool messages) are part of the new request's context —
        // otherwise the model wouldn't remember what it captured or
        // why.
        var msgs = baseMessages
        for t in transcript.suffix(20) {   // last ~20 turns is plenty
            let role: String
            switch t.role {
            case .user: role = "user"
            case .assistant: role = "assistant"
            case .tool: role = "tool"
            case .reasoning: continue
            }
            // Skip already-included history (baseMessages was built
            // with the prior transcript). Just append the NEW turns
            // since this user prompt — heuristic: anything after the
            // last user role in baseMessages.
            _ = role
        }
        msgs.append(ChatRequest.Message(role: "user", content: .parts(parts)))

        // Show a UI marker so the user sees the auto-continue happen.
        transcript.append(TerminalTurn(
            role: .tool,
            text: "📷 Screenshot captured — re-prompting with image attached…",
            exitCode: 0
        ))

        var request = ChatRequest(
            model: state.selectedModelPath?.lastPathComponent ?? "default",
            messages: msgs,
            stream: true,
            enableThinking: (reasoningEffort != "none"),
            reasoningEffort: (reasoningEffort == "none" || reasoningEffort == "thinking") ? nil : reasoningEffort,
            tools: tools,
            toolChoice: .auto
        )
        if maxToolIterations > 0 {
            request.maxToolIterations = maxToolIterations
        }

        // Fresh assistant turn for the auto-continue response.
        let activeAssistantId = UUID()
        transcript.append(TerminalTurn(id: activeAssistantId, role: .assistant, text: ""))

        let upstream = await engine.stream(request: request)
        do {
            for try await chunk in upstream {
                if Task.isCancelled { break }
                if let content = chunk.content, !content.isEmpty,
                   let idx = transcript.firstIndex(where: { $0.id == activeAssistantId }) {
                    transcript[idx].text += content
                }
                // Reasoning during auto-continue gets folded into the
                // assistant turn (we don't open a new collapsible
                // .reasoning turn here to keep the auto-continue
                // visually compact — user already saw the chips block
                // for the original turn).
            }
            if let last = transcript.last, last.role == .assistant, last.text.isEmpty {
                transcript.removeLast()
            }
        } catch {
            if !(error is CancellationError) {
                if let idx = transcript.lastIndex(where: { $0.role == .assistant && $0.text.isEmpty }) {
                    transcript.remove(at: idx)
                }
                transcript.append(TerminalTurn(
                    role: .tool,
                    text: "Auto-continue (with screenshot) error: \(error)",
                    exitCode: -1
                ))
            }
        }
    }

    /// Parse a bash tool result JSON blob into (display text, exit code).
    /// The engine's `ToolDispatcher` formats bash results as
    /// `{stdout, stderr, exit_code, timed_out?, killed?}`.
    private func decodeBashResult(_ jsonString: String?) -> (String, Int32) {
        guard let json = jsonString,
              let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return (jsonString ?? "(no output)", 0)
        }
        let stdout = (obj["stdout"] as? String) ?? ""
        let stderr = (obj["stderr"] as? String) ?? ""
        let exitCode = Int32((obj["exit_code"] as? Int) ?? 0)
        // Audit 2026-04-16 UX: `timed_out` and `killed` were silently
        // dropped. Prepend a visible banner so the user knows whether
        // empty output came from a timeout vs a no-output success.
        let timedOut = (obj["timed_out"] as? Bool) ?? false
        let killed = (obj["killed"] as? Bool) ?? false
        var parts: [String] = []
        if timedOut {
            parts.append("⏱ command timed out (process terminated)")
        } else if killed {
            parts.append("⚠ command killed before completion")
        }
        if !stdout.isEmpty { parts.append(stdout) }
        if !stderr.isEmpty { parts.append(stderr) }
        return (parts.joined(separator: "\n"), exitCode)
    }

    /// Raw shell fallback — used when no engine is loaded. Keeps the
    /// Terminal functional even without a model.
    private func runShellDirect(_ command: String) async {
        let bash = BashTool()
        let result = await bash.run(.init(command: command, cwd: cwd))
        let output = [result.stdout, result.stderr]
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        transcript.append(TerminalTurn(
            role: .tool,
            text: output.isEmpty ? "(no output)" : output,
            exitCode: result.exitCode
        ))
        // iter-56: persist the post-exec cwd so `cd foo` sticks across
        // commands. BashTool recovers the new cwd from a trailing
        // `pwd` marker it appends to the wrapped script; previously
        // the raw-shell fallback discarded `result.newCwd`, so every
        // `cd` was silently undone on the next command.
        //
        // iter-92 §119: also push the new cwd into the engine so if
        // the user later loads a model and switches to model-mode
        // bash, the engine's `terminalCwd` is already aligned with
        // what the UI displays. `updateTerminalCwd` is a no-op when
        // the value matches, so double-updates are cheap.
        if let newCwd = result.newCwd, newCwd != cwd {
            cwd = newCwd
            await state.engine.updateTerminalCwd(newCwd)
        }
    }
}

// MARK: - Models

struct TerminalTurn: Identifiable {
    // §371 — added `.reasoning` role so models that emit <think> blocks
    // or reasoning_content SSE chunks (Qwen3, GLM-5.1, DeepSeek-V3,
    // MiniMax-M2.7, Nemotron) render their thought stream inline with
    // the agentic loop instead of going silent while the model reasons
    // before a tool call. Rendered as a dimmed, collapsible block above
    // the assistant / tool turns that follow.
    enum Role { case user, assistant, tool, reasoning }
    let id: UUID
    var role: Role
    var text: String
    var exitCode: Int32? = nil
    /// §371 — collapse state for `.reasoning` turns. Defaults to
    /// expanded while streaming, collapses once the next non-reasoning
    /// chunk arrives so long thought streams don't clutter scrollback.
    var isExpanded: Bool = true

    init(id: UUID = UUID(), role: Role, text: String, exitCode: Int32? = nil, isExpanded: Bool = true) {
        self.id = id
        self.role = role
        self.text = text
        self.exitCode = exitCode
        self.isExpanded = isExpanded
    }
}

private struct TerminalTurnView: View {
    let turn: TerminalTurn
    /// §425b — when verbose=true, skip the long-output fold logic so
    /// tool stdout/stderr renders in full (matches `vmlxctl chat
    /// --verbose`). When verbose=false (default), the fold kicks in
    /// at 80 lines so long outputs don't drown the scrollback.
    var verbose: Bool = false

    /// Long-output collapse threshold (UI-6). Tool / model outputs over
    /// this many lines render with a fold: first FOLD_HEAD lines, an
    /// "expand N more lines" toggle, then the FOLD_TAIL trailing lines.
    /// The full text is still copyable from the expanded view.
    private static let foldThreshold = 80
    private static let foldHead = 25
    private static let foldTail = 15

    @State private var expanded: Bool = false
    /// §371 — separate expand toggle for `.reasoning` turns. Seeds from
    /// `turn.isExpanded` (which the stream loop flips to false once the
    /// next non-reasoning chunk arrives) so long thought streams auto-
    /// collapse but the user can re-expand them.
    @State private var reasoningExpanded: Bool? = nil

    private var reasoningIsOpen: Bool {
        reasoningExpanded ?? turn.isExpanded
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack(spacing: Theme.Spacing.sm) {
                Image(systemName: icon)
                    .foregroundStyle(color)
                Text(label)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                if let code = turn.exitCode {
                    Text("exit \(code)")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(code == 0 ? Theme.Colors.success : Theme.Colors.danger)
                }
                Spacer()
                if turn.role == .reasoning {
                    Button {
                        withAnimation(.easeInOut(duration: 0.15)) {
                            reasoningExpanded = !reasoningIsOpen
                        }
                    } label: {
                        Text(reasoningIsOpen ? "Collapse" : "Show \(lineCount) lines")
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.accent)
                    }
                    .buttonStyle(.plain)
                } else if foldedRanges != nil {
                    Button {
                        withAnimation(.easeInOut(duration: 0.15)) {
                            expanded.toggle()
                        }
                    } label: {
                        Text(expanded ? "Collapse" : "Show all \(lineCount) lines")
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.accent)
                    }
                    .buttonStyle(.plain)
                }
            }
            if turn.role != .reasoning || reasoningIsOpen {
                outputBody
                    .padding(Theme.Spacing.md)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .fill(Theme.Colors.surface)
                            .overlay(
                                RoundedRectangle(cornerRadius: Theme.Radius.md)
                                    .stroke(Theme.Colors.border, lineWidth: 1)
                            )
                    )
                    .opacity(turn.role == .reasoning ? 0.75 : 1.0)
            }
        }
    }

    /// Cached line-count + folded ranges so the chevron header label
    /// doesn't have to re-split on every redraw.
    private var lineCount: Int { turn.text.components(separatedBy: "\n").count }

    /// Returns (head, hidden-count, tail) when the turn should be folded.
    /// `nil` means render the full text. User-input turns are never folded
    /// (they're typed by hand and short by definition).
    private var foldedRanges: (head: String, hidden: Int, tail: String)? {
        // §425b — verbose mode: never fold. Render full output even
        // when it's hundreds of lines long. The user opted in by
        // enabling Verbose in the toolbar, so respect it.
        if verbose { return nil }
        guard turn.role != .user else { return nil }
        let lines = turn.text.components(separatedBy: "\n")
        guard lines.count > Self.foldThreshold else { return nil }
        let head = lines.prefix(Self.foldHead).joined(separator: "\n")
        let tail = lines.suffix(Self.foldTail).joined(separator: "\n")
        let hidden = lines.count - Self.foldHead - Self.foldTail
        return (head, hidden, tail)
    }

    @ViewBuilder
    private var outputBody: some View {
        if let folded = foldedRanges, !expanded {
            VStack(alignment: .leading, spacing: 6) {
                Text(folded.head).textSelection(.enabled)
                HStack(spacing: 6) {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.Colors.textLow)
                    Text("\(folded.hidden) lines hidden")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                    Spacer()
                }
                .padding(.vertical, 4)
                .padding(.horizontal, Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.sm)
                        .fill(Theme.Colors.surfaceHi)
                )
                Text(folded.tail).textSelection(.enabled)
            }
            .font(turn.role == .user ? Theme.Typography.body : Theme.Typography.mono)
            .foregroundStyle(Theme.Colors.textHigh)
        } else {
            Text(turn.text)
                .font(turn.role == .user ? Theme.Typography.body : Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textHigh)
                .textSelection(.enabled)
        }
    }

    private var icon: String {
        switch turn.role {
        case .user:      return "person.fill"
        case .assistant: return "sparkles"
        case .tool:      return "terminal.fill"
        case .reasoning: return "brain"
        }
    }
    private var label: String {
        switch turn.role {
        case .user:      return "You"
        case .assistant: return "Model"
        case .tool:      return "$"
        case .reasoning: return "thinking"
        }
    }
    private var color: SwiftUI.Color {
        switch turn.role {
        case .user:      return Theme.Colors.accent
        case .assistant: return Theme.Colors.accentHi
        case .tool:      return Theme.Colors.textMid
        case .reasoning: return Theme.Colors.textLow
        }
    }
}
