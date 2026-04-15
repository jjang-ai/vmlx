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

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.Colors.border)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                        ForEach(transcript) { turn in
                            TerminalTurnView(turn: turn)
                                .id(turn.id)
                        }
                    }
                    .padding(Theme.Spacing.lg)
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

    private var header: some View {
        HStack(spacing: Theme.Spacing.md) {
            Image(systemName: "terminal")
                .foregroundStyle(Theme.Colors.textHigh)
            Text("Terminal")
                .font(Theme.Typography.title)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
            Text(cwd.path)
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textLow)
                .lineLimit(1)
                .truncationMode(.head)
            Button {
                pickCwd()
            } label: {
                Image(systemName: "folder")
                    .foregroundStyle(Theme.Colors.textMid)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.vertical, Theme.Spacing.md)
        .background(Theme.Colors.surface)
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
                Text("Terminal mode gives the model full shell access")
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
                Button("I understand — enable Terminal mode") {
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

    private func pickCwd() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = cwd
        if panel.runModal() == .OK, let url = panel.url {
            cwd = url
        }
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
        // context for multi-turn shell sessions.
        var messages: [ChatRequest.Message] = [
            ChatRequest.Message(
                role: "system",
                content: .string("""
                    You are an expert terminal assistant with access to a \
                    `bash` tool that runs shell commands on the user's Mac. \
                    Current working directory: \(cwd.path). \
                    When the user asks for a task, call the `bash` tool with the \
                    appropriate command. Return concise explanations.
                    """)
            )
        ]
        for t in transcript {
            let role: String
            switch t.role {
            case .user: role = "user"
            case .assistant: role = "assistant"
            case .tool: role = "tool"
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

        // Inject the bash tool schema. Let the model choose when to call it.
        let tools: [ChatRequest.Tool] = [BashTool.openAISchema]

        let request = ChatRequest(
            model: state.selectedModelPath?.lastPathComponent ?? "default",
            messages: messages,
            stream: true,
            tools: tools,
            toolChoice: .auto
        )

        // Track the current assistant turn so we can append streaming
        // content in place. We create a new turn per final pass.
        var activeAssistantId = UUID()
        transcript.append(TerminalTurn(id: activeAssistantId, role: .assistant, text: ""))

        // Pull the stream from the actor in one `await` first, then
        // iterate locally. `Engine.stream(request:)` is actor-isolated,
        // so it cannot be called directly inside a `for try await in`
        // expression under Swift-6 strict concurrency — matches the
        // ChatViewModel pattern at `ChatViewModel.swift:350`.
        let upstream = await engine.stream(request: request)
        do {
            for try await chunk in upstream {
                if Task.isCancelled { break }

                // Append content into the active assistant turn.
                if let content = chunk.content, !content.isEmpty {
                    if let idx = transcript.firstIndex(where: { $0.id == activeAssistantId }) {
                        transcript[idx].text += content
                    }
                }

                // Tool-call lifecycle. `.started` creates a pending tool
                // turn, `.done` fills in the exit code + output.
                if let status = chunk.toolStatus {
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
                        if let idx = transcript.lastIndex(where: { $0.role == .tool && $0.exitCode == nil }) {
                            let (text, code) = decodeBashResult(status.message)
                            transcript[idx].text = text
                            transcript[idx].exitCode = code
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
        } catch {
            transcript.append(TerminalTurn(
                role: .tool,
                text: "engine error: \(error)",
                exitCode: -1
            ))
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
        var parts: [String] = []
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
    }
}

// MARK: - Models

struct TerminalTurn: Identifiable {
    enum Role { case user, assistant, tool }
    let id: UUID
    var role: Role
    var text: String
    var exitCode: Int32? = nil

    init(id: UUID = UUID(), role: Role, text: String, exitCode: Int32? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.exitCode = exitCode
    }
}

private struct TerminalTurnView: View {
    let turn: TerminalTurn

    /// Long-output collapse threshold (UI-6). Tool / model outputs over
    /// this many lines render with a fold: first FOLD_HEAD lines, an
    /// "expand N more lines" toggle, then the FOLD_TAIL trailing lines.
    /// The full text is still copyable from the expanded view.
    private static let foldThreshold = 80
    private static let foldHead = 25
    private static let foldTail = 15

    @State private var expanded: Bool = false

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
                if foldedRanges != nil {
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
        }
    }

    /// Cached line-count + folded ranges so the chevron header label
    /// doesn't have to re-split on every redraw.
    private var lineCount: Int { turn.text.components(separatedBy: "\n").count }

    /// Returns (head, hidden-count, tail) when the turn should be folded.
    /// `nil` means render the full text. User-input turns are never folded
    /// (they're typed by hand and short by definition).
    private var foldedRanges: (head: String, hidden: Int, tail: String)? {
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
        }
    }
    private var label: String {
        switch turn.role {
        case .user:      return "You"
        case .assistant: return "Model"
        case .tool:      return "$"
        }
    }
    private var color: SwiftUI.Color {
        switch turn.role {
        case .user:      return Theme.Colors.accent
        case .assistant: return Theme.Colors.accentHi
        case .tool:      return Theme.Colors.textMid
        }
    }
}
