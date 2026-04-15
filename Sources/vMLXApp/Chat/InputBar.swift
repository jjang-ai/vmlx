import SwiftUI
import UniformTypeIdentifiers
import vMLXEngine
import vMLXTheme

struct InputBar: View {
    @Environment(AppState.self) private var app
    @Bindable var vm: ChatViewModel
    @State private var showImporter = false

    // Input history recall state (Up/Down to cycle prior user turns in
    // the current chat, Esc to return to the draft). Mirrors the
    // behavior of terminal shells — Up walks backward, Down forward,
    // Esc restores the draft the user had in progress before they
    // started cycling. Reset on any normal keystroke so typing never
    // gets intercepted by the cycler.
    @State private var historyIndex: Int? = nil
    @State private var draftBeforeHistory: String = ""

    /// True when the send button should actually send a request.
    /// Requires: engine running, input not empty, and we're not already
    /// generating. While generating, the button flips to "stop" which
    /// IS enabled (so the user can cancel).
    private var canSend: Bool {
        let hasText = !vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !vm.pendingImages.isEmpty
        guard hasText else { return false }
        switch app.engineState {
        case .running, .standby:
            // Standby is OK — the JIT wake banner will fire as soon as
            // the request hits the server.
            return true
        case .loading, .stopped, .error:
            return false
        }
    }

    /// Button is tappable if we can send OR we're currently streaming
    /// (so the user can hit stop).
    private var buttonEnabled: Bool {
        canSend || vm.isGenerating
    }

    /// Placeholder text adapts to the engine state so the user knows
    /// why the button is disabled without hovering.
    private var placeholderText: String {
        switch app.engineState {
        case .running: return "Send a message…"
        case .loading: return "Loading model…"
        case .standby(.soft): return "Waking up model…"
        case .standby(.deep): return "Wake the model and send…"
        case .stopped: return "Load a model in the Server tab to chat"
        case .error: return "Engine error — retry in the Server tab"
        }
    }

    /// Hover text for the send button — tells the user exactly why they
    /// can't send if disabled.
    private var helpText: String {
        if vm.isGenerating { return "Stop (Esc)" }
        if canSend { return "Send (↵)" }
        switch app.engineState {
        case .stopped: return "Load a model first"
        case .loading: return "Wait for the model to finish loading"
        case .error(let msg): return "Engine error: \(msg)"
        default: return "Enter a message"
        }
    }

    // MARK: - History recall

    /// Most-recent-first list of user message content in the active chat.
    private func userHistory() -> [String] {
        vm.messages
            .filter { $0.role == .user && !$0.content.isEmpty }
            .map { $0.content }
            .reversed()
    }

    private func handleUpArrow() -> KeyPress.Result {
        let history = userHistory()
        guard !history.isEmpty else { return .ignored }

        // Only intercept Up when the field is empty or we're already
        // cycling — otherwise an Up arrow inside a multi-line draft
        // should move the caret, not recall history.
        if historyIndex == nil {
            if !vm.inputText.isEmpty { return .ignored }
            draftBeforeHistory = vm.inputText
            historyIndex = 0
            vm.inputText = history[0]
            return .handled
        }
        let next = min((historyIndex ?? -1) + 1, history.count - 1)
        historyIndex = next
        vm.inputText = history[next]
        return .handled
    }

    private func handleDownArrow() -> KeyPress.Result {
        guard let idx = historyIndex else { return .ignored }
        let history = userHistory()
        let next = idx - 1
        if next < 0 {
            historyIndex = nil
            vm.inputText = draftBeforeHistory
            return .handled
        }
        historyIndex = next
        if next < history.count {
            vm.inputText = history[next]
        }
        return .handled
    }

    private func handleEscape() -> KeyPress.Result {
        // Priority 1: if we're streaming a response, Esc cancels it.
        // The Send button's `helpText` advertises "Stop (Esc)" so the
        // shortcut MUST actually fire — labeled-lie audit 2026-04-15
        // finding #6.
        if vm.isGenerating {
            vm.stop()
            return .handled
        }
        // Priority 2: pop out of history-recall mode if active.
        guard historyIndex != nil else { return .ignored }
        historyIndex = nil
        vm.inputText = draftBeforeHistory
        return .handled
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            if !vm.pendingImages.isEmpty {
                HStack(spacing: Theme.Spacing.sm) {
                    ForEach(Array(vm.pendingImages.enumerated()), id: \.offset) { i, data in
                        #if canImport(AppKit)
                        if let img = NSImage(data: data) {
                            Image(nsImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 40, height: 40)
                                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.sm))
                                .overlay(alignment: .topTrailing) {
                                    Button { vm.pendingImages.remove(at: i) } label: {
                                        Image(systemName: "xmark.circle.fill")
                                            .font(.system(size: 12))
                                            .foregroundStyle(Theme.Colors.textMid)
                                    }
                                    .buttonStyle(.plain)
                                    .offset(x: 4, y: -4)
                                }
                        }
                        #endif
                    }
                }
            }

            HStack(alignment: .bottom, spacing: Theme.Spacing.sm) {
                Button {
                    showImporter = true
                } label: {
                    Image(systemName: "photo.badge.plus")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(Theme.Colors.accent)
                        .frame(width: 28, height: 28)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .fill(Theme.Colors.surfaceHi)
                        )
                }
                .buttonStyle(.plain)
                .help("Attach images (drag, paste, or pick from disk)")

                TextField(placeholderText, text: $vm.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textHigh)
                    .lineLimit(1...8)
                    .onSubmit { if canSend { vm.send() } }
                    .onChange(of: vm.inputText) { _, _ in
                        // Any keystroke while cycling drops us out of
                        // history mode so the user can keep editing the
                        // recalled text without arrow keys stealing focus.
                        // We only want to reset when the change wasn't
                        // driven by ourselves; a simple "did anything in
                        // the field diverge from the history snapshot?"
                        // check is enough here because we set the text
                        // atomically in the arrow handlers.
                        if let idx = historyIndex, idx >= 0 {
                            let snapshot = userHistory()
                            if idx < snapshot.count && vm.inputText != snapshot[idx] {
                                historyIndex = nil
                            }
                        }
                        // Audit R5 (P2): bump the engine idle timer on
                        // every keystroke so the model doesn't deep-sleep
                        // while the user is actively typing a long prompt.
                        // ChatViewModel.bumpIdleTimer hops to the engine
                        // actor and calls IdleTimer.reset() which is
                        // idempotent + cheap to call on every char.
                        vm.bumpIdleTimer()
                    }
                    .onKeyPress(.upArrow) { handleUpArrow() }
                    .onKeyPress(.downArrow) { handleDownArrow() }
                    .onKeyPress(.escape) { handleEscape() }
                    // Cmd+Return sends even when the input is multi-line
                    // (plain Return inserts a newline in multi-line mode).
                    // Mirrors the shortcut most chat UIs use for power
                    // users and is the #1 UX gap the audit flagged.
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.command) {
                            if canSend { vm.send() }
                            return .handled
                        }
                        return .ignored
                    }

                Button {
                    if vm.isGenerating {
                        vm.stop()
                    } else {
                        vm.send()
                    }
                } label: {
                    Image(systemName: vm.isGenerating ? "stop.fill" : "arrow.up")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(Theme.Colors.textHigh)
                        .frame(width: 28, height: 28)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .fill(buttonEnabled ? Theme.Colors.accent : Theme.Colors.surfaceHi)
                        )
                        .opacity(buttonEnabled ? 1.0 : 0.5)
                }
                .buttonStyle(.plain)
                .disabled(!buttonEnabled)
                .help(helpText)
            }
            .padding(Theme.Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.lg)
                    .fill(Theme.Colors.surface)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.lg)
                            .stroke(Theme.Colors.border, lineWidth: 1)
                    )
            )
        }
        .padding(Theme.Spacing.lg)
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.image],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                for url in urls {
                    if url.startAccessingSecurityScopedResource() {
                        defer { url.stopAccessingSecurityScopedResource() }
                        if let data = try? Data(contentsOf: url) {
                            vm.pendingImages.append(data)
                        }
                    }
                }
            }
        }
        // Drag-and-drop: drop any image file (PNG, JPEG, HEIC, WebP, etc)
        // anywhere on the input bar. SwiftUI's `.onDrop` for `.image` UTType
        // resolves URLs via the standard NSItemProvider path and we reuse
        // the same `pendingImages` append flow as the file picker.
        .onDrop(of: [.image, .fileURL], isTargeted: nil) { providers in
            for provider in providers {
                if provider.canLoadObject(ofClass: NSImage.self) {
                    _ = provider.loadDataRepresentation(forTypeIdentifier: "public.image") { data, _ in
                        guard let data else { return }
                        Task { @MainActor in vm.pendingImages.append(data) }
                    }
                } else {
                    _ = provider.loadObject(ofClass: URL.self) { url, _ in
                        guard let url, let data = try? Data(contentsOf: url) else { return }
                        Task { @MainActor in vm.pendingImages.append(data) }
                    }
                }
            }
            return true
        }
    }
}
