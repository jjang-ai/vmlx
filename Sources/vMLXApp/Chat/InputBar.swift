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
            || !vm.pendingVideos.isEmpty
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

    // SwiftUI body deliberately broken into a small number of computed
    // subviews. Keeping everything in one giant builder sends the Swift
    // type-checker into exponential territory ("unable to type-check in
    // reasonable time"). Each helper below stays well under that limit.

    @ViewBuilder
    private var attachedImages: some View {
        if !vm.pendingImages.isEmpty {
            HStack(spacing: Theme.Spacing.sm) {
                ForEach(Array(vm.pendingImages.enumerated()), id: \.offset) { i, data in
                    #if canImport(AppKit)
                    if let img = NSImage(data: data) {
                        thumbnailView(img: img, index: i)
                    }
                    #endif
                }
            }
        }
    }

    #if canImport(AppKit)
    @ViewBuilder
    private func thumbnailView(img: NSImage, index: Int) -> some View {
        Image(nsImage: img)
            .resizable()
            .scaledToFill()
            .frame(width: 40, height: 40)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.sm))
            .overlay(alignment: .topTrailing) {
                Button { vm.pendingImages.remove(at: index) } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.Colors.textMid)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Remove image \(index + 1)")
                .offset(x: 4, y: -4)
            }
    }
    #endif

    /// Pending-video thumbnails. Each chip shows a play-arrow badge
    /// + the filename tail + an x-to-remove button. Keeps the chat
    /// input bar a consistent media-attach surface. A background
    /// task generates an AVAssetImageGenerator preview frame so users
    /// see the first frame rather than a filmstrip placeholder.
    @ViewBuilder
    private var attachedVideos: some View {
        if !vm.pendingVideos.isEmpty {
            HStack(spacing: Theme.Spacing.sm) {
                ForEach(Array(vm.pendingVideos.enumerated()), id: \.offset) { i, url in
                    videoChip(url: url, index: i)
                }
            }
        }
    }

    @ViewBuilder
    private func videoChip(url: URL, index: Int) -> some View {
        HStack(spacing: Theme.Spacing.xs) {
            Image(systemName: "play.rectangle.fill")
                .font(.system(size: 14))
                .foregroundStyle(Theme.Colors.accent)
            Text(url.lastPathComponent)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: 140)
            Button {
                // Iter-23: remove AND delete the underlying temp file.
                // Attach-then-discard was leaking the staged copy
                // in /tmp/vmlx-chat-video-<UUID>.<ext> until the
                // 14-day sweep. Only delete if the URL is inside the
                // OS tempDir (we staged it there ourselves) to avoid
                // nuking source files a user might have pointed at
                // via future shortcut paths. Safe to blindly delete
                // temp-dir entries because they're created per-attach
                // with UUID filenames.
                let url = vm.pendingVideos[index]
                vm.pendingVideos.remove(at: index)
                let tmp = FileManager.default.temporaryDirectory
                    .standardizedFileURL.path
                if url.standardizedFileURL.path.hasPrefix(tmp) {
                    try? FileManager.default.removeItem(at: url)
                }
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.Colors.textMid)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove video \(index + 1)")
        }
        .padding(.horizontal, Theme.Spacing.sm)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                .fill(Theme.Colors.surfaceHi)
        )
    }

    private var attachButton: some View {
        Button { showImporter = true } label: {
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
        .accessibilityLabel("Attach images")
        .accessibilityHint("Opens a file picker to attach images to the next message")
        .help("Attach images (drag, paste, or pick from disk)")
    }

    private var sendButton: some View {
        Button {
            if vm.isGenerating { vm.stop() } else { vm.send() }
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
        .accessibilityLabel(vm.isGenerating ? "Stop generation" : "Send message")
        .accessibilityHint(vm.isGenerating
            ? "Cancels the in-flight response"
            : "Sends the current prompt to the loaded model")
        .help(helpText)
    }

    private var textField: some View {
        TextField(placeholderText, text: $vm.inputText, axis: .vertical)
            .textFieldStyle(.plain)
            .font(Theme.Typography.body)
            .foregroundStyle(Theme.Colors.textHigh)
            .lineLimit(1...8)
            .onSubmit { if canSend { vm.send() } }
            .onChange(of: vm.inputText) { _, _ in handleTextChange() }
            .onKeyPress(.upArrow) { handleUpArrow() }
            .onKeyPress(.downArrow) { handleDownArrow() }
            .onKeyPress(.escape) { handleEscape() }
            .onKeyPress(.return, phases: .down) { press in
                if press.modifiers.contains(.command) {
                    if canSend { vm.send() }
                    return .handled
                }
                return .ignored
            }
    }

    private func handleTextChange() {
        if let idx = historyIndex, idx >= 0 {
            let snapshot = userHistory()
            if idx < snapshot.count && vm.inputText != snapshot[idx] {
                historyIndex = nil
            }
        }
    }

    private var inputRow: some View {
        HStack(alignment: .bottom, spacing: Theme.Spacing.sm) {
            attachButton
            textField
            sendButton
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

    /// Rough token count (≈ chars / 4, matching OpenAI's ballpark guidance
    /// for English). Hidden on empty input so the bar stays clean; shown
    /// as a dim caption when the user is composing. Flips amber at 2k
    /// tokens and red at 8k as a visible context-pressure hint.
    /// Production note: a true tokenizer round-trip would be more accurate
    /// but requires an async hop to the Engine actor and doesn't feel
    /// responsive on every keystroke. The ≈/4 heuristic is good-enough
    /// for the "don't blow past context" use case.
    private var tokenCountHint: some View {
        Group {
            let chars = vm.inputText.count
            if chars > 0 {
                let approxTokens = (chars + 3) / 4
                let color: Color = approxTokens > 8000 ? Theme.Colors.danger
                    : approxTokens > 2000 ? Theme.Colors.warning
                    : Theme.Colors.textLow
                HStack(spacing: Theme.Spacing.xs) {
                    Spacer()
                    Text("≈\(approxTokens) tok · \(chars) chars")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(color)
                }
                .padding(.horizontal, Theme.Spacing.md)
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            attachedImages
            attachedVideos
            inputRow
            tokenCountHint
        }
        .padding(Theme.Spacing.lg)
        .fileImporter(
            isPresented: $showImporter,
            // Iter-15: accept both images and videos. The engine/VL path
            // already supports video_url ContentParts; the gap was UI-
            // only. Route based on UTI below.
            allowedContentTypes: [.image, .movie, .mpeg4Movie, .quickTimeMovie],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                for url in urls {
                    if url.startAccessingSecurityScopedResource() {
                        defer { url.stopAccessingSecurityScopedResource() }
                        // Preserve security scope by copying into temp
                        // for videos — the picker URL's scope ends at
                        // defer{} so a later send() dispatch wouldn't
                        // be able to read from it.
                        if Self.isVideoURL(url) {
                            if let staged = Self.stageVideoIntoTemp(url) {
                                vm.pendingVideos.append(staged)
                            }
                        } else if let data = try? Data(contentsOf: url) {
                            vm.pendingImages.append(data)
                        }
                    }
                }
            }
        }
        // Drag-and-drop: images and videos. Images inline as Data;
        // videos stage into temp (see stageVideoIntoTemp for the same
        // security-scope argument as the picker path).
        .onDrop(of: [.image, .movie, .mpeg4Movie, .quickTimeMovie, .fileURL],
                isTargeted: nil) { providers in
            for provider in providers {
                // Video-by-URL provider (Finder drop).
                if provider.hasItemConformingToTypeIdentifier("public.movie") {
                    _ = provider.loadObject(ofClass: URL.self) { url, _ in
                        guard let url else { return }
                        if let staged = Self.stageVideoIntoTemp(url) {
                            Task { @MainActor in vm.pendingVideos.append(staged) }
                        }
                    }
                    continue
                }
                if provider.canLoadObject(ofClass: NSImage.self) {
                    _ = provider.loadDataRepresentation(forTypeIdentifier: "public.image") { data, _ in
                        guard let data else { return }
                        Task { @MainActor in vm.pendingImages.append(data) }
                    }
                } else {
                    _ = provider.loadObject(ofClass: URL.self) { url, _ in
                        guard let url else { return }
                        if Self.isVideoURL(url) {
                            if let staged = Self.stageVideoIntoTemp(url) {
                                Task { @MainActor in vm.pendingVideos.append(staged) }
                            }
                        } else if let data = try? Data(contentsOf: url) {
                            Task { @MainActor in vm.pendingImages.append(data) }
                        }
                    }
                }
            }
            return true
        }
    }

    /// True when `url` looks like a video the engine's video_url
    /// ContentPart path can consume. The set matches the MIME types
    /// AVFoundation handles + the file-extension set in the picker.
    private static func isVideoURL(_ url: URL) -> Bool {
        let videoExt: Set<String> = [
            "mp4", "m4v", "mov", "qt", "webm", "mkv", "avi"
        ]
        return videoExt.contains(url.pathExtension.lowercased())
    }

    /// Copy a picked video into the OS temp dir so the security-
    /// scoped original can go out of scope without breaking later
    /// reads. Preserves filename. Returns `nil` on copy failure (the
    /// attach is silently dropped — InputBar uses NSImage-style
    /// best-effort semantics).
    private static func stageVideoIntoTemp(_ url: URL) -> URL? {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-chat-video-\(UUID().uuidString)")
            .appendingPathExtension(url.pathExtension)
        do {
            try FileManager.default.copyItem(at: url, to: tmp)
            return tmp
        } catch {
            return nil
        }
    }
}
