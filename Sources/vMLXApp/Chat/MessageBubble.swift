import SwiftUI
import vMLXTheme
import vMLXEngine

struct MessageBubble: View {
    let message: ChatMessage
    let reasoningEnabled: Bool
    /// True while the chat view-model has an in-flight stream. Drives:
    ///   * typing-dots placeholder (assistant + empty content)
    ///   * disabling edit/regenerate buttons
    let isGenerating: Bool
    /// True when this bubble is the last assistant message in the list.
    /// Regenerate is hidden on older assistant turns, mirroring the React
    /// MessageBubble's `isLastAssistant` prop.
    let isLastAssistant: Bool
    /// User-level preference — when true, suppress InlineToolCallCard
    /// rendering entirely. Mirrors ChatSettings.hideToolStatus from the
    /// chat-settings popover. Pre-iter-39 the toggle persisted to SQLite
    /// but the view unconditionally rendered tool cards — so flipping
    /// it had no visible effect.
    let hideToolStatus: Bool
    let onDelete: () -> Void
    let onEdit: (String) -> Void
    let onRegenerate: () -> Void
    /// Fork this chat from the current message into a new session, keeping
    /// everything strictly BEFORE the anchor. `nil` hides the button (for
    /// the first message, where branching is equivalent to New Chat).
    var onBranch: (() -> Void)? = nil

    @State private var hovered = false
    @Environment(\.appLocale) private var appLocale
    @State private var editing = false
    @State private var draft: String = ""
    @State private var showDeleteConfirm = false
    /// Index of the inline image the user clicked, or `nil` when no zoom
    /// overlay is visible. Bound to the full-screen overlay ZStack below.
    /// Exposed `internal` so unit tests can drive the toggle without a
    /// real mouse click.
    @State var zoomedImageIndex: Int? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            HStack(spacing: Theme.Spacing.sm) {
                Text(roleLabel)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Spacer()
                if hovered {
                    actionBar
                }
            }

            if editing {
                editor
            } else {
                if reasoningEnabled, let reasoning = message.reasoning, !reasoning.isEmpty {
                    // Reasoning phase is "active" only while the message
                    // is streaming AND no content tokens have arrived yet.
                    // As soon as the model starts emitting content, the
                    // header flips from "Thinking…" to "Thought" — even
                    // though the overall message stream is still live.
                    let reasoningActive = message.isStreaming && message.content.isEmpty
                    ReasoningBox(
                        text: reasoning,
                        isStreaming: message.isStreaming,
                        isActive: reasoningActive
                    )
                }

                // Inline tool-call cards. Rendered whenever the assistant
                // message carries tool calls — chat mode users running
                // MCP/OpenWebUI-style tools need to see invocations just
                // like Terminal mode does.
                //
                // Honors ChatSettings.hideToolStatus: power users who
                // don't want to see every sub-call (they trust the
                // model and just want the final answer) can flip the
                // chat-settings toggle and suppress this block.
                let toolCalls = message.inlineToolCalls
                if !toolCalls.isEmpty, !hideToolStatus {
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        ForEach(toolCalls) { tc in
                            InlineToolCallCard(toolCall: tc)
                        }
                    }
                }

                if !message.imageData.isEmpty {
                    HStack(spacing: Theme.Spacing.sm) {
                        ForEach(Array(message.imageData.enumerated()), id: \.offset) { idx, data in
                            #if canImport(AppKit)
                            if let nsimg = NSImage(data: data) {
                                // Tap → flag the index; the overlay ZStack
                                // at the root of the bubble picks it up.
                                Button {
                                    zoomedImageIndex = idx
                                } label: {
                                    Image(nsImage: nsimg)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(maxWidth: 240, maxHeight: 240)
                                        .cornerRadius(Theme.Radius.md)
                                }
                                .buttonStyle(.plain)
                                .help("Click to zoom")
                            }
                            #endif
                        }
                    }
                }

                if !message.videoPaths.isEmpty {
                    HStack(spacing: Theme.Spacing.sm) {
                        ForEach(Array(message.videoPaths.enumerated()), id: \.offset) { _, path in
                            videoChip(pathString: path)
                        }
                    }
                }

                contentView

                if let usage = message.usage {
                    MetricsStrip(usage: usage)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Theme.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(message.role == .user ? Theme.Colors.surface : Color.clear)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(message.role == .user ? Theme.Colors.border : Color.clear,
                                lineWidth: 1)
                )
        )
        .onHover { hovered = $0 }
        .confirmationDialog(
            "Delete this message?",
            isPresented: $showDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button(L10n.Common.delete.render(appLocale), role: .destructive) { onDelete() }
            Button(L10n.Common.cancel.render(appLocale), role: .cancel) { }
        } message: {
            Text("The message will be permanently removed from this chat. This can't be undone.")
        }
        .overlay {
            // Full-screen zoom overlay for inline images. Shown only while
            // `zoomedImageIndex != nil` and that index is in range for the
            // current message's imageData array. Tap anywhere to dismiss.
            // Theme tokens only — no hard-coded colors.
            if let idx = zoomedImageIndex,
               idx >= 0, idx < message.imageData.count {
                #if canImport(AppKit)
                if let img = NSImage(data: message.imageData[idx]) {
                    ZStack {
                        Theme.Colors.background
                            .opacity(0.92)
                            .ignoresSafeArea()
                        Image(nsImage: img)
                            .resizable()
                            .scaledToFit()
                            .padding(Theme.Spacing.xl)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { zoomedImageIndex = nil }
                    .transition(.opacity.animation(.easeOut(duration: 0.2)))
                    .onExitCommand { zoomedImageIndex = nil }
                }
                #endif
            }
        }
    }

    /// Streaming + empty assistant → typing dots. Otherwise either a streaming
    /// `StreamingTextView` (typewriter applies) or the full `MarkdownView`.
    /// Markdown is only used for finalized assistant messages — during the
    /// stream we keep the simple text path so the typewriter stays smooth.
    @ViewBuilder
    private var contentView: some View {
        if message.role == .assistant && message.isStreaming && message.content.isEmpty {
            TypingDots()
        } else if message.isStreaming {
            StreamingTextView(text: message.content, isStreaming: true)
        } else if message.role == .assistant {
            MarkdownView(text: message.content)
        } else {
            // User / system / tool — plain text, selectable.
            Text(message.content)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var editor: some View {
        VStack(alignment: .trailing, spacing: Theme.Spacing.sm) {
            TextEditor(text: $draft)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 80)
                .padding(Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surface)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
            HStack(spacing: Theme.Spacing.sm) {
                Button(L10n.Common.cancel.render(appLocale)) { editing = false }
                    .buttonStyle(.plain)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                Button(L10n.Common.save.render(appLocale)) {
                    onEdit(draft)
                    editing = false
                }
                .buttonStyle(.plain)
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.accent)
            }
        }
    }

    private var actionBar: some View {
        HStack(spacing: Theme.Spacing.sm) {
            if message.role == .user {
                Button {
                    draft = message.content
                    editing = true
                } label: {
                    Image(systemName: "pencil")
                        .font(.system(size: 10))
                        .foregroundStyle(isGenerating ? Theme.Colors.textLow.opacity(0.4)
                                                      : Theme.Colors.textLow)
                }
                .buttonStyle(.plain)
                .disabled(isGenerating)
                .help(isGenerating ? "Stop generating to edit" : "Edit message")
            }
            if message.role == .assistant && isLastAssistant {
                Button(action: onRegenerate) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 10))
                        .foregroundStyle(isGenerating ? Theme.Colors.textLow.opacity(0.4)
                                                      : Theme.Colors.textLow)
                }
                .buttonStyle(.plain)
                .disabled(isGenerating)
                .help(isGenerating ? "Stop generating to regenerate" : "Regenerate response")
            }
            if let onBranch {
                Button(action: onBranch) {
                    Image(systemName: "arrow.triangle.branch")
                        .font(.system(size: 10))
                        .foregroundStyle(isGenerating ? Theme.Colors.textLow.opacity(0.4)
                                                      : Theme.Colors.textLow)
                }
                .buttonStyle(.plain)
                .disabled(isGenerating)
                .help(isGenerating
                      ? "Stop generating to branch"
                      : "Fork this chat from before this message into a new session")
            }
            Button {
                showDeleteConfirm = true
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Colors.textLow)
            }
            .buttonStyle(.plain)
            .help("Delete message")
        }
    }

    /// Renders a persisted video attachment as a clickable chip. Click
    /// opens the file in Finder (user wanted a light-weight surface;
    /// QuickLook would need an NSViewRepresentable). Missing files
    /// (user deleted the temp staging) render the chip in muted color
    /// with a "file not found" tooltip rather than crashing.
    @ViewBuilder
    private func videoChip(pathString: String) -> some View {
        let url = URL(string: pathString) ?? URL(fileURLWithPath: pathString)
        let exists = FileManager.default.fileExists(atPath: url.path)
        Button {
            #if canImport(AppKit)
            if exists {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
            #endif
        } label: {
            HStack(spacing: Theme.Spacing.xs) {
                Image(systemName: "play.rectangle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(exists ? Theme.Colors.accent : Theme.Colors.textLow)
                Text(url.lastPathComponent)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(exists ? Theme.Colors.textMid : Theme.Colors.textLow)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 200)
            }
            .padding(.horizontal, Theme.Spacing.sm)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.sm)
                    .fill(Theme.Colors.surfaceHi.opacity(exists ? 1.0 : 0.6))
            )
        }
        .buttonStyle(.plain)
        .disabled(!exists)
        .help(exists ? "Reveal in Finder" : "File not found — temp staging may have been cleared")
    }

    private var roleLabel: String {
        switch message.role {
        case .user: return "You"
        case .assistant: return "Assistant"
        case .system: return "System"
        case .tool: return "Tool"
        }
    }
}

// MARK: - Typing Dots

/// Three dots that pulse with a 200ms stagger. Used while an assistant
/// message is streaming but no content has arrived yet.
private struct TypingDots: View {
    @State private var phase: Double = 0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Theme.Colors.textMid)
                    .frame(width: 5, height: 5)
                    .opacity(opacity(for: i))
            }
        }
        .frame(height: 16)
        .onAppear {
            withAnimation(.linear(duration: 0.9).repeatForever(autoreverses: false)) {
                phase = 1
            }
        }
    }

    private func opacity(for index: Int) -> Double {
        // Stagger by 200ms over a ~900ms loop: dot N peaks at phase = N*0.22.
        let stagger = Double(index) * 0.22
        let t = (phase - stagger).truncatingRemainder(dividingBy: 1.0)
        let s = t < 0 ? t + 1 : t
        // Triangle wave 0.3 → 1.0 → 0.3.
        let tri = s < 0.5 ? s * 2 : (1 - s) * 2
        return 0.3 + tri * 0.7
    }
}

// MARK: - Metrics Strip

/// Per-message metrics row mirroring `getMetricsItems` from
/// `panel/src/renderer/src/components/chat/chat-utils.ts:109-178`. One
/// horizontal line of `·`-separated entries in the caption / textLow style.
private struct MetricsStrip: View {
    let usage: StreamChunk.Usage
    @State private var expanded: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    expanded.toggle()
                }
            } label: {
                HStack(spacing: Theme.Spacing.xs) {
                    ForEach(Array(items.enumerated()), id: \.offset) { i, item in
                        if i > 0 {
                            Text("·")
                                .font(Theme.Typography.caption)
                                .foregroundStyle(Theme.Colors.textLow)
                        }
                        Text(item)
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textLow)
                    }
                    Spacer()
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 8, weight: .semibold))
                        .foregroundStyle(Theme.Colors.textLow)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if expanded {
                detailGrid
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, Theme.Spacing.xs)
    }

    /// Full metric breakdown. Shown on tap. Mirrors what the Electron
    /// MetricsStripExpanded component surfaces: token split, throughput,
    /// timing, cache attribution. Missing fields fall back to "—" so the
    /// grid shape stays stable whether or not a metric was captured.
    private var detailGrid: some View {
        VStack(alignment: .leading, spacing: 4) {
            Divider().background(Theme.Colors.border.opacity(0.5))
            detailColumns([
                ("Prompt tokens",     "\(usage.promptTokens)"),
                ("Completion tokens", "\(usage.completionTokens)"),
                ("Cached tokens",     "\(usage.cachedTokens)"),
                ("Total tokens",      "\(usage.promptTokens + usage.completionTokens)"),
            ])
            detailColumns([
                ("TTFT",     usage.ttftMs.map { String(format: "%.0fms", $0) } ?? "—"),
                ("Prefill",  usage.prefillMs.map { String(format: "%.0fms", $0) } ?? "—"),
                ("Total",    usage.totalMs.map { String(format: "%.2fs", $0 / 1000) } ?? "—"),
            ])
            detailColumns([
                ("Decode t/s",  usage.tokensPerSecond.map { String(format: "%.1f", $0) } ?? "—"),
                ("Prefill t/s", usage.promptTokensPerSecond.map { String(format: "%.0f", $0) } ?? "—"),
                ("Cache tier",  usage.cacheDetail ?? "—"),
            ])
        }
        .padding(Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                .fill(Theme.Colors.surface)
        )
    }

    private func detailColumns(_ pairs: [(String, String)]) -> some View {
        HStack(alignment: .top, spacing: Theme.Spacing.md) {
            ForEach(Array(pairs.enumerated()), id: \.offset) { _, pair in
                VStack(alignment: .leading, spacing: 1) {
                    Text(pair.0)
                        .font(.system(size: 9))
                        .foregroundStyle(Theme.Colors.textLow)
                    Text(pair.1)
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textMid)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var items: [String] {
        var out: [String] = []
        if usage.completionTokens > 0 {
            out.append("\(usage.completionTokens) tokens")
        }
        if let tps = usage.tokensPerSecond, tps > 0 {
            // Partial emissions during streaming get a "live" marker so
            // users can see rolling tok/s mid-generation. Final usage
            // shows the authoritative number with no marker.
            let suffix = usage.isPartial ? " live" : ""
            out.append(String(format: "%.1f t/s%@", tps, suffix))
        }
        if let pps = usage.promptTokensPerSecond, pps > 0 {
            out.append(String(format: "%.0f pp/s", pps))
        }
        if usage.promptTokens > 0 {
            var s = "\(usage.promptTokens) prompt"
            if usage.cachedTokens > 0 {
                // Only print the tier when we actually have a hit — a
                // "miss" label on a 0-cached line would be confusing.
                let tier = usage.cacheDetail.flatMap {
                    $0 == "miss" ? nil : $0
                }
                let detail = tier.map { " \($0)" } ?? ""
                s += " (\(usage.cachedTokens)\(detail) cached)"
            }
            out.append(s)
        }
        if let ttft = usage.ttftMs, ttft > 0 {
            out.append(String(format: "%.0fms TTFT", ttft))
        }
        if let prefill = usage.prefillMs, prefill > 0 {
            out.append(String(format: "%.0fms prefill", prefill))
        }
        if let total = usage.totalMs, total > 0 {
            out.append(String(format: "%.2fs total", total / 1000))
        }
        return out
    }
}
