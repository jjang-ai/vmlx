import SwiftUI
#if canImport(AppKit)
import AppKit
#endif
import vMLXTheme

/// Collapsible `<think>…</think>` reasoning block.
///
/// Behavior:
/// - **Auto-expands** while `isStreaming == true` so the user sees the
///   reasoning as it arrives.
/// - **Auto-collapses** on the streaming→done transition (once) so the
///   finished transcript stays compact. User can click to re-expand.
/// - Header shows "Thinking…" with a subtle pulsing dot while
///   `isActive == true`, "Thought" otherwise.
/// - `isStreaming` controls auto-collapse on the streaming→done
///   transition. `isActive` is a narrower signal — true only while the
///   *reasoning phase* is live (i.e. no content tokens have arrived yet).
///   Callers should flip `isActive` to false as soon as content begins
///   streaming, so the header label drops "Thinking…" immediately even
///   though the overall message stream is still live.
/// - Parent is responsible for hiding this entirely when reasoning is off.
struct ReasoningBox: View {
    let text: String
    let isStreaming: Bool
    let isActive: Bool
    @State private var expanded: Bool = true     // default expanded
    @State private var userCollapsed: Bool = false
    @State private var pulse: Double = 0.3
    /// Brief "Copied!" acknowledgment shown for 1.2s after the copy
    /// button fires so the user gets a visible confirmation. Replaces
    /// the char-count label during the acknowledgment window.
    @State private var copiedFlash: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            if expanded {
                Text(text)
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textMid)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, Theme.Spacing.md)
                    .padding(.bottom, Theme.Spacing.md)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(borderColor, lineWidth: 1)
                )
        )
        .onChange(of: isStreaming) { wasStreaming, nowStreaming in
            if wasStreaming && !nowStreaming && !userCollapsed {
                // Auto-collapse when the reasoning phase completes,
                // unless the user manually expanded it during stream.
                withAnimation(.easeInOut(duration: 0.25)) {
                    expanded = false
                }
            } else if !wasStreaming && nowStreaming {
                // New reasoning turn — reset state.
                userCollapsed = false
                expanded = true
            }
        }
        .onAppear {
            // If we render mid-stream (e.g. session restore during a
            // live turn), start pulsing immediately.
            if isActive { startPulse() }
        }
        .onChange(of: isActive) { _, nowActive in
            if nowActive { startPulse() }
        }
    }

    // MARK: - Header

    private var header: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.18)) {
                expanded.toggle()
                if !expanded { userCollapsed = true }
                else { userCollapsed = false }
            }
        } label: {
            HStack(spacing: Theme.Spacing.sm) {
                Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Theme.Colors.textLow)
                    .frame(width: 10)
                if isActive {
                    // Pulsing dot to signal "something is happening".
                    Circle()
                        .fill(Theme.Colors.accent)
                        .frame(width: 6, height: 6)
                        .opacity(pulse)
                }
                Text(isActive ? "Thinking…" : "Thought")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(isActive ? Theme.Colors.textHigh : Theme.Colors.textMid)
                Spacer()
                if copiedFlash {
                    Text("Copied!")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.success)
                        .transition(.opacity)
                } else if !isActive && !text.isEmpty {
                    Text("\(text.count) chars")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                }
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // Copy-reasoning-only button on the trailing edge of the
        // header. Appears only when reasoning has settled (non-
        // empty text, not actively streaming) so it doesn't steal
        // attention while the model is thinking. Ships independent
        // of the outer `MessageBubble` copy button which copies the
        // full message.
        .overlay(alignment: .trailing) {
            if !isActive && !text.isEmpty {
                Button {
                    copyReasoningToClipboard()
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.Colors.textLow)
                        .padding(.trailing, Theme.Spacing.sm)
                        .padding(.vertical, Theme.Spacing.sm)
                }
                .buttonStyle(.plain)
                .help("Copy reasoning only")
            }
        }
    }

    /// Writes `text` to the macOS pasteboard and shows a brief
    /// "Copied!" acknowledgment in the header.
    private func copyReasoningToClipboard() {
        #if canImport(AppKit)
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        #endif
        withAnimation(.easeInOut(duration: 0.15)) {
            copiedFlash = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            withAnimation(.easeInOut(duration: 0.2)) {
                copiedFlash = false
            }
        }
    }

    private var borderColor: Color {
        isActive ? Theme.Colors.accent.opacity(0.4) : Theme.Colors.border
    }

    /// Drive the header dot pulse via a repeating animation. Stopped
    /// implicitly when `isStreaming` flips off (the dot is removed).
    private func startPulse() {
        pulse = 0.3
        withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) {
            pulse = 1.0
        }
    }
}
