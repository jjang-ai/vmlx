import SwiftUI
import vMLXTheme

/// Renderer-side typewriter for streamed assistant text.
///
/// Ported 1:1 from the React `useTypewriter` hook in
/// `panel/src/renderer/src/components/chat/MessageBubble.tsx:99-142`.
/// Lessons (see MEMORY feedback_streaming_throttle.md):
///   * Throttling MUST live at the renderer. Upstream (engine, IPC, server)
///     batches in unpredictable bursts — main-process schemes failed 3 times.
///   * The algorithm:  on every text change, target = full.length.  On each
///     ~16ms tick, advance `displayed` by `max(1, ceil(remaining / 12))` so
///     the buffer drains in ~200ms regardless of burst size.
///   * If content shrinks (edit, regenerate), snap to match.
///   * If we lag more than 1 character behind a non-streaming message, snap.
struct StreamingTextView: View {
    let text: String
    var isStreaming: Bool = false

    @State private var displayed: String = ""
    @State private var displayedLen: Int = 0
    @State private var tickTask: Task<Void, Never>? = nil

    var body: some View {
        Text(displayed)
            .font(Theme.Typography.body)
            .foregroundStyle(Theme.Colors.textHigh)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .onAppear {
                displayed = text
                displayedLen = text.count
            }
            .onChange(of: text) { _, _ in onTextChange() }
            .onChange(of: isStreaming) { _, streaming in
                if !streaming {
                    tickTask?.cancel()
                    displayed = text
                    displayedLen = text.count
                }
            }
    }

    private func onTextChange() {
        let full = text
        let fullLen = full.count

        if !isStreaming {
            tickTask?.cancel()
            displayed = full
            displayedLen = fullLen
            return
        }
        // Content shrunk (rare correction) — snap.
        if fullLen < displayedLen {
            displayed = full
            displayedLen = fullLen
            return
        }
        if displayedLen >= fullLen { return }
        startTick()
    }

    private func startTick() {
        tickTask?.cancel()
        tickTask = Task { @MainActor in
            while !Task.isCancelled {
                let full = text
                let fullLen = full.count
                if displayedLen >= fullLen { return }
                let remaining = fullLen - displayedLen
                // Catch up within ~200ms (12 frames at 60fps), min 1 char/frame.
                let chars = max(1, Int((Double(remaining) / 12.0).rounded(.up)))
                let newLen = min(displayedLen + chars, fullLen)
                let endIdx = full.index(full.startIndex, offsetBy: newLen)
                displayed = String(full[..<endIdx])
                displayedLen = newLen
                if newLen >= fullLen { return }
                try? await Task.sleep(nanoseconds: 16_000_000) // ~60fps
            }
        }
    }
}
