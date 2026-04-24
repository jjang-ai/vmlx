import SwiftUI
import vMLXTheme

/// Scrollable transcript with auto-follow, scroll-up detection, and a floating
/// "↓ N new messages" button. Mirrors the React MessageList behavior at
/// `panel/src/renderer/src/components/chat/MessageList.tsx:136-144`:
///   * If the user scrolls up past a threshold, freeze auto-scroll.
///   * Track unread tail messages while frozen; show a snap-to-bottom pill.
///   * Tapping the pill (or sending a new prompt) re-engages auto-follow.
struct MessageList: View {
    @Environment(\.appLocale) private var appLocale: AppLocale
    @Bindable var vm: ChatViewModel
    @State private var userScrolledUp = false
    @State private var unreadCount = 0
    @State private var lastSeenCount = 0
    @State private var scrollOffset: CGFloat = 0
    @State private var contentHeight: CGFloat = 0
    @State private var viewportHeight: CGFloat = 0

    var body: some View {
        ScrollViewReader { proxy in
            ZStack(alignment: .bottomTrailing) {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                        ForEach(Array(vm.messages.enumerated()), id: \.element.id) { idx, message in
                            MessageBubble(
                                message: message,
                                reasoningEnabled: vm.reasoningEnabled,
                                isGenerating: vm.isGenerating,
                                isLastAssistant: isLastAssistant(idx),
                                hideToolStatus: vm.hideToolStatus,
                                onDelete: { vm.deleteMessage(message.id) },
                                onEdit: { vm.editMessage(message.id, newContent: $0) },
                                onRegenerate: { vm.regenerate(from: message.id) },
                                // Branch is hidden on the first message (idx==0
                                // branches into an empty chat, equivalent to
                                // New Chat); ChatViewModel also bails with a
                                // flash-banner defensively if called there.
                                onBranch: idx > 0
                                    ? { vm.branchSession(from: message.id) }
                                    : nil
                            )
                            .id(message.id)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(Theme.Spacing.lg)
                    .background(
                        GeometryReader { contentGeo in
                            Color.clear
                                .preference(key: ContentHeightKey.self,
                                            value: contentGeo.size.height)
                                .preference(
                                    key: ScrollOffsetKey.self,
                                    value: -contentGeo.frame(in: .named("scroll")).minY
                                )
                        }
                    )
                }
                .coordinateSpace(name: "scroll")
                .background(
                    GeometryReader { geo in
                        Color.clear
                            .preference(key: ViewportHeightKey.self, value: geo.size.height)
                    }
                )
                .background(Theme.Colors.background)
                .onPreferenceChange(ContentHeightKey.self) { contentHeight = $0 }
                .onPreferenceChange(ViewportHeightKey.self) { viewportHeight = $0 }
                .onPreferenceChange(ScrollOffsetKey.self) { offset in
                    scrollOffset = offset
                    let distanceFromBottom = max(0, contentHeight - viewportHeight - offset)
                    userScrolledUp = distanceFromBottom > 120
                    if !userScrolledUp {
                        unreadCount = 0
                        lastSeenCount = vm.messages.count
                    }
                }
                .onChange(of: vm.messages.count) { _, new in
                    if userScrolledUp {
                        unreadCount = max(0, new - lastSeenCount)
                    } else {
                        lastSeenCount = new
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                }
                // Streaming content is updated on every token — the old
                // `onChange(of: messages.last?.content)` fired with a
                // hard jump per delta, causing jitter. Instead we watch
                // contentHeight which naturally rises as text grows, and
                // use a short animation so the scroll feels smooth.
                // Throttle via `.animation(_:value:)` so SwiftUI coalesces
                // rapid updates.
                .onChange(of: contentHeight) { _, _ in
                    guard !userScrolledUp else { return }
                    withAnimation(.linear(duration: 0.08)) {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }

                if userScrolledUp {
                    scrollDownButton(proxy: proxy)
                        .padding(Theme.Spacing.lg)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }
        }
    }

    private func isLastAssistant(_ idx: Int) -> Bool {
        guard vm.messages[idx].role == .assistant else { return false }
        for j in (idx + 1)..<vm.messages.count where vm.messages[j].role == .assistant {
            return false
        }
        return true
    }

    @ViewBuilder
    private func scrollDownButton(proxy: ScrollViewProxy) -> some View {
        Button {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            unreadCount = 0
            lastSeenCount = vm.messages.count
            userScrolledUp = false
        } label: {
            HStack(spacing: Theme.Spacing.xs) {
                Image(systemName: "arrow.down")
                    .font(.system(size: 11, weight: .semibold))
                if unreadCount > 0 {
                    Text("\(unreadCount) new")
                        .font(Theme.Typography.caption)
                }
            }
            .foregroundStyle(Theme.Colors.textHigh)
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
            .background(
                Capsule()
                    .fill(Theme.Colors.surfaceHi)
                    .overlay(Capsule().stroke(Theme.Colors.border, lineWidth: 1))
            )
        }
        .buttonStyle(.plain)
        .help(L10n.Tooltip.scrollToBottom.render(appLocale))
    }
}

// MARK: - Preference keys

private struct ScrollOffsetKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
private struct ContentHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
private struct ViewportHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
