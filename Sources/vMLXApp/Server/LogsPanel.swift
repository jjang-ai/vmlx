import SwiftUI
import AppKit
import vMLXEngine
import vMLXTheme

/// Live-tail log viewer for the Server screen.
///
/// Subscribes to `Engine.logs.subscribe(...)` on appear and renders a
/// filterable, pausable, exportable feed. Mirrors the logs drawer in the
/// Electron app (`panel/src/renderer/components/ServerLogs.tsx`) while
/// staying strictly on Theme tokens.
struct LogsPanel: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale

    // Filter state
    @State private var minLevel: LogStore.Level = .info
    @State private var selectedCategories: Set<String> = []
    @State private var searchText: String = ""

    // Feed state
    @State private var lines: [LogStore.Line] = []
    @State private var pendingLines: [LogStore.Line] = []
    @State private var paused: Bool = false
    @State private var autoScroll: Bool = true
    @State private var compact: Bool = false
    @State private var reverseOrder: Bool = false

    // User-scroll detection for the floating "↓" pill that mirrors
    // Chat/Terminal scroll behavior. When the user scrolls up past the
    // 60px threshold, autoScroll freezes and a pill with the unread
    // count appears. Tapping it (or hitting Resume) snaps back.
    @State private var userScrolledUp: Bool = false
    @State private var unreadInScroll: Int = 0
    @State private var contentHeight: CGFloat = 0
    @State private var viewportHeight: CGFloat = 0
    @State private var scrollOffset: CGFloat = 0

    // Subscribe task lifetime
    @State private var streamTask: Task<Void, Never>? = nil

    /// Stream subscription level — always `.trace` so the panel sees
    /// every event the engine emits. Per-view filtering happens
    /// client-side via `visibleLines`. This way changing the level
    /// picker is instant and never drops scrollback (UI-8).
    private let streamMinLevel: LogStore.Level = .trace

    private let allCategories = ["engine", "server", "model", "cache", "tool", "mcp"]
    private let maxBuffer = 2000

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            header
            filterBar
            feedList
            footerBar
        }
        .padding(Theme.Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
        .task { await subscribeStream() }
        .onDisappear {
            streamTask?.cancel()
            streamTask = nil
        }
        // UI-8: changing the level picker no longer drops the buffer.
        // The stream subscribes at `.trace` and we filter client-side,
        // so toggling INFO ↔ DEBUG is purely a re-filter.
    }

    // MARK: - Sub-views

    private var header: some View {
        HStack {
            Text(L10n.ServerUI.logs.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Spacer()
            if !pendingLines.isEmpty {
                Button {
                    flushPending()
                } label: {
                    Text("\u{2193} \(pendingLines.count) new")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .padding(.horizontal, Theme.Spacing.sm)
                        .padding(.vertical, 2)
                        .background(
                            Capsule().fill(Theme.Colors.accent)
                        )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var filterBar: some View {
        HStack(spacing: Theme.Spacing.sm) {
            // Level picker
            Picker("", selection: $minLevel) {
                ForEach(LogStore.Level.allCases, id: \.self) { lvl in
                    Text(lvl.rawValue.uppercased()).tag(lvl)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(width: 110)

            // Category chips
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: Theme.Spacing.xs) {
                    ForEach(allCategories, id: \.self) { cat in
                        categoryChip(cat)
                    }
                }
            }

            // Search
            TextField(L10n.Common.search.render(appLocale), text: $searchText)
                .textFieldStyle(.plain)
                .font(Theme.Typography.body)
                .padding(.horizontal, Theme.Spacing.sm)
                .padding(.vertical, Theme.Spacing.xs)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.sm)
                        .fill(Theme.Colors.surfaceHi)
                )
                .frame(maxWidth: 200)
        }
    }

    @ViewBuilder
    private func categoryChip(_ cat: String) -> some View {
        let active = selectedCategories.contains(cat)
        Button {
            if active { selectedCategories.remove(cat) }
            else { selectedCategories.insert(cat) }
        } label: {
            Text(cat)
                .font(Theme.Typography.caption)
                .foregroundStyle(active ? Theme.Colors.textHigh : Theme.Colors.textMid)
                .padding(.horizontal, Theme.Spacing.sm)
                .padding(.vertical, 2)
                .background(
                    Capsule()
                        .fill(active ? Theme.Colors.accent : Theme.Colors.surfaceHi)
                )
        }
        .buttonStyle(.plain)
    }

    private var feedList: some View {
        let filtered = visibleLines
        return ScrollViewReader { proxy in
            ZStack(alignment: .bottomTrailing) {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: compact ? 1 : 3) {
                        ForEach(filtered) { line in
                            row(line)
                                .id(line.id)
                        }
                    }
                    .padding(Theme.Spacing.sm)
                    .background(
                        GeometryReader { contentGeo in
                            Color.clear
                                .preference(key: LogContentHeightKey.self,
                                            value: contentGeo.size.height)
                                .preference(
                                    key: LogScrollOffsetKey.self,
                                    value: -contentGeo.frame(in: .named("logScroll")).minY
                                )
                        }
                    )
                }
                .coordinateSpace(name: "logScroll")
                .background(
                    GeometryReader { geo in
                        Color.clear
                            .preference(key: LogViewportHeightKey.self,
                                        value: geo.size.height)
                    }
                )
                .frame(minHeight: 220, maxHeight: 420)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.background)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
                .onPreferenceChange(LogContentHeightKey.self) { contentHeight = $0 }
                .onPreferenceChange(LogViewportHeightKey.self) { viewportHeight = $0 }
                .onPreferenceChange(LogScrollOffsetKey.self) { offset in
                    scrollOffset = offset
                    let distance = max(0, contentHeight - viewportHeight - offset)
                    let nowUp = distance > 60
                    if !nowUp && userScrolledUp {
                        // User snapped back to the bottom — clear the pill.
                        unreadInScroll = 0
                    }
                    userScrolledUp = nowUp
                }
                .onChange(of: filtered.count) { _, _ in
                    guard autoScroll, !paused, !userScrolledUp,
                          let last = filtered.last else { return }
                    withAnimation(.linear(duration: 0.08)) {
                        proxy.scrollTo(last.id, anchor: reverseOrder ? .top : .bottom)
                    }
                }

                // Floating "scroll to bottom" pill — appears whenever
                // the user scrolls up past the threshold, regardless of
                // pause state. Tapping snaps back and clears the unread
                // counter. UI-2.
                if userScrolledUp, let last = filtered.last {
                    Button {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last.id,
                                           anchor: reverseOrder ? .top : .bottom)
                        }
                        userScrolledUp = false
                        unreadInScroll = 0
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "arrow.down")
                                .font(.system(size: 11, weight: .semibold))
                            if unreadInScroll > 0 {
                                Text("\(unreadInScroll) new")
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
                    .padding(Theme.Spacing.md)
                    .help(L10n.Tooltip.scrollToNewestLog.render(appLocale))
                }
            }
        }
    }

    @ViewBuilder
    private func row(_ line: LogStore.Line) -> some View {
        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
            Text(Self.formatTime(line.timestamp))
                .foregroundStyle(Theme.Colors.textLow)
            Text(line.level.rawValue.uppercased())
                .foregroundStyle(color(for: line.level))
                .frame(width: 44, alignment: .leading)
            Text(line.category)
                .foregroundStyle(Theme.Colors.textMid)
                .frame(width: 56, alignment: .leading)
            Text(line.message)
                .foregroundStyle(color(for: line.level))
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
        .font(Theme.Typography.mono)
        .lineLimit(compact ? 1 : nil)
    }

    private var footerBar: some View {
        HStack(spacing: Theme.Spacing.sm) {
            toolbarButton(paused ? "play.fill" : "pause.fill",
                          label: paused ? "Resume" : "Pause") {
                paused.toggle()
                if !paused { flushPending() }
            }
            toolbarButton(autoScroll ? "arrow.down.to.line" : "arrow.down",
                          label: autoScroll ? "Auto-scroll on" : "Auto-scroll off") {
                autoScroll.toggle()
            }
            toolbarButton(compact ? "rectangle.compress.vertical" : "rectangle.expand.vertical",
                          label: compact ? "Compact" : "Comfortable") {
                compact.toggle()
            }
            toolbarButton(reverseOrder ? "arrow.up" : "arrow.down",
                          label: reverseOrder ? "Newest first" : "Oldest first") {
                reverseOrder.toggle()
            }
            Spacer()
            toolbarButton("square.and.arrow.up", label: "Export") {
                Task { await exportLogs() }
            }
            toolbarButton("trash", label: "Clear") {
                Task {
                    await app.engine.logs.clear()
                    lines.removeAll()
                    pendingLines.removeAll()
                }
            }
        }
    }

    @ViewBuilder
    private func toolbarButton(
        _ icon: String,
        label: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: Theme.Spacing.xs) {
                Image(systemName: icon)
                Text(label)
            }
            .font(Theme.Typography.caption)
            .foregroundStyle(Theme.Colors.textMid)
            .padding(.horizontal, Theme.Spacing.sm)
            .padding(.vertical, Theme.Spacing.xs)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.sm)
                    .fill(Theme.Colors.surfaceHi)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Behavior

    private var visibleLines: [LogStore.Line] {
        let filter = LogFilter(
            minLevel: minLevel,
            categories: selectedCategories.isEmpty ? nil : selectedCategories,
            contains: searchText.isEmpty ? nil : searchText,
            since: nil
        )
        let matched = lines.filter(filter.matches)
        return reverseOrder ? matched.reversed() : matched
    }

    private func subscribeStream() async {
        streamTask?.cancel()
        let engine = app.engine
        let task = Task { @MainActor in
            // `engine.logs` is an actor-isolated `let`; crossing the actor
            // boundary to fetch it requires `await`. `subscribe` itself is
            // `nonisolated` so no further hop is needed.
            let store = await engine.logs
            // UI-8: subscribe at TRACE so we have everything client-side
            // and changing the level picker is just a filter, not a
            // resubscribe-and-drop-buffer cycle.
            let stream = store.subscribe(minLevel: streamMinLevel)
            for await line in stream {
                if Task.isCancelled { return }
                if paused {
                    pendingLines.append(line)
                    if pendingLines.count > maxBuffer {
                        pendingLines.removeFirst(pendingLines.count - maxBuffer)
                    }
                } else {
                    lines.append(line)
                    if lines.count > maxBuffer {
                        lines.removeFirst(lines.count - maxBuffer)
                    }
                    // Track "while-scrolled-up" unread count so the pill
                    // can show "↓ N new" without affecting the count of
                    // explicitly-paused buffered lines.
                    if userScrolledUp,
                       LogFilter(minLevel: minLevel,
                                 categories: selectedCategories.isEmpty ? nil : selectedCategories,
                                 contains: searchText.isEmpty ? nil : searchText,
                                 since: nil)
                       .matches(line)
                    {
                        unreadInScroll += 1
                    }
                }
            }
        }
        streamTask = task
    }

    private func flushPending() {
        lines.append(contentsOf: pendingLines)
        pendingLines.removeAll()
        if lines.count > maxBuffer {
            lines.removeFirst(lines.count - maxBuffer)
        }
    }

    private func exportLogs() async {
        let data = await app.engine.logs.export()
        await MainActor.run {
            let panel = NSSavePanel()
            panel.allowedContentTypes = []
            panel.nameFieldStringValue = "vmlx-logs-\(Int(Date().timeIntervalSince1970)).jsonl"
            panel.canCreateDirectories = true
            if panel.runModal() == .OK, let url = panel.url {
                try? data.write(to: url)
            }
        }
    }

    private func color(for level: LogStore.Level) -> Color {
        switch level {
        case .trace: return Theme.Colors.textLow
        case .debug: return Theme.Colors.textMid
        case .info:  return Theme.Colors.textHigh
        case .warn:  return Theme.Colors.warning
        case .error: return Theme.Colors.danger
        }
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()
    private static func formatTime(_ d: Date) -> String {
        timeFormatter.string(from: d)
    }
}

// MARK: - Preference keys (UI-2)

private struct LogContentHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
private struct LogViewportHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
private struct LogScrollOffsetKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
