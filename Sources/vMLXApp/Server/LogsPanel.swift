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

    // Subscribe task lifetime
    @State private var streamTask: Task<Void, Never>? = nil

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
        .onChange(of: minLevel) { _, _ in restartStream() }
    }

    // MARK: - Sub-views

    private var header: some View {
        HStack {
            Text("LOGS")
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
            TextField("Search", text: $searchText)
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
            ScrollView {
                LazyVStack(alignment: .leading, spacing: compact ? 1 : 3) {
                    ForEach(filtered) { line in
                        row(line)
                            .id(line.id)
                    }
                }
                .padding(Theme.Spacing.sm)
            }
            .frame(minHeight: 220, maxHeight: 420)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.background)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .stroke(Theme.Colors.border, lineWidth: 1)
                    )
            )
            .onChange(of: filtered.count) { _, _ in
                guard autoScroll, !paused, let last = filtered.last else { return }
                withAnimation(.linear(duration: 0.08)) {
                    proxy.scrollTo(last.id, anchor: reverseOrder ? .top : .bottom)
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
        let level = minLevel
        let task = Task { @MainActor in
            // `engine.logs` is an actor-isolated `let`; crossing the actor
            // boundary to fetch it requires `await`. `subscribe` itself is
            // `nonisolated` so no further hop is needed.
            let store = await engine.logs
            let stream = store.subscribe(minLevel: level)
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
                }
            }
        }
        streamTask = task
    }

    private func restartStream() {
        lines.removeAll()
        pendingLines.removeAll()
        Task { await subscribeStream() }
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
