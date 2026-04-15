import SwiftUI
import vMLXEngine
import vMLXTheme

/// Live request-log stream for the API tab.
///
/// Subscribes to `Engine.logs` filtered to `category == "server"` so every
/// Hummingbird access-log entry surfaces here without the user having to
/// open the Server tab's broader LogsPanel. Displays method + path +
/// status badge + latency, color-coded by status class (2xx / 3xx / 4xx
/// / 5xx). Auto-scrolls to the tail unless the user scrolls up; scroll
/// pill snaps back.
///
/// Read-only view — this panel never mutates the log buffer. `engine.logs`
/// is the single source of truth; the Server tab's LogsPanel sees the
/// same lines with its full filter stack.
struct RequestLogPanel: View {
    @Environment(AppState.self) private var app

    @State private var lines: [LogStore.Line] = []
    @State private var paused: Bool = false
    @State private var userScrolledUp: Bool = false
    @State private var unread: Int = 0
    @State private var contentHeight: CGFloat = 0
    @State private var viewportHeight: CGFloat = 0
    @State private var scrollOffset: CGFloat = 0
    @State private var streamTask: Task<Void, Never>? = nil

    private let maxBuffer = 500

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            header
            feedList
            footer
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .stroke(Theme.Colors.border, lineWidth: 1)
        )
        .task { await subscribe() }
        .onDisappear {
            streamTask?.cancel()
            streamTask = nil
        }
    }

    // MARK: - Subviews

    private var header: some View {
        HStack {
            Image(systemName: "network")
                .foregroundStyle(Theme.Colors.accent)
            Text("Live request log")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
            Text("\(lines.count) lines")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
    }

    private var feedList: some View {
        ScrollViewReader { proxy in
            ZStack(alignment: .bottomTrailing) {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(lines) { line in
                            row(line)
                                .id(line.id)
                        }
                    }
                    .padding(Theme.Spacing.sm)
                    .background(
                        GeometryReader { contentGeo in
                            Color.clear
                                .preference(key: ReqLogContentHeight.self,
                                            value: contentGeo.size.height)
                                .preference(
                                    key: ReqLogScrollOffset.self,
                                    value: -contentGeo.frame(in: .named("reqLog")).minY
                                )
                        }
                    )
                }
                .coordinateSpace(name: "reqLog")
                .background(
                    GeometryReader { geo in
                        Color.clear
                            .preference(key: ReqLogViewport.self,
                                        value: geo.size.height)
                    }
                )
                .frame(minHeight: 180, maxHeight: 280)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.background)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
                .onPreferenceChange(ReqLogContentHeight.self) { contentHeight = $0 }
                .onPreferenceChange(ReqLogViewport.self) { viewportHeight = $0 }
                .onPreferenceChange(ReqLogScrollOffset.self) { offset in
                    scrollOffset = offset
                    let distance = max(0, contentHeight - viewportHeight - offset)
                    let up = distance > 60
                    if !up && userScrolledUp { unread = 0 }
                    userScrolledUp = up
                }
                .onChange(of: lines.count) { _, _ in
                    guard !paused, !userScrolledUp, let last = lines.last else { return }
                    withAnimation(.linear(duration: 0.08)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }

                if userScrolledUp, let last = lines.last {
                    Button {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                        userScrolledUp = false
                        unread = 0
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "arrow.down")
                                .font(.system(size: 11, weight: .semibold))
                            if unread > 0 {
                                Text("\(unread) new")
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
                }
            }
        }
    }

    @ViewBuilder
    private func row(_ line: LogStore.Line) -> some View {
        let parts = Self.parse(line.message)
        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
            Text(Self.formatTime(line.timestamp))
                .foregroundStyle(Theme.Colors.textLow)
            Text(parts.method)
                .foregroundStyle(methodColor(parts.method))
                .frame(width: 60, alignment: .leading)
            Text(parts.path)
                .foregroundStyle(Theme.Colors.textHigh)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: Theme.Spacing.sm)
            if let status = parts.status {
                Text("\(status)")
                    .foregroundStyle(statusColor(status))
                    .font(Theme.Typography.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(
                        Capsule().fill(statusColor(status).opacity(0.15))
                    )
            }
            if let ms = parts.ms {
                Text("\(Self.formatMs(ms))")
                    .foregroundStyle(Theme.Colors.textMid)
                    .frame(width: 60, alignment: .trailing)
            }
        }
        .font(Theme.Typography.mono)
        .textSelection(.enabled)
    }

    private var footer: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Button(action: { paused.toggle() }) {
                HStack(spacing: 4) {
                    Image(systemName: paused ? "play.fill" : "pause.fill")
                    Text(paused ? "Resume" : "Pause")
                }
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
            }
            .buttonStyle(.plain)
            Spacer()
            Text("Shows the last \(maxBuffer) server-category log lines. Server tab's LogsPanel has full filters.")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
    }

    // MARK: - Behavior

    private func subscribe() async {
        streamTask?.cancel()
        let engine = app.engine
        let task = Task { @MainActor in
            let store = await engine.logs
            let stream = store.subscribe(minLevel: .info)
            for await line in stream {
                if Task.isCancelled { return }
                guard line.category == "server" else { continue }
                if paused {
                    if userScrolledUp { unread += 1 }
                    continue
                }
                lines.append(line)
                if lines.count > maxBuffer {
                    lines.removeFirst(lines.count - maxBuffer)
                }
                if userScrolledUp { unread += 1 }
            }
        }
        streamTask = task
    }

    // MARK: - Formatting

    /// Parse the RequestLoggerMiddleware format
    /// "{METHOD} {path} -> {status} ({ms}ms)" into structured fields.
    /// Falls back to raw text when the line doesn't match (e.g. server
    /// lifecycle messages like "Starting HTTP server on ...").
    private static func parse(_ message: String) -> (method: String, path: String, status: Int?, ms: Double?) {
        // Split on " -> " then " ("
        let parts = message.components(separatedBy: " -> ")
        guard parts.count == 2 else {
            return (method: "", path: message, status: nil, ms: nil)
        }
        let methodPath = parts[0].components(separatedBy: " ")
        let method = methodPath.first ?? ""
        let path = methodPath.dropFirst().joined(separator: " ")
        let rest = parts[1].components(separatedBy: " (")
        let status = Int(rest.first ?? "")
        var ms: Double? = nil
        if rest.count > 1 {
            let msStr = rest[1]
                .replacingOccurrences(of: "ms)", with: "")
                .replacingOccurrences(of: ")", with: "")
            ms = Double(msStr)
        }
        return (method, path, status, ms)
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
    private static func formatTime(_ d: Date) -> String {
        timeFormatter.string(from: d)
    }
    private static func formatMs(_ ms: Double) -> String {
        if ms < 1 { return String(format: "%.2fms", ms) }
        if ms < 1000 { return String(format: "%.0fms", ms) }
        return String(format: "%.1fs", ms / 1000)
    }

    private func methodColor(_ method: String) -> Color {
        switch method {
        case "GET":    return Theme.Colors.accent
        case "POST":   return Theme.Colors.success
        case "PUT":    return Theme.Colors.warning
        case "DELETE": return Theme.Colors.danger
        default:       return Theme.Colors.textMid
        }
    }
    private func statusColor(_ status: Int) -> Color {
        switch status {
        case 200..<300: return Theme.Colors.success
        case 300..<400: return Theme.Colors.accent
        case 400..<500: return Theme.Colors.warning
        default:        return Theme.Colors.danger
        }
    }
}

// MARK: - Preference keys

private struct ReqLogContentHeight: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
private struct ReqLogViewport: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
private struct ReqLogScrollOffset: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}
