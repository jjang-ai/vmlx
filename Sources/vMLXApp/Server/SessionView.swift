import SwiftUI
import vMLXEngine
import vMLXTheme

/// Detail view for the currently-selected session. Mirrors
/// `panel/src/renderer/src/components/sessions/SessionView.tsx`:
///   - Top bar: PID, host:port, latency, JANG/model-type badges
///   - Action buttons gated per state (Start / Stop / Wake / Cancel / Reconnect)
///   - Loading progress bar with phase label under the top bar
///   - Four bottom tabs: Logs / Performance / Cache / Benchmark
///   - Auto-opens the Logs tab when the session transitions to `.error`
struct SessionView: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale
    let session: Session

    @State private var tab: Tab = .logs

    enum Tab: String, CaseIterable, Identifiable {
        case logs        = "Logs"
        case performance = "Performance"
        case cache       = "Cache"
        case benchmark   = "Benchmark"
        case directories = "Directories"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            topBar
            if case .loading = session.state {
                SessionLoadBar(progress: session.loadProgress)
            }
            // iter-137 §212: render the engine-state error MESSAGE.
            // Prior behavior: the status pill said "Error", the
            // action-button row showed "Reconnect" + "Stop", and
            // startSession's flashBanner fired for ~3s — then the
            // actual failure reason vanished. Users staring at a
            // session card in error state had no way to see what
            // broke without opening Logs (which may not have the
            // friendly LoadOptions-level message — only the deeper
            // loader trace). Surface the `.error(let msg)` payload
            // right under the top bar so the reason is visible as
            // long as the session stays in error state.
            if case .error(let msg) = session.state {
                errorBanner(message: msg)
            }
            SessionConfigForm(sessionId: session.id)
                .padding(Theme.Spacing.lg)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .fill(Theme.Colors.surface)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
            tabbedPanel
        }
        .onChange(of: session.state) { _, new in
            if case .error = new { tab = .logs }
        }
    }

    private func errorBanner(message: String) -> some View {
        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Theme.Colors.danger)
                .font(.system(size: 14))
            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.ServerUI.engineError.render(appLocale))
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.danger)
                Text(message)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textHigh)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .padding(Theme.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.danger.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(Theme.Colors.danger.opacity(0.3), lineWidth: 1)
                )
        )
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(alignment: .center, spacing: Theme.Spacing.md) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: Theme.Spacing.xs) {
                    Text(session.displayName)
                        .font(Theme.Typography.title)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .lineLimit(1)
                    if session.isJANG { Badge(text: "JANG", color: Theme.Colors.accentHi) }
                    if session.isMXTQ { Badge(text: "MXTQ", color: Theme.Colors.accentHi) }
                    Badge(text: session.family.uppercased(), color: Theme.Colors.textMid)
                }
                HStack(spacing: Theme.Spacing.md) {
                    // iter-132 §158: remote sessions store the endpoint
                    // URL in `session.host` and `0` in `session.port` —
                    // rendering "https://api.openai.com/v1:0" on the
                    // detail view looked broken. Show as "ENDPOINT" +
                    // URL for remote, "HOST" + host:port for local.
                    if session.isRemote {
                        metaItem("ENDPOINT", session.host)
                    } else {
                        metaItem("HOST", "\(session.host):\(session.port)")
                        if let pid = session.pid {
                            metaItem("PID", "\(pid)")
                        }
                    }
                    if let ms = session.latencyMs {
                        metaItem("LATENCY", String(format: "%.0f ms", ms))
                    }
                    if !session.isRemote, case .standby(let depth) = session.state {
                        metaItem("STANDBY", depth == .deep ? "Deep sleep" : "Light sleep")
                    }
                }
            }
            Spacer()
            actionButtons
        }
    }

    private func metaItem(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(label)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Text(value)
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textHigh)
        }
    }

    // MARK: - Action buttons

    private var actionButtons: some View {
        HStack(spacing: Theme.Spacing.sm) {
            // iter-132 §158: remote sessions have no local lifecycle —
            // the endpoint is either reachable or not, and we can't
            // Start/Stop/Wake a server we don't own. Surface a quiet
            // "Remote" label so the spot isn't empty; Chat-send still
            // dispatches through RemoteEngineClient.
            if session.isRemote {
                Text(L10n.ServerUI.remote.render(appLocale))
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.accent)
                    .padding(.horizontal, Theme.Spacing.md)
                    .padding(.vertical, Theme.Spacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .stroke(Theme.Colors.accent, lineWidth: 1)
                    )
            } else {
                switch session.state {
                case .stopped:
                    buttonTile("Start", color: Theme.Colors.accent) { Task { await startSession() } }
                case .loading:
                    Text(L10n.ServerUI.starting.render(appLocale))
                        .font(Theme.Typography.bodyHi)
                        .foregroundStyle(Theme.Colors.textMid)
                    buttonTile("Cancel", color: Theme.Colors.danger) { Task { await app.engine.stop() } }
                case .running:
                    buttonTile("Stop", color: Theme.Colors.danger) { Task { await app.engine.stop() } }
                case .standby:
                    buttonTile("Wake", color: Theme.Colors.accent) { Task { await app.engine.wakeFromStandby() } }
                    buttonTile("Stop", color: Theme.Colors.danger) { Task { await app.engine.stop() } }
                case .error:
                    buttonTile("Reconnect", color: Theme.Colors.accent) { Task { await startSession() } }
                    buttonTile("Stop", color: Theme.Colors.danger) { Task { await app.engine.stop() } }
                }
            }
        }
    }

    @ViewBuilder
    private func buttonTile(_ title: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(color)
                )
        }
        .buttonStyle(.plain)
    }

    private func startSession() async {
        let resolved = await app.engine.settings.resolved(sessionId: session.id)
        let opts = Engine.LoadOptions(modelPath: session.modelPath, from: resolved)
        do {
            for try await event in await app.engine.load(opts) {
                if case .failed(let msg) = event {
                    app.flashBanner("Engine load failed: \(msg)")
                }
            }
        } catch {
            app.flashBanner("Engine load failed: \(error)")
        }
    }

    // MARK: - Bottom tabbed panel

    private var tabbedPanel: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            HStack(spacing: Theme.Spacing.xs) {
                ForEach(Tab.allCases) { t in
                    Button { tab = t } label: {
                        Text(t.rawValue)
                            .font(Theme.Typography.bodyHi)
                            .foregroundStyle(tab == t ? Theme.Colors.textHigh : Theme.Colors.textMid)
                            .padding(.horizontal, Theme.Spacing.md)
                            .padding(.vertical, Theme.Spacing.sm)
                            .background(
                                RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                    .fill(tab == t ? Theme.Colors.surfaceHi : Color.clear)
                            )
                    }
                    .buttonStyle(.plain)
                }
                Spacer()
            }
            Group {
                switch tab {
                case .logs:        LogsPanel()
                case .performance: PerformancePanel()
                case .cache:       CachePanel()
                case .benchmark:   BenchmarkPanel(sessionId: session.id,
                                                  modelId: session.modelPath.lastPathComponent)
                case .directories: ModelDirectoriesPanel()
                }
            }
        }
    }
}

/// Mirror of `SessionView.tsx:599-615` — the progress bar that sits between
/// the top bar and the config form while a model is loading. Shows the
/// determinate fraction and phase label straight from `LoadProgress`.
struct SessionLoadBar: View {
    let progress: LoadProgress?
    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack {
                Text(progress?.phase.rawValue.capitalized ?? "Loading")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                Spacer()
                if let f = progress?.fraction {
                    Text("\(Int(f * 100))%")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                }
            }
            LoadingBar(fraction: progress?.fraction)
            if let label = progress?.label, !label.isEmpty {
                Text(label)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
        }
        .padding(Theme.Spacing.md)
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
