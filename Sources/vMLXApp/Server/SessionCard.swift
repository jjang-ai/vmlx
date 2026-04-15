import SwiftUI
import vMLXEngine
import vMLXTheme

/// Per-session card shown in the `SessionDashboard`. Mirrors
/// `panel/src/renderer/src/components/sessions/SessionCard.tsx`. Status pill
/// covers all 5 engine states (running / stopped / error / loading / standby)
/// using the same color logic as `EngineStatusFooter`. During loading the
/// card shows an elapsed counter, phase label, and determinate progress bar.
struct SessionCard: View {
    let session: Session
    let isSelected: Bool
    let onSelect: () -> Void
    let onStart: () -> Void
    let onStop: () -> Void
    let onWake: () -> Void
    let onReconnect: () -> Void
    let onDelete: () -> Void

    @State private var loadingElapsed: Int = 0
    @State private var tickerTask: Task<Void, Never>? = nil

    var body: some View {
        Button(action: onSelect) {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                headerRow
                if case .loading = session.state {
                    loadingBlock
                }
                infoRow
                actionsRow
            }
            .padding(Theme.Spacing.md)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.lg)
                    .fill(Theme.Colors.surface)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.lg)
                            .stroke(isSelected ? Theme.Colors.accent : Theme.Colors.border,
                                    lineWidth: isSelected ? 1.5 : 1)
                    )
            )
        }
        .buttonStyle(.plain)
        .contextMenu {
            if case .standby = session.state { Button("Wake", action: onWake) }
            if session.isRunning { Button("Stop", action: onStop) }
            if case .error = session.state { Button("Reconnect", action: onReconnect) }
            Button("Open logs", action: onSelect)
            Divider()
            Button("Delete", role: .destructive, action: onDelete)
        }
        .onAppear {
            if case .loading = session.state { startTicker() }
        }
        .onDisappear { stopTicker() }
        .onChange(of: session.state) { _, newState in
            if case .loading = newState { startTicker() } else { stopTicker() }
        }
    }

    // MARK: - Header

    private var headerRow: some View {
        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: Theme.Spacing.xs) {
                    Text(session.displayName)
                        .font(Theme.Typography.bodyHi)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if session.isJANG {
                        Badge(text: "JANG", color: Theme.Colors.accentHi)
                    }
                    if session.isMXTQ {
                        Badge(text: "MXTQ", color: Theme.Colors.accentHi)
                    }
                }
                Text(session.family)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            Spacer()
            statusPill
        }
    }

    // MARK: - Status pill

    private var statusPill: some View {
        HStack(spacing: Theme.Spacing.xs) {
            Circle()
                .fill(dotColor)
                .frame(width: 6, height: 6)
            Text(statusLabel)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
        }
        .padding(.horizontal, Theme.Spacing.sm)
        .padding(.vertical, Theme.Spacing.xs)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                .fill(Theme.Colors.surfaceHi)
        )
    }

    private var dotColor: Color {
        switch session.state {
        case .stopped: return Theme.Colors.textLow
        case .loading: return Theme.Colors.accent
        case .running: return Theme.Colors.success
        case .standby: return Theme.Colors.warning
        case .error:   return Theme.Colors.danger
        }
    }

    private var statusLabel: String {
        switch session.state {
        case .stopped:        return "Stopped"
        case .loading:        return "Loading \(formatElapsed(loadingElapsed))"
        case .running:        return "Running"
        case .standby(.soft): return "Light Sleep"
        case .standby(.deep): return "Deep Sleep"
        case .error:          return "Error"
        }
    }

    // MARK: - Loading block

    private var loadingBlock: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            LoadingBar(fraction: session.loadProgress?.fraction)
            if let label = session.loadProgress?.label, !label.isEmpty {
                Text(label)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
        }
    }

    // MARK: - Info row

    private var infoRow: some View {
        HStack(spacing: Theme.Spacing.md) {
            Text("\(session.host):\(session.port)")
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textMid)
            if let pid = session.pid {
                Text("PID \(pid)")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
            Spacer()
        }
    }

    // MARK: - Actions row

    private var actionsRow: some View {
        HStack(spacing: Theme.Spacing.sm) {
            switch session.state {
            case .stopped, .error:
                actionButton("Start", color: Theme.Colors.accent, action: onStart)
            case .standby:
                actionButton("Wake", color: Theme.Colors.accent, action: onWake)
                actionButton("Stop", color: Theme.Colors.danger, action: onStop)
            case .running:
                actionButton("Stop", color: Theme.Colors.danger, action: onStop)
            case .loading:
                actionButton("Cancel", color: Theme.Colors.danger, action: onStop)
            }
            Spacer()
        }
    }

    @ViewBuilder
    private func actionButton(_ label: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textHigh)
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.xs)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.sm)
                        .fill(color)
                )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Elapsed ticker

    private func startTicker() {
        stopTicker()
        loadingElapsed = 0
        tickerTask = Task { @MainActor in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if Task.isCancelled { return }
                loadingElapsed += 1
            }
        }
    }

    private func stopTicker() {
        tickerTask?.cancel()
        tickerTask = nil
    }

    private func formatElapsed(_ s: Int) -> String {
        if s < 60 { return "\(s)s" }
        return "\(s / 60)m \(s % 60)s"
    }
}

/// Small rounded color badge used for JANG/MXTQ markers on the card.
struct Badge: View {
    let text: String
    let color: Color
    var body: some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold))
            .foregroundStyle(Theme.Colors.textHigh)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(color.opacity(0.22))
            )
            .overlay(
                Capsule().stroke(color.opacity(0.5), lineWidth: 0.5)
            )
    }
}
