import SwiftUI
import vMLXEngine
import vMLXTheme

/// Dedicated window that lists every download — active, queued, completed,
/// failed, cancelled. Per-row Pause / Resume / Cancel / Retry / Open-in-Finder.
struct DownloadsWindow: View {
    @Environment(AppState.self) private var state

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.Colors.border)
            if sortedJobs.isEmpty {
                emptyView
            } else {
                ScrollView {
                    LazyVStack(spacing: Theme.Spacing.sm) {
                        ForEach(sortedJobs) { job in
                            DownloadRow(job: job)
                        }
                    }
                    .padding(Theme.Spacing.lg)
                }
            }
            Divider().background(Theme.Colors.border)
            footer
        }
        .frame(minWidth: 640, minHeight: 420)
        .background(Theme.Colors.background)
    }

    private var sortedJobs: [DownloadManager.Job] {
        state.downloadJobs.sorted { a, b in
            func rank(_ s: DownloadManager.Status) -> Int {
                switch s {
                case .downloading: return 0
                case .queued: return 1
                case .paused: return 2
                case .completed: return 3
                case .failed: return 4
                case .cancelled: return 5
                }
            }
            let ra = rank(a.status), rb = rank(b.status)
            if ra != rb { return ra < rb }
            return a.startedAt > b.startedAt
        }
    }

    private var header: some View {
        HStack {
            Text("Downloads")
                .font(Theme.Typography.title)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.surface)
    }

    private var footer: some View {
        HStack {
            Spacer()
            Button("Clear completed") {
                Task { await state.downloadManager.clearCompleted() }
            }
            .buttonStyle(.plain)
            .font(Theme.Typography.bodyHi)
            .foregroundStyle(Theme.Colors.textMid)
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.surfaceHi)
            )
        }
        .padding(Theme.Spacing.md)
        .background(Theme.Colors.surface)
    }

    private var emptyView: some View {
        VStack(spacing: Theme.Spacing.md) {
            Spacer()
            Image(systemName: "tray")
                .font(.system(size: 40))
                .foregroundStyle(Theme.Colors.textLow)
            Text("No downloads yet")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textMid)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct DownloadRow: View {
    let job: DownloadManager.Job
    @Environment(AppState.self) private var state

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Spacing.md) {
            Image(systemName: iconName)
                .font(.system(size: 20))
                .foregroundStyle(iconColor)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                HStack {
                    Text(job.displayName)
                        .font(Theme.Typography.bodyHi)
                        .foregroundStyle(Theme.Colors.textHigh)
                    Spacer()
                    StatusPill(status: job.status)
                }
                Text(job.repo)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .lineLimit(1)
                    .truncationMode(.middle)

                ProgressLine(fraction: job.fraction)

                HStack(spacing: Theme.Spacing.md) {
                    Text("\(DownloadFormat.bytes(job.receivedBytes)) / \(DownloadFormat.bytes(job.totalBytes))")
                    Text("•")
                    Text(DownloadFormat.speed(job.bytesPerSecond))
                    Text("•")
                    Text("ETA \(DownloadFormat.eta(job.etaSeconds))")
                    Spacer()
                    actionButtons
                }
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)

                if let err = job.error, !err.isEmpty {
                    Text(err)
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.danger)
                        .lineLimit(2)
                }
            }
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

    @ViewBuilder
    private var actionButtons: some View {
        let id = job.id
        switch job.status {
        case .downloading:
            rowButton("pause.fill", "Pause") {
                Task { await state.downloadManager.pause(id) }
            }
            rowButton("xmark", "Cancel") {
                Task { await state.downloadManager.cancel(id) }
            }
        case .paused:
            rowButton("play.fill", "Resume") {
                Task { await state.downloadManager.resume(id) }
            }
            rowButton("xmark", "Cancel") {
                Task { await state.downloadManager.cancel(id) }
            }
        case .failed:
            rowButton("arrow.clockwise", "Retry") {
                Task { await state.downloadManager.resume(id) }
            }
        case .completed:
            if let path = job.localPath {
                rowButton("folder", "Open") {
                    NSWorkspace.shared.activateFileViewerSelecting([path])
                }
            }
        case .queued, .cancelled:
            EmptyView()
        }
    }

    private func rowButton(_ icon: String, _ label: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .frame(width: 14)
                .padding(Theme.Spacing.xs)
        }
        .buttonStyle(.plain)
        .foregroundStyle(Theme.Colors.textMid)
        .help(label)
    }

    private var iconName: String {
        switch job.status {
        case .downloading: return "arrow.down.circle"
        case .paused: return "pause.circle"
        case .queued: return "clock"
        case .completed: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        case .cancelled: return "xmark.circle"
        }
    }
    private var iconColor: SwiftUI.Color {
        switch job.status {
        case .downloading, .queued: return Theme.Colors.accent
        case .paused: return Theme.Colors.warning
        case .completed: return Theme.Colors.success
        case .failed: return Theme.Colors.danger
        case .cancelled: return Theme.Colors.textLow
        }
    }
}

private struct StatusPill: View {
    let status: DownloadManager.Status
    var body: some View {
        Text(status.rawValue.uppercased())
            .font(.system(size: 9, weight: .bold, design: .default))
            .foregroundStyle(Theme.Colors.textHigh)
            .padding(.horizontal, Theme.Spacing.sm)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(pillColor.opacity(0.25))
            )
            .overlay(
                Capsule().stroke(pillColor, lineWidth: 1)
            )
    }
    private var pillColor: SwiftUI.Color {
        switch status {
        case .downloading: return Theme.Colors.accent
        case .queued: return Theme.Colors.textMid
        case .paused: return Theme.Colors.warning
        case .completed: return Theme.Colors.success
        case .failed: return Theme.Colors.danger
        case .cancelled: return Theme.Colors.textLow
        }
    }
}

private struct ProgressLine: View {
    let fraction: Double
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Theme.Colors.surfaceHi)
                Capsule()
                    .fill(Theme.Colors.accent)
                    .frame(width: max(2, geo.size.width * fraction))
                    .animation(.easeOut(duration: 0.25), value: fraction)
            }
        }
        .frame(height: 4)
    }
}
