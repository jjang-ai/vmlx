import SwiftUI
import vMLXEngine
import vMLXTheme

/// Thin pinned bar that sits above the main detail view whenever ANY download
/// is active. Clicking the bar opens the Downloads window.
///
/// Per `feedback_download_window.md`: NEVER silent. This bar auto-reveals the
/// instant a `.started` event arrives (the AppState observer ensures the
/// `DownloadsWindow` also auto-opens on first start).
struct DownloadStatusBar: View {
    @Environment(AppState.self) private var state
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let active = state.downloadJobs.filter {
            $0.status == .downloading || $0.status == .queued || $0.status == .paused
        }
        if active.isEmpty {
            EmptyView()
        } else {
            let totalSpeed = active.reduce(0.0) { $0 + $1.bytesPerSecond }
            let primary = active.first!
            Button {
                openWindow(id: "downloads")
            } label: {
                HStack(spacing: Theme.Spacing.md) {
                    Image(systemName: "arrow.down.circle.fill")
                        .foregroundStyle(Theme.Colors.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: Theme.Spacing.sm) {
                            Text("\(active.count) download\(active.count == 1 ? "" : "s")")
                                .font(Theme.Typography.bodyHi)
                                .foregroundStyle(Theme.Colors.textHigh)
                            Text(primary.displayName)
                                .font(Theme.Typography.caption)
                                .foregroundStyle(Theme.Colors.textMid)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Text(DownloadFormat.speed(totalSpeed))
                                .font(Theme.Typography.caption)
                                .foregroundStyle(Theme.Colors.textMid)
                        }
                        MiniBar(fraction: primary.fraction)
                    }
                }
                .padding(.horizontal, Theme.Spacing.lg)
                .padding(.vertical, Theme.Spacing.sm)
                .background(Theme.Colors.surface)
                .overlay(
                    Rectangle()
                        .fill(Theme.Colors.border)
                        .frame(height: 1),
                    alignment: .bottom
                )
            }
            .buttonStyle(.plain)
        }
    }
}

private struct MiniBar: View {
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
        .frame(height: 2)
    }
}

/// Formatting helpers shared by the bar and the window.
enum DownloadFormat {
    static func bytes(_ v: Int64) -> String {
        let bcf = ByteCountFormatter()
        bcf.allowedUnits = [.useKB, .useMB, .useGB]
        bcf.countStyle = .file
        return bcf.string(fromByteCount: v)
    }
    static func speed(_ bps: Double) -> String {
        if bps <= 0 { return "—" }
        return bytes(Int64(bps)) + "/s"
    }
    static func eta(_ secs: Double?) -> String {
        guard let s = secs, s.isFinite, s > 0 else { return "—" }
        if s < 60 { return "\(Int(s))s" }
        if s < 3600 { return "\(Int(s / 60))m" }
        return String(format: "%.1fh", s / 3600)
    }
}

extension DownloadManager.Job {
    var fraction: Double {
        guard totalBytes > 0 else { return 0 }
        return min(1.0, Double(receivedBytes) / Double(totalBytes))
    }
}
