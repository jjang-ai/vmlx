import SwiftUI
import vMLXEngine
import vMLXTheme

/// Dedicated window that lists every download — active, queued, completed,
/// failed, cancelled. Per-row Pause / Resume / Cancel / Retry / Open-in-Finder.
struct DownloadsWindow: View {
    @Environment(AppState.self) private var state
    @Environment(\.appLocale) private var appLocale

    /// §250: tab between active-downloads list and Hub model search.
    /// Stored @State so each window instance keeps its own selection.
    @State private var tab: Tab = .downloads
    enum Tab: String, CaseIterable, Identifiable {
        case downloads = "Downloads"
        case search = "Search Hub"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            tabBar
            Divider().background(Theme.Colors.border)
            switch tab {
            case .downloads: downloadsTab
            case .search: ModelSearchPanel()
            }
        }
        .frame(minWidth: 720, minHeight: 480)
        .background(Theme.Colors.background)
    }

    private var tabBar: some View {
        HStack {
            Picker("", selection: $tab) {
                ForEach(Tab.allCases) { t in Text(t.rawValue).tag(t) }
            }
            .pickerStyle(.segmented)
            .frame(width: 280)
            Spacer()
        }
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.vertical, Theme.Spacing.sm)
        .background(Theme.Colors.surface)
    }

    @ViewBuilder
    private var downloadsTab: some View {
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
            Text(L10n.Downloads.title.render(appLocale))
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
            Button(L10n.Downloads.clearCompleted.render(appLocale)) {
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
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                HStack {
                    Spacer()
                    VStack(spacing: Theme.Spacing.sm) {
                        Image(systemName: "arrow.down.circle")
                            .font(.system(size: 44, weight: .light))
                            .foregroundStyle(Theme.Colors.accent)
                        Text(L10n.Downloads.empty.render(appLocale))
                            .font(Theme.Typography.title)
                            .foregroundStyle(Theme.Colors.textHigh)
                        Text(L10n.Downloads.emptyHint.render(appLocale))
                            .font(Theme.Typography.body)
                            .foregroundStyle(Theme.Colors.textMid)
                    }
                    Spacer()
                }
                .padding(.top, Theme.Spacing.xl)

                instructionCard
                hfTokenCallout
                pathsCard
            }
            .padding(Theme.Spacing.lg)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var instructionCard: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Label("How to start a download", systemImage: "1.circle.fill")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)

            Text(L10n.DownloadsUI.imageTabHint.render(appLocale))
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
                .fixedSize(horizontal: false, vertical: true)

            Text(L10n.DownloadsUI.fromCLI.render(appLocale))
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
                .padding(.top, Theme.Spacing.xs)

            Text("vmlx pull mlx-community/Qwen3-32B-4bit")
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textHigh)
                .padding(Theme.Spacing.sm)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.Colors.surfaceHi)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.sm))
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .stroke(Theme.Colors.border, lineWidth: 1)
        )
    }

    private var hfTokenCallout: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Label("Gated repos (Llama, Gemma, Mistral large)", systemImage: "lock.shield")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Text(L10n.DownloadsUI.gatedLicenseHint.render(appLocale))
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
            VStack(alignment: .leading, spacing: 4) {
                Text("1. Visit the model page on huggingface.co and click **Request access**.")
                Text("2. In the **API** tab below, paste a HuggingFace token in the *HuggingFace access token* card and click **Save & Test**.")
                Text("3. Retry the download — gated files will now succeed.")
            }
            .font(Theme.Typography.caption)
            .foregroundStyle(Theme.Colors.textMid)

            HStack {
                Link(destination: URL(string: "https://huggingface.co/settings/tokens")!) {
                    Label("Get a token", systemImage: "arrow.up.right.square")
                        .font(Theme.Typography.body)
                }
                .foregroundStyle(Theme.Colors.accent)
                Spacer()
            }
            .padding(.top, Theme.Spacing.xs)
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .stroke(Theme.Colors.border, lineWidth: 1)
        )
    }

    private var pathsCard: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Label("Where downloads land", systemImage: "folder")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Text(L10n.DownloadsUI.cacheHint.render(appLocale))
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
            Text("~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/main/")
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textMid)
                .padding(Theme.Spacing.sm)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.Colors.surfaceHi)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.sm))
            Text(L10n.DownloadsUI.pauseResumeHint.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .stroke(Theme.Colors.border, lineWidth: 1)
        )
    }
}

private struct DownloadRow: View {
    let job: DownloadManager.Job
    @Environment(AppState.self) private var state
    @Environment(\.appLocale) private var appLocale

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
                    Text(L10n.DownloadsUI.etaFormat.format(locale: appLocale, DownloadFormat.eta(job.etaSeconds) as NSString))
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
                // O7 §293 — targeted HF auth CTA. When the sibling
                // fetch returned 401/403, show a Fix button that
                // switches to the API tab + scrolls the user to the
                // HuggingFaceTokenCard. Much better UX than the
                // generic error hint which users miss.
                if job.requiresHFAuth {
                    HStack(spacing: 6) {
                        Image(systemName: "key.horizontal.fill")
                            .foregroundStyle(Theme.Colors.warning)
                        Text(L10n.DownloadsUI.gatedPrompt.render(appLocale))
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textMid)
                        Spacer()
                        Button(L10n.Downloads.openSettings.render(appLocale)) {
                            NotificationCenter.default.post(
                                name: .vmlxOpenHuggingFaceTokenCard,
                                object: nil)
                        }
                        .buttonStyle(.borderless)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Theme.Colors.accent)
                    }
                    .padding(.top, 4)
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
            rowButton("pause.fill", L10n.Common.pause.render(appLocale)) {
                Task { await state.downloadManager.pause(id) }
            }
            rowButton("xmark", L10n.Common.cancel.render(appLocale)) {
                Task { await state.downloadManager.cancel(id) }
            }
        case .paused:
            rowButton("play.fill", L10n.Common.resume.render(appLocale)) {
                Task { await state.downloadManager.resume(id) }
            }
            rowButton("xmark", L10n.Common.cancel.render(appLocale)) {
                Task { await state.downloadManager.cancel(id) }
            }
        case .failed:
            rowButton("arrow.clockwise", L10n.Common.retry.render(appLocale)) {
                Task { await state.downloadManager.resume(id) }
            }
        case .completed:
            if let path = job.localPath {
                rowButton("folder", L10n.Common.open.render(appLocale)) {
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
        // `help` drives the AppKit tooltip; `accessibilityLabel` drives
        // VoiceOver. Both share the same human-readable label so the
        // two assistive layers never diverge.
        .help(label)
        .accessibilityLabel(label)
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
