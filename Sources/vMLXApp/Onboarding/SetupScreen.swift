import SwiftUI
import vMLXEngine
import vMLXTheme

/// First-launch wizard. Shown as a sheet over `RootView` when the
/// `vmlx.firstLaunchComplete` user default is missing. Three steps:
///   1. Welcome — intro + "Get Started"
///   2. Model picker — lists models already in HF cache, otherwise
///      recommends a download (which routes through DownloadManager)
///   3. Done — "Finish" button calls `AppState.markFirstLaunchComplete()`
struct SetupScreen: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale
    @ObservedObject private var hfAuth = HuggingFaceAuth.shared
    @State private var step: Int = 0
    @State private var entries: [ModelLibrary.ModelEntry] = []
    @State private var loading: Bool = true
    @State private var selected: ModelLibrary.ModelEntry? = nil

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(Theme.Spacing.xl)
            Divider()
            footer
        }
        .background(Theme.Colors.background)
        .task(id: step) {
            if step == 1 {
                loading = true
                entries = await app.engine.scanModels(force: false)
                loading = false
            }
        }
    }

    private var header: some View {
        HStack {
            Text(L10n.Setup.welcome.render(appLocale))
                .font(Theme.Typography.title)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
            Text(L10n.Setup.stepOfFormat.format(locale: appLocale, Int64(step + 1)))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
        .padding(Theme.Spacing.lg)
    }

    @ViewBuilder
    private var content: some View {
        switch step {
        case 0: welcomeStep
        case 1: modelStep
        default: doneStep
        }
    }

    private var welcomeStep: some View {
        VStack(spacing: Theme.Spacing.lg) {
            Image(systemName: "sparkles")
                .font(.system(size: 48))
                .foregroundStyle(Theme.Colors.accent)
            Text(L10n.Setup.runSOTA.render(appLocale))
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Text("vMLX serves chat, embeddings, images, and tool calls over OpenAI / Anthropic / Ollama APIs — 100% on-device.")
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 500)
        }
    }

    @ViewBuilder
    private var modelStep: some View {
        if loading {
            VStack(spacing: Theme.Spacing.md) {
                ProgressView()
                Text(L10n.Setup.scanningCache.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        } else if entries.isEmpty {
            VStack(spacing: Theme.Spacing.md) {
                EmptyStateView(
                    systemImage: "tray",
                    title: "No models found",
                    caption: "vMLX didn't find any models in your Hugging Face cache. We recommend downloading Qwen3-0.6B-8bit — a fast, capable starter model (~0.6GB).",
                    cta: ("Download Qwen3-0.6B", downloadStarter)
                )
                hfTokenBanner
            }
        } else {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                Text(L10n.Setup.pickModel.render(appLocale))
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textHigh)
                ScrollView {
                    VStack(spacing: Theme.Spacing.xs) {
                        ForEach(entries) { e in
                            Button {
                                selected = e
                                app.selectedModelPath = e.canonicalPath
                            } label: {
                                HStack {
                                    Image(systemName: selected?.id == e.id ? "checkmark.circle.fill" : "circle")
                                        .foregroundStyle(Theme.Colors.accent)
                                    VStack(alignment: .leading) {
                                        Text(e.displayName)
                                            .font(Theme.Typography.bodyHi)
                                            .foregroundStyle(Theme.Colors.textHigh)
                                        Text("\(e.family) · \(e.modality.rawValue)")
                                            .font(Theme.Typography.caption)
                                            .foregroundStyle(Theme.Colors.textLow)
                                    }
                                    Spacer()
                                }
                                .padding(Theme.Spacing.sm)
                                .background(
                                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                                        .fill(selected?.id == e.id ? Theme.Colors.surfaceHi : Theme.Colors.surface)
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .frame(maxHeight: 280)
                hfTokenBanner
            }
        }
    }

    /// S1 §307 — gentle reminder about HF token state. If the user has
    /// a token stored we show a tiny "✓ Hugging Face token stored"
    /// footnote; if they don't we explain why they'd want one, with a
    /// deep-link to the API tab's token card. Non-blocking — the
    /// starter model (Qwen3-0.6B-8bit) is a public mlx-community repo
    /// and doesn't need a token.
    @ViewBuilder
    private var hfTokenBanner: some View {
        if hfAuth.hasToken {
            HStack(spacing: 6) {
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(Theme.Colors.success)
                Text(L10n.Setup.tokenSaved.render(appLocale))
                    .foregroundStyle(Theme.Colors.textLow)
            }
            .font(Theme.Typography.caption)
        } else {
            HStack(alignment: .top, spacing: 6) {
                Image(systemName: "key.horizontal")
                    .foregroundStyle(Theme.Colors.warning)
                VStack(alignment: .leading, spacing: 2) {
                    Text(L10n.Setup.tokenMissing.render(appLocale))
                        .foregroundStyle(Theme.Colors.textMid)
                    Button {
                        app.mode = .api
                        NotificationCenter.default.post(
                            name: .vmlxOpenHuggingFaceTokenCard, object: nil)
                    } label: {
                        Text(L10n.Setup.addTokenCTA.render(appLocale))
                            .foregroundStyle(Theme.Colors.accent)
                    }
                    .buttonStyle(.plain)
                }
            }
            .font(Theme.Typography.caption)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, Theme.Spacing.sm)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.warning.opacity(0.08))
            )
        }
    }

    private var doneStep: some View {
        VStack(spacing: Theme.Spacing.lg) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundStyle(Theme.Colors.success)
            Text(L10n.Setup.allSet.render(appLocale))
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Text(L10n.Setup.allSetBody.render(appLocale))
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 500)
        }
    }

    private var footer: some View {
        HStack {
            if step > 0 {
                Button(L10n.Setup.back.render(appLocale)) { step -= 1 }
                    .buttonStyle(.plain)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            Spacer()
            if step < 2 {
                Button(L10n.Setup.next.render(appLocale)) { step += 1 }
                    .buttonStyle(.borderedProminent)
            } else {
                Button(L10n.Setup.finish.render(appLocale)) { app.markFirstLaunchComplete() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(Theme.Spacing.lg)
    }

    private func downloadStarter() {
        Task {
            _ = await app.downloadManager.enqueue(
                repo: "mlx-community/Qwen3-0.6B-8bit",
                displayName: "Qwen3-0.6B-8bit"
            )
        }
    }
}
