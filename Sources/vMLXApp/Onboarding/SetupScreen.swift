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
            Text("Welcome to vMLX")
                .font(Theme.Typography.title)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
            Text("Step \(step + 1) of 3")
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
            Text("Run state-of-the-art LLMs on Apple Silicon")
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
                Text("Scanning Hugging Face cache…")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        } else if entries.isEmpty {
            EmptyStateView(
                systemImage: "tray",
                title: "No models found",
                caption: "vMLX didn't find any models in your Hugging Face cache. We recommend downloading Qwen3-0.6B-8bit — a fast, capable starter model (~0.6GB).",
                cta: ("Download Qwen3-0.6B", downloadStarter)
            )
        } else {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                Text("Pick a model to use for chat")
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
            }
        }
    }

    private var doneStep: some View {
        VStack(spacing: Theme.Spacing.lg) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundStyle(Theme.Colors.success)
            Text("You're all set")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Text("Open the Server tab to start the engine, then head to Chat. Keys for remote clients live under the API tab.")
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 500)
        }
    }

    private var footer: some View {
        HStack {
            if step > 0 {
                Button("Back") { step -= 1 }
                    .buttonStyle(.plain)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            Spacer()
            if step < 2 {
                Button(step == 0 ? "Get Started" : "Next") { step += 1 }
                    .buttonStyle(.borderedProminent)
            } else {
                Button("Finish") {
                    // If the user picked a model in step 2, auto-load it
                    // into a fresh session and land them on Chat. This is
                    // what first-launch users actually want — otherwise
                    // they're dropped on an empty screen with no running
                    // engine and have to re-click the model picker.
                    if let picked = selected {
                        let path = picked.canonicalPath
                        app.mode = .chat
                        Task { await app.chatViewModelRef?.startModel(at: path) }
                    }
                    app.markFirstLaunchComplete()
                }
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
