import SwiftUI
import vMLXEngine
import vMLXTheme
#if canImport(AppKit)
import AppKit
#endif

/// Model Directories — list + add + remove panel for the folders the
/// `ModelLibrary` walks during its disk scan. The default
/// `~/.cache/huggingface/hub/` is always present and not removable
/// (deleting it would orphan every locally-cached HuggingFace model);
/// every other entry is a user-added directory and can be removed.
///
/// User directories are persisted in the ModelLibrary's SQLite store so
/// they survive relaunches. Adding a directory triggers an immediate
/// `scan(force:true)` so the user sees their new models in the picker
/// without restarting the app.
struct ModelDirectoriesPanel: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale

    @State private var userDirs: [URL] = []
    @State private var isScanning = false
    @State private var lastScanResultCount: Int? = nil
    @State private var bannerMessage: String? = nil
    @State private var dirToConfirmRemove: URL? = nil

    /// Free-form repo input for starting arbitrary HuggingFace downloads
    /// without having to leave the Server tab. Format: `{org}/{repo}`,
    /// same as `vmlx pull`.
    @State private var pullRepo: String = ""
    @State private var isEnqueuing: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            header
            defaultDirRow
            customDirsList
            footer
            Divider().background(Theme.Colors.border)
            pullRepoRow
            if let msg = bannerMessage {
                bannerView(msg)
            }
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
        .task { await refreshDirs() }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(alignment: .top, spacing: Theme.Spacing.md) {
            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.ModelDirs.modelDirectories.render(appLocale))
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)
                Text(L10n.ModelDirs.folderScanHelp.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            Spacer()
            Button {
                Task { await rescan() }
            } label: {
                HStack(spacing: 4) {
                    if isScanning {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .controlSize(.mini)
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                    Text(L10n.ModelDirs.rescan.render(appLocale))
                        .font(Theme.Typography.body)
                }
                .foregroundStyle(Theme.Colors.textHigh)
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.xs)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surfaceHi)
                )
            }
            .buttonStyle(.plain)
            .disabled(isScanning)
            .help(L10n.Tooltip.rescanDirs.render(appLocale))
        }
    }

    /// Read-only row for the system default `~/.cache/huggingface/hub/`.
    /// Always present even when SwiftPM hasn't seen any models yet.
    private var defaultDirRow: some View {
        let defaultURL = URL(fileURLWithPath: NSString(
            string: "~/.cache/huggingface/hub").expandingTildeInPath)
        return dirRow(
            url: defaultURL,
            isDefault: true,
            isMissing: !FileManager.default.fileExists(atPath: defaultURL.path)
        )
    }

    private var customDirsList: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            HStack {
                Text(L10n.ModelDirs.customDirectories.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Spacer()
                Text("\(userDirs.count) added")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
            if userDirs.isEmpty {
                Text(L10n.ModelDirs.noCustomDirs.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                    .padding(.vertical, Theme.Spacing.sm)
            } else {
                ForEach(userDirs, id: \.self) { url in
                    dirRow(
                        url: url,
                        isDefault: false,
                        isMissing: !FileManager.default.fileExists(atPath: url.path)
                    )
                }
            }
        }
    }

    private var footer: some View {
        HStack {
            Button {
                pickAndAddDir()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "plus")
                    Text(L10n.ModelDirs.addDirectory.render(appLocale))
                }
                .foregroundStyle(.white)
                .font(Theme.Typography.bodyHi)
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.accent)
                )
            }
            .buttonStyle(.plain)
            .help(L10n.Tooltip.pickFolder.render(appLocale))
            Spacer()
            if let count = lastScanResultCount {
                Text(L10n.ModelDirs.lastScanFormat.format(locale: appLocale, Int64(count)))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        }
    }

    /// Free-form repo download. Lets the user pull any HuggingFace repo
    /// directly from the Server tab without bouncing over to Image. The
    /// Downloads window auto-opens on first progress event so there is
    /// no silent-download behavior. HF token (from Keychain) is injected
    /// automatically via the default DownloadManager binding.
    private var pullRepoRow: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text(L10n.ModelDirs.downloadByHF.render(appLocale))
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            HStack(spacing: Theme.Spacing.sm) {
                TextField("mlx-community/Qwen3-32B-4bit",
                          text: $pullRepo)
                    .textFieldStyle(.plain)
                    .font(Theme.Typography.mono)
                    .padding(.horizontal, Theme.Spacing.sm)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.sm)
                            .fill(Theme.Colors.surfaceHi)
                    )
                    .onSubmit { Task { await enqueuePull() } }

                Button {
                    Task { await enqueuePull() }
                } label: {
                    HStack(spacing: 4) {
                        if isEnqueuing {
                            ProgressView().controlSize(.mini)
                        } else {
                            Image(systemName: "arrow.down.circle.fill")
                        }
                        Text(isEnqueuing ? "Starting…" : "Download")
                    }
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(.white)
                    .padding(.horizontal, Theme.Spacing.md)
                    .padding(.vertical, Theme.Spacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .fill(Theme.Colors.accent)
                    )
                }
                .buttonStyle(.plain)
                .disabled(isEnqueuing
                          || pullRepo.trimmingCharacters(in: .whitespaces).isEmpty
                          || !pullRepo.contains("/"))
                .help(L10n.Tooltip.hfDownload.render(appLocale))
            }
            Text(L10n.ModelDirs.hfFormatHint.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
    }

    private func enqueuePull() async {
        let repo = pullRepo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repo.isEmpty, repo.contains("/") else {
            bannerMessage = "Enter a repo in {org}/{repo} format"
            return
        }
        isEnqueuing = true
        defer { isEnqueuing = false }
        let displayName = repo.split(separator: "/").last.map(String.init) ?? repo
        _ = await app.downloadManager.enqueue(
            repo: repo, displayName: displayName
        )
        bannerMessage = "Queued \(repo) — opening Downloads window…"
        pullRepo = ""
    }

    private func bannerView(_ msg: String) -> some View {
        Text(msg)
            .font(Theme.Typography.caption)
            .foregroundStyle(Theme.Colors.textHigh)
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.accent.opacity(0.18))
            )
    }

    // MARK: - Row

    private func dirRow(url: URL, isDefault: Bool, isMissing: Bool) -> some View {
        HStack(spacing: Theme.Spacing.md) {
            Image(systemName: isMissing ? "folder.badge.questionmark"
                                        : (isDefault ? "lock.fill" : "folder.fill"))
                .foregroundStyle(isMissing ? Theme.Colors.warning : Theme.Colors.textMid)
                .font(.system(size: 13))
            VStack(alignment: .leading, spacing: 1) {
                Text(url.path)
                    .font(Theme.Typography.mono)
                    .foregroundStyle(isMissing ? Theme.Colors.warning : Theme.Colors.textHigh)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if isDefault {
                    Text(L10n.ModelDirs.hfCacheDefault.render(appLocale))
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                } else if isMissing {
                    Text(L10n.ModelDirs.missingOnDisk.render(appLocale))
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.warning)
                }
            }
            Spacer()
            Button {
                #if canImport(AppKit)
                NSWorkspace.shared.activateFileViewerSelecting([url])
                #endif
            } label: {
                Image(systemName: "folder.badge.gear")
                    .foregroundStyle(Theme.Colors.textMid)
            }
            .buttonStyle(.plain)
            .help(L10n.Tooltip.revealInFinder.render(appLocale))

            if !isDefault {
                Button {
                    dirToConfirmRemove = url
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(Theme.Colors.danger.opacity(0.85))
                }
                .buttonStyle(.plain)
                .help(L10n.Tooltip.removeDir.render(appLocale))
            }
        }
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surfaceHi)
        )
        .confirmationDialog(
            "Remove this directory?",
            isPresented: Binding(
                get: { dirToConfirmRemove == url },
                set: { if !$0 { dirToConfirmRemove = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button(L10n.Misc.removeFromScanList.render(appLocale), role: .destructive) {
                Task { await removeDir(url) }
            }
            Button(L10n.Common.cancel.render(appLocale), role: .cancel) { }
        } message: {
            Text("vMLX will stop scanning this folder for models. The folder and any files inside it stay on disk untouched.")
        }
    }

    // MARK: - Actions

    private func refreshDirs() async {
        userDirs = await app.engine.modelLibrary.userDirs()
    }

    private func rescan() async {
        isScanning = true
        defer { isScanning = false }
        let results = await app.engine.modelLibrary.scan(force: true)
        lastScanResultCount = results.count
        bannerMessage = "Rescanned — found \(results.count) model\(results.count == 1 ? "" : "s")."
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            if bannerMessage?.contains("Rescanned") == true {
                bannerMessage = nil
            }
        }
    }

    @MainActor
    private func pickAndAddDir() {
        #if canImport(AppKit)
        // Iter 128 (vmlx#121 / #133): macOS 26 + ad-hoc-signed builds
        // lose the XPC link to the file picker. NSOpenPanelSafe falls
        // back to a manual-path text alert when XPC failure is
        // detected, so users can always type/paste a directory path.
        let result = NSOpenPanelSafe.pick(configure: { panel in
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            panel.title = "Pick a model directory"
            panel.prompt = "Add directory"
            panel.message = "Choose a folder vMLX should scan for model directories. Each immediate subfolder is treated as one model."
        }, fallbackTitle: L10n.PickerFallback.modelDirTitle.render(appLocale),
           fallbackMessage: L10n.PickerFallback.modelDirMessage.render(appLocale),
           canChooseFiles: false)
        if let url = result.url {
            if result.usedFallback {
                bannerMessage = L10n.PickerFallback.manualPathBanner.format(
                    locale: appLocale, url.lastPathComponent)
            }
            Task { await addDir(url) }
        } else if let reason = result.failureReason {
            bannerMessage = L10n.PickerFallback.manualPathFailure.format(
                locale: appLocale, reason)
        }
        #endif
    }

    private func addDir(_ url: URL) async {
        await app.engine.modelLibrary.addUserDir(url)
        await refreshDirs()
        await rescan()
        bannerMessage = "Added \(url.lastPathComponent). Rescanning…"
    }

    private func removeDir(_ url: URL) async {
        await app.engine.modelLibrary.removeUserDir(url)
        dirToConfirmRemove = nil
        await refreshDirs()
        await rescan()
        bannerMessage = "Removed \(url.lastPathComponent) from the scan list."
    }
}
