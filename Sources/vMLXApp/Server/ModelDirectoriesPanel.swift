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

    @State private var userDirs: [URL] = []
    @State private var isScanning = false
    @State private var lastScanResultCount: Int? = nil
    @State private var bannerMessage: String? = nil
    @State private var dirToConfirmRemove: URL? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            header
            defaultDirRow
            customDirsList
            footer
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
                Text("Model directories")
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)
                Text("Folders the library scans for downloaded models.")
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
                    Text("Rescan")
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
            .help("Force a fresh disk walk of every model directory")
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
                Text("CUSTOM DIRECTORIES")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Spacer()
                Text("\(userDirs.count) added")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
            if userDirs.isEmpty {
                Text("No custom directories. Add one to scan an external drive or a workspace folder full of safetensors.")
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
                    Text("Add directory…")
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
            .help("Pick a folder containing model directories to scan")
            Spacer()
            if let count = lastScanResultCount {
                Text("Last scan: \(count) model\(count == 1 ? "" : "s")")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        }
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
                    Text("HuggingFace cache (default — cannot be removed)")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                } else if isMissing {
                    Text("Missing on disk — folder may have been deleted or the drive unmounted")
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
            .help("Reveal in Finder")

            if !isDefault {
                Button {
                    dirToConfirmRemove = url
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(Theme.Colors.danger.opacity(0.85))
                }
                .buttonStyle(.plain)
                .help("Remove this directory from the scan list (does not delete files)")
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
            Button("Remove from scan list", role: .destructive) {
                Task { await removeDir(url) }
            }
            Button("Cancel", role: .cancel) { }
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

    private func pickAndAddDir() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Pick a model directory"
        panel.prompt = "Add directory"
        panel.message = "Choose a folder vMLX should scan for model directories. Each immediate subfolder is treated as one model."
        if panel.runModal() == .OK, let url = panel.url {
            Task { await addDir(url) }
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
