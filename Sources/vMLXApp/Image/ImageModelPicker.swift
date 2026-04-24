// SPDX-License-Identifier: Apache-2.0
//
// ImageModelPicker — replaces the hardcoded model list in the original
// scaffold. Binds to `appState.engine.modelLibrary.entries()` and surfaces
// two sections:
//
//   • Generate: Flux Schnell, Flux Dev, Z-Image Turbo
//   • Edit:     Qwen Image Edit
//
// Each row shows: display name, size, downloaded dot, JANG/MXTQ badge.
// Rows that aren't downloaded surface a "Download" button which kicks
// `DownloadManager.enqueue(...)` — per feedback_download_window.md, that
// auto-opens the Downloads window on the first `.started` event, so there
// is no silent download path.
//
// Theming: Theme.* tokens only, zero hardcoded colors.

import SwiftUI
import vMLXEngine
import vMLXTheme

struct ImageModelPicker: View {
    @Environment(AppState.self) private var appState
    @Binding var selected: ImageCatalogModel?
    @Binding var mode: ImageScreen.Tab

    @State private var entries: [ModelLibrary.ModelEntry] = []

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            section(title: "Generate", models: ImageCatalog.generate)
            section(title: "Edit",     models: ImageCatalog.edit)
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
        .task { await refresh() }
    }

    private func section(title: String, models: [ImageCatalogModel]) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text(title.uppercased())
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .padding(.horizontal, Theme.Spacing.sm)
            ForEach(models) { m in
                ImageModelRow(
                    model: m,
                    isDownloaded: entryFor(m) != nil,
                    isSelected: selected?.id == m.id,
                    size: entryFor(m)?.totalSizeBytes ?? m.approxSizeBytes,
                    badges: badgesFor(m),
                    onSelect: {
                        selected = m
                        mode = (m.kind == .edit) ? .edit : .generate
                    },
                    onDownload: {
                        Task {
                            await appState.downloadManager.enqueue(
                                repo: m.repo, displayName: m.displayName
                            )
                        }
                    }
                )
            }
        }
    }

    private func entryFor(_ m: ImageCatalogModel) -> ModelLibrary.ModelEntry? {
        // Loose match: display name or repo contains the model's search
        // fragment. Avoids regex and lives in the explicit catalog, per
        // feedback_no_regex_explicit_settings.md.
        entries.first { entry in
            let needle = m.libraryMatchFragment.lowercased()
            return entry.displayName.lowercased().contains(needle)
                || entry.canonicalPath.path.lowercased().contains(needle)
        }
    }

    private func badgesFor(_ m: ImageCatalogModel) -> [String] {
        var out: [String] = []
        // §381 — lead with runnability so users see it before JANG/quant
        // metadata. Scaffolded entries get a muted "Not ready" tag.
        if !m.ready { out.append("Not ready") }
        guard let e = entryFor(m) else { return out }
        if e.isJANG { out.append("JANG") }
        if e.isMXTQ { out.append("MXTQ") }
        if let b = e.quantBits { out.append("\(b)-bit") }
        return out
    }

    private func refresh() async {
        entries = await appState.engine.modelLibrary.entries()
            .filter { $0.modality == .image || $0.family.lowercased().contains("flux") }
    }
}

// MARK: - Row

private struct ImageModelRow: View {
    let model: ImageCatalogModel
    let isDownloaded: Bool
    let isSelected: Bool
    let size: Int64
    let badges: [String]
    let onSelect: () -> Void
    let onDownload: () -> Void

    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Circle()
                .fill(isDownloaded ? Theme.Colors.success : Theme.Colors.textLow)
                .frame(width: 6, height: 6)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: Theme.Spacing.xs) {
                    Text(model.displayName)
                        .font(Theme.Typography.bodyHi)
                        .foregroundStyle(Theme.Colors.textHigh)
                    ForEach(badges, id: \.self) { b in
                        Text(b)
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textMid)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Theme.Colors.surfaceHi)
                            )
                    }
                }
                Text(sizeLabel)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
            Spacer()
            if isDownloaded {
                Button(action: onSelect) {
                    Text(isSelected ? "Selected" : "Select")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(isSelected ? Theme.Colors.textHigh : Theme.Colors.textMid)
                        .padding(.horizontal, Theme.Spacing.sm)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                .fill(isSelected ? Theme.Colors.accent : Theme.Colors.surfaceHi)
                        )
                }
                .buttonStyle(.plain)
            } else {
                Button(action: onDownload) {
                    Label("Download", systemImage: "arrow.down.circle")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .padding(.horizontal, Theme.Spacing.sm)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                .fill(Theme.Colors.accent)
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { if isDownloaded { onSelect() } }
        .padding(.horizontal, Theme.Spacing.sm)
        .padding(.vertical, Theme.Spacing.xs)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                .fill(isSelected ? Theme.Colors.surfaceHi : Color.clear)
        )
    }

    private var sizeLabel: String {
        if size <= 0 { return "—" }
        let gb = Double(size) / 1_073_741_824.0
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(size) / 1_048_576.0
        return String(format: "%.0f MB", mb)
    }
}

// MARK: - Catalog (single source of truth for known image models)

struct ImageCatalogModel: Identifiable, Hashable {
    enum Kind { case generate, edit }
    let id: String
    let displayName: String
    let repo: String
    let kind: Kind
    /// Substring used to match this model against a `ModelLibrary.ModelEntry`.
    let libraryMatchFragment: String
    /// Fallback size shown when the model hasn't been downloaded yet.
    let approxSizeBytes: Int64
    /// §381 — `true` when the Swift-side DiT + encoder forward passes
    /// are actually ported and the model runs end-to-end. `false` for
    /// scaffolded entries (Flux1 Schnell/Dev, Flux2 Klein, Qwen-Image
    /// Edit, FIBO) which register into the picker but throw
    /// FluxError.notImplemented on generate. The picker renders a
    /// "Not yet runnable" badge + disables selection so users don't
    /// waste the 24 GB download on a stub.
    let ready: Bool
}

enum ImageCatalog {
    static let generate: [ImageCatalogModel] = [
        ImageCatalogModel(
            id: "flux-schnell",
            displayName: "Flux Schnell",
            repo: "black-forest-labs/FLUX.1-schnell",
            kind: .generate,
            libraryMatchFragment: "flux.1-schnell",
            approxSizeBytes: 23_800_000_000,
            ready: false
        ),
        ImageCatalogModel(
            id: "flux-dev",
            displayName: "Flux Dev",
            repo: "black-forest-labs/FLUX.1-dev",
            kind: .generate,
            libraryMatchFragment: "flux.1-dev",
            approxSizeBytes: 23_800_000_000,
            ready: false
        ),
        ImageCatalogModel(
            id: "z-image-turbo",
            displayName: "Z-Image Turbo",
            repo: "mlx-community/Z-Image-Turbo-mlx",
            kind: .generate,
            libraryMatchFragment: "z-image-turbo",
            approxSizeBytes: 6_800_000_000,
            ready: true
        ),
    ]
    static let edit: [ImageCatalogModel] = [
        ImageCatalogModel(
            id: "qwen-image-edit",
            displayName: "Qwen Image Edit",
            repo: "mlx-community/Qwen-Image-Edit",
            kind: .edit,
            libraryMatchFragment: "qwen-image-edit",
            approxSizeBytes: 54_000_000_000,
            ready: false
        ),
    ]
    static let all: [ImageCatalogModel] = generate + edit
}
