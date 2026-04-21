import SwiftUI
import vMLXEngine
import vMLXTheme

/// §250 — HuggingFace Hub search panel. Lets users find MLX-compatible
/// models by free-text query + format filter, inspect gated/size/last-
/// modified metadata, and hand off to `DownloadManager.enqueue` with one
/// click. Replaces the "type org/repo by hand" onboarding.
///
/// Keeps state entirely local to the view — no bindings needed from
/// `AppState`. DownloadManager is read off the shared AppState so
/// clicking `Download` fires the same path as the existing Add-by-URL flow.
struct ModelSearchPanel: View {
    @Environment(AppState.self) private var state

    @State private var query: String = ""
    @State private var filter: FilterMode = .mlx
    @State private var results: [HuggingFaceSearchResult] = []
    @State private var isSearching: Bool = false
    @State private var lastError: String? = nil
    @State private var searchTask: Task<Void, Never>? = nil

    /// Repos currently being downloaded so the row action button flips
    /// from "Download" to "In library" immediately after click.
    @State private var inFlightRepos: Set<String> = []

    enum FilterMode: String, CaseIterable, Identifiable {
        case mlx = "mlx"
        case safetensors = "safetensors"
        case textGen = "text-generation"
        case imageGen = "text-to-image"
        case vlm = "image-text-to-text"
        var id: String { rawValue }
        var label: String {
            switch self {
            case .mlx: return "MLX"
            case .safetensors: return "Safetensors"
            case .textGen: return "Text-gen"
            case .imageGen: return "Image-gen"
            case .vlm: return "VLM"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            searchBar
            Divider().background(Theme.Colors.border)
            if let err = lastError {
                errorBanner(err)
            }
            content
        }
    }

    private var searchBar: some View {
        HStack(spacing: Theme.Spacing.sm) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(Theme.Colors.textMid)
                TextField("Search HuggingFace models — e.g. qwen, gemma, flux",
                          text: $query)
                    .textFieldStyle(.plain)
                    .font(Theme.Typography.body)
                    .onSubmit { trigger() }
                if !query.isEmpty {
                    Button(action: { query = ""; results = []; lastError = nil }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(Theme.Colors.textLow)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.surfaceHi)
            )

            Picker("Filter", selection: $filter) {
                ForEach(FilterMode.allCases) { m in
                    Text(m.label).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 280)

            Button(action: trigger) {
                if isSearching {
                    ProgressView().controlSize(.small)
                } else {
                    Text("Search")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isSearching || query.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(Theme.Spacing.md)
        .background(Theme.Colors.surface)
    }

    private func errorBanner(_ text: String) -> some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Theme.Colors.warning)
            Text(text)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
            Spacer()
            Button("Dismiss") { lastError = nil }
                .buttonStyle(.plain)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
        .padding(Theme.Spacing.sm)
        .background(Theme.Colors.surfaceHi)
    }

    @ViewBuilder
    private var content: some View {
        if results.isEmpty && !isSearching {
            emptyHint
        } else {
            ScrollView {
                LazyVStack(spacing: Theme.Spacing.sm) {
                    ForEach(results) { row in
                        ModelSearchRow(
                            row: row,
                            inFlight: inFlightRepos.contains(row.modelId),
                            onDownload: { download(row) }
                        )
                    }
                }
                .padding(Theme.Spacing.md)
            }
        }
    }

    private var emptyHint: some View {
        VStack(spacing: Theme.Spacing.sm) {
            Spacer(minLength: Theme.Spacing.xl)
            Image(systemName: "sparkle.magnifyingglass")
                .font(.system(size: 36, weight: .light))
                .foregroundStyle(Theme.Colors.textMid)
            Text("Find MLX-compatible models")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Text("Search the HuggingFace Hub. Popular filters: MLX, VLM, image-gen.")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private func trigger() {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        searchTask?.cancel()
        isSearching = true
        lastError = nil
        let filterTag = filter.rawValue
        searchTask = Task {
            let hf = HuggingFaceSearch(token: HuggingFaceAuth.shared.currentToken())
            do {
                let out = try await hf.search(
                    query: q,
                    filters: [filterTag],
                    sort: "downloads",
                    limit: 30
                )
                if Task.isCancelled { return }
                await MainActor.run {
                    self.results = out
                    self.isSearching = false
                }
            } catch {
                if Task.isCancelled { return }
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.isSearching = false
                    self.results = []
                }
            }
        }
    }

    private func download(_ row: HuggingFaceSearchResult) {
        Task { @MainActor in
            _ = await state.downloadManager.enqueue(
                repo: row.modelId,
                displayName: row.modelId
            )
            inFlightRepos.insert(row.modelId)
        }
    }
}

/// One search result row.
private struct ModelSearchRow: View {
    let row: HuggingFaceSearchResult
    let inFlight: Bool
    let onDownload: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Spacing.md) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(row.modelId)
                        .font(Theme.Typography.bodyHi)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if row.gated {
                        Text("Gated")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(Theme.Colors.warning)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .overlay(
                                Capsule().stroke(Theme.Colors.warning.opacity(0.5),
                                                 lineWidth: 0.5)
                            )
                    }
                }
                HStack(spacing: Theme.Spacing.md) {
                    Label(Self.humanCount(row.downloads), systemImage: "arrow.down")
                    Label("\(row.likes)", systemImage: "heart")
                    if let p = row.pipeline {
                        Text(p).font(Theme.Typography.caption)
                    }
                    if let d = row.lastModified {
                        Text(Self.relativeDate(d))
                            .font(Theme.Typography.caption)
                    }
                }
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)

                if !row.tags.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(row.tags.prefix(6), id: \.self) { tag in
                            Text(tag)
                                .font(.system(size: 9))
                                .foregroundStyle(Theme.Colors.textLow)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(
                                    Capsule().fill(Theme.Colors.surfaceHi)
                                )
                        }
                    }
                }
            }
            Spacer()
            Button(action: onDownload) {
                Label(inFlight ? "Queued" : "Download",
                      systemImage: inFlight ? "checkmark" : "arrow.down.circle")
            }
            .buttonStyle(.bordered)
            .disabled(inFlight)
        }
        .padding(Theme.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(Theme.Colors.border, lineWidth: 0.5)
                )
        )
    }

    private static func humanCount(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fk", Double(n) / 1_000) }
        return "\(n)"
    }

    private static func relativeDate(_ d: Date) -> String {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f.localizedString(for: d, relativeTo: Date())
    }
}
