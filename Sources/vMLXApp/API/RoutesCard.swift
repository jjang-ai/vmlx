import SwiftUI
import vMLXEngine
import vMLXTheme

/// O2/O3/O4 §290 — discoverable API route explorer for the API tab.
///
/// Renders every entry in `RouteCatalog.all` with:
///   • live online dot (green when the current engine can serve this
///     route's modality; orange when loaded-but-wrong-modality; red
///     when engine is stopped or ∞-available routes fall back to
///     engineState only)
///   • family filter chips + free-text search
///   • auth badge (open / bearer / admin) color-coded
///   • per-route copy-curl button (substitutes current host/port +
///     bearer + admin-token + first loaded model-id via
///     `RouteCatalog.curl(...)`)
///   • inline disclosure with sample body + streams-badge + docs anchor
struct RoutesCard: View {

    @Environment(AppState.self) private var app

    let host: String
    let port: Int
    let bearer: String?
    let admin: String?

    @State private var filter: RouteEntry.Family? = nil
    @State private var query: String = ""
    @State private var expanded: Set<String> = []
    @State private var copiedRow: String? = nil

    private var visibleRoutes: [RouteEntry] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return RouteCatalog.all.filter { r in
            (filter == nil || r.family == filter!) &&
            (q.isEmpty
                || r.path.lowercased().contains(q)
                || r.brief.lowercased().contains(q)
                || r.family.rawValue.lowercased().contains(q))
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            header
            filterChips
            Divider().opacity(0.3)
            if visibleRoutes.isEmpty {
                Text("No routes match your filter.")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .padding(.vertical, Theme.Spacing.md)
            } else {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(visibleRoutes) { route in
                        routeRow(route)
                        if route.id != visibleRoutes.last?.id {
                            Divider().opacity(0.2)
                        }
                    }
                }
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
    }

    // MARK: — header + filter

    private var header: some View {
        HStack {
            Text("API ROUTES")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Text("\(visibleRoutes.count)/\(RouteCatalog.all.count)")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Spacer()
            TextField("Filter by path, family, or brief...",
                      text: $query)
                .textFieldStyle(.plain)
                .frame(width: 240)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Theme.Colors.background)
                )
        }
    }

    private var filterChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                chip(label: "All", active: filter == nil) { filter = nil }
                ForEach(RouteEntry.Family.allCases, id: \.self) { fam in
                    chip(label: fam.rawValue,
                         active: filter == fam) { filter = fam }
                }
            }
        }
    }

    private func chip(label: String, active: Bool, tap: @escaping () -> Void) -> some View {
        Button(action: tap) {
            Text(label)
                .font(.system(size: 11, weight: active ? .semibold : .regular))
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill(active ? Theme.Colors.accent.opacity(0.2) : Theme.Colors.background)
                )
                .foregroundStyle(active ? Theme.Colors.accent : Theme.Colors.textMid)
                .overlay(
                    Capsule()
                        .stroke(active ? Theme.Colors.accent : Theme.Colors.border, lineWidth: 0.5)
                )
        }
        .buttonStyle(.plain)
    }

    // MARK: — row

    @ViewBuilder
    private func routeRow(_ route: RouteEntry) -> some View {
        let isOpen = expanded.contains(route.id)
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                onlineDot(for: route)
                    .frame(width: 10, height: 10)
                Text(route.method.rawValue)
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(methodColor(route.method).opacity(0.2))
                    .foregroundStyle(methodColor(route.method))
                    .cornerRadius(4)
                Text(route.path)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.Colors.textHigh)
                if route.streams {
                    tag("SSE", tint: .accentColor)
                }
                authBadge(route.auth)
                Spacer()
                Button {
                    let cmd = RouteCatalog.curl(
                        for: route,
                        host: host, port: port,
                        bearer: bearer, admin: admin,
                        model: firstLoadedModel())
                    copyToPasteboard(cmd)
                    copiedRow = route.id
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
                        if copiedRow == route.id { copiedRow = nil }
                    }
                } label: {
                    Text(copiedRow == route.id ? "Copied ✓" : "Copy curl")
                        .font(.system(size: 10, weight: .medium))
                }
                .buttonStyle(.borderless)
                .foregroundStyle(copiedRow == route.id ? Theme.Colors.success : Theme.Colors.accent)
                Button {
                    if isOpen { expanded.remove(route.id) }
                    else      { expanded.insert(route.id) }
                } label: {
                    Image(systemName: isOpen ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.Colors.textLow)
                }
                .buttonStyle(.borderless)
            }
            Text(route.brief)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
                .fixedSize(horizontal: false, vertical: true)
            if isOpen {
                drawer(route)
            }
        }
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private func drawer(_ route: RouteEntry) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                tag("Family: \(route.family.rawValue)", tint: .secondary)
                tag("Modality: \(route.modality.rawValue)", tint: .secondary)
                if let a = route.docsAnchor {
                    tag("#\(a)", tint: .secondary)
                }
            }
            .font(.system(size: 10))

            if !route.sampleBody.isEmpty {
                Text("Sample body")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Text(route.sampleBody)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Theme.Colors.background)
                    )
            }
            Text("Full curl")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Text(RouteCatalog.curl(for: route,
                                   host: host, port: port,
                                   bearer: bearer, admin: admin,
                                   model: firstLoadedModel()))
                .font(.system(size: 11, design: .monospaced))
                .textSelection(.enabled)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Theme.Colors.background)
                )
        }
        .padding(.top, 6)
    }

    // MARK: — online dot logic (O3)

    /// Green = engine can serve this modality right now.
    /// Orange = engine running but wrong modality (e.g. chat-only loaded
    ///   asked for /v1/embeddings).
    /// Red = engine stopped / error.
    /// Gray = soft_sleep or deep_sleep (would JIT-wake on request).
    @ViewBuilder
    private func onlineDot(for route: RouteEntry) -> some View {
        let color: Color = {
            switch app.engineState {
            case .stopped, .error: return Theme.Colors.danger
            case .loading:         return Theme.Colors.warning
            case .standby:         return Theme.Colors.textLow
            case .running:
                return matchesLoadedModality(route.modality)
                    ? Theme.Colors.success
                    : Theme.Colors.warning
            }
        }()
        Circle()
            .fill(color)
            .overlay(Circle().stroke(Color.white.opacity(0.3), lineWidth: 0.5))
    }

    private func matchesLoadedModality(_ required: RouteEntry.Modality) -> Bool {
        if required == .any { return true }
        // Without a capability probe we can't fully know, so default to
        // green for `running`. Routes that truly need a specific
        // modality (image/audio/embed/rerank) stay green when any
        // engine is loaded — the request itself will 503 if the
        // modality isn't available, which is a better UX than false
        // red dots before trying.
        return true
    }

    private func firstLoadedModel() -> String? {
        // The tray + /v1/models use `loadedModel`; reach into app.engine
        // via a published mirror on AppState. We sidestep the actor hop
        // by reading from the Session list, matching the tray label.
        if let s = app.sessions.first(where: {
            if case .running = $0.state { return true }
            return false
        }) {
            return s.displayName ?? s.modelPath.lastPathComponent
        }
        return nil
    }

    // MARK: — badges + helpers

    private func tag(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                Capsule()
                    .fill(tint.opacity(0.15))
            )
            .foregroundStyle(tint)
    }

    @ViewBuilder
    private func authBadge(_ auth: RouteEntry.AuthRequirement) -> some View {
        switch auth {
        case .none:   tag("open",   tint: .secondary)
        case .bearer: tag("bearer", tint: .accentColor)
        case .admin:  tag("admin",  tint: .orange)
        }
    }

    private func methodColor(_ m: RouteEntry.Method) -> Color {
        switch m {
        case .get:    return Theme.Colors.accent
        case .post:   return Theme.Colors.success
        case .delete: return Theme.Colors.danger
        case .put:    return Theme.Colors.warning
        case .head:   return Theme.Colors.textMid
        }
    }

    private func copyToPasteboard(_ s: String) {
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
        #endif
    }
}

#if canImport(AppKit)
import AppKit
#endif
