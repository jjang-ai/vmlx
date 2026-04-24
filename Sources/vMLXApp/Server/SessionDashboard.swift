import SwiftUI
import vMLXEngine
import vMLXTheme

/// UI model for one server session shown in the dashboard. Mirrors the
/// Electron `Session` interface (`sessions/SessionCard.tsx:5-22`) with only
/// the fields the Swift Server screen currently renders.
///
/// `state` + `loadProgress` are fed by `AppState.observePerSessionEngine(_:engine:)`
/// — a long-lived subscription to THIS session's own `Engine.subscribeState()`,
/// keyed by session id in `AppState.perSessionStateObservers`. Each Session
/// card reflects its own engine's lifecycle independently of which session
/// is currently selected in the UI.
struct Session: Identifiable, Equatable {
    let id: UUID
    var displayName: String
    var modelPath: URL
    var family: String
    var isJANG: Bool
    var isMXTQ: Bool
    var quantBits: Int?
    var host: String
    var port: Int
    var pid: Int?
    var latencyMs: Double?
    var state: EngineState
    var loadProgress: LoadProgress?
    /// §359 — snapshot of `SessionSettings.modelAlias` captured at
    /// session-start time. Displayed as a badge next to the model name
    /// so the user can confirm their alias is in effect. nil when no
    /// alias is set (the gateway then registers only the canonical
    /// display name). Alias is load-time-only — changing it mid-session
    /// updates SQLite but the Session.aliasSnapshot is refreshed only
    /// on the next Start/Restart.
    var aliasSnapshot: String? = nil
    /// iter-128 §154: session is bound to a remote OpenAI/Ollama/Anthropic
    /// endpoint via `SessionSettings.remoteURL` instead of serving a local
    /// model. Local engine stays `.stopped` for these — it's genuinely
    /// never loaded — but the session is functionally "ready" once the
    /// user has configured a valid URL + API key. Before §154, SessionCard
    /// rendered this as literal "Stopped / Start" which misled users into
    /// repeatedly clicking a no-op Start button.
    var isRemote: Bool = false

    /// True while the engine is actively serving requests.
    var isRunning: Bool {
        if case .running = state { return true }
        return false
    }
    /// True while the engine is in either standby depth.
    var isStandby: Bool {
        if case .standby = state { return true }
        return false
    }
    /// Running/loading/standby sessions live in the "Running" group; the
    /// "Inactive" group holds stopped + error sessions, matching
    /// `SessionDashboard.tsx:272-320`.
    var isActiveGroup: Bool {
        switch state {
        case .running, .loading, .standby: return true
        case .stopped, .error:              return false
        }
    }
}

/// Top-level Server screen dashboard — lists all sessions grouped by
/// Running / Inactive, with a "+" button that opens the model picker to
/// create a new session. Multi-session: each row in `app.sessions` has
/// its own `Engine` actor (`AppState.engines[UUID]`) and its own state
/// observer (`AppState.perSessionStateObservers[UUID]`), so background
/// engines keep their `state` up to date even when another session is
/// selected in the UI.
struct SessionDashboard: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale
    @Binding var selection: UUID?
    @State private var showCreatePopover = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            header

            if app.sessions.isEmpty {
                emptyState
            } else {
                if !runningGroup.isEmpty {
                    groupLabel("Running")
                    sessionGrid(runningGroup)
                }
                if !inactiveGroup.isEmpty {
                    groupLabel("Inactive")
                    sessionGrid(inactiveGroup)
                }
            }
        }
    }

    private var runningGroup: [Session]  { app.sessions.filter { $0.isActiveGroup } }
    private var inactiveGroup: [Session] { app.sessions.filter { !$0.isActiveGroup } }

    private var header: some View {
        HStack(alignment: .center, spacing: Theme.Spacing.md) {
            VStack(alignment: .leading, spacing: 2) {
                // §349 — localized title + subtitle.
                Text(L10n.Server.title.render(appLocale))
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)
                Text(L10n.Server.subtitle.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            Spacer()
            Button {
                showCreatePopover = true
            } label: {
                HStack(spacing: Theme.Spacing.xs) {
                    Image(systemName: "arrow.down.doc.fill")
                    // §349 — localized "Load model" button.
                    Text(L10n.Server.loadModel.render(appLocale))
                }
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.accent)
                )
            }
            .buttonStyle(.plain)
            .help(L10n.Tooltip.sessionDashboardLoad.render(appLocale))
            .popover(isPresented: $showCreatePopover, arrowEdge: .top) {
                CreateSessionPopover(isPresented: $showCreatePopover) { url in
                    Task { await createSession(for: url) }
                }
                .frame(width: 380)
            }
        }
    }

    private func groupLabel(_ text: String) -> some View {
        Text(text.uppercased())
            .font(Theme.Typography.caption)
            .foregroundStyle(Theme.Colors.textLow)
            .padding(.top, Theme.Spacing.sm)
    }

    private func sessionGrid(_ list: [Session]) -> some View {
        LazyVGrid(columns: [
            GridItem(.flexible(minimum: 280), spacing: Theme.Spacing.md),
            GridItem(.flexible(minimum: 280), spacing: Theme.Spacing.md),
        ], alignment: .leading, spacing: Theme.Spacing.md) {
            ForEach(list) { session in
                SessionCard(
                    session: session,
                    isSelected: selection == session.id,
                    onSelect: { selection = session.id },
                    onStart:   { Task { await startSession(session.id) } },
                    onStop:    { Task { await stopSession(session.id) } },
                    onWake:    { Task { await wakeSession(session.id) } },
                    onReconnect: { Task { await startSession(session.id) } },
                    onDelete:  { deleteSession(session.id) }
                )
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: Theme.Spacing.lg) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(Theme.Colors.textLow)
            VStack(spacing: Theme.Spacing.sm) {
                // §349 — localized empty-state title + hint.
                Text(L10n.Server.noModelLoaded.render(appLocale))
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.Colors.textHigh)
                Text(L10n.Server.noModelHint.render(appLocale))
                    .font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textMid)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Button {
                showCreatePopover = true
            } label: {
                HStack(spacing: Theme.Spacing.sm) {
                    Image(systemName: "arrow.down.doc.fill")
                        .font(.system(size: 13, weight: .semibold))
                    // §349 — localized CTA.
                    Text(L10n.Server.loadModel.render(appLocale))
                        .font(.system(size: 14, weight: .semibold))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, Theme.Spacing.xl)
                .padding(.vertical, Theme.Spacing.md)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.accent)
                )
            }
            .buttonStyle(.plain)
            .popover(isPresented: $showCreatePopover, arrowEdge: .top) {
                CreateSessionPopover(isPresented: $showCreatePopover) { url in
                    Task { await createSession(for: url) }
                }
                .frame(width: 380)
            }
            // §349 — localized models-source hint.
            Text(L10n.Server.modelsSourceHint.render(appLocale))
                .font(.system(size: 11))
                .foregroundStyle(Theme.Colors.textLow)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(Theme.Spacing.xxl)
        .padding(.vertical, Theme.Spacing.xxl)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
    }

    // MARK: - Session lifecycle actions
    //
    // Thin wrappers around `AppState.createSession` / `startSession` /
    // `stopSession`. The real logic lives on AppState so other surfaces
    // (Chat model picker's ▶/■ buttons, tray) share the exact same load
    // path. Keep the view-specific side effects (selection binding,
    // popover dismissal) local to the dashboard.

    private func createSession(for url: URL) async {
        let id = await app.createSession(forModel: url)
        selection = id
        showCreatePopover = false
    }

    private func startSession(_ id: UUID) async { await app.startSession(id) }
    private func stopSession(_ id: UUID)  async { await app.stopSession(id) }

    private func wakeSession(_ id: UUID) async {
        await app.engine(for: id).wakeFromStandby()
    }

    private func deleteSession(_ id: UUID) {
        app.sessions.removeAll { $0.id == id }
        if selection == id { selection = app.sessions.first?.id }
        app.removeEngine(for: id)
        if app.selectedServerSessionId == id {
            app.selectedServerSessionId = app.sessions.first?.id
            app.rebindEngineObserver()
        }
        Task { await app.engine.settings.deleteSession(id) }
    }

    // MARK: - Tiny display-only heuristics (JANG/MXTQ badges on the card)

    private func detectFamily(_ url: URL) -> String { SessionHeuristics.family(url) }
    private func detectJANG(_ url: URL) -> Bool { SessionHeuristics.isJANG(url) }
    private func detectMXTQ(_ url: URL) -> Bool { SessionHeuristics.isMXTQ(url) }
}

/// Display-only name-based heuristics for the Session card's badges and
/// family label. Extracted so `AppState.hydrateSessionsFromSettings()`
/// and `SessionDashboard.createSession(for:)` share the same rules —
/// otherwise a persisted session would show different badges than a
/// freshly-created one. Kept non-authoritative: the engine's real
/// capability detector (`CapabilityDetector`) is the source of truth
/// for loading; these are just card decorations.
enum SessionHeuristics {
    /// Collect all path segments, starting from the leaf, that look
    /// like human-readable model identifiers (i.e. NOT a commit SHA or
    /// cache-internal dir like `snapshots`/`blobs`). Used by both
    /// `family(...)` and `displayName(...)` so HuggingFace cache paths
    /// like `~/.cache/huggingface/hub/models--qwen--Qwen3-35B/snapshots/6068dbe.../`
    /// resolve to `Qwen3-35B` instead of the bare SHA.
    static func readableSegments(_ url: URL) -> [String] {
        // Walk leaf → root, skipping segments that are:
        //   - hex-SHA-like (40 chars all hex) — HF commit snapshots
        //   - known HF cache-internal names ("snapshots", "blobs", "refs")
        //   - the `hub` / `huggingface` sentinels
        //   - empty or "/"
        var out: [String] = []
        var cur = url.resolvingSymlinksInPath()
        let skipNames: Set<String> = [
            "snapshots", "blobs", "refs", "hub",
            "huggingface", "cache", ".cache", "MLXModels",
        ]
        while cur.path != "/" && !cur.pathComponents.isEmpty {
            let name = cur.lastPathComponent
            if name.isEmpty || name == "/" { break }
            let isHexSha = name.count >= 40
                && name.count <= 64
                && name.allSatisfy { $0.isHexDigit }
            let isCacheInternal = skipNames.contains(name)
            if !isHexSha && !isCacheInternal {
                // HuggingFace cache convention: `models--ORG--NAME`
                // with a DOUBLE-hyphen separator between the three
                // fields. The actual model name itself almost always
                // contains single hyphens (e.g. `Gemma-4-26B-A4B-it-
                // JANG_2L`), so the earlier `split(separator:"-")`
                // implementation happily shattered the whole string
                // and picked the last single-hyphen token — leaving
                // every HF-cached JANG session titled "JANG_2L" /
                // "JANG_4M" / "JANG_4S" instead of the full model
                // name. Split on the literal `--` separator so the
                // third field survives intact.
                if name.hasPrefix("models--") {
                    let parts = name.components(separatedBy: "--")
                    if let tail = parts.last, !tail.isEmpty {
                        out.append(tail)
                    } else {
                        out.append(name)
                    }
                } else {
                    out.append(name)
                }
            }
            cur = cur.deletingLastPathComponent()
        }
        return out
    }

    /// Preferred display name for a session. Falls back in this order:
    ///   1. leaf if it looks human-readable
    ///   2. nearest ancestor that looks human-readable
    ///   3. the original leaf (so the user at least sees *something*
    ///      even for pathological paths — better than an empty string)
    static func displayName(_ url: URL) -> String {
        let segs = readableSegments(url)
        if let first = segs.first, !first.isEmpty { return first }
        return url.lastPathComponent
    }

    /// Returns the stored name if it's readable, nil otherwise.
    /// Rejects 40+ hex-char SHAs that slipped through earlier because
    /// `createSession` used `url.lastPathComponent` which on HF cache
    /// paths is a commit SHA. Session rows saved before the 2026-04-16
    /// fix persist those bad names in SQLite — this sanitizer drops
    /// them so the session shows a fresh human-readable name instead.
    static func sanitizedStoredName(_ name: String?) -> String? {
        guard let n = name, !n.isEmpty else { return nil }
        // 40-64 hex chars = SHA-like, skip.
        if n.count >= 40 && n.count <= 64 && n.allSatisfy({ $0.isHexDigit }) {
            return nil
        }
        // Bare quant-suffix names (e.g. "JANG_2L", "JANG_4M", "JANGTQ2")
        // leaked into SQLite before the `--` splitter bug was fixed
        // (readableSegments was splitting `models--ORG--Name-...-JANG_2L`
        // on single hyphens and picking the trailing quant token). Reject
        // them so the hydrator re-derives the full model name.
        let upper = n.uppercased()
        if upper.hasPrefix("JANG_") || upper.hasPrefix("JANGTQ") {
            return nil
        }
        return n
    }

    static func family(_ url: URL) -> String {
        // Scan readable segments, not just the leaf. Otherwise a path like
        // `.../Qwen3-35B/snapshots/6068dbe…` would miss the `qwen` match
        // and return the generic `"model"` fallback.
        let candidates = readableSegments(url).map { $0.lowercased() }
        for n in candidates {
            for f in ["qwen", "mistral", "gemma", "llama", "glm",
                      "nemotron", "minimax", "deepseek", "phi",
                      "kimi", "granite", "jamba", "olmo", "lfm",
                      "baichuan", "mimo", "bailing"] {
                if n.contains(f) { return f }
            }
        }
        return "model"
    }
    static func isJANG(_ url: URL) -> Bool {
        let candidates = readableSegments(url).map { $0.lowercased() }
        return candidates.contains { $0.contains("jang") || $0.contains("mlxq") }
    }
    /// JANGTQ-format bundle. The engine flips the authoritative
    /// `isMXTQ` at load time via `jang_config.weight_format == "mxtq"`,
    /// but the card badge is drawn from the folder name BEFORE the
    /// engine has opened the config. Match both substrings so names
    /// like `Qwen3.5-35B-A3B-JANGTQ_2L` AND `Qwen3.5-A3B-MXTQ` badge
    /// consistently — the previous heuristic only found `"mxtq"` and
    /// silently dropped the badge on folders using the `"jangtq"`
    /// naming convention (which is what `jang_tools` actually writes).
    static func isMXTQ(_ url: URL) -> Bool {
        let candidates = readableSegments(url).map { $0.lowercased() }
        return candidates.contains { $0.contains("mxtq") || $0.contains("jangtq") }
    }
}

/// Tiny popover shown by the "New session" button. Uses the existing
/// `ModelPickerRow` so we reuse the model-library-backed picker.
private struct CreateSessionPopover: View {
    @Binding var isPresented: Bool
    @State private var pickedPath: URL? = nil
    @Environment(\.appLocale) private var appLocale: AppLocale
    let onCreate: (URL) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            // §349 — localized "New session" popover title.
            Text(L10n.Server.newSession.render(appLocale))
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            ModelPickerRow(path: $pickedPath)
            HStack {
                Spacer()
                Button(L10n.Common.cancel.render(appLocale)) { isPresented = false }
                Button(L10n.Misc.create.render(appLocale)) {
                    if let p = pickedPath {
                        onCreate(p)
                    }
                }
                .disabled(pickedPath == nil)
            }
        }
        .padding(Theme.Spacing.lg)
    }
}
