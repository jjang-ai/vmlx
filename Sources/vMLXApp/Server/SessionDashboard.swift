import SwiftUI
import vMLXEngine
import vMLXTheme

/// UI model for one server session shown in the dashboard. Mirrors the
/// Electron `Session` interface (`sessions/SessionCard.tsx:5-22`) with only
/// the fields the Swift Server screen currently renders. Additional fields
/// (remoteUrl, type=remote, config JSON) land when multi-engine arrives.
///
/// Phase 3 note: `state` is currently a mirror of the single global
/// `AppState.engineState` for the one-and-only active session. When
/// multi-process lands, each Session will have its own live state feed.
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
/// create a new session. Single-session is the current reality; the data
/// model is multi-session-ready so phase 4 can flip it on without a rewrite.
struct SessionDashboard: View {
    @Environment(AppState.self) private var app
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
                Text("Server")
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)
                Text("Pick a model and start serving on a local port")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            Spacer()
            Button {
                showCreatePopover = true
            } label: {
                HStack(spacing: Theme.Spacing.xs) {
                    Image(systemName: "arrow.down.doc.fill")
                    Text("Load model")
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
            .help("Pick a model from disk and load it into a new server session")
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
                Text("No model loaded")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.Colors.textHigh)
                Text("Pick a model from your local cache to start chatting,\nrun a server, or generate images.")
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
                    Text("Load a model")
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
            Text("Models are detected from `~/.cache/huggingface/hub/`.\nUse the Downloads window to pull new ones.")
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

    private func createSession(for url: URL) async {
        let id = UUID()
        var settings = SessionSettings(modelPath: url)
        settings.displayName = url.lastPathComponent
        // Instantiate a dedicated Engine for this session and select it so
        // `app.engine` resolves to the new one for subsequent calls.
        let sessionEngine = app.engine(for: id)
        app.selectedServerSessionId = id
        app.rebindEngineObserver()
        settings = await sessionEngine.applySessionSettings(id, settings)
        let session = Session(
            id: id,
            displayName: settings.displayName ?? url.lastPathComponent,
            modelPath: url,
            family: detectFamily(url),
            isJANG: detectJANG(url),
            isMXTQ: detectMXTQ(url),
            quantBits: nil,
            host: settings.host ?? "127.0.0.1",
            port: settings.port ?? 8000,
            pid: nil,
            latencyMs: nil,
            state: .stopped,
            loadProgress: nil
        )
        app.sessions.append(session)
        selection = id
        showCreatePopover = false
    }

    private func startSession(_ id: UUID) async {
        guard let idx = app.sessions.firstIndex(where: { $0.id == id }) else { return }
        let s = app.sessions[idx]
        let eng = app.engine(for: id)
        app.selectedServerSessionId = id
        app.rebindEngineObserver()

        // Remote sessions skip the local engine load + HTTP listener
        // entirely. The session is a thin proxy — Chat / Terminal will
        // dispatch directly to the configured remote endpoint via
        // RemoteEngineClient. We just mark the session "running" so the
        // UI lifecycle indicator flips green.
        let remoteSettings = await eng.settings.session(id)
        if let r = remoteSettings, r.isRemote {
            if let idx2 = app.sessions.firstIndex(where: { $0.id == id }) {
                app.sessions[idx2].host = r.remoteURL ?? ""
                app.sessions[idx2].port = 0
                app.sessions[idx2].pid = Int(ProcessInfo.processInfo.processIdentifier)
            }
            return
        }

        var opts = Engine.LoadOptions(modelPath: s.modelPath)
        let resolved = await eng.settings.resolved(sessionId: id)
        opts = Engine.LoadOptions(modelPath: s.modelPath, from: resolved)
        do {
            for try await event in await eng.load(opts) {
                if case .failed(let msg) = event {
                    app.flashBanner("Engine load failed: \(msg)")
                    return
                }
            }
        } catch {
            app.flashBanner("Engine load failed: \(error)")
            return
        }

        // Persist this as the "last loaded session" so the next launch
        // can auto-reload it. Only written on successful load.
        UserDefaults.standard.set(id.uuidString, forKey: "vmlx.lastLoadedSessionId")

        // After the model is loaded, start the per-session HTTP listener
        // so external clients (OpenAI/Ollama/Anthropic) can hit it. This
        // replaces the previous dead path where `Start Session` only
        // loaded the model and left the tray stuck on PID:nil.
        //
        // LAN toggle resolution: if the per-session `lan` flag (or the
        // global default) is true, bind to `0.0.0.0` so other machines
        // on the LAN can reach the gateway. Otherwise use the explicit
        // host field (defaulting to 127.0.0.1) so loopback stays the
        // safe default for single-machine use.
        let lanOn = resolved.settings.defaultLAN
        let bindHost: String
        if lanOn {
            bindHost = "0.0.0.0"
        } else {
            bindHost = resolved.settings.defaultHost.isEmpty
                ? "127.0.0.1"
                : resolved.settings.defaultHost
        }
        let bindPort = s.port

        let http = app.httpServer(for: id)
        do {
            try await http.start(
                host: bindHost,
                port: bindPort,
                apiKey: resolved.settings.apiKey,
                adminToken: resolved.settings.adminToken,
                logLevel: LogStore.Level(rawValue: resolved.settings.defaultLogLevel) ?? .info
            )
            if let idx2 = app.sessions.firstIndex(where: { $0.id == id }) {
                app.sessions[idx2].host = bindHost
                app.sessions[idx2].port = bindPort
                app.sessions[idx2].pid = Int(ProcessInfo.processInfo.processIdentifier)
            }
            // UI-9 gateway registration: now that the model is loaded and
            // the library scan is fresh, register every display name with
            // the gateway so its /v1/chat/completions handler can route
            // incoming requests by `model` field. The gateway listener
            // itself starts/stops based on GlobalSettings.gatewayEnabled.
            await app.gateway.registerEngine(eng)
            await app.ensureGatewayRunning()
        } catch {
            app.flashBanner("HTTP listener failed: \(error)")
        }
    }

    private func stopSession(_ id: UUID) async {
        let eng = app.engine(for: id)
        await app.gateway.unregisterEngine(eng)
        await app.httpServer(for: id).stop()
        await eng.stop()
        if let idx = app.sessions.firstIndex(where: { $0.id == id }) {
            app.sessions[idx].pid = nil
        }
    }

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

    private func detectFamily(_ url: URL) -> String {
        let n = url.lastPathComponent.lowercased()
        for f in ["qwen", "mistral", "gemma", "llama", "glm", "nemotron", "minimax", "deepseek"] {
            if n.contains(f) { return f }
        }
        return "model"
    }
    private func detectJANG(_ url: URL) -> Bool {
        let n = url.lastPathComponent.lowercased()
        return n.contains("jang") || n.contains("mlxq")
    }
    private func detectMXTQ(_ url: URL) -> Bool {
        url.lastPathComponent.lowercased().contains("mxtq")
    }
}

/// Tiny popover shown by the "New session" button. Uses the existing
/// `ModelPickerRow` so we reuse the model-library-backed picker.
private struct CreateSessionPopover: View {
    @Binding var isPresented: Bool
    @State private var pickedPath: URL? = nil
    let onCreate: (URL) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text("New session")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            ModelPickerRow(path: $pickedPath)
            HStack {
                Spacer()
                Button("Cancel") { isPresented = false }
                Button("Create") {
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
