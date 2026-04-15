import SwiftUI
import vMLXEngine
import vMLXTheme

/// Persisted appearance preference. Read by the root views to pick a
/// `preferredColorScheme`; toggled from the Settings screen. `.auto`
/// means "follow system" — SwiftUI treats `nil` as system-managed.
enum AppearanceMode: String, CaseIterable, Identifiable {
    case auto = "auto"
    case light = "light"
    case dark = "dark"

    var id: String { rawValue }

    var colorScheme: ColorScheme? {
        switch self {
        case .auto:  return nil
        case .light: return .light
        case .dark:  return .dark
        }
    }

    var label: String {
        switch self {
        case .auto:  return "System"
        case .light: return "Light"
        case .dark:  return "Dark"
        }
    }

    /// SF Symbol used by the top-bar appearance cycle button
    /// (`ChatScreen.ChatTopBar`). Matches macOS's system appearance
    /// conventions: half-filled circle for Auto, sun for Light,
    /// moon for Dark.
    var iconName: String {
        switch self {
        case .auto:  return "circle.lefthalf.filled"
        case .light: return "sun.max"
        case .dark:  return "moon"
        }
    }
}

@main
struct vMLXApp: App {
    @State private var appState = AppState()
    /// Persisted via `@AppStorage` so the pick survives relaunches.
    /// Default is `.dark` to match the pre-existing hardcoded value
    /// and not surprise existing users on upgrade.
    @AppStorage("vmlx.appearance") private var appearanceRaw: String = AppearanceMode.dark.rawValue
    private var appearance: AppearanceMode {
        AppearanceMode(rawValue: appearanceRaw) ?? .dark
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(appState)
                .preferredColorScheme(appearance.colorScheme)
                .background(Theme.Colors.background)
                .frame(minWidth: 1100, minHeight: 720)
        }
        .windowStyle(.hiddenTitleBar)

        WindowGroup(id: "downloads") {
            DownloadsWindow()
                .environment(appState)
                .preferredColorScheme(appearance.colorScheme)
                .background(Theme.Colors.background)
        }
        .windowResizability(.contentSize)

        // Menu bar server control center. Uses `.window` style so the
        // popover can host sliders, steppers, text fields and disclosure
        // groups — the default `.menu` style is limited to buttons +
        // submenus. ADDITIVE scene — does not affect the main WindowGroup.
        // Icon tracks `appState.engineState` live.
        MenuBarExtra("vMLX", systemImage: TrayItem.icon(for: appState.engineState)) {
            TrayItem()
                .environment(appState)
        }
        .menuBarExtraStyle(.window)
    }
}

/// Global, cross-screen state. One per window.
@Observable
@MainActor
final class AppState {
    enum Mode: String, CaseIterable, Identifiable {
        case chat = "Chat"
        case server = "Server"
        case image = "Image"
        case terminal = "Terminal"
        case api = "API"
        var id: String { rawValue }
    }

    var mode: Mode = .chat

    /// Per-session engines. Each entry in `sessions` may own a live `Engine`
    /// actor keyed by session UUID. `defaultEngine` is the catch-all the UI
    /// falls back to when no session is selected (e.g. first launch, Chat
    /// mode without a Server session). Every other code path should prefer
    /// `activeEngine` or `engine(for:)`.
    private let defaultEngine = Engine()
    var engines: [UUID: Engine] = [:]

    init() {
        // Recover from any force-quit mid-stream BEFORE the UI reads
        // SQLite. Flips `is_streaming = 1` rows back to 0 and tags them
        // with ` [interrupted]`. Called exactly once on app launch,
        // ahead of every ChatViewModel.attach(app:).
        Database.shared.markAllStreamingAsInterrupted()
    }

    /// The engine currently driving UI events. Resolves to the engine of the
    /// selected server session if present, else the default fallback. Keeping
    /// this as a non-optional keeps the 40+ existing `app.engine.xxx` call
    /// sites compiling without invasive nil-handling.
    var engine: Engine {
        if let id = selectedServerSessionId, let e = engines[id] { return e }
        return defaultEngine
    }

    /// Strictly-typed accessor for call sites that want to distinguish
    /// "no active session" from "default fallback". Returns nil when the
    /// selected session has no engine AND no session is selected.
    var activeEngine: Engine? {
        if let id = selectedServerSessionId, let e = engines[id] { return e }
        return nil
    }

    /// Look up (or lazily create) the engine for a given session id. Newly
    /// created engines are automatically bound to `HuggingFaceAuth.shared`
    /// so gated-repo downloads triggered from that session pick up the
    /// user's stored HF token without any extra plumbing.
    func engine(for id: UUID) -> Engine {
        if let e = engines[id] { return e }
        let fresh = Engine()
        engines[id] = fresh
        Task { @MainActor in
            let dm = await fresh.downloadManager
            HuggingFaceAuth.shared.bind(dm)
        }
        return fresh
    }

    /// Tear down a per-session engine (called from SessionDashboard delete).
    func removeEngine(for id: UUID) {
        if let e = engines.removeValue(forKey: id) {
            Task { await e.stop() }
        }
        if let srv = httpServers.removeValue(forKey: id) {
            Task { await srv.stop() }
        }
    }

    // MARK: - Per-session HTTP servers

    /// Hummingbird `Server` supervisors keyed by session id. Populated on
    /// first `httpServer(for:)` call; started on `SessionDashboard.startSession`
    /// and stopped on `stopSession`/`deleteSession`. See `HTTPServerActor`.
    var httpServers: [UUID: HTTPServerActor] = [:]

    /// Look up (or lazily create) the HTTP server supervisor for a session.
    func httpServer(for id: UUID) -> HTTPServerActor {
        if let srv = httpServers[id] { return srv }
        let fresh = HTTPServerActor(engine: engine(for: id), sessionId: id)
        httpServers[id] = fresh
        return fresh
    }

    var engineState: EngineState = .stopped
    var loadProgress: LoadProgress? = nil
    var selectedModelPath: URL? = nil
    var activeSessionId: UUID? = nil

    /// Server-screen session list. Mirrors the Electron `sessions` IPC feed
    /// (`SessionsContext.tsx`). Each entry is a running *or* inactive model
    /// session with its per-session settings blob (port, model path, etc).
    /// Phase 3 wires this to SettingsStore + the live engine state of the
    /// single active session; multi-process multi-engine support lands later.
    var sessions: [Session] = []
    /// The session the ServerScreen is currently focused on. Initially the
    /// only active session (if any); SessionDashboard drives this.
    var selectedServerSessionId: UUID? = nil

    /// First-launch wizard shown when `vmlx.firstLaunchComplete` hasn't been
    /// set. `SetupScreen` toggles the flag when the user finishes.
    var showSetup: Bool = !UserDefaults.standard.bool(forKey: "vmlx.firstLaunchComplete")
    func markFirstLaunchComplete() {
        UserDefaults.standard.set(true, forKey: "vmlx.firstLaunchComplete")
        showSetup = false
    }

    /// Server lifecycle callbacks used by the rich `TrayItem` popover. Bound
    /// in `RootView.task` to dispatch the corresponding engine calls (load,
    /// stop, softSleep, deepSleep, wake). The defaults are no-ops so the
    /// popover can render before the RootView mounts without crashing. All
    /// six are idempotent on the engine side — calling `stop` while already
    /// stopped is a no-op, `wake` on `.running` is a no-op, etc.
    var onTrayStartServer: () -> Void = {}
    var onTrayStopServer: () -> Void = {}
    var onTrayRestartServer: () -> Void = {}
    var onTraySoftSleepServer: () -> Void = {}
    var onTrayDeepSleepServer: () -> Void = {}
    var onTrayWakeServer: () -> Void = {}

    // Download manager — shared across all screens. Per
    // `feedback_download_window.md`, the Downloads window auto-opens on the
    // first `.started` event so the user ALWAYS sees the download.
    let downloadManager = DownloadManager()
    var downloadJobs: [DownloadManager.Job] = []
    var hasAutoOpenedDownloadsWindow = false

    /// Count of downloads that have finished successfully this session.
    /// Increments on every `.completed` event; drives the "+1" badge that
    /// flashes on the ServerScreen model picker dropdown for 3 seconds
    /// after a download lands. Pure UI signal — nothing persists.
    var downloadedModelCount: Int = 0
    var recentDownloadBadgeExpiresAt: Date? = nil

    /// Cmd-K command bar visibility. Bound to RootView's `.sheet`. Toggle
    /// rather than set so the Cmd-K shortcut closes the palette if it's
    /// already open (per keyboard-shortcut constraint).
    var showCommandBar: Bool = false
    /// Weak handle the ChatScreen publishes so the command bar can drive
    /// session switching. Non-optional after ChatScreen mounts; the bar
    /// falls back to mode-only actions when nil.
    var chatViewModelRef: ChatViewModel? = nil

    // Simple in-process warning banner (non-blocking).
    var banner: String? = nil
    func flashBanner(_ msg: String) {
        banner = msg
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 3_500_000_000)
            if banner == msg { banner = nil }
        }
    }

    /// Convenience: true while server is up and ready to serve requests.
    var serverRunning: Bool {
        if case .running = engineState { return true }
        return false
    }

    /// Subscribe to engine state once at app launch — keeps `engineState` and
    /// `loadProgress` in sync with the actor. Started from `RootView.task`.
    /// Always tracks the *active* engine (the one returned by `engine`).
    /// When the active engine swaps (user picks a different server session),
    /// `rebindEngineObserver()` cancels the old Task and re-subscribes.
    private var engineObserver: Task<Void, Never>? = nil

    func observeEngine() async {
        rebindEngineObserver()
    }

    func rebindEngineObserver() {
        engineObserver?.cancel()
        let current = engine
        engineObserver = Task { @MainActor [weak self] in
            guard let self else { return }
            for await next in await current.subscribeState() {
                if Task.isCancelled { break }
                self.engineState = next
                if case .loading(let p) = next {
                    self.loadProgress = p
                } else {
                    self.loadProgress = nil
                }
            }
        }
    }

    /// Subscribe to the DownloadManager and keep `downloadJobs` in sync.
    /// On the first `.started` event, opens the Downloads window via the
    /// provided `openWindow` closure — nothing silent, ever.
    func observeDownloads(openWindow: @escaping (String) -> Void) async {
        for await event in await downloadManager.subscribe() {
            switch event {
            case .started(let job):
                if !hasAutoOpenedDownloadsWindow {
                    hasAutoOpenedDownloadsWindow = true
                    openWindow("downloads")
                }
                upsert(job)
            case .progress(let job):
                upsert(job)
            case .completed(let job):
                upsert(job)
                // Force a model library rescan so the newly-downloaded model
                // appears in the ServerScreen picker immediately — otherwise
                // the 5-minute freshness window hides it until manual refresh.
                // Mirrors Electron's post-download modelScanner kick.
                downloadedModelCount += 1
                recentDownloadBadgeExpiresAt = Date().addingTimeInterval(3.0)
                // Fire a system notification so the user sees the
                // completion even when the app is in the background.
                // Silent complete was the UI audit's #5 gap.
                DownloadNotifier.notifyCompleted(job)
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    await Self.handleDownloadCompletion(
                        job, library: self.engine.modelLibrary)
                    // Badge auto-expires after 3 seconds.
                    try? await Task.sleep(nanoseconds: 3_100_000_000)
                    if let exp = self.recentDownloadBadgeExpiresAt, exp <= Date() {
                        self.recentDownloadBadgeExpiresAt = nil
                    }
                }
            case .paused(let id), .resumed(let id), .cancelled(let id):
                if let job = await downloadManager.job(id) { upsert(job) }
            case .failed(let id, _):
                if let job = await downloadManager.job(id) { upsert(job) }
            }
        }
    }

    /// Pure helper — forces a `ModelLibrary` rescan in response to a
    /// completed download job. Extracted so tests can drive the integration
    /// without spinning up a full AppState/Engine graph. The job argument
    /// is currently unused by the rescan itself but kept in the signature
    /// so richer behaviors (e.g. targeted entry insertion) can land later
    /// without breaking callers.
    static func handleDownloadCompletion(
        _ job: DownloadManager.Job,
        library: vMLXEngine.ModelLibrary
    ) async {
        _ = job
        _ = await library.scan(force: true)
    }

    private func upsert(_ job: DownloadManager.Job) {
        if let idx = downloadJobs.firstIndex(where: { $0.id == job.id }) {
            downloadJobs[idx] = job
        } else {
            downloadJobs.append(job)
        }
    }
}

struct RootView: View {
    @Environment(AppState.self) private var state
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var s = state
        NavigationSplitView {
            Sidebar(mode: $s.mode)
        } detail: {
            VStack(spacing: 0) {
                UpdateAvailableBanner()
                DownloadStatusBar()
                ZStack(alignment: .top) {
                    Group {
                        switch state.mode {
                        case .chat:   ChatScreen()
                        case .server: ServerScreen()
                        case .image:  ImageScreen()
                        case .terminal: TerminalScreen()
                        case .api:    APIScreen()
                        }
                    }
                    if let msg = state.banner {
                        BannerView(message: msg)
                            .padding(.top, Theme.Spacing.md)
                            .transition(.opacity)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Theme.Colors.background)
            }
        }
        .background(Theme.Colors.background)
        .task { await state.observeEngine() }
        .task {
            await state.observeDownloads { id in
                openWindow(id: id)
            }
        }
        .task {
            // HuggingFace token lifecycle: load from Keychain on launch,
            // then bind the default engine's DownloadManager so every
            // subsequent download picks up the stored token. Per-session
            // engines bind lazily as they're created (see AppState.engine).
            HuggingFaceAuth.shared.loadFromKeychain()
            let defaultDM = await state.engine.downloadManager
            HuggingFaceAuth.shared.bind(defaultDM)
        }
        .task {
            // Bind the TrayItem lifecycle callbacks once RootView mounts.
            // Each callback dispatches the corresponding Engine actor call
            // on a background task so the popover button tap remains
            // responsive. `start` re-loads using the last-used LoadOptions
            // (via `lastLoadOptions`); `restart` stops then re-loads with
            // the same retained options.
            let st = state
            st.onTrayStartServer = {
                Task { @MainActor in
                    let eng = st.engine
                    if let opts = await eng.lastLoadOptions {
                        let stream = await eng.load(opts)
                        for try await _ in stream {}
                    } else if let path = st.selectedModelPath {
                        let opts = Engine.LoadOptions(modelPath: path)
                        let stream = await eng.load(opts)
                        for try await _ in stream {}
                    } else {
                        st.flashBanner("Pick a model in the Server tab first")
                        st.mode = .server
                    }
                }
            }
            st.onTrayStopServer = {
                Task { await st.engine.stop() }
            }
            st.onTrayRestartServer = {
                Task { @MainActor in
                    let eng = st.engine
                    let opts = await eng.lastLoadOptions
                    await eng.stop()
                    if let opts {
                        let stream = await eng.load(opts)
                        for try await _ in stream {}
                    }
                }
            }
            st.onTraySoftSleepServer = {
                Task { try? await st.engine.softSleep() }
            }
            st.onTrayDeepSleepServer = {
                Task { try? await st.engine.deepSleep() }
            }
            st.onTrayWakeServer = {
                Task { try? await st.engine.wake() }
            }
        }
        .sheet(isPresented: Binding(get: { state.showSetup },
                                    set: { state.showSetup = $0 })) {
            SetupScreen()
                .environment(state)
                .frame(minWidth: 640, minHeight: 420)
        }
        .sheet(isPresented: Binding(get: { state.showCommandBar },
                                    set: { state.showCommandBar = $0 })) {
            CommandBar(chatVM: state.chatViewModelRef)
                .environment(state)
        }
        .background(
            // Invisible buttons carrying global Cmd-N / Cmd-K / Cmd-Shift-T
            // shortcuts. Chat-specific shortcuts (Cmd-W, Up-arrow recall)
            // live on the ChatScreen so they only fire in-context.
            ZStack {
                Button("") {
                    state.showCommandBar.toggle()
                }
                .keyboardShortcut("k", modifiers: .command)
                .hidden()

                Button("") {
                    state.chatViewModelRef?.newSession()
                    state.mode = .chat
                }
                .keyboardShortcut("n", modifiers: .command)
                .hidden()

                Button("") {
                    state.chatViewModelRef?.reopenLastClosed()
                }
                .keyboardShortcut("t", modifiers: [.command, .shift])
                .hidden()
            }
            .frame(width: 0, height: 0)
        )
    }
}

private struct Sidebar: View {
    @Binding var mode: AppState.Mode
    @Environment(AppState.self) private var appState

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text("vMLX")
                .font(Theme.Typography.title)
                .foregroundStyle(Theme.Colors.textHigh)
                .padding(.horizontal, Theme.Spacing.lg)
                .padding(.top, Theme.Spacing.lg)
                .padding(.bottom, Theme.Spacing.md)

            ForEach(AppState.Mode.allCases) { m in
                Button {
                    mode = m
                } label: {
                    HStack(spacing: Theme.Spacing.sm) {
                        Image(systemName: icon(for: m))
                            .frame(width: 16)
                            .foregroundStyle(mode == m ? Theme.Colors.textHigh : Theme.Colors.textMid)
                        Text(m.rawValue)
                            .font(Theme.Typography.bodyHi)
                            .foregroundStyle(mode == m ? Theme.Colors.textHigh : Theme.Colors.textMid)
                        Spacer()
                    }
                    .padding(.horizontal, Theme.Spacing.md)
                    .padding(.vertical, Theme.Spacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .fill(mode == m ? Theme.Colors.surfaceHi : Color.clear)
                    )
                    .padding(.horizontal, Theme.Spacing.sm)
                }
                .buttonStyle(.plain)
            }

            Spacer()

            // Footer — live engine status. Bound to AppState.engineState which
            // is fed by Engine.subscribeState() in AppState.observeEngine().
            EngineStatusFooter(state: appState.engineState,
                               progress: appState.loadProgress)
                .padding(.horizontal, Theme.Spacing.lg)
                .padding(.bottom, Theme.Spacing.lg)
        }
        .frame(minWidth: 210)
        .background(Theme.Colors.surface)
    }

    private func icon(for m: AppState.Mode) -> String {
        switch m {
        case .chat:   return "bubble.left.and.bubble.right"
        case .server: return "server.rack"
        case .image:  return "photo.on.rectangle"
        case .terminal: return "terminal"
        case .api:    return "network"
        }
    }
}

/// Sidebar footer — renders the live engine state with a colored dot, label,
/// and (when loading) a thin determinate progress bar with phase label.
///
/// State → presentation map:
///   .stopped         → grey  "Stopped"
///   .loading(p)      → blue  "<phase>"  + progress bar (indeterminate if p.fraction == nil)
///   .running         → green "Running"
///   .standby(.soft)  → yellow "Light sleep"
///   .standby(.deep)  → yellow "Deep sleep"
///   .error(msg)      → red   "Error" (msg shown on hover)
struct EngineStatusFooter: View {
    let state: EngineState
    let progress: LoadProgress?

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack(spacing: Theme.Spacing.sm) {
                Circle()
                    .fill(dotColor)
                    .frame(width: 6, height: 6)
                Text(label)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                Spacer()
                if case .loading = state, let frac = progress?.fraction {
                    Text("\(Int(frac * 100))%")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                }
            }
            if case .loading = state {
                LoadingBar(fraction: progress?.fraction)
                if let phaseLabel = progress?.label, !phaseLabel.isEmpty {
                    Text(phaseLabel)
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
        }
        .help(hoverText)
    }

    private var dotColor: SwiftUI.Color {
        switch state {
        case .stopped:        return Theme.Colors.textLow
        case .loading:        return Theme.Colors.accent
        case .running:        return Theme.Colors.success
        case .standby:        return Theme.Colors.warning
        case .error:          return Theme.Colors.danger
        }
    }

    private var label: String {
        switch state {
        case .stopped:        return "Stopped"
        case .loading(let p): return Self.phaseLabel(p.phase)
        case .running:        return "Running"
        case .standby(.soft): return "Light sleep"
        case .standby(.deep): return "Deep sleep"
        case .error:          return "Error"
        }
    }

    /// Map the raw phase enum to a user-friendly label. The progress
    /// detail (`p.label`) carries the specific step; this is just the
    /// high-level bucket for the status pill.
    static func phaseLabel(_ phase: LoadProgress.Phase) -> String {
        switch phase {
        case .downloading: return "Downloading"
        case .reading:     return "Detecting"
        case .quantizing:  return "Quantizing"
        case .applying:    return "Loading"
        case .warmup:      return "Warming up"
        case .finalizing:  return "Finishing"
        }
    }

    private var hoverText: String {
        switch state {
        case .error(let msg): return msg
        case .loading(let p): return p.label
        default:              return label
        }
    }
}

/// 2-pixel-tall determinate-or-indeterminate progress bar. When `fraction`
/// is nil, renders an animated barber-pole sweep instead of a fixed fill.
struct LoadingBar: View {
    let fraction: Double?
    @State private var sweepX: CGFloat = -0.4

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Theme.Colors.surfaceHi)
                if let f = fraction {
                    Capsule()
                        .fill(Theme.Colors.accent)
                        .frame(width: max(2, geo.size.width * CGFloat(f.clamped01)))
                        .animation(.easeOut(duration: 0.25), value: f)
                } else {
                    Capsule()
                        .fill(Theme.Colors.accent)
                        .frame(width: geo.size.width * 0.35)
                        .offset(x: geo.size.width * sweepX)
                        .onAppear {
                            withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) {
                                sweepX = 1.05
                            }
                        }
                }
            }
            .clipShape(Capsule())
        }
        .frame(height: 3)
    }
}

private extension Double {
    var clamped01: Double { max(0, min(1, self)) }
}

struct BannerView: View {
    let message: String
    var body: some View {
        Text(message)
            .font(Theme.Typography.bodyHi)
            .foregroundStyle(Theme.Colors.textHigh)
            .padding(.horizontal, Theme.Spacing.lg)
            .padding(.vertical, Theme.Spacing.sm)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.surfaceHi)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .stroke(Theme.Colors.border, lineWidth: 1)
                    )
            )
    }
}

/// Shared rounded surface with subtle border — Linear card look.
struct Panel<Content: View>: View {
    let content: () -> Content
    init(@ViewBuilder _ content: @escaping () -> Content) { self.content = content }
    var body: some View {
        content()
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.lg)
                    .fill(Theme.Colors.surface)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.lg)
                            .stroke(Theme.Colors.border, lineWidth: 1)
                    )
            )
    }
}

/// "Endpoint not wired" banner for screens blocked on engine work.
struct NotWiredBanner: View {
    let note: String
    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(Theme.Colors.warning)
            Text(note)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
            Spacer()
        }
        .padding(Theme.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surfaceHi)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
    }
}
