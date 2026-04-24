import MLX
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

/// AppKit lifecycle bridge so we can hook `applicationWillTerminate`.
///
/// SwiftUI's pure-declarative lifecycle doesn't expose a pre-quit hook;
/// without this adaptor, debounced SettingsStore writes (temperature /
/// model-path / session port) scheduled within ~500 ms of quit can lose
/// to the process exit. We call `engine.settings.flushPending()` and
/// `engine.stop()` synchronously (2 s timeout each) so SQLite commits
/// land on disk and the Metal runtime releases shaders.
final class VMLXAppDelegate: NSObject, NSApplicationDelegate {

    weak var appState: AppState?

    func applicationWillTerminate(_ notification: Notification) {
        // §246: install a no-op MLX error handler BEFORE kicking off the
        // async flush + stop. If a load or stream task is mid-eval when
        // the main runloop tears down, MLX's `ThreadPool::enqueue` can
        // fire `[Not allowed on stopped ThreadPool]` on its way out. The
        // default handler calls `fatalError` → SIGTRAP → users see a
        // "vMLX quit unexpectedly" dialog on clean quit. Swallowing to
        // stderr during termination is the right behavior: we're exiting
        // anyway, the load's partial state is irrelevant, and a crash
        // report at quit-time spooks users. Reproduced 2026-04-21 on
        // 2.0.0-beta.9 when osascript-quit was issued during warmup.
        MLX.setErrorHandler({ msg, _ in
            if let m = msg.map({ String(cString: $0) }) {
                FileHandle.standardError.write(Data("MLX-terminate: \(m)\n".utf8))
            }
        })

        guard let app = appState else { return }
        let flushSem = DispatchSemaphore(value: 0)
        Task.detached {
            await app.engine.settings.flushPending()
            await app.engine.stop()
            flushSem.signal()
        }
        _ = flushSem.wait(timeout: .now() + .seconds(2))
    }
}

@main
struct vMLXApp: App {
    @State private var appState = AppState()
    @NSApplicationDelegateAdaptor(VMLXAppDelegate.self) private var appDelegate
    /// Persisted via `@AppStorage` so the pick survives relaunches.
    /// Default is `.dark` to match the pre-existing hardcoded value
    /// and not surprise existing users on upgrade.
    @AppStorage("vmlx.appearance") private var appearanceRaw: String = AppearanceMode.dark.rawValue
    private var appearance: AppearanceMode {
        AppearanceMode(rawValue: appearanceRaw) ?? .dark
    }
    /// §349 — UI language. Stored in UserDefaults (NOT the engine
    /// SettingsStore) so the translation layer is a purely visual
    /// concern. `AppLocalePreference.seedIfAbsent()` runs in RootView
    /// on first appearance to match the system language once.
    @AppStorage(AppLocalePreference.userDefault)
    private var uiLanguageRaw: String = AppLocale.fromSystem().rawValue
    private var uiLocale: AppLocale {
        AppLocale(rawValue: uiLanguageRaw) ?? .en
    }
    /// Publish appState into the delegate on first access so it can
    /// flush settings during `applicationWillTerminate`. `init()` on
    /// App structs can't capture `@State` safely, but `body` is called
    /// after SwiftUI wires State — so doing it lazily inside body is
    /// the documented-safe place.
    private func linkDelegate() {
        if appDelegate.appState !== appState {
            appDelegate.appState = appState
        }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(appState)
                .environment(\.appLocale, uiLocale)
                .preferredColorScheme(appearance.colorScheme)
                .background(Theme.Colors.background)
                // §F4 — make every Text in the window copy-selectable.
                // Applies to error banners, notifications, settings
                // labels, help text, message bodies. Individual Text
                // views can still opt-out with `.textSelection(.disabled)`.
                .textSelection(.enabled)
                .onAppear {
                    linkDelegate()
                    AppLocalePreference.seedIfAbsent()
                }
                .frame(minWidth: 1100, minHeight: 720)
        }
        .windowStyle(.hiddenTitleBar)

        WindowGroup(id: "downloads") {
            DownloadsWindow()
                .environment(appState)
                .environment(\.appLocale, uiLocale)
                .preferredColorScheme(appearance.colorScheme)
                .background(Theme.Colors.background)
                .textSelection(.enabled)
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
                .environment(\.appLocale, uiLocale)
        }
        .menuBarExtraStyle(.window)

        // §349 — standard macOS Settings window. Reachable via Cmd-,
        // from any vMLX window, the menu bar (vMLX → Settings…), and
        // the in-app View menu entry below. Hosts the language picker
        // + appearance mode in a single General section for now; more
        // sections (Cache, Network, Shortcuts) slot in later without
        // changing the scene.
        Settings {
            SettingsScreen()
                .environment(appState)
                .environment(\.appLocale, uiLocale)
                .preferredColorScheme(appearance.colorScheme)
                .textSelection(.enabled)
        }

        // Scene-level `.commands { ... }` block registers File/View/Window
        // menu-bar entries so users can hit standard macOS shortcuts.
        // Without this the menu bar only shows the default SwiftUI menus
        // and keyboard shortcuts are reachable only from inside the app's
        // focused tab, not from the menu bar.
        .commands {
            // Replace the default "New" command (Cmd-N) with "New Chat"
            // wired to ChatViewModel.newSession so Cmd-N from any tab
            // lands on a fresh chat.
            CommandGroup(replacing: .newItem) {
                Button(L10n.Menu.newChat.render(uiLocale)) {
                    appState.mode = .chat
                    appState.chatViewModelRef?.newSession()
                }
                .keyboardShortcut("n", modifiers: [.command])

                Button(L10n.Menu.reopenLastClosed.render(uiLocale)) {
                    appState.mode = .chat
                    appState.chatViewModelRef?.reopenLastClosed()
                }
                .keyboardShortcut("t", modifiers: [.command, .shift])
            }

            // Edit menu — wire Cmd-Z to ChatViewModel's in-memory undo
            // stack for destructive actions (delete session, delete
            // message, clear-all). Replaces the default "Undo" group
            // so SwiftUI doesn't route Cmd-Z to the system text-edit
            // undo (which wouldn't know about our DB-backed actions).
            CommandGroup(replacing: .undoRedo) {
                Button(appState.chatViewModelRef?.topUndoLabel
                        .map { "\(L10n.Menu.undo.render(uiLocale)) \($0)" }
                       ?? L10n.Menu.undo.render(uiLocale)) {
                    if let label = appState.chatViewModelRef?.undo() {
                        appState.flashBanner("Undid: \(label)")  // L10N-EXEMPT: transient banner, TODO localize separately
                    }
                }
                .keyboardShortcut("z", modifiers: [.command])
                .disabled(appState.chatViewModelRef?.topUndoLabel == nil)
            }

            // View menu — tab switching. Cmd-1..Cmd-5 route to the five
            // top-level modes so keyboard users can navigate without
            // touching the sidebar.
            CommandMenu(L10n.Menu.view.render(uiLocale)) {
                Button(L10n.Mode.chat.render(uiLocale))     { appState.mode = .chat }
                    .keyboardShortcut("1", modifiers: [.command])
                Button(L10n.Mode.server.render(uiLocale))   { appState.mode = .server }
                    .keyboardShortcut("2", modifiers: [.command])
                Button(L10n.Mode.image.render(uiLocale))    { appState.mode = .image }
                    .keyboardShortcut("3", modifiers: [.command])
                Button(L10n.Mode.terminal.render(uiLocale)) { appState.mode = .terminal }
                    .keyboardShortcut("4", modifiers: [.command])
                Button(L10n.Mode.api.render(uiLocale))      { appState.mode = .api }  // L10N-EXEMPT: product term
                    .keyboardShortcut("5", modifiers: [.command])
                Divider()
                Button(L10n.Menu.commandPalette.render(uiLocale)) {
                    appState.showCommandBar.toggle()
                }
                .keyboardShortcut("k", modifiers: [.command])
            }

            // §349 — quick access to the Settings window so users can
            // flip UI language without memorizing Cmd-,. The standard
            // Settings shortcut still works; this is just a second
            // discovery path from the app menu. Button text reads
            // through the L10n catalog so it renders in the active
            // UI locale (en / ja / ko / zh-Hans).
            CommandGroup(after: .appSettings) {
                Button(L10n.Menu.languageAndSettings.render(uiLocale)) {
                    NSApp.sendAction(Selector(("showSettingsWindow:")),
                                     to: nil, from: nil)
                }
            }

            // Window → Downloads. Uses the named WindowGroup defined
            // above so macOS-level "bring window forward" logic works.
            CommandGroup(after: .windowArrangement) {
                Divider()
                Button(L10n.Menu.downloads.render(uiLocale)) { openDownloadsWindow() }
                    .keyboardShortcut("d", modifiers: [.command, .shift])
            }
        }
    }

    /// AppKit fallback to focus or open the Downloads window from the
    /// menu-bar command. SwiftUI's `@Environment(\.openWindow)` can't be
    /// captured at Scene scope, so we reach into NSApp — the named
    /// WindowGroup id "downloads" matches the scene declaration above.
    private func openDownloadsWindow() {
        #if canImport(AppKit)
        // If a Downloads window is already open, bring it forward.
        for win in NSApp.windows {
            if win.identifier?.rawValue.contains("downloads") == true {
                win.makeKeyAndOrderFront(nil)
                return
            }
        }
        // Otherwise, post the `openWindow` URL scheme SwiftUI synthesises
        // for named WindowGroups (see Apple's SwiftUI Scene documentation
        // — the URL form is honoured by NSApp.openURL).
        if let url = URL(string: "vmlx://downloads") {
            NSWorkspace.shared.open(url)
        }
        #endif
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

    /// §358 — persistent gateway status, so the tray + sidebar can show
    /// the actual bound port (or "offline" badge) without relying on the
    /// 3-second flashBanner. Populated every time `ensureGatewayRunning`
    /// runs. Off when gateway is disabled in GlobalSettings.
    enum GatewayStatus: Sendable, Equatable {
        case disabled
        case running(boundPort: Int, requestedPort: Int)
        case failed(port: Int, message: String)
    }
    var gatewayStatus: GatewayStatus = .disabled

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
        // Sweep temp-staged chat videos older than 14 days so
        // `/tmp/vmlx-chat-video-<UUID>.<ext>` orphans don't
        // accumulate forever when users attach videos but never
        // delete the containing chat. Live chats referencing swept
        // files degrade to "file not found" in the MessageBubble
        // chip — harmless. Cost: one FileManager scan on launch.
        Self.sweepStaleChatVideoStagings()

        // H1 §271 — thermal throttle surface. Prime the level on launch
        // and wire a NotificationCenter observer so the tray updates
        // whenever the OS posts a thermal-state change. Weak closure
        // capture is safe because AppState lives for the app's full
        // lifetime. The observer token is kept so deinit could remove
        // it; we let it leak with the app.
        self.thermalLevel = ThermalMonitor.currentLevel()
        NotificationCenter.default.addObserver(
            forName: ThermalMonitor.didChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.thermalLevel = ThermalMonitor.currentLevel()
            }
        }

        // H4 §273 — subscribe to the LogStore at `.error` so the tray
        // can render a red-dot badge with an unacknowledged count. The
        // `logs` property is actor-isolated on Engine, so we have to
        // `await self.defaultEngine.logs` first to pull it off the
        // actor, then call the nonisolated `subscribe(minLevel:)` from
        // the detached Task. The inner for-await hops back to MainActor
        // on every entry to bump the count safely.
        let engineRef = self.defaultEngine
        Task { [weak self] in
            let store = await engineRef.logs
            let stream = store.subscribe(minLevel: .error)
            for await _ in stream {
                await MainActor.run {
                    self?.errorLogCount += 1
                }
            }
        }
    }

    /// H4 §273 — user acknowledged the tray error-count badge.
    /// Called when the user opens the LogsPanel OR hovers the tray dot.
    /// Resets the count so the red dot disappears until a NEW error is
    /// appended post-acknowledgement.
    @MainActor
    func acknowledgeErrors() {
        self.errorLogCount = 0
        self.lastErrorLogAcknowledged = Date()
    }

    /// Delete temp-staged chat video files older than 14 days.
    /// Called from AppState.init; failures logged and ignored.
    private static func sweepStaleChatVideoStagings() {
        let tmp = FileManager.default.temporaryDirectory
        let fm = FileManager.default
        let cutoff = Date().addingTimeInterval(-14 * 24 * 3600)
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .contentModificationDateKey]
        guard let entries = try? fm.contentsOfDirectory(
            at: tmp, includingPropertiesForKeys: Array(keys), options: .skipsHiddenFiles)
        else { return }
        for url in entries
        where url.lastPathComponent.hasPrefix("vmlx-chat-video-") {
            let values = try? url.resourceValues(forKeys: keys)
            guard values?.isRegularFile == true,
                  let mtime = values?.contentModificationDate,
                  mtime < cutoff
            else { continue }
            try? fm.removeItem(at: url)
        }
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
    ///
    /// Phase 4 multi-engine: every newly-created engine also gets a
    /// long-lived state observer keyed by session id, so each `Session`
    /// row in `sessions` reflects ITS OWN engine's load / running /
    /// standby / error state — not a mirror of whichever engine is
    /// currently selected. Without this, a background engine's state
    /// change (e.g. idle auto-sleep) never reaches its card in
    /// SessionDashboard because the global `engineObserver` only
    /// subscribes to `self.engine` (the selected one).
    func engine(for id: UUID) -> Engine {
        if let e = engines[id] { return e }
        let fresh = Engine()
        engines[id] = fresh
        Task { @MainActor [weak self] in
            let dm = await fresh.downloadManager
            HuggingFaceAuth.shared.bind(dm)
            // Fan-in: forward this engine's download events into the
            // shared AppState.downloadJobs list so HTTP-initiated pulls
            // (Ollama /api/pull, OpenAI image-side downloads, etc.) show
            // in the Downloads window — the audit 2026-04-15 found that
            // pulls via HTTP went to engine.downloadManager and never
            // surfaced in the UI list. Per-engine subscription is fine
            // because Swift's AsyncStream subscribers are independent.
            guard let self else { return }
            await self.forwardDownloadEvents(from: dm)
        }
        observePerSessionEngine(id, engine: fresh)
        return fresh
    }

    /// Per-session engine state observers. Keyed by session id, NOT by
    /// the currently-selected session, so background sessions still
    /// have their `Session.state` kept in sync with their engine's
    /// lifecycle. Replaces the Phase-3 single-mirror model where only
    /// the selected engine's state was observed.
    private var perSessionStateObservers: [UUID: Task<Void, Never>] = [:]

    /// Start a long-lived subscription to `engine.subscribeState()` and
    /// mirror each state event into the matching `sessions[idx].state`
    /// and `.loadProgress`. Cancels any prior observer for this id so
    /// calling `engine(for:)` on an existing id is a no-op.
    func observePerSessionEngine(_ id: UUID, engine: Engine) {
        perSessionStateObservers[id]?.cancel()
        perSessionStateObservers[id] = Task { @MainActor [weak self] in
            guard let self else { return }
            for await next in await engine.subscribeState() {
                if Task.isCancelled { break }
                guard let idx = self.sessions.firstIndex(where: { $0.id == id })
                else { continue }
                self.sessions[idx].state = next
                if case .loading(let p) = next {
                    self.sessions[idx].loadProgress = p
                } else {
                    self.sessions[idx].loadProgress = nil
                }
                // App Nap prevention: hold a userInitiated activity token
                // while ANY session is actively loading or running, release
                // when all sessions are idle. macOS can otherwise suspend
                // the process during long generations (> several minutes)
                // which manifests as decode cratering or hangs after a
                // background window switch. See audit 2026-04-16.
                self.refreshAppNapActivity()
            }
        }
    }

    // MARK: - App Nap prevention

    /// `NSProcessInfo.beginActivity` token held while any engine is
    /// non-idle. Nil when every session is `.stopped` / `.error`. macOS
    /// releases the token on process exit, but we explicitly `endActivity`
    /// when we can so the system is free to nap between sessions.
    private var appNapActivityToken: NSObjectProtocol?

    /// Reconcile the activity token against current session states.
    /// Called from the per-session state observer and also after session
    /// removal to make sure we release the token when the last active
    /// engine goes idle.
    private func refreshAppNapActivity() {
        let anyActive = sessions.contains { s in
            switch s.state {
            case .running, .loading, .standby: return true
            case .stopped, .error:              return false
            }
        }
        if anyActive, appNapActivityToken == nil {
            appNapActivityToken = ProcessInfo.processInfo.beginActivity(
                options: [.userInitiated, .idleSystemSleepDisabled],
                reason: "vMLX model is loading or generating"
            )
        } else if !anyActive, let token = appNapActivityToken {
            ProcessInfo.processInfo.endActivity(token)
            appNapActivityToken = nil
        }
    }

    /// Cancel the per-session state observer for a removed session.
    /// Called from session-delete paths in `SessionDashboard` so we
    /// don't leak observer tasks after the Engine actor is dropped.
    func stopObservingPerSessionEngine(_ id: UUID) {
        perSessionStateObservers[id]?.cancel()
        perSessionStateObservers.removeValue(forKey: id)
    }

    /// Pipes events from a per-engine DownloadManager into the shared
    /// AppState job list. Subscribes once per engine on creation; unbinds
    /// implicitly when the engine is removed (the AsyncStream completes
    /// on actor deinit).
    func forwardDownloadEvents(from dm: DownloadManager) async {
        for await event in await dm.subscribe() {
            switch event {
            case .started(let job):
                if !hasAutoOpenedDownloadsWindow,
                   let openWindowClosure = appOpenWindow
                {
                    hasAutoOpenedDownloadsWindow = true
                    openWindowClosure("downloads")
                }
                upsert(job)
            case .progress(let job): upsert(job)
            case .completed(let job):
                upsert(job)
                // Per-engine fan-in covers HTTP-initiated pulls (Ollama
                // /api/pull, image endpoints, etc). Match the AppState path:
                // fire a system notification on completion so backgrounded
                // clients surface the success.
                DownloadNotifier.notifyCompleted(job)
            case .paused(let id), .resumed(let id), .cancelled(let id):
                if let job = await dm.job(id) { upsert(job) }
            case .failed(let id, let message):
                if let job = await dm.job(id) {
                    upsert(job)
                    // Same symmetry for failures — no silent background drop.
                    DownloadNotifier.notifyFailed(job, message: message)
                }
            }
        }
    }

    /// Captured at observeDownloads() time so per-engine fan-in can also
    /// auto-open the window on first `.started`. Nil before main app runs.
    private var appOpenWindow: ((String) -> Void)? = nil

    /// Public hook for the `vmlx://downloads` URL handler + anyone else
    /// that wants to surface the Downloads window without SwiftUI's
    /// `@Environment(\.openWindow)` in scope. Safely no-ops when
    /// RootView hasn't mounted yet (opener not captured).
    func openDownloadsWindowIfReady() {
        appOpenWindow?("downloads")
    }

    /// Ensure the gateway listener is running per current GlobalSettings.
    /// Safe to call from any session-start path — if the user has the
    /// toggle off, this is a no-op. If on, binds to (gatewayHost,
    /// gatewayPort) using the apiKey + adminToken from global settings.
    /// Idempotent: calling while already running does nothing.
    func ensureGatewayRunning() async {
        let global = await engine.settings.global()
        guard global.gatewayEnabled else {
            gatewayStatus = .disabled
            return
        }
        let host = global.gatewayLAN ? "0.0.0.0" : "127.0.0.1"
        let level = LogStore.Level(rawValue: global.defaultLogLevel) ?? .info
        do {
            try await gateway.start(
                host: host,
                port: global.gatewayPort,
                apiKey: global.apiKey,
                adminToken: global.adminToken,
                logLevel: level,
                defaultEngine: engine,
                allowedOrigins: global.corsOrigins
            )
            // §358 — surface auto-bump + duplicate-model warnings as
            // one-shot banners. If the requested gateway port was taken
            // and we bumped to the next free slot, tell the user once
            // so their API clients don't silently hit the wrong port.
            if let note = await gateway.drainAutoBumpNote() {
                flashBanner(note)
            }
            // Sync the in-memory gatewayStatus so the tray/sidebar can
            // show "Gateway on :8081" (actual bound) vs configured 8080.
            let bound = await gateway.port
            let requested = await gateway.requestedPort
            gatewayStatus = .running(boundPort: bound, requestedPort: requested)
        } catch {
            flashBanner("Gateway failed to start on \(host):\(global.gatewayPort) — \(error)")
            gatewayStatus = .failed(port: global.gatewayPort,
                                    message: "\(error)")
        }
    }

    /// Drain duplicate-model-registration warnings from the gateway and
    /// surface one flashBanner per offending display name. Called from
    /// `startSession` right after `gateway.registerEngine` lands, since
    /// that's the one spot where duplicates can appear.
    func drainGatewayDuplicateWarnings() async {
        let dupes = await gateway.drainDuplicateWarnings()
        for name in dupes {
            flashBanner(
                "Duplicate model name \"\(name)\" — gateway will route requests to the last-loaded session. Rename one in Server → Model alias to resolve."
            )
        }
    }

    /// Tear down the gateway listener. Called when the user flips the
    /// toggle off from the Tray. Idempotent.
    func stopGateway() async {
        await gateway.stop()
    }

    /// Tear down a per-session engine (called from SessionDashboard delete).
    /// Cancels the per-session state observer first so we don't leak a
    /// Task writing into a removed session's array slot.
    func removeEngine(for id: UUID) {
        stopObservingPerSessionEngine(id)
        if let e = engines.removeValue(forKey: id) {
            Task { await e.stop() }
        }
        if let srv = httpServers.removeValue(forKey: id) {
            Task { await srv.stop() }
        }
        // Reconcile app-nap activity in case the engine we just removed
        // was the last non-idle one. Without this, a session-delete while
        // that session was mid-generation would leak the activity token
        // forever (well, until app quit).
        refreshAppNapActivity()
    }

    // MARK: - Per-session HTTP servers

    /// Hummingbird `Server` supervisors keyed by session id. Populated on
    /// first `httpServer(for:)` call; started on `SessionDashboard.startSession`
    /// and stopped on `stopSession`/`deleteSession`. See `HTTPServerActor`.
    var httpServers: [UUID: HTTPServerActor] = [:]

    /// Gateway supervisor — single port that fans out to per-session
    /// engines by ChatRequest.model. Lazy: only spun up if the user
    /// flips on `GlobalSettings.gatewayEnabled` from the Tray.
    let gateway = GatewayActor()

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

    /// Seconds remaining until the currently-pending idle sleep event
    /// fires (soft-sleep or deep-sleep). `nil` when the IdleTimer is
    /// disabled or both events have already fired. Polled once per
    /// second from `rebindEngineObserver`'s spawn Task. Drives the tray
    /// "Sleeps in MM:SS" label so users stop being surprised when their
    /// model suddenly light-sleeps mid-afternoon. Repeated user ask:
    /// the idle countdown used to be totally invisible.
    var idleCountdownSeconds: TimeInterval? = nil
    var idleCountdownKindIsDeep: Bool = false

    /// A7 §259 — dual-stage idle countdown. Both soft and deep remainders
    /// are surfaced simultaneously so TrayItem can render
    /// "soft 4:12 · deep 14:12" instead of flip-flopping between one or
    /// the other. `nil` when the timer is disabled or that stage has
    /// already fired. The legacy `idleCountdownSeconds` above is kept
    /// so older surfaces (SessionCard, CLI status) don't need to be
    /// touched all at once.
    var idleCountdownSoftSeconds: TimeInterval? = nil
    var idleCountdownDeepSeconds: TimeInterval? = nil

    /// H1 §271 — thermal throttle surface. `ThermalMonitor.currentLevel()`
    /// wraps ProcessInfo.thermalState; this field mirrors it so the
    /// tray / chat banners can render a warning when the Mac is
    /// throttling (decode tok/s drops ~40% on `.serious`). Updated on
    /// app launch AND whenever the OS posts
    /// `.thermalStateDidChangeNotification` — see `rebindEngineObserver`
    /// for the NotificationCenter hook.
    var thermalLevel: ThermalMonitor.Level = .nominal

    /// Count of `.error`-level log entries since last acknowledgement.
    /// Drives the tray red-dot badge so users notice silent errors
    /// without opening the Logs panel. Reset via `acknowledgeErrors()`.
    /// H4 §273.
    var errorLogCount: Int = 0
    var lastErrorLogAcknowledged: Date = .distantPast

    /// Server-screen session list. Mirrors the Electron `sessions` IPC feed
    /// (`SessionsContext.tsx`). Each entry is a running *or* inactive model
    /// session with its per-session settings blob (port, model path, etc).
    ///
    /// Each row's `state` / `loadProgress` is maintained by a dedicated
    /// per-session observer spawned in `engine(for:)` — see
    /// `perSessionStateObservers` and `observePerSessionEngine(_:engine:)`.
    /// Multiple engines coexist concurrently; the global `engineState` /
    /// `loadProgress` pair tracks only the CURRENTLY-SELECTED session
    /// (driven by `rebindEngineObserver`).
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
        // Phase-4 follow-up: AppState.sessions used to live entirely in
        // memory, so quitting the app dropped every open server session.
        // SessionSettings persist per-id in SettingsDB; hydrate from there
        // on launch so users come back to the same session list they left.
        await hydrateSessionsFromSettings()
    }

    /// Rebuild `sessions` from persisted `SessionSettings` on launch so
    /// server sessions survive app restart. Each row gets a dedicated
    /// Engine actor + per-session state observer (same plumbing as
    /// `SessionDashboard.createSession`). Engines start in `.stopped` —
    /// the user clicks Start to reload the model.
    func hydrateSessionsFromSettings() async {
        guard sessions.isEmpty else { return }  // idempotent
        let store = await engine.settings
        let ids = await store.allSessionIDs()
        var restored: [Session] = []
        for id in ids {
            guard let s = await store.session(id) else { continue }
            let modelPath = s.modelPath
            // engine(for:) spawns the Engine actor + per-session state
            // observer; we never need to hold the returned actor here,
            // the dict lookup in subsequent dispatch paths picks it up.
            _ = engine(for: id)
            // 2026-04-18 iter-19: path-vanished guard. Users do delete
            // model dirs from Finder / HF-cache cleanups / `rm -rf`
            // between runs. Pre-fix the session card rendered cleanly
            // but hitting Start produced a cryptic engine-level load
            // error. Now we check upfront: if the path no longer
            // resolves to a directory, hydrate the session in `.error`
            // state so the UI surfaces a red banner immediately and
            // Start is disabled. Remote sessions (isRemote) skip this
            // guard — the endpoint is remote, the local path doesn't
            // apply.
            let fm = FileManager.default
            var isDir: ObjCBool = false
            let pathExists = fm.fileExists(
                atPath: modelPath.path, isDirectory: &isDir) && isDir.boolValue
            let missingLocalModel = !s.isRemote && !pathExists
            let hydratedState: EngineState = missingLocalModel
                ? .error("Model directory not found at \(modelPath.lastPathComponent) — re-download or point to a different model")
                : .stopped

            let session = Session(
                id: id,
                // HuggingFace cache paths end in a commit SHA
                // (`snapshots/<40-hex-sha>/`), which showed up as the
                // displayName fallback. `SessionHeuristics.displayName`
                // walks up to the first human-readable segment so the
                // session card shows e.g. `Qwen3-35B` instead of a
                // truncated hash. User-supplied `s.displayName` still
                // wins when set — but `sanitizedStoredName` rejects
                // bad names saved before the fix landed (SQLite rows
                // still carrying the SHA from old `createSession`).
                displayName: SessionHeuristics.sanitizedStoredName(s.displayName)
                    ?? SessionHeuristics.displayName(modelPath),
                modelPath: modelPath,
                family: SessionHeuristics.family(modelPath),
                isJANG: SessionHeuristics.isJANG(modelPath),
                isMXTQ: SessionHeuristics.isMXTQ(modelPath),
                quantBits: nil,
                host: s.host ?? "127.0.0.1",
                port: s.port ?? 8000,
                pid: nil,
                latencyMs: nil,
                state: hydratedState,
                loadProgress: nil,
                isRemote: s.isRemote
            )
            restored.append(session)
        }
        if !restored.isEmpty {
            sessions = restored
            // Pick the first hydrated session so the Server tab opens on
            // a meaningful row instead of the empty-state screen.
            if selectedServerSessionId == nil {
                selectedServerSessionId = restored.first?.id
                rebindEngineObserver()
            }
        }
    }

    func rebindEngineObserver() {
        engineObserver?.cancel()
        let current = engine
        engineObserver = Task { @MainActor [weak self] in
            guard let self else { return }
            // Sibling Task — 1 Hz idle-countdown poller. Lives alongside
            // the state-subscription loop so both cancel together when
            // the engine swaps. Writes `idleCountdownSeconds` so the
            // tray / chat banner can render "Sleeps in MM:SS". Poll is
            // cheap (single actor hop) and stops when the timer returns
            // nil (disabled / fired / no pending event).
            let countdownTask = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    guard let self else { break }
                    // iter-61: gate the countdown on `.running`. The
                    // IdleTimer keeps its internal `softFired`/`deepFired`
                    // booleans but those don't flip when the user
                    // MANUALLY soft-sleeps from the tray — so
                    // `nextSleepCountdown()` keeps returning a non-nil
                    // remaining time even though the engine is already
                    // in `.standby(.soft)`. That made the tray show
                    // "Sleeps in 3:45" next to a moon icon, which is
                    // a user-visible UI lie. Gate the countdown on
                    // the engine actually being in `.running` — only
                    // then can the idle timer legitimately fire.
                    let runningNow: Bool = {
                        if case .running = current.state { return true }
                        return false
                    }()
                    if runningNow,
                       let next = await current.idleTimer.nextSleepCountdown()
                    {
                        self.idleCountdownSeconds = next.seconds
                        self.idleCountdownKindIsDeep = (next.kind == .deepSleep)
                    } else {
                        self.idleCountdownSeconds = nil
                    }
                    // A7 §259 — also poll the dual-stage remainders so
                    // the tray can render both countdowns side-by-side.
                    // Gated on `.running` / `.standby(.soft)` — deep
                    // remainder is still meaningful once soft has fired.
                    let softOrRunning: Bool = {
                        switch current.state {
                        case .running: return true
                        case .standby(.soft): return true
                        default: return false
                        }
                    }()
                    if softOrRunning {
                        let pair = await current.idleTimer.sleepCountdowns()
                        self.idleCountdownSoftSeconds = pair.soft
                        self.idleCountdownDeepSeconds = pair.deep
                    } else {
                        self.idleCountdownSoftSeconds = nil
                        self.idleCountdownDeepSeconds = nil
                    }
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
            }
            defer { countdownTask.cancel() }
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

    // MARK: - Public session lifecycle
    //
    // These three helpers used to live as `private func`s inside
    // `SessionDashboard`. Lifted onto `AppState` so non-server surfaces
    // (Chat model picker, Terminal, tray, etc.) can start / stop / create
    // sessions WITHOUT the user having to pivot to the Server tab —
    // repeated user ask: "way to directly easily start/stop models from
    // the chat page". SessionDashboard now calls through here so both
    // surfaces share the same lifecycle path (no drift between buttons).

    /// Look up an existing session row for a model path, or return nil if
    /// none exists yet. Comparison uses `standardizedFileURL` so a
    /// resolved HF snapshot path matches the session's stored path even
    /// when symlinks / double-slashes differ.
    @MainActor
    func sessionId(forModelPath url: URL) -> UUID? {
        let canonical = url.standardizedFileURL.resolvingSymlinksInPath()
        return sessions.first { s in
            s.modelPath.standardizedFileURL.resolvingSymlinksInPath() == canonical
        }?.id
    }

    /// Create a fresh server session row for this model path and append
    /// it to `sessions`. Does NOT start the engine — caller must follow up
    /// with `startSession(_:)`. Returns the new session id.
    @MainActor
    func createSession(forModel url: URL) async -> UUID {
        let id = UUID()
        var settings = SessionSettings(modelPath: url)
        settings.displayName = SessionHeuristics.displayName(url)
        let eng = engine(for: id)
        selectedServerSessionId = id
        rebindEngineObserver()
        settings = await eng.applySessionSettings(id, settings)
        let session = Session(
            id: id,
            displayName: settings.displayName ?? SessionHeuristics.displayName(url),
            modelPath: url,
            family: SessionHeuristics.family(url),
            isJANG: SessionHeuristics.isJANG(url),
            isMXTQ: SessionHeuristics.isMXTQ(url),
            quantBits: nil,
            host: settings.host ?? "127.0.0.1",
            port: settings.port ?? 8000,
            pid: nil,
            latencyMs: nil,
            state: .stopped,
            loadProgress: nil,
            isRemote: settings.isRemote
        )
        sessions.append(session)
        return id
    }

    /// Start (= load weights + bring up HTTP listener) the session with
    /// this id. Mirrors the previous `SessionDashboard.startSession`
    /// flow. Safe to call for a session that is already running — the
    /// engine's `load(_:)` is idempotent.
    @MainActor
    func startSession(_ id: UUID) async {
        guard let idx = sessions.firstIndex(where: { $0.id == id }) else { return }
        let s = sessions[idx]
        let eng = engine(for: id)
        selectedServerSessionId = id
        // Also surface the newly-active model path globally. Consumers
        // that still read `selectedModelPath` (cmd+k quick picker, tray
        // title, pre-Gateway chat send() guard, older tests) stay in
        // sync — prevents the "loaded but chat still says not loaded"
        // class of bug.
        selectedModelPath = s.modelPath
        rebindEngineObserver()

        let remoteSettings = await eng.settings.session(id)
        if let r = remoteSettings, r.isRemote {
            if let idx2 = sessions.firstIndex(where: { $0.id == id }) {
                sessions[idx2].host = r.remoteURL ?? ""
                sessions[idx2].port = 0
                sessions[idx2].pid = Int(ProcessInfo.processInfo.processIdentifier)
            }
            return
        }

        let resolved = await eng.settings.resolved(sessionId: id)
        let opts = Engine.LoadOptions(modelPath: s.modelPath, from: resolved)
        do {
            for try await event in await eng.load(opts) {
                if case .failed(let msg) = event {
                    flashBanner("Engine load failed: \(msg)")
                    return
                }
            }
        } catch {
            flashBanner("Engine load failed: \(error)")
            return
        }

        let lanOn = resolved.settings.defaultLAN
        let bindHost = lanOn
            ? "0.0.0.0"
            : (resolved.settings.defaultHost.isEmpty ? "127.0.0.1" : resolved.settings.defaultHost)
        let http = httpServer(for: id)
        do {
            // iter-52: TLS plumbing. CLI had it since v1; SwiftUI
            // sessions didn't — users with ssl_keyfile/ssl_certfile
            // set in GlobalSettings got plain HTTP regardless.
            // Empty strings → nil so `Server.run()` falls back to the
            // HTTP path, non-empty → HTTPS via HummingbirdTLS.
            let tlsKey = resolved.settings.sslKeyFile
            let tlsCert = resolved.settings.sslCertFile
            try await http.start(
                host: bindHost,
                port: s.port,
                apiKey: resolved.settings.apiKey,
                adminToken: resolved.settings.adminToken,
                logLevel: LogStore.Level(rawValue: resolved.settings.defaultLogLevel) ?? .info,
                allowedOrigins: resolved.settings.corsOrigins,
                rateLimitPerMinute: resolved.settings.rateLimit,
                tlsKeyPath: tlsKey.isEmpty ? nil : tlsKey,
                tlsCertPath: tlsCert.isEmpty ? nil : tlsCert
            )
            if let idx2 = sessions.firstIndex(where: { $0.id == id }) {
                sessions[idx2].host = bindHost
                sessions[idx2].port = s.port
                sessions[idx2].pid = Int(ProcessInfo.processInfo.processIdentifier)
            }
            await gateway.registerEngine(eng)
            // §358 — surface duplicate display-name registrations. If this
            // session advertises a model that another session already did,
            // tell the user once so they can rename via Server → Model alias.
            await drainGatewayDuplicateWarnings()
            await ensureGatewayRunning()
        } catch {
            flashBanner("HTTP listener failed: \(error)")
            await eng.stop()
            if let idx2 = sessions.firstIndex(where: { $0.id == id }) {
                sessions[idx2].pid = nil
            }
        }
    }

    /// Stop (= unload + tear down HTTP listener) the session with this id.
    /// Safe for already-stopped sessions — engine.stop() + http.stop()
    /// are both idempotent.
    @MainActor
    func stopSession(_ id: UUID) async {
        let eng = engine(for: id)
        await gateway.unregisterEngine(eng)
        await httpServer(for: id).stop()
        await eng.stop()
        if let idx = sessions.firstIndex(where: { $0.id == id }) {
            sessions[idx].pid = nil
        }
    }

    /// Subscribe to the DownloadManager and keep `downloadJobs` in sync.
    /// On the first `.started` event, opens the Downloads window via the
    /// provided `openWindow` closure — nothing silent, ever.
    func observeDownloads(openWindow: @escaping (String) -> Void) async {
        // Capture so per-engine fan-in (forwardDownloadEvents) can also
        // auto-open the window on first .started for HTTP-initiated pulls.
        appOpenWindow = openWindow
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
            case .failed(let id, let message):
                if let job = await downloadManager.job(id) {
                    upsert(job)
                    // Surface the failure via the same system-notification
                    // path `.completed` uses — otherwise a background failure
                    // is silent, contradicting `feedback_download_window.md`
                    // "no silent downloads, ever" (the rule applies equally
                    // to failed outcomes).
                    DownloadNotifier.notifyFailed(job, message: message)
                }
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

/// O7 §293 — app-wide notifications used by the Downloads → HF token CTA.
/// `vmlxOpenHuggingFaceTokenCard` flips the sidebar to .api; once the
/// APIScreen is mounted it re-fires `vmlxFocusHuggingFaceTokenField`
/// which HuggingFaceTokenCard subscribes to, focusing the TextField
/// and scrolling the card into view.
extension Notification.Name {
    public static let vmlxOpenHuggingFaceTokenCard =
        Notification.Name("vmlx.openHuggingFaceTokenCard")
    public static let vmlxFocusHuggingFaceTokenField =
        Notification.Name("vmlx.focusHuggingFaceTokenField")
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
        // `vmlx://` deep-link handler. Called by macOS when the user
        // (or Shortcuts, or another app) opens a `vmlx://...` URL.
        // Paths are best-effort — unknown paths fall back to the
        // chat tab so the URL at least opens vMLX.
        .onOpenURL { url in
            handleIncomingURL(url, state: state)
        }
        // Drag-drop model files onto the main window. Drops are matched
        // by extension via `CFBundleDocumentTypes` in project.yml; the
        // handler walks up to find the containing model dir and adds
        // it to the library.
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            handleModelDrop(providers, state: state)
        }
        // O7 §293 — DownloadsWindow posts this when a job surfaces a
        // 401/403 from HuggingFace. Switch to the API tab so the user
        // lands on the HuggingFaceTokenCard, then broadcast a second
        // notification the card subscribes to so it can focus the
        // text field and scroll itself into view.
        .onReceive(NotificationCenter.default
            .publisher(for: .vmlxOpenHuggingFaceTokenCard)
        ) { _ in
            state.mode = .api
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                NotificationCenter.default.post(
                    name: .vmlxFocusHuggingFaceTokenField,
                    object: nil)
            }
        }
        .task { await state.observeEngine() }
        .task {
            await state.observeDownloads { id in
                openWindow(id: id)
            }
        }
        .task {
            // UI-9: if the gateway was on last session, bring it back up
            // at launch. Session registrations happen as each session
            // starts, so an empty model list on first boot just returns
            // a 404 until a session loads — fine behavior.
            await state.ensureGatewayRunning()
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
                // iter-55: previously just `engine.stop()` which left
                // the HTTP listener running with a dead engine —
                // requests arriving after the tray Stop would fail
                // with EngineError.notLoaded. Route through the full
                // `stopSession` teardown so the listener + gateway
                // registration unwind cleanly. Falls back to bare
                // `engine.stop()` when no active session is selected
                // (rare; happens before any session has loaded).
                Task { @MainActor in
                    if let id = st.selectedServerSessionId ?? st.sessions.first?.id {
                        await st.stopSession(id)
                    } else {
                        await st.engine.stop()
                    }
                }
            }
            st.onTrayRestartServer = {
                Task { @MainActor in
                    let eng = st.engine
                    let opts = await eng.lastLoadOptions
                    // Full teardown to drop the HTTP listener, then
                    // re-load. Matches onTrayStopServer shape so
                    // restart is a stop+start pair, not a silent
                    // engine-only reload.
                    if let id = st.selectedServerSessionId ?? st.sessions.first?.id {
                        await st.stopSession(id)
                    } else {
                        await eng.stop()
                    }
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

// MARK: - vmlx:// URL scheme + model drag-drop handlers

/// Handle a `vmlx://...` deep-link. Registered via
/// `CFBundleURLTypes` in project.yml; called by macOS when the URL is
/// dispatched through `NSWorkspace.shared.open(_:)` or Shortcuts.
///
/// Paths:
///   - `vmlx://chat/new`         — Chat tab + new session
///   - `vmlx://chat/<UUID>`      — Chat tab + jump to that session
///   - `vmlx://server/<UUID>`    — Server tab + select that session
///   - `vmlx://downloads`        — no-op (Downloads window comes up
///                                  automatically via `.onOpenURL`
///                                  routing to the named WindowGroup)
///   - anything else             — fall back to Chat tab so vMLX
///                                  at least opens
@MainActor
fileprivate func handleIncomingURL(_ url: URL, state: AppState) {
    guard url.scheme == "vmlx" else { return }
    let host = url.host ?? ""
    let path = url.pathComponents.filter { $0 != "/" }
    switch host {
    case "chat":
        state.mode = .chat
        if let first = path.first, first == "new" {
            state.chatViewModelRef?.newSession()
        } else if let first = path.first, let uuid = UUID(uuidString: first) {
            state.chatViewModelRef?.selectSession(uuid)
        }
    case "server":
        state.mode = .server
        if let first = path.first, let uuid = UUID(uuidString: first) {
            state.selectedServerSessionId = uuid
            state.rebindEngineObserver()
        }
    case "downloads":
        // Tell AppState to route the window open via its public
        // opener hook. If we got here before RootView mounted, the
        // opener is nil — the `.onOpenURL` hop will then open the
        // main window and the user can hit Cmd-Shift-D.
        state.openDownloadsWindowIfReady()
    default:
        state.mode = .chat
    }
}

/// Drop handler for model files (`.safetensors` / `.gguf` / `.mlxq`)
/// dragged onto the main window. Walks up to the containing directory
/// (a single-file drop of `model-00001-of-00004.safetensors` points at
/// the model bundle, not the file itself), adds it as a user model
/// directory so the scanner picks it up, and flashes a banner.
///
/// Returns `true` iff at least one valid provider was accepted.
@MainActor
fileprivate func handleModelDrop(_ providers: [NSItemProvider], state: AppState) -> Bool {
    var accepted = false
    for p in providers {
        guard p.canLoadObject(ofClass: URL.self) else { continue }
        accepted = true
        _ = p.loadObject(ofClass: URL.self) { url, _ in
            guard let url else { return }
            // Single-file drop: walk up to the containing directory.
            // Drag of an already-expanded model directory: use the
            // directory itself.
            var targetDir = url
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
               !isDir.boolValue
            {
                targetDir = url.deletingLastPathComponent()
            }
            Task { @MainActor in
                let lib = await state.engine.modelLibrary
                await lib.addUserDir(targetDir)
                let entries = await lib.scan(force: true)
                // §358 — honest discovery toast. Count what we found IN
                // the folder the user just added (not the full library),
                // split by modality so "added 3 image + 7 text models"
                // tells them the image-gen picker just got populated.
                let lowered = targetDir.path.lowercased()
                let inThisDir = entries.filter { entry in
                    entry.canonicalPath.path.lowercased().hasPrefix(lowered)
                }
                let text = inThisDir.filter { $0.modality != .image }.count
                let img  = inThisDir.filter { $0.modality == .image }.count
                let parts: [String] = {
                    var p: [String] = []
                    if text > 0 { p.append("\(text) text") }
                    if img > 0 { p.append("\(img) image") }
                    return p
                }()
                let summary = parts.isEmpty ? "no models detected" : parts.joined(separator: " + ") + " models"
                state.flashBanner("Added \(targetDir.lastPathComponent) — \(summary)")
            }
        }
    }
    return accepted
}

private struct Sidebar: View {
    @Binding var mode: AppState.Mode
    @Environment(AppState.self) private var appState
    @Environment(\.appLocale) private var appLocale: AppLocale

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
                        Text(label(for: m))
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

    private func label(for m: AppState.Mode) -> String {
        switch m {
        case .chat:     return L10n.Mode.chat.render(appLocale)
        case .server:   return L10n.Mode.server.render(appLocale)
        case .image:    return L10n.Mode.image.render(appLocale)
        case .terminal: return L10n.Mode.terminal.render(appLocale)
        case .api:      return L10n.Mode.api.render(appLocale)
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
    // Honor the system-wide Reduce Motion accessibility setting. When the
    // user has it on, the indeterminate barber-pole sweep is replaced with
    // a centered static bar — still clearly "loading" but no motion.
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Theme.Colors.surfaceHi)
                if let f = fraction {
                    Capsule()
                        .fill(Theme.Colors.accent)
                        .frame(width: max(2, geo.size.width * CGFloat(f.clamped01)))
                        .animation(reduceMotion ? nil : .easeOut(duration: 0.25), value: f)
                } else if reduceMotion {
                    // Static "busy" indicator — 60% wide, centered — so the
                    // bar is visibly non-zero without any sweep animation.
                    Capsule()
                        .fill(Theme.Colors.accent)
                        .frame(width: geo.size.width * 0.6)
                        .offset(x: geo.size.width * 0.2)
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
