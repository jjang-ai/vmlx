// SPDX-License-Identifier: Apache-2.0
//
// Multi-engine gateway supervisor. Owns one GatewayServer task and a
// per-model registry mapping `ChatRequest.model` strings → Engine. Engines
// are registered as sessions start and unregistered as they stop. The
// gateway port is opt-in via GlobalSettings.gatewayEnabled +
// GlobalSettings.gatewayPort.

import Foundation
import vMLXEngine
import vMLXServer

/// Supervises the optional gateway HTTP listener and the model→engine
/// registry that drives request routing.
actor GatewayActor {

    // MARK: - Registry state

    /// Model display name → Engine. Each entry is keyed by the canonical
    /// display name used by `/v1/models` (`ModelEntry.displayName`).
    private var registry: [String: WeakEngineBox] = [:]

    /// Insertion order so the default engine resolution stays deterministic.
    private var insertionOrder: [String] = []

    private var runTask: Task<Void, Error>?
    private(set) var host: String = "127.0.0.1"
    /// Port the caller *requested* when start() was invoked.
    private(set) var requestedPort: Int = 8080
    /// Port we actually bound — may differ from `requestedPort` when the
    /// requested port was taken (Ollama on 8080 is the common case) and
    /// the `allowAutoBump` flag in `start()` kicked us to the next free
    /// slot. UI should display this value, not `requestedPort`.
    private(set) var port: Int = 8080
    private(set) var lastError: String?

    /// Last collision we auto-recovered from. Surfaced to the UI as a
    /// one-line banner so the user knows we bumped off their originally
    /// configured port (e.g. 8080 → 8081 when Ollama is running).
    private(set) var lastAutoBumpNote: String?

    /// Duplicate model-name registrations discovered since the last UI
    /// pull. Keyed by model display name → list of engine ObjectIdentifiers
    /// that tried to claim it. UI polls this via `drainDuplicateWarnings()`
    /// to surface a one-shot banner ("two sessions both register 'llama-7b';
    /// requests will route to the last-loaded one").
    private var duplicateWarnings: [String] = []

    var isRunning: Bool { runTask != nil }

    // MARK: - Registry API

    /// Register an engine under every display name it currently advertises.
    /// Call from `SessionDashboard.startSession` once the engine has loaded
    /// (so its ModelLibrary scan is fresh).
    ///
    /// §359 — legacy behavior was "register every model in the library"
    /// which leaked the full on-disk catalog through `/v1/models` (this
    /// session didn't load those). Use `registerEngine(_:loadedModel:alias:)`
    /// instead for new call sites.
    func registerEngine(_ engine: Engine) async {
        let entries = await engine.modelLibrary.entries()
        for entry in entries {
            if let existing = registry[entry.displayName]?.engine,
               existing !== engine {
                duplicateWarnings.append(entry.displayName)
            }
            registry[entry.displayName] = WeakEngineBox(engine: engine)
            if !insertionOrder.contains(entry.displayName) {
                insertionOrder.append(entry.displayName)
            }
        }
    }

    /// §359 — targeted registration: register this engine under ONLY the
    /// model it loaded + an optional user-supplied alias. Replaces the
    /// "register everything in the library" path for session startups.
    ///
    /// - Parameters:
    ///   - engine: the Engine actor for the session that just finished loading
    ///   - canonicalName: the `ModelEntry.displayName` of the loaded model
    ///   - alias: optional user-supplied nickname (`SessionSettings.modelAlias`).
    ///     When set, `/v1/chat/completions` requests with `model: "<alias>"`
    ///     route to this engine, and `/v1/models` advertises both names.
    func registerEngine(
        _ engine: Engine,
        canonicalName: String,
        alias: String?
    ) async {
        // Collision detection against both canonical + alias.
        var toRegister: [String] = [canonicalName]
        if let a = alias, !a.isEmpty, a != canonicalName {
            toRegister.append(a)
        }
        for name in toRegister {
            if let existing = registry[name]?.engine, existing !== engine {
                duplicateWarnings.append(name)
            }
            registry[name] = WeakEngineBox(engine: engine)
            if !insertionOrder.contains(name) {
                insertionOrder.append(name)
            }
        }
    }

    /// Drain queued duplicate-name warnings. Each returned name represents
    /// a model display-name that two or more engines tried to register; the
    /// engine that registered last wins all gateway routing for that name.
    /// Caller (AppState.observeEngine) surfaces these as flashBanners.
    func drainDuplicateWarnings() -> [String] {
        let w = duplicateWarnings
        duplicateWarnings.removeAll()
        return w
    }

    /// Drop every registry entry pointing at this engine. Called on session
    /// stop / delete.
    func unregisterEngine(_ engine: Engine) {
        let stale = registry.filter { _, box in box.engine === engine }
        for key in stale.keys {
            registry.removeValue(forKey: key)
            insertionOrder.removeAll { $0 == key }
        }
    }

    /// Resolve a `ChatRequest.model` string to the engine that should
    /// handle it. Falls back to the first registered engine if the string
    /// is empty (defaults to "use whatever you've got"). Returns nil only
    /// when the string is non-empty AND no match exists — surfaces as a
    /// 404 with the available-models list.
    func resolve(_ model: String) -> Engine? {
        if model.isEmpty {
            for key in insertionOrder {
                if let engine = registry[key]?.engine { return engine }
            }
            return nil
        }
        if let engine = registry[model]?.engine { return engine }
        // iter-86 §114: tolerant bare-repo match. MUST require a `/`
        // boundary so a client passing `model: "4bit"` doesn't
        // accidentally match a registry key ending in `-4bit`
        // (`mlx-community/gemma-4-e2b-it-4bit`). The previous
        // unbounded suffix fallback (hasSuffix on the raw model
        // string) picked whichever engine registered first, which
        // is non-deterministic under concurrent load.
        for (k, box) in registry where k.hasSuffix("/" + model) {
            if let engine = box.engine { return engine }
        }
        return nil
    }

    /// All currently-registered engines (deduped). Used by `/v1/models`
    /// and the model-not-found 404.
    func allEngines() -> [Engine] {
        var seen = Set<ObjectIdentifier>()
        var out: [Engine] = []
        for key in insertionOrder {
            guard let engine = registry[key]?.engine else { continue }
            let id = ObjectIdentifier(engine)
            if !seen.contains(id) {
                seen.insert(id)
                out.append(engine)
            }
        }
        return out
    }

    // MARK: - Lifecycle

    /// Start the gateway listener. Called when the user flips the
    /// gateway toggle on, or at app launch if the persisted setting is
    /// already on. Re-binding to a new (host, port) tears down the old
    /// task before starting fresh.
    func start(
        host: String,
        port: Int,
        apiKey: String?,
        adminToken: String?,
        logLevel: LogStore.Level,
        defaultEngine: Engine,
        allowedOrigins: [String] = ["*"],
        allowAutoBump: Bool = true
    ) async throws {
        if let task = runTask, host == self.host, port == self.port, !task.isCancelled {
            return
        }
        await stop()

        self.host = host
        self.requestedPort = port
        self.port = port
        self.lastError = nil
        self.lastAutoBumpNote = nil

        // §358 — auto-bump on collision. Ollama binds 8080 by default,
        // which is also our gateway default. Previous behavior threw a
        // portInUse and the user saw a 3-sec banner but never a bound
        // gateway, so cross-app routing silently died. Now: scan for
        // the next free port in [port, port+32], record the bump, and
        // bind there. UI reads `port` for display and `lastAutoBumpNote`
        // for the one-shot "moved from 8080→8081 (was taken)" banner.
        var bound = port
        if !Engine.isPortFree(port) {
            if allowAutoBump {
                if let fallback = Self.findFreePort(startingAt: port + 1, limit: 32) {
                    bound = fallback
                    self.port = bound
                    self.lastAutoBumpNote =
                        "Gateway port \(port) was taken — bound to \(bound) instead. " +
                        "Change the configured port in Tray → Server if you need a stable target."
                } else {
                    throw EngineError.portInUse(port)
                }
            } else {
                throw EngineError.portInUse(port)
            }
        }
        let _ = bound  // suppress unused warning — already written to self.port

        // Capture self weakly inside the resolver closures so the gateway
        // doesn't pin this actor in memory if the user later disables it.
        let server = GatewayServer(
            host: host,
            port: self.port,
            apiKey: apiKey,
            adminToken: adminToken,
            logLevel: logLevel,
            defaultEngine: defaultEngine,
            resolver: { [weak self] model in
                guard let self else { return nil }
                return await self.resolve(model)
            },
            enumerate: { [weak self] in
                guard let self else { return [] }
                return await self.allEngines()
            },
            allowedOrigins: allowedOrigins
        )
        runTask = Task {
            do {
                try await server.run()
            } catch {
                await self.recordRunError("\(error)")
                throw error
            }
        }
    }

    /// Cancel + drain the gateway task. Idempotent.
    func stop() async {
        guard let task = runTask else { return }
        runTask = nil
        task.cancel()
        _ = try? await task.value
    }

    private func recordRunError(_ message: String) {
        lastError = message
    }

    /// Drain the auto-bump note (one-shot) so the UI banner fires once
    /// per actual bump. Called from AppState.ensureGatewayRunning right
    /// after a successful start.
    func drainAutoBumpNote() -> String? {
        let note = lastAutoBumpNote
        lastAutoBumpNote = nil
        return note
    }

    /// Internal port scan — actor-isolated because Engine.isPortFree is
    /// synchronous and cheap, so no need to escape the actor queue.
    private static func findFreePort(startingAt start: Int, limit: Int) -> Int? {
        var p = start
        let end = min(start + limit, 65_535)
        while p <= end {
            if Engine.isPortFree(p) { return p }
            p += 1
        }
        return nil
    }
}

/// Weak Engine reference holder so the registry doesn't pin engines in
/// memory after their session is deleted. Engine is a class-bound actor
/// so the weak reference works as expected.
private final class WeakEngineBox: @unchecked Sendable {
    weak var engine: Engine?
    init(engine: Engine) { self.engine = engine }
}
