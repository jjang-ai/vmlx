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
    private(set) var port: Int = 8080
    private(set) var lastError: String?

    var isRunning: Bool { runTask != nil }

    // MARK: - Registry API

    /// Register an engine under every display name it currently advertises.
    /// Call from `SessionDashboard.startSession` once the engine has loaded
    /// (so its ModelLibrary scan is fresh).
    func registerEngine(_ engine: Engine) async {
        let entries = await engine.modelLibrary.entries()
        for entry in entries {
            registry[entry.displayName] = WeakEngineBox(engine: engine)
            if !insertionOrder.contains(entry.displayName) {
                insertionOrder.append(entry.displayName)
            }
        }
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
        allowedOrigins: [String] = ["*"]
    ) async throws {
        if let task = runTask, host == self.host, port == self.port, !task.isCancelled {
            return
        }
        await stop()

        self.host = host
        self.port = port
        self.lastError = nil

        guard Engine.isPortFree(port) else {
            throw EngineError.portInUse(port)
        }

        // Capture self weakly inside the resolver closures so the gateway
        // doesn't pin this actor in memory if the user later disables it.
        let server = GatewayServer(
            host: host,
            port: port,
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
}

/// Weak Engine reference holder so the registry doesn't pin engines in
/// memory after their session is deleted. Engine is a class-bound actor
/// so the weak reference works as expected.
private final class WeakEngineBox: @unchecked Sendable {
    weak var engine: Engine?
    init(engine: Engine) { self.engine = engine }
}
