// SPDX-License-Identifier: Apache-2.0
//
// Per-session Hummingbird server supervisor for the SwiftUI app.
//
// Previously the SwiftUI Session tab only loaded the model — the HTTP
// listener was never actually started, so every session showed PID:nil
// in the tray and no external clients could hit the loaded model. The
// only HTTP entrypoint was `vMLXCLI/main.swift`.
//
// This actor owns the `vMLXServer.Server` task for exactly one session.
// `SessionDashboard.startSession` starts it after the model loads;
// `stopSession` cancels and awaits teardown. The actor is keyed on the
// session UUID in `AppState.httpServers` so multiple sessions can run
// simultaneously on different ports.

import Foundation
import vMLXEngine
import vMLXServer

/// Supervises a background Hummingbird `Server` task for a single session.
actor HTTPServerActor {

    // MARK: - State

    private let engine: Engine
    private let sessionId: UUID
    private var runTask: Task<Void, Error>?
    private(set) var host: String = "127.0.0.1"
    private(set) var port: Int = 8000
    private(set) var lastError: String?

    /// `true` while a run task is live, whether or not it has finished
    /// binding its listener. Used by the UI to show running/stopped state.
    var isRunning: Bool { runTask != nil }

    // MARK: - Lifecycle

    init(engine: Engine, sessionId: UUID) {
        self.engine = engine
        self.sessionId = sessionId
    }

    /// Start the HTTP listener. If a task is already running, tears it
    /// down first so the new bind can take effect (e.g. after a port change).
    ///
    /// - Parameters:
    ///   - host: The bind host — `"127.0.0.1"` for loopback-only, `"0.0.0.0"`
    ///     when the user flipped the LAN toggle.
    ///   - port: The TCP port to bind.
    ///   - apiKey: Optional bearer token; when nil, `BearerAuthMiddleware`
    ///     is a no-op and every request is admitted.
    ///   - adminToken: Optional admin-token; when nil, `AdminAuthMiddleware`
    ///     is a no-op. When set, gates `/admin/*` and `/v1/cache/*`.
    func start(host: String, port: Int, apiKey: String?, adminToken: String? = nil, logLevel: LogStore.Level = .info, allowedOrigins: [String] = ["*"], rateLimitPerMinute: Int = 0, tlsKeyPath: String? = nil, tlsCertPath: String? = nil) async throws {
        // If we're already running on the same endpoint, nothing to do.
        if let task = runTask, host == self.host, port == self.port, !task.isCancelled {
            return
        }
        // Otherwise tear down the existing task before starting fresh.
        await stop()

        self.host = host
        self.port = port
        self.lastError = nil

        // Sanity check the bind upfront so the caller gets a clean error
        // instead of a fatal trap deep inside Hummingbird. `Server.run`
        // does its own preflight check too (fixes mlxstudio #44) but we
        // want to surface it synchronously to the UI so a failed bind
        // doesn't leave the run task in a half-started state.
        guard Engine.isPortFree(port) else {
            throw EngineError.portInUse(port)
        }

        let srv = Server(
            engine: engine,
            host: host,
            port: port,
            apiKey: apiKey,
            adminToken: adminToken,
            logLevel: logLevel,
            tlsKeyPath: tlsKeyPath,
            tlsCertPath: tlsCertPath,
            rateLimitPerMinute: rateLimitPerMinute,
            allowedOrigins: allowedOrigins
        )
        runTask = Task {
            do {
                try await srv.run()
            } catch {
                // Record the error so the next status poll surfaces it.
                await self.recordRunError("\(error)")
                throw error
            }
        }
    }

    /// Gracefully cancel the run task and wait for it to finish. If there
    /// is no task, this is a no-op.
    func stop() async {
        guard let task = runTask else { return }
        runTask = nil
        task.cancel()
        // Await the cancellation; ignore the throw since we expect
        // `CancellationError` as the normal shutdown path.
        _ = try? await task.value
    }

    /// Background helper used by the `runTask` closure to write the
    /// error field from a non-isolated context.
    private func recordRunError(_ message: String) {
        lastError = message
    }

    /// A stable identifier for `AppState.httpServers` keying.
    var id: UUID { sessionId }
}
