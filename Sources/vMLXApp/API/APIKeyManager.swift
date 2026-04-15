import Foundation

/// Thin wrapper around the `api_keys` SQLite table (see `Database.swift`).
/// API keys generated here are surfaced in the API screen and in bearer-auth
/// mode are enforced by `vMLXServer.BearerAuthMiddleware`.
///
/// The store is a `@MainActor` singleton bridging to the existing
/// `Database.shared` and posts a simple AsyncStream of change events so
/// the API screen can subscribe without manual reloads.
@MainActor
final class APIKeyManager {
    static let shared = APIKeyManager()

    /// Live view of keys — anyone observing can read this directly between
    /// change events. `APIScreen` rebuilds off `subscribe()` + `list()`.
    private(set) var cached: [Database.APIKeyRow] = []

    private var continuations: [UUID: AsyncStream<[Database.APIKeyRow]>.Continuation] = [:]

    private init() {
        cached = Database.shared.allAPIKeys()
    }

    // MARK: - API

    func list() -> [Database.APIKeyRow] {
        cached = Database.shared.allAPIKeys()
        return cached
    }

    /// Generates a 32-char URL-safe random key with the `vmlx_` prefix.
    /// Schema matches `server.py::BearerAuthMiddleware` — the middleware just
    /// does a byte-equals compare, so prefix is purely cosmetic.
    @discardableResult
    func generate(label: String) -> Database.APIKeyRow {
        let value = "vmlx_" + Self.randomURLSafe(length: 32)
        let row = Database.APIKeyRow(
            id: UUID().uuidString,
            label: label.isEmpty ? "Untitled" : label,
            value: value,
            createdAt: Date(),
            lastUsedAt: nil
        )
        Database.shared.insertAPIKey(row)
        reloadAndBroadcast()
        return row
    }

    func revoke(id: String) {
        Database.shared.deleteAPIKey(id: id)
        reloadAndBroadcast()
    }

    func markUsed(id: String) {
        Database.shared.touchAPIKey(id: id, at: Date())
        reloadAndBroadcast()
    }

    /// Subscribe to a stream of snapshots. The first element is the current
    /// state; every mutation pushes a new snapshot.
    func subscribe() -> AsyncStream<[Database.APIKeyRow]> {
        AsyncStream { cont in
            let token = UUID()
            cont.yield(cached)
            continuations[token] = cont
            cont.onTermination = { [weak self] _ in
                Task { @MainActor in self?.continuations.removeValue(forKey: token) }
            }
        }
    }

    // MARK: - internal

    private func reloadAndBroadcast() {
        cached = Database.shared.allAPIKeys()
        for (_, c) in continuations { c.yield(cached) }
    }

    /// URL-safe random characters (a-z, A-Z, 0-9, -_). 6 bits per char → 32
    /// chars ≈ 192 bits of entropy, well past what we need for bearer auth.
    private static func randomURLSafe(length: Int) -> String {
        let alphabet = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        var out = ""
        out.reserveCapacity(length)
        for _ in 0..<length {
            out.append(alphabet[Int.random(in: 0..<alphabet.count)])
        }
        return out
    }
}
