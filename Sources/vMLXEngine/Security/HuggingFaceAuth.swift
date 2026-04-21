import Foundation

/// HuggingFace authentication coordinator.
///
/// Owns the user-facing lifecycle of a HF access token: read it out of the
/// Keychain on app launch, validate it against the HF API, push it into any
/// `DownloadManager` actors that care, and surface the authenticated
/// username (if any) so UI can render "Signed in as @alice" banners.
///
/// This is the only correct place in the app to mutate a HF token. Direct
/// `KeychainHelper.save(.hfToken, ...)` calls should be avoided — going
/// through `HuggingFaceAuth` guarantees the DownloadManager and the UI
/// state stay in lockstep.
@MainActor
public final class HuggingFaceAuth: ObservableObject {
    public static let shared = HuggingFaceAuth()

    // MARK: - Published state

    /// Whether a token is currently stored in the Keychain. Purely derived —
    /// updated whenever `save`, `clear`, or `loadFromKeychain` runs.
    @Published public private(set) var hasToken: Bool = false

    /// Username resolved via `/api/whoami-v2`. Nil until a successful
    /// validation call has completed. `save(validate: true)` fills this in.
    @Published public private(set) var username: String? = nil

    /// Result of the last validation attempt. Drives error banners in the UI.
    public enum ValidationState: Equatable {
        case unknown
        case validating
        case valid(username: String)
        case invalid(reason: String)
    }
    @Published public private(set) var validation: ValidationState = .unknown

    // MARK: - Download manager binding

    /// Download manager instances that should receive `setHFAuthToken` any
    /// time the token changes. Held weakly so the auth singleton can outlive
    /// any individual engine.
    private var boundManagers: [WeakDownloadManagerRef] = []

    /// Register a DownloadManager to receive token updates. Call this when
    /// the Engine spins up so gated-repo downloads pick up the current token.
    public func bind(_ manager: DownloadManager) {
        boundManagers.removeAll { $0.value == nil }
        boundManagers.append(WeakDownloadManagerRef(value: manager))
        // Push the current token immediately so freshly-bound managers are
        // auth'd from the first request.
        let token = KeychainHelper.load(.hfToken)
        Task { await manager.setHFAuthToken(token) }
    }

    // MARK: - Token management

    /// Read the current token from the Keychain. Call at app launch before
    /// any download is triggered. Also triggers a background validation if
    /// a token is present and we don't yet have a cached username.
    public func loadFromKeychain() {
        let token = KeychainHelper.load(.hfToken)
        hasToken = (token?.isEmpty == false)
        pushToAllManagers(token: token)
        if hasToken && username == nil {
            Task { await self.validate() }
        }
    }

    /// Save a new token. `validate=true` (default) round-trips to
    /// /api/whoami-v2 before committing; on failure the Keychain is left
    /// untouched and `.invalid` is published. `validate=false` trusts the
    /// caller and writes unconditionally (useful for tests / migration).
    @discardableResult
    public func save(token: String, validate: Bool = true) async -> ValidationState {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            clear()
            return .invalid(reason: "Empty token")
        }

        if validate {
            self.validation = .validating
            let result = await Self.validateToken(trimmed)
            switch result {
            case .valid(let user):
                _ = KeychainHelper.save(.hfToken, value: trimmed)
                self.username = user
                self.hasToken = true
                self.validation = .valid(username: user)
                self.pushToAllManagers(token: trimmed)
                return .valid(username: user)
            case .invalid(let reason):
                self.validation = .invalid(reason: reason)
                return .invalid(reason: reason)
            case .unknown, .validating:
                return self.validation
            }
        } else {
            _ = KeychainHelper.save(.hfToken, value: trimmed)
            self.hasToken = true
            self.validation = .unknown
            self.pushToAllManagers(token: trimmed)
            return .unknown
        }
    }

    /// Forget the token. Clears the Keychain, unsets the username, pushes
    /// nil to all bound DownloadManagers.
    /// Current token (if any). Used by consumers like the HF search
    /// panel that need to attach the token to an API call without going
    /// through the DownloadManager binding path. Returns nil when no
    /// token is stored.
    public func currentToken() -> String? {
        let t = KeychainHelper.load(.hfToken)
        return (t?.isEmpty ?? true) ? nil : t
    }

    public func clear() {
        _ = KeychainHelper.delete(.hfToken)
        self.hasToken = false
        self.username = nil
        self.validation = .unknown
        self.pushToAllManagers(token: nil)
    }

    /// Re-validate the current token against HF. Useful for a "Test" button.
    public func validate() async {
        guard let token = KeychainHelper.load(.hfToken), !token.isEmpty else {
            self.validation = .invalid(reason: "No token")
            return
        }
        self.validation = .validating
        self.validation = await Self.validateToken(token)
        if case .valid(let user) = self.validation {
            self.username = user
            self.hasToken = true
        }
    }

    // MARK: - Internal

    private func pushToAllManagers(token: String?) {
        boundManagers.removeAll { $0.value == nil }
        for ref in boundManagers {
            if let mgr = ref.value {
                Task { await mgr.setHFAuthToken(token) }
            }
        }
    }

    /// Hit the HF `/api/whoami-v2` endpoint to verify a token is live and
    /// extract the username. Uses an ephemeral URLSession so nothing is
    /// cached between validations.
    private static func validateToken(_ token: String) async -> ValidationState {
        guard let url = URL(string: "https://huggingface.co/api/whoami-v2") else {
            return .invalid(reason: "Bad URL")
        }
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.cachePolicy = .reloadIgnoringLocalCacheData
        do {
            let (data, response) = try await URLSession(configuration: .ephemeral)
                .data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return .invalid(reason: "Network error")
            }
            if http.statusCode == 401 || http.statusCode == 403 {
                return .invalid(reason: "Token rejected by HuggingFace")
            }
            if http.statusCode >= 400 {
                return .invalid(reason: "HF returned \(http.statusCode)")
            }
            // whoami-v2 shape: {"name":"alice","type":"user", ...}
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let name = obj["name"] as? String
            {
                return .valid(username: name)
            }
            return .invalid(reason: "Malformed response")
        } catch {
            return .invalid(reason: error.localizedDescription)
        }
    }
}

/// Weak reference holder so `HuggingFaceAuth.boundManagers` doesn't retain
/// the engine. Swift arrays can't hold `weak` directly.
private final class WeakDownloadManagerRef {
    weak var value: DownloadManager?
    init(value: DownloadManager) { self.value = value }
}
