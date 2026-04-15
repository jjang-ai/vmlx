import Foundation
import Security

/// Generic macOS Keychain wrapper used for secrets that must NOT be stored
/// in SQLite or UserDefaults. Today this backs the HuggingFace access token
/// (for gated repo downloads); tomorrow it can back remote-endpoint API keys,
/// notarization app-specific passwords, and anything else the user types in
/// that shouldn't survive in plaintext on disk.
///
/// The helper uses `kSecClassGenericPassword` with a fixed service string per
/// secret. Items are scoped to the vMLX bundle via its bundle identifier —
/// not the app sandbox, since vMLX ships as a non-sandboxed Developer ID
/// distribution — so every secret lives under vMLX's ACL.
///
/// Error handling is deliberately permissive: save/load errors are logged to
/// stderr and surface as `nil` (load) or `false` (save) so the UI can treat
/// "no token yet" identically to "keychain unavailable". Callers that need
/// to distinguish can use `loadStatus` / `saveStatus` to get the raw OSStatus.
public enum KeychainHelper {

    /// Canonical service strings — one per secret. Keeping them in an enum
    /// makes it obvious when a new secret is added and prevents typos.
    public enum Service: String {
        case hfToken = "ai.jangq.vmlx.hf_token"
        case remoteAPIKey = "ai.jangq.vmlx.remote_api_key"
    }

    // MARK: - Public API

    /// Save a secret. Passing `nil` or empty deletes the entry (equivalent
    /// to `delete`). Returns true on success.
    @discardableResult
    public static func save(_ service: Service, account: String = "default", value: String?) -> Bool {
        guard let value, !value.isEmpty else {
            return delete(service, account: account)
        }
        return saveStatus(service, account: account, value: value) == errSecSuccess
    }

    /// Load a secret. Returns nil if missing or if the keychain is locked.
    public static func load(_ service: Service, account: String = "default") -> String? {
        var result: AnyObject?
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service.rawValue,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Delete a secret. Returns true if the deletion succeeded or the item
    /// was already missing. `errSecItemNotFound` is treated as success.
    @discardableResult
    public static func delete(_ service: Service, account: String = "default") -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service.rawValue,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    /// True if a non-empty secret is stored for this service/account.
    public static func has(_ service: Service, account: String = "default") -> Bool {
        load(service, account: account)?.isEmpty == false
    }

    // MARK: - Raw OSStatus (for diagnostic UI)

    public static func saveStatus(_ service: Service, account: String = "default", value: String) -> OSStatus {
        guard let data = value.data(using: .utf8) else { return errSecParam }

        // Try to update first — if the item exists we replace its data.
        let lookup: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service.rawValue,
            kSecAttrAccount as String: account,
        ]
        let attrs: [String: Any] = [kSecValueData as String: data]
        let updateStatus = SecItemUpdate(lookup as CFDictionary, attrs as CFDictionary)
        if updateStatus == errSecSuccess { return errSecSuccess }

        // Item didn't exist — insert it. `kSecAttrAccessibleWhenUnlocked` is
        // the default but we set it explicitly so the behaviour survives a
        // future security defaults change from Apple.
        var add = lookup
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked
        return SecItemAdd(add as CFDictionary, nil)
    }
}
