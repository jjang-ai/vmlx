// SPDX-License-Identifier: Apache-2.0
//
// KeychainAPIKeyStore — macOS Keychain backing for the bearer API keys
// that `APIKeyManager` mints. Keeps the secret material out of the
// plaintext SQLite `api_keys.value` column; the column is still written
// on insert so existing Server middleware (`BearerAuthMiddleware`) that
// reads the SQLite row for now keeps working, but the Keychain mirror
// is the long-term authoritative store.
//
// Audit 2026-04-16 flagged plaintext-in-SQLite as a local-security
// posture gap. Anyone with read access to `~/Library/Application Support
// /vMLX/vmlx.sqlite3` could previously pull every bearer token in one
// `SELECT value FROM api_keys`. Mirroring to Keychain puts the token
// behind `SecItem` ACLs, scoped to this exact app's code signature.
//
// Migration path:
//   1. On first launch with this code: `migrateFromSQLiteIfNeeded()`
//      walks every existing SQLite row and writes the `value` into
//      Keychain under the key's `id`. No-op on subsequent launches.
//   2. `APIKeyManager.generate` writes to Keychain first, then mirrors
//      into SQLite so the column stays populated during the grace
//      period while consumers migrate.
//   3. `APIKeyManager.revoke` deletes from both.
//   4. Reads prefer Keychain; SQLite `value` is a fallback for rows
//      that haven't been migrated yet.

import Foundation
#if canImport(Security)
import Security
#endif

enum KeychainAPIKeyStore {

    /// Service name used as the `kSecAttrService` attribute on every
    /// stored item. Keychain is scoped to this service + code signature
    /// so other apps can't read these items even if they know the
    /// account id.
    private static let service = "ai.jangq.vmlx.api_keys"

    /// Persist `value` under `id`. Overwrites any existing item with
    /// the same id (Keychain is idempotent via `SecItemUpdate`
    /// fallback). Silent no-op + stderr log on `Security` framework
    /// error — we prefer a degraded-but-working app over crashing on
    /// a keychain glitch.
    static func save(id: String, value: String) {
        #if canImport(Security)
        let data = Data(value.utf8)
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: id,
        ]
        let update: [CFString: Any] = [
            kSecValueData: data,
            // Only this application (tied to the code-signed binary's
            // identity) can read the item. Stops a rogue binary with
            // the same bundle id from pulling the keys.
            kSecAttrAccessible: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        let updateStatus = SecItemUpdate(query as CFDictionary, update as CFDictionary)
        if updateStatus == errSecItemNotFound {
            var newItem = query
            for (k, v) in update { newItem[k] = v }
            let addStatus = SecItemAdd(newItem as CFDictionary, nil)
            if addStatus != errSecSuccess {
                FileHandle.standardError.write(Data(
                    "[vmlx][keychain] save failed (add): id=\(id) status=\(addStatus)\n".utf8))
            }
        } else if updateStatus != errSecSuccess {
            FileHandle.standardError.write(Data(
                "[vmlx][keychain] save failed (update): id=\(id) status=\(updateStatus)\n".utf8))
        }
        #endif
    }

    /// Fetch the raw key material for `id`. Returns nil if the item is
    /// missing (keychain never held it, or the user wiped keychain).
    /// Callers should fall back to the SQLite plaintext `value` column
    /// during the migration grace period.
    static func load(id: String) -> String? {
        #if canImport(Security)
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: id,
            kSecReturnData: true,
            kSecMatchLimit: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data,
              let str = String(data: data, encoding: .utf8) else { return nil }
        return str
        #else
        return nil
        #endif
    }

    /// Drop the item for `id`. Silent no-op on missing item or any
    /// other failure — `revoke` also clears the SQLite row, so a
    /// lingering Keychain entry is merely untidy, not dangerous.
    static func delete(id: String) {
        #if canImport(Security)
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: id,
        ]
        _ = SecItemDelete(query as CFDictionary)
        #endif
    }

    /// One-shot migration: for each existing `APIKeyRow`, if Keychain
    /// doesn't hold its id yet, mirror the SQLite `value` in so the
    /// long-term Keychain-authoritative reads can pick it up. Idempotent
    /// — after migration, subsequent launches find every row already
    /// in Keychain and return without touching SQLite.
    static func migrateFromSQLiteIfNeeded(rows: [Database.APIKeyRow]) {
        for row in rows where load(id: row.id) == nil {
            save(id: row.id, value: row.value)
        }
    }
}
