// SPDX-License-Identifier: Apache-2.0
//
// Raw sqlite3 persistence for the SettingsStore. Lives in its own file
// (~/Library/Application Support/vMLX/settings.sqlite3) so it can migrate
// independently of the chat DB and the model library DB.
//
// Schema (v1):
//   global_settings (id TEXT PRIMARY KEY, settings_json TEXT, updated_at REAL)
//   session_settings (id TEXT PRIMARY KEY, settings_json TEXT, updated_at REAL)
//   chat_settings   (id TEXT PRIMARY KEY, settings_json TEXT, updated_at REAL)

import Foundation
import SQLite3

public final class SettingsDB: @unchecked Sendable {

    private var db: OpaquePointer?
    private let path: String

    public init(customPath: URL? = nil) {
        let fm = FileManager.default
        let url: URL
        if let customPath {
            url = customPath
        } else {
            let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                     appropriateFor: nil, create: true)
            let dir = (appSup ?? URL(fileURLWithPath: NSTemporaryDirectory()))
                .appendingPathComponent("vMLX", isDirectory: true)
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            url = dir.appendingPathComponent("settings.sqlite3")
        }
        self.path = url.path

        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        if sqlite3_open_v2(path, &db, flags, nil) != SQLITE_OK {
            NSLog("vMLX: SettingsDB sqlite3_open failed at \(path)")
        }
        runSQL("PRAGMA journal_mode=WAL;")
        runSQL("PRAGMA synchronous=NORMAL;")
        runSQL("PRAGMA foreign_keys=OFF;")
        runMigrations()
    }

    deinit {
        if db != nil { sqlite3_close(db) }
    }

    public var storagePath: String { path }

    // MARK: - Migrations

    /// Sequential migrations. Index 0 -> v1, index 1 -> v2, etc.
    private lazy var migrations: [(OpaquePointer?) throws -> Void] = [
        { db in
            func execOne(_ sql: String) throws {
                var err: UnsafeMutablePointer<CChar>?
                if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
                    let msg = err.map { String(cString: $0) } ?? "?"
                    sqlite3_free(err)
                    throw SettingsDBError.migration("v1: \(msg)")
                }
            }
            try execOne("""
            CREATE TABLE IF NOT EXISTS global_settings (
                id TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """)
            try execOne("""
            CREATE TABLE IF NOT EXISTS session_settings (
                id TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """)
            try execOne("""
            CREATE TABLE IF NOT EXISTS chat_settings (
                id TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """)
        },
    ]

    public var latestVersion: Int { migrations.count }

    private func runMigrations() {
        let current = userVersion()
        guard current < latestVersion else { return }
        for v in current..<latestVersion {
            do {
                try migrations[v](db)
                setUserVersion(v + 1)
            } catch {
                NSLog("vMLX: SettingsDB migration to v\(v + 1) failed: \(error)")
                return
            }
        }
    }

    private func userVersion() -> Int {
        var stmt: OpaquePointer?
        var v = 0
        if sqlite3_prepare_v2(db, "PRAGMA user_version;", -1, &stmt, nil) == SQLITE_OK {
            if sqlite3_step(stmt) == SQLITE_ROW {
                v = Int(sqlite3_column_int(stmt, 0))
            }
        }
        sqlite3_finalize(stmt)
        return v
    }

    private func setUserVersion(_ v: Int) {
        runSQL("PRAGMA user_version=\(v);")
    }

    public func currentUserVersion() -> Int { userVersion() }

    // MARK: - CRUD

    public func getGlobal() -> Data? {
        fetchJSON(table: "global_settings", id: GlobalSettings.singletonID)
    }

    public func setGlobal(_ data: Data) {
        writeJSON(table: "global_settings", id: GlobalSettings.singletonID, data: data)
    }

    public func getSession(_ id: UUID) -> Data? {
        fetchJSON(table: "session_settings", id: id.uuidString)
    }

    public func setSession(_ id: UUID, _ data: Data) {
        writeJSON(table: "session_settings", id: id.uuidString, data: data)
    }

    public func deleteSession(_ id: UUID) {
        deleteRow(table: "session_settings", id: id.uuidString)
    }

    public func getChat(_ id: UUID) -> Data? {
        fetchJSON(table: "chat_settings", id: id.uuidString)
    }

    public func setChat(_ id: UUID, _ data: Data) {
        writeJSON(table: "chat_settings", id: id.uuidString, data: data)
    }

    public func deleteChat(_ id: UUID) {
        deleteRow(table: "chat_settings", id: id.uuidString)
    }

    public func allSessionIDs() -> [UUID] {
        var out: [UUID] = []
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT id FROM session_settings;", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let p = sqlite3_column_text(stmt, 0),
                   let u = UUID(uuidString: String(cString: p)) {
                    out.append(u)
                }
            }
        }
        sqlite3_finalize(stmt)
        return out
    }

    public func allChatIDs() -> [UUID] {
        var out: [UUID] = []
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT id FROM chat_settings;", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let p = sqlite3_column_text(stmt, 0),
                   let u = UUID(uuidString: String(cString: p)) {
                    out.append(u)
                }
            }
        }
        sqlite3_finalize(stmt)
        return out
    }

    // MARK: - Helpers

    private func fetchJSON(table: String, id: String) -> Data? {
        let sql = "SELECT settings_json FROM \(table) WHERE id=?;"
        var stmt: OpaquePointer?
        var out: Data?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT)
            if sqlite3_step(stmt) == SQLITE_ROW,
               let p = sqlite3_column_text(stmt, 0) {
                out = Data(String(cString: p).utf8)
            }
        }
        sqlite3_finalize(stmt)
        return out
    }

    private func writeJSON(table: String, id: String, data: Data) {
        let sql = """
        INSERT INTO \(table) (id, settings_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            settings_json=excluded.settings_json,
            updated_at=excluded.updated_at;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT)
        let str = String(data: data, encoding: .utf8) ?? "{}"
        sqlite3_bind_text(stmt, 2, str, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(stmt, 3, Date().timeIntervalSince1970)
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }

    private func deleteRow(table: String, id: String) {
        let sql = "DELETE FROM \(table) WHERE id=?;"
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    private func runSQL(_ sql: String) {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "?"
            NSLog("vMLX SettingsDB runSQL failed: \(msg)")
            sqlite3_free(err)
        }
    }
}

public enum SettingsDBError: Error, CustomStringConvertible {
    case migration(String)
    public var description: String {
        switch self {
        case .migration(let s): return "SettingsDB migration failed: \(s)"
        }
    }
}

extension GlobalSettings {
    public static let singletonID = "global"
}

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
