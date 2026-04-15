import Foundation
import SQLite3

/// Minimal SQLite wrapper for sessions + messages. Uses the system `libsqlite3`
/// so we pick up zero third-party dependencies — important for App Store review
/// (fewer SBOM entries, no license audits, no transitive code). GRDB would have
/// been nicer ergonomically but adds a package and a larger binary surface, and
/// our query set here is tiny.
///
/// Storage location: `~/Library/Application Support/vMLX/vmlx.sqlite3`
/// WAL mode enabled to match the Electron app.
@MainActor
final class Database {
    static let shared = Database()

    private var db: OpaquePointer?

    private init() {
        open()
        migrate()
    }

    deinit {
        if db != nil { sqlite3_close(db) }
    }

    private func open() {
        let fm = FileManager.default
        let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                 appropriateFor: nil, create: true)
        let dir = (appSup ?? URL(fileURLWithPath: NSTemporaryDirectory()))
            .appendingPathComponent("vMLX", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appendingPathComponent("vmlx.sqlite3").path
        if sqlite3_open(path, &db) != SQLITE_OK {
            NSLog("vMLX: sqlite3_open failed at \(path)")
        }
        runSQL("PRAGMA journal_mode=WAL;")
        runSQL("PRAGMA foreign_keys=ON;")
    }

    private func migrate() {
        runSQL("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            model_path TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        """)
        runSQL("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            reasoning TEXT,
            tool_calls_json TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """)
        runSQL("CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, created_at);")
        runSQL("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_used_at REAL
        );
        """)

        // Schema version bump: add `is_streaming` column to messages so
        // we can recover from mid-stream force-quits. We check PRAGMA
        // user_version to decide whether to ALTER — SQLite doesn't
        // support `ADD COLUMN IF NOT EXISTS`.
        let version = currentUserVersion()
        if version < 1 {
            // ALTER may fail (column already exists from a prior test
            // run on an unbumped version) — ignore the failure, the
            // column-existence check below is cheap.
            runSQL("ALTER TABLE messages ADD COLUMN is_streaming INTEGER NOT NULL DEFAULT 0;")
            runSQL("PRAGMA user_version = 1;")
        }
    }

    private func currentUserVersion() -> Int {
        var stmt: OpaquePointer?
        var value: Int = 0
        if sqlite3_prepare_v2(db, "PRAGMA user_version;", -1, &stmt, nil) == SQLITE_OK {
            if sqlite3_step(stmt) == SQLITE_ROW {
                value = Int(sqlite3_column_int(stmt, 0))
            }
        }
        sqlite3_finalize(stmt)
        return value
    }

    /// Recover from force-quit mid-stream: any messages left with
    /// `is_streaming = 1` get flipped back to 0 and tagged with an
    /// ` [interrupted]` suffix so the user sees what happened. Call
    /// exactly once on app launch, BEFORE any session loads. Mirrors
    /// Electron's `sessions.ts::markInterrupted` on startup.
    func markAllStreamingAsInterrupted() {
        runSQL("""
        UPDATE messages
           SET is_streaming = 0,
               content = content || ' [interrupted]'
         WHERE is_streaming = 1;
        """)
    }

    // MARK: - API keys (used by APIKeyManager)

    struct APIKeyRow: Identifiable, Hashable, Sendable {
        let id: String
        var label: String
        var value: String
        var createdAt: Date
        var lastUsedAt: Date?
    }

    func allAPIKeys() -> [APIKeyRow] {
        var out: [APIKeyRow] = []
        var stmt: OpaquePointer?
        let sql = "SELECT id, label, value, created_at, last_used_at FROM api_keys ORDER BY created_at DESC"
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                let id = cstr(stmt, 0)
                let label = cstr(stmt, 1)
                let value = cstr(stmt, 2)
                let created = sqlite3_column_double(stmt, 3)
                let lastUsed: Date? = sqlite3_column_type(stmt, 4) == SQLITE_NULL
                    ? nil
                    : Date(timeIntervalSince1970: sqlite3_column_double(stmt, 4))
                out.append(APIKeyRow(id: id, label: label, value: value,
                                     createdAt: Date(timeIntervalSince1970: created),
                                     lastUsedAt: lastUsed))
            }
        }
        sqlite3_finalize(stmt)
        return out
    }

    func insertAPIKey(_ row: APIKeyRow) {
        let sql = """
        INSERT INTO api_keys (id, label, value, created_at, last_used_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            label=excluded.label,
            value=excluded.value,
            last_used_at=excluded.last_used_at;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        sqlite3_bind_text(stmt, 1, row.id, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, row.label, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 3, row.value, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(stmt, 4, row.createdAt.timeIntervalSince1970)
        if let lu = row.lastUsedAt {
            sqlite3_bind_double(stmt, 5, lu.timeIntervalSince1970)
        } else {
            sqlite3_bind_null(stmt, 5)
        }
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }

    func deleteAPIKey(id: String) {
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "DELETE FROM api_keys WHERE id=?", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    func touchAPIKey(id: String, at: Date = Date()) {
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "UPDATE api_keys SET last_used_at=? WHERE id=?", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_double(stmt, 1, at.timeIntervalSince1970)
            sqlite3_bind_text(stmt, 2, id, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    private func runSQL(_ sql: String) {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "?"
            NSLog("vMLX sqlite failed: \(msg)")
            sqlite3_free(err)
        }
    }

    // MARK: - Sessions

    func allSessions() -> [ChatSession] {
        var results: [ChatSession] = []
        let sql = "SELECT id, title, model_path, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                let id = UUID(uuidString: cstr(stmt, 0)) ?? UUID()
                let title = cstr(stmt, 1)
                let mp = sqlite3_column_type(stmt, 2) == SQLITE_NULL ? nil : cstr(stmt, 2)
                let c = sqlite3_column_double(stmt, 3)
                let u = sqlite3_column_double(stmt, 4)
                results.append(ChatSession(
                    id: id, title: title, modelPath: mp,
                    createdAt: Date(timeIntervalSince1970: c),
                    updatedAt: Date(timeIntervalSince1970: u)
                ))
            }
        }
        sqlite3_finalize(stmt)
        return results
    }

    func upsertSession(_ s: ChatSession) {
        let sql = """
        INSERT INTO sessions (id, title, model_path, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title=excluded.title,
            model_path=excluded.model_path,
            updated_at=excluded.updated_at;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        sqlite3_bind_text(stmt, 1, s.id.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, s.title, -1, SQLITE_TRANSIENT)
        if let mp = s.modelPath {
            sqlite3_bind_text(stmt, 3, mp, -1, SQLITE_TRANSIENT)
        } else {
            sqlite3_bind_null(stmt, 3)
        }
        sqlite3_bind_double(stmt, 4, s.createdAt.timeIntervalSince1970)
        sqlite3_bind_double(stmt, 5, s.updatedAt.timeIntervalSince1970)
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }

    func deleteSession(_ id: UUID) {
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "DELETE FROM sessions WHERE id=?", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, id.uuidString, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    // MARK: - Messages

    func messages(for sessionId: UUID) -> [ChatMessage] {
        var results: [ChatMessage] = []
        let sql = """
        SELECT id, session_id, role, content, reasoning, tool_calls_json, created_at, is_streaming
        FROM messages WHERE session_id=? ORDER BY created_at ASC
        """
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, sessionId.uuidString, -1, SQLITE_TRANSIENT)
            while sqlite3_step(stmt) == SQLITE_ROW {
                let id = UUID(uuidString: cstr(stmt, 0)) ?? UUID()
                let sid = UUID(uuidString: cstr(stmt, 1)) ?? sessionId
                let role = ChatMessage.Role(rawValue: cstr(stmt, 2)) ?? .user
                let content = cstr(stmt, 3)
                let reasoning = sqlite3_column_type(stmt, 4) == SQLITE_NULL ? nil : cstr(stmt, 4)
                let tc = sqlite3_column_type(stmt, 5) == SQLITE_NULL ? nil : cstr(stmt, 5)
                let ts = sqlite3_column_double(stmt, 6)
                let isStreaming = sqlite3_column_int(stmt, 7) != 0
                results.append(ChatMessage(
                    id: id, sessionId: sid, role: role, content: content,
                    reasoning: reasoning, toolCallsJSON: tc,
                    createdAt: Date(timeIntervalSince1970: ts),
                    isStreaming: isStreaming
                ))
            }
        }
        sqlite3_finalize(stmt)
        return results
    }

    func upsertMessage(_ m: ChatMessage) {
        let sql = """
        INSERT INTO messages (id, session_id, role, content, reasoning, tool_calls_json, created_at, is_streaming)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            content=excluded.content,
            reasoning=excluded.reasoning,
            tool_calls_json=excluded.tool_calls_json,
            is_streaming=excluded.is_streaming;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        sqlite3_bind_text(stmt, 1, m.id.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, m.sessionId.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 3, m.role.rawValue, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 4, m.content, -1, SQLITE_TRANSIENT)
        if let r = m.reasoning {
            sqlite3_bind_text(stmt, 5, r, -1, SQLITE_TRANSIENT)
        } else { sqlite3_bind_null(stmt, 5) }
        if let tc = m.toolCallsJSON {
            sqlite3_bind_text(stmt, 6, tc, -1, SQLITE_TRANSIENT)
        } else { sqlite3_bind_null(stmt, 6) }
        sqlite3_bind_double(stmt, 7, m.createdAt.timeIntervalSince1970)
        sqlite3_bind_int(stmt, 8, m.isStreaming ? 1 : 0)
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }

    func deleteMessage(_ id: UUID) {
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "DELETE FROM messages WHERE id=?", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, id.uuidString, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    func deleteMessages(after createdAt: Date, in sessionId: UUID) {
        let sql = "DELETE FROM messages WHERE session_id=? AND created_at>=?"
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, sessionId.uuidString, -1, SQLITE_TRANSIENT)
            sqlite3_bind_double(stmt, 2, createdAt.timeIntervalSince1970)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    // MARK: - helpers

    private func cstr(_ stmt: OpaquePointer?, _ col: Int32) -> String {
        guard let p = sqlite3_column_text(stmt, col) else { return "" }
        return String(cString: p)
    }
}

// SQLite `SQLITE_TRANSIENT` constant isn't bridged; re-declare here.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
