import Foundation
import SQLite3

/// Persistent store for `ModelLibrary`. Backed by its own SQLite file at
/// `~/Library/Application Support/vMLX/models.sqlite3` — deliberately split
/// from the chat/sessions DB (`vmlx.sqlite3`) so schema migrations evolve
/// independently and a corrupt index here can't nuke chat history.
///
/// Thread-safety: the class itself is not actor-isolated, but all callers
/// currently funnel through the `ModelLibrary` actor so access is serialized.
/// Connection is opened with `SQLITE_OPEN_FULLMUTEX` for defence in depth.
public final class ModelLibraryDB: @unchecked Sendable {

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
            url = dir.appendingPathComponent("models.sqlite3")
        }
        self.path = url.path

        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        if sqlite3_open_v2(path, &db, flags, nil) != SQLITE_OK {
            NSLog("vMLX: ModelLibraryDB sqlite3_open failed at \(path)")
        }
        runSQL("PRAGMA journal_mode=WAL;")
        runSQL("PRAGMA synchronous=NORMAL;")
        migrate()
    }

    deinit {
        if db != nil { sqlite3_close(db) }
    }

    // MARK: - Migration

    private func migrate() {
        let v = userVersion()
        if v < 1 {
            runSQL("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                canonical_path TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                family TEXT NOT NULL,
                modality TEXT NOT NULL,
                total_size_bytes INTEGER NOT NULL,
                is_jang INTEGER NOT NULL,
                is_mxtq INTEGER NOT NULL,
                quant_bits INTEGER,
                detected_at REAL NOT NULL,
                source TEXT NOT NULL
            );
            """)
            runSQL("CREATE INDEX IF NOT EXISTS idx_models_family ON models(family);")
            runSQL("CREATE INDEX IF NOT EXISTS idx_models_modality ON models(modality);")
            runSQL("""
            CREATE TABLE IF NOT EXISTS user_dirs (
                url TEXT PRIMARY KEY,
                added_at REAL NOT NULL
            );
            """)
            runSQL("PRAGMA user_version=1;")
        }
        if v < 2 {
            // Add capabilities_json column. Existing rows get '{}' which
            // triggers re-detection on the next scan (empty JSON decodes
            // to `.unknown`, which the library diff layer then overwrites).
            runSQL("ALTER TABLE models ADD COLUMN capabilities_json TEXT NOT NULL DEFAULT '{}';")
            runSQL("PRAGMA user_version=2;")
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

    private func runSQL(_ sql: String) {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "?"
            NSLog("vMLX ModelLibraryDB runSQL failed: \(msg)")
            sqlite3_free(err)
        }
    }

    // MARK: - Models CRUD

    public func upsert(_ e: ModelLibrary.ModelEntry) {
        let sql = """
        INSERT INTO models (id, canonical_path, display_name, family, modality,
                            total_size_bytes, is_jang, is_mxtq, quant_bits,
                            detected_at, source, capabilities_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            canonical_path=excluded.canonical_path,
            display_name=excluded.display_name,
            family=excluded.family,
            modality=excluded.modality,
            total_size_bytes=excluded.total_size_bytes,
            is_jang=excluded.is_jang,
            is_mxtq=excluded.is_mxtq,
            quant_bits=excluded.quant_bits,
            detected_at=excluded.detected_at,
            source=excluded.source,
            capabilities_json=excluded.capabilities_json;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        sqlite3_bind_text(stmt, 1, e.id, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, e.canonicalPath.path, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 3, e.displayName, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 4, e.family, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 5, e.modality.rawValue, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(stmt, 6, e.totalSizeBytes)
        sqlite3_bind_int(stmt, 7, e.isJANG ? 1 : 0)
        sqlite3_bind_int(stmt, 8, e.isMXTQ ? 1 : 0)
        if let q = e.quantBits {
            sqlite3_bind_int(stmt, 9, Int32(q))
        } else {
            sqlite3_bind_null(stmt, 9)
        }
        sqlite3_bind_double(stmt, 10, e.detectedAt.timeIntervalSince1970)
        sqlite3_bind_text(stmt, 11, encodeSource(e.source), -1, SQLITE_TRANSIENT)
        let capsJSON = encodeCapabilities(e.capabilities)
        sqlite3_bind_text(stmt, 12, capsJSON, -1, SQLITE_TRANSIENT)
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }

    private func encodeCapabilities(_ c: ModelCapabilities) -> String {
        guard let data = try? JSONEncoder().encode(c),
              let s = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return s
    }

    private func decodeCapabilities(_ s: String) -> ModelCapabilities {
        guard !s.isEmpty, s != "{}",
              let data = s.data(using: .utf8),
              let caps = try? JSONDecoder().decode(ModelCapabilities.self, from: data) else {
            return .unknown
        }
        return caps
    }

    public func all() -> [ModelLibrary.ModelEntry] {
        var out: [ModelLibrary.ModelEntry] = []
        let sql = """
        SELECT id, canonical_path, display_name, family, modality,
               total_size_bytes, is_jang, is_mxtq, quant_bits, detected_at, source,
               capabilities_json
        FROM models ORDER BY display_name ASC;
        """
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let e = rowToEntry(stmt) { out.append(e) }
            }
        }
        sqlite3_finalize(stmt)
        return out
    }

    public func byId(_ id: String) -> ModelLibrary.ModelEntry? {
        let sql = """
        SELECT id, canonical_path, display_name, family, modality,
               total_size_bytes, is_jang, is_mxtq, quant_bits, detected_at, source,
               capabilities_json
        FROM models WHERE id=? LIMIT 1;
        """
        var stmt: OpaquePointer?
        var out: ModelLibrary.ModelEntry?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT)
            if sqlite3_step(stmt) == SQLITE_ROW {
                out = rowToEntry(stmt)
            }
        }
        sqlite3_finalize(stmt)
        return out
    }

    public func purge(_ ids: Set<String>) {
        guard !ids.isEmpty else { return }
        runSQL("BEGIN;")
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "DELETE FROM models WHERE id=?;", -1, &stmt, nil) == SQLITE_OK {
            for id in ids {
                sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT)
                sqlite3_step(stmt)
                sqlite3_reset(stmt)
            }
        }
        sqlite3_finalize(stmt)
        runSQL("COMMIT;")
    }

    /// Most-recent `detected_at` (unix seconds) across all entries, or nil if empty.
    public func mostRecentDetectedAt() -> Date? {
        var stmt: OpaquePointer?
        var out: Date?
        if sqlite3_prepare_v2(db, "SELECT MAX(detected_at) FROM models;", -1, &stmt, nil) == SQLITE_OK {
            if sqlite3_step(stmt) == SQLITE_ROW,
               sqlite3_column_type(stmt, 0) != SQLITE_NULL {
                out = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 0))
            }
        }
        sqlite3_finalize(stmt)
        return out
    }

    // MARK: - User dirs

    public func userDirs() -> [URL] {
        var out: [URL] = []
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT url FROM user_dirs ORDER BY added_at ASC;",
                              -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let p = sqlite3_column_text(stmt, 0) {
                    out.append(URL(fileURLWithPath: String(cString: p)))
                }
            }
        }
        sqlite3_finalize(stmt)
        return out
    }

    public func addUserDir(_ url: URL) {
        var stmt: OpaquePointer?
        let sql = "INSERT OR IGNORE INTO user_dirs (url, added_at) VALUES (?, ?);"
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, url.path, -1, SQLITE_TRANSIENT)
            sqlite3_bind_double(stmt, 2, Date().timeIntervalSince1970)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    public func removeUserDir(_ url: URL) {
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "DELETE FROM user_dirs WHERE url=?;", -1, &stmt, nil) == SQLITE_OK {
            sqlite3_bind_text(stmt, 1, url.path, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    // MARK: - Row decoding

    private func rowToEntry(_ stmt: OpaquePointer?) -> ModelLibrary.ModelEntry? {
        guard let stmt else { return nil }
        let id = cstr(stmt, 0)
        let path = cstr(stmt, 1)
        let name = cstr(stmt, 2)
        let family = cstr(stmt, 3)
        let modalityRaw = cstr(stmt, 4)
        let size = sqlite3_column_int64(stmt, 5)
        let isJang = sqlite3_column_int(stmt, 6) != 0
        let isMxtq = sqlite3_column_int(stmt, 7) != 0
        let qb: Int? = sqlite3_column_type(stmt, 8) == SQLITE_NULL
            ? nil : Int(sqlite3_column_int(stmt, 8))
        let ts = sqlite3_column_double(stmt, 9)
        let sourceRaw = cstr(stmt, 10)
        let capsRaw = cstr(stmt, 11)
        let modality = ModelLibrary.Modality(rawValue: modalityRaw) ?? .unknown
        return ModelLibrary.ModelEntry(
            id: id,
            canonicalPath: URL(fileURLWithPath: path),
            displayName: name,
            family: family,
            modality: modality,
            totalSizeBytes: size,
            isJANG: isJang,
            isMXTQ: isMxtq,
            quantBits: qb,
            detectedAt: Date(timeIntervalSince1970: ts),
            source: decodeSource(sourceRaw),
            capabilities: decodeCapabilities(capsRaw)
        )
    }

    private func cstr(_ stmt: OpaquePointer?, _ col: Int32) -> String {
        guard let p = sqlite3_column_text(stmt, col) else { return "" }
        return String(cString: p)
    }

    private func encodeSource(_ s: ModelLibrary.Source) -> String {
        switch s {
        case .hfCache: return "hf"
        case .downloaded: return "dl"
        case .userDir(let u): return "user:\(u.path)"
        }
    }

    private func decodeSource(_ s: String) -> ModelLibrary.Source {
        if s == "hf" { return .hfCache }
        if s == "dl" { return .downloaded }
        if s.hasPrefix("user:") {
            return .userDir(URL(fileURLWithPath: String(s.dropFirst(5))))
        }
        return .hfCache
    }
}

// SQLITE_TRANSIENT is not bridged.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
