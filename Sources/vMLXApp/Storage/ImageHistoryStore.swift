// SPDX-License-Identifier: Apache-2.0
//
// Image generation history store. A tiny, standalone SQLite wrapper (mirrors
// the pattern used by Database.swift — pure libsqlite3, no third-party
// deps) that persists every image generation / edit so the Image screen's
// history sidebar can re-load past prompts + settings.
//
// Unlike Database, this store accepts a custom path at construction time
// so unit tests can hit an isolated temp DB. In the app we instantiate one
// shared instance keyed on the default app-support directory.
//
// Schema:
//   image_generations (
//       id                TEXT PRIMARY KEY,
//       model_alias       TEXT NOT NULL,
//       prompt            TEXT NOT NULL,
//       source_image_path TEXT,
//       mask_path         TEXT,
//       settings_json     TEXT NOT NULL,
//       output_path       TEXT,
//       created_at        REAL NOT NULL,
//       duration_ms       INTEGER,
//       status            TEXT NOT NULL   -- pending|completed|failed|cancelled
//   )

import Foundation
import SQLite3

/// Persisted image generation record.
public struct ImageGenerationRecord: Identifiable, Hashable, Codable {
    public enum Status: String, Codable, Sendable {
        case pending, completed, failed, cancelled
    }
    public var id: UUID
    public var modelAlias: String
    public var prompt: String
    public var sourceImagePath: String?
    public var maskPath: String?
    public var settingsJSON: String
    public var outputPath: String?
    public var createdAt: Date
    public var durationMs: Int?
    public var status: Status

    public init(
        id: UUID = UUID(),
        modelAlias: String,
        prompt: String,
        sourceImagePath: String? = nil,
        maskPath: String? = nil,
        settingsJSON: String,
        outputPath: String? = nil,
        createdAt: Date = .now,
        durationMs: Int? = nil,
        status: Status = .pending
    ) {
        self.id = id
        self.modelAlias = modelAlias
        self.prompt = prompt
        self.sourceImagePath = sourceImagePath
        self.maskPath = maskPath
        self.settingsJSON = settingsJSON
        self.outputPath = outputPath
        self.createdAt = createdAt
        self.durationMs = durationMs
        self.status = status
    }
}

/// SQLite-backed image history. Not @MainActor isolated — the UI calls it
/// from the main thread anyway, but tests can drive it from any thread.
final class ImageHistoryStore {

    static let shared = ImageHistoryStore()

    private var db: OpaquePointer?

    /// Production initializer — opens at the default app-support location.
    init() {
        let fm = FileManager.default
        let appSup = try? fm.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = (appSup ?? URL(fileURLWithPath: NSTemporaryDirectory()))
            .appendingPathComponent("vMLX", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appendingPathComponent("image_history.sqlite3").path
        openAndMigrate(path: path)
    }

    /// Test initializer — opens at the caller-supplied URL so unit tests
    /// can round-trip against a temp DB without touching the real store.
    init(customPath: URL) {
        try? FileManager.default.createDirectory(
            at: customPath.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        openAndMigrate(path: customPath.path)
    }

    deinit {
        if db != nil { sqlite3_close(db) }
    }

    private func openAndMigrate(path: String) {
        if sqlite3_open(path, &db) != SQLITE_OK {
            NSLog("vMLX: image_history sqlite3_open failed at \(path)")
        }
        runSQL("PRAGMA journal_mode=WAL;")
        runSQL("PRAGMA foreign_keys=ON;")
        runSQL("""
        CREATE TABLE IF NOT EXISTS image_generations (
            id TEXT PRIMARY KEY,
            model_alias TEXT NOT NULL,
            prompt TEXT NOT NULL,
            source_image_path TEXT,
            mask_path TEXT,
            settings_json TEXT NOT NULL,
            output_path TEXT,
            created_at REAL NOT NULL,
            duration_ms INTEGER,
            status TEXT NOT NULL
        );
        """)
        runSQL("CREATE INDEX IF NOT EXISTS ix_image_gen_created ON image_generations(created_at DESC);")
    }

    private func runSQL(_ sql: String) {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "?"
            NSLog("vMLX image_history sqlite failed: \(msg)")
            sqlite3_free(err)
        }
    }

    // MARK: - CRUD

    @discardableResult
    func upsert(_ r: ImageGenerationRecord) -> Bool {
        let sql = """
        INSERT INTO image_generations (
            id, model_alias, prompt, source_image_path, mask_path,
            settings_json, output_path, created_at, duration_ms, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            prompt=excluded.prompt,
            output_path=excluded.output_path,
            duration_ms=excluded.duration_ms,
            status=excluded.status,
            settings_json=excluded.settings_json,
            source_image_path=excluded.source_image_path,
            mask_path=excluded.mask_path;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return false }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, r.id.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, r.modelAlias, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 3, r.prompt, -1, SQLITE_TRANSIENT)
        bindOptText(stmt, 4, r.sourceImagePath)
        bindOptText(stmt, 5, r.maskPath)
        sqlite3_bind_text(stmt, 6, r.settingsJSON, -1, SQLITE_TRANSIENT)
        bindOptText(stmt, 7, r.outputPath)
        sqlite3_bind_double(stmt, 8, r.createdAt.timeIntervalSince1970)
        if let d = r.durationMs {
            sqlite3_bind_int(stmt, 9, Int32(d))
        } else {
            sqlite3_bind_null(stmt, 9)
        }
        sqlite3_bind_text(stmt, 10, r.status.rawValue, -1, SQLITE_TRANSIENT)
        return sqlite3_step(stmt) == SQLITE_DONE
    }

    func all(limit: Int = 500) -> [ImageGenerationRecord] {
        var results: [ImageGenerationRecord] = []
        let sql = """
        SELECT id, model_alias, prompt, source_image_path, mask_path,
               settings_json, output_path, created_at, duration_ms, status
        FROM image_generations
        ORDER BY created_at DESC
        LIMIT ?;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(limit))
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = UUID(uuidString: cstr(stmt, 0)) ?? UUID()
            let alias = cstr(stmt, 1)
            let prompt = cstr(stmt, 2)
            let src = optCstr(stmt, 3)
            let mask = optCstr(stmt, 4)
            let settings = cstr(stmt, 5)
            let out = optCstr(stmt, 6)
            let ts = sqlite3_column_double(stmt, 7)
            let dur: Int? = sqlite3_column_type(stmt, 8) == SQLITE_NULL
                ? nil : Int(sqlite3_column_int(stmt, 8))
            let statusRaw = cstr(stmt, 9)
            results.append(ImageGenerationRecord(
                id: id,
                modelAlias: alias,
                prompt: prompt,
                sourceImagePath: src,
                maskPath: mask,
                settingsJSON: settings,
                outputPath: out,
                createdAt: Date(timeIntervalSince1970: ts),
                durationMs: dur,
                status: ImageGenerationRecord.Status(rawValue: statusRaw) ?? .completed
            ))
        }
        return results
    }

    @discardableResult
    func delete(_ id: UUID) -> Bool {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(
            db, "DELETE FROM image_generations WHERE id=?", -1, &stmt, nil
        ) == SQLITE_OK else { return false }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id.uuidString, -1, SQLITE_TRANSIENT)
        return sqlite3_step(stmt) == SQLITE_DONE
    }

    // MARK: - helpers

    private func bindOptText(_ stmt: OpaquePointer?, _ col: Int32, _ value: String?) {
        if let v = value {
            sqlite3_bind_text(stmt, col, v, -1, SQLITE_TRANSIENT)
        } else {
            sqlite3_bind_null(stmt, col)
        }
    }

    private func cstr(_ stmt: OpaquePointer?, _ col: Int32) -> String {
        guard let p = sqlite3_column_text(stmt, col) else { return "" }
        return String(cString: p)
    }

    private func optCstr(_ stmt: OpaquePointer?, _ col: Int32) -> String? {
        if sqlite3_column_type(stmt, col) == SQLITE_NULL { return nil }
        return cstr(stmt, col)
    }
}

// Re-declare SQLITE_TRANSIENT inside this file so it doesn't depend on
// Database.swift's private copy.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
