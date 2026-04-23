// Copyright © 2026 Jinho Jang. All rights reserved.
//
// §355 — Block-level disk cache for paged KV blocks.

import CryptoKit
import Foundation
import SQLite3
import os

public final class BlockDiskCache: @unchecked Sendable {

    public let cacheDir: URL
    public let maxSizeBytes: Int
    public let modelKey: String
    private let blocksDir: URL

    private var db: OpaquePointer?
    private let lock = OSAllocatedUnfairLock()

    public private(set) var hits: Int = 0
    public private(set) var misses: Int = 0
    public private(set) var stores: Int = 0
    public private(set) var evictions: Int = 0

    private let ioQueue = DispatchQueue(
        label: "net.jangqai.vmlx.blockdiskcache.io",
        qos: .utility)

    public init(
        cacheDir: URL,
        maxSizeGB: Double = 10.0,
        modelKey: String = "default"
    ) throws {
        self.cacheDir = cacheDir
        self.maxSizeBytes = Int(maxSizeGB * 1_073_741_824)
        self.modelKey = modelKey
        self.blocksDir = cacheDir.appendingPathComponent("blocks", isDirectory: true)

        try FileManager.default.createDirectory(
            at: blocksDir, withIntermediateDirectories: true)

        let dbPath = cacheDir.appendingPathComponent("block_index.db").path
        guard sqlite3_open(dbPath, &db) == SQLITE_OK else {
            throw BlockDiskCacheError.openFailed(dbPath)
        }
        try createSchemaIfNeeded()
    }

    deinit {
        if let db { sqlite3_close(db) }
    }

    private func createSchemaIfNeeded() throws {
        let ddl = """
        CREATE TABLE IF NOT EXISTS blocks (
            hash         TEXT    NOT NULL,
            model_key    TEXT    NOT NULL,
            token_count  INTEGER NOT NULL,
            byte_size    INTEGER NOT NULL,
            last_access  REAL    NOT NULL,
            created_at   REAL    NOT NULL,
            PRIMARY KEY (hash, model_key)
        );
        CREATE INDEX IF NOT EXISTS idx_blocks_lru
            ON blocks(model_key, last_access);
        """
        var err: UnsafeMutablePointer<CChar>? = nil
        guard sqlite3_exec(db, ddl, nil, nil, &err) == SQLITE_OK else {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(err)
            throw BlockDiskCacheError.schemaFailed(msg)
        }
    }

    public static func blockHash(
        tokenIDs: [Int], modelKey: String
    ) -> String {
        var hasher = SHA256()
        var bytes = [UInt8]()
        bytes.reserveCapacity(tokenIDs.count * 4 + modelKey.utf8.count + 1)
        for tid in tokenIDs {
            let v = UInt32(truncatingIfNeeded: tid)
            bytes.append(UInt8(v & 0xff))
            bytes.append(UInt8((v >> 8) & 0xff))
            bytes.append(UInt8((v >> 16) & 0xff))
            bytes.append(UInt8((v >> 24) & 0xff))
        }
        bytes.append(contentsOf: modelKey.utf8)
        bytes.append(0)
        hasher.update(data: Data(bytes))
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    public func fetch(blockHash: String) -> Data? {
        lock.lock(); defer { lock.unlock() }
        guard let db else { return nil }
        var stmt: OpaquePointer?
        let sql = "SELECT byte_size FROM blocks WHERE hash = ? AND model_key = ?"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            misses += 1
            return nil
        }
        defer { sqlite3_finalize(stmt) }
        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, blockHash, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 2, modelKey, -1, TRANSIENT)
        guard sqlite3_step(stmt) == SQLITE_ROW else {
            misses += 1
            return nil
        }

        let url = shardURL(for: blockHash)
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            misses += 1
            _ = deleteIndexRow(blockHash: blockHash)
            return nil
        }
        hits += 1
        touchAccessTime(blockHash: blockHash)
        return data
    }

    public func store(
        blockHash: String, tokenCount: Int, payload: Data
    ) {
        let url = shardURL(for: blockHash)
        let bytes = payload.count
        let now = Date().timeIntervalSince1970
        let key = modelKey

        ioQueue.async { [weak self] in
            guard let self else { return }

            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)

            do {
                try payload.write(to: url, options: .atomic)
            } catch {
                FileHandle.standardError.write(
                    "[BlockDiskCache] write failed \(url.path): \(error)\n"
                        .data(using: .utf8)!)
                return
            }

            self.lock.lock()
            defer { self.lock.unlock() }
            guard let db = self.db else { return }
            let sql = """
            INSERT OR REPLACE INTO blocks
                (hash, model_key, token_count, byte_size, last_access, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            sqlite3_bind_text(stmt, 1, blockHash, -1, TRANSIENT)
            sqlite3_bind_text(stmt, 2, key, -1, TRANSIENT)
            sqlite3_bind_int64(stmt, 3, Int64(tokenCount))
            sqlite3_bind_int64(stmt, 4, Int64(bytes))
            sqlite3_bind_double(stmt, 5, now)
            sqlite3_bind_double(stmt, 6, now)
            _ = sqlite3_step(stmt)
            self.stores += 1

            self.evictIfOverBudgetLocked()
        }
    }

    private func evictIfOverBudgetLocked() {
        guard let db else { return }
        let total = totalSizeBytesLocked()
        guard total > maxSizeBytes else { return }

        var toDelete: [(hash: String, bytes: Int)] = []
        var runningTotal = total
        let sql = """
        SELECT hash, byte_size FROM blocks
        WHERE model_key = ?
        ORDER BY last_access ASC
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        defer { sqlite3_finalize(stmt) }
        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, modelKey, -1, TRANSIENT)
        while sqlite3_step(stmt) == SQLITE_ROW && runningTotal > maxSizeBytes {
            guard let cstr = sqlite3_column_text(stmt, 0) else { continue }
            let hash = String(cString: cstr)
            let bytes = Int(sqlite3_column_int64(stmt, 1))
            toDelete.append((hash, bytes))
            runningTotal -= bytes
        }

        for entry in toDelete {
            let url = shardURL(for: entry.hash)
            try? FileManager.default.removeItem(at: url)
            _ = deleteIndexRow(blockHash: entry.hash)
            evictions += 1
        }
    }

    private func totalSizeBytesLocked() -> Int {
        guard let db else { return 0 }
        var stmt: OpaquePointer?
        let sql = "SELECT COALESCE(SUM(byte_size), 0) FROM blocks WHERE model_key = ?"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return 0 }
        defer { sqlite3_finalize(stmt) }
        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, modelKey, -1, TRANSIENT)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(stmt, 0))
    }

    private func deleteIndexRow(blockHash: String) -> Bool {
        guard let db else { return false }
        var stmt: OpaquePointer?
        let sql = "DELETE FROM blocks WHERE hash = ? AND model_key = ?"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return false }
        defer { sqlite3_finalize(stmt) }
        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, blockHash, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 2, modelKey, -1, TRANSIENT)
        return sqlite3_step(stmt) == SQLITE_DONE
    }

    private func touchAccessTime(blockHash: String) {
        let key = modelKey
        let now = Date().timeIntervalSince1970
        ioQueue.async { [weak self] in
            guard let self else { return }
            self.lock.lock()
            defer { self.lock.unlock() }
            guard let db = self.db else { return }
            var stmt: OpaquePointer?
            let sql = "UPDATE blocks SET last_access = ? WHERE hash = ? AND model_key = ?"
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            sqlite3_bind_double(stmt, 1, now)
            sqlite3_bind_text(stmt, 2, blockHash, -1, TRANSIENT)
            sqlite3_bind_text(stmt, 3, key, -1, TRANSIENT)
            _ = sqlite3_step(stmt)
        }
    }

    public struct Stats: Sendable {
        public let entryCount: Int
        public let totalBytes: Int
        public let hits: Int
        public let misses: Int
        public let stores: Int
        public let evictions: Int
        public let hitRate: Double
        public let maxBytes: Int
    }

    public func snapshot() -> Stats {
        lock.lock(); defer { lock.unlock() }
        var entryCount = 0
        if let db {
            var stmt: OpaquePointer?
            let sql = "SELECT COUNT(*) FROM blocks WHERE model_key = ?"
            if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
                let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
                sqlite3_bind_text(stmt, 1, modelKey, -1, TRANSIENT)
                if sqlite3_step(stmt) == SQLITE_ROW {
                    entryCount = Int(sqlite3_column_int64(stmt, 0))
                }
                sqlite3_finalize(stmt)
            }
        }
        let totalBytes = totalSizeBytesLocked()
        let totalLookups = hits + misses
        let rate = totalLookups > 0 ? Double(hits) / Double(totalLookups) : 0
        return Stats(
            entryCount: entryCount,
            totalBytes: totalBytes,
            hits: hits,
            misses: misses,
            stores: stores,
            evictions: evictions,
            hitRate: rate,
            maxBytes: maxSizeBytes)
    }

    private func shardURL(for blockHash: String) -> URL {
        let prefix = String(blockHash.prefix(2))
        return blocksDir
            .appendingPathComponent(prefix, isDirectory: true)
            .appendingPathComponent("\(blockHash).safetensors")
    }
}

public enum BlockDiskCacheError: Error, CustomStringConvertible {
    case openFailed(String)
    case schemaFailed(String)

    public var description: String {
        switch self {
        case .openFailed(let p): return "BlockDiskCache: failed to open SQLite at \(p)"
        case .schemaFailed(let m): return "BlockDiskCache: schema setup failed — \(m)"
        }
    }
}
