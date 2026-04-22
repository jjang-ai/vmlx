import Foundation

// MARK: - Debug bundle exporter (Q5 §301)
//
// `Engine.exportDebugBundle()` gathers non-sensitive diagnostics into a
// single zip archive that operators can attach to an issue report. It
// bundles:
//
//   • logs.ndjson           — full LogStore ring (info+) in NDJSON
//   • resolved-settings.json — current resolved settings + per-field
//                              tier trace (global/session/chat/request)
//   • cache-stats.json      — output of `cacheStats()` (paged + disk)
//   • engine-state.json     — {state, loaded model name, boot time}
//
// What we DO NOT include (PII / secrets policy):
//   • No chat messages, prompts, responses, reasoning, or tool calls.
//   • No API keys (bearer/admin). The keys card is explicit redaction.
//   • No file paths outside the bundle itself.
//   • No HuggingFace token. The keychain-backed blob never leaves.
//
// The zip is built by shelling out to `/usr/bin/zip` (present on every
// macOS install). Ships Developer ID-signed DMG; no third-party lib.

extension Engine {

    /// Build a debug-bundle zip and return the file URL. Caller is
    /// responsible for moving / attaching / deleting the file.
    /// `destination` defaults to `~/Downloads/vmlx-debug-<timestamp>.zip`
    /// — use a tmpdir + move for atomicity if the UI is handling it.
    public func exportDebugBundle(
        to destination: URL? = nil
    ) async throws -> URL {
        let fm = FileManager.default

        // Scratch dir under /tmp for staging the individual files
        // before we zip them. Using .itemReplacementDirectory returns
        // a fresh unique path, so we don't clobber a prior export.
        let staging = try fm.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: fm.homeDirectoryForCurrentUser,
            create: true
        )
        defer { try? fm.removeItem(at: staging) }

        // --- logs.ndjson -------------------------------------------------
        let logsData = await self.logs.export()
        try logsData.write(to: staging.appendingPathComponent("logs.ndjson"))

        // --- resolved-settings.json --------------------------------------
        // ResolvedSettings isn't Encodable (too many imported types +
        // legacy ObjC bridging on some fields). Mirror + reflect it
        // into a plain dict so JSONSerialization can handle it and we
        // don't drag Codable conformance onto every field.
        let resolved = await settings.resolved()
        let mirror = Mirror(reflecting: resolved)
        var resolvedDict: [String: Any] = [:]
        for child in mirror.children {
            guard let label = child.label else { continue }
            resolvedDict[label] = Self.jsonSafe(child.value)
        }
        let resolvedJSON = try JSONSerialization.data(
            withJSONObject: resolvedDict,
            options: [.prettyPrinted, .sortedKeys, .fragmentsAllowed]
        )
        try resolvedJSON.write(to: staging.appendingPathComponent("resolved-settings.json"))

        // --- cache-stats.json --------------------------------------------
        // cacheStats returns `[String: Any]`, so JSONSerialization instead
        // of JSONEncoder. The default encoder doesn't handle `[String: Any]`.
        let cacheBlob: [String: Any]
        do {
            cacheBlob = try await cacheStats()
        } catch {
            // Bundle still usable without it — attach the failure reason.
            cacheBlob = ["error": "\(error)"]
        }
        let cacheData = try JSONSerialization.data(
            withJSONObject: cacheBlob,
            options: [.prettyPrinted, .sortedKeys]
        )
        try cacheData.write(to: staging.appendingPathComponent("cache-stats.json"))

        // --- engine-state.json -------------------------------------------
        let stateBlob: [String: Any] = [
            "state": String(describing: state),
            "bundle_generated_at": ISO8601DateFormatter().string(from: Date()),
            "platform": "darwin",
            "host_arch": hostArch(),
        ]
        let stateData = try JSONSerialization.data(
            withJSONObject: stateBlob,
            options: [.prettyPrinted, .sortedKeys]
        )
        try stateData.write(to: staging.appendingPathComponent("engine-state.json"))

        // --- zip ---------------------------------------------------------
        let finalURL: URL = destination ?? {
            let ts = ISO8601DateFormatter.filename.string(from: Date())
            let downloads = fm.urls(for: .downloadsDirectory, in: .userDomainMask).first
                ?? fm.homeDirectoryForCurrentUser
            return downloads.appendingPathComponent("vmlx-debug-\(ts).zip")
        }()
        // Remove a pre-existing file at the destination — zip refuses
        // to overwrite silently, and the UI button promises a file.
        try? fm.removeItem(at: finalURL)

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        proc.arguments = ["-j", "-q", finalURL.path,
                          staging.appendingPathComponent("logs.ndjson").path,
                          staging.appendingPathComponent("resolved-settings.json").path,
                          staging.appendingPathComponent("cache-stats.json").path,
                          staging.appendingPathComponent("engine-state.json").path]
        let stderr = Pipe()
        proc.standardError = stderr
        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            let msg = String(data: stderr.fileHandleForReading.readDataToEndOfFile(),
                             encoding: .utf8) ?? ""
            throw EngineError.notImplemented("debug-bundle zip failed: \(msg)")
        }

        await self.log(.info, "engine",
                       "Debug bundle exported: \(finalURL.lastPathComponent)")
        return finalURL
    }

    /// Reduce an arbitrary `Any` value to something JSONSerialization
    /// accepts. Optionals unwrap; URLs/UUIDs/enums flatten to String;
    /// collections recurse; everything else falls back to `String(describing:)`.
    fileprivate static func jsonSafe(_ value: Any) -> Any {
        let m = Mirror(reflecting: value)
        // Optional
        if m.displayStyle == .optional {
            if let inner = m.children.first?.value {
                return jsonSafe(inner)
            }
            return NSNull()
        }
        switch value {
        case let v as String: return v
        case let v as Int: return v
        case let v as Int64: return v
        case let v as Double: return v
        case let v as Float: return Double(v)
        case let v as Bool: return v
        case let v as URL: return v.path
        case let v as UUID: return v.uuidString
        case let v as Date: return ISO8601DateFormatter().string(from: v)
        case let arr as [Any]: return arr.map { jsonSafe($0) }
        case let dict as [String: Any]:
            var out: [String: Any] = [:]
            for (k, v) in dict { out[k] = jsonSafe(v) }
            return out
        default:
            // Struct / enum with associated values — reflect one level.
            if m.displayStyle == .struct, !m.children.isEmpty {
                var out: [String: Any] = [:]
                for child in m.children {
                    guard let label = child.label else { continue }
                    out[label] = jsonSafe(child.value)
                }
                return out
            }
            return String(describing: value)
        }
    }

    private nonisolated func hostArch() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) {
                String(cString: $0)
            }
        }
        return machine
    }
}

private extension JSONEncoder {
    static let debugBundle: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()
}

private extension ISO8601DateFormatter {
    /// Filename-safe timestamp: `2026-04-22T00-42-17Z` — no colons so
    /// Finder/Safari don't mangle the download.
    static let filename: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withFullDate, .withTime, .withTimeZone]
        return f
    }()
}
