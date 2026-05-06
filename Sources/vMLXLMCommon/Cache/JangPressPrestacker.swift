// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressPrestacker
//
// Some JANGTQ MoE bundles store routed experts as thousands of separate
// `experts.N.w{1,2,3}.tq_*` tensors. The Swift model sanitizers used to
// combine those tensors with MLX.stacked(...), which materializes the whole
// routed expert bank into resident Metal buffers and defeats mmap-backed
// JangPress. This helper builds a low-RAM safetensors overlay where those
// per-expert tensors are stream-copied into the pre-stacked
// `switch_mlp.*.tq_*` layout that TurboQuantSwitchGLU already consumes.

import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public enum JangPressPrestacker {
    private static let version = "v3"
    private static let outputName = "jangpress-prestacked.safetensors"
    private static let hotCoreName = "jangpress-hot-core.safetensors"

    public static func prepareBundleIfNeeded(
        originalURL: URL,
        enabled: Bool
    ) throws -> URL {
        guard enabled else { return originalURL }

        let env = ProcessInfo.processInfo.environment
        if env["JANGPRESS_PRESTACK"] == "0" {
            return originalURL
        }

        let originalURL = originalURL.resolvingSymlinksInPath()
        let sidecar = originalURL.appendingPathComponent("jangtq_runtime.safetensors")
        guard FileManager.default.fileExists(atPath: sidecar.path) else {
            return originalURL
        }

        do {
            let scan = try scanBundle(originalURL)
            guard !scan.groups.isEmpty else { return originalURL }

            let cacheURL = try cacheDirectory(for: originalURL, files: scan.files)
            try ensureOverlayLinks(from: originalURL, to: cacheURL)
            let outputURL = cacheURL.appendingPathComponent(outputName)
            let hotCoreURL = cacheURL.appendingPathComponent(hotCoreName)
            let manifestURL = cacheURL.appendingPathComponent("jangpress-prestack-manifest.json")

            if FileManager.default.fileExists(atPath: outputURL.path),
               FileManager.default.fileExists(atPath: hotCoreURL.path),
               FileManager.default.fileExists(atPath: manifestURL.path)
            {
                log("using existing prestacked overlay \(cacheURL.path)")
                return cacheURL
            }

            try FileManager.default.createDirectory(
                at: cacheURL, withIntermediateDirectories: true)
            let plan = try buildWritePlan(from: scan.groups)
            guard !plan.isEmpty else { return originalURL }
            let hotPlan = buildHotCorePlan(from: scan.tensors, routedPlan: plan)
            guard !hotPlan.isEmpty else { return originalURL }
            try writeSafetensors(plan: plan, to: outputURL)
            try writeSafetensors(plan: hotPlan, to: hotCoreURL)
            try writeManifest(plan: plan, source: originalURL, to: manifestURL)
            let totalBytes = plan.reduce(UInt64(0)) { $0 + $1.totalBytes }
            let hotBytes = hotPlan.reduce(UInt64(0)) { $0 + $1.totalBytes }
            log(String(format:
                "wrote %d prestacked routed tensors (%.1f GB) and hot core %.1f GB into %@",
                plan.count, Double(totalBytes) / 1_073_741_824.0,
                Double(hotBytes) / 1_073_741_824.0,
                outputURL.path))
            return cacheURL
        } catch {
            if env["JANGPRESS_PRESTACK_STRICT"] == "1" {
                throw error
            }
            log("prestack failed, falling back to original bundle: \(error)")
            return originalURL
        }
    }

    private struct SourceFile {
        var url: URL
        var size: UInt64
        var modified: TimeInterval
    }

    private struct TensorSource {
        var key: String
        var expert: Int
        var fileURL: URL
        var byteOffset: UInt64
        var byteLength: UInt64
        var dtype: String
        var shape: [Int]
    }

    private struct TensorGroup {
        var outputKey: String
        var byExpert: [Int: TensorSource] = [:]
    }

    private struct BundleScan {
        var files: [SourceFile]
        var tensors: [TensorSource]
        var groups: [String: TensorGroup]
    }

    private struct WriteTensor {
        var key: String
        var dtype: String
        var shape: [Int]
        var sources: [TensorSource]
        var dataOffset: UInt64 = 0

        var totalBytes: UInt64 {
            sources.reduce(UInt64(0)) { $0 + $1.byteLength }
        }
    }

    private static func scanBundle(_ directory: URL) throws -> BundleScan {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: directory, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])
        else {
            return BundleScan(files: [], tensors: [], groups: [:])
        }

        var files: [SourceFile] = []
        var tensors: [TensorSource] = []
        var groups: [String: TensorGroup] = [:]
        var existingStackedKeys = Set<String>()
        var pending: [(String, TensorSource)] = []

        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors",
                  url.lastPathComponent != "jangtq_runtime.safetensors",
                  url.lastPathComponent != outputName,
                  url.lastPathComponent != hotCoreName,
                  !isDerivedStackedArtifact(url.lastPathComponent)
            else { continue }

            // Shadow bundles are symlink farms. Hash and read the resolved
            // safetensors target, not the symlink node, otherwise rebuilding
            // the shadow dir refreshes symlink mtimes and creates another
            // 100+ GB prestack cache for identical model bytes.
            let resolvedURL = url.resolvingSymlinksInPath()
            let values = try resolvedURL.resourceValues(
                forKeys: [.fileSizeKey, .contentModificationDateKey])
            files.append(SourceFile(
                url: resolvedURL,
                size: UInt64(values.fileSize ?? 0),
                modified: values.contentModificationDate?.timeIntervalSince1970 ?? 0))

            let header = try readSafetensorsHeader(resolvedURL)
            for (key, value) in header.tensors {
                if isStackedRoutedKey(key) {
                    existingStackedKeys.insert(key)
                }
                guard let dtype = value["dtype"] as? String,
                      let shape = value["shape"] as? [Int],
                      let offsets = value["data_offsets"] as? [UInt64],
                      offsets.count == 2,
                      offsets[1] >= offsets[0]
                else { continue }

                let match = matchPerExpertKey(key)
                let source = TensorSource(
                    key: key,
                    expert: match?.expert ?? -1,
                    fileURL: resolvedURL,
                    byteOffset: header.dataBase + offsets[0],
                    byteLength: offsets[1] - offsets[0],
                    dtype: dtype,
                    shape: shape)
                tensors.append(source)
                if let match {
                    pending.append((match.outputKey, source))
                }
            }
        }

        for (outputKey, source) in pending where !existingStackedKeys.contains(outputKey) {
            let expert = source.expert
            var group = groups[outputKey] ?? TensorGroup(outputKey: outputKey)
            group.byExpert[expert] = source
            groups[outputKey] = group
        }

        groups = groups.filter { _, group in
            guard let maxExpert = group.byExpert.keys.max(), group.byExpert[0] != nil else {
                return false
            }
            return (0...maxExpert).allSatisfy { group.byExpert[$0] != nil }
        }

        return BundleScan(
            files: files.sorted { $0.url.path < $1.url.path },
            tensors: tensors,
            groups: groups)
    }

    private struct HeaderRead {
        var dataBase: UInt64
        var tensors: [String: [String: Any]]
    }

    private static func readSafetensorsHeader(_ url: URL) throws -> HeaderRead {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else {
            throw NSError(domain: "JangPressPrestacker", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "short safetensors header: \(url.path)"])
        }
        let headerLength = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let headerData = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard headerData.count == Int(headerLength) else {
            throw NSError(domain: "JangPressPrestacker", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "truncated safetensors header: \(url.path)"])
        }
        let json = try JSONSerialization.jsonObject(with: headerData)
        guard let dict = json as? [String: Any] else {
            throw NSError(domain: "JangPressPrestacker", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "invalid safetensors JSON: \(url.path)"])
        }
        var tensors: [String: [String: Any]] = [:]
        for (key, value) in dict where key != "__metadata__" {
            guard let item = value as? [String: Any] else { continue }
            var normalized = item
            if let shape = item["shape"] as? [NSNumber] {
                normalized["shape"] = shape.map { $0.intValue }
            }
            if let offsets = item["data_offsets"] as? [NSNumber] {
                normalized["data_offsets"] = offsets.map { $0.uint64Value }
            }
            tensors[key] = normalized
        }
        return HeaderRead(dataBase: 8 + headerLength, tensors: tensors)
    }

    private static func buildWritePlan(from groups: [String: TensorGroup]) throws -> [WriteTensor] {
        var result: [WriteTensor] = []
        for group in groups.values.sorted(by: { $0.outputKey < $1.outputKey }) {
            let experts = group.byExpert.keys.sorted()
            guard experts.first == 0, experts.last == experts.count - 1 else { continue }
            let sources = experts.compactMap { group.byExpert[$0] }
            guard let first = sources.first else { continue }
            guard sources.allSatisfy({ $0.dtype == first.dtype && $0.shape == first.shape }) else {
                continue
            }
            let perTensorBytes = sources.reduce(UInt64(0)) { $0 + $1.byteLength }
            let expectedBytes = UInt64(sources.count) * first.byteLength
            guard perTensorBytes == expectedBytes else { continue }
            result.append(WriteTensor(
                key: group.outputKey,
                dtype: first.dtype,
                shape: [sources.count] + first.shape,
                sources: sources))
        }
        assignDataOffsets(&result)
        return result
    }

    private static func buildHotCorePlan(
        from tensors: [TensorSource],
        routedPlan: [WriteTensor]
    ) -> [WriteTensor] {
        let routedKeys = Set(routedPlan.flatMap { $0.sources.map(\.key) })
        var seen = Set<String>()
        var result: [WriteTensor] = []
        for source in tensors.sorted(by: { $0.key < $1.key }) {
            guard !routedKeys.contains(source.key),
                  !isPerExpertTQBitsKey(source.key),
                  seen.insert(source.key).inserted
            else { continue }
            result.append(WriteTensor(
                key: source.key,
                dtype: source.dtype,
                shape: source.shape,
                sources: [source]))
        }
        assignDataOffsets(&result)
        return result
    }

    private static func assignDataOffsets(_ tensors: inout [WriteTensor]) {
        var offset: UInt64 = 0
        for i in tensors.indices {
            tensors[i].dataOffset = offset
            offset += tensors[i].totalBytes
        }
    }

    private static func writeSafetensors(plan: [WriteTensor], to outputURL: URL) throws {
        let tmpURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent(".\(outputURL.lastPathComponent).tmp-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: tmpURL.path, contents: nil)

        let outFD = systemOpen(tmpURL.path, O_WRONLY | O_TRUNC)
        guard outFD >= 0 else {
            throw posixError("open", path: tmpURL.path)
        }
        defer { _ = systemClose(outFD) }

        var header: [String: Any] = [
            "__metadata__": [
                "format": "pt",
                "jangpress_prestacked": "1",
                "jangpress_prestack_version": version,
            ]
        ]
        for item in plan {
            header[item.key] = [
                "dtype": item.dtype,
                "shape": item.shape,
                "data_offsets": [item.dataOffset, item.dataOffset + item.totalBytes],
            ]
        }
        var headerData = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        let dataBaseAlignment = 4096
        let misalignment = (8 + headerData.count) % dataBaseAlignment
        if misalignment != 0 {
            headerData.append(Data(
                repeating: 0x20,
                count: dataBaseAlignment - misalignment))
        }
        var headerLength = UInt64(headerData.count).littleEndian
        try withUnsafeBytes(of: &headerLength) { raw in
            try writeAll(fd: outFD, raw.baseAddress!, raw.count)
        }
        try headerData.withUnsafeBytes { raw in
            try writeAll(fd: outFD, raw.baseAddress!, raw.count)
        }

        try copyTensorPayloads(plan: plan, outFD: outFD)
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }
        try FileManager.default.moveItem(at: tmpURL, to: outputURL)
    }

    private static func copyTensorPayloads(plan: [WriteTensor], outFD: Int32) throws {
        let bufferSize = 4 * 1024 * 1024
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: bufferSize, alignment: 4096)
        defer { buffer.deallocate() }
        var fdCache: [String: Int32] = [:]
        defer {
            for fd in fdCache.values {
                _ = systemClose(fd)
            }
        }

        for item in plan {
            for source in item.sources {
                let path = source.fileURL.path
                let fd: Int32
                if let cached = fdCache[path] {
                    fd = cached
                } else {
                    let opened = systemOpen(path, O_RDONLY)
                    guard opened >= 0 else { throw posixError("open", path: path) }
                    fdCache[path] = opened
                    fd = opened
                }

                var remaining = source.byteLength
                var offset = source.byteOffset
                while remaining > 0 {
                    let toRead = min(UInt64(bufferSize), remaining)
                    let n = systemPread(fd, buffer, Int(toRead), Int64(offset))
                    guard n > 0 else {
                        throw posixError("pread", path: path)
                    }
                    try writeAll(fd: outFD, buffer, n)
                    remaining -= UInt64(n)
                    offset += UInt64(n)
                }
            }
        }
    }

    private static func writeAll(fd: Int32, _ pointer: UnsafeRawPointer, _ count: Int) throws {
        var written = 0
        while written < count {
            let n = systemWrite(fd, pointer.advanced(by: written), count - written)
            guard n > 0 else { throw posixError("write", path: "fd:\(fd)") }
            written += n
        }
    }

    private static func cacheDirectory(for originalURL: URL, files: [SourceFile]) throws -> URL {
        let env = ProcessInfo.processInfo.environment
        let root: URL
        if let override = env["JANGPRESS_PRESTACK_CACHE_DIR"], !override.isEmpty {
            root = URL(fileURLWithPath: override)
        } else {
            root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("vmlx", isDirectory: true)
                .appendingPathComponent("jangpress-prestack", isDirectory: true)
        }
        let hash = stableHash(([version, originalURL.path] + files.map {
            "\($0.url.lastPathComponent):\($0.size):\($0.modified)"
        }).joined(separator: "|"))
        return root
            .appendingPathComponent(originalURL.lastPathComponent + "-" + hash, isDirectory: true)
    }

    private static func ensureOverlayLinks(from source: URL, to cache: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        let entries = try fm.contentsOfDirectory(
            at: source, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        for entry in entries where entry.lastPathComponent != outputName
            && entry.lastPathComponent != hotCoreName
        {
            let dst = cache.appendingPathComponent(entry.lastPathComponent)
            if entry.pathExtension == "safetensors"
                && entry.lastPathComponent != "jangtq_runtime.safetensors"
            {
                if fm.fileExists(atPath: dst.path) {
                    try? fm.removeItem(at: dst)
                }
                continue
            }
            if isDerivedStackedArtifact(entry.lastPathComponent) {
                if fm.fileExists(atPath: dst.path) {
                    try? fm.removeItem(at: dst)
                }
                continue
            }
            if fm.fileExists(atPath: dst.path) { continue }
            try fm.createSymbolicLink(at: dst, withDestinationURL: entry.resolvingSymlinksInPath())
        }
    }

    private static func isDerivedStackedArtifact(_ name: String) -> Bool {
        // `jangtq_stacked.safetensors` is a converter/debug artifact that
        // duplicates routed experts in an eager resident layout. Loading it
        // beside `jangpress-prestacked.safetensors` injects stale
        // `packed`/`norms` keys into TurboQuantSwitchLinear and defeats the
        // whole mmap-backed JangPress path.
        name == "jangtq_stacked.safetensors" || name == "jangtq_stacked.json"
    }

    private static func writeManifest(plan: [WriteTensor], source: URL, to url: URL) throws {
        let total = plan.reduce(UInt64(0)) { $0 + $1.totalBytes }
        let object: [String: Any] = [
            "version": version,
            "source": source.path,
            "tensor_count": plan.count,
            "bytes": total,
            "created": Date().timeIntervalSince1970,
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: [.atomic])
    }

    private struct KeyMatch {
        var expert: Int
        var outputKey: String
    }

    private static func matchPerExpertKey(_ key: String) -> KeyMatch? {
        if let m = match(
            key,
            #"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "model.layers.\(m[1]).block_sparse_moe.switch_mlp.\(projName(m[3])).\(m[4])")
        }
        if let m = match(
            key,
            #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "layers.\(m[1]).ffn.switch_mlp.\(projName(m[3])).\(m[4])")
        }
        if let m = match(
            key,
            #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#
        ) {
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "layers.\(m[1]).ffn.switch_mlp.\(m[3]).\(m[4])")
        }
        if let m = match(
            key,
            #"^(language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(w[123])\.(tq_packed|tq_norms)$"#
        ) {
            let root = m[1]
            return KeyMatch(
                expert: Int(m[3])!,
                outputKey: "\(root)model.layers.\(m[2]).mlp.switch_mlp.\(projName(m[4])).\(m[5])")
        }
        if let m = match(
            key,
            #"^(language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#
        ) {
            let root = m[1]
            return KeyMatch(
                expert: Int(m[3])!,
                outputKey: "\(root)model.layers.\(m[2]).mlp.switch_mlp.\(m[4]).\(m[5])")
        }
        if let m = match(
            key,
            #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(tq_packed|tq_norms)$"#
        ) {
            let fc = m[3] == "up_proj" ? "fc1" : "fc2"
            return KeyMatch(
                expert: Int(m[2])!,
                outputKey: "backbone.layers.\(m[1]).mixer.switch_mlp.\(fc).\(m[4])")
        }
        return nil
    }

    private static func isStackedRoutedKey(_ key: String) -> Bool {
        key.contains(".switch_mlp.") &&
            (key.hasSuffix(".tq_packed") || key.hasSuffix(".tq_norms"))
    }

    private static func isPerExpertTQBitsKey(_ key: String) -> Bool {
        match(key, #"^layers\.\d+\.ffn\.experts\.\d+\.(w[123]|gate_proj|up_proj|down_proj)\.tq_bits$"#) != nil
            || match(
                key,
                #"^(language_model\.)?model\.layers\.\d+\.mlp\.experts\.\d+\.(w[123]|gate_proj|up_proj|down_proj)\.tq_bits$"#
            ) != nil
            || match(
                key,
                #"^(language_model\.)?model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.(w[123]|gate_proj|up_proj|down_proj)\.tq_bits$"#
            ) != nil
    }

    private static func projName(_ raw: String) -> String {
        switch raw {
        case "w1": return "gate_proj"
        case "w2": return "down_proj"
        case "w3": return "up_proj"
        default: return raw
        }
    }

    private static func match(_ value: String, _ pattern: String) -> [String]? {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = value as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let m = re.firstMatch(in: value, range: range) else { return nil }
        var captures: [String] = []
        for i in 0..<m.numberOfRanges {
            let r = m.range(at: i)
            captures.append(r.location == NSNotFound ? "" : ns.substring(with: r))
        }
        return captures
    }

    private static func stableHash(_ input: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }
        return String(format: "%016llx", hash)
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data("[JangPressPrestacker] \(message)\n".utf8))
    }

    private static func posixError(_ op: String, path: String) -> NSError {
        NSError(domain: "JangPressPrestacker", code: Int(errno),
            userInfo: [NSLocalizedDescriptionKey: "\(op)(\(path)) failed: \(String(cString: strerror(errno)))"])
    }

    #if canImport(Darwin)
    private static func systemOpen(_ path: String, _ flags: Int32) -> Int32 {
        Darwin.open(path, flags, 0)
    }
    private static func systemClose(_ fd: Int32) -> Int32 { Darwin.close(fd) }
    private static func systemPread(
        _ fd: Int32, _ buffer: UnsafeMutableRawPointer, _ count: Int, _ offset: Int64
    ) -> Int {
        Darwin.pread(fd, buffer, count, off_t(offset))
    }
    private static func systemWrite(
        _ fd: Int32, _ buffer: UnsafeRawPointer, _ count: Int
    ) -> Int {
        Darwin.write(fd, buffer, count)
    }
    #elseif canImport(Glibc)
    private static func systemOpen(_ path: String, _ flags: Int32) -> Int32 {
        Glibc.open(path, flags, 0)
    }
    private static func systemClose(_ fd: Int32) -> Int32 { Glibc.close(fd) }
    private static func systemPread(
        _ fd: Int32, _ buffer: UnsafeMutableRawPointer, _ count: Int, _ offset: Int64
    ) -> Int {
        Glibc.pread(fd, buffer, count, off_t(offset))
    }
    private static func systemWrite(
        _ fd: Int32, _ buffer: UnsafeRawPointer, _ count: Int
    ) -> Int {
        Glibc.write(fd, buffer, count)
    }
    #endif
}
