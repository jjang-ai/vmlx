// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressShard — page-cache-backed view of a safetensors shard.
//
// PURPOSE
// =======
// The vm_purgable_control approach in `JangPressMachCache`
// requires copying weight bytes into a fresh VM region — which doubles
// RAM usage at load time. This module is the alternative: mmap the
// safetensors file directly, parse its tensor index, and let the
// kernel's page cache handle resident vs swapped-out state. Combined
// with `madvise(MADV_DONTNEED)` on dormant byte ranges, the kernel
// reclaims those pages without us ever copying.
//
// SAFETENSORS LAYOUT (recap)
// ==========================
//
//   bytes  0..8        little-endian uint64: header_size in bytes
//   bytes  8..8+H      JSON header: { "tensor_name": { "dtype": "F16",
//                       "shape": [...], "data_offsets": [start, end] } }
//   bytes  8+H..end    tensor data, where each tensor's bytes live at
//                       absolute file offset `8 + H + start`.
//
// USAGE
// =====
//
//   let shard = try JangPressShard(path: shardURL)
//   defer { shard.close() }
//
//   if let range = shard.byteRange(for: "model.layers.0.mlp.switch_mlp.gate_proj.weight") {
//       shard.advise(.willNeed, range: range)        // pre-fault
//       // ... use bytes via shard.bytes(in: range) ...
//       shard.advise(.dontNeed, range: range)        // ask kernel to evict
//   }
//
// SAFETY
// ======
// • The mmap region stays read-only (PROT_READ). Modifications are
//   never written back.
// • `MADV_DONTNEED` tells the kernel "I don't need these pages now"
//   — they may be reclaimed but the file is the source of truth, so
//   re-access transparently re-faults from disk.
// • `MADV_FREE` (alternative) is more aggressive — kernel can drop
//   immediately. Only useful for anonymous memory, not file-backed.
// • The shard view shares pages with the kernel page cache. Other
//   processes / MLX itself reading the same file see the same pages.
//   No RAM doubling.

import Foundation
import Darwin

public enum JangPressShardError: Error, CustomStringConvertible {
    case openFailed(URL, errno: Int32)
    case statFailed(URL, errno: Int32)
    case mmapFailed(URL, errno: Int32)
    case truncatedHeader(URL)
    case malformedHeaderJSON(URL, String)
    case headerSizeOutOfBounds(URL, declared: UInt64, fileSize: UInt64)
    case unknownTensor(name: String)

    public var description: String {
        switch self {
        case .openFailed(let url, let e): return "open(\(url.lastPathComponent)) failed errno=\(e)"
        case .statFailed(let url, let e): return "fstat(\(url.lastPathComponent)) failed errno=\(e)"
        case .mmapFailed(let url, let e): return "mmap(\(url.lastPathComponent)) failed errno=\(e)"
        case .truncatedHeader(let url): return "safetensors header truncated in \(url.lastPathComponent)"
        case .malformedHeaderJSON(let url, let m): return "safetensors header JSON malformed in \(url.lastPathComponent): \(m)"
        case .headerSizeOutOfBounds(let url, let h, let f): return "header size \(h) > file size \(f) in \(url.lastPathComponent)"
        case .unknownTensor(let n): return "tensor not found: \(n)"
        }
    }
}

public enum JangPressAdvice: Sendable {
    case willNeed     // MADV_WILLNEED — kernel should pre-fault these
    case dontNeed     // MADV_DONTNEED — kernel may reclaim these now
    case sequential   // MADV_SEQUENTIAL — read-ahead aggressively
    case random       // MADV_RANDOM — disable read-ahead

    var rawValue: Int32 {
        switch self {
        case .willNeed:    return Int32(MADV_WILLNEED)
        case .dontNeed:    return Int32(MADV_DONTNEED)
        case .sequential:  return Int32(MADV_SEQUENTIAL)
        case .random:      return Int32(MADV_RANDOM)
        }
    }
}

public struct TensorDescriptor: Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let dataOffset: UInt64        // absolute byte offset in shard file
    public let dataLength: UInt64        // tensor byte length
}

public final class JangPressShard: @unchecked Sendable {

    public let url: URL
    public let fileSize: UInt64
    public let baseAddress: UnsafeRawPointer
    public private(set) var tensors: [String: TensorDescriptor] = [:]

    private let fd: Int32
    private let pageSize: Int

    public init(path: URL) throws {
        self.url = path
        self.pageSize = Int(getpagesize())

        // 1. open + fstat
        let fdLocal = open(path.path, O_RDONLY)
        guard fdLocal >= 0 else { throw JangPressShardError.openFailed(path, errno: errno) }
        self.fd = fdLocal

        var st = stat()
        guard fstat(fdLocal, &st) == 0 else {
            close(fdLocal)
            throw JangPressShardError.statFailed(path, errno: errno)
        }
        self.fileSize = UInt64(st.st_size)

        // 2. mmap whole file PROT_READ
        guard let raw = mmap(nil, Int(self.fileSize), PROT_READ, MAP_SHARED, fdLocal, 0),
              raw != UnsafeMutableRawPointer(bitPattern: -1) else {
            close(fdLocal)
            throw JangPressShardError.mmapFailed(path, errno: errno)
        }
        self.baseAddress = UnsafeRawPointer(raw)

        // 3. parse safetensors header (first 8 bytes = header size)
        guard fileSize >= 8 else {
            munmap(raw, Int(fileSize))
            close(fdLocal)
            throw JangPressShardError.truncatedHeader(path)
        }
        let headerSize = baseAddress.load(as: UInt64.self).littleEndian
        guard 8 + headerSize <= fileSize else {
            munmap(raw, Int(fileSize))
            close(fdLocal)
            throw JangPressShardError.headerSizeOutOfBounds(path, declared: headerSize, fileSize: fileSize)
        }

        let headerJSONData = Data(bytes: baseAddress.advanced(by: 8), count: Int(headerSize))
        let dataAreaStart: UInt64 = 8 + headerSize
        let parsed: [String: Any]
        do {
            guard let obj = try JSONSerialization.jsonObject(with: headerJSONData)
                as? [String: Any] else {
                throw JangPressShardError.malformedHeaderJSON(path, "not a JSON object")
            }
            parsed = obj
        } catch {
            munmap(raw, Int(fileSize))
            close(fdLocal)
            throw JangPressShardError.malformedHeaderJSON(path, "\(error)")
        }

        // 4. extract tensor descriptors. The `__metadata__` key (if
        //    present) is non-tensor metadata.
        var byName: [String: TensorDescriptor] = [:]
        byName.reserveCapacity(parsed.count)
        for (name, value) in parsed {
            if name == "__metadata__" { continue }
            guard
                let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Any],
                offsets.count == 2,
                let startNum = (offsets[0] as? NSNumber)?.uint64Value,
                let endNum = (offsets[1] as? NSNumber)?.uint64Value
            else {
                continue
            }
            byName[name] = TensorDescriptor(
                name: name,
                dtype: dtype,
                shape: shape,
                dataOffset: dataAreaStart + startNum,
                dataLength: endNum - startNum
            )
        }
        self.tensors = byName
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: baseAddress), Int(fileSize))
        close(fd)
    }

    // MARK: - Header-only sniff (NEW iter 19)
    //
    // Pre-parse the safetensors header without mmap'ing the data area.
    // Used by JangPressMmapTier to decide whether a shard contains
    // routed-expert tensors before paying the cost of mmap'ing the
    // whole file. Caller passes the parsed names through their regex;
    // if no match → skip the shard.
    //
    // Cost: one open + one read of (8 + headerSize) bytes (typically
    // < 1 MB). Compared to mmap'ing a 5 GB shard, this is ~5000× cheaper
    // and avoids the page-cache competition documented in
    // JANGPRESS-DEEP-TRACE.md Issue 5b.

    /// Returns just the tensor names in the shard's safetensors header,
    /// without mmap'ing the data area. Returns nil if header parse fails
    /// (caller should fall back to full open via init).
    public static func sniffTensorNames(at path: URL) -> [String]? {
        let fd = open(path.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }

        // Read first 8 bytes — header size.
        var sizeBytes = [UInt8](repeating: 0, count: 8)
        let n = sizeBytes.withUnsafeMutableBufferPointer { buf -> Int in
            read(fd, buf.baseAddress, 8)
        }
        guard n == 8 else { return nil }
        let headerSize: UInt64 = sizeBytes.withUnsafeBytes { ptr in
            ptr.load(as: UInt64.self).littleEndian
        }
        // Cap header size at 32 MB — anything larger is suspicious and
        // not worth pre-reading.
        guard headerSize > 0 && headerSize < 32 * 1024 * 1024 else { return nil }

        var headerBytes = [UInt8](repeating: 0, count: Int(headerSize))
        let h = headerBytes.withUnsafeMutableBufferPointer { buf -> Int in
            read(fd, buf.baseAddress, Int(headerSize))
        }
        guard h == Int(headerSize) else { return nil }

        let headerData = Data(headerBytes)
        guard let obj = try? JSONSerialization.jsonObject(with: headerData)
                as? [String: Any] else { return nil }
        // Filter out __metadata__ + any non-tensor keys.
        return obj.keys.filter { $0 != "__metadata__" }
    }

    // MARK: - Tensor lookup

    /// Returns the absolute byte range in the shard file for the given
    /// tensor name. nil if the tensor isn't in this shard.
    public func byteRange(for tensorName: String) -> Range<UInt64>? {
        guard let d = tensors[tensorName] else { return nil }
        return d.dataOffset..<(d.dataOffset + d.dataLength)
    }

    public func descriptor(for tensorName: String) -> TensorDescriptor? {
        tensors[tensorName]
    }

    public func bytes(in range: Range<UInt64>) -> UnsafeRawBufferPointer {
        let start = baseAddress.advanced(by: Int(range.lowerBound))
        let count = Int(range.upperBound - range.lowerBound)
        return UnsafeRawBufferPointer(start: start, count: count)
    }

    // MARK: - madvise

    /// Advise the kernel about the given byte range. `range` is rounded
    /// outward to page boundaries since `madvise` operates on pages.
    @discardableResult
    public func advise(_ advice: JangPressAdvice, range: Range<UInt64>) -> Bool {
        let alignedStart = UInt64(range.lowerBound) & ~UInt64(pageSize - 1)
        let endRoundUp   = (range.upperBound + UInt64(pageSize - 1)) & ~UInt64(pageSize - 1)
        let length = Int(endRoundUp - alignedStart)
        guard length > 0 else { return false }
        let addr = UnsafeMutableRawPointer(mutating: baseAddress.advanced(by: Int(alignedStart)))
        let rc = madvise(addr, length, advice.rawValue)
        return rc == 0
    }

    /// Apply advice to the entire data region (everything after the
    /// safetensors header).
    @discardableResult
    public func adviseEntireDataArea(_ advice: JangPressAdvice) -> Bool {
        let header = baseAddress.load(as: UInt64.self).littleEndian
        let dataStart: UInt64 = 8 + header
        return advise(advice, range: dataStart..<fileSize)
    }

    /// Synchronous read-touch — force pages in the range to be resident
    /// by reading one byte per page. Stronger than `madvise(WILLNEED)`,
    /// which is only a hint and races the matmul on cold ranges.
    ///
    /// Cost: one cache-line read per page in the range. Page-fault
    /// latency dominates; this only adds an L2-resident byte read on
    /// already-resident pages. The cold path (hit-after-fault) is the
    /// expensive part either way — this just moves it from inside the
    /// matmul to the prefault site, where it can run concurrently with
    /// neighbouring layers' compute.
    @discardableResult
    public func touchRangeResident(range: Range<UInt64>) -> Int {
        let alignedStart = UInt64(range.lowerBound) & ~UInt64(pageSize - 1)
        let endRoundUp   = (range.upperBound + UInt64(pageSize - 1)) & ~UInt64(pageSize - 1)
        guard endRoundUp > alignedStart else { return 0 }
        let pageStride = Int(pageSize)
        let basePtr = baseAddress.advanced(by: Int(alignedStart))
        let totalLen = Int(endRoundUp - alignedStart)
        var touched = 0
        var off = 0
        // Use a volatile-style read through a UInt8 pointer to keep the
        // optimiser from eliding the load. We don't store the value;
        // the side-effect we care about is the page fault.
        let bytes = basePtr.assumingMemoryBound(to: UInt8.self)
        while off < totalLen {
            _ = bytes[off]
            touched += 1
            off += pageStride
        }
        return touched
    }

    /// Force the kernel to invalidate (drop) clean pages in the given
    /// range via `msync(MS_INVALIDATE | MS_ASYNC)`. This is the
    /// Darwin-specific path to actually evict file-backed pages —
    /// stronger than `madvise(DONTNEED)` which the macOS kernel
    /// treats as a hint and may ignore.
    ///
    /// Use this when you're confident a region is dormant for an
    /// extended period (e.g. JANGPress's quiesce-time compaction).
    /// Contrast `advise(.dontNeed, …)` which is the soft hint path.
    @discardableResult
    public func forceInvalidate(range: Range<UInt64>) -> Bool {
        let alignedStart = UInt64(range.lowerBound) & ~UInt64(pageSize - 1)
        let endRoundUp   = (range.upperBound + UInt64(pageSize - 1)) & ~UInt64(pageSize - 1)
        let length = Int(endRoundUp - alignedStart)
        guard length > 0 else { return false }
        let addr = UnsafeMutableRawPointer(mutating: baseAddress.advanced(by: Int(alignedStart)))
        // MS_INVALIDATE | MS_ASYNC: invalidate cached copies of these
        // pages without forcing a synchronous write-back (we mapped
        // PROT_READ so there's nothing to write anyway).
        let rc = msync(addr, length, MS_INVALIDATE | MS_ASYNC)
        return rc == 0
    }
}
