// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressMachCache — leverage macOS purgeable memory to keep
// dormant MoE expert tiles compressed-in-place under memory pressure,
// then transparently re-fault them when routing demand arrives.
//
// THE IDEA
// ========
// MoE inference touches only `top_k` experts per token (typically 2-3% of
// the total expert count). The rest are dormant from one decode step to
// the next. macOS's compressed-memory machinery can compress dormant
// pages with WKdm (hardware-accelerated, ~5-30 µs decompress per 16 KB
// page) — but only if it considers them inactive. We tell the kernel
// "these tiles are okay to compress / drop" via `vm_purgable_control`
// with the VOLATILE state and a low priority. When the router selects
// a tile, we flip it to NONVOLATILE — which:
//
//   • Restores resident pages immediately (free / no decompress needed),
//   • Decompresses compressed pages on-demand (~ms scale),
//   • Refaults discarded pages from the on-disk shard (~ms-tens of ms).
//
// Net effect: under memory pressure, dormant experts cost almost no
// physical RAM. Under no pressure, the kernel doesn't compress and
// performance is identical to the all-resident baseline.
//
// WHY THIS BEATS A USER-SPACE LZ4 CACHE
// =====================================
// 1. WKdm is hardware-accelerated. We can't beat it from user-space.
// 2. Compression is adaptive — kernel only does it under pressure.
//    Idle conditions: zero overhead.
// 3. Eviction policy is the kernel's standard LRU-with-priority.
// 4. Per-page granularity (16 KB) is finer than per-expert.
// 5. Memory-pressure response is real-time via standard dispatch sources.
//
// TRADE-OFFS
// ==========
// • macOS-only. Equivalent on Linux is `madvise(MADV_FREE)` + zswap; the
//   semantics differ enough that we keep this Mach-specific.
// • VM region count is a finite resource (~100k regions). DSV4-Flash has
//   256 experts × 43 layers ≈ 11k regions — comfortable. Larger MoEs
//   (Kimi K2.6 at 384×61=23k regions) are still well under the limit.
// • Refault-from-disk latency is the worst case. Mitigated by
//   `MADV_WILLNEED` prefetch and predictive routing-history.
// • Volatility flag is per-region; it doesn't pin individual pages,
//   so a NONVOLATILE region whose pages haven't been touched in a
//   long time may still need decompress on the first read. That's
//   the kernel's choice — we just request the state.
//
// INTEGRATION POINTS
// ==================
// • `JangLoader` populates the cache during model load — instead of
//   keeping raw expert weight arrays in MLXArray-resident form, expert
//   tiles get registered here. Hot-set initially is ALL experts (full
//   resident); kernel pressure drives migration to compressed form
//   over time.
// • `SwitchGLU` / `SwitchLinear` consult `acquire(layerId:expertIds:)`
//   before launching the per-expert matmul kernel; releases happen
//   after the kernel completes (overlap with next layer's attention).
// • `CacheCoordinator` is the natural owner — it already orchestrates
//   the `Memory → L1 → L2` cascade for prefix cache. We extend it
//   with a fourth dimension: routed-expert weights.
//
// COMPATIBILITY WITH OTHER CACHES
// ===============================
// • TurboQuant KV cache: orthogonal — KV is per-token-position, expert
//   tile is per-expert weight. Both run simultaneously.
// • L1 / L2 BlockDiskCache: orthogonal. L2 stores prefix-cache blocks;
//   expert tiles are model weights, not cache.
// • Hybrid SSM: orthogonal. SSM state is per-layer, not per-expert.
//   `enableSSMReDerive` re-runs the prompt forward, which transparently
//   re-acquires whatever experts the routing chose; if the kernel had
//   compressed them, the WKdm decompress fires and the run succeeds
//   slightly slower (one-time cost; subsequent decode is fully warm).

import Foundation
import Darwin
import os

// MARK: - Errors

public enum JangPressMachError: Error, CustomStringConvertible {
    case vmAllocateFailed(kern_return_t)
    case vmPurgableControlFailed(kern_return_t)
    case mmapFailed(Int32)
    case unknownExpert(layer: Int, expert: Int)
    case alreadyDiscarded(layer: Int, expert: Int)

    public var description: String {
        switch self {
        case .vmAllocateFailed(let kr):
            return "vm_allocate failed: kr=\(kr)"
        case .vmPurgableControlFailed(let kr):
            return "vm_purgable_control failed: kr=\(kr)"
        case .mmapFailed(let err):
            return "mmap failed: errno=\(err)"
        case .unknownExpert(let l, let e):
            return "unknown expert (layer=\(l), expert=\(e))"
        case .alreadyDiscarded(let l, let e):
            return "expert tile (layer=\(l), expert=\(e)) was DISCARDED — refault not yet wired"
        }
    }
}

// MARK: - Configuration

public struct JangPressMachConfig: Sendable {
    /// Fraction of experts to keep always-non-volatile (cannot be
    /// compressed by kernel). Range 0.0–1.0. Default 0.0 = let the
    /// kernel manage every tile by routing-frequency LRU.
    public var alwaysHotFraction: Double

    /// Predictive prefetch — touch likely-next-layer experts during the
    /// current layer's attention compute. Reduces decompress stalls.
    public var enablePrefetch: Bool

    /// Per-expert disk-refault enabled. Requires `bundlePath` to be set
    /// at register-time; without it, a discarded tile will be reported
    /// as `alreadyDiscarded` rather than refaulted. Default off until
    /// the on-disk indexer is wired (currently we just `mmap` the
    /// safetensors file directly so kernel can refault from inode).
    public var enableDiskRefault: Bool

    /// User-facing percentage knob (0–100). Maps to `alwaysHotFraction`
    /// such that `manualCompressPercent = 100 - alwaysHotFraction*100`.
    /// E.g. `manualCompressPercent = 70` means 30 % hot, 70 % open to
    /// compression. Wired through `Engine.LoadOptions.jangPressCompressPct`.
    public var manualCompressPercent: Int?

    public init(
        alwaysHotFraction: Double = 0.0,
        enablePrefetch: Bool = true,
        enableDiskRefault: Bool = false,
        manualCompressPercent: Int? = nil
    ) {
        self.alwaysHotFraction = alwaysHotFraction
        self.enablePrefetch = enablePrefetch
        self.enableDiskRefault = enableDiskRefault
        self.manualCompressPercent = manualCompressPercent
    }
}

// MARK: - Tile

/// Per-expert weight tile state. The pointer fields are only safe to
/// touch under `JangPressMachCache`'s lock — `@unchecked` is
/// the right escape hatch here.
public struct JangPressTile: @unchecked Sendable {
    public let layerId: Int
    public let expertId: Int
    /// Base address of the VM region holding the tile bytes. Stable
    /// across acquire/release calls (region doesn't move).
    public let baseAddress: UnsafeMutableRawPointer
    /// Size of the tile in bytes (page-rounded internally).
    public let size: Int
    /// Mach VM region handle.
    fileprivate let region: vm_address_t
    /// Optional disk fallback when kernel discards the region.
    public let diskURL: URL?
    public let diskOffset: UInt64
    /// Tracked routing frequency (for hot-set scoring).
    fileprivate var accessCount: UInt64
    fileprivate var lastAccessTick: UInt64
}

// MARK: - Stats

public struct JangPressMachStats: Sendable {
    public var totalTiles: Int
    public var totalBytesAllocated: Int
    public var hotPinned: Int          // alwaysHot, never volatile
    public var currentlyVolatile: Int
    public var currentlyNonVolatile: Int
    public var acquireCount: UInt64
    public var releaseCount: UInt64
    public var refaultCount: UInt64    // kernel had to refault from disk
    public var discardCount: UInt64    // tile was DROPPED (had to disk-refault)
    public var pressureLowCount: UInt64
    public var pressureWarnCount: UInt64
    public var pressureCriticalCount: UInt64
}

// MARK: - Cache

public final class JangPressMachCache: @unchecked Sendable {

    private let config: JangPressMachConfig
    private let lock = OSAllocatedUnfairLock()
    private var tiles: [TileKey: JangPressTile] = [:]
    private var hotPinned: Set<TileKey> = []
    private var stats = JangPressMachStats(
        totalTiles: 0, totalBytesAllocated: 0,
        hotPinned: 0, currentlyVolatile: 0, currentlyNonVolatile: 0,
        acquireCount: 0, releaseCount: 0,
        refaultCount: 0, discardCount: 0,
        pressureLowCount: 0, pressureWarnCount: 0, pressureCriticalCount: 0)
    private var pressureSource: DispatchSourceMemoryPressure?

    private let log = Logger(subsystem: "ai.jangq.vmlx", category: "RoutedExpertCache")

    /// Composite key (layer, expert).
    fileprivate struct TileKey: Hashable, Sendable {
        let layer: Int
        let expert: Int
    }

    public init(config: JangPressMachConfig = .init()) {
        self.config = config
        installPressureMonitor()
    }

    deinit {
        pressureSource?.cancel()
        // Free all VM regions on shutdown.
        for tile in tiles.values {
            _ = vm_deallocate(mach_task_self_, tile.region, vm_size_t(tile.size))
        }
    }

    // MARK: - Registration

    /// Register an expert tile by copying `bytes` into a new purgeable
    /// VM region. Returns the tile descriptor.
    @discardableResult
    public func register(
        layer: Int,
        expert: Int,
        bytes: UnsafeRawBufferPointer,
        diskURL: URL? = nil,
        diskOffset: UInt64 = 0
    ) throws -> JangPressTile {
        let pageSize = Int(vm_kernel_page_size)
        let alignedSize = ((bytes.count + pageSize - 1) / pageSize) * pageSize

        var addr: vm_address_t = 0
        let kr = vm_allocate(
            mach_task_self_,
            &addr,
            vm_size_t(alignedSize),
            VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE
        )
        guard kr == KERN_SUCCESS else {
            throw JangPressMachError.vmAllocateFailed(kr)
        }

        // Copy weights into the new region
        if let src = bytes.baseAddress, bytes.count > 0 {
            memcpy(UnsafeMutableRawPointer(bitPattern: UInt(addr))!, src, bytes.count)
        }

        let key = TileKey(layer: layer, expert: expert)
        let tile = JangPressTile(
            layerId: layer,
            expertId: expert,
            baseAddress: UnsafeMutableRawPointer(bitPattern: UInt(addr))!,
            size: alignedSize,
            region: addr,
            diskURL: diskURL,
            diskOffset: diskOffset,
            accessCount: 0,
            lastAccessTick: 0
        )

        lock.withLock {
            tiles[key] = tile
            stats.totalTiles += 1
            stats.totalBytesAllocated += alignedSize

            // Initial state: NON-VOLATILE so first decode runs at full speed.
            // Pressure events transition cold tiles to VOLATILE.
            stats.currentlyNonVolatile += 1
        }

        return tile
    }

    /// Mark a set of experts as "always hot" — they never get flipped
    /// to volatile, so the kernel can never compress or evict them.
    /// Used for the top-N most-routed experts when `alwaysHotFraction > 0`.
    public func pinHot(layer: Int, experts: [Int]) {
        lock.withLock {
            for e in experts {
                hotPinned.insert(TileKey(layer: layer, expert: e))
            }
            stats.hotPinned = hotPinned.count
        }
    }

    // MARK: - Routing-time API

    /// Mark a list of experts NON-VOLATILE before the matmul fires.
    /// Returns the tile descriptors, in the same order requested.
    public func acquire(layer: Int, experts: [Int]) throws -> [JangPressTile] {
        var out: [JangPressTile] = []
        out.reserveCapacity(experts.count)
        try lock.withLockUnchecked {
            for e in experts {
                let key = TileKey(layer: layer, expert: e)
                guard var tile = tiles[key] else {
                    throw JangPressMachError.unknownExpert(layer: layer, expert: e)
                }

                // Skip the kernel call if already pinned hot.
                if !hotPinned.contains(key) {
                    var state: Int32 = VM_PURGABLE_NONVOLATILE
                    let kr = vm_purgable_control(
                        mach_task_self_,
                        tile.region,
                        VM_PURGABLE_SET_STATE,
                        &state
                    )
                    guard kr == KERN_SUCCESS else {
                        throw JangPressMachError.vmPurgableControlFailed(kr)
                    }

                    // If the kernel had already discarded, the previous state
                    // returned via the `state` out-arg is VM_PURGABLE_EMPTY.
                    // The region's pages are now zeroed — we have to refault
                    // from disk to make it usable.
                    if state == VM_PURGABLE_EMPTY {
                        stats.discardCount += 1
                        if let url = tile.diskURL {
                            try refaultFromDisk(tile: tile, diskURL: url, offset: tile.diskOffset)
                            stats.refaultCount += 1
                        } else {
                            throw JangPressMachError.alreadyDiscarded(layer: layer, expert: e)
                        }
                    }
                }

                tile.accessCount &+= 1
                tile.lastAccessTick = mach_absolute_time()
                tiles[key] = tile
                out.append(tile)
                stats.acquireCount &+= 1
            }
        }
        return out
    }

    /// Release a list of experts back to VOLATILE so the kernel can
    /// compress them under pressure. Hot-pinned experts ignore this call.
    public func release(layer: Int, experts: [Int]) {
        lock.withLock {
            for e in experts {
                let key = TileKey(layer: layer, expert: e)
                guard let tile = tiles[key] else { continue }
                if hotPinned.contains(key) { continue }

                var state: Int32 = VM_PURGABLE_VOLATILE
                let kr = vm_purgable_control(
                    mach_task_self_,
                    tile.region,
                    VM_PURGABLE_SET_STATE,
                    &state
                )
                if kr == KERN_SUCCESS {
                    stats.releaseCount &+= 1
                } else {
                    log.warning("release failed (kr=\(kr)) layer=\(layer) expert=\(e)")
                }
            }
        }
    }

    // MARK: - Disk refault

    private func refaultFromDisk(tile: JangPressTile, diskURL: URL, offset: UInt64) throws {
        let fd = open(diskURL.path, O_RDONLY)
        guard fd >= 0 else { throw JangPressMachError.mmapFailed(errno) }
        defer { close(fd) }

        let dst = tile.baseAddress
        var read = 0
        while read < tile.size {
            let n = pread(fd, dst.advanced(by: read), tile.size - read,
                          off_t(offset) + off_t(read))
            if n <= 0 { throw JangPressMachError.mmapFailed(errno) }
            read += n
        }
    }

    // MARK: - Memory-pressure listener

    private func installPressureMonitor() {
        let q = DispatchQueue(label: "ai.jangq.vmlx.routed-expert-pressure", qos: .utility)
        let src = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical, .normal],
            queue: q
        )
        src.setEventHandler { [weak self] in
            guard let self else { return }
            let event = src.data
            self.lock.withLock {
                if event.contains(.normal)   { self.stats.pressureLowCount   &+= 1 }
                if event.contains(.warning)  { self.stats.pressureWarnCount  &+= 1 }
                if event.contains(.critical) { self.stats.pressureCriticalCount &+= 1 }
            }
            self.log.notice("memory pressure event: \(String(describing: event))")
            // Future: under critical pressure we can pre-emptively
            // VOLATILE-flip cold tiles past the hot fraction. For now
            // the kernel handles that via the existing VOLATILE flag.
        }
        src.activate()
        pressureSource = src
    }

    // MARK: - Stats

    public func snapshot() -> JangPressMachStats {
        lock.withLock { stats }
    }
}
