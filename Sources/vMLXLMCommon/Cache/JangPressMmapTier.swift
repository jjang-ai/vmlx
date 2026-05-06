// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressMmapTier — bundle-aware mmap+madvise tier for routed
// MoE expert weights.
//
// PURPOSE
// =======
// Open every safetensors shard in a bundle once at load time as a
// read-only `JangPressShard`. Walk the shards' tensor indexes to
// identify which entries are routed-expert tiles (per-architecture
// regex patterns). Build an in-memory map of (layer, expert) →
// (shard, byteRange). At inference time:
//
//   • acquire(layer, experts) → call madvise(.willNeed) on those
//     ranges so the kernel pre-faults.
//   • release(layer, experts) → call madvise(.dontNeed) so the
//     kernel can reclaim those pages under pressure.
//
// COMPARED TO JangPressMachCache
// =======================================
// Different tradeoff space:
//
//   JangPressMachCache (vm_purgable_control)
//     • Owns its own copy of weights in fresh purgeable VM regions.
//     • Independent of how MLX stores tensors.
//     • Doubles RAM at load (until MLX integration replaces the
//       canonical storage with our region — gated on MLX-swift fork).
//     • Kernel uses WKdm to compress dormant pages.
//
//   JangPressMmapTier (file-backed mmap + madvise)
//     • Uses the bundle file as the source of truth.
//     • No extra RAM — pages are file-backed, shared with the kernel
//       page cache. Multiple opens of the same file share pages.
//     • Discard reclaims pages back to the file (no compression);
//       refault re-reads from disk.
//     • Doesn't conflict with MLX — they hold ANOTHER copy in their
//       allocator. Our mmap is a parallel read-only view.
//     • Win comes when memory pressure removes our mmap pages —
//       MLX's copies are still resident, so the model still works.
//       BUT: in the future, if MLX is taught to read from our mmap
//       directly (replacing its allocator), we save the duplicate
//       copy entirely.
//
// REGEX PATTERNS PER FAMILY
// =========================
// Routed-expert tile names follow architecture-family conventions.
// We support the most common shapes today:
//
//   model.layers.<L>.mlp.switch_mlp.<gate|up|down>_proj.weight
//     (Qwen 3.5/3.6, GLM 4/5, MiniMax, Laguna stacked-expert format)
//
//   model.layers.<L>.mlp.experts.<E>.<gate|up|down>_proj.weight
//     (DSV3 / DSV4 / Kimi K2.x per-expert format)
//
// The patterns are matched per-bundle at construction time — we
// detect which scheme is in use by counting tensor names that match
// each.

import Foundation

public struct JangPressMmapConfig: Sendable {
    /// Path to a bundle directory holding safetensors shards. We open
    /// every `*.safetensors` file in this directory.
    public let bundleURL: URL

    /// 0..100 — fraction of the routed-expert pool to keep MADV_WILLNEED
    /// ("hot") at all times. Bottom (1 - hotPct/100) gets MADV_DONTNEED
    /// when idle. Default 30 % hot.
    public var hotPercent: Int

    /// When true, the entire data area of every shard starts with
    /// MADV_DONTNEED. The first inference's `acquire()` calls switch
    /// the active set to MADV_WILLNEED. Use this for memory-tight
    /// startups; under low pressure on a roomy machine prefer false
    /// (default) so all weights start resident.
    public var startCold: Bool

    public init(bundleURL: URL, hotPercent: Int = 30, startCold: Bool = false) {
        self.bundleURL = bundleURL
        self.hotPercent = max(0, min(100, hotPercent))
        self.startCold = startCold
    }
}

public final class JangPressMmapTier: @unchecked Sendable {

    public let config: JangPressMmapConfig

    /// Shards opened from the bundle. Held strongly so the mmap
    /// regions stay alive for the lifetime of the tier.
    /// Lazily populated by `ensureBuilt()` on first use.
    public var shards: [URL: JangPressShard] {
        ensureBuilt()
        return _shards
    }
    private var _shards: [URL: JangPressShard] = [:]

    /// (layer, expert) → list of (shard, byteRange) for the three
    /// expert projections (gate / up / down). One expert may contribute
    /// multiple ranges (one per projection).
    public struct ExpertRanges: Sendable {
        public let layer: Int
        public let expert: Int
        public let parts: [(shard: URL, range: Range<UInt64>)]
        public var totalBytes: UInt64 {
            parts.reduce(0) { $0 + ($1.range.upperBound - $1.range.lowerBound) }
        }
    }
    /// Lazily populated by `ensureBuilt()` on first use.
    public var experts: [TileKey: ExpertRanges] {
        ensureBuilt()
        return _experts
    }
    private var _experts: [TileKey: ExpertRanges] = [:]

    /// Tile identifier — matches the shape used by
    /// `JangPressMachCache`.
    public struct TileKey: Hashable, Sendable {
        public let layer: Int
        public let expert: Int
        public init(layer: Int, expert: Int) {
            self.layer = layer
            self.expert = expert
        }
    }

    /// iter 24: build state machine. Init is now O(1) and does no I/O.
    /// All shard opens, header parses, and tile indexing are deferred
    /// to `ensureBuilt()` which fires on first acquire/release/snapshot.
    /// This addresses the load-time SIGKILL on memory-tight hosts where
    /// JangPress was racing MLX's pread for limited RAM.
    private let buildLock = NSLock()
    private var didBuild = false
    private var buildError: Error?

    public init(config: JangPressMmapConfig) throws {
        self.config = config
        // No work in init. The bundle URL is validated lazily inside
        // ensureBuilt() so that Engine.load completes regardless of
        // whether the bundle actually exists / is readable. A missing
        // bundle becomes a noop tier (acquire/release no-op cleanly)
        // rather than a hard load failure.
    }

    /// Lazily perform shard sniff + mmap + tile indexing. Idempotent;
    /// concurrent callers will block on `buildLock` and observe the
    /// already-built state. Errors are captured + re-thrown on the
    /// first call only (subsequent calls observe no-op state).
    private func ensureBuilt() {
        // Fast path — already built.
        if didBuild { return }
        buildLock.lock(); defer { buildLock.unlock() }
        if didBuild { return }
        defer { didBuild = true }

        let fm = FileManager.default
        let shardURLs: [URL]
        do {
            shardURLs = try fm.contentsOfDirectory(
                at: config.bundleURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ).filter {
                $0.pathExtension == "safetensors"
                    && !JangLoader.shouldSkipModelSafetensorsFile($0)
            }
        } catch {
            buildError = error
            FileHandle.standardError.write(Data(
                "[JangPressMmapTier] enumerate \(config.bundleURL.path) failed: \(error) — tier inert\n".utf8))
            return
        }

        // iter 19: header-sniff each shard FIRST to identify which
        // contain routed-expert tensors. Skip mmap'ing shards that have
        // only attention/embed/lm_head/etc.
        var openedShards: [URL: JangPressShard] = [:]
        var skippedCount = 0
        for url in shardURLs {
            guard let names = JangPressShard.sniffTensorNames(at: url) else {
                do {
                    openedShards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressMmapTier] sniff+open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
                continue
            }
            let hasRoutedExpert = names.contains(where: { name in
                Self.parseRoutedExpertName(name) != nil
            })
            if hasRoutedExpert {
                do {
                    openedShards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressMmapTier] open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
            } else {
                skippedCount += 1
            }
        }
        self._shards = openedShards
        if skippedCount > 0 {
            FileHandle.standardError.write(Data(
                "[JangPressMmapTier] lazy-built: sniffed \(shardURLs.count) shards, mmap'd \(openedShards.count), skipped \(skippedCount) (no routed experts)\n".utf8))
        }

        // Walk tensor names + build (layer, expert) → byte-range map.
        // Iter 25: stacked tiles get split into per-expert sub-ranges
        // using their shape[0] (= numExperts). This drops re-fault
        // latency on stacked layouts (Holo3/Qwen3.6/Cascade-2) from
        // 65-198 ms (whole 67-304 MB tile faulted) to ~5 ms (only the
        // single expert's sub-range faulted). Per-expert layouts are
        // unaffected.
        var byKey: [TileKey: [(URL, Range<UInt64>)]] = [:]
        var stackedSplitCount = 0
        var stackedTileCount = 0
        for (url, shard) in openedShards {
            for name in shard.tensors.keys {
                guard let parsed = Self.parseRoutedTile(name),
                      let range = shard.byteRange(for: name)
                else { continue }

                switch parsed {
                case .perExpert(let layer, let expert):
                    byKey[TileKey(layer: layer, expert: expert), default: []]
                        .append((url, range))

                case .stacked(let layer):
                    // shape[0] = numExperts. Split byte range into N
                    // equal sub-ranges so each expert can be advised
                    // independently. Falls back to a single synthetic
                    // expert=0 entry if shape isn't usable (e.g.
                    // descriptor missing or shape[0] == 1).
                    guard let desc = shard.descriptor(for: name),
                          let numExperts = desc.shape.first,
                          numExperts > 1
                    else {
                        byKey[TileKey(layer: layer, expert: 0), default: []]
                            .append((url, range))
                        continue
                    }
                    stackedTileCount += 1
                    let totalBytes = range.upperBound - range.lowerBound
                    // sanity: total bytes must divide cleanly. If they
                    // don't (rare — shouldn't happen for safetensors),
                    // fall back to whole-tile registration.
                    let bytesPerExpert = totalBytes / UInt64(numExperts)
                    if bytesPerExpert * UInt64(numExperts) != totalBytes
                        || bytesPerExpert == 0 {
                        byKey[TileKey(layer: layer, expert: 0), default: []]
                            .append((url, range))
                        continue
                    }
                    stackedSplitCount += numExperts
                    for e in 0..<numExperts {
                        let start = range.lowerBound + UInt64(e) * bytesPerExpert
                        let end = start + bytesPerExpert
                        byKey[TileKey(layer: layer, expert: e), default: []]
                            .append((url, start..<end))
                    }
                }
            }
        }
        var built: [TileKey: ExpertRanges] = [:]
        for (key, parts) in byKey {
            built[key] = ExpertRanges(
                layer: key.layer,
                expert: key.expert,
                parts: parts.map { (shard: $0.0, range: $0.1) }
            )
        }
        self._experts = built
        if stackedTileCount > 0 {
            FileHandle.standardError.write(Data(
                "[JangPressMmapTier] stacked-split: \(stackedTileCount) stacked tensors → \(stackedSplitCount) per-expert sub-ranges (iter 25)\n".utf8))
        }

        // startCold: mark every routed range DONTNEED right after building.
        // This is the post-MLX-pread reclaim — kernel page cache for these
        // bytes is redundant once MLX has copied to Metal buffer.
        if config.startCold {
            for (_, ranges) in built {
                for part in ranges.parts {
                    if let shard = openedShards[part.shard] {
                        shard.advise(.dontNeed, range: part.range)
                    }
                }
            }
        }
    }

    // MARK: - Routing API

    /// Pre-fault the given experts. Triggers `ensureBuilt()` on first
    /// use; thereafter does the corresponding number of
    /// `madvise(MADV_WILLNEED)` syscalls — one per (gate/up/down) part.
    ///
    /// `VMLX_JANGPRESS_FORCE_PREFAULT=1` upgrades the path to a
    /// synchronous read-touch (one byte per page) so cold pages are
    /// guaranteed resident before the matmul reads them. WILLNEED alone
    /// is only a hint and races the matmul on cold MoE bundles —
    /// observable as 0.15 tok/s on Laguna XS.2 at first turn, where the
    /// matmul faulted in pages serially.
    public func acquire(layer: Int, experts: [Int]) {
        ensureBuilt()
        let forcePrefault = JangPressMmapTier.forcePrefaultEnabled
        for e in experts {
            guard let r = self._experts[TileKey(layer: layer, expert: e)] else { continue }
            for part in r.parts {
                if forcePrefault {
                    _shards[part.shard]?.touchRangeResident(range: part.range)
                } else {
                    _shards[part.shard]?.advise(.willNeed, range: part.range)
                }
            }
        }
    }

    /// `VMLX_JANGPRESS_FORCE_PREFAULT` — opt-in synchronous read-touch
    /// in `acquire()` for users who want guaranteed-resident pages
    /// before the matmul. Default off (WILLNEED hint preserves prior
    /// behaviour).
    static var forcePrefaultEnabled: Bool {
        let raw = ProcessInfo.processInfo.environment[
            "VMLX_JANGPRESS_FORCE_PREFAULT"]?.lowercased()
        return raw == "1" || raw == "true" || raw == "on" || raw == "yes"
    }

    /// Mark the given experts as MADV_DONTNEED — kernel can reclaim
    /// their pages under pressure.
    public func release(layer: Int, experts: [Int]) {
        ensureBuilt()
        for e in experts {
            guard let r = self._experts[TileKey(layer: layer, expert: e)] else { continue }
            for part in r.parts {
                _shards[part.shard]?.advise(.dontNeed, range: part.range)
            }
        }
    }

    /// Stronger version of `release` — uses `msync(MS_INVALIDATE)` to
    /// force the kernel to drop pages immediately rather than treating
    /// it as a hint. Use during quiesce-time compaction when you're
    /// confident these experts will stay dormant for ≥30 s. Cost: the
    /// next acquire pays a disk re-fault (~ms per tile).
    public func forceRelease(layer: Int, experts: [Int]) {
        ensureBuilt()
        for e in experts {
            guard let r = self._experts[TileKey(layer: layer, expert: e)] else { continue }
            for part in r.parts {
                _shards[part.shard]?.forceInvalidate(range: part.range)
            }
        }
    }

    /// iter 24: bulk-release every routed range with `madvise(DONTNEED)`.
    /// Designed to run RIGHT AFTER first inference completes — at that
    /// point MLX-swift has finished pread'ing the safetensors into its
    /// own Metal buffers, so the kernel page cache for those same bytes
    /// is pure redundancy. Dropping it gives RAM back for other apps.
    ///
    /// This is the v1 "real win" of JangPress on JANGTQ bundles where
    /// MLX always copies (cf. the v2 storage-hook fork that would let
    /// MLX wrap our mmap directly).
    public func releaseAllRoutedRanges() {
        ensureBuilt()
        for (_, ranges) in _experts {
            for part in ranges.parts {
                _shards[part.shard]?.advise(.dontNeed, range: part.range)
            }
        }
    }

    /// Same as `releaseAllRoutedRanges` but uses msync(MS_INVALIDATE) for
    /// guaranteed reclaim regardless of pressure level.
    public func forceReleaseAllRoutedRanges() {
        ensureBuilt()
        for (_, ranges) in _experts {
            for part in ranges.parts {
                _shards[part.shard]?.forceInvalidate(range: part.range)
            }
        }
    }

    // MARK: - Stats

    public struct Stats: Sendable {
        public var shardCount: Int
        public var expertCount: Int
        public var totalRoutedBytes: UInt64
        public var byLayer: [Int: Int]
        public var built: Bool
    }

    /// Returns a stats snapshot. Will trigger `ensureBuilt()` to populate
    /// real data; for the "is this initialized?" probe use
    /// `snapshotIfBuilt()` instead.
    public func snapshot() -> Stats {
        ensureBuilt()
        var byLayer: [Int: Int] = [:]
        var total: UInt64 = 0
        for (key, r) in _experts {
            byLayer[key.layer, default: 0] += 1
            total += r.totalBytes
        }
        return Stats(
            shardCount: _shards.count,
            expertCount: _experts.count,
            totalRoutedBytes: total,
            byLayer: byLayer,
            built: didBuild)
    }

    /// Probe-only snapshot: does NOT trigger ensureBuilt(). Returns
    /// zeros + built=false until something actually uses the tier.
    public func snapshotIfBuilt() -> Stats {
        buildLock.lock(); defer { buildLock.unlock() }
        if !didBuild {
            return Stats(shardCount: 0, expertCount: 0, totalRoutedBytes: 0, byLayer: [:], built: false)
        }
        var byLayer: [Int: Int] = [:]
        var total: UInt64 = 0
        for (key, r) in _experts {
            byLayer[key.layer, default: 0] += 1
            total += r.totalBytes
        }
        return Stats(
            shardCount: _shards.count,
            expertCount: _experts.count,
            totalRoutedBytes: total,
            byLayer: byLayer,
            built: true)
    }

    // MARK: - Tensor name parsing

    // VL bundles wrap the language tower under various namespace prefixes:
    //   • Plain text: `model.layers.<L>...`
    //   • Holo3 VL outside: `language_model.model.layers.<L>...`
    //   • Qwen3.6 VL inside: `model.language_model.layers.<L>...`
    //   • Plain (some affines): `language_model.layers.<L>...`
    // The shared prefix matches zero or more `model.` / `language_model.`
    // chunks before the trailing `layers.` anchor. This costs one
    // backtrack step per call (cheap) and covers all observed layouts.
    private static let vlPrefix = #"(?:(?:model|language_model)\.)*"#

    // Pattern A — Qwen/GLM/MiniMax fp16 stacked layout:
    //   [<vlPrefix>]layers.<L>.mlp.switch_mlp.<gate|up|down>_proj.weight
    private static let switchMlpRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.weight$"#)

    // Pattern B — Mistral 4 / DSV3.x / Kimi K2 / Ling per-expert layout:
    //   [<vlPrefix>]layers.<L>.mlp.experts.<E>.<gate|up|down>_proj.weight
    //   [<vlPrefix>]layers.<L>.mlp.experts.<E>.<gate|up|down>_proj.tq_packed
    private static let perExpertMlpRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.experts\.(\d+)\.(?:gate|up|down)_proj\.(?:weight|tq_packed|tq_norms)$"#)

    // Pattern C — Laguna / Qwen3.6 JANGTQ stacked:
    //   [<vlPrefix>]layers.<L>.mlp.experts.<gate_up_proj|down_proj>.tq_packed
    private static let jangtqStackedRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)\.tq_packed$"#)

    // Pattern D — JANG_2L / MXFP4 affine stacked:
    //   [<vlPrefix>]layers.<L>.mlp.experts.<gate_up_proj|down_proj>.weight
    private static let affineStackedRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)\.weight$"#)

    // Pattern G — Holo3 / Qwen3.5MoE JANGTQ switch_mlp (per-projection
    // TQ-packed, one stacked tile per layer per projection):
    //   [<vlPrefix>]layers.<L>.mlp.switch_mlp.<gate|up|down>_proj.tq_packed
    private static let switchMlpJangtqRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:tq_packed|tq_norms)$"#)

    // Pattern N — Gemma 4 VLM JANG/SWA MoE stacked:
    //   model.language_model.layers.<L>.switch_mlp.<gate|up|down>_proj.weight
    // Gemma's exported text tower omits the `.mlp.` namespace used by
    // Qwen/Laguna, but it is still a stacked expert bank on axis 0.
    private static let gemmaSwitchMlpRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$"#)

    // Pattern H — MiniMax M2 / M2.7 per-expert JANGTQ:
    //   [<vlPrefix>]layers.<L>.block_sparse_moe.experts.<E>.w[123].tq_packed
    private static let minimaxBlockSparseRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.tq_packed$"#)

    // Pattern I — MiniMax affine JANG (no .tq_packed suffix):
    //   [<vlPrefix>]layers.<L>.block_sparse_moe.experts.<E>.w[123].weight
    private static let minimaxBlockSparseAffineRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.weight$"#)

    // Pattern O — MiniMax M2.7 JANGPressPrestacker overlay:
    //   [<vlPrefix>]layers.<L>.block_sparse_moe.switch_mlp.<gate|up|down>_proj.tq_packed
    // The overlay stacks all experts on axis 0, so page advice must
    // split by shape[0] just like the other stacked switch_mlp layouts.
    private static let minimaxBlockSparseSwitchMlpRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.block_sparse_moe\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$"#)

    // Pattern J — Nemotron Omni / Cascade nemotron_h JANGTQ:
    //   backbone.layers.<L>.mixer.experts.<E>.<gate|up|down>_proj.tq_packed
    // Nvidia uses `backbone.layers` + `mixer` (since hybrid SSM/attn
    // mixer pattern), with the same projection trio as Qwen.
    private static let nemotronMixerRegex = try! NSRegularExpression(
        pattern: #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(?:gate|up|down)_proj\.tq_packed$"#)

    // Pattern K — Nemotron affine variant:
    //   backbone.layers.<L>.mixer.experts.<E>.<gate|up|down>_proj.weight
    private static let nemotronMixerAffineRegex = try! NSRegularExpression(
        pattern: #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(?:gate|up|down)_proj\.weight$"#)

    // Pattern L — Nemotron stacked switch_mlp (one tile per layer):
    //   backbone.layers.<L>.mixer.switch_mlp.<fc1|fc2>.(weight|tq_packed|tq_norms)
    // JANGPressPrestacker rewrites per-expert Nemotron JANGTQ tensors
    // into this `fc1/fc2.tq_*` layout, so the mmap tier must recognize
    // it or the canonical overlay is skipped as "no routed experts".
    // Bias/scale sidecars are tiny and not part of the routed TQ bank.
    private static let nemotronSwitchMlpRegex = try! NSRegularExpression(
        pattern: #"^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.fc[12]\.(?:weight|tq_packed|tq_norms)$"#)

    // Pattern M — Nemotron Cascade-2 affine stacked switch_mlp:
    //   backbone.layers.<L>.mixer.switch_mlp.<gate|up|down>_proj.weight
    private static let nemotronSwitchMlpAffineRegex = try! NSRegularExpression(
        pattern: #"^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.(?:gate|up|down)_proj\.weight$"#)

    // Pattern P — DeepSeek V3/V4 canonical prestacked JANGTQ:
    //   layers.<L>.ffn.switch_mlp.<gate|up|down>_proj.tq_packed
    // New converter bundles may write DeepSeek-family routed experts already
    // stacked in the main shards under `ffn.switch_mlp`; Swift model
    // sanitizers remap that namespace to the module's `mlp.switch_mlp`.
    private static let deepseekFfnSwitchMlpRegex = try! NSRegularExpression(
        pattern: #"^layers\.(\d+)\.ffn\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$"#)

    // Pattern E — DeepSeek V4 per-expert JANGTQ (NEW iter 12).
    // Note the differences from pattern B:
    //   • NO `model.` prefix (DSV4's own naming convention)
    //   • `ffn` instead of `mlp`
    //   • `w1` / `w2` / `w3` instead of gate/up/down_proj
    //   • `.tq_packed` (or `.tq_norms` / `.tq_bits`) suffix
    //
    // Catches both routed AND hash-routed (DSV4 L0-L2) layers since
    // they share the same physical naming — only the router upstream
    // differs. Component H from CACHE-ARCHITECTURE.md.
    //
    //   layers.<L>.ffn.experts.<E>.<w1|w2|w3>.tq_packed
    //   layers.<L>.ffn.experts.<E>.<gate|up|down>_proj.tq_packed
    private static let dsv4PerExpertRegex = try! NSRegularExpression(
        pattern: #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(?:w[123]|(?:gate|up|down)_proj)\.(?:tq_packed|tq_norms)$"#)

    // Pattern F — DeepSeek V4 per-expert affine (e.g. JANG_2L of DSV4):
    //   layers.<L>.ffn.experts.<E>.<w1|w2|w3>.weight
    //   layers.<L>.ffn.experts.<E>.<gate|up|down>_proj.weight
    private static let dsv4PerExpertAffineRegex = try! NSRegularExpression(
        pattern: #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(?:w[123]|(?:gate|up|down)_proj)\.weight$"#)

    /// Result of parsing a routed-expert tensor name.
    /// - perExpert: tensor holds ONE expert (gate_proj/up_proj/down_proj
    ///   per-expert layouts B/E/F/H/I/J/K). Caller indexes by (layer, expert).
    /// - stacked: tensor holds ALL N experts along axis 0 (layouts
    ///   A/C/D/G/L/M/N/O). Caller must look up tensor shape[0] = numExperts
    ///   and split the byte range into N equal sub-ranges (iter 25).
    public enum ParsedTile: Sendable {
        case perExpert(layer: Int, expert: Int)
        case stacked(layer: Int)
    }

    /// Parse a tensor name and return its tile classification, else nil.
    /// Iter 25: replaces the older `(layer, expert)` API with a typed
    /// case so the indexer can split stacked tiles into per-expert
    /// sub-ranges (using the tensor's shape[0] = numExperts) instead
    /// of treating the whole stack as a single synthetic-id-0 tile.
    /// Re-acquire on a stacked layout drops from O(67-304 MB tile faulted)
    /// to O(per-expert sub-range faulted) — same cost as a per-expert
    /// layout's tile.
    public static func parseRoutedTile(_ name: String) -> ParsedTile? {
        let range = NSRange(name.startIndex..<name.endIndex, in: name)

        // Per-expert patterns (B + E + F + H + I + J + K) FIRST — they
        // have numeric expert ids in path so we want fine-grained per-
        // expert tracking, not the synthetic id 0 of the stacked layouts.
        for perExpertRegex in [perExpertMlpRegex, dsv4PerExpertRegex, dsv4PerExpertAffineRegex,
                               minimaxBlockSparseRegex, minimaxBlockSparseAffineRegex,
                               nemotronMixerRegex, nemotronMixerAffineRegex] {
            if let m = perExpertRegex.firstMatch(in: name, range: range), m.numberOfRanges >= 3 {
                guard
                    let lr = Range(m.range(at: 1), in: name),
                    let er = Range(m.range(at: 2), in: name),
                    let layer = Int(name[lr]),
                    let expert = Int(name[er])
                else { return nil }
                return .perExpert(layer: layer, expert: expert)
            }
        }

        // Stacked patterns (A + C + D + G + L + M + N + O). One tensor =
        // ALL experts of that layer along axis 0.
        for regex in [switchMlpRegex, jangtqStackedRegex, affineStackedRegex,
                      switchMlpJangtqRegex, gemmaSwitchMlpRegex,
                      minimaxBlockSparseSwitchMlpRegex,
                      nemotronSwitchMlpRegex, nemotronSwitchMlpAffineRegex,
                      deepseekFfnSwitchMlpRegex] {
            if let m = regex.firstMatch(in: name, range: range), m.numberOfRanges >= 2 {
                guard
                    let lr = Range(m.range(at: 1), in: name),
                    let layer = Int(name[lr])
                else { continue }
                return .stacked(layer: layer)
            }
        }

        return nil
    }

    /// Legacy thunk for callers that don't care about stacked-vs-per-expert
    /// distinction. Returns synthetic expert=0 for stacked layouts —
    /// behavior matches the pre-iter-25 contract.
    public static func parseRoutedExpertName(_ name: String) -> (layer: Int, expert: Int)? {
        switch parseRoutedTile(name) {
        case .perExpert(let l, let e): return (l, e)
        case .stacked(let l): return (l, 0)
        case .none: return nil
        }
    }
}
