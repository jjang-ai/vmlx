// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressEmbedTier — page-level Zipfian compression for the
// embedding table and lm_head. Component F from
// `Cache/CACHE-ARCHITECTURE.md`.
//
// MOTIVATION
// ==========
// `model.embed_tokens.weight` and `model.lm_head.weight` are the two
// vocab-sized matrices in the model. Per decode step we touch:
//
//   • embed_tokens: ONE row (the input token id)
//   • lm_head:      ONE row when greedy / sampling; the entire matrix
//                   for argmax over vocab. Most production sampling
//                   uses temperature/top-p which DOES touch every row,
//                   but the post-softmax distribution is Zipfian — a
//                   handful of rows dominate the probability mass.
//
// On a 128 K vocab × 4096 hidden bf16 model that's ~1 GB per matrix.
// If we identify the top-1 % most-frequent token rows (~1.3 K rows),
// pin them MADV_WILLNEED, and let the rest be MADV_DONTNEED, the
// kernel can keep the auxiliary mmap pages evictable. This does not
// replace MLX's canonical heap/Metal copy of embed/lm_head weights, so
// it must be reported as page-cache pressure behavior rather than
// guaranteed resident model-RAM savings.
//
// COMPATIBILITY WITH JANGPRESS ROUTED-EXPERT TIER
// ===============================================
// This tier is independent of `JangPressMmapTier`. They both use
// `JangPressShard` for the underlying mmap+madvise primitive but
// don't share state. A bundle can have both active simultaneously:
//
//   JangPressMmapTier   — covers routed-expert tiles
//   JangPressEmbedTier   — covers embed_tokens + lm_head rows
//   (JangPressMachCache for .mach backend, mutually exclusive
//    with JangPressMmapTier per Engine.LoadOptions selection)
//
// This module is **scaffold-only** in iter 8 — the public API + tests
// land but it's not yet integrated into the engine. Once the routed-
// expert path proves out on a real bundle the same wiring pattern
// extends here.

import Foundation

// MARK: - Errors

public enum JangPressEmbedError: Error, CustomStringConvertible {
    case missingEmbeddingTensor(URL)
    case rowSizeUnknown(tensor: String)

    public var description: String {
        switch self {
        case .missingEmbeddingTensor(let url):
            return "no embed_tokens.weight or lm_head.weight in \(url.lastPathComponent)"
        case .rowSizeUnknown(let t):
            return "cannot infer row size for \(t)"
        }
    }
}

// MARK: - Config

public struct JangPressEmbedConfig: Sendable {
    public let bundleURL: URL

    /// 0..100 — fraction of vocab kept MADV_WILLNEED. The remainder
    /// is MADV_DONTNEED-eligible. Default 1 % — Zipfian distributions
    /// concentrate most activations in the top ~1 % of vocab.
    public var hotPercent: Int

    /// If true, scan only `model.embed_tokens.weight`. Skips
    /// `model.lm_head.weight` (e.g. for tied embeddings where it
    /// doesn't exist as a separate tensor).
    public var skipLMHead: Bool

    public init(bundleURL: URL, hotPercent: Int = 1, skipLMHead: Bool = false) {
        self.bundleURL = bundleURL
        self.hotPercent = max(0, min(100, hotPercent))
        self.skipLMHead = skipLMHead
    }
}

// MARK: - Tier

public final class JangPressEmbedTier: @unchecked Sendable {

    public let config: JangPressEmbedConfig

    /// Shards that hold embed_tokens and/or lm_head.
    /// Lazily populated by `ensureBuilt()` on first use.
    public var shards: [URL: JangPressShard] {
        ensureBuilt()
        return _shards
    }
    private var _shards: [URL: JangPressShard] = [:]

    /// Per-tensor metadata we need at acquire/release time.
    public struct TensorView: Sendable {
        public let name: String
        public let shard: URL
        public let dtypeBytes: Int        // bytes per scalar (bf16=2, fp32=4)
        public let vocabSize: Int
        public let hiddenSize: Int
        public let dataOffset: UInt64     // absolute byte offset of row 0 in shard file
    }
    public var embedTokens: TensorView? {
        ensureBuilt()
        return _embedTokens
    }
    private var _embedTokens: TensorView?

    public var lmHead: TensorView? {
        ensureBuilt()
        return _lmHead
    }
    private var _lmHead: TensorView?

    /// Per-token-id activation count. Updated by `recordTokenActivity`
    /// during the first ~1000 decode steps; the warm-up window builds
    /// the Zipfian profile.
    private var tokenFrequency: [Int: UInt64] = [:]
    private var observedSamples: UInt64 = 0

    /// iter 24: build state machine. Init is now O(1) — actual file I/O
    /// is deferred to `ensureBuilt()` which fires on first use.
    private let buildLock = NSLock()
    private var didBuild = false

    public init(config: JangPressEmbedConfig) throws {
        self.config = config
        // No work in init. ensureBuilt() does sniff + open + tensor lookup.
    }

    private func ensureBuilt() {
        if didBuild { return }
        buildLock.lock(); defer { buildLock.unlock() }
        if didBuild { return }
        defer { didBuild = true }

        // Embed + LM head tensor name candidates. Different model
        // families use different canonical names; we accept all common
        // variants.
        let embedCandidates: Set<String> = [
            "model.embed_tokens.weight",     // Llama, Mistral, Qwen, Gemma, GLM, etc.
            "embed_tokens.weight",           // some bundles drop the model. prefix
            "embed.weight",                  // DeepSeek-V4
            "language_model.embed_tokens.weight",  // VL wrappers
            "model.embed.weight",            // edge case
        ]
        let headCandidates: Set<String> = [
            "lm_head.weight",                // most architectures
            "head.weight",                   // DeepSeek-V4
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ]
        let allCandidates = embedCandidates.union(headCandidates)

        // iter 20: header-sniff each shard FIRST. Only mmap the shards
        // that actually contain embed_tokens or lm_head — typically
        // just one shard out of 86 on DSV4. Saves ~37 seconds at load
        // by eliminating 85 redundant JangPressShard.init mmap+parse
        // cycles.
        let fm = FileManager.default
        let shardURLs = (try? fm.contentsOfDirectory(
            at: config.bundleURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
        )?.filter {
            $0.pathExtension == "safetensors"
                && !JangLoader.shouldSkipModelSafetensorsFile($0)
        } ?? []

        var skippedCount = 0
        for url in shardURLs {
            guard let names = JangPressShard.sniffTensorNames(at: url) else {
                do {
                    self._shards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressEmbedTier] sniff+open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
                continue
            }
            let hasEmbedOrHead = names.contains(where: { allCandidates.contains($0) })
            if hasEmbedOrHead {
                do {
                    self._shards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressEmbedTier] open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
            } else {
                skippedCount += 1
            }
        }
        if skippedCount > 0 {
            FileHandle.standardError.write(Data(
                "[JangPressEmbedTier] lazy-built: sniffed \(shardURLs.count) shards, mmap'd \(self._shards.count), skipped \(skippedCount) (no embed/lm_head)\n".utf8))
        }

        // 2. Locate the embedding + LM head tensors among the (small)
        // set of shards we actually opened. Note: we write to the
        // underscored backing fields (`_embedTokens` / `_lmHead`)
        // because the public computed properties trigger ensureBuilt()
        // (would re-enter the build lock).
        for (url, shard) in _shards {
            if self._embedTokens == nil {
                for name in embedCandidates {
                    if let d = shard.descriptor(for: name) {
                        self._embedTokens = Self.makeView(name: name, shard: url, descriptor: d)
                        break
                    }
                }
            }
            if self._lmHead == nil, !config.skipLMHead {
                for name in headCandidates {
                    if let d = shard.descriptor(for: name) {
                        self._lmHead = Self.makeView(name: name, shard: url, descriptor: d)
                        break
                    }
                }
            }
        }
    }

    /// Compute row-size from descriptor. Defaults to bf16 if dtype
    /// can't be parsed.
    private static func makeView(
        name: String, shard: URL, descriptor: TensorDescriptor
    ) -> TensorView? {
        guard descriptor.shape.count == 2 else { return nil }
        let vocab = descriptor.shape[0]
        let hidden = descriptor.shape[1]
        let dtypeBytes: Int
        switch descriptor.dtype {
        case "F32", "I32", "U32": dtypeBytes = 4
        case "F16", "BF16", "I16", "U16": dtypeBytes = 2
        case "I8", "U8", "F8_E4M3", "F8_E5M2": dtypeBytes = 1
        default: dtypeBytes = 2  // assume bf16 — most JANGTQ embeds
        }
        return TensorView(
            name: name, shard: shard, dtypeBytes: dtypeBytes,
            vocabSize: vocab, hiddenSize: hidden,
            dataOffset: descriptor.dataOffset)
    }

    // MARK: - Routing-time API

    /// Per-decode-step hook. Records token activity for the warm-up
    /// profile. **RAM/CPU safety (Eric directive 2026-05-04):**
    ///
    /// On long prompts (e.g. 32k tokens) the naive per-token dict
    /// update runs 32k inserts and triggers ~17 dictionary resizes
    /// (each resize copies the whole bucket array). Caps:
    ///
    ///  • **Stride sampling** — observe every Nth token instead of
    ///    every token. The Zipfian frequency profile is preserved
    ///    (skipping deterministic positions doesn't bias which IDs
    ///    are hot). Stride defaults to 8; 32k-token prompts now do
    ///    4k inserts instead of 32k.
    ///  • **Distinct-token cap** — once `tokenFrequency.count`
    ///    crosses `maxDistinctTokens` (default 8192), we stop
    ///    accepting NEW IDs. Existing entries still tick their
    ///    counter. Caps the dict at 8192 × ~32 B ≈ 256 KB
    ///    regardless of prompt diversity.
    ///
    /// Both knobs are env-tunable for benchmarking
    /// (`VMLX_JANGPRESS_TOKEN_STRIDE`, `VMLX_JANGPRESS_MAX_DISTINCT`).
    public func recordTokenActivity(_ tokenIds: [Int]) {
        let cap = Self.maxDistinctTokens
        // Stride sampling kicks in ONLY for long token arrays — short
        // arrays (per-step routing observations, small prompts) keep
        // 1:1 semantics. Threshold: 256 tokens. Below that, every
        // element is recorded; above, we sample every Nth.
        let stride: Int
        if tokenIds.count >= Self.strideActivationThreshold {
            stride = Self.tokenStride
        } else {
            stride = 1
        }
        var idx = 0
        while idx < tokenIds.count {
            let t = tokenIds[idx]
            if let existing = tokenFrequency[t] {
                tokenFrequency[t] = existing &+ 1
                observedSamples &+= 1
            } else if tokenFrequency.count < cap {
                tokenFrequency[t] = 1
                observedSamples &+= 1
            }
            // else: dict is at cap and this is a new ID — drop it.
            // The frequency profile we already have is the load-
            // bearing signal; new tail-IDs that arrive after cap are
            // by definition rare (or we'd already have them).
            idx += stride
        }
    }

    /// Token-array length above which stride sampling kicks in.
    /// Override via `VMLX_JANGPRESS_STRIDE_THRESHOLD`. Default 256.
    private static var strideActivationThreshold: Int {
        if let raw = ProcessInfo.processInfo.environment["VMLX_JANGPRESS_STRIDE_THRESHOLD"],
           let parsed = Int(raw), parsed > 0
        {
            return parsed
        }
        return 256
    }

    /// Stride for `recordTokenActivity` sampling. Override via
    /// `VMLX_JANGPRESS_TOKEN_STRIDE`. Min 1 (every token), default 8.
    private static var tokenStride: Int {
        if let raw = ProcessInfo.processInfo.environment["VMLX_JANGPRESS_TOKEN_STRIDE"],
           let parsed = Int(raw), parsed > 0
        {
            return parsed
        }
        return 8
    }

    /// Hard cap on distinct-token entries in `tokenFrequency`. Override
    /// via `VMLX_JANGPRESS_MAX_DISTINCT`. Default 8192 ≈ 256 KB worst
    /// case for the dict storage.
    private static var maxDistinctTokens: Int {
        if let raw = ProcessInfo.processInfo.environment["VMLX_JANGPRESS_MAX_DISTINCT"],
           let parsed = Int(raw), parsed > 0
        {
            return parsed
        }
        return 8192
    }

    /// After warm-up, set advise on the bottom (1 - hotPercent)% of
    /// vocab rows to MADV_DONTNEED. The hottest rows are kept
    /// MADV_WILLNEED. Idempotent — safe to call multiple times.
    public func applyZipfianAdvise() {
        guard !tokenFrequency.isEmpty else { return }
        let sorted = tokenFrequency.sorted { $0.value > $1.value }
        let total = sorted.count
        guard let embed = embedTokens else { return }

        let hotCount = max(1, Int(Double(embed.vocabSize) * Double(config.hotPercent) / 100.0))
        let hotIds = Set(sorted.prefix(hotCount).map { $0.key })

        // Mark hot rows WILLNEED, the rest DONTNEED, on each tensor view.
        for view in [embed, lmHead].compactMap({ $0 }) {
            guard let shard = shards[view.shard] else { continue }
            let rowBytes = UInt64(view.hiddenSize * view.dtypeBytes)

            // For each row in vocab, advise based on hot/cold status.
            // We could batch into runs of consecutive cold rows for
            // fewer madvise calls — TODO performance.
            for rowId in 0..<view.vocabSize {
                let start = view.dataOffset + UInt64(rowId) * rowBytes
                let end = start + rowBytes
                let advice: JangPressAdvice = hotIds.contains(rowId) ? .willNeed : .dontNeed
                shard.advise(advice, range: start..<end)
            }
        }
        _ = total
    }

    // MARK: - Stats

    public struct Stats: Sendable {
        public var hasEmbedTokens: Bool
        public var hasLMHead: Bool
        public var vocabSize: Int
        public var hiddenSize: Int
        public var observedTokenSamples: UInt64
        public var distinctTokensSeen: Int
        public var hotPercent: Int
    }

    public func snapshot() -> Stats {
        Stats(
            hasEmbedTokens: embedTokens != nil,
            hasLMHead: lmHead != nil,
            vocabSize: embedTokens?.vocabSize ?? 0,
            hiddenSize: embedTokens?.hiddenSize ?? 0,
            observedTokenSamples: observedSamples,
            distinctTokensSeen: tokenFrequency.count,
            hotPercent: config.hotPercent)
    }

    /// iter 24: probe-only snapshot that doesn't trigger ensureBuilt().
    /// Returns hasEmbedTokens=false / zeros until something actually
    /// uses the tier (so /v1/cache/jangpress doesn't force a full
    /// sniff pass just for stats).
    public func snapshotIfBuilt() -> Stats {
        buildLock.lock(); defer { buildLock.unlock() }
        if !didBuild {
            return Stats(
                hasEmbedTokens: false, hasLMHead: false,
                vocabSize: 0, hiddenSize: 0,
                observedTokenSamples: observedSamples,
                distinctTokensSeen: tokenFrequency.count,
                hotPercent: config.hotPercent)
        }
        return Stats(
            hasEmbedTokens: _embedTokens != nil,
            hasLMHead: _lmHead != nil,
            vocabSize: _embedTokens?.vocabSize ?? 0,
            hiddenSize: _embedTokens?.hiddenSize ?? 0,
            observedTokenSamples: observedSamples,
            distinctTokensSeen: tokenFrequency.count,
            hotPercent: config.hotPercent)
    }
}
