// vMLX Examples — RuntimeShared
//
// Common scaffolding for the per-model runtime examples
// (DSV4-Flash, Laguna, Mistral 3.5).
//
// Each per-model example sets up `Engine.LoadOptions` with the same
// default stack and then drives one or more chat turns through the
// engine, printing reasoning + content + tool calls as they stream.
//
// Defaults below mirror what `vmlxctl serve` ships with so that the
// example output matches production behavior:
//
//   • enableTurboQuant      — KV cache compresses prefix to TurboQuant
//                             encoded form once prefill ends; window stays
//                             float for low-latency token append.
//
//   • enableJANG            — JANG repack on load (codebook + Hadamard
//                             rotation + bookend bits). Required for any
//                             JANGTQ bundle.
//
//   • enablePrefixCache     — RAM-resident prefix cache (multi-turn).
//
//   • enableDiskCache       — L1 disk cache (whole-prefix shards).
//
//   • enableBlockDiskCache  — optional block-level persistent L2. It stores
//                             paged KV blocks as safetensors payloads and
//                             materializes hits back into pinned CacheBlock
//                             instances. Default OFF because it is an
//                             additional disk-throughput tradeoff.
//
//   • enableSSMCompanion    — SSM companion cache for hybrid (Mamba +
//                             attention) models, e.g. Nemotron-H.
//
//   • enableSSMReDerive     — Async SSM warm path. After a thinking-mode
//                             turn ends, `reDeriveSSMStates` runs ONE
//                             prompt-only forward pass to build a
//                             cache-key-aligned SSM state for the next
//                             turn. Costs ~2 s on a 2K prompt; without it
//                             every subsequent turn re-prefills from scratch.
//
// The live cache cascade is
// `Paged/Memory → prompt-level DiskCache → block-level BlockDiskCache → cold`
// orchestrated by `CacheCoordinator`.
//
// JANGTQ runtime is on the critical path: `JangLoader` parses
// `jang_config.json`, swaps Linear modules to JANGTQDenseLinear, and
// reads `jangtq_runtime.safetensors` (the deterministic codebook +
// Hadamard sign sidecar). Without that sidecar the loader hard-errors.

import Foundation
import vMLXEngine
import vMLXLMCommon

// MARK: - Standard LoadOptions for examples

public enum RuntimeShared {

    /// LoadOptions tuned for the production default chat path.
    /// Cache stack ON, TurboQuant ON unless compile-first suppresses it,
    /// JANG repack ON, SSM re-derive ON.
    public static func makeLoadOptions(
        bundle: URL,
        cacheDir: URL? = nil,
        kvCacheBits: Int = 4,
        kvCacheGroupSize: Int = 64,
        slidingWindowMode: String = "auto"
    ) -> Engine.LoadOptions {
        var opt = Engine.LoadOptions(modelPath: bundle)

        // KV cache: TurboQuant on (default true). bits/group set explicitly
        // so the example output is deterministic across machines.
        opt.enableTurboQuant = true
        opt.turboQuantBits = kvCacheBits
        opt.kvCacheQuantization = "tq"
        opt.kvCacheGroupSize = kvCacheGroupSize

        // JANGTQ runtime ON — required for any `weight_format == "mxtq"` bundle.
        opt.enableJANG = true

        // Cache cascade: Memory → L1 → L2.
        opt.enableMemoryCache = true
        opt.enablePrefixCache = true
        opt.enableDiskCache = true
        opt.enableBlockDiskCache = false
        if let cacheDir {
            opt.diskCacheDir = cacheDir.path
            opt.blockDiskCacheDir = cacheDir.appendingPathComponent("blocks").path
        }

        // SSM hybrid: re-derive ON. Costs 1 prompt-only prefill at turn end,
        // saves a full prefill on the *next* turn. No-op for non-hybrid models.
        opt.enableSSMCompanion = true
        opt.enableSSMReDerive = true

        // Production default — honor the model's config. Compile-first SWA
        // families (Laguna/Gemma/Mistral) can still take the bounded fast path
        // under `auto`; explicit `long` is a slower full-context escape hatch.
        opt.slidingWindowMode = slidingWindowMode

        return opt
    }

    /// Print the runtime knobs that this example will run with so the
    /// reader can confirm the cache stack actually engaged.
    public static func reportLoadOptions(_ o: Engine.LoadOptions) {
        print("[runtime] knobs:")
        print("  TurboQuant       = \(o.enableTurboQuant)  (bits=\(o.turboQuantBits) gs=\(o.kvCacheGroupSize))")
        print("  JANG repack      = \(o.enableJANG)")
        print("  PrefixCache      = \(o.enablePrefixCache)")
        print("  DiskCache (L1)   = \(o.enableDiskCache)  dir=\(o.diskCacheDir.isEmpty ? "<default>" : o.diskCacheDir)")
        print("  BlockDiskCache (L2) = \(o.enableBlockDiskCache)  dir=\(o.blockDiskCacheDir.isEmpty ? "<default>" : o.blockDiskCacheDir)")
        print("  SSMCompanion     = \(o.enableSSMCompanion)")
        print("  SSMReDerive      = \(o.enableSSMReDerive)")
        print("  SlidingWindow    = \(o.slidingWindowMode) (size=\(o.slidingWindowSize))")
    }

    /// Drain a load stream to completion. Throws on `.failed`.
    public static func awaitLoad(_ engine: Engine, options: Engine.LoadOptions) async throws {
        let stream = await engine.load(options)
        for try await event in stream {
            if case .failed(let reason) = event {
                fatalError("[runtime] load failed: \(reason)")
            }
        }
    }

    /// One-line cache-stat printer — call after every turn to see L1/L2 hits.
    public static func reportCacheStats(_ engine: Engine) async {
        do {
            let stats = try await engine.cacheStats()
            // Minimal pretty-print of the parts we care about.
            if let paged = stats["paged"] as? [String: Any] {
                let hit = paged["hitCount"] ?? 0
                let miss = paged["missCount"] ?? 0
                let rate = paged["hitRate"] as? Double ?? 0.0
                print("[cache.paged]   hit=\(hit) miss=\(miss) rate=\(String(format: "%.2f", rate))")
            }
            if let l1 = stats["disk"] as? [String: Any] {
                print("[cache.disk]    \(l1)")
            }
            if let l2 = stats["blockDisk"] as? [String: Any] {
                print("[cache.blockDisk] \(l2)")
            }
        } catch {
            print("[cache] stats error: \(error)")
        }
    }

    /// Convenience constructors for chat messages.
    public static func userMsg(_ text: String) -> ChatRequest.Message {
        ChatRequest.Message(role: "user", content: .string(text))
    }
    public static func systemMsg(_ text: String) -> ChatRequest.Message {
        ChatRequest.Message(role: "system", content: .string(text))
    }
    public static func assistantMsg(_ text: String) -> ChatRequest.Message {
        ChatRequest.Message(role: "assistant", content: .string(text))
    }

    /// Build a ChatRequest with sensible defaults. The `model` field is
    /// only used by HTTP callers; in-process we ignore it.
    public static func makeRequest(
        _ messages: [ChatRequest.Message],
        maxTokens: Int = 256,
        temperature: Double = 0.0,
        topP: Double? = nil,
        enableThinking: Bool = false,
        reasoningEffort: String? = nil,
        thinkingBudget: Int? = nil,
        tools: [ChatRequest.Tool]? = nil
    ) -> ChatRequest {
        ChatRequest(
            model: "in-process",
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            tools: tools,
            includeReasoning: true,
            thinkingBudget: thinkingBudget
        )
    }

    /// Tag set used by `assertNoLeak` — these markers must never appear in
    /// `content`. Covers `<think>` family, harmony channel format, DSML,
    /// and GLM-style `<tool_call>` markup.
    public static let leakTags: [String] = [
        "<think>", "</think>",
        "<thought>", "</thought>",
        "<|channel>", "<|channel|>", "<channel|>", "|thought",
        "<|start|>", "<|message|>",
        "｜DSML｜", "</DSML>",
        "<tool_call>",
    ]

    public static func assertNoLeak(_ content: String, label: String = "content") {
        for tag in leakTags where content.contains(tag) {
            fatalError("[\(label)] LEAKED tag \(tag) in: \(content.prefix(120))")
        }
    }

    public struct DrainResult: Sendable {
        public var reasoning: String
        public var content: String
        public var tools: [String]
        public var finalUsage: StreamChunk.Usage?
        public var lastUsage: StreamChunk.Usage?
        public var partialUsageCount: Int
        public var contentChunkCount: Int
        public var reasoningChunkCount: Int
        public var finishReason: String?

        public init(
            reasoning: String,
            content: String,
            tools: [String],
            finalUsage: StreamChunk.Usage?,
            lastUsage: StreamChunk.Usage?,
            partialUsageCount: Int,
            contentChunkCount: Int,
            reasoningChunkCount: Int,
            finishReason: String?
        ) {
            self.reasoning = reasoning
            self.content = content
            self.tools = tools
            self.finalUsage = finalUsage
            self.lastUsage = lastUsage
            self.partialUsageCount = partialUsageCount
            self.contentChunkCount = contentChunkCount
            self.reasoningChunkCount = reasoningChunkCount
            self.finishReason = finishReason
        }
    }

    /// Drain a stream printing each event into reasoning/content/toolcalls
    /// counters. Returns full usage/timing metadata for production probes.
    public static func drainStreamDetailed(
        _ engine: Engine,
        _ req: ChatRequest,
        printContent: Bool = true,
        printReasoningTick: Bool = false
    ) async throws -> DrainResult {
        var reasoning = ""
        var content = ""
        var tools: [String] = []
        var finalUsage: StreamChunk.Usage? = nil
        var lastUsage: StreamChunk.Usage? = nil
        var partialUsageCount = 0
        var contentChunkCount = 0
        var reasoningChunkCount = 0
        var finishReason: String? = nil

        let stream = await engine.stream(request: req)
        for try await chunk in stream {
            if let r = chunk.reasoning, !r.isEmpty {
                reasoning += r
                reasoningChunkCount += 1
                if printReasoningTick { print("R", terminator: "") }
            }
            if let c = chunk.content, !c.isEmpty {
                content += c
                contentChunkCount += 1
                if printContent { print(c, terminator: "") }
            }
            if let calls = chunk.toolCalls, !calls.isEmpty {
                for call in calls {
                    tools.append("\(call.function.name)(\(call.function.arguments))")
                }
            }
            if let usage = chunk.usage {
                lastUsage = usage
                if usage.isPartial {
                    partialUsageCount += 1
                } else {
                    finalUsage = usage
                }
            }
            if let reason = chunk.finishReason {
                finishReason = reason
            }
        }
        if printContent || printReasoningTick { print("") }
        return DrainResult(
            reasoning: reasoning,
            content: content,
            tools: tools,
            finalUsage: finalUsage,
            lastUsage: lastUsage,
            partialUsageCount: partialUsageCount,
            contentChunkCount: contentChunkCount,
            reasoningChunkCount: reasoningChunkCount,
            finishReason: finishReason)
    }

    /// Drain a stream printing each event into reasoning/content/toolcalls
    /// counters. Returns final (reasoning, content, toolNames).
    public static func drainStream(
        _ engine: Engine,
        _ req: ChatRequest,
        printContent: Bool = true,
        printReasoningTick: Bool = false
    ) async throws -> (reasoning: String, content: String, tools: [String]) {
        let result = try await drainStreamDetailed(
            engine, req,
            printContent: printContent,
            printReasoningTick: printReasoningTick)
        return (result.reasoning, result.content, result.tools)
    }
}
