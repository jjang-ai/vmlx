// SPDX-License-Identifier: Apache-2.0
//
// Settings type hierarchy for the 4-tier resolver:
//   global defaults → session config → chat override → request override
//
// Field-name parity references:
//
//   Python CLI flags:
//     vmlx_engine/cli.py
//       serve_parser block at line 881 (full list of ~70 --flags)
//
//   Electron SQLite schema (for forward-compat import from Electron vMLX):
//     panel/src/main/database.ts
//       CREATE TABLE sessions       (line 259)
//       CREATE TABLE chats          (line 203)
//       CREATE TABLE chat_overrides (line 228, migrated cols 350-429)
//       CREATE TABLE settings       (line 245)
//
// Every field in the Python cli.py `serve` subparser has a home in one of the
// four tiers. Same field names where reasonable. A snake_case alias list is
// provided via CodingKeys so the JSON blobs round-trip cleanly with the
// Electron column names for any future migration tool.
//
// Sendability: every struct here is a pure-value `Sendable`, no reference
// fields. Safe to pass across actor boundaries.

import Foundation

// MARK: - Shared enums / helper types

public enum SettingsTier: String, Sendable, Codable {
    case global
    case session
    case chat
    case request
    case builtin  // compiled-in default when nothing else populated a field
}

public struct MCPServer: Codable, Sendable, Equatable, Hashable {
    public var name: String
    public var command: String
    public var args: [String]
    public var env: [String: String]
    public var enabled: Bool
    public init(name: String, command: String, args: [String] = [],
                env: [String: String] = [:], enabled: Bool = true) {
        self.name = name; self.command = command; self.args = args
        self.env = env; self.enabled = enabled
    }
}

public enum EngineKindCodable: String, Codable, Sendable {
    case simple
    case batched
}

// MARK: - GlobalSettings (tier 1 — every field has a value)

/// Every field here is non-optional: this is the fallback source-of-truth.
/// Mirrors the full `vmlx-engine serve` CLI surface + a few vMLX.app-only
/// preferences (idle thresholds, MCP servers, UI defaults).
public struct GlobalSettings: Codable, Sendable, Equatable {

    // MARK: Server binding
    public var defaultHost: String = "127.0.0.1"   // cli.py --host
    public var defaultPort: Int = 8000             // cli.py --port
    public var defaultLAN: Bool = false            // convenience toggle: true → 0.0.0.0
    public var defaultLogLevel: String = "info"    // cli.py --log-level
    public var allowedOrigins: [String] = ["*"]    // cli.py --allowed-origins / CORS

    // MARK: Gateway (multi-engine multiplexer, UI-9)
    /// When true, vMLX binds an additional Hummingbird listener that
    /// dispatches /v1/chat/completions etc. by ChatRequest.model field
    /// to the matching session's Engine. Per-session listeners keep
    /// running unchanged. Off by default — opt in from Tray → Server.
    public var gatewayEnabled: Bool = false
    public var gatewayPort: Int = 8080
    public var gatewayLAN: Bool = false

    // MARK: Engine load — defaults mirror `vmlx_engine/cli.py` except
    // two Mac-specific tunings noted below.
    public var engineKind: EngineKindCodable = .batched
    // Mac override: single-user laptop doesn't benefit from 256-way batch
    // concurrency (Python server default); 5 is the sweet spot for 1-user chat.
    public var maxNumSeqs: Int = 5                 // cli.py default=256 (server), 5 (vMLX Mac)
    // Python engine has separate prefill_batch_size (how many prompts to
    // chunk together during prefill) and completion_batch_size (how many
    // requests to decode in parallel). Swift's BatchEngine collapses both
    // into `maxNumSeqs` (max concurrent sequences) + `prefillStepSize`
    // (tokens per forward-pass step). The separate Python knobs would be
    // orphan fields in Swift (set but never read), so they're intentionally
    // absent. Use `maxNumSeqs` for request concurrency and `prefillStepSize`
    // for per-step prompt chunking.
    // Mac override: 1024 prefill step keeps first-token latency low on
    // long prompts where 2048 would stall the first render tick.
    public var prefillStepSize: Int = 1024         // cli.py default=2048 (server), 1024 (vMLX Mac)
    public var continuousBatching: Bool = true    // cli.py --continuous-batching
    public var streamInterval: Int = 1            // cli.py --stream-interval

    // Prefix cache
    public var enablePrefixCache: Bool = true     // cli.py --enable-prefix-cache / --disable-prefix-cache
    public var prefixCacheSize: Int = 100         // cli.py --prefix-cache-size
    public var prefixCacheMaxBytes: Int64 = 0     // cli.py --prefix-cache-max-bytes (0 = unlimited)
    public var cacheMemoryMB: Int = 0             // cli.py --cache-memory-mb (0 = auto)
    public var memoryAwareCache: Bool = true      // inverse of cli.py --no-memory-aware-cache

    // Paged cache
    public var usePagedCache: Bool = true         // cli.py --use-paged-cache
    public var pagedCacheBlockSize: Int = 64      // cli.py --paged-cache-block-size
    public var maxCacheBlocks: Int = 1000         // cli.py --max-cache-blocks default=1000

    /// Hard ceiling on prompt tokens a single request may submit.
    /// When a chat template renders more than this many tokens, the
    /// route handler rejects the request with a 400 BEFORE entering
    /// prefill — which is the only way to guard against MLX Metal
    /// `fatalError` on OOM (Swift `try/catch` can't intercept a
    /// fatalError, only a thrown error). Set to 256k by default so
    /// most legitimate long-context requests pass through; users
    /// running 1M-context Minimax can raise it.
    ///
    /// Set to 0 to disable the ceiling entirely (for testing; not
    /// recommended in production).
    public var maxPromptTokens: Int = 262_144

    // KV cache quantization. `none | q4 | q8 | turboquant`. Default is
    // `none` as of perf audit 2026-04-16 — the previous `turboquant`
    // default carried a 40-96% decode-speed tax on MoE / hybrid models:
    // Nemotron-Cascade-2-30B-A3B 2.4 → 59.8 tok/s (25× speedup),
    // Gemma-4-26B-A4B 34.9 → 49.0 tok/s (+40%), Qwen3.5-9B 69.5 → 78.3
    // tok/s (+13%). The TQ compress+dequant cycle per decode step on the
    // attention-half of layers dominated generation time. Users who need
    // the 4× KV memory headroom for long contexts can opt back in via
    // settings, and JANG models with a calibrated `turboquant` block in
    // `jang_config.json` still auto-activate TQ via the
    // `loadedJangConfig?.turboquant` path in `Stream.buildGenerateParameters`.
    public var kvCacheQuantization: String = "none"  // cli.py --kv-cache-quantization: none|q4|q8|turboquant
    public var kvCacheGroupSize: Int = 64            // cli.py --kv-cache-group-size

    // Disk caches.
    //
    // L2 prompt-level disk cache is **default-on** as of 2026-04-14.
    // The v2 unified TQDiskSerializer format handles every supported
    // cache kind (plain KVCacheSimple, QuantizedKVCache, TurboQuantKVCache,
    // MambaCache, the Nemotron-H / Qwen3.5-A3B / Jamba / FalconH1 hybrid
    // mix, and VL JANG) and round-trip is live-verified on the 2026-04-13
    // cross-matrix. With TurboQuant also default-on, the on-disk payload
    // is ~26× smaller than raw float16 so the 10 GB default budget covers
    // many multi-turn sessions without the user doing anything.
    //
    // Users can still disable via the Server settings panel toggle or
    // `--disable-disk-cache` CLI flag.
    public var enableDiskCache: Bool = true       // cli.py --enable-disk-cache / --disable-disk-cache
    public var diskCacheDir: String = ""          // cli.py --disk-cache-dir
    public var diskCacheMaxGB: Double = 10.0      // cli.py --disk-cache-max-gb
    public var enableBlockDiskCache: Bool = false // cli.py --enable-block-disk-cache
    public var blockDiskCacheDir: String = ""     // cli.py --block-disk-cache-dir
    public var blockDiskCacheMaxGB: Double = 10.0 // cli.py --block-disk-cache-max-gb

    // L1.5 byte-budgeted memory cache (MemoryAwarePrefixCache). Sits
    // between the paged L1 and disk L2, storing whole-prompt KV payloads
    // with LRU + memory-pressure eviction. Default ON — the combined
    // L1/L1.5/L2 stack gives the best multi-turn cache hit rate and
    // the memory pressure monitor drops entries before vmsignal fires,
    // so there is no OOM risk on constrained-RAM hardware.
    public var enableMemoryCache: Bool = true
    public var memoryCachePercent: Double = 0.30
    public var memoryCacheTTLMinutes: Double = 0

    // TurboQuant KV-cache compression. **Default on for every model
    // (MLX + JANG alike) in vMLX.** Previously the flag only mattered
    // for JANG-stamped models; now it applies to every chat/server
    // session regardless of loader so the hot path compresses KV with
    // zero generation-speed overhead.
    //
    // `turboQuantBits` is the per-side bit budget passed to
    // `.turboQuant(keyBits: N, valueBits: N)`. Lower = more compression.
    // Typical range 3–8: 3 bits/side ≈ 4.7x compression (aggressive),
    // 4 bits/side ≈ 3.6x (sweet spot), 8 bits/side ≈ near-lossless
    // reference. The 4-bit default matches the upstream TurboQuant
    // paper's production recommendation.
    //
    // Wired into the generation loop at `Stream.buildGenerateParameters`
    // → `GenerateParameters.kvMode = .turboQuant(...)`. Hybrid-SSM
    // models are safe: `maybeQuantizeKVCache` only compresses plain
    // `KVCacheSimple` layers and skips `MambaCache`/`RotatingKVCache`/
    // `CacheList` — Nemotron-H, Qwen3-Next, Jamba, FalconH1 etc. keep
    // their SSM paths untouched.
    // Default-ON (iter-64 — user directive): TurboQuant KV cache is the
    // NATIVE DEFAULT for vMLX v2. Production priority is memory savings
    // on long contexts over raw decode throughput. Users who want to
    // A/B against plain KV can flip `enableTurboQuant=false` via the
    // Server tab's Cache section, the `vmlxctl serve --disable-turboquant`
    // flag, or the `VMLX_DISABLE_TURBO_QUANT=1` env killswitch.
    //
    // History: iter-16 (2026-04-16) flipped this to false after a perf
    // audit found 25-40% decode regressions on MoE/hybrid models. The
    // user directive in iter-64 explicitly reinstates default-on because
    // (a) most production queries need long context more than peak tok/s,
    // (b) MLA models already skip TQ via `cacheTypeIsMLA` guard at
    // Stream.swift:~2146, (c) hybrid-SSM mamba layers skip TQ via
    // `maybeQuantizeKVCache`'s `KVCacheSimple`-only compression (SSM
    // state + rotating windows pass through untouched), (d) the env
    // killswitch makes A/B testing trivial.
    //
    // JANG models with calibrated `turboquant` blocks in their
    // `jang_config.json` still auto-activate TQ through the explicit
    // `loadedJangConfig?.turboquant` check in Stream.swift — that path
    // does NOT read this flag, so calibrated models keep their ship-time
    // behavior regardless of this default.
    public var enableTurboQuant: Bool = true
    public var turboQuantBits: Int = 4

    // Hybrid-SSM companion cache re-derive. When a thinking-template
    // turn finishes on a hybrid model, the post-generation SSM state
    // is contaminated (see `Cache/SSMReDerive.swift` for the long
    // rationale). With this flag on, the engine runs a fresh
    // prompt-only forward pass synchronously right after the normal
    // cache store and installs the CLEAN state in the SSM companion
    // cache keyed on the stripped prompt hash. Next turn's fetch
    // gets a position-accurate SSM state and hybrid models can
    // actually cache-hit on multi-turn thinking-model chat.
    //
    // Cost: one extra chunked-prefill pass (no decode) per turn.
    // ~2s on a 2K prompt. Users who prioritize responsiveness
    // over hybrid cache reuse can disable this.
    public var enableSSMReDerive: Bool = true

    // JANG
    public var enableJANG: Bool = true            // vMLX-only: repack JANG models on load
    public var quantizationOverride: String = "" // cli.py --image-quantize equivalent + ad-hoc override

    // SSM companion
    public var enableSSMCompanion: Bool = true    // vMLX-only: hybrid SSM companion cache

    // Smelt mode (partial expert loading)
    public var smelt: Bool = false                // cli.py --smelt
    public var smeltExperts: Int = 50             // cli.py --smelt-experts
    public var smeltMode: String = "default"      // doc-only --smelt-mode (per MEMORY)

    // Flash MoE
    public var flashMoe: Bool = false             // cli.py --flash-moe
    public var flashMoeSlotBank: Int = 64         // cli.py --flash-moe-slot-bank default=64
    public var flashMoePrefetch: String = "none"  // cli.py --flash-moe-prefetch: none|temporal
    public var flashMoeIoSplit: Int = 4           // cli.py --flash-moe-io-split

    // Distributed
    public var distributed: Bool = false          // cli.py --distributed
    public var distributedHost: String = ""       // doc-only --distributed-host
    public var distributedPort: Int = 9100        // doc-only --distributed-port
    public var distributedMode: String = "pipeline" // cli.py --distributed-mode: pipeline|tensor
    public var clusterSecret: String = ""         // cli.py --cluster-secret
    public var workerNodes: String = ""           // cli.py --worker-nodes

    // Speculative / PLD
    public var enableJit: Bool = false            // cli.py --enable-jit
    public var speculativeModel: String = ""      // cli.py --speculative-model
    public var numDraftTokens: Int = 3            // cli.py --num-draft-tokens default=3
    public var enablePld: Bool = false            // cli.py --enable-pld
    public var pldSummaryInterval: Int = 487      // cli.py --pld-summary-interval default=487

    // JANG-DFlash speculative decoding (block diffusion drafter + DDTree).
    // Swift-native: implementation lives in
    // `Sources/vMLXLMCommon/DFlash/JangDFlashSpecDec.swift`. Requires a
    // target model conforming to `JangDFlashTarget` (MiniMax only today)
    // and a safetensors drafter checkpoint at `dflashDrafterPath`. When
    // enabled but any precondition is missing the engine logs a warning
    // and falls back to the standard token iterator.
    public var dflash: Bool = false
    public var dflashDrafterPath: String = ""
    public var dflashBlockSize: Int = 16            // JangDFlashSpecConfig.blockSize
    public var dflashTopK: Int = 4                  // JangDFlashSpecConfig.topK
    public var dflashNumPaths: Int = 60             // JangDFlashSpecConfig.numPaths
    public var dflashTapLayers: String = "10,22,34,46,58"  // comma-sep int indices
    public var dflashTargetHiddenDim: Int = 3072    // JangDFlashSpecConfig.targetHiddenDim

    // Tool calling
    //
    // iter-50 audit note on `enableAutoToolChoice`: this is a vLLM-
    // compat flag (`--enable-auto-tool-choice`) that in vLLM gates
    // whether the server auto-parses tool calls from model output.
    // The Swift engine ALWAYS auto-parses when `request.tools` is
    // present — parser selection flows through `caps.toolParser ?? resolved.settings.defaultToolParser`
    // in Stream.swift. The flag is persisted through the resolver
    // but has no runtime gate.
    //
    // iter-102 note: the original note claimed "kept for CLI compat
    // so `--enable-auto-tool-choice` on the command line doesn't
    // error out" — but `vmlxctl` (ArgumentParser) doesn't actually
    // define this flag, so passing it would error with "unexpected
    // argument". The flag is inert: it's decoded from GlobalSettings
    // JSON dumps + SessionSettings JSON overrides (so migrating
    // Python users' configs doesn't need a schema change) but
    // nothing reads it. Future work: either honor it (e.g. to
    // hard-disable parsing for scripted clients) or drop it along
    // with the resolver wire-through and migrate config consumers.
    public var enableAutoToolChoice: Bool = false // cli.py --enable-auto-tool-choice (dead gate — see audit note)
    public var defaultToolParser: String = ""      // cli.py --tool-call-parser default=None (auto-detected from model)
    public var defaultReasoningParser: String = "" // cli.py --reasoning-parser default=None (auto-detected from model)

    // Inference defaults (from cli.py --default-*)
    public var defaultEnableThinking: Bool? = nil    // cli.py --default-enable-thinking (tri-state)
    public var defaultTemperature: Double = 0.7      // cli.py --default-temperature
    public var defaultTopP: Double = 0.9             // server.py _FALLBACK_TOP_P = 0.9
    public var defaultTopK: Int = 0                  // NOT in cli.py; mlx-lm implicit 0 = disabled unless request sets it
    public var defaultMinP: Double = 0.0             // cli.py --default-min-p
    public var defaultRepetitionPenalty: Double = 1.0 // cli.py --default-repetition-penalty
    public var defaultMaxTokens: Int = 32768         // cli.py --max-tokens
    public var defaultSystemPrompt: String? = nil    // cli.py --default-system-prompt

    /// Maximum tool-call rounds per `/v1/chat/completions` request.
    /// Each round = one generation pass + one batch of tool calls
    /// executed server-side. Once the limit is hit the loop exits
    /// and the caller sees whatever content came out of the last
    /// turn. The old behavior was a hardcoded `10` inside
    /// `Stream.maxToolIterations`; now it's a tunable knob so
    /// long-running agent workflows (Cline / Aider-style) can raise
    /// it without a rebuild. `SessionSettings.maxToolIterations`
    /// (the per-chat override) already existed.
    public var defaultMaxToolIterations: Int = 10   // cli.py --max-tool-iterations

    // Chat template
    public var chatTemplate: String = ""             // cli.py --chat-template
    public var chatTemplateKwargs: String = ""       // cli.py --chat-template-kwargs (JSON blob)
    public var isMllm: Bool? = nil                   // cli.py --is-mllm (tri-state)
    public var embeddingModel: String = ""           // cli.py --embedding-model

    // Idle lifecycle — NOT in cli.py. Python has no auto-sleep; sleep is
    // admin-only via `/admin/soft-sleep` + `/admin/deep-sleep`.
    //
    // Default ON for the Swift/desktop app because:
    //   * Users leave sessions running between short conversations, and
    //     holding ~30 GB of weights resident hurts every other app on
    //     an M-series Mac that shares unified memory.
    //   * Both transitions are transparent — `softSleep` only drops
    //     caches (weights stay hot), and any HTTP request wakes the
    //     engine automatically before dispatch (see
    //     `OpenAIRoutes`/`GatewayServer` — `wakeFromStandby()`).
    //   * Chat / Terminal paths also call `wakeFromStandby()` before
    //     streaming (B2/B3 fix) so there's no user-visible stall.
    //
    // Users who dislike the behavior can turn it off in the Lifecycle
    // section of SessionConfigForm.
    public var idleEnabled: Bool = true
    public var idleSoftSec: Double = 300             // vMLX-only UI (used only when idleEnabled)
    public var idleDeepSec: Double = 900             // vMLX-only UI (used only when idleEnabled)
    public var wakeTimeout: Int = 300                // cli.py --wake-timeout default=300

    // Auth / security
    public var apiKey: String? = nil                 // cli.py --api-key default=None
    public var adminToken: String? = nil             // doc-only --admin-token (from MEMORY)
    public var corsOrigins: [String] = ["*"]         // cli.py --allowed-origins default="*"
    public var rateLimit: Int = 0                    // cli.py --rate-limit default=0
    public var requestTimeout: Double = 300          // cli.py --timeout default=300.0
    /// PEM-format TLS key file. When both `sslKeyFile` and `sslCertFile`
    /// are set, the Hummingbird server binds with a TLS context. Maps to
    /// Python's `--ssl-keyfile`. Empty string disables TLS.
    public var sslKeyFile: String = ""
    /// PEM-format TLS cert file. See `sslKeyFile`.
    public var sslCertFile: String = ""
    // NOTE: No `loadTimeoutSec` field. Python's `serve` has NO load timeout.
    // If you want a hung HF download killer wait for an upstream flag rather
    // than baking in a made-up default.

    // API gateway
    public var inferenceEndpoints: String = ""       // cli.py --inference-endpoints

    // MCP
    public var mcpConfigPath: String = ""            // cli.py --mcp-config
    public var mcpServers: [MCPServer] = []

    // Misc vMLX.app UI defaults
    public var streamUsageInChunks: Bool = true
    public var hideToolStatusDefault: Bool = false

    // MARK: Image generation defaults (vMLX.app UI)
    //
    // Mirrors the Electron ImageSettings component. Every field here is the
    // *default* — live UI state lives in the Image screen view model and can
    // override these per-generation. Written/read via SettingsStore so users
    // see their last-used values on relaunch.
    public var imageDefaultSteps: Int = 4            // Flux Schnell default; Dev=30, Z-Image=8
    public var imageDefaultGuidance: Double = 3.5    // Dev=3.5, Schnell ignores
    public var imageDefaultWidth: Int = 1024
    public var imageDefaultHeight: Int = 1024
    public var imageDefaultSeed: Int = -1            // -1 = random
    public var imageDefaultNumImages: Int = 1
    public var imageDefaultScheduler: String = "default"
    public var imageDefaultStrength: Double = 0.75   // edit-mode img2img strength
    public var imageDefaultModelAlias: String = ""   // last-used model alias

    public init() {}
}

// MARK: - SessionSettings (tier 2 — session-level overrides, all optional)

/// Per-running-session config. Stored keyed by a UUID that the app generates
/// when the user creates a server session. Most fields are optional (nil =
/// inherit from global). The non-optional `modelPath` is required because a
/// session can't exist without a model.
public struct SessionSettings: Codable, Sendable, Equatable {
    public var modelPath: URL
    public var modelAlias: String? = nil      // cli.py --served-model-name
    public var displayName: String? = nil
    public var host: String? = nil
    public var port: Int? = nil
    public var lan: Bool? = nil
    public var quantizationOverride: String? = nil

    // Engine load overrides
    public var engineKind: EngineKindCodable? = nil
    public var maxNumSeqs: Int? = nil
    public var prefillStepSize: Int? = nil
    public var maxCacheBlocks: Int? = nil
    // iter-46: paged-cache block size, per-session override. Defaults
    // to GlobalSettings.pagedCacheBlockSize when nil. Determines the
    // minimum prompt length that can produce a paged-cache hit —
    // sub-block prompts return nil from fetchPrefix entirely.
    public var pagedCacheBlockSize: Int? = nil

    public var enableTurboQuant: Bool? = nil
    public var turboQuantBits: Int? = nil
    public var enableJANG: Bool? = nil
    public var enablePrefixCache: Bool? = nil
    public var enableSSMCompanion: Bool? = nil
    public var enableBlockDiskCache: Bool? = nil
    public var enableDiskCache: Bool? = nil
    // Per-session disk-cache dir + cap. Added 2026-04-16 because the
    // SessionConfigForm UI lets users redirect the cache to an external
    // SSD per session; the field was missing on SessionSettings, blocking
    // the Release build.
    public var diskCacheDir: String? = nil
    public var diskCacheMaxGB: Double? = nil
    public var enableMemoryCache: Bool? = nil
    public var memoryCachePercent: Double? = nil
    public var memoryCacheTTLMinutes: Double? = nil
    public var kvCacheQuantization: String? = nil

    public var flashMoe: Bool? = nil
    public var flashMoeSlotBank: Int? = nil
    public var flashMoePrefetch: String? = nil
    public var flashMoeIoSplit: Int? = nil
    public var smelt: Bool? = nil
    public var smeltExperts: Int? = nil
    public var smeltMode: String? = nil

    // DFlash (per-session overrides)
    public var dflash: Bool? = nil
    public var dflashDrafterPath: String? = nil
    public var dflashBlockSize: Int? = nil
    public var dflashTopK: Int? = nil
    public var dflashNumPaths: Int? = nil
    public var dflashTapLayers: String? = nil
    public var dflashTargetHiddenDim: Int? = nil

    public var distributed: Bool? = nil
    public var distributedHost: String? = nil
    public var distributedPort: Int? = nil
    public var distributedMode: String? = nil
    public var clusterSecret: String? = nil
    public var workerNodes: String? = nil

    // Inference defaults (tier overrides)
    public var defaultEnableThinking: Bool? = nil
    public var defaultTemperature: Double? = nil
    public var defaultTopP: Double? = nil
    public var defaultTopK: Int? = nil
    public var defaultMinP: Double? = nil
    public var defaultRepetitionPenalty: Double? = nil
    public var defaultMaxTokens: Int? = nil
    public var defaultSystemPrompt: String? = nil

    public var defaultReasoningParser: String? = nil
    public var defaultToolParser: String? = nil
    public var enableAutoToolChoice: Bool? = nil

    public var chatTemplate: String? = nil
    public var chatTemplateKwargs: String? = nil

    // Idle
    public var idleEnabled: Bool? = nil
    public var idleSoftSec: Double? = nil
    public var idleDeepSec: Double? = nil

    // Auth
    public var apiKey: String? = nil
    public var adminToken: String? = nil
    public var corsOrigins: [String]? = nil

    // MCP
    public var mcpConfigPath: String? = nil
    public var mcpServers: [MCPServer]? = nil

    // Remote endpoint (new in DEV-6). When `remoteURL` is non-nil the
    // session is treated as a proxy: Chat/Terminal bypass the local
    // Engine actor and `RemoteEngineClient` makes HTTP calls to the
    // remote server instead. `modelPath` is still required but is
    // treated as a placeholder display name in that mode — the real
    // model id lives in `remoteModelName`.
    public var remoteURL: String? = nil
    public var remoteProtocol: String? = nil   // "openai" | "ollama" | "anthropic"
    public var remoteAPIKey: String? = nil     // stored in Keychain by session UUID
    public var remoteModelName: String? = nil  // id sent in ChatRequest.model

    /// True if this session is configured as a thin proxy to a remote
    /// OpenAI/Ollama/Anthropic-compatible endpoint. Drives the UI label,
    /// the chat dispatch path, and the SessionDashboard lifecycle buttons.
    public var isRemote: Bool {
        guard let u = remoteURL, !u.isEmpty else { return false }
        return URL(string: u) != nil
    }

    public init(modelPath: URL) { self.modelPath = modelPath }
}

// MARK: - ChatSettings (tier 3 — per-chat overrides, all optional)

/// Per-chat overrides. Mirrors Electron `chat_overrides` table columns
/// (panel/src/main/database.ts:228 + migrations 350-429).
public struct ChatSettings: Codable, Sendable {
    public var modelAlias: String? = nil

    public var temperature: Double? = nil
    public var topP: Double? = nil
    public var topK: Int? = nil
    public var minP: Double? = nil
    public var repetitionPenalty: Double? = nil
    public var maxTokens: Int? = nil
    public var systemPrompt: String? = nil
    public var stopSequences: [String]? = nil

    public var enableThinking: Bool? = nil
    public var reasoningEffort: String? = nil       // Electron chat_overrides.reasoning_effort
    public var wireApi: String? = nil               // Electron chat_overrides.wire_api

    public var toolChoice: String? = nil            // "auto" | "none" | name
    public var tools: [ChatRequest.Tool]? = nil
    public var maxToolIterations: Int? = nil
    public var builtinToolsEnabled: Bool? = nil
    public var hideToolStatus: Bool? = nil
    public var mcpEnabled: Bool? = nil

    public var workingDirectory: String? = nil
    public var webSearchEnabled: Bool? = nil
    public var fetchUrlEnabled: Bool? = nil
    public var fileToolsEnabled: Bool? = nil
    public var searchToolsEnabled: Bool? = nil
    public var shellEnabled: Bool? = nil
    public var gitEnabled: Bool? = nil
    public var utilityToolsEnabled: Bool? = nil
    public var braveSearchEnabled: Bool? = nil
    public var toolResultMaxChars: Int? = nil

    public init() {}
}

// MARK: - RequestOverride (tier 4 — populated from HTTP body)

/// Highest-priority tier. Populated per-request from the chat completion /
/// /api/chat / /v1/messages body. Usually sparse — the client only sets the
/// fields they explicitly want to override.
public struct RequestOverride: Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var maxTokens: Int?
    public var systemPrompt: String?
    public var stopSequences: [String]?
    public var enableThinking: Bool?
    public var reasoningEffort: String?
    public var toolChoice: String?
    public var tools: [ChatRequest.Tool]?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        maxTokens: Int? = nil,
        systemPrompt: String? = nil,
        stopSequences: [String]? = nil,
        enableThinking: Bool? = nil,
        reasoningEffort: String? = nil,
        toolChoice: String? = nil,
        tools: [ChatRequest.Tool]? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.systemPrompt = systemPrompt
        self.stopSequences = stopSequences
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.toolChoice = toolChoice
        self.tools = tools
    }

    /// Convenience: build a RequestOverride from a ChatRequest body.
    public static func from(_ req: ChatRequest) -> RequestOverride {
        RequestOverride(
            temperature: req.temperature,
            topP: req.topP,
            topK: req.topK,
            minP: req.minP,
            repetitionPenalty: req.repetitionPenalty,
            maxTokens: req.maxTokens,
            systemPrompt: nil,
            stopSequences: req.stop,
            enableThinking: req.enableThinking,
            reasoningEffort: req.reasoningEffort,
            toolChoice: nil,
            tools: req.tools
        )
    }
}

// MARK: - ResolvedSettings (output of the 4-tier resolver)

/// Every field from GlobalSettings plus a `resolutionTrace` debugging field
/// that records which tier contributed which value. The trace is keyed by
/// the Swift property name for readability.
public struct ResolvedSettings: Sendable {
    public var settings: GlobalSettings
    public var resolutionTrace: [String: SettingsTier]

    public init(settings: GlobalSettings, trace: [String: SettingsTier] = [:]) {
        self.settings = settings
        self.resolutionTrace = trace
    }

    // Passthrough accessors for the most-used fields so call sites don't
    // have to keep saying `.settings.defaultTemperature`.
    public var temperature: Double { settings.defaultTemperature }
    public var topP: Double { settings.defaultTopP }
    public var topK: Int { settings.defaultTopK }
    public var minP: Double { settings.defaultMinP }
    public var repetitionPenalty: Double { settings.defaultRepetitionPenalty }
    public var maxTokens: Int { settings.defaultMaxTokens }
    public var enableThinking: Bool? { settings.defaultEnableThinking }
    public var systemPrompt: String? { settings.defaultSystemPrompt }
}

// MARK: - SettingsChange (subscription events)

public enum SettingsChange: Sendable, Equatable {
    case global
    case session(UUID)
    case chat(UUID)
}
