import Foundation

/// O1 §290 — the single-source-of-truth catalog of every HTTP route
/// vMLX registers. Hand-authored against the six route files in
/// `Sources/vMLXServer/Routes/` so the API screen renders a live
/// discoverable list instead of users guessing paths from source.
///
/// When a new route lands, add an entry here AND register it in the
/// corresponding Routes/*.swift. The RoutesCard view keys off this
/// table for the filter chips, per-route online dots, copy-curl, and
/// the route-inspector drawer.

public struct RouteEntry: Identifiable, Hashable, Sendable {
    public var id: String { "\(method) \(path)" }
    public let method: Method
    public let path: String
    /// Family grouping for the filter chips: OpenAI, Ollama, Anthropic,
    /// Admin, MCP, Metrics, Gateway.
    public let family: Family
    /// One-line brief shown next to the path.
    public let brief: String
    /// Auth requirement — used to badge routes + filter.
    public let auth: AuthRequirement
    /// True when the happy path streams SSE (text/event-stream).
    public let streams: Bool
    /// Modality this route depends on being loaded. Used to grey the
    /// online dot when the currently-loaded engine lacks the modality.
    public let modality: Modality
    /// Sample JSON body for copy-curl. Empty string for GET routes.
    public let sampleBody: String
    /// Docs anchor (relative URL or section header). Optional.
    public let docsAnchor: String?

    public enum Method: String, Sendable, Codable {
        case get = "GET"
        case post = "POST"
        case delete = "DELETE"
        case put = "PUT"
        case head = "HEAD"
    }

    public enum Family: String, CaseIterable, Sendable, Codable {
        case liveness = "Liveness"
        case openAI = "OpenAI"
        case ollama = "Ollama"
        case anthropic = "Anthropic"
        case admin = "Admin"
        case cache = "Cache"
        case mcp = "MCP"
        case metrics = "Metrics"
    }

    public enum AuthRequirement: String, Sendable, Codable {
        /// Anyone — no auth middleware applied.
        case none
        /// `Authorization: Bearer <apiKey>` required when set, else open.
        case bearer
        /// `Authorization: Bearer <adminToken>` required when set; admin
        /// routes gate destructive + lifecycle operations.
        case admin
    }

    public enum Modality: String, Sendable, Codable {
        case any           // works regardless of loaded model (e.g. /health)
        case text          // needs a chat / text model
        case embedding     // needs an embedding model
        case image         // needs an image model
        case audio         // needs a TTS or STT model
        case rerank        // needs a reranker model
    }
}

public enum RouteCatalog {
    /// Full table — 52 routes as of §290.
    public static let all: [RouteEntry] = [
        // MARK: Liveness
        .init(method: .get, path: "/health",
              family: .liveness, brief: "Engine state, inflight count, idle countdowns",
              auth: .none, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "health"),
        .init(method: .get, path: "/metrics",
              family: .metrics, brief: "Prometheus gauges: GPU / RAM / CPU / tok-per-sec / queue",
              auth: .none, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "metrics"),

        // MARK: OpenAI
        .init(method: .post, path: "/v1/chat/completions",
              family: .openAI, brief: "Streaming chat completion with tools + reasoning",
              auth: .bearer, streams: true, modality: .text,
              sampleBody: #"{"model":"<model>","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
              docsAnchor: "chat-completions"),
        .init(method: .delete, path: "/v1/chat/completions/:id/cancel",
              family: .openAI, brief: "Cancel an in-flight chat completion by request id",
              auth: .bearer, streams: false, modality: .text, sampleBody: "",
              docsAnchor: "chat-completions-cancel"),
        .init(method: .post, path: "/v1/completions",
              family: .openAI, brief: "Legacy text completion",
              auth: .bearer, streams: true, modality: .text,
              sampleBody: #"{"model":"<model>","prompt":"Once upon","max_tokens":64}"#,
              docsAnchor: "completions"),
        .init(method: .delete, path: "/v1/completions/:id/cancel",
              family: .openAI, brief: "Cancel a legacy completion",
              auth: .bearer, streams: false, modality: .text, sampleBody: "",
              docsAnchor: "completions-cancel"),
        .init(method: .post, path: "/v1/embeddings",
              family: .openAI, brief: "Dense embeddings (base64 or float vector)",
              auth: .bearer, streams: false, modality: .embedding,
              sampleBody: #"{"model":"<emb>","input":"the quick brown fox","encoding_format":"base64"}"#,
              docsAnchor: "embeddings"),
        .init(method: .post, path: "/v1/rerank",
              family: .openAI, brief: "Cross-encoder rerank of documents by query",
              auth: .bearer, streams: false, modality: .rerank,
              sampleBody: #"{"model":"<rr>","query":"swift concurrency","documents":["GCD","actors","dispatch"]}"#,
              docsAnchor: "rerank"),
        .init(method: .get, path: "/v1/models",
              family: .openAI, brief: "List known models across library roots + loaded flag",
              auth: .bearer, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "models"),
        .init(method: .post, path: "/v1/images/generations",
              family: .openAI, brief: "Flux / Z-Image generation (seed, n, response_format)",
              auth: .bearer, streams: false, modality: .image,
              sampleBody: #"{"model":"<img>","prompt":"a neon city","n":1,"size":"1024x1024","seed":42}"#,
              docsAnchor: "images-generate"),
        .init(method: .post, path: "/v1/images/edits",
              family: .openAI, brief: "Image-conditioned edit (img2img + mask)",
              auth: .bearer, streams: false, modality: .image,
              sampleBody: #"{"model":"<img>","prompt":"make it night","image":"<base64>","n":1}"#,
              docsAnchor: "images-edit"),
        .init(method: .post, path: "/v1/audio/transcriptions",
              family: .openAI, brief: "Whisper-compatible speech-to-text (multipart)",
              auth: .bearer, streams: false, modality: .audio, sampleBody: "",
              docsAnchor: "audio-transcribe"),
        .init(method: .post, path: "/v1/audio/translations",
              family: .openAI, brief: "Whisper speech-to-English translation",
              auth: .bearer, streams: false, modality: .audio, sampleBody: "",
              docsAnchor: "audio-translate"),
        .init(method: .post, path: "/v1/audio/speech",
              family: .openAI, brief: "TTS — Kokoro / Parler / Dia voice synthesis",
              auth: .bearer, streams: false, modality: .audio,
              sampleBody: #"{"model":"kokoro","input":"hello world","voice":"af_sky"}"#,
              docsAnchor: "audio-speech"),
        .init(method: .get, path: "/v1/audio/voices",
              family: .openAI, brief: "Enumerate available TTS voices",
              auth: .bearer, streams: false, modality: .audio, sampleBody: "",
              docsAnchor: "audio-voices"),
        .init(method: .post, path: "/v1/responses",
              family: .openAI, brief: "OpenAI Responses API (tools + thinking blocks)",
              auth: .bearer, streams: true, modality: .text,
              sampleBody: #"{"model":"<model>","input":"hi","stream":true}"#,
              docsAnchor: "responses"),
        .init(method: .delete, path: "/v1/responses/:id/cancel",
              family: .openAI, brief: "Cancel an in-flight Responses session",
              auth: .bearer, streams: false, modality: .text, sampleBody: "",
              docsAnchor: "responses-cancel"),

        // MARK: Anthropic
        .init(method: .post, path: "/v1/messages",
              family: .anthropic, brief: "Anthropic Messages API with tool_use + thinking",
              auth: .bearer, streams: true, modality: .text,
              sampleBody: #"{"model":"<model>","messages":[{"role":"user","content":"hi"}],"max_tokens":512}"#,
              docsAnchor: "messages"),
        .init(method: .post, path: "/v1/messages/count_tokens",
              family: .anthropic, brief: "Anthropic token count estimator",
              auth: .bearer, streams: false, modality: .text,
              sampleBody: #"{"model":"<model>","messages":[{"role":"user","content":"hi"}]}"#,
              docsAnchor: "count-tokens"),

        // MARK: Ollama
        .init(method: .post, path: "/api/chat",
              family: .ollama, brief: "Ollama streaming chat (tool_calls two-chunk shape)",
              auth: .bearer, streams: true, modality: .text,
              sampleBody: #"{"model":"<model>","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
              docsAnchor: "ollama-chat"),
        .init(method: .post, path: "/api/generate",
              family: .ollama, brief: "Ollama generate (context[] legacy array)",
              auth: .bearer, streams: true, modality: .text,
              sampleBody: #"{"model":"<model>","prompt":"hi","stream":true}"#,
              docsAnchor: "ollama-generate"),
        .init(method: .post, path: "/api/embed",
              family: .ollama, brief: "Ollama embedding (native format)",
              auth: .bearer, streams: false, modality: .embedding,
              sampleBody: #"{"model":"<emb>","input":["first","second"]}"#,
              docsAnchor: "ollama-embed"),
        .init(method: .post, path: "/api/embeddings",
              family: .ollama, brief: "Legacy single-string embeddings endpoint",
              auth: .bearer, streams: false, modality: .embedding,
              sampleBody: #"{"model":"<emb>","prompt":"hello"}"#,
              docsAnchor: "ollama-embeddings"),
        .init(method: .get, path: "/api/tags",
              family: .ollama, brief: "List installed models in Ollama format",
              auth: .bearer, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "ollama-tags"),
        .init(method: .get, path: "/api/ps",
              family: .ollama, brief: "Currently-loaded models (Ollama format)",
              auth: .bearer, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "ollama-ps"),
        .init(method: .get, path: "/api/version",
              family: .ollama, brief: "Ollama version string (drift-locked for Copilot)",
              auth: .none, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "ollama-version"),
        .init(method: .post, path: "/api/show",
              family: .ollama, brief: "Model details (architecture, quantization, license)",
              auth: .bearer, streams: false, modality: .any,
              sampleBody: #"{"name":"<model>"}"#,
              docsAnchor: "ollama-show"),
        .init(method: .post, path: "/api/pull",
              family: .ollama, brief: "Download a model from HuggingFace (Ollama-shape stream)",
              auth: .bearer, streams: true, modality: .any,
              sampleBody: #"{"name":"mlx-community/Qwen3-0.6B-8bit"}"#,
              docsAnchor: "ollama-pull"),
        .init(method: .delete, path: "/api/delete",
              family: .ollama, brief: "Delete a model from the library",
              auth: .bearer, streams: false, modality: .any,
              sampleBody: #"{"name":"<model>"}"#,
              docsAnchor: "ollama-delete"),
        .init(method: .post, path: "/api/copy",
              family: .ollama, brief: "Duplicate a model under a new alias",
              auth: .bearer, streams: false, modality: .any,
              sampleBody: #"{"source":"<model>","destination":"<alias>"}"#,
              docsAnchor: "ollama-copy"),
        .init(method: .post, path: "/api/create",
              family: .ollama, brief: "Create a model from a Modelfile (Ollama compat)",
              auth: .bearer, streams: true, modality: .any,
              sampleBody: #"{"name":"<alias>","modelfile":"FROM <base>\nPARAMETER temperature 0.7"}"#,
              docsAnchor: "ollama-create"),
        .init(method: .post, path: "/api/push",
              family: .ollama, brief: "Publish a local model to HuggingFace Hub (§187)",
              auth: .bearer, streams: true, modality: .any,
              sampleBody: #"{"name":"<alias>"}"#,
              docsAnchor: "ollama-push"),
        .init(method: .head, path: "/api/blobs/:digest",
              family: .ollama, brief: "Check if a blob is already cached",
              auth: .bearer, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "ollama-blobs"),

        // MARK: Admin (admin-token gated)
        .init(method: .post, path: "/admin/soft-sleep",
              family: .admin, brief: "Flush caches, keep weights resident",
              auth: .admin, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "admin-soft-sleep"),
        .init(method: .post, path: "/admin/deep-sleep",
              family: .admin, brief: "Unload weights; GPU memory drops ≥90%",
              auth: .admin, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "admin-deep-sleep"),
        .init(method: .post, path: "/admin/wake",
              family: .admin, brief: "Reload weights (optional model override body)",
              auth: .admin, streams: false, modality: .any,
              sampleBody: #"{"model":"/path/to/model"}"#,
              docsAnchor: "admin-wake"),
        .init(method: .post, path: "/admin/benchmark",
              family: .admin, brief: "Run the in-process BenchSuite against loaded model",
              auth: .admin, streams: false, modality: .text, sampleBody: "",
              docsAnchor: "admin-benchmark"),
        .init(method: .post, path: "/admin/dflash/load",
              family: .admin, brief: "Attach a JANG-DFlash drafter for speculative decoding",
              auth: .admin, streams: false, modality: .text,
              sampleBody: #"{"drafter":"/path/to/drafter.safetensors"}"#,
              docsAnchor: "admin-dflash-load"),
        .init(method: .post, path: "/admin/dflash/unload",
              family: .admin, brief: "Detach the DFlash drafter",
              auth: .admin, streams: false, modality: .text, sampleBody: "",
              docsAnchor: "admin-dflash-unload"),
        .init(method: .get, path: "/admin/dflash",
              family: .admin, brief: "Current DFlash drafter status + stats",
              auth: .admin, streams: false, modality: .text, sampleBody: "",
              docsAnchor: "admin-dflash-status"),
        .init(method: .delete, path: "/admin/models/:id",
              family: .admin, brief: "Remove a model entry (guarded while loading)",
              auth: .admin, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "admin-models-delete"),
        // S3 §309 — live log-level swap (paired with R4).
        .init(method: .get, path: "/admin/log-level",
              family: .admin, brief: "Current LogStore global minimum level",
              auth: .admin, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "admin-log-level-get"),
        .init(method: .post, path: "/admin/log-level",
              family: .admin, brief: "Change LogStore verbosity without restart",
              auth: .admin, streams: false, modality: .any,
              sampleBody: #"{"level":"debug"}"#,
              docsAnchor: "admin-log-level-set"),

        // MARK: Cache (admin-gated)
        .init(method: .get, path: "/v1/cache/stats",
              family: .cache, brief: "Prefix / paged / memory / disk / SSM / TQ stats",
              auth: .admin, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "cache-stats"),
        .init(method: .post, path: "/v1/cache/clear",
              family: .cache, brief: "Drop every tier's cache entries",
              auth: .admin, streams: false, modality: .any,
              sampleBody: "{}",
              docsAnchor: "cache-clear"),
        .init(method: .get, path: "/v1/cache/entries",
              family: .cache, brief: "Enumerate cache entries with byte sizes",
              auth: .admin, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "cache-entries"),
        .init(method: .post, path: "/v1/cache/warm",
              family: .cache, brief: "Prefill a prompt into the paged cache",
              auth: .admin, streams: false, modality: .text,
              sampleBody: #"{"prompt":"Hello world","model":"<model>"}"#,
              docsAnchor: "cache-warm"),
        .init(method: .get, path: "/v1/cache",
              family: .cache, brief: "Cache subsystem root / overview",
              auth: .admin, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "cache-root"),

        // MARK: MCP
        .init(method: .get, path: "/v1/mcp/servers",
              family: .mcp, brief: "List registered MCP servers + connection state",
              auth: .bearer, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "mcp-servers"),
        .init(method: .get, path: "/v1/mcp/tools",
              family: .mcp, brief: "Flat tool list across all registered MCP servers",
              auth: .bearer, streams: false, modality: .any, sampleBody: "",
              docsAnchor: "mcp-tools"),
        .init(method: .post, path: "/v1/mcp/execute",
              family: .mcp, brief: "Call an MCP tool directly (bypasses chat loop)",
              auth: .bearer, streams: false, modality: .any,
              sampleBody: #"{"server":"<name>","tool":"<tool>","arguments":{}}"#,
              docsAnchor: "mcp-execute"),
        .init(method: .get, path: "/mcp/:server/**",
              family: .mcp, brief: "Raw MCP passthrough (streams SSE for tool calls)",
              auth: .bearer, streams: true, modality: .any, sampleBody: "",
              docsAnchor: "mcp-passthrough"),
    ]

    /// Convenience: routes grouped by family, preserving `all` order.
    public static func byFamily() -> [(RouteEntry.Family, [RouteEntry])] {
        var out: [RouteEntry.Family: [RouteEntry]] = [:]
        for r in all { out[r.family, default: []].append(r) }
        return RouteEntry.Family.allCases.compactMap { f in
            guard let rs = out[f], !rs.isEmpty else { return nil }
            return (f, rs)
        }
    }

    /// Substitute tokens in a sample body + full URL for copy-curl.
    /// - `{host}`, `{port}`, `{bearer}`, `{admin}`, `{model}` replaced.
    public static func curl(
        for route: RouteEntry,
        scheme: String = "http",
        host: String,
        port: Int,
        bearer: String? = nil,
        admin: String? = nil,
        model: String? = nil
    ) -> String {
        let url = "\(scheme)://\(host):\(port)\(route.path)"
        var parts: [String] = ["curl", "-sS", "-X", route.method.rawValue]
        if route.auth == .admin, let t = admin, !t.isEmpty {
            parts += ["-H", "\"Authorization: Bearer \(t)\""]
        } else if route.auth == .bearer, let t = bearer, !t.isEmpty {
            parts += ["-H", "\"Authorization: Bearer \(t)\""]
        }
        if !route.sampleBody.isEmpty {
            parts += ["-H", "\"Content-Type: application/json\""]
            var body = route.sampleBody
            if let m = model, !m.isEmpty {
                body = body.replacingOccurrences(of: "<model>", with: m)
                body = body.replacingOccurrences(of: "<emb>", with: m)
                body = body.replacingOccurrences(of: "<img>", with: m)
                body = body.replacingOccurrences(of: "<rr>", with: m)
            }
            // Escape single quotes for POSIX shell wrapping in '...'
            let escaped = body.replacingOccurrences(of: "'", with: #"'"'"'"#)
            parts += ["-d", "'\(escaped)'"]
        }
        parts.append("\"\(url)\"")
        return parts.joined(separator: " ")
    }
}
