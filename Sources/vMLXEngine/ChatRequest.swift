import Foundation
import vMLXLMCommon

/// Wire-compatible chat request. Mirrors the Python `ChatRequest` Pydantic model
/// in vmlx_engine/server.py so the Swift HTTP server can decode the same payloads.
public struct ChatRequest: Codable, Sendable {
    public var model: String
    public var messages: [Message]
    public var stream: Bool?
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var stop: [String]?
    public var seed: Int?
    public var enableThinking: Bool?
    public var reasoningEffort: String?
    public var tools: [Tool]?
    public var toolChoice: ToolChoice?
    /// Optional session scope — vMLX extension. When set, the 4-tier
    /// settings resolver merges session-level overrides on top of the
    /// global defaults. Non-OpenAI field; ignored by clients that don't
    /// know about it.
    public var sessionId: String?
    /// Optional chat scope — vMLX extension. When set, chat-level
    /// overrides merge on top of session + global. Non-OpenAI.
    public var chatId: String?

    // MARK: - Decoded-but-partially-supported OpenAI parameters
    //
    // iter-101 §128: previously this block documented "reject with
    // 400 until real support lands". The actual contract evolved
    // (see the long comment at the top of `validate()`):
    //
    //   - `n != 1` and `logprobs == true` still hard-reject with
    //     400 — the semantic mismatch is severe enough that silently
    //     producing one completion or omitting logprobs would break
    //     the caller's downstream expectation.
    //   - `frequency_penalty` / `presence_penalty` are RANGE-CHECKED
    //     and they are now fully wired (§173) — `Stream.buildGenerateParameters`
    //     forwards them into `GenerateParameters` and the sampler's
    //     `FrequencyPenalty` / `PresencePenalty` LogitProcessors apply
    //     them per-token. `top_logprobs` is RANGE-CHECKED and accepted
    //     for SDK round-trip but the engine still doesn't emit
    //     alternative-token rows (only the chosen-token logprob).
    //   - `logit_bias` / `response_format` remain
    //     accepted-but-unwired — accepted silently by `validate()`
    //     so the absent path doesn't break SDK round-trip, but the
    //     engine ignores them. See FIXME(iter-96 §174) on logitBias.
    //
    // Remove a field from the accept-silent list below AND extend
    // `Engine.stream` to actually honor it when real support lands.

    /// OpenAI `n` — number of completions to generate per request.
    /// vMLX always generates one. `n > 1` is rejected with 400
    /// because silently producing one when the caller asked for
    /// five corrupts downstream code.
    public var n: Int?

    /// OpenAI `logprobs` — when true, include per-token log probabilities
    /// in the response. Collected by `LogprobsCollector` in Evaluate.swift
    /// on post-penalty logits and emitted as `choices[0].logprobs.content`
    /// in OpenAI wire format (with `bytes: [Int]` UTF-8 byte arrays).
    public var logprobs: Bool?

    /// OpenAI `top_logprobs` — how many top alternatives to include per
    /// token position (0–20). Requires `logprobs == true`.
    public var topLogprobs: Int?

    /// OpenAI `frequency_penalty` — distinct from
    /// `repetition_penalty`. Range-checked ([-2, 2]). Wired to the
    /// sampler as of iter-95 §173 — `Stream.buildGenerateParameters`
    /// forwards non-zero values into `GenerateParameters.frequencyPenalty`
    /// which the `FrequencyPenaltyContext` LogitProcessor applies
    /// on every decode step.
    public var frequencyPenalty: Double?

    /// OpenAI `presence_penalty` — distinct from
    /// `repetition_penalty`. Range-checked ([-2, 2]). Wired to the
    /// sampler as of iter-95 §173 (see `frequencyPenalty` above).
    public var presencePenalty: Double?

    /// OpenAI `logit_bias` — per-token sampling bias dictionary.
    /// Wire format: `{"<token_id>": <bias>, …}` where bias is a
    /// number typically in [-100, 100]; -100 effectively bans the
    /// token, +100 forces it.
    ///
    /// Iter 143: WIRED through to the sampler via `LogitBiasContext`
    /// (Evaluate.swift). Stream.swift's buildGenerateParameters
    /// parses string keys → Int token IDs and forwards as
    /// `GenerateParameters.logitBias`. Out-of-vocab IDs are silently
    /// ignored (see LogitBiasContext.process). Resolves the prior
    /// iter-96 §174 FIXME.
    public var logitBias: [String: Double]?

    /// OpenAI `response_format` — JSON-object / JSON-schema
    /// structured output. `text` (or nil) yields plain text; `json_object`
    /// biases sampling toward valid JSON via a prompt-level system message
    /// injection and a final-output validation pass; `json_schema`
    /// additionally validates against the provided schema.
    public var responseFormat: ResponseFormat?

    /// OpenAI `stream_options` — per-stream configuration. Currently
    /// honors `include_usage`: when true, the final SSE frame before
    /// `[DONE]` carries a usage payload (prompt/completion/total tokens
    /// plus cached breakdown). Default is false per OpenAI spec.
    public var streamOptions: StreamOptions?

    /// Anthropic-compat extension: when true, `reasoning_content` is
    /// included in non-streaming responses (streaming already emits it
    /// as a separate delta key). Distinct from `enable_thinking`, which
    /// controls *whether* reasoning is generated at all.
    public var includeReasoning: Bool?

    /// OpenAI `chat_template_kwargs` — per-request template variables
    /// merged into the Jinja render context alongside `reasoning_effort`.
    /// Lets callers pass model-specific switches (e.g. `enable_tools`,
    /// `assistant_prefix`) without a CLI rebuild.
    public var chatTemplateKwargs: [String: JSONValue]?

    public struct ResponseFormat: Codable, Sendable {
        public var type: String?
        public var jsonSchema: JSONSchemaSpec?

        public init(type: String? = nil, jsonSchema: JSONSchemaSpec? = nil) {
            self.type = type
            self.jsonSchema = jsonSchema
        }

        enum CodingKeys: String, CodingKey {
            case type
            case jsonSchema = "json_schema"
        }
    }

    /// OpenAI `response_format.json_schema` envelope.
    /// `schema` holds the JSON Schema draft used for validation.
    public struct JSONSchemaSpec: Codable, Sendable {
        public var name: String?
        public var description: String?
        public var schema: JSONValue?
        public var strict: Bool?

        public init(name: String? = nil, description: String? = nil,
                    schema: JSONValue? = nil, strict: Bool? = nil) {
            self.name = name
            self.description = description
            self.schema = schema
            self.strict = strict
        }
    }

    public struct StreamOptions: Codable, Sendable {
        public var includeUsage: Bool?
        public init(includeUsage: Bool? = nil) {
            self.includeUsage = includeUsage
        }
        enum CodingKeys: String, CodingKey {
            case includeUsage = "include_usage"
        }
    }

    public struct Message: Codable, Sendable {
        public var role: String   // "system" | "user" | "assistant" | "tool"
        public var content: ContentValue?
        public var name: String?
        public var toolCalls: [ToolCall]?
        public var toolCallId: String?

        public init(
            role: String,
            content: ContentValue? = nil,
            name: String? = nil,
            toolCalls: [ToolCall]? = nil,
            toolCallId: String? = nil
        ) {
            self.role = role
            self.content = content
            self.name = name
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
        }

        enum CodingKeys: String, CodingKey {
            case role, content, name
            case toolCalls = "tool_calls"
            case toolCallId = "tool_call_id"
        }
    }

    public init(
        model: String,
        messages: [Message],
        stream: Bool? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        stop: [String]? = nil,
        seed: Int? = nil,
        enableThinking: Bool? = nil,
        reasoningEffort: String? = nil,
        tools: [Tool]? = nil,
        toolChoice: ToolChoice? = nil,
        includeReasoning: Bool? = nil,
        sessionId: String? = nil,
        chatId: String? = nil,
        thinkingBudget: Int? = nil
    ) {
        self.model = model
        self.messages = messages
        self.stream = stream
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.stop = stop
        self.seed = seed
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.tools = tools
        self.toolChoice = toolChoice
        self.includeReasoning = includeReasoning
        self.sessionId = sessionId
        self.chatId = chatId
        self.thinkingBudget = thinkingBudget
    }

    /// Multimodal content: either a plain string OR an array of parts (text/image_url).
    public enum ContentValue: Codable, Sendable {
        case string(String)
        case parts([ContentPart])

        public init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .string(s); return }
            self = .parts(try c.decode([ContentPart].self))
        }
        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .string(let s): try c.encode(s)
            case .parts(let p):  try c.encode(p)
            }
        }
    }

    public struct ContentPart: Codable, Sendable {
        public var type: String   // "text" | "image_url" | "video_url" | "audio_url" | "input_audio"
        public var text: String?
        public var imageUrl: ImageURL?
        public var videoUrl: VideoURL?
        public var audioUrl: AudioURL?
        public var inputAudio: InputAudio?

        public struct AudioURL: Codable, Sendable {
            public var url: String
            public init(url: String) {
                self.url = url
            }

            /// Returns a local audio file URL. Supports file://, absolute
            /// paths, data:audio/*;base64, and bounded HTTP(S) downloads.
            public func loadAudioLocalURL() async -> URL? {
                let raw = url
                if raw.hasPrefix("file://") { return URL(string: raw) }
                if raw.hasPrefix("/") { return URL(fileURLWithPath: raw) }
                if raw.hasPrefix("data:") {
                    guard let comma = raw.firstIndex(of: ","),
                          let bytes = Data(
                            base64Encoded: String(raw[raw.index(after: comma)...]),
                            options: .ignoreUnknownCharacters)
                    else { return nil }
                    let ext: String
                    if raw.contains("audio/wav") || raw.contains("audio/x-wav") {
                        ext = "wav"
                    } else if raw.contains("audio/mpeg") || raw.contains("audio/mp3") {
                        ext = "mp3"
                    } else if raw.contains("audio/flac") {
                        ext = "flac"
                    } else {
                        ext = "wav"
                    }
                    let tmp = FileManager.default.temporaryDirectory
                        .appendingPathComponent("vmlx-audio-\(UUID().uuidString).\(ext)")
                    do {
                        try bytes.write(to: tmp)
                        return tmp
                    } catch {
                        return nil
                    }
                }
                if raw.hasPrefix("http://") || raw.hasPrefix("https://") {
                    guard let u = URL(string: raw) else { return nil }
                    do {
                        var req = URLRequest(url: u)
                        req.timeoutInterval = 20
                        let (data, _) = try await URLSession.shared.data(for: req)
                        guard data.count <= 128 * 1024 * 1024 else { return nil }
                        let ext = u.pathExtension.isEmpty ? "wav" : u.pathExtension
                        let tmp = FileManager.default.temporaryDirectory
                            .appendingPathComponent("vmlx-audio-\(UUID().uuidString).\(ext)")
                        try data.write(to: tmp)
                        return tmp
                    } catch {
                        return nil
                    }
                }
                return nil
            }
        }

        public struct InputAudio: Codable, Sendable {
            public var data: String?
            public var format: String?

            public init(data: String? = nil, format: String? = nil) {
                self.data = data
                self.format = format
            }

            public func loadAudioLocalURL() -> URL? {
                guard let data,
                      let bytes = Data(base64Encoded: data, options: .ignoreUnknownCharacters)
                else { return nil }
                let ext = (format?.isEmpty == false ? format! : "wav")
                    .lowercased()
                    .trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
                let tmp = FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx-audio-\(UUID().uuidString).\(ext.isEmpty ? "wav" : ext)")
                do {
                    try bytes.write(to: tmp)
                    return tmp
                } catch {
                    return nil
                }
            }
        }

        public struct VideoURL: Codable, Sendable {
            public var url: String
            public init(url: String) {
                self.url = url
            }
            /// Returns a local file URL that `AVURLAsset` can consume.
            /// - For `file://` inputs, returns the URL as-is.
            /// - For `http(s)://`, downloads bytes to a temp file and
            ///   returns that file URL.
            /// - For bare paths, wraps them in `file://`.
            /// - For `data:` URLs, decodes the base64 body to a temp file
            ///   under the system temp dir.
            /// Returns nil on fetch/decode failure.
            public func loadVideoLocalURL() async -> URL? {
                let raw = url
                if raw.hasPrefix("file://") {
                    return URL(string: raw)
                }
                if raw.hasPrefix("/") {
                    return URL(fileURLWithPath: raw)
                }
                if raw.hasPrefix("data:") {
                    guard let comma = raw.firstIndex(of: ","),
                          let bytes = Data(
                            base64Encoded: String(raw[raw.index(after: comma)...]),
                            options: .ignoreUnknownCharacters)
                    else { return nil }
                    // Best-effort extension — default to .mp4 when not provided.
                    let ext = raw.contains("video/mp4") ? "mp4"
                           : raw.contains("video/quicktime") ? "mov"
                           : "mp4"
                    let tmp = FileManager.default.temporaryDirectory
                        .appendingPathComponent("vmlx-video-\(UUID().uuidString).\(ext)")
                    do {
                        try bytes.write(to: tmp)
                        return tmp
                    } catch {
                        return nil
                    }
                }
                if raw.hasPrefix("http://") || raw.hasPrefix("https://") {
                    guard let u = URL(string: raw) else { return nil }
                    // §334 — bound remote fetch by time + size. Prior
                    // to this, URLSession.shared.data(from:) used the
                    // default 60s timeout and had NO size cap: a
                    // malicious `video_url` could hang the request
                    // path for a full minute + stream megabytes into
                    // process memory before the catch/return-nil on
                    // the `data.write(to: tmp)` line. Videos over
                    // ~512 MB are unlikely to be legitimate in a chat
                    // context — cap the read to 512 MB and fail fast
                    // on hangs at 20 s.
                    do {
                        var req = URLRequest(url: u)
                        req.timeoutInterval = 20
                        let (data, _) = try await URLSession.shared.data(for: req)
                        guard data.count <= 512 * 1024 * 1024 else { return nil }
                        let ext = (u.pathExtension.isEmpty ? "mp4" : u.pathExtension)
                        let tmp = FileManager.default.temporaryDirectory
                            .appendingPathComponent("vmlx-video-\(UUID().uuidString).\(ext)")
                        try data.write(to: tmp)
                        return tmp
                    } catch {
                        return nil
                    }
                }
                return nil
            }
        }
        public struct ImageURL: Codable, Sendable {
            public var url: String
            public var detail: String?

            public init(url: String, detail: String? = nil) {
                self.url = url
                self.detail = detail
            }

            /// Decode the URL field to raw image bytes. Handles four formats:
            ///  1. `data:image/...;base64,<b64>` (OpenAI chat vision)
            ///  2. `file:///absolute/path.png`
            ///  3. `http(s)://host/path.png`
            ///  4. bare base64 (Ollama `images: [String]`)
            public func loadImageData() async -> Data? {
                let raw = url
                // Data URL
                if raw.hasPrefix("data:") {
                    if let comma = raw.firstIndex(of: ",") {
                        let b64 = String(raw[raw.index(after: comma)...])
                        return Data(base64Encoded: b64,
                                    options: .ignoreUnknownCharacters)
                    }
                    return nil
                }
                // File URL
                if raw.hasPrefix("file://") {
                    if let u = URL(string: raw) {
                        return try? Data(contentsOf: u)
                    }
                    return nil
                }
                // HTTP(S) URL
                if raw.hasPrefix("http://") || raw.hasPrefix("https://") {
                    guard let u = URL(string: raw) else { return nil }
                    // §334 — same timeout + size cap as
                    // `loadVideoLocalURL`. Images are typically
                    // <10 MB; cap at 64 MB which comfortably holds
                    // any legitimate 4K RGBA image plus metadata
                    // overhead. 20 s timeout matches the video path.
                    do {
                        var req = URLRequest(url: u)
                        req.timeoutInterval = 20
                        let (data, _) = try await URLSession.shared.data(for: req)
                        guard data.count <= 64 * 1024 * 1024 else { return nil }
                        return data
                    } catch {
                        return nil
                    }
                }
                // Bare base64 (Ollama convention)
                return Data(base64Encoded: raw, options: .ignoreUnknownCharacters)
            }
        }

        public init(
            type: String,
            text: String? = nil,
            imageUrl: ImageURL? = nil,
            videoUrl: VideoURL? = nil,
            audioUrl: AudioURL? = nil,
            inputAudio: InputAudio? = nil
        ) {
            self.type = type
            self.text = text
            self.imageUrl = imageUrl
            self.videoUrl = videoUrl
            self.audioUrl = audioUrl
            self.inputAudio = inputAudio
        }

        enum CodingKeys: String, CodingKey {
            case type, text
            case imageUrl = "image_url"
            case videoUrl = "video_url"
            case audioUrl = "audio_url"
            case inputAudio = "input_audio"
        }
    }

    public struct Tool: Codable, Sendable {
        public var type: String   // "function"
        public var function: Function
        public init(type: String, function: Function) {
            self.type = type
            self.function = function
        }
        public struct Function: Codable, Sendable {
            public var name: String
            public var description: String?
            public var parameters: JSONValue?
            public init(name: String, description: String?, parameters: JSONValue?) {
                self.name = name
                self.description = description
                self.parameters = parameters
            }
        }
    }

    public struct ToolCall: Codable, Sendable {
        public var id: String
        public var type: String
        public var function: Function
        public struct Function: Codable, Sendable {
            public var name: String
            public var arguments: String
        }
    }

    public enum ToolChoice: Codable, Sendable {
        case auto
        case none
        case required
        case function(name: String)

        public init(from decoder: Decoder) throws {
            if let s = try? decoder.singleValueContainer().decode(String.self) {
                switch s {
                case "auto": self = .auto
                case "none": self = .none
                case "required": self = .required
                default: self = .function(name: s)
                }
                return
            }
            // {"type":"function","function":{"name":"..."}}
            struct Wrap: Codable { let function: F; struct F: Codable { let name: String } }
            let w = try decoder.singleValueContainer().decode(Wrap.self)
            self = .function(name: w.function.name)
        }
        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .auto: try c.encode("auto")
            case .none: try c.encode("none")
            case .required: try c.encode("required")
            case .function(let n): try c.encode(n)
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, stop, seed, tools
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
        case toolChoice = "tool_choice"
        case sessionId = "session_id"
        case chatId = "chat_id"
        case n
        case logprobs
        case topLogprobs = "top_logprobs"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case logitBias = "logit_bias"
        case responseFormat = "response_format"
        case streamOptions = "stream_options"
        case includeReasoning = "include_reasoning"
        case chatTemplateKwargs = "chat_template_kwargs"
        case thinkingBudget = "thinking_budget"
        case maxToolIterations = "max_tool_iterations"
        case maxCompletionTokens = "max_completion_tokens"
        // mlxstudio #100 — Continue extension + Anthropic-shaped clients
        // send the nested form `reasoning: { effort: "medium" }` instead
        // of the flat OpenAI `reasoning_effort: "medium"`. Without this
        // alias the field decoded to nil → `effortImpliesThinking` was
        // false → enable_thinking defaulted to false → reasoning never
        // fired even when the user explicitly requested it. Decoded
        // into `reasoningContainer`, folded into `reasoningEffort` via
        // `applyReasoningContainerAlias()` post-decode.
        case reasoningContainer = "reasoning"
    }

    /// Anthropic-shaped reasoning container: `{ "effort": "low|medium|high|max", "type": "..." }`.
    /// Only `effort` is consumed today — other fields are forward-compat
    /// stubs. See `applyReasoningContainerAlias()` for the fold logic.
    public struct ReasoningContainer: Codable, Sendable {
        public var effort: String?
        public var type: String?
        public init(effort: String? = nil, type: String? = nil) {
            self.effort = effort
            self.type = type
        }
    }
    public var reasoningContainer: ReasoningContainer?

    /// OpenAI v2 alias for `max_tokens`. Stored as-is via Codable
    /// auto-synthesis; the route handler folds it into `maxTokens` via
    /// `applyMaxCompletionTokensAlias()` after decoding. Audit P1-API-1.
    public var maxCompletionTokens: Int?

    /// Post-decode hook that folds `max_completion_tokens` into
    /// `max_tokens` when the latter is nil. Idempotent; safe to call
    /// multiple times. Route handlers should call this once on every
    /// decoded ChatRequest before passing it to `Engine.stream()`.
    public mutating func applyMaxCompletionTokensAlias() {
        if maxTokens == nil, let v = maxCompletionTokens, v > 0 {
            maxTokens = v
        }
    }

    /// mlxstudio #100 — fold the nested `reasoning: { effort: "..." }`
    /// container into the flat `reasoning_effort` field. Continue
    /// extension and other Anthropic-shaped clients send the nested
    /// form. The flat field wins if both are present (matches
    /// max_completion_tokens precedence).
    /// Idempotent.
    public mutating func applyReasoningContainerAlias() {
        if reasoningEffort == nil,
           let effort = reasoningContainer?.effort,
           !effort.isEmpty
        {
            reasoningEffort = effort
        }
    }

    /// Anthropic-compat extension: caps the number of tokens generated
    /// inside the reasoning/`<think>` segment. When the budget is
    /// exhausted mid-reasoning, the engine stops emitting additional
    /// reasoning deltas and routes overflow as visible content without
    /// surfacing structural tags.
    /// 0 disables the cap.
    public var thinkingBudget: Int?

    /// vMLX extension: per-request override for the max number of
    /// tool-execution iterations the outer loop will run. Falls back to
    /// `defaultMaxToolIterations` from the settings tier when nil. Caps
    /// runaway tool loops from the request layer instead of forcing a
    /// session-wide setting change.
    public var maxToolIterations: Int?

    /// Validate field ranges + structure. Throws `ChatRequestValidationError`
    /// with a descriptive message on any out-of-range or malformed input.
    ///
    /// Called from the HTTP route handlers before dispatching to
    /// `Engine.stream(request:)` so bad client input surfaces as a
    /// clean 400 Bad Request instead of a mid-stream fatal.
    /// Ranges mirror OpenAI / Anthropic spec bounds where both specs
    /// agree, and pick the wider interval otherwise.
    public func validate() throws {
        if messages.isEmpty {
            throw ChatRequestValidationError(
                field: "messages", reason: "must be non-empty")
        }
        for (idx, m) in messages.enumerated() {
            let role = m.role
            let validRoles: Set<String> = [
                "system", "user", "assistant", "tool", "developer", "function",
            ]
            if !validRoles.contains(role) {
                let allowed = validRoles.sorted().joined(separator: ", ")
                throw ChatRequestValidationError(
                    field: "messages[\(idx)].role",
                    reason: "unknown role `\(role)` — expected one of \(allowed)")
            }
        }
        if let t = temperature, !(t >= 0 && t <= 2) {
            throw ChatRequestValidationError(
                field: "temperature", reason: "must be in [0, 2], got \(t)")
        }
        if let p = topP, !(p > 0 && p <= 1) {
            throw ChatRequestValidationError(
                field: "top_p", reason: "must be in (0, 1], got \(p)")
        }
        if let k = topK, k < 0 {
            throw ChatRequestValidationError(
                field: "top_k", reason: "must be >= 0, got \(k)")
        }
        if let p = minP, !(p >= 0 && p <= 1) {
            throw ChatRequestValidationError(
                field: "min_p", reason: "must be in [0, 1], got \(p)")
        }
        if let rp = repetitionPenalty, !(rp > 0 && rp <= 5) {
            throw ChatRequestValidationError(
                field: "repetition_penalty",
                reason: "must be in (0, 5], got \(rp)")
        }
        if let mt = maxTokens, mt <= 0 {
            throw ChatRequestValidationError(
                field: "max_tokens", reason: "must be > 0, got \(mt)")
        }
        // §438 — F5 from API audit. Reject pathologically large
        // max_tokens at request boundary instead of letting it
        // through to forward() where it would either OOM the KV
        // cache (paged or hybrid) or silently truncate. Cap at
        // 1M tokens — above any real model's context window today.
        // Per-model context-window enforcement requires the loaded
        // model's `maxPositionEmbeddings` and lives in Stream.swift,
        // but a conservative absolute upper bound here catches
        // misconfigured clients before they hit the loader.
        if let mt = maxTokens, mt > 1_000_000 {
            throw ChatRequestValidationError(
                field: "max_tokens",
                reason: "exceeds absolute upper bound 1_000_000, got \(mt)")
        }
        if let s = stop, s.count > 16 {
            throw ChatRequestValidationError(
                field: "stop",
                reason: "at most 16 stop sequences supported, got \(s.count)")
        }
        // iter-114 §140: empty stop-sequence string is silently filtered
        // at AhoCorasick init (see `AhoCorasick.swift:58 where !p.isEmpty`)
        // so it's not a crash risk, but the API caller who sent
        // `stop: [""]` expecting a real match gets a silent no-op —
        // generation ignores their stop and runs to max_tokens. Reject
        // with a clean 400 so SDK clients see the actual contract.
        // Per-entry check so the error message names the offending index.
        if let s = stop {
            for (i, entry) in s.enumerated() where entry.isEmpty {
                throw ChatRequestValidationError(
                    field: "stop[\(i)]",
                    reason: "stop sequences must be non-empty strings")
            }
        }

        // iter-121 §147: tools field validation. OpenAI spec requires
        // tool `function.name` to match `^[a-zA-Z0-9_-]{1,64}$` and
        // we had no enforcement at all. Consequences of the gap:
        // (1) count-blowup — a client could attach 1000 tools; each
        // serializes into the chat template prompt, easily filling
        // the context window before the first user turn. (2) prompt
        // injection — a tool named `x\n\nIgnore prior instructions`
        // splices an attacker-controlled instruction into the
        // serialized template. (3) missing-name — empty-string
        // tool.name passes JSON decode but produces a malformed
        // <tool_call> block at format time. Enforce the spec: cap at
        // 128 tools, require each name to be non-empty + 1-64 ASCII
        // `[a-zA-Z0-9_-]`, error message names the offending index.
        if let tools = tools {
            if tools.count > 128 {
                throw ChatRequestValidationError(
                    field: "tools",
                    reason: "at most 128 tools supported, got \(tools.count)")
            }
            let nameCharset = CharacterSet(charactersIn:
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
            for (i, t) in tools.enumerated() {
                let name = t.function.name
                if name.isEmpty {
                    throw ChatRequestValidationError(
                        field: "tools[\(i)].function.name",
                        reason: "tool name must be non-empty")
                }
                if name.count > 64 {
                    throw ChatRequestValidationError(
                        field: "tools[\(i)].function.name",
                        reason: "tool name must be at most 64 characters, got \(name.count)")
                }
                if name.rangeOfCharacter(from: nameCharset.inverted) != nil {
                    throw ChatRequestValidationError(
                        field: "tools[\(i)].function.name",
                        reason: "tool name must match ^[a-zA-Z0-9_-]{1,64}$, got `\(name)`")
                }
            }
        }

        // MARK: Decoded-but-not-yet-fully-implemented OpenAI fields.
        //
        // **Silent accept.** Every mainstream OpenAI client (openai-python,
        // openai-node, LangChain, LiteLLM, LlamaIndex, Open WebUI, Cline,
        // Aider, Continue.dev, Chatbox) always sends `presence_penalty=0`,
        // `frequency_penalty=0`, often `logprobs=false`, often
        // `response_format={"type":"text"}` as defaults from their config.
        // Returning 400 on these breaks every single SDK round-trip on the
        // first request, regardless of whether the user actually meant to
        // toggle the feature. The Swift engine previously rejected all five
        // — broke ~100% of OpenAI SDK callers in the wild.
        //
        // The new contract: validate **range** of values that ARE present,
        // accept-silently otherwise. The values are still parsed and
        // available on `ChatRequest` for any future implementation; we just
        // don't fail the request when they're set to a no-op default OR a
        // value we can't honor yet. Range-checking still rejects nonsense
        // (`presence_penalty=42`, `top_p=-1`).
        //
        // The `n != 1` case stays a hard reject because vMLX truly cannot
        // produce multiple completions in one pass — silently producing
        // one when the caller asked for five would corrupt downstream code.
        if let n = n, n != 1 {
            throw ChatRequestValidationError(
                field: "n",
                reason: "vMLX generates exactly one completion per request; `n=\(n)` is not supported. Issue multiple requests or use batching.")
        }
        if let fp = frequencyPenalty, !(fp >= -2 && fp <= 2) {
            throw ChatRequestValidationError(
                field: "frequency_penalty",
                reason: "must be in [-2, 2], got \(fp)")
        }
        if let pp = presencePenalty, !(pp >= -2 && pp <= 2) {
            throw ChatRequestValidationError(
                field: "presence_penalty",
                reason: "must be in [-2, 2], got \(pp)")
        }
        if let tlp = topLogprobs, !(tlp >= 0 && tlp <= 20) {
            throw ChatRequestValidationError(
                field: "top_logprobs",
                reason: "must be in [0, 20], got \(tlp)")
        }
        // PR #99 (iter-96 §123 + §163 B1-B6 fixups): logprobs is now
        // supported via Evaluate.swift's LogprobsCollector. Replaces the
        // pre-PR hard-reject guard (audit 2026-04-15) that surfaced a 400
        // when callers asked for logprobs. `logit_bias` / `response_format`
        // remain accepted-but-unwired — the Engine ignores the values and
        // the response omits the corresponding fields. See the `logitBias`
        // property doc above for the processor-wiring plan.
        //
        // `frequency_penalty` / `presence_penalty` were wired in §173.
    }
}

/// Thrown by `ChatRequest.validate()` when an input field is
/// out-of-range or malformed. Route handlers catch this and return
/// a 400 response with `{"error": {"message": ..., "type": "invalid_request_error"}}`.
public struct ChatRequestValidationError: Error, CustomStringConvertible, Sendable {
    public let field: String
    public let reason: String
    public init(field: String, reason: String) {
        self.field = field
        self.reason = reason
    }
    public var description: String {
        "invalid request: \(field): \(reason)"
    }
}

/// One streaming delta emitted by `Engine.stream`. Maps 1:1 to OpenAI SSE chunks.
///
/// Field semantics — matches `vmlx_engine/server.py::_stream_chat_completions`:
/// - `content` — visible assistant text. May be empty during reasoning phase.
/// - `reasoning` — `<think>` content. UI must render this OR fall through to
///   content when `enable_thinking == false` to honor §15 of NO-REGRESSION.
/// - `toolCallDelta` — incremental tool-call argument streaming
///   (`tool_calls[].function.arguments` arrives in chunks like content).
/// - `toolStatus` — non-final tool-call lifecycle events (started/running/done)
///   used by the chat UI's `InlineToolCall` cards.
/// - `finishReason` — final chunk sets this; UI uses to flip streaming → done.
/// - `usage` — final chunk carries metrics; per-message UI strip reads this.
public struct StreamChunk: Sendable {
    public var content: String?
    public var reasoning: String?
    public var toolCalls: [ChatRequest.ToolCall]?
    public var toolCallDelta: ToolCallDelta?
    public var toolStatus: ToolStatus?
    public var finishReason: String?
    public var usage: Usage?
    /// Per-token log probability data. Non-nil when `logprobs: true`
    /// was set on the request. Each entry corresponds to one generated token.
    public var logprobs: [TokenLogprob]?

    public init(
        content: String? = nil,
        reasoning: String? = nil,
        toolCalls: [ChatRequest.ToolCall]? = nil,
        toolCallDelta: ToolCallDelta? = nil,
        toolStatus: ToolStatus? = nil,
        finishReason: String? = nil,
        usage: Usage? = nil,
        logprobs: [TokenLogprob]? = nil
    ) {
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.toolCallDelta = toolCallDelta
        self.toolStatus = toolStatus
        self.finishReason = finishReason
        self.usage = usage
        self.logprobs = logprobs
    }

    /// Per-message metrics surfaced under each assistant turn. Mirrors
    /// `panel/src/renderer/src/lib/chat-utils.ts::getMetricsItems` 1:1.
    public struct Usage: Sendable {
        public var promptTokens: Int
        public var completionTokens: Int
        public var cachedTokens: Int
        /// Generation throughput (decoded tokens / second).
        public var tokensPerSecond: Double?
        /// Prefill throughput (prompt tokens / second).
        public var promptTokensPerSecond: Double?
        /// Time to first token (ms).
        public var ttftMs: Double?
        /// Prefill wall-clock time (ms). Populated from
        /// `GenerateCompletionInfo.promptTime * 1000` on the final chunk;
        /// `nil` on partial usage emissions during streaming.
        public var prefillMs: Double?
        /// Total wall-clock time from request received to final chunk (ms).
        public var totalMs: Double?
        /// Human-readable cache breakdown e.g. `"paged+ssm(23)+tq"`.
        public var cacheDetail: String?
        /// `true` when this usage is a live partial emitted DURING the stream
        /// (not the authoritative final totals). UI can use this to drive a
        /// "live" indicator without disturbing the finalized metrics strip.
        public var isPartial: Bool

        public init(
            promptTokens: Int = 0,
            completionTokens: Int = 0,
            cachedTokens: Int = 0,
            tokensPerSecond: Double? = nil,
            promptTokensPerSecond: Double? = nil,
            ttftMs: Double? = nil,
            prefillMs: Double? = nil,
            totalMs: Double? = nil,
            cacheDetail: String? = nil,
            isPartial: Bool = false
        ) {
            self.promptTokens = promptTokens
            self.completionTokens = completionTokens
            self.cachedTokens = cachedTokens
            self.tokensPerSecond = tokensPerSecond
            self.promptTokensPerSecond = promptTokensPerSecond
            self.ttftMs = ttftMs
            self.prefillMs = prefillMs
            self.totalMs = totalMs
            self.cacheDetail = cacheDetail
            self.isPartial = isPartial
        }
    }

    /// Incremental tool-call streaming. One delta per OpenAI chunk —
    /// `index` identifies which tool call is being appended to.
    public struct ToolCallDelta: Sendable {
        public var index: Int
        public var id: String?
        public var name: String?
        public var argumentsDelta: String?
        public init(index: Int, id: String? = nil, name: String? = nil, argumentsDelta: String? = nil) {
            self.index = index
            self.id = id
            self.name = name
            self.argumentsDelta = argumentsDelta
        }
    }

    /// Tool-call lifecycle event for the inline tool-call card UI.
    public struct ToolStatus: Sendable {
        public enum Phase: String, Sendable { case started, running, done, error }
        public var toolCallId: String
        public var name: String
        public var phase: Phase
        public var message: String?
        public init(toolCallId: String, name: String, phase: Phase, message: String? = nil) {
            self.toolCallId = toolCallId
            self.name = name
            self.phase = phase
            self.message = message
        }
    }
}

/// Minimal JSON value type for tool parameter passthrough.
public indirect enum JSONValue: Codable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let n = try? c.decode(Double.self) { self = .number(n); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let a = try? c.decode([JSONValue].self) { self = .array(a); return }
        if let o = try? c.decode([String: JSONValue].self) { self = .object(o); return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "unknown JSON value")
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null:        try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let n): try c.encode(n)
        case .string(let s): try c.encode(s)
        case .array(let a):  try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }
}
