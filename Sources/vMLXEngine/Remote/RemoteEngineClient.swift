import Foundation

/// HTTP client that speaks to a remote vMLX / OpenAI / Ollama / Anthropic
/// server as if it were a local `Engine`. Used by Chat + Terminal modes
/// when a session is configured with `SessionSettings.remoteURL`.
///
/// Design:
///
/// - Same `stream(request:)` return type as `Engine.stream(request:)` so
///   the view-model dispatch path can be a one-liner ternary. Returns an
///   `AsyncThrowingStream<StreamChunk, Error>`.
///
/// - Protocol is a string so it can round-trip through the Settings DB
///   without a new Codable case. Values: "openai" (default, broad compat),
///   "ollama" (maps /api/chat), "anthropic" (maps /v1/messages).
///
/// - Auth: Bearer for openai + vmlx, X-API-Key for ollama if provided,
///   x-api-key + anthropic-version for anthropic. The `apiKey` is sourced
///   from SessionSettings but resolved by the caller — this actor treats
///   it as opaque.
///
/// - Cancellation: `cancelStream()` cancels the in-flight URLSession data
///   task. The client is one-shot per stream call — no session pooling —
///   so cancelling also abandons the bytes iterator.
///
/// - Errors: HTTP >= 400 are surfaced as `RemoteEngineClientError.http` so
///   the UI can show a clean banner. Network errors propagate unwrapped.
public actor RemoteEngineClient {

    // MARK: - Types

    public enum Kind: String, Sendable {
        case openai, ollama, anthropic
        public init(rawOrDefault raw: String?) {
            guard let raw, let k = Kind(rawValue: raw.lowercased()) else {
                self = .openai
                return
            }
            self = k
        }
    }

    public enum RemoteError: Error, LocalizedError {
        case badURL
        case http(status: Int, body: String)
        case malformedResponse(String)
        case notImplemented(Kind)
        case cancelled

        public var errorDescription: String? {
            switch self {
            case .badURL: return "Bad remote URL"
            case .http(let s, let b): return "Remote returned \(s): \(b.prefix(200))"
            case .malformedResponse(let msg): return "Malformed response: \(msg)"
            case .notImplemented(let k): return "Remote \(k.rawValue) support not implemented"
            case .cancelled: return "Cancelled"
            }
        }
    }

    // MARK: - Config

    public let endpoint: URL
    public let kind: Kind
    public let apiKey: String?
    public let modelName: String

    // MARK: - State

    private var liveTask: URLSessionDataTask?

    public init(endpoint: URL, kind: Kind, apiKey: String?, modelName: String) {
        self.endpoint = endpoint
        self.kind = kind
        self.apiKey = (apiKey?.isEmpty == true) ? nil : apiKey
        self.modelName = modelName
    }

    // MARK: - Public API (mirrors Engine.stream)

    /// Start a streaming chat request against the remote endpoint. Returns
    /// an AsyncThrowingStream of `StreamChunk` values shaped identically to
    /// the local `Engine.stream(request:)` output, so the view layer can
    /// swap engines without a branch.
    public func stream(request: ChatRequest) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    switch self.kind {
                    case .openai:
                        try await self.streamOpenAI(request: request, to: continuation)
                    case .ollama:
                        try await self.streamOllama(request: request, to: continuation)
                    case .anthropic:
                        try await self.streamAnthropic(request: request, to: continuation)
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: RemoteError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Cancel the in-flight request, if any.
    public func cancelStream() {
        liveTask?.cancel()
        liveTask = nil
    }

    /// GET /v1/models (OpenAI) or /api/tags (Ollama). Returns model ids.
    /// Used by the UI to populate a picker of what's available on the
    /// remote endpoint before the user hits Send.
    public func listModels() async throws -> [String] {
        switch kind {
        case .openai:
            let url = endpoint.appendingPathComponent("v1").appendingPathComponent("models")
            var req = URLRequest(url: url)
            applyAuth(to: &req)
            let (data, response) = try await URLSession.shared.data(for: req)
            try checkHTTP(response, body: data)
            struct ModelsResp: Decodable { let data: [ModelRow] }
            struct ModelRow: Decodable { let id: String }
            let parsed = try JSONDecoder().decode(ModelsResp.self, from: data)
            return parsed.data.map(\.id)
        case .ollama:
            let url = endpoint.appendingPathComponent("api").appendingPathComponent("tags")
            var req = URLRequest(url: url)
            applyAuth(to: &req)
            let (data, response) = try await URLSession.shared.data(for: req)
            try checkHTTP(response, body: data)
            struct TagsResp: Decodable { let models: [TagRow] }
            struct TagRow: Decodable { let name: String }
            let parsed = try JSONDecoder().decode(TagsResp.self, from: data)
            return parsed.models.map(\.name)
        case .anthropic:
            // Anthropic has no public listing endpoint; return a static set
            // of currently-supported model ids so the UI picker still works.
            return [
                "claude-opus-4-5",
                "claude-sonnet-4-5",
                "claude-haiku-4-5",
            ]
        }
    }

    // MARK: - OpenAI-compatible streaming

    private func streamOpenAI(
        request: ChatRequest,
        to continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws {
        let url = endpoint
            .appendingPathComponent("v1")
            .appendingPathComponent("chat")
            .appendingPathComponent("completions")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        applyAuth(to: &req)

        // Rewrite model id to whatever the remote endpoint expects and
        // force stream=true — the caller may have set it elsewhere.
        var body = request
        body.model = modelName
        body.stream = true
        req.httpBody = try JSONEncoder().encode(body)

        let (bytes, response) = try await URLSession.shared.bytes(for: req)
        try checkHTTPAsync(response)

        // Server-Sent Events parser: each chunk is `data: {json}\n\n`.
        // iter-97 §124: accumulate raw bytes, not per-byte Unicode
        // scalars. The earlier per-byte-scalar pipeline (see git
        // history) treated every UInt8 as a separate scalar in
        // U+0000..U+00FF, corrupting UTF-8 multi-byte sequences —
        // a non-ASCII content byte `0xE2` would become scalar
        // U+00E2 instead of being part of a `[0xE2, 0x9C, 0x85]`
        // (✅) sequence. data(using: .utf8) then re-encoded each
        // character to a 2-byte UTF-8 sequence, producing garbled
        // bytes that didn't match the original payload. JSON decode
        // happened to still succeed (structural chars are ASCII)
        // but string VALUES contained mojibake. Real breakage for
        // remote OpenAI-compatible servers serving emoji / accented
        // chars / CJK content. Fix: accumulate as [UInt8], decode
        // to String with String(bytes:encoding: .utf8) on the line
        // boundary.
        var lineBytes: [UInt8] = []
        for try await byte in bytes {
            if Task.isCancelled { throw CancellationError() }
            lineBytes.append(byte)
            if byte == 0x0A {  // '\n'
                defer { lineBytes.removeAll(keepingCapacity: true) }
                guard let line = String(bytes: lineBytes, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                else { continue }
                guard line.hasPrefix("data: ") else { continue }
                let payload = String(line.dropFirst("data: ".count))
                if payload == "[DONE]" { break }
                if let chunk = Self.decodeOpenAIChunk(payload) {
                    continuation.yield(chunk)
                }
            }
        }
    }

    /// Decode one SSE payload into a StreamChunk. Returns nil on malformed
    /// JSON — we log-and-skip rather than tearing down the whole stream on
    /// a single bad line, matching how most OpenAI SDKs behave.
    private static func decodeOpenAIChunk(_ payload: String) -> StreamChunk? {
        guard let data = payload.data(using: .utf8) else { return nil }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let choices = obj["choices"] as? [[String: Any]] ?? []
        guard let first = choices.first else { return nil }
        let delta = first["delta"] as? [String: Any] ?? [:]
        let content = delta["content"] as? String
        let reasoning = delta["reasoning"] as? String
            ?? delta["reasoning_content"] as? String
        let finish = first["finish_reason"] as? String

        // Parse streaming tool_calls delta. OpenAI sends fragments keyed by
        // index with function.name + function.arguments substrings; we pass
        // them through as-is so downstream parsers can merge.
        var toolCalls: [ChatRequest.ToolCall]? = nil
        if let rawCalls = delta["tool_calls"] as? [[String: Any]] {
            var out: [ChatRequest.ToolCall] = []
            for raw in rawCalls {
                let id = raw["id"] as? String ?? ""
                let type = raw["type"] as? String ?? "function"
                let fn = raw["function"] as? [String: Any] ?? [:]
                let name = fn["name"] as? String ?? ""
                let args = fn["arguments"] as? String ?? ""
                out.append(ChatRequest.ToolCall(
                    id: id, type: type,
                    function: ChatRequest.ToolCall.Function(
                        name: name, arguments: args
                    )
                ))
            }
            toolCalls = out.isEmpty ? nil : out
        }

        // Usage only appears on the final chunk when stream_options
        // .include_usage is set on the request.
        var usage: StreamChunk.Usage? = nil
        if let u = obj["usage"] as? [String: Any] {
            let prompt = u["prompt_tokens"] as? Int ?? 0
            let completion = u["completion_tokens"] as? Int ?? 0
            usage = StreamChunk.Usage(
                promptTokens: prompt,
                completionTokens: completion
            )
        }

        return StreamChunk(
            content: content,
            reasoning: reasoning,
            toolCalls: toolCalls,
            toolStatus: nil,
            finishReason: finish,
            usage: usage
        )
    }

    // MARK: - Ollama streaming

    private func streamOllama(
        request: ChatRequest,
        to continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws {
        let url = endpoint.appendingPathComponent("api").appendingPathComponent("chat")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        applyAuth(to: &req)

        // Ollama request shape: { model, messages, stream:true, options:{...} }
        var options: [String: Any] = [:]
        if let t = request.temperature { options["temperature"] = t }
        if let p = request.topP { options["top_p"] = p }
        if let k = request.topK { options["top_k"] = k }
        if let m = request.maxTokens { options["num_predict"] = m }

        let messages: [[String: Any]] = request.messages.map { m in
            var out: [String: Any] = ["role": m.role]
            switch m.content {
            case .string(let s): out["content"] = s
            case .parts(let parts):
                // Ollama accepts content as a string; flatten multimodal parts.
                out["content"] = parts.compactMap { $0.text }.joined(separator: "\n")
            case .none: out["content"] = ""
            }
            return out
        }

        let body: [String: Any] = [
            "model": modelName,
            "messages": messages,
            "stream": true,
            "options": options,
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (bytes, response) = try await URLSession.shared.bytes(for: req)
        try checkHTTPAsync(response)

        // Ollama NDJSON: one JSON object per line, terminated by `"done":true`.
        // iter-97 §124: same UTF-8-safe byte accumulation as the OpenAI
        // path. Ollama responses routinely carry non-ASCII content
        // (Chinese Qwen responses, tokens with accents) — the earlier
        // per-byte-scalar pipeline mojibake'd them.
        var lineBytes: [UInt8] = []
        for try await byte in bytes {
            if Task.isCancelled { throw CancellationError() }
            lineBytes.append(byte)
            if byte == 0x0A {
                defer { lineBytes.removeAll(keepingCapacity: true) }
                guard let line = String(bytes: lineBytes, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                else { continue }
                guard !line.isEmpty,
                      let data = line.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                else { continue }
                let msg = obj["message"] as? [String: Any]
                let content = msg?["content"] as? String
                let done = obj["done"] as? Bool ?? false
                let chunk = StreamChunk(
                    content: content,
                    reasoning: nil,
                    toolCalls: nil,
                    toolStatus: nil,
                    finishReason: done ? "stop" : nil,
                    usage: nil
                )
                continuation.yield(chunk)
                if done { break }
            }
        }
    }

    // MARK: - Anthropic streaming

    private func streamAnthropic(
        request: ChatRequest,
        to continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws {
        let url = endpoint.appendingPathComponent("v1").appendingPathComponent("messages")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        req.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        if let key = apiKey {
            req.setValue(key, forHTTPHeaderField: "x-api-key")
        }

        // Anthropic: system is a top-level field, not a message role.
        var system: String? = nil
        var messages: [[String: Any]] = []
        for m in request.messages {
            let text: String
            switch m.content {
            case .string(let s): text = s
            case .parts(let parts): text = parts.compactMap { $0.text }.joined(separator: "\n")
            case .none: text = ""
            }
            if m.role == "system" {
                system = (system.map { $0 + "\n" } ?? "") + text
            } else {
                messages.append(["role": m.role, "content": text])
            }
        }

        var body: [String: Any] = [
            "model": modelName,
            "messages": messages,
            "stream": true,
            "max_tokens": request.maxTokens ?? 4096,
        ]
        if let s = system { body["system"] = s }
        if let t = request.temperature { body["temperature"] = t }
        if let p = request.topP { body["top_p"] = p }
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (bytes, response) = try await URLSession.shared.bytes(for: req)
        try checkHTTPAsync(response)

        // Anthropic SSE stream: event: content_block_delta → data: {delta:{text}}
        // iter-97 §124: UTF-8-safe byte accumulation. Anthropic's
        // content_block_delta text field is the user-visible response
        // body — corrupting it on any non-ASCII character was a real
        // breakage for Claude responses containing emoji or accented
        // chars.
        var lineBytes: [UInt8] = []
        for try await byte in bytes {
            if Task.isCancelled { throw CancellationError() }
            lineBytes.append(byte)
            if byte == 0x0A {
                defer { lineBytes.removeAll(keepingCapacity: true) }
                guard let line = String(bytes: lineBytes, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                else { continue }
                guard line.hasPrefix("data: ") else { continue }
                let payload = String(line.dropFirst("data: ".count))
                if payload == "[DONE]" { break }
                guard let data = payload.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                else { continue }
                let type = obj["type"] as? String
                if type == "content_block_delta" {
                    if let delta = obj["delta"] as? [String: Any],
                       let text = delta["text"] as? String
                    {
                        continuation.yield(StreamChunk(
                            content: text, reasoning: nil,
                            toolCalls: nil, toolStatus: nil,
                            finishReason: nil, usage: nil
                        ))
                    }
                } else if type == "message_stop" {
                    break
                }
            }
        }
    }

    // MARK: - HTTP helpers

    private func applyAuth(to request: inout URLRequest) {
        guard let key = apiKey, !key.isEmpty else { return }
        switch kind {
        case .openai, .ollama:
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        case .anthropic:
            request.setValue(key, forHTTPHeaderField: "x-api-key")
        }
    }

    private func checkHTTP(_ response: URLResponse, body: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        if http.statusCode >= 400 {
            let msg = String(data: body, encoding: .utf8) ?? ""
            throw RemoteError.http(status: http.statusCode, body: msg)
        }
    }

    private func checkHTTPAsync(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        if http.statusCode >= 400 {
            throw RemoteError.http(status: http.statusCode, body: "streaming error")
        }
    }
}
