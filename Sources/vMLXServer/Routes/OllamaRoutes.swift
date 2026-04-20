import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import vMLXEngine

/// Ollama-compatible routes.
///
/// Python source: `vmlx_engine/server.py`
///   - POST /api/chat      — line 3221, adapts via ollama_adapter.py then calls stream_chat_completion
///   - POST /api/generate  — line 3318
///   - GET  /api/tags      — line 3177
///   - POST /api/show      — line 3203
///   - POST /api/pull      — line 3396 (no-op)
///   - GET  /api/version   — line 3197
///   - GET  /api/ps        — line 3186
///
/// Key behavioral requirement: `enable_thinking` 3-tier precedence
///   per-request > chat_template_kwargs > server default (server.py:3279-3285).
public enum OllamaRoutes {

    public static func register<Context: RequestContext>(
        on router: Router<Context>,
        engine: Engine
    ) {
        router.get("/api/version") { _, _ -> Response in
            // Pin to a recent Ollama version string. VS Code Copilot and
            // Open WebUI feature-gate on >= 0.6.4 (NO-REGRESSION-CHECKLIST §8f);
            // the older "0.1.0-vmlx" silently disabled tool-call paths in
            // those clients. Mirrors Python server.py at v1.3.50.
            OpenAIRoutes.json(["version": "0.12.6"])
        }

        // GET /api/tags — enumerate local models.
        //
        // Ollama clients call this to populate their model picker and
        // will show an empty list if this endpoint returns `{"models":[]}`.
        // We serialize `ModelLibrary.entries()` into Ollama's schema:
        // `{name, model, modified_at, size, digest, details}`.
        router.get("/api/tags") { _, _ -> Response in
            // On-demand scan — keeps `/api/tags` in sync with disk even
            // when the server was started with `vmlxctl serve --model <path>`
            // (which skips the HF cache walk). Freshness window makes
            // subsequent hits a no-op.
            _ = await engine.modelLibrary.scan(force: false)
            let entries = await engine.modelLibrary.entries()
            let models: [[String: Any]] = entries.map { e in
                let iso = ISO8601DateFormatter().string(from: e.detectedAt)
                return [
                    "name": e.displayName,
                    "model": e.displayName,
                    "modified_at": iso,
                    "size": e.totalSizeBytes,
                    "digest": e.id,
                    "details": [
                        "parent_model": "",
                        "format": e.isJANG ? "jang" : (e.isMXTQ ? "mxtq" : "mlx"),
                        "family": e.family,
                        "families": [e.family],
                        "parameter_size": Self.humanParamSize(e.totalSizeBytes, quantBits: e.quantBits),
                        "quantization_level": e.quantBits.map { "Q\($0)" } ?? "",
                    ] as [String: Any],
                ]
            }
            return OpenAIRoutes.json(["models": models])
        }

        // GET /api/ps — currently-loaded model list. Since the Swift app
        // runs one engine per session, report whatever is loaded now.
        //
        // iter-37: was returning `path.lastPathComponent` which for HF
        // cache layouts is the 40-char snapshot hash, not the clean
        // `org/repo` slug. Ollama clients (Copilot, Open WebUI) key off
        // this field to find the model — if the name is a hash, the
        // picker can't match anything. Now we look up the ModelLibrary
        // entry by canonical path and return its resolved displayName,
        // with `path.lastPathComponent` as a safe fallback when the
        // library doesn't have an entry yet (first-load race).
        router.get("/api/ps") { _, _ -> Response in
            let loadedPath = await engine.loadedModelPath
            guard let path = loadedPath else {
                return OpenAIRoutes.json(["models": []])
            }
            // Ensure the library has scanned at least once so
            // `displayName` is actually populated. `scan(force:false)`
            // hits the 5-minute freshness cache, so this is cheap.
            _ = await engine.modelLibrary.scan(force: false)
            let entries = await engine.modelLibrary.entries()
            let canonical = path.resolvingSymlinksInPath().standardizedFileURL
            let displayName = entries.first(where: {
                $0.canonicalPath.standardizedFileURL == canonical
            })?.displayName ?? path.lastPathComponent
            let iso = ISO8601DateFormatter().string(from: Date())
            return OpenAIRoutes.json([
                "models": [[
                    "name": displayName,
                    "model": displayName,
                    "size": 0,
                    "digest": "",
                    "details": [:] as [String: Any],
                    "expires_at": iso,
                    "size_vram": 0,
                ]]
            ])
        }

        // POST /api/show — return modelfile-style details for a named model.
        router.post("/api/show") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 1024 * 1024)
            let data = Data(buffer: body)
            let obj = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            let name = (obj["name"] as? String) ?? (obj["model"] as? String) ?? ""
            _ = await engine.modelLibrary.scan(force: false)
            let entries = await engine.modelLibrary.entries()
            guard let e = entries.first(where: { $0.displayName == name }) else {
                return OpenAIRoutes.errorJSON(.notFound, "model not found: \(name)")
            }
            // Capabilities array — Ollama 0.20.x clients (GitHub Copilot,
            // Open WebUI) filter the picker by this field. Extracted to
            // `OllamaCapabilities` so `swift test` covers the classifier.
            let caps = OllamaCapabilities.capabilities(family: e.family,
                                                       modality: e.modality)
            // iter-115 §141: the `modelfile` field previously emitted
            // `# vMLX JANG: true\nPATH $HOME/.cache/huggingface/...`
            // which leaked the user's absolute filesystem path on disk.
            // Any API-key-holding client (including LAN peers when the
            // server is bound to 0.0.0.0) could learn home-dir naming,
            // HF-cache layout, and sometimes credential-adjacent path
            // components. Replace with the Ollama-standard `FROM <name>`
            // stanza — what real Ollama emits and what SDK clients
            // actually parse (openai-python, LangChain, Copilot all
            // ignore everything except FROM + PARAMETER + TEMPLATE).
            let modelfile = """
                # vMLX model
                FROM \(e.displayName)
                # format: \(e.isJANG ? "jang" : "mlx")

                """
            return OpenAIRoutes.json([
                "license": "",
                "modelfile": modelfile,
                "parameters": "",
                "template": "",
                "details": [
                    "parent_model": "",
                    "format": e.isJANG ? "jang" : "mlx",
                    "family": e.family,
                    "families": [e.family],
                    "parameter_size": Self.humanParamSize(e.totalSizeBytes, quantBits: e.quantBits),
                    "quantization_level": e.quantBits.map { "Q\($0)" } ?? "",
                ] as [String: Any],
                "capabilities": caps,
            ])
        }

        // POST /api/pull — dispatch to the engine's DownloadManager and
        // stream NDJSON progress events back in Ollama's wire format.
        // Body: `{"name": "<repo>" [, "stream": true]}`. The repo string
        // is treated as a HuggingFace `org/model` id (Ollama's own
        // registry isn't mirrored — vMLX pulls from HF Hub instead).
        //
        // NDJSON format mirrors `ollama pull`:
        //   {"status":"pulling manifest"}
        //   {"status":"downloading","digest":"sha256:...","total":N,"completed":N}
        //   ...
        //   {"status":"success"}
        //
        // Previously this returned a hardcoded `{"status":"success"}`
        // immediately, so Open WebUI / Ollama CLI / LangChain Ollama
        // clients all thought the model had been pulled when nothing
        // had actually downloaded. Build-state audit 2026-04-15 #3.
        router.post("/api/pull") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 64 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let repo = (obj["name"] as? String) ?? (obj["model"] as? String),
                  !repo.isEmpty
            else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing 'name' or 'model'")
            }
            let isStream = (obj["stream"] as? Bool) ?? true

            // Kick off the download via the engine's DownloadManager.
            // Returns immediately; the run task fires .progress / .completed
            // events on the shared subscribe stream.
            let displayName = repo.split(separator: "/").last.map(String.init) ?? repo
            let jobId = await engine.downloadManager.enqueue(
                repo: repo, displayName: displayName)

            if !isStream {
                // Non-streaming: block until the job terminates, then
                // return a single success/failure JSON.
                for await event in await engine.downloadManager.subscribe() {
                    switch event {
                    case .completed(let job) where job.id == jobId:
                        return OpenAIRoutes.json(["status": "success"])
                    case .failed(let id, let err) where id == jobId:
                        return OpenAIRoutes.errorJSON(
                            .internalServerError,
                            "pull failed: \(err)")
                    default: break
                    }
                }
                return OpenAIRoutes.json(["status": "success"])
            }

            // Streaming: emit Ollama-shape NDJSON status events for the
            // lifecycle of THIS job id. Each line is a JSON object on its
            // own line, terminated with \n. The final line is either
            // `{"status":"success"}` or `{"status":"error","message":...}`.
            var headers: HTTPFields = [:]
            headers[.contentType] = "application/x-ndjson"
            headers[.cacheControl] = "no-cache"
            return Response(
                status: .ok,
                headers: headers,
                body: ResponseBody { writer in
                    let allocator = ByteBufferAllocator()
                    func emit(_ payload: [String: Any]) async throws {
                        let data = try JSONSerialization.data(withJSONObject: payload)
                        var buf = allocator.buffer(capacity: data.count + 1)
                        buf.writeBytes(data)
                        buf.writeBytes("\n".utf8)
                        try await writer.write(buf)
                    }

                    try await emit(["status": "pulling manifest"])

                    for await event in await engine.downloadManager.subscribe() {
                        switch event {
                        case .progress(let job) where job.id == jobId:
                            var line: [String: Any] = [
                                "status": "downloading",
                                "digest": "sha256:\(job.id.uuidString.prefix(12).lowercased())",
                            ]
                            if job.totalBytes > 0 {
                                line["total"] = job.totalBytes
                                line["completed"] = job.receivedBytes
                            }
                            try await emit(line)
                        case .completed(let job) where job.id == jobId:
                            try await emit(["status": "success"])
                            try await writer.finish(nil)
                            return
                        case .failed(let id, let err) where id == jobId:
                            try await emit([
                                "status": "error",
                                "message": "\(err)",
                            ])
                            try await writer.finish(nil)
                            return
                        case .cancelled(let id) where id == jobId:
                            try await emit([
                                "status": "error",
                                "message": "cancelled",
                            ])
                            try await writer.finish(nil)
                            return
                        default:
                            break
                        }
                    }
                    try await writer.finish(nil)
                }
            )
        }

        // POST /api/chat — streaming NDJSON or single JSON
        router.post("/api/chat") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let ollamaBody = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid JSON")
            }
            let model = (ollamaBody["model"] as? String) ?? "default"
            let isStream = (ollamaBody["stream"] as? Bool) ?? true

            // Translate Ollama body → ChatRequest. Minimal mapping; full translation
            // lives in vmlx_engine/api/ollama_adapter.py (ollama_chat_to_openai).
            guard let chatReq = Self.ollamaToChatRequest(ollamaBody) else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing messages")
            }
            // 2026-04-18 validate parity — see AnthropicRoutes for the
            // background. Reject out-of-range temperature / negative
            // max_tokens etc with 400 before invoking the engine.
            do {
                try chatReq.validate()
            } catch let err as ChatRequestValidationError {
                return OpenAIRoutes.errorJSON(.badRequest, err.description)
            } catch {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid request: \(error)")
            }

            await engine.wakeFromStandby()
            // 4-tier settings resolution happens inside `Engine.stream`.
            let upstream = await engine.stream(request: chatReq)

            if isStream {
                var headers: HTTPFields = [:]
                headers[.contentType] = "application/x-ndjson"
                headers[.cacheControl] = "no-cache"
                return Response(
                    status: .ok,
                    headers: headers,
                    body: JSONLEncoder.ollamaChatStream(model: model, upstream: upstream)
                )
            }

            // Non-streaming — collect and return a single JSON object.
            var content = ""
            var reasoning = ""
            var usage: StreamChunk.Usage? = nil
            var finishReason: String? = nil
            do {
                for try await chunk in upstream {
                    if let c = chunk.content { content += c }
                    if let r = chunk.reasoning { reasoning += r }
                    if let u = chunk.usage { usage = u }
                    if let fr = chunk.finishReason { finishReason = fr }
                }
            } catch let err as EngineError {
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
            var message: [String: Any] = ["role": "assistant", "content": content]
            if !reasoning.isEmpty { message["thinking"] = reasoning }
            var obj: [String: Any] = [
                "model": model,
                "created_at": JSONLEncoder.iso8601Now(),
                "message": message,
                "done": true,
                "done_reason": finishReason ?? "stop",
            ]
            // iter-105 §183: route through the shared applyOllamaTimings
            // helper so /api/chat non-stream emits the full timing
            // envelope (prompt_eval_count / eval_count / total_duration /
            // prompt_eval_duration / eval_duration / load_duration) and
            // stays in lockstep with /api/generate non-stream (line
            // ~437 below) plus both streaming encoders (JSONLEncoder
            // lines 115/186). Prior inline assignment emitted only
            // the two count fields — latency UIs (Copilot, LangChain,
            // Open WebUI, OllamaJS) that key off *_duration got zeros
            // for non-stream /api/chat while the same model hit via
            // /api/generate reported real numbers. Hard rule #6.
            JSONLEncoder.applyOllamaTimings(into: &obj, usage: usage)
            return OpenAIRoutes.json(obj)
        }

        // POST /api/generate — Ollama's text-completion endpoint.
        //
        // Wraps `prompt` as a single user message and routes through
        // `engine.stream`. Streaming emits NDJSON with `response`
        // per-token, matching Ollama's CLI wire format.
        //
        // Differences from `/api/chat`:
        //   - single `prompt` string, not `messages[]`
        //   - response stream emits `{"model":"...", "response":"...",
        //     "done":false}` per token, then a `done:true` final frame
        //     with timing/eval fields.
        router.post("/api/generate") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid JSON body")
            }
            let model = (obj["model"] as? String) ?? "default"
            let prompt = (obj["prompt"] as? String) ?? ""
            if prompt.isEmpty {
                return OpenAIRoutes.errorJSON(.badRequest, "missing 'prompt'")
            }
            let isStream = (obj["stream"] as? Bool) ?? true
            let options = (obj["options"] as? [String: Any]) ?? [:]

            var messages: [ChatRequest.Message] = []
            if let system = obj["system"] as? String, !system.isEmpty {
                messages.append(.init(role: "system", content: .string(system)))
            }
            messages.append(.init(role: "user", content: .string(prompt)))

            let chatReq = ChatRequest(
                model: model,
                messages: messages,
                stream: isStream,
                maxTokens: options["num_predict"] as? Int,
                temperature: options["temperature"] as? Double,
                topP: options["top_p"] as? Double,
                topK: options["top_k"] as? Int,
                minP: options["min_p"] as? Double,
                repetitionPenalty: options["repeat_penalty"] as? Double,
                stop: options["stop"] as? [String],
                seed: options["seed"] as? Int,
                enableThinking: obj["think"] as? Bool
            )
            // 2026-04-18 validate parity — see /api/chat above.
            do {
                try chatReq.validate()
            } catch let err as ChatRequestValidationError {
                return OpenAIRoutes.errorJSON(.badRequest, err.description)
            } catch {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid request: \(error)")
            }
            await engine.wakeFromStandby()
            let upstream = await engine.stream(request: chatReq)

            if isStream {
                var headers: HTTPFields = [:]
                headers[.contentType] = "application/x-ndjson"
                headers[.cacheControl] = "no-cache"
                return Response(
                    status: .ok,
                    headers: headers,
                    body: JSONLEncoder.ollamaGenerateStream(model: model, upstream: upstream)
                )
            }

            // Non-streaming: collect into a single JSON blob.
            var content = ""
            var usage: StreamChunk.Usage? = nil
            do {
                for try await chunk in upstream {
                    if let c = chunk.content { content += c }
                    if let u = chunk.usage { usage = u }
                }
            } catch let err as EngineError {
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
            var out: [String: Any] = [
                "model": model,
                "created_at": ISO8601DateFormatter().string(from: Date()),
                "response": content,
                "done": true,
                "done_reason": "stop",
            ]
            // iter-64: share the timing-envelope helper with the NDJSON
            // streaming encoders so non-stream + stream emit identical
            // fields on their final `done:true` chunk. Inline logic
            // previously lived in §93 (iter-63).
            JSONLEncoder.applyOllamaTimings(into: &out, usage: usage)
            return OpenAIRoutes.json(out)
        }

        // POST /api/embeddings and /api/embed — thin adapters around the
        // OpenAI-compatible `/v1/embeddings` path. Ollama accepts two
        // shapes: legacy `{"prompt": "..."}` (single string) and the
        // newer `{"input": [...]}`. Both map to the OpenAI `input` field,
        // which `engine.embeddings` already understands.
        let ollamaEmbedHandler: @Sendable (Request, Context) async throws -> Response = { req, _ in
            var req = req
            let body = try await req.collectBody(upTo: 16 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid JSON body")
            }
            var openAI: [String: Any] = [:]
            if let model = obj["model"] as? String { openAI["model"] = model }
            if let prompt = obj["prompt"] {
                openAI["input"] = prompt   // legacy single-string shape
            } else if let input = obj["input"] {
                openAI["input"] = input
            } else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing 'prompt' or 'input'")
            }
            // iter-63: JIT wake — mirror §89 (OpenAI) and §91 (gateway).
            // Ollama `/api/embeddings` + `/api/embed` against a soft-
            // slept engine was returning 503 notLoaded, breaking tools
            // like LangChain/Copilot/OllamaJS that auto-reconnect.
            await engine.wakeFromStandby()
            do {
                let result = try await engine.embeddings(request: openAI)
                // Ollama shape: {"embedding": [...]} for a single input,
                // {"embeddings": [[...],[...]]} for a list. Extract from
                // the OpenAI-style {"data": [{"embedding": [...]}]} envelope.
                let data = (result["data"] as? [[String: Any]]) ?? []
                // iter-106 §132: engine.embeddings stores vectors as
                // [Float] inside the [String: Any] response dict. A prior
                // `as? [Double]` cast ALWAYS returned nil — Swift array
                // casts require element-type identity, no bridging — so
                // this Ollama adapter was silently shipping empty vectors
                // for months. Cast to [Float] (the real underlying type);
                // JSONSerialization will bridge the floats through
                // NSNumber on the wire regardless.
                let vectors: [[Float]] = data.compactMap {
                    $0["embedding"] as? [Float]
                }
                if vectors.count == 1 {
                    return OpenAIRoutes.json([
                        "embedding": vectors[0],
                        "model": result["model"] ?? "",
                    ])
                }
                return OpenAIRoutes.json([
                    "embeddings": vectors,
                    "model": result["model"] ?? "",
                ])
            } catch let err as EngineError {
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
        }
        router.post("/api/embeddings", use: ollamaEmbedHandler)
        router.post("/api/embed", use: ollamaEmbedHandler)

        // DELETE /api/delete — remove a downloaded model. Body: `{"name":"qwen3:8b"}`.
        // Resolves the entry by displayName match against ModelLibrary,
        // then calls `deleteEntry(byId:)` which rm -r's the canonical path
        // under safety-fenced known roots.
        router.delete("/api/delete") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 16 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = (obj["name"] as? String) ?? (obj["model"] as? String),
                  !name.isEmpty
            else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing 'name' or 'model'")
            }
            let library = await engine.modelLibrary
            let entries = await library.entries()
            guard let match = entries.first(where: {
                $0.displayName == name
                    || $0.displayName.lowercased() == name.lowercased()
                    || $0.canonicalPath.lastPathComponent == name
            }) else {
                return OpenAIRoutes.errorJSON(.notFound, "model not found: \(name)")
            }
            do {
                let ok = try await library.deleteEntry(byId: match.id)
                if !ok {
                    return OpenAIRoutes.errorJSON(.notFound, "delete failed (entry vanished)")
                }
                return OpenAIRoutes.json([
                    "status": "success",
                    "deleted": match.displayName,
                ])
            } catch let err as EngineError {
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
        }

        // POST /api/copy — Ollama-style alias creation. Body:
        // `{"source":"qwen3:8b","destination":"qwen-team:8b"}`.
        // vMLX has no concept of an editable display alias separate from
        // the on-disk path, so this returns 501 with a structured note
        // explaining the gap (better than silently 200-ing on a no-op).
        // When ModelLibrary grows aliases, swap the body for a real copy.
        router.post("/api/copy") { _, _ -> Response in
            return OpenAIRoutes.errorJSON(
                .notImplemented,
                "vMLX does not yet support model aliasing. Use /api/pull to download a copy under a new name.")
        }

        // POST /api/create — Ollama Modelfile build. Out-of-scope for vMLX
        // (we don't have a Modelfile equivalent). Returns 501 with an
        // explicit reason so SDK callers don't think creation succeeded.
        router.post("/api/create") { _, _ -> Response in
            return OpenAIRoutes.errorJSON(
                .notImplemented,
                "vMLX does not support Ollama Modelfile creation. Use vMLX's quantizer / converter tools and /api/pull instead.")
        }
    }

    /// Minimal Ollama → ChatRequest translation. Full logic: vmlx_engine/api/ollama_adapter.py.
    /// Handles: model, messages (string content only), options.{temperature,top_p,top_k,num_predict,stop},
    /// tools, think (enable_thinking).
    /// Convert raw model-on-disk bytes into the "7B"/"70B" strings
    /// Ollama clients expect in `details.parameter_size`. Uses the
    /// weight-bytes-to-params ratio from the detected quantization:
    /// 16 bits/weight (fp16/bf16) → 2 bytes per param, 8 bits → 1,
    /// 4 bits → 0.5, etc. Falls back to assuming 2 bytes/param when
    /// `quant_bits` is nil.
    static func humanParamSize(_ bytes: Int64, quantBits: Int? = nil) -> String {
        let bytesPerParam: Double
        if let q = quantBits {
            bytesPerParam = Double(q) / 8.0
        } else {
            bytesPerParam = 2.0
        }
        guard bytesPerParam > 0 else { return "" }
        let params = Double(bytes) / bytesPerParam
        if params >= 1_000_000_000 {
            return String(format: "%.1fB", params / 1_000_000_000)
        }
        if params >= 1_000_000 {
            return String(format: "%.1fM", params / 1_000_000)
        }
        return "\(Int(params))"
    }

    static func ollamaToChatRequest(_ body: [String: Any]) -> ChatRequest? {
        guard let rawMessages = body["messages"] as? [[String: Any]] else { return nil }
        let messages: [ChatRequest.Message] = rawMessages.compactMap { m in
            guard let role = m["role"] as? String else { return nil }
            let textContent = (m["content"] as? String) ?? ""
            // Ollama carries images as a sibling `images: [String]` field
            // of raw base64 strings. Translate each to an OpenAI-style
            // `image_url` part so `Stream.extractImages` can decode it
            // via `ContentPart.ImageURL.loadImageData()`'s bare-base64
            // fallback. Text is preserved as a `text` part when any
            // images are present.
            let rawImages = (m["images"] as? [String]) ?? []
            let contentVal: ChatRequest.ContentValue?
            if !rawImages.isEmpty {
                var parts: [ChatRequest.ContentPart] = []
                if !textContent.isEmpty {
                    parts.append(ChatRequest.ContentPart(
                        type: "text", text: textContent))
                }
                for b64 in rawImages {
                    parts.append(ChatRequest.ContentPart(
                        type: "image_url",
                        imageUrl: .init(url: b64)))
                }
                contentVal = .parts(parts)
            } else if !textContent.isEmpty {
                contentVal = .string(textContent)
            } else {
                contentVal = nil
            }
            return ChatRequest.Message(
                role: role, content: contentVal, name: nil,
                toolCalls: nil, toolCallId: nil
            )
        }

        // Ollama tools — same OpenAI-function shape as /v1/chat/completions.
        // Ollama clients (Open WebUI, LangChain's OllamaFunctions) send:
        //   tools: [{type:"function", function:{name, description, parameters}}]
        // We decode via Codable round-trip since ChatRequest.Tool.Function
        // doesn't expose a public memberwise init.
        var tools: [ChatRequest.Tool]? = nil
        if let rawTools = body["tools"] as? [[String: Any]], !rawTools.isEmpty {
            var collected: [ChatRequest.Tool] = []
            for t in rawTools {
                if let data = try? JSONSerialization.data(withJSONObject: t),
                   let tool = try? JSONDecoder().decode(
                       ChatRequest.Tool.self, from: data)
                {
                    collected.append(tool)
                }
            }
            if !collected.isEmpty { tools = collected }
        }

        let opts = (body["options"] as? [String: Any]) ?? [:]
        // P1-API-4: surface unknown Ollama options. Mirostat / num_ctx /
        // repeat_last_n etc are accepted by the wire but vMLX has no
        // direct mapping yet. Log them so the user knows they were
        // received but ignored — beats silent drop.
        let knownOptionKeys: Set<String> = [
            "num_predict", "temperature", "top_p", "top_k", "min_p",
            "repeat_penalty", "stop", "seed",
        ]
        // Iter-26: gate the unsupported-options warning on the same
        // VL debug flag pattern. Ollama clients that send typo'd
        // options (common in ops scripts) would blast stderr on every
        // request. Warning still lands via `VMLX_OLLAMA_DEBUG=1`.
        let unknownKeys = opts.keys.filter { !knownOptionKeys.contains($0) }
        if !unknownKeys.isEmpty,
           ProcessInfo.processInfo.environment["VMLX_OLLAMA_DEBUG"] == "1"
        {
            let names = unknownKeys.sorted().joined(separator: ", ")
            FileHandle.standardError.write(Data(
                "[ollama] /api/chat: ignored unsupported options: \(names)\n".utf8))
        }
        let req = ChatRequest(
            model: (body["model"] as? String) ?? "default",
            messages: messages,
            stream: body["stream"] as? Bool,
            maxTokens: opts["num_predict"] as? Int,
            temperature: opts["temperature"] as? Double,
            topP: opts["top_p"] as? Double,
            topK: opts["top_k"] as? Int,
            minP: opts["min_p"] as? Double,
            repetitionPenalty: opts["repeat_penalty"] as? Double,
            stop: opts["stop"] as? [String],
            seed: opts["seed"] as? Int,
            enableThinking: body["think"] as? Bool,
            reasoningEffort: nil,
            tools: tools,
            toolChoice: nil
        )
        return req
    }
}

// Public inits live in vMLXEngine/ChatRequest.swift.
