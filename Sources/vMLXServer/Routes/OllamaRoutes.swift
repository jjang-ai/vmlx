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
            //
            // iter-114 §192: stamp the last human-review date so
            // future audits can spot-check whether this pin still
            // sits above every Ollama-SDK client's feature gate.
            // Automatic fail-after-N-months would be brittle; a
            // visible stamp + periodic audit is the sweet spot.
            // Previous reviews:
            //   - 2026-04-15 (iter-50): bumped from 0.6.2 to 0.12.6
            //     to satisfy newer Copilot Chat gates
            //   - last-reviewed: 2026-04-20 — still comfortably
            //     above the observed >=0.6.4 minimum for Copilot
            //     and Open WebUI. No change needed yet.
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
            let matched = entries.first(where: {
                $0.canonicalPath.standardizedFileURL == canonical
            })
            let displayName = matched?.displayName ?? path.lastPathComponent
            let iso = ISO8601DateFormatter().string(from: Date())
            // iter-123 §198: was emitting `size:0, digest:"", details:{}`
            // — Ollama CLI `ollama ps` displays these three as
            // `0 B`, blank, and `Unknown` respectively, and Copilot /
            // LangChain clients that probe `/api/ps` for a model's
            // family/quantization to pick decode flags got nothing
            // useful. Populate from the matched ModelLibrary entry
            // when present (same shape as `/api/tags` and `/api/show`
            // so clients don't have to learn per-route schemas).
            // Falls back to the old zero/empty values when the library
            // hasn't matched the loaded path (fresh model outside the
            // scan root) — preserves the current "at least the name
            // is right" behavior rather than throwing.
            var model: [String: Any] = [
                "name": displayName,
                "model": displayName,
                "expires_at": iso,
                "size_vram": 0,
            ]
            if let e = matched {
                model["size"] = e.totalSizeBytes
                model["digest"] = e.id
                model["details"] = [
                    "parent_model": "",
                    "format": e.isJANG ? "jang" : (e.isMXTQ ? "mxtq" : "mlx"),
                    "family": e.family,
                    "families": [e.family],
                    "parameter_size": Self.humanParamSize(e.totalSizeBytes, quantBits: e.quantBits),
                    "quantization_level": e.quantBits.map { "Q\($0)" } ?? "",
                ] as [String: Any]
            } else {
                model["size"] = 0
                model["digest"] = ""
                model["details"] = [:] as [String: Any]
            }
            return OpenAIRoutes.json(["models": [model]])
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
            // iter-ralph §231 (H9): validate repo name shape before handing
            // to DownloadManager. Ollama's /api/pull accepts HuggingFace
            // `org/model` ids (optionally with `:tag` suffix or `@rev`
            // ref). Without validation a caller could pass
            // `../../../etc/passwd` / `/tmp/evil` / a network-looking URL
            // and — depending on what DownloadManager's hf-hub path
            // resolver decides — read or write outside the model cache.
            // HF repo ids match `^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}`
            // with optional `:tag` or `@revision`. Reject path-like
            // strings, absolute paths, URLs, and anything with shell
            // metachars.
            let repoPattern = #"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}(:[A-Za-z0-9._-]+|@[A-Za-z0-9._-]+)?$"#
            let isValidRepoShape: Bool = {
                if repo.hasPrefix("/") || repo.hasPrefix(".") { return false }
                if repo.contains("://") || repo.contains("..") { return false }
                let bad = CharacterSet(charactersIn: " \t\n\r<>|&;*?$`\"'\\")
                if repo.rangeOfCharacter(from: bad) != nil { return false }
                return repo.range(of: repoPattern, options: .regularExpression) != nil
            }()
            if !isValidRepoShape {
                return OpenAIRoutes.errorJSON(
                    .badRequest,
                    "invalid repo name: expected `org/model[:tag]` HuggingFace id, got \(repo)")
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

            // iter-112 §190: warn-log inbound keep_alive (silent drop
            // before today — see the helper for the full rationale).
            Self.warnIfKeepAlivePresent(ollamaBody, route: "/api/chat")

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

            // iter-112 §190: warn-log inbound keep_alive (silent drop
            // before today — see warnIfKeepAlivePresent for the full
            // rationale). Shared helper with /api/chat so both routes
            // emit identical logs.
            Self.warnIfKeepAlivePresent(obj, route: "/api/generate")

            // iter-111 §189: Ollama's /api/generate accepts
            // `context: [int]` — the legacy stateless-continuation
            // contract where the client sends the previous turn's
            // final token array and gets a continuation without
            // resending the whole conversation. vMLX previously
            // silently dropped inbound context[] (not read anywhere)
            // AND silently omitted it from the response. Legacy
            // Ollama clients (pre-v0.1.14) thought their conversation
            // was continuing but were actually starting fresh every
            // turn — a silent correctness bug that hurt output
            // quality when users relied on the legacy pattern.
            //
            // FIXME(iter-111 §189): wiring the real behavior needs
            // access to the loaded model's tokenizer through the
            // Engine container (same blocker as /v1/messages/
            // count_tokens iter-108 §186). When that Engine entry
            // point lands we can:
            //   (a) accept inbound context[] and decode → strip
            //       from the prompt the model would naturally re-
            //       prefill, OR more simply
            //   (b) accept context[] as a pre-tokenized prefix and
            //       feed directly into the paged cache.
            // On the response side we'd emit the final token array
            // so clients can round-trip.
            //
            // Until then, warn on detection and DO NOT emit a fake
            // empty `context: []` in the response — absence is the
            // honest signal that the server doesn't maintain the
            // contract. Clients that depend on it can detect the
            // gap cleanly instead of seeing a zero-length array
            // they mistakenly think will continue the session.
            if let inboundContext = obj["context"] as? [Int],
               !inboundContext.isEmpty
            {
                FileHandle.standardError.write(Data(
                    "[ollama] /api/generate: ignored inbound `context: [int]` (\(inboundContext.count) tokens) — vMLX does not yet support legacy stateless continuation, prompt will be re-prefilled from scratch. See iter-111 §189 FIXME.\n".utf8))
            }

            var messages: [ChatRequest.Message] = []
            if let system = obj["system"] as? String, !system.isEmpty {
                messages.append(.init(role: "system", content: .string(system)))
            }
            messages.append(.init(role: "user", content: .string(prompt)))

            // iter-110 §188: parity warning with /api/chat. Previously
            // /api/generate had NO unknown-option detection — users
            // setting mirostat / num_ctx / repeat_last_n / tfs_z /
            // typical_p got silent no-op. Route through the shared
            // helper so both Ollama-shape routes emit identical logs.
            Self.warnUnsupportedOllamaOptions(options, route: "/api/generate")
            // §329 Ollama `think` accepts Bool OR "low"/"medium"/"high".
            let (thinkBool, thinkEffort) = Self.parseOllamaThinkField(obj["think"])
            var chatReq = ChatRequest(
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
                enableThinking: thinkBool,
                reasoningEffort: thinkEffort
            )
            // iter-110 §188: match /api/chat's post-init freq/presence
            // assignment. Ollama's options panel in Open WebUI exposes
            // both fields and the engine-side sampler wiring from
            // iter-95 §173 honors them — but /api/generate was
            // silently dropping both.
            chatReq.frequencyPenalty = options["frequency_penalty"] as? Double
            chatReq.presencePenalty = options["presence_penalty"] as? Double
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
                // iter-113 §191: Ollama spec requires prompt_eval_count
                // + total_duration + load_duration on every embed
                // response. Prior adapter shipped bare {embedding(s),
                // model} — RAG pipelines (LangChain OllamaEmbeddings,
                // ollama-js embed client, Open WebUI RAG panel) saw
                // zeros for every vMLX embedding even though the engine
                // had real token counts available in its usage dict.
                // Hard rule #6 parity with /api/chat + /api/generate
                // (both emit via applyOllamaTimings since iter-64 §118).
                //
                // total_duration + load_duration aren't tracked
                // independently on this path (no GenerateParameters
                // instrumentation for pure-embed flow), so emit 0 —
                // same approach applyOllamaTimings uses for
                // load_duration in the gen path. Clients keying on
                // the field's presence don't break; clients reading
                // the value see the correct "not tracked" signal.
                let promptEvalCount: Int = {
                    if let usage = result["usage"] as? [String: Any],
                       let tokens = usage["prompt_tokens"] as? Int
                    { return tokens }
                    return 0
                }()
                if vectors.count == 1 {
                    return OpenAIRoutes.json([
                        "embedding": vectors[0],
                        "model": result["model"] ?? "",
                        "prompt_eval_count": promptEvalCount,
                        "total_duration": 0,
                        "load_duration": 0,
                    ])
                }
                return OpenAIRoutes.json([
                    "embeddings": vectors,
                    "model": result["model"] ?? "",
                    "prompt_eval_count": promptEvalCount,
                    "total_duration": 0,
                    "load_duration": 0,
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

        // iter-109 §187: /api/push + /api/blobs were raw 404 before
        // today — Ollama's registry-upload path that vMLX has no
        // equivalent of. Label them 501 with Ollama-shape error
        // bodies + FIXME anchors so SDK clients (ollama-js uploader,
        // LangChain's export flow) can feature-detect the gap
        // instead of seeing a network-layer 404 that looks like a
        // broken proxy. Same pattern as /api/copy + /api/create
        // (iter-83 labeled-stub convention).

        // POST /api/push — Ollama push-to-registry. vMLX doesn't
        // mirror Ollama's Modelfile registry and has no account
        // tokens to push with. Label 501; clients using Ollama's
        // built-in sharing workflow will see the explicit error
        // rather than a mysterious 404.
        //
        // FIXME(iter-109 §187): if vMLX ever publishes to HF Hub
        // as a user-initiated action, wire this through
        // DownloadManager in reverse (uploadManager), using the
        // HF token flow. For now the honest state is "we don't
        // publish models from the engine".
        router.post("/api/push") { _, _ -> Response in
            return OpenAIRoutes.errorJSON(
                .notImplemented,
                "vMLX does not push to the Ollama registry — model publishing is not wired. Use vMLX's quantizer / converter + HuggingFace CLI externally if you want to share a model.")
        }

        // HEAD /api/blobs/:digest — Ollama uploader probe. Before a
        // POST upload the Ollama client HEADs this endpoint to ask
        // "is this blob already stored?" — a 200 means skip upload,
        // a 404 means proceed. Since vMLX has no blob store, the
        // safest wire-level response is 501 (not 404) so the client
        // surfaces the capability gap rather than proceeding to POST.
        //
        // FIXME(iter-109 §187): if we grow a content-addressed
        // weight store (e.g., for smelt-mode partial-expert caching),
        // this is where the HEAD probe would hook in — return 200
        // when `blobs[digest].exists`, 404 otherwise. For now the
        // honest answer is 501 because there's no store to probe.
        router.head("/api/blobs/:digest") { _, _ -> Response in
            return OpenAIRoutes.errorJSON(
                .notImplemented,
                "vMLX has no Ollama-style content-addressed blob store.")
        }

        // POST /api/blobs/:digest — Ollama blob upload. Paired with
        // the HEAD above; if the uploader sees 404 on HEAD it POSTs
        // the actual bytes. vMLX has nowhere to put them so 501 is
        // the honest response.
        router.post("/api/blobs/:digest") { _, _ -> Response in
            return OpenAIRoutes.errorJSON(
                .notImplemented,
                "vMLX has no Ollama-style content-addressed blob store — use /api/pull to fetch from HuggingFace Hub instead.")
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
        warnUnsupportedOllamaOptions(opts, route: "/api/chat")
        // §329 Ollama `think` accepts Bool OR String ("low"/"medium"/
        // "high"). Prior to this fix only the Bool form was read, so
        // Ollama clients setting `think: "medium"` (Ollama 0.12+
        // reasoning-effort API) got silent-drop — enableThinking
        // stayed nil and reasoning_effort never wired.
        let (thinkBool, thinkEffort) = Self.parseOllamaThinkField(body["think"])
        var req = ChatRequest(
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
            enableThinking: thinkBool,
            reasoningEffort: thinkEffort,
            tools: tools,
            toolChoice: nil
        )
        // iter-110 §188: Ollama options{} also carries OpenAI-style
        // frequency_penalty + presence_penalty on some clients (Open
        // WebUI options panel, Copilot). ChatRequest has these fields
        // since iter-95 §173 and the sampler honors them — previously
        // the Ollama translator wasn't forwarding them, so sliders
        // users set in Open WebUI silently did nothing. Post-init
        // assignment mirrors the /v1/completions + /v1/responses
        // wiring from iter-97 + iter-98.
        req.frequencyPenalty = opts["frequency_penalty"] as? Double
        req.presencePenalty = opts["presence_penalty"] as? Double
        return req
    }

    /// iter-110 §188: shared warn helper for unsupported Ollama
    /// options — used by both /api/chat (via `ollamaToChatRequest`)
    /// and /api/generate (inline opts-to-ChatRequest mapping).
    ///
    /// Previously the warning logic was inlined inside
    /// `ollamaToChatRequest` AND gated on `VMLX_OLLAMA_DEBUG=1` so
    /// under default operation users setting num_ctx / mirostat /
    /// repeat_last_n / tfs_z / typical_p had NO indication those
    /// options were silently dropped. /api/generate had no warning
    /// logic at all — strictly worse. Hard rule #1 of the
    /// production-readiness loop: never silent no-op. Unify both
    /// routes around this single helper, emit unconditionally to
    /// stderr so the warning is visible in the server log.
    /// iter-112 §190: warn-log inbound Ollama `keep_alive` field.
    ///
    /// Ollama's /api/chat + /api/generate accept top-level
    /// `keep_alive` to control how long the model stays loaded
    /// after a request — five wire shapes: `"5m"` duration string,
    /// `"300s"` seconds-with-unit string, bare `300` seconds, `0`
    /// unload-immediately, `-1` keep-forever. Ollama clients
    /// (Copilot, LangChain, ollama-js) rely on this to coordinate
    /// memory with other workloads sharing the same machine.
    ///
    /// vMLX silently dropped it entirely before iter-112. Users
    /// setting `keep_alive: 0` expecting the model to unload had
    /// it stay resident indefinitely under vMLX's own idle timer;
    /// users setting `-1` for a long-lived session got surprised
    /// soft-sleeps when the idle timer fired. Classic silent-drop
    /// that hard rule #1 forbids.
    ///
    /// FIXME(iter-112 §190): wire keep_alive to the per-session
    /// idle-timer via `Engine.softSleep()` / `Engine.deepSleep()`
    /// so clients can control the sleep schedule per-request:
    ///   - `keep_alive == 0` → `softSleep()` immediately after
    ///     response completes (equivalent to Ollama's "unload now")
    ///   - `keep_alive > 0` → push the idle deadline out to
    ///     `Date().addingTimeInterval(N)` before any already-
    ///     scheduled deeper sleep
    ///   - `keep_alive < 0` → disable idle sleep for this session
    ///     until an explicit stop or another keep_alive request
    /// The idle-timer runs in a separate actor task today — the
    /// wire-through needs a per-session mutator on the scheduler
    /// (doesn't exist yet) rather than each handler reaching into
    /// the actor directly. Until then, surface the inbound value
    /// so operators can see what clients were expecting.
    static func warnIfKeepAlivePresent(
        _ body: [String: Any],
        route: String
    ) {
        guard let rawKeepAlive = body["keep_alive"] else { return }
        // Stringify for the log regardless of wire shape (string /
        // int / float). Accept everything so the operator sees
        // exactly what the client sent.
        let repr = "\(rawKeepAlive)"
        FileHandle.standardError.write(Data(
            "[ollama] \(route): ignored `keep_alive: \(repr)` — vMLX idle-timer / soft-sleep rules are not yet per-request configurable. See iter-112 §190 FIXME.\n".utf8))
    }

    /// §329 — Parse the Ollama `think` field which accepts both Bool
    /// (legacy) and String ("low"/"medium"/"high", Ollama 0.12+ reasoning
    /// effort API). Returns (enableThinking, reasoningEffort).
    ///
    /// Shapes:
    ///   - `true` → (true, nil)              — thinking on, default effort
    ///   - `false` → (false, nil)            — thinking off
    ///   - `"low"/"medium"/"high"` → (true, effort) — §223 auto-maps
    ///     non-"none" effort to enable_thinking=true at the Stream layer,
    ///     but set it explicitly here for clarity.
    ///   - `"none"` / `""` → (false, "none")
    ///   - anything else / missing → (nil, nil)
    static func parseOllamaThinkField(_ raw: Any?) -> (Bool?, String?) {
        if let b = raw as? Bool {
            return (b, nil)
        }
        guard let s = raw as? String else { return (nil, nil) }
        let lower = s.lowercased()
        switch lower {
        case "none", "":
            return (false, "none")
        case "low", "medium", "high":
            return (true, lower)
        default:
            return (nil, nil)
        }
    }

    static func warnUnsupportedOllamaOptions(
        _ opts: [String: Any],
        route: String
    ) {
        let knownOptionKeys: Set<String> = [
            // Directly wired through to ChatRequest / GenerateParameters.
            "num_predict", "temperature", "top_p", "top_k", "min_p",
            "repeat_penalty", "stop", "seed",
            "frequency_penalty", "presence_penalty",
        ]
        let unknownKeys = opts.keys.filter { !knownOptionKeys.contains($0) }
        guard !unknownKeys.isEmpty else { return }
        let names = unknownKeys.sorted().joined(separator: ", ")
        FileHandle.standardError.write(Data(
            "[ollama] \(route): ignored unsupported options: \(names) (not yet wired — mirostat / num_ctx / repeat_last_n / tfs_z / typical_p etc have no vMLX-side mapping)\n".utf8))
    }
}

// Public inits live in vMLXEngine/ChatRequest.swift.
