import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import vMLXEngine

/// Admin + health routes.
///
/// Python source: `vmlx_engine/server.py`
///   - GET  /health             — line 1704
///   - POST /admin/soft-sleep   — line 1880
///   - POST /admin/deep-sleep   — line 1927
///   - POST /admin/wake         — line 2020
///   - GET  /v1/cache/stats     — line 2141
public enum AdminRoutes {

    public static func register<Context: RequestContext>(
        on router: Router<Context>,
        engine: Engine
    ) {
        // `/health` — liveness + real engine state. Monitors, uptime
        // probes, and Ollama-compatible clients need to distinguish
        // running vs soft-sleeping vs deep-sleeping vs error.
        // Pre-iter-38 this was a static `{status:ok, engine:vmlx-swift}`
        // regardless of whether the engine had loaded a model, had
        // crashed, or was asleep — so probes couldn't tell a busy server
        // from a dead one. `status:ok` kept for backward compat with
        // anything grepping for that literal.
        router.get("/health") { _, _ -> Response in
            let state = engine.state
            let (stateStr, detail): (String, String?) = {
                switch state {
                case .stopped:         return ("stopped", nil)
                case .loading(let p):  return ("loading", p.phase.rawValue)
                case .running:         return ("running", nil)
                case .standby(.soft):  return ("soft_sleep", nil)
                case .standby(.deep):  return ("deep_sleep", nil)
                case .error(let msg):  return ("error", msg)
                }
            }()
            let loadedPath = await engine.loadedModelPath
            var model: String? = nil
            if let lp = loadedPath {
                _ = await engine.modelLibrary.scan(force: false)
                let entries = await engine.modelLibrary.entries()
                let canonical = lp.resolvingSymlinksInPath().standardizedFileURL
                model = entries.first(where: {
                    $0.canonicalPath.standardizedFileURL == canonical
                })?.displayName ?? lp.lastPathComponent
            }
            // iter-85 §163: expose real scheduler pattern so callers
            // don't assume they're hitting a continuous-batching
            // engine. Swift vMLX serializes MLX Metal work per-engine
            // because MTLCommandBuffer is not concurrency-safe — see
            // GenerationLock.swift. Surfaces queue depth so clients
            // that send bursts can tell they're queued behind an
            // in-flight request rather than actually interleaved.
            let lock = await engine.generationLock
            let inflight = await lock.isHeld ? 1 : 0
            let waiting = await lock.waitingCount
            var body: [String: Any] = [
                "status": "ok",
                "engine": "vmlx-swift",
                "state": stateStr,
                "scheduling": "serial-fifo",
                "inflight": inflight,
                "waiting": waiting,
            ]
            if let model { body["model"] = model }
            if let detail { body["detail"] = detail }
            return OpenAIRoutes.json(body)
        }

        router.post("/admin/soft-sleep") { _, _ -> Response in
            do {
                try await engine.softSleep()
                return OpenAIRoutes.json(["status": "sleeping"])
            } catch let err as EngineError {
                // iter-ralph §230 (H3): route EngineError through the
                // shared mapper so .promptTooLong → 413, .modelNotFound
                // → 404, .notLoaded → 503 get the right status codes
                // instead of collapsing to 501.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.notImplemented, "\(error)")
            }
        }

        router.post("/admin/deep-sleep") { _, _ -> Response in
            do {
                try await engine.deepSleep()
                return OpenAIRoutes.json(["status": "deep_sleeping"])
            } catch let err as EngineError {
                // iter-ralph §230 (H3): route EngineError through the
                // shared mapper so .promptTooLong → 413, .modelNotFound
                // → 404, .notLoaded → 503 get the right status codes
                // instead of collapsing to 501.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.notImplemented, "\(error)")
            }
        }

        // POST /admin/wake — optional JSON body `{model: "<path>"}` lets the
        // caller swap to a different model on wake (otherwise the retained
        // `lastLoadOptions` from the previous load is replayed). Empty body
        // = replay prior load.
        router.post("/admin/wake") { req, _ -> Response in
            var req = req
            let body = try? await req.collectBody(upTo: 64 * 1024)
            var override: Engine.LoadOptions? = nil
            if let body,
               let obj = try? JSONSerialization.jsonObject(with: Data(buffer: body))
                   as? [String: Any],
               let modelPath = obj["model"] as? String,
               !modelPath.isEmpty
            {
                let url = URL(fileURLWithPath: modelPath)
                // iter-152 §220: fence against ModelLibrary knownRoots so an
                // admin-token holder cannot hand in an arbitrary path
                // (/etc/..., someone else's ~, a Cloudflare tunnel secret)
                // and have the engine read it as a "model". Same shape as
                // /admin/models/{id} delete fence.
                guard await engine.modelLibrary.isPathUnderKnownRoots(url) else {
                    return OpenAIRoutes.errorJSON(
                        .badRequest,
                        "model path must live under a configured model root")
                }
                override = Engine.LoadOptions(modelPath: url)
            }
            do {
                try await engine.wake(override: override)
                return OpenAIRoutes.json(["status": "awake"])
            } catch let err as EngineError {
                // iter-ralph §230 (H3): route EngineError through the
                // shared mapper so .promptTooLong → 413, .modelNotFound
                // → 404, .notLoaded → 503 get the right status codes
                // instead of collapsing to 501.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.notImplemented, "\(error)")
            }
        }

        // JANG-DFlash admin surface. `/admin/dflash` returns a JSON
        // status blob; `/admin/dflash/load` takes `{"path": "..."}`
        // and loads the drafter checkpoint; `/admin/dflash/unload`
        // drops the current drafter. Mirrors Python's planned
        // speculative-decode admin routes (v1.3.x).
        router.get("/admin/dflash") { _, _ -> Response in
            let ready = await engine.dflashIsReady()
            // iter-128 §203: drafter_path was a home-dir path leak —
            // stored drafter sits under ~/.mlxstudio/models/.../drafter
            // by convention, so the raw path discloses the user's
            // library layout to any admin-token holder. Same family
            // as §141/§142/§143/§202. The other AdapterRoutes and
            // OpenAIRoutes surfaces already redact; this was the
            // remaining un-redacted spec-decoder path field. Echo
            // sites (POST /admin/dflash/load) legitimately echo what
            // the caller just sent so those keep the raw path.
            let rawPath = await engine.dflashDrafterPath()?.path
            let redacted = rawPath.map(OpenAIRoutes.redactHomeDir)
            let settings = await engine.settings.global()
            return OpenAIRoutes.json([
                "enabled": settings.dflash,
                "ready": ready,
                "drafter_path": redacted as Any,
                "block_size": settings.dflashBlockSize,
                "top_k": settings.dflashTopK,
                "num_paths": settings.dflashNumPaths,
                "tap_layers": settings.dflashTapLayers,
                "target_hidden_dim": settings.dflashTargetHiddenDim,
            ])
        }

        router.post("/admin/dflash/load") { req, _ -> Response in
            var req = req
            guard let buf = try? await req.collectBody(upTo: 64 * 1024),
                  let obj = try? JSONSerialization.jsonObject(with: Data(buffer: buf))
                      as? [String: Any],
                  let path = obj["path"] as? String, !path.isEmpty
            else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing `path` field")
            }
            // iter-152 §220: same knownRoots fence as /admin/wake — without
            // this an admin-token holder could point the DFlash drafter
            // loader at an arbitrary file, and the mlx safetensors reader
            // would happily start memory-mapping it.
            let drafterURL = URL(fileURLWithPath: path)
            guard await engine.modelLibrary.isPathUnderKnownRoots(drafterURL) else {
                return OpenAIRoutes.errorJSON(
                    .badRequest,
                    "drafter path must live under a configured model root")
            }
            do {
                try await engine.loadDFlashDrafter(from: drafterURL)
                let ready = await engine.dflashIsReady()
                return OpenAIRoutes.json([
                    "status": "loaded",
                    "ready": ready,
                    "drafter_path": path,
                ])
            } catch let err as EngineError {
                // iter-ralph §230 (H3): shared EngineError mapper.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
        }

        router.post("/admin/dflash/unload") { _, _ -> Response in
            await engine.unloadDFlashDrafter()
            return OpenAIRoutes.json(["status": "unloaded"])
        }

        router.get("/admin/cache/stats") { _, _ -> Response in
            // Alias of /v1/cache/stats
            do {
                let stats = try await engine.cacheStats()
                return OpenAIRoutes.json(Self.redactStats(stats))
            } catch let err as EngineError {
                // iter-ralph §230 (H3): route EngineError through the
                // shared mapper so .promptTooLong → 413, .modelNotFound
                // → 404, .notLoaded → 503 get the right status codes
                // instead of collapsing to 501.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.notImplemented, "\(error)")
            }
        }

        // POST /admin/cache/clear — drop every tier's cache entries
        // without unloading the model. Symmetric with the CachePanel
        // "Clear caches" button + the corresponding `/v1/cache/clear`
        // shape for external monitors. Pre-iter-41 only the Swift UI
        // could invoke this via `Engine.clearCaches()` — HTTP callers
        // had no way to flush and would see stale stats until the
        // next model reload.
        router.post("/admin/cache/clear") { _, _ -> Response in
            await engine.clearCaches()
            return OpenAIRoutes.json([
                "object": "cache.clear",
                "status": "cleared",
            ])
        }
        // Client-facing alias under /v1 so SDKs following the OpenAI
        // shape hit a path that looks like OpenAI's conventional
        // namespace. **Note (iter-75)**: this IS still gated by
        // `AdminAuthMiddleware` (`/v1/cache/*` is in the gate list) —
        // the original comment claiming this path skipped auth was
        // stale. Flushing every cache tier is destructive to running
        // inference sessions (loses prefix hits, kills disk L2) so
        // it belongs behind the admin token.
        router.post("/v1/cache/clear") { _, _ -> Response in
            await engine.clearCaches()
            return OpenAIRoutes.json([
                "object": "cache.clear",
                "status": "cleared",
            ])
        }

        router.get("/v1/cache/stats") { _, _ -> Response in
            do {
                let stats = try await engine.cacheStats()
                return OpenAIRoutes.json(Self.redactStats(stats))
            } catch let err as EngineError {
                // iter-ralph §230 (H3): route EngineError through the
                // shared mapper so .promptTooLong → 413, .modelNotFound
                // → 404, .notLoaded → 503 get the right status codes
                // instead of collapsing to 501.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.notImplemented, "\(error)")
            }
        }

        // GET /v1/cache/entries — list cached prompts (paged + disk + memory).
        // Returns the same shape as `cacheStats()` plus per-tier entry
        // counts. Python's `/v1/cache/entries` walks the prefix-cache trie
        // and the L2 disk index; Swift exposes the same data via
        // `cacheCoordinator.entriesSummary()`.
        router.get("/v1/cache/entries") { _, _ -> Response in
            do {
                let entries = try await engine.cacheEntries()
                return OpenAIRoutes.json(Self.redactStats(entries))
            } catch let err as EngineError {
                // iter-ralph §230 (H3): route EngineError through the
                // shared mapper so .promptTooLong → 413, .modelNotFound
                // → 404, .notLoaded → 503 get the right status codes
                // instead of collapsing to 501.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.notImplemented, "\(error)")
            }
        }

        // POST /v1/cache/warm — preload prompts into the prefix cache.
        // Body: `{"model": "<path>", "prompts": ["...", "..."]}`. Runs a
        // 1-token generation per prompt so each one populates the prefix
        // cache naturally (which is exactly how Python warms the cache —
        // it just sends a 1-token request per warmup string). Returns the
        // count of prompts processed and the post-warm cache stats.
        router.post("/v1/cache/warm") { req, _ -> Response in
            var req = req
            guard let buf = try? await req.collectBody(upTo: 1 * 1024 * 1024),
                  let obj = try? JSONSerialization.jsonObject(with: Data(buffer: buf))
                      as? [String: Any]
            else {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid JSON body")
            }
            guard let prompts = obj["prompts"] as? [String], !prompts.isEmpty else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing or empty `prompts`")
            }
            // iter-152 §220: cap prompt count + per-prompt length so an
            // admin-token holder can't schedule a 10k-prompt warmup run
            // that ties up the scheduler and blows memory. 64 prompts ×
            // 64KB each lines up with the 1MB body limit above and is
            // well above any realistic warm-set.
            let maxPromptCount = 64
            let maxPromptBytes = 64 * 1024
            guard prompts.count <= maxPromptCount else {
                return OpenAIRoutes.errorJSON(
                    .badRequest,
                    "too many prompts (max \(maxPromptCount))")
            }
            if let oversize = prompts.first(where: {
                $0.utf8.count > maxPromptBytes
            }) {
                return OpenAIRoutes.errorJSON(
                    .badRequest,
                    "prompt too long (max \(maxPromptBytes) bytes, got \(oversize.utf8.count))")
            }
            let model = (obj["model"] as? String) ?? ""
            do {
                let warmed = try await engine.cacheWarm(prompts: prompts, model: model)
                let stats = try await engine.cacheStats()
                var resp: [String: Any] = [
                    "object": "cache.warm",
                    "warmed": warmed,
                    "stats": stats,
                ]
                resp["model"] = model
                return OpenAIRoutes.json(resp)
            } catch let err as EngineError {
                // iter-ralph §230 (H3): shared EngineError mapper.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
        }

        // DELETE /v1/cache — flush all cache tiers (paged, disk, memory,
        // SSM state). Mirrors Python's `DELETE /v1/cache` admin endpoint.
        router.delete("/v1/cache") { _, _ -> Response in
            await engine.clearCaches()
            do {
                let stats = try await engine.cacheStats()
                return OpenAIRoutes.json([
                    "object": "cache.flush",
                    "cleared": true,
                    "stats": stats,
                ])
            } catch {
                return OpenAIRoutes.json([
                    "object": "cache.flush",
                    "cleared": true,
                ])
            }
        }

        // POST /admin/benchmark — run a BenchSuite against the loaded chat
        // model and return the final BenchReport as JSON. Body shape:
        //     { "suite": "decode256" | "prefill1024" | "cacheTurn5" }
        // Default suite is `decode256`. Runs synchronously (drains
        // Engine.benchmark's AsyncThrowingStream to the `.done(report:)`
        // or `.failed(msg)` terminator), then serializes headline +
        // extended fields. iter-68 gap (Engine.benchmark existed since
        // iter-62 but was only callable via the in-app BenchmarkPanel).
        router.post("/admin/benchmark") { req, _ -> Response in
            var req = req
            let body = try? await req.collectBody(upTo: 4 * 1024)
            var suiteName = "decode256"
            if let body,
               let obj = try? JSONSerialization.jsonObject(with: Data(buffer: body))
                   as? [String: Any],
               let s = obj["suite"] as? String, !s.isEmpty
            {
                suiteName = s
            }
            guard let suite = Engine.BenchSuite(rawValue: suiteName) else {
                let valid = Engine.BenchSuite.allCases.map { $0.rawValue }
                return OpenAIRoutes.errorJSON(
                    .badRequest,
                    "unknown suite '\(suiteName)'. Valid: \(valid.joined(separator: ", "))"
                )
            }
            do {
                for try await event in await engine.benchmark(suite: suite) {
                    switch event {
                    case .progress:
                        continue
                    case .failed(let msg):
                        return OpenAIRoutes.errorJSON(.internalServerError, msg)
                    case .done(let r):
                        var payload: [String: Any] = [
                            "object": "benchmark.run",
                            "suite": r.suite.rawValue,
                            "model_id": r.modelId,
                            "tokens_per_sec": r.tokensPerSec,
                            "ttft_ms": r.ttftMs,
                            "total_ms": r.totalMs,
                            "cache_hit_rate": r.cacheHitRate,
                            "notes": r.notes,
                        ]
                        if let v = r.tpotMs { payload["tpot_ms"] = v }
                        if let v = r.generationTps { payload["generation_tps"] = v }
                        if let v = r.processingTps { payload["processing_tps"] = v }
                        if let v = r.peakMemoryGB { payload["peak_memory_gb"] = v }
                        if let v = r.promptTokens { payload["prompt_tokens"] = v }
                        if let v = r.completionTokens { payload["completion_tokens"] = v }
                        return OpenAIRoutes.json(payload)
                    }
                }
                return OpenAIRoutes.errorJSON(.internalServerError,
                    "benchmark stream ended without a terminal event")
            } catch let err as EngineError {
                // iter-ralph §230 (H3): shared EngineError mapper.
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
        }

        // Fixes vmlx #57: user-facing model delete.
        // DELETE /admin/models/{id} removes the model's on-disk files and
        // drops it from the ModelLibrary cache. Returns 404 if the id is
        // unknown, 400 if the path fails the safety fence.
        router.delete("/admin/models/:id") { req, _ -> Response in
            guard let id = req.uri.path.split(separator: "/").last.map(String.init),
                  !id.isEmpty
            else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing model id")
            }
            do {
                let ok = try await engine.modelLibrary.deleteEntry(byId: id)
                if !ok {
                    return OpenAIRoutes.errorJSON(.notFound, "no such model: \(id)")
                }
                return OpenAIRoutes.json([
                    "status": "deleted",
                    "id": id,
                ])
            } catch {
                return OpenAIRoutes.errorJSON(.badRequest, "\(error)")
            }
        }
    }

    /// iter-127 §202: sanitize cache-stats payload before wire emission.
    ///
    /// `Engine.cacheStats()` returns `disk.directory` as a raw absolute
    /// filesystem path (`disk.cacheDir.path` at Engine.swift:2443). That
    /// leaks the user's home-dir layout — same class of disclosure that
    /// iter-117 §143 closed for `/v1/models`, iter-115 §141 closed for
    /// `/api/show`, iter-116 §142 closed for `/v1/adapters`. Clients
    /// (Copilot, dashboards, LAN peers once `0.0.0.0` is bound) only
    /// need to know that a disk tier exists, not where on disk it
    /// lives. The in-app `CachePanel` reads stats directly from the
    /// Engine actor so it still sees the real path — only the HTTP
    /// surface is redacted, same layer-boundary pattern as the rest
    /// of the §143-family fixes.
    ///
    /// Redaction walks `[String: Any]` recursively because `/v1/cache/
    /// entries` wraps `stats` in an outer dict (`{stats: {...}, entries:
    /// [...]}`) — a shallow dig would miss the nested `disk.directory`.
    static func redactStats(_ obj: [String: Any]) -> [String: Any] {
        var out: [String: Any] = [:]
        for (k, v) in obj {
            if k == "directory", let s = v as? String {
                out[k] = OpenAIRoutes.redactHomeDir(s)
            } else if let nested = v as? [String: Any] {
                out[k] = redactStats(nested)
            } else {
                out[k] = v
            }
        }
        return out
    }
}
