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
        router.get("/health") { _, _ -> Response in
            OpenAIRoutes.json([
                "status": "ok",
                "engine": "vmlx-swift",
            ])
        }

        router.post("/admin/soft-sleep") { _, _ -> Response in
            do {
                try await engine.softSleep()
                return OpenAIRoutes.json(["status": "sleeping"])
            } catch {
                return OpenAIRoutes.errorJSON(.notImplemented, "\(error)")
            }
        }

        router.post("/admin/deep-sleep") { _, _ -> Response in
            do {
                try await engine.deepSleep()
                return OpenAIRoutes.json(["status": "deep_sleeping"])
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
                override = Engine.LoadOptions(modelPath: url)
            }
            do {
                try await engine.wake(override: override)
                return OpenAIRoutes.json(["status": "awake"])
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
            let path = await engine.dflashDrafterPath()?.path
            let settings = await engine.settings.global()
            return OpenAIRoutes.json([
                "enabled": settings.dflash,
                "ready": ready,
                "drafter_path": path as Any,
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
            do {
                try await engine.loadDFlashDrafter(from: URL(fileURLWithPath: path))
                let ready = await engine.dflashIsReady()
                return OpenAIRoutes.json([
                    "status": "loaded",
                    "ready": ready,
                    "drafter_path": path,
                ])
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
                return OpenAIRoutes.json(stats)
            } catch {
                return OpenAIRoutes.errorJSON(.notImplemented, "\(error)")
            }
        }

        router.get("/v1/cache/stats") { _, _ -> Response in
            do {
                let stats = try await engine.cacheStats()
                return OpenAIRoutes.json(stats)
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
                return OpenAIRoutes.json(entries)
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
}
