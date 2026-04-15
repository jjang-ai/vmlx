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

        // Prometheus text-format metrics endpoint.
        //
        // Closes the "observability black box" gap the audit flagged
        // as #9 priority. Reads the live `MetricsCollector.currentSnapshot()`
        // and renders a small set of gauges + counters in the classic
        // prometheus_text_format_0.0.4 shape so any scrape config
        // (Grafana Agent, Telegraf, Datadog, node_exporter-style
        // dashboards) can ingest without custom parsing.
        //
        // Current metric surface:
        //
        //   # HELP vmlx_gpu_memory_bytes_used MLX GPU memory currently in use
        //   # TYPE vmlx_gpu_memory_bytes_used gauge
        //   vmlx_gpu_memory_bytes_used 12345678
        //
        //   # HELP vmlx_gpu_memory_bytes_peak Peak MLX GPU memory since start
        //   # TYPE vmlx_gpu_memory_bytes_peak gauge
        //   vmlx_gpu_memory_bytes_peak 23456789
        //
        //   # HELP vmlx_ram_bytes_used Resident process memory
        //   # TYPE vmlx_ram_bytes_used gauge
        //   vmlx_ram_bytes_used 98765432
        //
        //   # HELP vmlx_ram_bytes_total Host physical memory
        //   # TYPE vmlx_ram_bytes_total gauge
        //   vmlx_ram_bytes_total 137438953472
        //
        //   # HELP vmlx_cpu_percent Process CPU percentage
        //   # TYPE vmlx_cpu_percent gauge
        //   vmlx_cpu_percent 12.5
        //
        //   # HELP vmlx_tokens_per_second_decode Rolling decode throughput (5s)
        //   # TYPE vmlx_tokens_per_second_decode gauge
        //   vmlx_tokens_per_second_decode 42.7
        //
        //   # HELP vmlx_tokens_per_second_prefill Rolling prefill throughput (5s)
        //   # TYPE vmlx_tokens_per_second_prefill gauge
        //   vmlx_tokens_per_second_prefill 814.3
        //
        //   # HELP vmlx_queue_depth Pending requests in the engine queue
        //   # TYPE vmlx_queue_depth gauge
        //   vmlx_queue_depth 0
        //
        //   # HELP vmlx_active_requests In-flight streaming requests
        //   # TYPE vmlx_active_requests gauge
        //   vmlx_active_requests 1
        //
        // Text format is served with `Content-Type: text/plain; version=0.0.4`
        // which is what Prometheus's scraper negotiates via Accept header.
        // Authenticated via the same bearer-token middleware as the
        // rest of `/admin/*` — operators using scraper sidecars need
        // to set the bearer in their scrape config.
        router.get("/metrics") { _, _ -> Response in
            let s = await engine.metrics.currentSnapshot()
            var body = ""
            func line(_ name: String, _ help: String, _ type: String, _ value: Double) {
                body += "# HELP \(name) \(help)\n"
                body += "# TYPE \(name) \(type)\n"
                body += "\(name) \(value)\n"
            }
            line("vmlx_gpu_memory_bytes_used",
                 "MLX GPU memory currently in use",
                 "gauge", Double(s.gpuMemBytesUsed))
            line("vmlx_gpu_memory_bytes_peak",
                 "Peak MLX GPU memory since start",
                 "gauge", Double(s.gpuMemBytesPeak))
            line("vmlx_ram_bytes_used",
                 "Resident process memory",
                 "gauge", Double(s.ramBytesUsed))
            line("vmlx_ram_bytes_total",
                 "Host physical memory",
                 "gauge", Double(s.ramBytesTotal))
            line("vmlx_cpu_percent",
                 "Process CPU percentage",
                 "gauge", s.cpuPercent)
            line("vmlx_tokens_per_second_decode",
                 "Rolling decode throughput over the last 5 seconds",
                 "gauge", s.tokensPerSecondRolling)
            line("vmlx_tokens_per_second_prefill",
                 "Rolling prefill throughput over the last 5 seconds",
                 "gauge", s.promptTokensPerSecondRolling)
            line("vmlx_queue_depth",
                 "Pending requests in the engine queue",
                 "gauge", Double(s.queueDepth))
            line("vmlx_active_requests",
                 "In-flight streaming requests",
                 "gauge", Double(s.activeRequests))
            var headers: HTTPFields = [:]
            headers[.contentType] = "text/plain; version=0.0.4; charset=utf-8"
            return Response(
                status: .ok,
                headers: headers,
                body: .init(byteBuffer: .init(string: body)))
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
