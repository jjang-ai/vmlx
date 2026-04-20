import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import NIOFoundationCompat
import vMLXEngine

/// LoRA / DoRA runtime hot-swap routes.
///
/// Matches the semantics of Python `vmlx_engine` 's `/v1/adapters/*`
/// surface, but exposed via `Engine.loadAdapter` / `unloadAdapter` /
/// `fuseAdapter` / `listAdapter` (defined in `EngineAdapters.swift`).
///
/// Supported requests:
///
///   - `GET  /v1/adapters`        — returns `{"active": <ActiveAdapterInfo|null>}`
///   - `POST /v1/adapters/load`   — body `{"path": "/abs/path/to/adapter"}`;
///                                  returns the active adapter info on success
///   - `POST /v1/adapters/unload` — unloads the active adapter; idempotent
///   - `POST /v1/adapters/fuse`   — permanently fuses the active adapter
///                                  into the base weights (destructive)
///
/// Errors surface as structured JSON with the same shape as the other
/// routes (`{"error": {"message": ..., "type": "invalid_request"}}`).
public enum AdapterRoutes {

    public static func register<Context: RequestContext>(
        on router: Router<Context>,
        engine: Engine
    ) {
        router.get("/v1/adapters") { _, _ -> Response in
            let active = await engine.listAdapter()
            return OpenAIRoutes.json(encoded(active: active))
        }

        router.post("/v1/adapters/load") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 64 * 1024)
            let bytes = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                  let path = obj["path"] as? String, !path.isEmpty
            else {
                return OpenAIRoutes.errorJSON(
                    .badRequest,
                    "POST /v1/adapters/load expects JSON body {\"path\": \"...\"}"
                )
            }
            do {
                try await engine.loadAdapter(directory: URL(fileURLWithPath: path))
                let active = await engine.listAdapter()
                return OpenAIRoutes.json(encoded(active: active))
            } catch {
                return OpenAIRoutes.errorJSON(.badRequest, "\(error)")
            }
        }

        router.post("/v1/adapters/unload") { _, _ -> Response in
            do {
                try await engine.unloadAdapter()
                return OpenAIRoutes.json(["status": "unloaded"])
            } catch {
                return OpenAIRoutes.errorJSON(.badRequest, "\(error)")
            }
        }

        router.post("/v1/adapters/fuse") { _, _ -> Response in
            do {
                try await engine.fuseAdapter()
                let active = await engine.listAdapter()
                return OpenAIRoutes.json(encoded(active: active))
            } catch {
                return OpenAIRoutes.errorJSON(.badRequest, "\(error)")
            }
        }
    }

    /// Flatten an optional `ActiveAdapterInfo` into a plain
    /// `[String: Any]` dict suitable for `OpenAIRoutes.json`.
    private static func encoded(active: Engine.ActiveAdapterInfo?) -> [String: Any] {
        guard let a = active else {
            return ["active": NSNull()]
        }
        return [
            "active": [
                // iter-117 §143: delegate to shared OpenAIRoutes.redactHomeDir
                // (was §142's private copy; promoted for /v1/models reuse).
                "path": OpenAIRoutes.redactHomeDir(a.path),
                "name": (a.path as NSString).lastPathComponent,
                "fine_tune_type": a.fineTuneType,
                "rank": a.rank,
                "scale": a.scale,
                "num_layers": a.numLayers,
                "fused": a.fused,
            ] as [String: Any]
        ]
    }
}
