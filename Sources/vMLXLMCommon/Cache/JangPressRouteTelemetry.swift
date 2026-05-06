// SPDX-License-Identifier: Apache-2.0
//
// Route telemetry bridge for JangPress.
//
// The JangPress controller can only make a real hot/cold routed-expert
// decision if model forwards report which experts the router selected.
// This file keeps that dependency one-way: vMLXEngine installs the
// active controller in a TaskLocal around generation; vMLXLLM model
// code records top-k indices through this vMLXLMCommon helper without
// importing the engine.

import Cmlx
import Foundation
import MLX

public enum JangPressRouteTelemetry {
    @TaskLocal public static var controller: JangPressController?

    /// Bounded route telemetry for JangPress.
    ///
    /// Reading router indices with `asArray` inside a model forward can
    /// synchronize MLX/Metal. That is unsafe for large prefill tensors, so
    /// production telemetry only reads small decode-step top-k tensors
    /// (default <= 32 integer IDs) and skips large prefill observations.
    /// Operators can still force-disable with
    /// `VMLX_JANGPRESS_ROUTE_TELEMETRY=0`, or force-enable/raise the
    /// limit for diagnostics with `VMLX_JANGPRESS_ROUTE_MAX_INDICES`.
    public static var isEnabled: Bool {
        telemetryOverride != false
    }

    private static var telemetryOverride: Bool? {
        guard let raw = ProcessInfo.processInfo.environment[
            "VMLX_JANGPRESS_ROUTE_TELEMETRY"]?.lowercased()
        else {
            return nil
        }
        if raw == "0" || raw == "false" || raw == "off" || raw == "no" {
            return false
        }
        if raw == "1" || raw == "true" || raw == "on" || raw == "yes" {
            return true
        }
        return nil
    }

    public static var maxIndicesPerObservation: Int {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["VMLX_JANGPRESS_ROUTE_MAX_INDICES"],
           let parsed = Int(raw), parsed > 0
        {
            return parsed
        }
        return 32
    }

    public static func recordTopK(layer: Int, indices: MLXArray) {
        guard isEnabled else { return }
        guard indices.size > 0 else { return }

        let forceControllerTelemetry = telemetryOverride == true
        guard let controller else {
            observeCanonicalAdvisorIfSafe(layer: layer, indices: indices)
            return
        }
        if !forceControllerTelemetry && !controller.routePrefetchEnabled {
            observeCanonicalAdvisorIfSafe(layer: layer, indices: indices)
            return
        }
        guard indices.size <= maxIndicesPerObservation else {
            controller.recordSkippedRouteObservation()
            return
        }
        if isTracerArray(indices) {
            controller.recordSkippedRouteObservation()
            return
        }

        let k = max(indices.dim(-1), 1)
        let flat = indices.asType(.int32).flattened().asArray(Int32.self)
        guard !flat.isEmpty else { return }

        var pairs: [(layer: Int, experts: [Int])] = []
        pairs.reserveCapacity(max(flat.count / k, 1))
        var cursor = 0
        while cursor < flat.count {
            let end = min(cursor + k, flat.count)
            var experts: [Int] = []
            experts.reserveCapacity(end - cursor)
            for raw in flat[cursor..<end] {
                let e = Int(raw)
                if e >= 0 {
                    experts.append(e)
                }
            }
            if !experts.isEmpty {
                pairs.append((layer: layer, experts: experts))
            }
            cursor += k
        }
        if !pairs.isEmpty {
            controller.recordBatchRoutes(pairs)
        }

        // Router-aware canonical mmap advisor — bypass entry that
        // reuses the host-realized Int32 indices instead of re-reading
        // the MLXArray. Tracer + size guards already happened above so
        // we know `flat` is real device data within `maxIndicesPerObservation`.
        // The advisor itself is a no-op when its `configure(...)` was
        // not called (default disabled), so dense / non-mmap models do
        // not pay any cost here.
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layer, experts: flat)
    }

    private static func observeCanonicalAdvisorIfSafe(layer: Int, indices: MLXArray) {
        // The advisor has its own disabled fast path, size guard, and
        // tracer-array guard. Call it directly so non-JangPress / router
        // advice disabled decode does not pay an extra C tracer probe on
        // every routed layer. This mirrors the reference repo's direct
        // advisor call while preserving the controller readback path above.
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layer, indices: indices)
    }

    private static func isTracerArray(_ indices: MLXArray) -> Bool {
        // MLX compile traces model forwards with tracer arrays. Side-channel
        // telemetry must never call `asArray` on those values: there is no
        // realized data pointer yet, and attempting a host read has caused
        // Laguna compiled-decode SIGSEGVs in `MLXArray.asArray`.
        var isTracer = false
        if mlx_array_is_tracer(&isTracer, indices.ctx) == 0, isTracer {
            return true
        }
        return false
    }
}
