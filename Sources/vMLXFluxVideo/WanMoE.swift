import Foundation
@preconcurrency import MLX
import vMLXFluxKit

// MARK: - Wan 2.2 high-noise / low-noise expert switch
//
// Wan 2.2 ships TWO transformer checkpoints — a "high noise" expert and
// a "low noise" expert. The diffusion sampler picks one OR the other for
// each step based on the current sigma:
//
//   sigma > sigmaThreshold  →  high-noise expert
//   sigma <= sigmaThreshold →  low-noise expert
//
// The threshold (default 0.875 in `WanModelConfig`) is part of the
// checkpoint config; it may also come in as a fraction-of-steps boundary
// (e.g. 0.875 means "first 87.5% of the sigma schedule uses high-noise").
//
// This module is a pure routing helper. The two checkpoints themselves
// load through the same `WanDiTModel` skeleton with separate `LoadedWeights`
// trees. `WANModel.swift` instantiates `WanMoE` once at load time, then
// asks it which transformer to call per step.

public enum WanExpert: Sendable, Equatable {
    /// Single-model variants (Wan 2.1 1.3B / 14B, Wan 2.2 TI2V-5B). The
    /// `dual_model` flag in WanModelConfig is false; the same transformer
    /// runs every step.
    case single

    /// Dual-model Wan 2.2 (T2V-14B, I2V-14B).
    case dual(boundary: Float)
}

public struct WanMoE: Sendable {
    public let mode: WanExpert

    public init(mode: WanExpert) {
        self.mode = mode
    }

    /// Convenience: build from `dualModel` + `boundary` fields straight
    /// out of the checkpoint config.
    public static func fromConfig(dualModel: Bool, boundary: Float) -> WanMoE {
        if dualModel {
            return WanMoE(mode: .dual(boundary: boundary))
        }
        return WanMoE(mode: .single)
    }

    /// True when the high-noise expert should run for this step.
    /// Reference: `wan22.py` schedules — the boundary is expressed as a
    /// fraction of the sigma schedule (0.875 → switch when 87.5% of
    /// timesteps are done). Equivalent to comparing the *normalized* step
    /// progress against the boundary.
    public func useHighNoise(stepFraction: Float) -> Bool {
        switch mode {
        case .single:
            return false
        case .dual(let boundary):
            // Above the boundary in noise = high-noise expert.
            // stepFraction is `step / total_steps`; the early steps have
            // the most noise in flow-match, so high-noise = early = small
            // stepFraction relative to (1 - boundary).
            return stepFraction < (1.0 - boundary)
        }
    }

    /// String label for the active expert at this step. Useful for
    /// progress events and `isPlaceholder` reporting.
    public func expertLabel(stepFraction: Float) -> String {
        switch mode {
        case .single:
            return "single"
        case .dual:
            return useHighNoise(stepFraction: stepFraction) ? "high-noise" : "low-noise"
        }
    }
}
