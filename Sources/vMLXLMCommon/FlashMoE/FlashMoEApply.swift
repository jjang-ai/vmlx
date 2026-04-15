// SPDX-License-Identifier: Apache-2.0
//
// FlashMoE.apply — model traversal that swaps MoE blocks in-place.
//
// In Python this was a runtime `getattr`/`setattr` walk (see
// `vmlx_engine/models/flash_moe_integration.py:apply_flash_moe`).
// Swift is statically typed, so we rely on a protocol opt-in:
// models that support Flash MoE conform to `FlashMoEReplaceable`
// and expose their layer list + per-layer replacement hook.
//
// The traversal itself is module-agnostic — it iterates layers,
// checks which ones have MoE weights in the `ExpertIndex`, and asks
// the layer to install the streaming shim via `replaceMoEBlock(with:)`
// (Qwen/Mistral/MiniMax/Nemotron text path) or
// `replaceSwitchGLU(with:)` (Gemma 4 router+experts sibling path).

import Foundation
import MLXNN

// MARK: - Protocols

/// A model (typically an LLM) that opts into Flash MoE by exposing
/// its transformer layer list.
public protocol FlashMoEReplaceable: AnyObject {
    /// The list of transformer decoder layers, in index order.
    var flashMoELayers: [FlashMoELayer] { get }
}

/// A transformer layer that opts into Flash MoE by exposing a hook
/// to install the streaming shim. Each layer implements exactly one
/// of the two replacement calls — the traversal picks whichever the
/// layer indicates via `flashMoELayout`.
public protocol FlashMoELayer: AnyObject {
    /// Which MoE layout this layer uses, so the traversal knows which
    /// replacement call to issue. `.textPath` means the MoE is wrapped
    /// under `mlp`/`block_sparse_moe`/`mixer` and the whole block is
    /// replaced. `.gemma4` means the layer has sibling `router` +
    /// `experts` and only `experts.switch_glu` is replaced.
    var flashMoELayout: FlashMoELayout { get }

    /// Install a `FlashMoEBlock` in place of the existing MoE block
    /// (layouts `.textPathSwitchGLU`, `.textPathSwitchMLP`). The layer
    /// is responsible for calling `updateModule(key:_:)` on itself.
    ///
    /// Default implementation is a no-op — layers that use the
    /// Gemma 4 sibling layout never receive this call.
    func replaceMoEBlock(with block: FlashMoEBlock) throws

    /// Install a `FlashMoESwitchGLUShim` in place of the existing
    /// `experts.switch_glu` (layout `.gemma4`). The layer is responsible
    /// for calling `updateModule(key:_:)` on `experts`.
    ///
    /// Default implementation is a no-op — layers that use the text
    /// path never receive this call.
    func replaceSwitchGLU(with shim: FlashMoESwitchGLUShim) throws
}

/// Default no-op implementations so conformers only override the one
/// that matches their layout.
extension FlashMoELayer {
    public func replaceMoEBlock(with block: FlashMoEBlock) throws {}
    public func replaceSwitchGLU(with shim: FlashMoESwitchGLUShim) throws {}
}

/// Which layout a layer uses, so `FlashMoE.apply` picks the right
/// replacement call.
public enum FlashMoELayout: Sendable, Equatable {
    /// Text-path SwitchGLU (Qwen/Mistral/MiniMax). Replace the whole
    /// MoE block via `replaceMoEBlock(with:)`.
    case textPathSwitchGLU
    /// Text-path SwitchMLP (Nemotron, 2-projection). Replace the whole
    /// MoE block via `replaceMoEBlock(with:)`.
    case textPathSwitchMLP
    /// Gemma 4 sibling layout (router + experts). Replace
    /// `experts.switch_glu` via `replaceSwitchGLU(with:)`.
    case gemma4
    /// Layer has no MoE; the traversal skips it.
    case none
}

// MARK: - Apply

/// Flash MoE application result: how many layers were patched, broken
/// down by layout. Returned by `FlashMoE.apply` so callers can log or
/// surface the counts in engine stats.
public struct FlashMoEApplyResult: Sendable, Equatable {
    public var layersPatched: Int = 0
    public var textPathPatched: Int = 0
    public var gemma4Patched: Int = 0
    /// Layers that had MoE weights in the index but no conforming
    /// replacement hook (either `.none` layout or the layer didn't
    /// conform to `FlashMoELayer`). Logged as warnings.
    public var layersSkipped: Int = 0
}

/// Flash MoE application namespace.
public enum FlashMoE {

    /// Apply Flash MoE to a model by walking its layer list and
    /// installing streaming shims in MoE layers.
    ///
    /// - Parameters:
    ///   - model: A model conforming to `FlashMoEReplaceable`. If the
    ///     concrete model type doesn't conform, the caller should
    ///     skip Flash MoE entirely — the feature is opt-in.
    ///   - loader: The expert loader carrying the slot bank + index.
    ///   - activation: Optional inner activation. When `.default`,
    ///     text-path uses `silu(gate) * up` and Gemma 4 uses its own
    ///     GeGLU via the original `SwitchGLU.activation` pulled out
    ///     at swap time.
    /// - Returns: Counts of patched layers, broken down by layout.
    public static func apply(
        to model: FlashMoEReplaceable,
        loader: FlashMoEExpertLoader,
        activation: FlashMoEActivation = .default
    ) throws -> FlashMoEApplyResult {
        var result = FlashMoEApplyResult()
        let layers = model.flashMoELayers
        let index = loader.index

        for (layerIdx, layer) in layers.enumerated() {
            // Skip layers that have no MoE weights in the index.
            guard index.layers[layerIdx] != nil else { continue }

            switch layer.flashMoELayout {
            case .textPathSwitchGLU, .textPathSwitchMLP:
                let isGLU = (layer.flashMoELayout == .textPathSwitchGLU)
                let block = FlashMoEBlock(
                    loader: loader,
                    layerIdx: layerIdx,
                    isSwitchGLU: isGLU,
                    activation: activation
                )
                try layer.replaceMoEBlock(with: block)
                result.layersPatched += 1
                result.textPathPatched += 1

            case .gemma4:
                let shim = FlashMoESwitchGLUShim(
                    loader: loader,
                    layerIdx: layerIdx,
                    activation: activation
                )
                try layer.replaceSwitchGLU(with: shim)
                result.layersPatched += 1
                result.gemma4Patched += 1

            case .none:
                result.layersSkipped += 1
            }
        }
        return result
    }
}

// `FlashMoEExpertLoader.index` is a public stored property on the class
// itself, so no extension is needed here.
