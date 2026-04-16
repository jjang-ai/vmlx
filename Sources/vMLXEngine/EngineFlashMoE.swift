// Copyright © 2025 JANG. All rights reserved.
//
// Flash MoE engine integration (Phase 2b).
//
// Builds the ExpertIndex + SlotBankCache + FlashMoEExpertLoader, then
// calls `FlashMoE.apply(to:loader:)` on the loaded model when it
// conforms to `FlashMoEReplaceable` AND the user opted into Flash MoE
// via `GlobalSettings.flashMoe`. Non-conforming models are silently
// skipped so this is safe to call unconditionally at load time.
//
// Phase 2a shipped the shim types (`FlashMoEBlock`,
// `FlashMoESwitchGLUShim`, `FlashMoEExpertLoader`) plus the traversal
// (`FlashMoE.apply`) and the per-layer protocol (`FlashMoELayer`).
// Phase 2b is the engine wire-up + per-model conformance. The first
// reference conformance is `Qwen3MoE` — see its decoder layer for
// the integration pattern. Other MoE models (Mistral, MiniMax,
// Nemotron, Gemma 4, GLM4MoE, OlmoE, LFM2MoE, BailingMoe, PhiMoE)
// will fall through the `FlashMoEReplaceable` check as no-ops until
// they also add conformance — the settings toggle is safe to enable
// globally even before every model is wired.

import Foundation
import vMLXLMCommon

extension Engine {

    /// Apply Flash MoE expert streaming to the loaded chat model, if
    /// the feature is enabled in settings AND the model opts in via
    /// `FlashMoEReplaceable`. Called from `Engine.load` after the
    /// cache coordinator is ready and before warmup, so the first
    /// forward pass sees the streaming shims.
    ///
    /// Failures are logged as warnings rather than thrown — Flash MoE
    /// is an optimization, not a correctness-critical feature, so
    /// missing expert files or unexpected model layouts shouldn't
    /// block model load.
    internal func applyFlashMoEIfEnabled(
        container: vMLXLMCommon.ModelContainer, opts: LoadOptions
    ) async {
        let g = await settings.global()
        guard g.flashMoe else { return }

        // Read the underlying model out of the container's actor.
        let model: any LanguageModel = await container.perform { ctx in
            ctx.model
        }

        guard let replaceable = model as? FlashMoEReplaceable else {
            await self.logs.append(
                .info, category: "flashMoE",
                "skipping: model type does not conform to FlashMoEReplaceable"
            )
            return
        }

        // Build the expert index by scanning the model directory's
        // safetensors headers. `ExpertIndex.build` returns an empty
        // index if the model has no `layers.N.mlp.experts.*` weights
        // (i.e., a non-MoE model accidentally flagged as `flashMoe`).
        let index: ExpertIndex
        do {
            index = try ExpertIndex.build(modelPath: opts.modelPath)
        } catch {
            await self.logs.append(
                .warn, category: "flashMoE",
                "ExpertIndex.build failed: \(error) — disabling Flash MoE for this load"
            )
            return
        }
        if index.layers.isEmpty {
            await self.logs.append(
                .info, category: "flashMoE",
                "no MoE layers found in \(opts.modelPath.lastPathComponent) — skipping"
            )
            return
        }

        // Build the slot-bank cache. When the user leaves the default
        // (64), auto-size from num_hidden_layers × num_experts_per_tok
        // × 1.5 so deep MoE models (MiniMax M2.5 with 60+ layers and
        // 8 experts-per-tok, Gemma 4 MoE with 128 experts) don't
        // thrash the slot bank. Mirrors the Python guidance in
        // `project_flash_moe_slot_bank_sizing.md`. Clamped to [64, 2048]
        // so auto-sizing never shrinks below the legacy default.
        let userSlots = max(8, g.flashMoeSlotBank)
        let autoSlots = autoSizeSlotBank(modelPath: opts.modelPath)
        let effectiveSlots: Int
        if g.flashMoeSlotBank <= 64, let auto = autoSlots, auto > userSlots {
            effectiveSlots = min(2048, auto)
            await self.logs.append(
                .info, category: "flashMoE",
                "slot bank auto-sized: user=\(userSlots) → auto=\(effectiveSlots) "
                + "(from layers × experts_per_tok × 1.5)"
            )
        } else {
            effectiveSlots = userSlots
        }
        let cache = SlotBankCache(maxSlots: effectiveSlots)
        let loader = FlashMoEExpertLoader(
            index: index, cache: cache, ioSplit: max(1, g.flashMoeIoSplit))

        do {
            let result = try FlashMoE.apply(to: replaceable, loader: loader)
            await self.logs.append(
                .info, category: "flashMoE",
                "applied: \(result.layersPatched) layers patched " +
                "(\(result.textPathPatched) text-path, " +
                "\(result.gemma4Patched) Gemma 4), " +
                "\(result.layersSkipped) skipped"
            )
        } catch {
            await self.logs.append(
                .warn, category: "flashMoE",
                "FlashMoE.apply threw: \(error) — falling back to native MoE forward"
            )
        }
    }

    /// Parse config.json at `modelPath` and compute the recommended slot
    /// bank size as `num_hidden_layers × num_experts_per_tok × 1.5`.
    /// Returns nil if either field is missing. Both fields are also
    /// checked under `text_config` for VLM wrappers (Gemma 4, Qwen3.5-VL).
    private func autoSizeSlotBank(modelPath: URL) -> Int? {
        let cfgURL = modelPath.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        func lookup(_ dict: [String: Any]) -> (Int, Int)? {
            let layers = dict["num_hidden_layers"] as? Int
            // Audit 2026-04-16 Gemma 4 perf fix: add `top_k_experts`
            // (Gemma 4's config key) and `num_selected_experts` (some
            // DeepSeek derivatives). Before this, Gemma 4 26B-A4B
            // returned nil → slot bank stuck at UI default 256 → ~40
            // tok/s because the 240-activation working-set thrashed
            // the LRU bank on every decode step.
            let k = dict["num_experts_per_tok"] as? Int
                ?? dict["moe_top_k"] as? Int
                ?? dict["top_k_experts"] as? Int
                ?? dict["num_selected_experts"] as? Int
                ?? dict["top_k"] as? Int
            if let l = layers, let kk = k { return (l, kk) }
            return nil
        }
        var pair = lookup(obj)
        if pair == nil, let text = obj["text_config"] as? [String: Any] {
            pair = lookup(text)
        }
        guard let (layers, k) = pair else { return nil }
        let recommended = Int(ceil(Double(layers * k) * 1.5))
        return max(64, recommended)
    }
}
