// Copyright © 2025 JANG. All rights reserved.
//
// Synchronous "prompt-only" SSM state re-derivation for hybrid
// (Mamba + attention) models on thinking-template turns.
//
// Note: `MLX.eval(...)` below is the MLX Swift API for materializing
// lazy MLXArray computation graphs — it is NOT JavaScript eval and
// has nothing to do with arbitrary code execution.
//
// THE PROBLEM
// ===========
// Hybrid models (Nemotron-H, Qwen3.5-A3B, Jamba, FalconH1, Qwen3-Next,
// GraniteMoeHybrid, MiMoV2Flash, BaichuanM1, LFM2 / LFM2MoE) interleave
// Mamba SSM layers with attention layers. The SSM state is cumulative
// and path-dependent: after a forward pass over `prompt + generated`
// the SSM state reflects that full position, not just `prompt`.
//
// When the chat template adds a generation-prompt suffix
// (`<|im_start|>assistant\n` / `<|turn>model\n` / etc.), the cache
// coordinator strips those trailing tokens from the hash key
// (`CacheCoordinator.fetch(genPromptLen:)`) so multi-turn chat can
// reuse prior prefill state. But the stored SSM state is the live
// post-generation state — its position is `prompt + generated`, not
// the stripped prompt. Restoring it on the next turn would feed the
// model an SSM state ahead of the matched token position → garbled
// output.
//
// Python v1.3.36 mitigates this by SKIPPING SSM storage when
// `gen_prompt_len > 0` (see `CacheCoordinator.shouldSkipSSMStorage`).
// Correct, but wastes cache: every thinking-model turn re-computes
// the whole hybrid prefix from scratch.
//
// THIS FIX
// ========
// `reDeriveSSMStates(model:, tokens:, prefillStepSize:)` runs a FRESH
// forward pass on just the prompt-only tokens, through a fresh cache,
// and extracts the resulting SSM state. That state's position matches
// the stripped prompt hash key exactly — no contamination, no offset
// mismatch, and the next turn can cache-hit cleanly.
//
// Called synchronously from `cacheStoreAction` at turn-end, right
// after the normal `storeAfterGeneration`. By then the stream has
// already yielded the last visible token, so the user doesn't see
// the re-derive latency as "thinking" time — they see it as a
// brief pause before the next input box goes live.
//
// Cost: one extra chunked-prefill pass over the prompt (no decode
// loop, no sampling). On a 2K prompt at ~1000 prefill tok/s that's
// ~2 seconds. On long contexts it dominates so the watcher MUST be
// skippable via `GlobalSettings.enableSSMReDerive`.

import Foundation
import MLX
import MLXNN

/// Re-derive the SSM companion state for a hybrid model by running a
/// fresh prompt-only forward pass.
/// Convenience wrapper used by the `cacheStoreAction` closures in
/// `Evaluate.swift`. Decides whether re-derive should fire based on
/// the feature flag + model hybridness + genPromptLen, runs the
/// prompt-only forward pass, and stores the clean state directly in
/// `coordinator.ssmStateCache`. Swallows errors so re-derive never
/// breaks the main generation path.
public func maybeReDeriveSSMState(
    coordinator: CacheCoordinator,
    model: any LanguageModel,
    promptTokenIds: [Int],
    genPromptLen: Int,
    enableSSMReDerive: Bool
) {
    // iter-45: stderr breadcrumb so users + operators can see whether
    // the SSM helper watcher actually fires. The counter (reDerives)
    // only bumps on the happy path; without logs, a user seeing
    // reDerives=0 couldn't tell if the feature is disabled, the model
    // isn't hybrid, the prompt has no gen-suffix, or the MLX prep
    // step threw. These `cache/ssm-rederive` lines make the decision
    // tree visible.
    func log(_ status: String) {
        let line = "[vmlx][cache/ssm-rederive] \(status) hybrid=\(coordinator.isHybrid) genGP=\(genPromptLen) promptLen=\(promptTokenIds.count) enabled=\(enableSSMReDerive)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    guard enableSSMReDerive else { log("skip/disabled"); return }
    guard coordinator.isHybrid else { log("skip/not-hybrid"); return }
    guard genPromptLen > 0 else { log("skip/no-gen-prompt"); return }
    guard promptTokenIds.count > genPromptLen else {
        log("skip/prompt-shorter-than-gp"); return
    }

    let stripped = Array(promptTokenIds.prefix(promptTokenIds.count - genPromptLen))
    guard !stripped.isEmpty else { log("skip/empty-stripped"); return }

    do {
        guard let states = try reDeriveSSMStates(
            model: model, tokens: stripped
        ) else { log("skip/no-ssm-states"); return }
        coordinator.ssmStateCache.store(
            ssmStates: states,
            tokens: stripped,
            boundary: stripped.count
        )
        // Surface the re-derive event as a stats counter so
        // users watching the CachePanel can see "hybrid SSM helper
        // watcher" activity instead of wondering whether it ever ran.
        coordinator.ssmStateCache.markReDeriveFired()
        log("ok/stored stateCount=\(states.count)")
    } catch {
        // Best-effort. A failed re-derive just means the next turn
        // re-prefills normally — no worse than the pre-2026-04-14
        // contamination-skip behavior. Log the specific error so
        // operators can see if this is a recurring failure mode on
        // their platform/model combo.
        log("fail/\(error)")
    }
}

public func reDeriveSSMStates(
    model: any LanguageModel,
    tokens: [Int],
    prefillStepSize: Int = 512
) throws -> [MLXArray]? {
    guard !tokens.isEmpty else { return nil }

    let freshCache: [KVCache] = model.newCache(parameters: nil)

    // Pure-attention model check: if no cache layer looks Mamba-ish,
    // there's no SSM state to re-derive and we bail immediately.
    let hasSSMLayer = freshCache.contains { cache in
        let desc = String(describing: type(of: cache))
        return desc.contains("Mamba") || desc.contains("Arrays")
    }
    guard hasSSMLayer else { return nil }

    // Shape tokens as [1, L] — every prefill call site in Evaluate.swift
    // and BatchEngine expects the batch axis for sliding-window attention.
    let tokenArray = MLXArray(tokens.map { Int32($0) })
        .reshaped([1, tokens.count])
    let input = LMInput(text: LMInput.Text(tokens: tokenArray))

    // iter-45 FIX: the default `LLMModel.prepare` only runs chunks of
    // length `prefillStepSize` through the forward pass, and returns
    // any remaining tail as a `PrepareResult.tokens(...)` for the
    // caller to run. Prior re-derive code discarded that tail — so
    // for short prompts (len ≤ prefillStepSize, the common case for
    // chat single-turn requests) ZERO forward passes happened and the
    // SSM cache stayed empty, producing `skip/no-ssm-states` on every
    // attempt. Fix: collect the prepare tail and explicitly run it
    // through `callAsFunction` so the whole prompt touches every
    // mamba layer, writing `conv_state` + `hidden_state` into the
    // fresh cache.
    let result = try model.prepare(
        input, cache: freshCache, windowSize: prefillStepSize)
    switch result {
    case .tokens(let tail):
        if tail.tokens.size > 0 {
            let tailInput: MLXArray
            if tail.tokens.shape.count >= 2 {
                tailInput = tail.tokens
            } else {
                tailInput = tail.tokens.reshaped([1, tail.tokens.size])
            }
            _ = model.callAsFunction(tailInput, cache: freshCache)
        }
    case .logits:
        break
    @unknown default:
        break
    }
    MLX.eval(freshCache)

    let states = extractSSMStates(from: freshCache)
    return states.isEmpty ? nil : states
}
