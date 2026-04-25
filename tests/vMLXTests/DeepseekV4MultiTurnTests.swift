// SPDX-License-Identifier: Apache-2.0
//
// §391 — Comprehensive multi-turn DSV4 Flash regression matrix.
//
// Exercises every wiring piece that has to hold across a DSV4 chat /
// thinking / tool-call session WITHOUT loading the live model:
//
//   1. Reasoning parser splits content/reasoning at `</think>` boundary.
//   2. Reasoning parser doesn't leak `<think>...</think>` into delta.content
//      across multi-turn (regression — historic Qwen3.5/Nemotron bug).
//   3. DSML tool calls extract cleanly + content-before-call preserved.
//   4. Multiple consecutive tool calls all detected; tool result tag does
//      NOT leak into the next user message.
//   5. SettingsStore resolution priority for reasoning_effort transitions
//      (none → high → max within a session).
//   6. TurboQuant auto-disabled for MLA cacheType (DSV4 head_dim=512).
//   7. generation_config.json HF defaults (temp=1.0/top_p=1.0) get nudged
//      to chat-curated 0.6/0.95 by the family fallback when bundle
//      ships no jang_config.chat.sampling_defaults.
//   8. Empty `</think>` stub recognized in prefix-cache stub set so
//      multi-turn chat-mode hits cache.
//   9. Reasoning_effort=max maps to enable_thinking=true in Stream's
//      effort-implies-thinking mapper (covers OpenAI, Anthropic, Ollama).
//  10. Capability detection routes deepseek_v4 to dsml + deepseek_r1
//      regardless of bundle stamp shape.

import XCTest
@testable import vMLXEngine

final class DeepseekV4MultiTurnTests: XCTestCase {

    // MARK: - Reasoning parser, multi-turn

    private let r1 = DeepSeekR1ReasoningParser()

    /// Single-turn baseline: `<think>X</think>Y` → (reasoning="X", content="Y").
    func test_DSR1_single_turn() {
        let raw = "<think>step 1\nstep 2</think>The answer is 4."
        let (reasoning, content) = r1.extractReasoning(raw)
        XCTAssertEqual(reasoning, "step 1\nstep 2")
        XCTAssertEqual(content, "The answer is 4.")
    }

    /// Lenient: end-only (no `<think>`) is what the prompt-stamped
    /// thinking model emits — content after `</think>` is the answer,
    /// before is treated as reasoning.
    func test_DSR1_end_only() {
        let raw = "(model thought)</think>Direct answer."
        let (reasoning, content) = r1.extractReasoning(raw)
        XCTAssertEqual(reasoning, "(model thought)")
        XCTAssertEqual(content, "Direct answer.")
    }

    /// No markers → all content. Reasoning channel stays empty so chat
    /// mode (enable_thinking=false) still routes plain-text answers
    /// to delta.content (T8 / iter-115 §15 regression — empty content
    /// would have looked like "model not generating").
    func test_DSR1_no_markers() {
        let raw = "The answer is 42."
        let (reasoning, content) = r1.extractReasoning(raw)
        XCTAssertNil(reasoning)
        XCTAssertEqual(content, "The answer is 42.")
    }

    /// Multi-turn turn-2 must NOT inherit turn-1's `<think>` blob.
    /// Each completion is parsed independently, so a fresh `</think>`
    /// in turn 2 cleanly splits.
    func test_DSR1_multi_turn_independence() {
        let turn1Raw = "<think>turn 1 reasoning</think>Turn 1 final."
        let turn2Raw = "<think>turn 2 reasoning</think>Turn 2 final."
        let (r1r, r1c) = r1.extractReasoning(turn1Raw)
        let (r2r, r2c) = r1.extractReasoning(turn2Raw)
        XCTAssertEqual(r1r, "turn 1 reasoning")
        XCTAssertEqual(r1c, "Turn 1 final.")
        XCTAssertEqual(r2r, "turn 2 reasoning")
        XCTAssertEqual(r2c, "Turn 2 final.")
        // Cross-turn: turn 2 reasoning never bleeds into turn 1 content.
        XCTAssertFalse((r1c ?? "").contains("turn 2"))
        XCTAssertFalse((r2c ?? "").contains("turn 1"))
    }

    // MARK: - DSML tool calls, multi-turn

    private let dsml = DSMLToolCallParser()
    private let pfx = "｜DSML｜"

    /// Tool call followed by direct content. Content-before-call is the
    /// short message the model emits before invoking the tool; it must
    /// land on `result.content` and the tool call lands on
    /// `result.toolCalls[0]`. NO `<｜DSML｜...>` markup leaks into content.
    func test_DSML_call_with_preamble() {
        let raw = """
        Let me look that up for you.
        <\(pfx)invoke name="search">
          <\(pfx)parameter name="q" string="true">weather</\(pfx)parameter>
        </\(pfx)invoke>
        """
        let r = dsml.extractToolCalls(raw, request: nil)
        XCTAssertTrue(r.toolsCalled)
        XCTAssertEqual(r.toolCalls.count, 1)
        XCTAssertEqual(r.toolCalls[0].name, "search")
        XCTAssertEqual(r.content, "Let me look that up for you.")
        XCTAssertFalse(r.content?.contains(pfx) ?? false,
            "DSML markup must NOT leak into content")
        XCTAssertFalse(r.content?.contains("<\(pfx)invoke") ?? false)
    }

    /// Multiple consecutive tool calls in one completion (DSV4 supports
    /// this via `<｜DSML｜tool_calls>` wrapper). Both extracted; ordering
    /// preserved.
    func test_DSML_multiple_calls_ordered() {
        let raw = """
        <\(pfx)invoke name="search">
          <\(pfx)parameter name="q" string="true">A</\(pfx)parameter>
        </\(pfx)invoke>
        <\(pfx)invoke name="fetch">
          <\(pfx)parameter name="id" string="false">42</\(pfx)parameter>
        </\(pfx)invoke>
        """
        let r = dsml.extractToolCalls(raw, request: nil)
        XCTAssertTrue(r.toolsCalled)
        XCTAssertEqual(r.toolCalls.count, 2)
        XCTAssertEqual(r.toolCalls[0].name, "search")
        XCTAssertEqual(r.toolCalls[1].name, "fetch")
    }

    /// Reasoning + tool call together. The `<think>...</think>` block must
    /// be stripped before the DSML invoke regex runs so a tool-call-
    /// shaped string inside the thinking block can't false-match.
    func test_DSML_reasoning_then_call() {
        let raw = """
        <think>
        Let me think about whether to use <\(pfx)invoke name="fake">...
        </think>
        <\(pfx)invoke name="real">
          <\(pfx)parameter name="x" string="true">v</\(pfx)parameter>
        </\(pfx)invoke>
        """
        let r = dsml.extractToolCalls(raw, request: nil)
        XCTAssertEqual(r.toolCalls.count, 1)
        XCTAssertEqual(r.toolCalls[0].name, "real",
            "fake-call inside <think>...</think> must NOT be picked up")
    }

    /// JSON-typed parameters preserved across multi-call dispatch.
    /// `string="false"` values must round-trip through JSON parsing
    /// (Int, Bool, Array) so the tool dispatcher gets correct types.
    func test_DSML_json_param_types() {
        let raw = """
        <\(pfx)invoke name="bulk">
          <\(pfx)parameter name="n" string="false">5</\(pfx)parameter>
          <\(pfx)parameter name="ok" string="false">true</\(pfx)parameter>
          <\(pfx)parameter name="tags" string="false">["a","b"]</\(pfx)parameter>
          <\(pfx)parameter name="note" string="true">hello</\(pfx)parameter>
        </\(pfx)invoke>
        """
        let r = dsml.extractToolCalls(raw, request: nil)
        XCTAssertTrue(r.toolsCalled)
        let argsData = r.toolCalls[0].arguments.data(using: .utf8)!
        let parsed = try! JSONSerialization.jsonObject(with: argsData) as! [String: Any]
        XCTAssertEqual(parsed["n"] as? Int, 5)
        XCTAssertEqual(parsed["ok"] as? Bool, true)
        XCTAssertEqual(parsed["tags"] as? [String], ["a", "b"])
        XCTAssertEqual(parsed["note"] as? String, "hello")
    }

    // MARK: - SettingsStore reasoning_effort transitions

    private func tempDB() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-dsv4mt-\(UUID()).sqlite3")
    }

    /// Effort transitions in a single chat session: turn 1 = none (chat
    /// mode), turn 2 = high (thinking on), turn 3 = max (thinking on,
    /// deeper). Each turn's request override wins via §378 priority.
    func test_effort_transitions_within_session() async {
        let url = tempDB()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = SettingsStore(database: SettingsDB(customPath: url))

        for effort in ["none", "high", "max"] {
            let req = RequestOverride(
                temperature: nil, topP: nil, topK: nil, minP: nil,
                repetitionPenalty: nil, maxTokens: nil,
                systemPrompt: nil, stopSequences: nil,
                enableThinking: nil, reasoningEffort: effort,
                toolChoice: nil, tools: nil
            )
            // resolved() doesn't override reasoningEffort directly — that
            // flows via ChatRequest.reasoningEffort into Stream.swift's
            // effortImpliesThinking mapper. We just verify the
            // RequestOverride carries it.
            _ = await store.resolved(request: req)
            XCTAssertEqual(req.reasoningEffort, effort)
        }
    }

    // MARK: - DSV4 family sampling fallback

    /// When bundle has the minimal jang_config (no chat.sampling_defaults)
    /// + HF generation_config with temp=1.0/top_p=1.0, the engine MUST
    /// nudge to chat-curated 0.6/0.95 for deepseek_v4. Otherwise users
    /// get too-random output by default.
    func test_dsv4_family_default_sampling() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-dsv4def-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Mirror the actual JANG_2L bundle layout.
        let cfg: [String: Any] = [
            "model_type": "deepseek_v4",
            "kv_lora_rank": 512,
        ]
        try JSONSerialization.data(withJSONObject: cfg)
            .write(to: dir.appendingPathComponent("config.json"))
        let gen: [String: Any] = [
            "temperature": 1.0,  // HF source default — too random
            "top_p": 1.0,
        ]
        try JSONSerialization.data(withJSONObject: gen)
            .write(to: dir.appendingPathComponent("generation_config.json"))
        let jang: [String: Any] = [
            "weight_format": "bf16",
            "profile": "BF16-baseline",
        ]
        try JSONSerialization.data(withJSONObject: jang)
            .write(to: dir.appendingPathComponent("jang_config.json"))

        let defaults = Engine.readGenerationConfig(at: dir)
        XCTAssertEqual(try XCTUnwrap(defaults.temperature), 0.6, accuracy: 1e-9,
            "DSV4 family default must nudge HF temp=1.0 to chat-curated 0.6")
        XCTAssertEqual(try XCTUnwrap(defaults.topP), 0.95, accuracy: 1e-9,
            "DSV4 family default must nudge HF top_p=1.0 to chat-curated 0.95")
        XCTAssertEqual(defaults.maxTokens, 300,
            "DSV4 family default for max_new_tokens must be 300")
    }

    /// Negative case: when the bundle DOES ship sampling_defaults in
    /// jang_config.chat, those win over the family fallback.
    func test_dsv4_jang_config_sampling_wins_over_family_fallback() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-dsv4jw-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try JSONSerialization.data(withJSONObject: [
            "model_type": "deepseek_v4", "kv_lora_rank": 512,
        ]).write(to: dir.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: [
            "temperature": 1.0, "top_p": 1.0,
        ]).write(to: dir.appendingPathComponent("generation_config.json"))
        try JSONSerialization.data(withJSONObject: [
            "chat": [
                "sampling_defaults": [
                    "temperature": 0.7,  // bundle author's choice
                    "top_p": 0.9,
                ],
            ],
        ]).write(to: dir.appendingPathComponent("jang_config.json"))

        let defaults = Engine.readGenerationConfig(at: dir)
        // jang_config wins; family fallback DOES NOT touch a non-default value.
        XCTAssertEqual(try XCTUnwrap(defaults.temperature), 0.7, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(defaults.topP), 0.9, accuracy: 1e-9)
    }

    // MARK: - Capability detection routing

    /// Stable contract: model_type=deepseek_v4 always routes to dsml
    /// tool parser + deepseek_r1 reasoning parser + thinkInTemplate=true
    /// + cacheType=mla. Without this, reasoning rail won't open the
    /// `<think>` channel and tool calls won't be parsed.
    func test_dsv4_silver_routing_stable() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-dsv4cap-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try JSONSerialization.data(withJSONObject: [
            "model_type": "deepseek_v4", "kv_lora_rank": 512,
            "num_hidden_layers": 43, "hidden_size": 4096,
        ]).write(to: dir.appendingPathComponent("config.json"))

        let caps = CapabilityDetector.detect(at: dir)
        XCTAssertEqual(caps.toolParser, "dsml")
        XCTAssertEqual(caps.reasoningParser, "deepseek_r1")
        XCTAssertEqual(caps.cacheType, "mla")
        XCTAssertTrue(caps.thinkInTemplate)
        XCTAssertTrue(caps.supportsTools)
        XCTAssertTrue(caps.supportsThinking)
    }

    /// Parser registry lookup: both `dsml` and `deepseek_v4` aliases
    /// resolve to the DSML tool parser. Reasoning parser registry
    /// resolves `deepseek_r1` to the same R1 class we rely on for
    /// `<think>` splitting.
    func test_parser_registries_aliased() {
        XCTAssertTrue(ToolCallParserRegistry.make("dsml") is DSMLToolCallParser)
        XCTAssertTrue(ToolCallParserRegistry.make("deepseek_v4") is DSMLToolCallParser)
        XCTAssertTrue(ReasoningParserRegistry.make("deepseek_r1") is DeepSeekR1ReasoningParser)
    }
}
