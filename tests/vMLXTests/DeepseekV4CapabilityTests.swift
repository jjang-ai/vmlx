// SPDX-License-Identifier: Apache-2.0
//
// §385 DSV4 capability detection guard.
//
// Ensures the model_type=deepseek_v4 silver row resolves correctly
// via CapabilityDetector, so JANG and JANGTQ bundles that stamp
// `config.json::model_type = "deepseek_v4"` pick up:
//   - cacheType = "mla"           (head_dim=512 MLA latent)
//   - toolParser = "dsml"         (new DSV4 tool envelope)
//   - reasoningParser = "deepseek_r1"  (<think>...</think>)
//   - thinkInTemplate = true
//   - family = "deepseek_v4"

import XCTest
@testable import vMLXEngine

final class DeepseekV4CapabilityTests: XCTestCase {
    private func tempDir(name: String) -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-dsv4cap-\(UUID())-\(name)")
        try! FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    private func write(_ json: [String: Any], to file: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted])
        try data.write(to: file)
    }

    /// Bare config.json with model_type=deepseek_v4 + kv_lora_rank marker
    /// for MLA must resolve to the deepseek_v4 silver row.
    func testSilverRowForDSV4() throws {
        let d = tempDir(name: "silver")
        defer { try? FileManager.default.removeItem(at: d) }
        try write([
            "model_type": "deepseek_v4",
            "kv_lora_rank": 512,
            "num_hidden_layers": 43,
            "hidden_size": 4096,
        ], to: d.appendingPathComponent("config.json"))

        let caps = CapabilityDetector.detect(at: d)
        XCTAssertEqual(caps.family, "deepseek_v4")
        XCTAssertEqual(caps.modelType, "deepseek_v4")
        XCTAssertEqual(caps.cacheType, "mla")
        XCTAssertEqual(caps.toolParser, "dsml")
        XCTAssertEqual(caps.reasoningParser, "deepseek_r1")
        XCTAssertTrue(caps.thinkInTemplate)
        XCTAssertTrue(caps.supportsTools)
        XCTAssertTrue(caps.supportsThinking)
    }

    /// §389 — bundle that ships NO `tokenizer_config.json::chat_template`
    /// (DSV4 uses Python encoding/encoding_dsv4.py which Swift can't
    /// exec) must still get a usable Jinja template via the built-in
    /// fallback in `silverChatTemplate`. Without this, in-app chat /
    /// /v1/chat/completions / /api/chat would all fail at template
    /// render time with "no chat_template available".
    func testBuiltinChatTemplateForDSV4() throws {
        let d = tempDir(name: "tpl")
        defer { try? FileManager.default.removeItem(at: d) }
        // Mirror the actual JANG_2L bundle layout: HF config + minimal
        // tokenizer_config.json without a chat_template.
        try write([
            "model_type": "deepseek_v4",
            "kv_lora_rank": 512,
        ], to: d.appendingPathComponent("config.json"))
        try write([
            "add_bos_token": false,
            "add_eos_token": false,
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            // explicitly NO chat_template field
        ], to: d.appendingPathComponent("tokenizer_config.json"))

        let caps = CapabilityDetector.detect(at: d)
        XCTAssertEqual(caps.modelType, "deepseek_v4")
        let tpl = try XCTUnwrap(caps.chatTemplate,
            "DSV4 bundle without tokenizer chat_template MUST get the built-in fallback")
        // Sanity-check the Jinja covers all three modes.
        XCTAssertTrue(tpl.contains("<｜begin▁of▁sentence｜>"))
        XCTAssertTrue(tpl.contains("<｜User｜>"))
        XCTAssertTrue(tpl.contains("<｜Assistant｜>"))
        XCTAssertTrue(tpl.contains("<think>"))
        XCTAssertTrue(tpl.contains("</think>"))
        XCTAssertTrue(tpl.contains("<｜end▁of▁sentence｜>"))
        XCTAssertTrue(tpl.contains("Reasoning Effort"),
            "max-effort prefix must be present so reasoning_effort=max gets the documented system blurb")
        XCTAssertTrue(tpl.contains("<tool_result>"),
            "tool_result tag must be in the template so tool messages render correctly")
    }

    /// jang_config.capabilities override must win over silver. If a
    /// bundle explicitly writes tool_parser=dsml + reasoning_parser=
    /// deepseek_r1 via the newer capabilities stamp shape, detection
    /// must honor those exact values.
    func testJANGStampOverridesSilver() throws {
        let d = tempDir(name: "jangstamp")
        defer { try? FileManager.default.removeItem(at: d) }
        try write([
            "model_type": "deepseek_v4",
            "kv_lora_rank": 512,
        ], to: d.appendingPathComponent("config.json"))
        try write([
            "capabilities": [
                "tool_parser": "dsml",
                "reasoning_parser": "deepseek_r1",
                "think_in_template": true,
                "cache_type": "mla",
                "family": "deepseek_v4",
                "supports_tools": true,
                "supports_thinking": true,
            ],
        ], to: d.appendingPathComponent("jang_config.json"))

        let caps = CapabilityDetector.detect(at: d)
        XCTAssertEqual(caps.toolParser, "dsml")
        XCTAssertEqual(caps.reasoningParser, "deepseek_r1")
        XCTAssertEqual(caps.cacheType, "mla")
        XCTAssertEqual(caps.family, "deepseek_v4")
        // detectionSource should be jangStamped not silver
        XCTAssertEqual(caps.detectionSource, .jangStamped)
    }
}
