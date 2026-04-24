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
