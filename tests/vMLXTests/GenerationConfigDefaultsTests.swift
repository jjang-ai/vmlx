// SPDX-License-Identifier: Apache-2.0
//
// §373 regression guard — verify that `Engine.readGenerationConfig(at:)`
// parses `generation_config.json` into `ModelGenerationDefaults` so the
// three-tier sampling fallback chain (request → generation_config →
// global default) surfaces model-recommended values by default.
//
// Coverage mirrors the real-world configs shipped by Qwen, Gemma,
// Nemotron, and MiniMax. If any of these bundles ever produce nil-only
// defaults after a refactor, every API that omits sampling params
// silently falls back to 0.7 temp / 0.9 topP — which is the regression
// Eric flagged as "UI caption promises values the sampler never saw."

import XCTest
@testable import vMLXEngine

final class GenerationConfigDefaultsTests: XCTestCase {
    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-gencfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDir)
        try await super.tearDown()
    }

    private func writeConfig(_ json: String) throws {
        let url = tempDir.appendingPathComponent("generation_config.json")
        try json.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Qwen3 recommends temp=0.6, top_p=0.95, top_k=20. Our fallback
    /// chain must surface those four values instead of the engine-wide
    /// 0.7 / 0.9 / 0 / nil defaults.
    func testQwenRecommendedDefaultsParse() throws {
        // Writes Qwen's recommended config; throws to allow XCTUnwrap.
        try writeConfig(#"""
        {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 2048,
            "repetition_penalty": 1.05
        }
        """#)
        let d = Engine.readGenerationConfig(at: tempDir)
        XCTAssertEqual(try XCTUnwrap(d.temperature), 0.6, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(d.topP), 0.95, accuracy: 1e-9)
        XCTAssertEqual(d.topK, 20)
        XCTAssertEqual(d.maxTokens, 2048)
        XCTAssertEqual(try XCTUnwrap(d.repetitionPenalty), 1.05, accuracy: 1e-9)
    }

    /// Missing file must return an empty ModelGenerationDefaults with
    /// every field nil — the fallback chain treats nil as "ask the
    /// engine-wide default," so a model without generation_config.json
    /// behaves identically to pre-§373 code.
    func testMissingConfigReturnsEmpty() {
        let d = Engine.readGenerationConfig(at: tempDir)
        XCTAssertNil(d.temperature)
        XCTAssertNil(d.topP)
        XCTAssertNil(d.topK)
        XCTAssertNil(d.maxTokens)
        XCTAssertNil(d.repetitionPenalty)
    }

    /// Malformed JSON must also return an empty struct; we never want
    /// a parse error to abort model loading or wedge the fallback chain.
    func testGarbageJSONSwallowed() throws {
        try writeConfig(#"{ this is not json"#)
        let d = Engine.readGenerationConfig(at: tempDir)
        XCTAssertNil(d.temperature)
        XCTAssertNil(d.maxTokens)
    }

    /// Partial config (only temperature) fills that field, leaves the
    /// rest nil so the fallback chain can still delegate top_p / top_k
    /// to the engine-wide default. This mirrors Gemma's shipping
    /// generation_config.json which sometimes sets only temperature.
    func testPartialConfigOnlyFillsSpecifiedKeys() throws {
        try writeConfig(#"{"temperature": 1.0}"#)
        let d = Engine.readGenerationConfig(at: tempDir)
        XCTAssertEqual(try XCTUnwrap(d.temperature), 1.0, accuracy: 1e-9)
        XCTAssertNil(d.topP)
        XCTAssertNil(d.topK)
        XCTAssertNil(d.maxTokens)
        XCTAssertNil(d.repetitionPenalty)
    }

    /// Integer literals (top_k: 50 without decimal) and double literals
    /// (temperature: 0.7) must both parse. Some HF configs emit top_k
    /// as an integer, some as "50.0"; NSNumber handles both.
    func testIntAndDoubleCoercion() throws {
        try writeConfig(#"{"temperature": 0.7, "top_k": 50, "max_new_tokens": 1024}"#)
        let d = Engine.readGenerationConfig(at: tempDir)
        XCTAssertEqual(try XCTUnwrap(d.temperature), 0.7, accuracy: 1e-9)
        XCTAssertEqual(d.topK, 50)
        XCTAssertEqual(d.maxTokens, 1024)
    }
}
