// SPDX-License-Identifier: Apache-2.0
//
// §378 regression guard — verify the sampling fallback priority chain
// that buildGenerateParameters (Stream.swift) uses to combine:
//   1. ChatRequest.temperature/topP/etc.  (explicit HTTP body)
//   2. session/chat override via SettingsStore cascade
//   3. ModelGenerationDefaults (generation_config.json)
//   4. compiled-in global default
//
// The first §373 draft had tiers 2 and 3 swapped, so a user who set
// session temperature=1.2 got silently replaced with the model's
// recommended 0.6 from generation_config.json. §378 uses
// ResolvedSettings.resolutionTrace to tell whether tier 2 actually
// populated the field; if the trace says `.global`, the compiled-in
// builtin is in use and the model default can safely take over.
//
// Covers only the SettingsStore resolver / trace behavior — the
// Stream.swift consumer is integration-covered by the harness.

import XCTest
@testable import vMLXEngine

final class SamplingFallbackPriorityTests: XCTestCase {

    private func tempDBPath() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-fallback-\(UUID()).sqlite3")
    }

    /// When no one overrides temperature at session/chat/request level,
    /// `resolutionTrace["defaultTemperature"]` must be `.global` so the
    /// downstream sampler knows it's safe to swap in the model's
    /// recommended value from generation_config.json.
    func testTraceSaysGlobalWhenNothingOverrides() async throws {
        let url = tempDBPath()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = SettingsStore(database: SettingsDB(customPath: url))
        let r = await store.resolved()
        XCTAssertEqual(r.resolutionTrace["defaultTemperature"], .global,
                       "Untouched defaultTemperature must trace to .global")
        XCTAssertEqual(r.resolutionTrace["defaultTopP"], .global)
        XCTAssertEqual(r.resolutionTrace["defaultTopK"], .global)
        XCTAssertEqual(r.resolutionTrace["defaultMaxTokens"], .global)
        XCTAssertEqual(r.resolutionTrace["defaultRepetitionPenalty"], .global)
    }

    /// When the session sets defaultTemperature, the trace must record
    /// `.session`. Stream.swift §378 reads this and does NOT substitute
    /// the generation_config value — the user's explicit override wins.
    func testTraceSaysSessionWhenSessionOverrides() async throws {
        let url = tempDBPath()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = SettingsStore(database: SettingsDB(customPath: url))

        let id = UUID()
        var s = SessionSettings(modelPath: URL(fileURLWithPath: "/dev/null"))
        s.defaultTemperature = 1.2
        await store.setSession(id, s)

        let r = await store.resolved(sessionId: id)
        XCTAssertEqual(r.temperature, 1.2, accuracy: 1e-9,
                       "Session override must surface through resolved.temperature")
        XCTAssertEqual(r.resolutionTrace["defaultTemperature"], .session,
                       "Session-sourced temperature must trace to .session")
    }

    /// When the request body sets temperature, the trace must record
    /// `.request`. Same §378 rule — no generation_config substitution.
    func testTraceSaysRequestWhenRequestOverrides() async throws {
        let url = tempDBPath()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = SettingsStore(database: SettingsDB(customPath: url))

        let override = RequestOverride(
            temperature: 0.33,
            topP: nil, topK: nil, minP: nil,
            repetitionPenalty: nil, maxTokens: nil,
            systemPrompt: nil, stopSequences: nil,
            enableThinking: nil, reasoningEffort: nil,
            toolChoice: nil, tools: nil
        )
        let r = await store.resolved(request: override)
        XCTAssertEqual(r.temperature, 0.33, accuracy: 1e-9)
        XCTAssertEqual(r.resolutionTrace["defaultTemperature"], .request)
    }

    /// request > session priority: if BOTH set, request wins and trace
    /// records `.request`.
    func testRequestOverridesSessionPriority() async throws {
        let url = tempDBPath()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = SettingsStore(database: SettingsDB(customPath: url))

        let id = UUID()
        var s = SessionSettings(modelPath: URL(fileURLWithPath: "/dev/null"))
        s.defaultTemperature = 0.9
        await store.setSession(id, s)

        let override = RequestOverride(
            temperature: 0.1,
            topP: nil, topK: nil, minP: nil,
            repetitionPenalty: nil, maxTokens: nil,
            systemPrompt: nil, stopSequences: nil,
            enableThinking: nil, reasoningEffort: nil,
            toolChoice: nil, tools: nil
        )
        let r = await store.resolved(sessionId: id, request: override)
        XCTAssertEqual(r.temperature, 0.1, accuracy: 1e-9,
                       "Request must beat session in cascade")
        XCTAssertEqual(r.resolutionTrace["defaultTemperature"], .request)
    }
}
