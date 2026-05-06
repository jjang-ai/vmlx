// SPDX-License-Identifier: Apache-2.0
//
// chat_template_kwargs decode helper — shared across OpenAI Chat,
// OpenAI Responses, Anthropic /v1/messages, and Ollama /api/chat /
// /api/generate so all four protocols forward the same shape into
// `ChatRequest.chatTemplateKwargs` (which Stream.swift merges into
// `templateExtras` last so caller wins over engine-set extras).
//
// Type inference for `[String: ChatRequest.JSONValue]` was tripping
// the Swift compiler when called inline at the route construction
// sites; this helper isolates the type so the surrounding closures
// stay simple.

import Foundation
import vMLXEngine

/// Decode an arbitrary `[String: Any]` chat_template_kwargs payload
/// into the typed `[String: JSONValue]` that ChatRequest expects.
/// Returns nil on any serialization / decode failure — caller
/// continues without the kwargs (graceful degradation; templates that
/// don't read those vars still render correctly).
func decodeChatTemplateKwargs(_ kwargs: [String: Any]) -> [String: JSONValue]? {
    guard let data = try? JSONSerialization.data(withJSONObject: kwargs) else {
        return nil
    }
    return try? JSONDecoder().decode([String: JSONValue].self, from: data)
}
