// EngineTTS — routes `/v1/audio/speech` requests into the vMLXTTS module.
//
// Kept separate from Engine.swift to preserve the 2000-line chat core.
// The TTS backend itself lives in `Sources/vMLXTTS/` and is selectable
// between a PlaceholderSynth (always available) and a neural Kokoro
// backend (scaffolded, not yet ported — see KokoroBackend.swift).

import Foundation
import vMLXTTS

extension Engine {

    /// Synthesize speech for an OpenAI-compatible `/v1/audio/speech`
    /// request. Returns raw audio bytes plus metadata so the server
    /// layer can set content-type and diagnostic headers.
    public func synthesizeSpeech(request: [String: Any]) async throws -> TTSResult {
        do {
            return try TTSEngine().synthesize(request: request)
        } catch let e as TTSError {
            // iter-120 §146: split TTSError into user-input errors (400
            // via EngineError.invalidRequest) vs server-capability
            // errors (500 via notImplemented). Previously both fell into
            // the same .notImplemented bucket, so a client POSTing `{}`
            // (missing `input`) or `{response_format: "flac"}` (valid
            // format but unsupported on our backend) got a 500. OpenAI
            // spec rejects both of those with 400, and SDK clients
            // expect 400 to drive retry/fallback logic.
            switch e {
            case .missingInput, .unsupportedFormat:
                throw EngineError.invalidRequest("speech: \(e.description)")
            case .encoderUnavailable, .modelNotPorted:
                throw EngineError.notImplemented("speech: \(e.description)")
            }
        }
    }
}
