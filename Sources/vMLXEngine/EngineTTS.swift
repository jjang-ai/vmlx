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
            switch e {
            case .missingInput:
                throw EngineError.notImplemented("speech: \(e.description)")
            case .unsupportedFormat, .encoderUnavailable, .modelNotPorted:
                throw EngineError.notImplemented("speech: \(e.description)")
            }
        }
    }
}
