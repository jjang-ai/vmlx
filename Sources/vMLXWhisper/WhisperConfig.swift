// Copyright © 2026 vMLX. Native port of mlx-whisper for Swift.
//
// Configuration container for OpenAI Whisper models hosted on the HF Hub
// (mlx-community/whisper-*). Mirrors the Python `ModelDimensions`
// dataclass used by mlx-examples/whisper — same field names, same JSON
// layout, so a `config.json` from an mlx-community whisper checkpoint
// drops in without a translation layer.

import Foundation

public struct WhisperConfig: Codable, Sendable {
    public let nMels: Int
    public let nAudioCtx: Int
    public let nAudioState: Int
    public let nAudioHead: Int
    public let nAudioLayer: Int
    public let nVocab: Int
    public let nTextCtx: Int
    public let nTextState: Int
    public let nTextHead: Int
    public let nTextLayer: Int

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case nAudioCtx = "n_audio_ctx"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nVocab = "n_vocab"
        case nTextCtx = "n_text_ctx"
        case nTextState = "n_text_state"
        case nTextHead = "n_text_head"
        case nTextLayer = "n_text_layer"
    }

    /// Multilingual checkpoints carry n_vocab == 51865 (v1/v2) or 51866
    /// (v3). English-only checkpoints carry 51864. Needed for language-
    /// token indexing during decoding.
    public var isMultilingual: Bool { nVocab >= 51865 }

    /// Whisper v3 bumped n_mels from 80 → 128. Preserve for mel setup.
    public var nMelsActual: Int { nMels }
}
