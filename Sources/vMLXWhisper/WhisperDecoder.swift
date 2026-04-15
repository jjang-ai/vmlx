// Copyright © 2026 vMLX. Whisper greedy decoder + public transcribe
// entry point.
//
// Implements a simplified version of mlx-examples/whisper's
// DecodingTask: no temperature fallback, no beam search, no word-
// level timestamps — just greedy argmax over the logits, stopping at
// the `<|endoftext|>` token. This is sufficient for the first
// `/v1/audio/transcriptions` milestone and can be extended in-place
// without changing the public API.
//
// TODO (deferred):
//   - Temperature fallback (0→0.2→…→1.0) with compression-ratio +
//     logprob thresholds.
//   - Timestamp-aware decoding & segment emission.
//   - Beam search.
//   - Voice-activity-detection-based 30-second chunk stitching.

import Foundation
import MLX

public struct TranscribeResult: Sendable {
    public let text: String
    public let language: String
    public let durationSeconds: Double
    public let tokens: [Int]
}

public enum WhisperDecoder {

    /// Run a full audio-file → text transcription. `samples` is the
    /// raw 16 kHz mono Float32 PCM. `language` may be nil (English-
    /// only models ignore it; multilingual falls back to "en").
    public static func transcribe(
        loaded: LoadedWhisper,
        samples: [Float],
        language: String? = nil,
        task: String = "transcribe",
        maxNewTokens: Int = 224
    ) -> TranscribeResult {
        let config = loaded.config
        let model = loaded.model
        let tokenizer = loaded.tokenizer

        let durationSec = Double(samples.count) / Double(WhisperAudio.sampleRate)

        // Currently we only decode the first 30-second chunk. Longer
        // audio gets its tail truncated. (Follow-up: stitched chunked
        // decoding via seek offsets + timestamp tokens.)
        let mel = WhisperAudio.melArray(
            from: samples, nMels: config.nMels)
        let audioFeatures = model.encoder(mel)

        var tokens = tokenizer.initialPromptTokens(
            language: language, task: task, withoutTimestamps: true)
        let promptLen = tokens.count

        for _ in 0 ..< maxNewTokens {
            let tokenArray = MLXArray(tokens.map { Int32($0) }, [1, tokens.count])
            let logits = model.decoder(tokenArray, audioFeatures: audioFeatures)
            // logits shape: [1, T, vocab]; take the last step.
            let last = logits[0, logits.dim(1) - 1, 0...]
            // Argmax.
            let nextTok = Int(argMax(last).item(Int32.self))
            if nextTok == tokenizer.special.eot { break }
            tokens.append(nextTok)
            if tokens.count >= config.nTextCtx - 1 { break }
        }

        let generated = Array(tokens[promptLen ..< tokens.count])
        let text = tokenizer.decode(generated)
        let effectiveLang = language ?? "en"
        return TranscribeResult(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            language: effectiveLang,
            durationSeconds: durationSec,
            tokens: generated)
    }
}
