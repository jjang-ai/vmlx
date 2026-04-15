// KokoroBackend — scaffolding for the neural TTS backend.
//
// Chosen model family: **Kokoro** (StyleTTS-based, ~80 MB, Apache 2.0).
// Rationale:
//   • smallest high-quality open TTS we could port to MLX/Swift
//   • single speaker embedding file (voices are ~256-dim style vectors
//     baked into one .pt per voice — trivial to serialize as safetensors)
//   • HiFi-GAN vocoder is a well-known stack; an MLX port exists in
//     `mlx-examples/kokoro` (reference Python) and in the hf mlx-community
//     org under `mlx-community/Kokoro-82M-*`
//   • English-only G2P via a bundled CMU-dict subset keeps us off eSpeak
//
// PORT STATUS: NOT YET IMPLEMENTED.
//
// Work items (ordered):
//   1. Config.swift — decode `config.json` (hidden_dim, n_mels, voice ids)
//   2. Phonemizer — ship a compact CMU-dict subset (~30k entries) as a
//      Swift resource; fall back to letter-to-sound rules on miss.
//   3. TextEncoder — 6-layer transformer, MLXNN port.
//   4. DurationPredictor — 3-layer LSTM + linear head; MLX LSTM exists.
//   5. StyleEncoder — loads the per-voice style vector.
//   6. Decoder — FFT-based upsampler feeding the HiFi-GAN vocoder.
//   7. HiFiGAN — transposed-conv upsample + residual blocks; MLXNN.
//   8. Weight mapping — safetensors → MLX module tree.
//   9. Sampling at 24 kHz → Int16 PCM buffer.
//
// Drop-in hook: once implemented, `TTSEngine.synthesize(_:)` switches
// from `PlaceholderSynth.renderPCM` to `KokoroBackend.renderPCM` based
// on whether a Kokoro model directory is resolvable on disk.
//
// Suggested model download (Mac Studio / MacBook):
//   huggingface-cli download mlx-community/Kokoro-82M-bf16
//   → ~/.cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16
//
// The module is left empty on purpose — the compile budget for this
// session is the placeholder path + route wiring. See the vMLXTTS
// README block at the top of `TTSEngine.swift` for handoff context.

import Foundation

public enum KokoroBackend {
    /// Returns `nil` — not yet ported. When implemented, this will
    /// produce mono 24 kHz Int16 PCM from a phonemized utterance.
    public static func renderPCM(text: String,
                                 voice: String,
                                 speed: Double,
                                 modelDirectory: URL) -> [Int16]? {
        return nil
    }
}
