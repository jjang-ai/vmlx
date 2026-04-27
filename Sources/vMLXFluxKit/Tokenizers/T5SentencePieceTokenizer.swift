// SPDX-License-Identifier: Apache-2.0
//
// T5SentencePieceTokenizer — thin adapter over swift-transformers'
// AutoTokenizer for FLUX.1's T5-XXL caption encoder.
//
// FLUX.1 ships its T5-XXL tokenizer under `text_encoder_2/tokenizer.json`
// in HuggingFace tokenizers JSON format (Unigram model derived from
// the original SentencePiece). swift-transformers loads that natively,
// so this wrapper just routes the directory and applies T5 padding /
// EOS conventions on top of the raw token IDs.
//
// T5 v1.1 vocab specifics:
//   - EOS = 1 (always appended after the prompt tokens)
//   - PAD = 0 (right-pad to `maxLen`)
//   - BOS = (none) — T5 has no BOS
//
// FLUX.1 Schnell uses maxLen=256 (matches Track 1's
// `T5XXLEncoder(maxSeqLen: 256)`). FLUX.1 Dev uses maxLen=512.
//
// Reference:
//   /tmp/mflux-ref/src/mflux/models/flux/model/flux_text_encoder/t5_encoder/

import Foundation
@preconcurrency import Tokenizers

public struct T5SentencePieceTokenizer: Sendable {
    public let inner: any Tokenizers.Tokenizer
    public let maxLen: Int

    /// T5 v1.1 EOS / PAD token IDs.
    public static let eosTokenID: Int = 1
    public static let padTokenID: Int = 0

    public init(inner: any Tokenizers.Tokenizer, maxLen: Int) {
        self.inner = inner
        self.maxLen = maxLen
    }

    /// Async loader. Reads `text_encoder_2/tokenizer.json` from a FLUX.1
    /// snapshot. Falls back to the snapshot root if the tokenizer file is
    /// hoisted to top level (some HF mirrors do this for the T5 piece).
    ///
    /// `maxLen` MUST match the T5 encoder's `maxSeqLen` so the output
    /// shape `(B, maxLen, 4096)` lines up with the DiT's `txtIn` slot.
    public static func load(modelPath: URL, maxLen: Int) async throws -> T5SentencePieceTokenizer {
        let teDir = modelPath.appendingPathComponent("text_encoder_2")
        let dirToUse: URL
        if FileManager.default.fileExists(atPath:
            teDir.appendingPathComponent("tokenizer.json").path)
        {
            dirToUse = teDir
        } else if FileManager.default.fileExists(atPath:
            modelPath.appendingPathComponent("tokenizer.json").path)
        {
            dirToUse = modelPath
        } else if FileManager.default.fileExists(atPath:
            modelPath.appendingPathComponent("tokenizer_2/tokenizer.json").path)
        {
            // Diffusers-format snapshots ship the T5 tokenizer under
            // `tokenizer_2/` separate from the encoder weights.
            dirToUse = modelPath.appendingPathComponent("tokenizer_2")
        } else {
            throw FluxError.weightsNotFound(teDir)
        }
        let upstream = try await AutoTokenizer.from(modelFolder: dirToUse)
        return T5SentencePieceTokenizer(inner: upstream, maxLen: maxLen)
    }

    /// Encode `prompt` to a fixed-length T5 token-id list:
    ///   - run the upstream tokenizer (no special tokens — T5 has no BOS)
    ///   - append EOS=1 if the upstream output didn't already end with EOS
    ///   - truncate to `maxLen`, ensuring the last token stays EOS=1
    ///   - right-pad with PAD=0 to `maxLen`
    ///
    /// The output count is exactly `maxLen` so the caller can reshape to
    /// `(1, maxLen)` for the encoder.
    public func encode(_ prompt: String) -> [Int] {
        var ids = inner.encode(text: prompt, addSpecialTokens: true)
        if ids.last != Self.eosTokenID {
            ids.append(Self.eosTokenID)
        }
        if ids.count > maxLen {
            // Truncate but keep EOS at the tail.
            ids = Array(ids.prefix(maxLen - 1)) + [Self.eosTokenID]
        } else if ids.count < maxLen {
            ids += Array(repeating: Self.padTokenID, count: maxLen - ids.count)
        }
        return ids
    }
}
