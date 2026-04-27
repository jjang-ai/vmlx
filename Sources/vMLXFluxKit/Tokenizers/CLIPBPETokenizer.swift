// SPDX-License-Identifier: Apache-2.0
//
// CLIPBPETokenizer — thin adapter over swift-transformers' AutoTokenizer
// for FLUX.1's CLIP-L pooled text encoder.
//
// FLUX.1 ships its CLIP-L tokenizer under `tokenizer/` in either:
//   - `tokenizer/tokenizer.json` (HF unified format)
//   - `tokenizer/{vocab.json, merges.txt}` (legacy openai/clip format)
//
// swift-transformers handles either layout — we just route the dir and
// apply CLIP's padding / special-token conventions on top.
//
// CLIP-L vocab specifics (matches openai/clip-vit-large-patch14):
//   - BOS  = 49406  (`<|startoftext|>`)
//   - EOS  = 49407  (`<|endoftext|>`)
//   - PAD  = 49407  (CLIP pads with EOS, not zero)
//   - max length = 77
//
// Reference:
//   /tmp/mflux-ref/src/mflux/models/flux/model/flux_text_encoder/clip_encoder/

import Foundation
@preconcurrency import Tokenizers

public struct CLIPBPETokenizer: Sendable {
    public let inner: any Tokenizers.Tokenizer
    public let maxLen: Int

    public static let bosTokenID: Int = 49406
    public static let eosTokenID: Int = 49407
    public static let padTokenID: Int = 49407
    public static let defaultMaxLen: Int = 77

    public init(inner: any Tokenizers.Tokenizer, maxLen: Int = CLIPBPETokenizer.defaultMaxLen) {
        self.inner = inner
        self.maxLen = maxLen
    }

    /// Async loader. Reads `tokenizer/` from a FLUX.1 snapshot. Falls
    /// back to the snapshot root if the tokenizer.json is hoisted there.
    public static func load(modelPath: URL, maxLen: Int = CLIPBPETokenizer.defaultMaxLen) async throws -> CLIPBPETokenizer {
        let tokDir = modelPath.appendingPathComponent("tokenizer")
        let dirToUse: URL
        if FileManager.default.fileExists(atPath: tokDir.appendingPathComponent("tokenizer.json").path)
            || FileManager.default.fileExists(atPath: tokDir.appendingPathComponent("vocab.json").path)
        {
            dirToUse = tokDir
        } else if FileManager.default.fileExists(atPath:
            modelPath.appendingPathComponent("text_encoder/tokenizer.json").path)
        {
            // Some snapshots colocate the CLIP tokenizer next to the
            // CLIP encoder weights under `text_encoder/`.
            dirToUse = modelPath.appendingPathComponent("text_encoder")
        } else {
            throw FluxError.weightsNotFound(tokDir)
        }
        let upstream = try await AutoTokenizer.from(modelFolder: dirToUse)
        return CLIPBPETokenizer(inner: upstream, maxLen: maxLen)
    }

    /// Encode `prompt` to a fixed-length CLIP-L token-id list:
    ///   - run the upstream tokenizer with special tokens disabled
    ///     so we control BOS / EOS placement explicitly
    ///   - prepend BOS=49406, append EOS=49407
    ///   - truncate to `maxLen`, ensuring the last token is EOS
    ///   - right-pad with PAD=49407 to `maxLen`
    ///
    /// The output count is exactly `maxLen` (default 77) so the caller
    /// can reshape to `(1, 77)` for the CLIP encoder.
    public func encode(_ prompt: String) -> [Int] {
        // We deliberately re-add BOS/EOS to match the reference impl's
        // explicit ID placement; if the underlying tokenizer already
        // added them, we strip duplicates so we don't end up with two
        // BOS / EOS in a row.
        var raw = inner.encode(text: prompt, addSpecialTokens: true)
        if raw.first == Self.bosTokenID { raw.removeFirst() }
        if raw.last == Self.eosTokenID { raw.removeLast() }

        var ids: [Int] = [Self.bosTokenID]
        ids.append(contentsOf: raw)
        ids.append(Self.eosTokenID)

        if ids.count > maxLen {
            // Truncate but keep EOS at the tail.
            ids = Array(ids.prefix(maxLen - 1)) + [Self.eosTokenID]
        } else if ids.count < maxLen {
            ids += Array(repeating: Self.padTokenID, count: maxLen - ids.count)
        }
        return ids
    }
}
