// SPDX-License-Identifier: Apache-2.0
//
// ZImageTokenizer — thin adapter over swift-transformers' AutoTokenizer
// for Z-Image's `text_encoder/tokenizer.json` (Qwen2 BPE).
//
// Z-Image uses Qwen-2's tokenizer (vocab 151936) shipped under the
// model snapshot's `text_encoder/` subdirectory. We rely on
// swift-transformers to load + run BPE; this wrapper only handles
// directory routing, max-length truncation, and the awkward fact that
// `AutoTokenizer.from(modelFolder:)` is async (callers want to invoke
// it inside a synchronous loader path).
//
// Reference: /tmp/mflux-ref/src/mflux/models/z_image/model/z_image_text_encoder/

import Foundation
@preconcurrency import Tokenizers

public struct ZImageTokenizer: Sendable {
    public let inner: any Tokenizers.Tokenizer
    public let maxLen: Int

    /// Default max prompt length. mflux uses 512 for Z-Image Turbo's
    /// caption encoder; longer prompts get truncated to keep the DiT
    /// caption-feature length bounded.
    public static let defaultMaxLen = 512

    public init(inner: any Tokenizers.Tokenizer, maxLen: Int = ZImageTokenizer.defaultMaxLen) {
        self.inner = inner
        self.maxLen = maxLen
    }

    /// Async loader. Reads `text_encoder/` from `modelPath` and hydrates
    /// a Qwen2 tokenizer. The text_encoder subdir is the canonical Z-Image
    /// tokenizer location (it ships next to the encoder safetensors,
    /// not under a separate `tokenizer/` subdir).
    public static func load(modelPath: URL, maxLen: Int = ZImageTokenizer.defaultMaxLen) async throws -> ZImageTokenizer {
        let teDir = modelPath.appendingPathComponent("text_encoder")
        let dirToUse: URL
        if FileManager.default.fileExists(atPath:
            teDir.appendingPathComponent("tokenizer.json").path)
        {
            dirToUse = teDir
        } else if FileManager.default.fileExists(atPath:
            modelPath.appendingPathComponent("tokenizer.json").path)
        {
            // Fallback for snapshots that hoist tokenizer.json to the
            // top level (some HF mirrors do this).
            dirToUse = modelPath
        } else {
            throw FluxError.weightsNotFound(teDir)
        }
        let upstream = try await AutoTokenizer.from(modelFolder: dirToUse)
        return ZImageTokenizer(inner: upstream, maxLen: maxLen)
    }

    /// Encode `prompt` to token ids (Int). Truncates at `maxLen` to keep
    /// the text-encoder forward bounded.
    public func encode(_ prompt: String) -> [Int] {
        var ids = inner.encode(text: prompt, addSpecialTokens: true)
        if ids.count > maxLen {
            ids = Array(ids.prefix(maxLen))
        }
        return ids
    }
}
