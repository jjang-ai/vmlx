// Copyright © 2026 vMLX. Whisper tokenizer adapter.
//
// mlx-community Whisper checkpoints ship a HuggingFace `tokenizer.json`
// (plus the legacy `vocab.json`/`merges.txt`). swift-transformers'
// `AutoTokenizer` knows how to load the former. This file wraps the
// tokenizer and exposes the Whisper-specific special token IDs the
// greedy decoder needs.

import Foundation
import Tokenizers

public struct WhisperSpecialTokens {
    public let sot: Int
    public let eot: Int
    public let noTimestamps: Int
    public let transcribe: Int
    public let translate: Int
    public let noSpeech: Int?
    public let timestampBegin: Int
    /// Map from BCP-47 language code (e.g. "en", "ja") to its token id.
    public let languageToToken: [String: Int]
}

public final class WhisperTokenizer {
    public let tokenizer: any Tokenizer
    public let special: WhisperSpecialTokens
    public let isMultilingual: Bool

    init(
        tokenizer: any Tokenizer,
        special: WhisperSpecialTokens,
        isMultilingual: Bool
    ) {
        self.tokenizer = tokenizer
        self.special = special
        self.isMultilingual = isMultilingual
    }

    /// Load a whisper tokenizer from a local directory (must contain
    /// `tokenizer.json`). Resolves all whisper special token IDs using
    /// the tokenizer's `convertTokenToId`.
    public static func load(from dir: URL, isMultilingual: Bool) async throws -> WhisperTokenizer {
        // swift-transformers' AutoTokenizer takes either a Hub repo or a
        // local URL. We use the local-directory loader.
        let tok = try await AutoTokenizer.from(modelFolder: dir)
        let special = try resolveSpecials(tokenizer: tok, isMultilingual: isMultilingual)
        return WhisperTokenizer(
            tokenizer: tok, special: special, isMultilingual: isMultilingual)
    }

    private static func resolveSpecials(
        tokenizer: any Tokenizer, isMultilingual: Bool
    ) throws -> WhisperSpecialTokens {
        func tid(_ t: String) -> Int? { tokenizer.convertTokenToId(t) }

        guard let sot = tid("<|startoftranscript|>"),
              let eot = tid("<|endoftext|>"),
              let noTs = tid("<|notimestamps|>"),
              let transcribe = tid("<|transcribe|>"),
              let translate = tid("<|translate|>")
        else {
            throw WhisperError.tokenizerMissingSpecials
        }
        let noSpeech = tid("<|nospeech|>") ?? tid("<|nocaptions|>")
        // <|0.00|> is the first timestamp token.
        guard let tsBegin = tid("<|0.00|>") else {
            throw WhisperError.tokenizerMissingSpecials
        }

        // Language tokens: <|en|>, <|ja|>, ... Present only in
        // multilingual checkpoints. Whisper v3 has 100 language codes.
        var langMap: [String: Int] = [:]
        if isMultilingual {
            for code in languageCodes {
                if let id = tid("<|\(code)|>") {
                    langMap[code] = id
                }
            }
        }

        return WhisperSpecialTokens(
            sot: sot, eot: eot, noTimestamps: noTs,
            transcribe: transcribe, translate: translate,
            noSpeech: noSpeech, timestampBegin: tsBegin,
            languageToToken: langMap)
    }

    /// Decode a sequence of token ids to text, stripping any special
    /// tokens (SOT, language, task, timestamps, EOT).
    public func decode(_ ids: [Int]) -> String {
        let filtered = ids.filter { $0 < special.timestampBegin && $0 != special.eot }
        return tokenizer.decode(tokens: filtered)
    }

    /// Build the initial decoder prompt. OpenAI's convention is
    /// `[sot, <lang>, <task>, <|notimestamps|>]`. English-only models
    /// drop the language + task tokens and start with just `[sot]`.
    public func initialPromptTokens(
        language: String?, task: String, withoutTimestamps: Bool
    ) -> [Int] {
        var tokens: [Int] = [special.sot]
        if isMultilingual {
            let lang = language ?? "en"
            if let id = special.languageToToken[lang] {
                tokens.append(id)
            }
            tokens.append(task == "translate" ? special.translate : special.transcribe)
        }
        if withoutTimestamps {
            tokens.append(special.noTimestamps)
        }
        return tokens
    }
}

public enum WhisperError: LocalizedError {
    case configurationMissing(String)
    case weightsMissing
    case tokenizerMissingSpecials
    case audioDecodeFailed(String)
    case modelNotLoaded

    public var errorDescription: String? {
        switch self {
        case .configurationMissing(let file):
            return "whisper: missing \(file)"
        case .weightsMissing:
            return "whisper: no .safetensors weights found in model directory"
        case .tokenizerMissingSpecials:
            return "whisper: tokenizer.json missing expected <|startoftranscript|> / <|transcribe|> / language tokens"
        case .audioDecodeFailed(let msg):
            return "whisper: \(msg)"
        case .modelNotLoaded:
            return "whisper: transcription requested but no whisper model is loaded"
        }
    }
}

/// The 99 languages recognized by Whisper (plus the "yue" addition in
/// v3). Used to look up `<|LANG|>` token IDs during decoder prompt
/// construction. List copied verbatim from OpenAI whisper's
/// `tokenizer.py`.
private let languageCodes: [String] = [
    "en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar","sv",
    "it","id","hi","fi","vi","he","uk","el","ms","cs","ro","da","hu","ta","no",
    "th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa","lv","bn","sr",
    "az","sl","kn","et","mk","br","eu","is","hy","ne","mn","bs","kk","sq","sw",
    "gl","mr","pa","si","km","sn","yo","so","af","oc","ka","be","tg","sd","gu",
    "am","yi","lo","uz","fo","ht","ps","tk","nn","mt","sa","lb","my","bo","tl",
    "mg","as","tt","haw","ln","ha","ba","jw","su","yue",
]
