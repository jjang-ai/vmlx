// Copyright © 2024 Apple Inc.

import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?

    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

extension Tokenizer {
    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokenIds: [Int]) -> String {
        decode(tokenIds: tokenIds, skipSpecialTokens: false)
    }

    public var eosTokenId: Int? {
        guard let eosToken else { return nil }
        return convertTokenToId(eosToken)
    }

    public var unknownTokenId: Int? {
        guard let unknownToken else { return nil }
        return convertTokenToId(unknownToken)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]]
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: tools, additionalContext: nil)
    }
}

public enum TokenizerError: LocalizedError {
    case missingChatTemplate

    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            "This tokenizer does not have a chat template."
        }
    }
}

public protocol StreamingDetokenizer: IteratorProtocol<String> {
    mutating func append(token: Int)
}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: any Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokenIds: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokenIds: segmentTokens)
        let new = newSegment.suffix(newSegment.count - segment.count)

        // Iter 143 — Laguna `????` UTF-8 corruption fix.
        //
        // The original guard `new.last == "\u{fffd}"` handles ONLY the
        // case where a multi-byte sequence is split such that the
        // incomplete bytes land at the very end of the suffix. But for
        // tokenizers that decode each token independently then string-
        // concatenate (sentencepiece-derived Mistral / Laguna), an
        // emoji can split such that the FFFD lands EARLIER in the
        // suffix while the trailing bytes of the next token decode as
        // ordinary text. Example:
        //
        //   Token A → "\u{fffd}"          (incomplete leading bytes)
        //   Token B → "\u{fffd}world"     (trailing bytes + plain text)
        //
        // In that case `new.last == "d"`, the original guard fires
        // false, and we yield "FFFD FFFD world" — the visible `???`
        // on Laguna live tests 2026-05-03.
        //
        // Fix: guard on `new.contains("\u{fffd}")` so any embedded
        // replacement char defers emission until the next token is
        // appended. Once subsequent tokens complete the UTF-8
        // sequence, the joined decode of segmentTokens drops the FFFDs
        // and the natural suffix delta produces clean text.
        if new.contains("\u{fffd}") {
            return nil
        }

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return String(new)
    }
}
