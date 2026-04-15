import Foundation

/// A protocol for loading tokenizers from local directories.
public protocol TokenizerLoader: Sendable {
    func load(from directory: URL) async throws -> any Tokenizer
}
