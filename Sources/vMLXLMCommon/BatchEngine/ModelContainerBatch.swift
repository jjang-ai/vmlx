// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

// MARK: - ModelContainer Batch Engine Integration

extension ModelContainer {

    /// Create a ``BatchEngine`` from this model container.
    ///
    /// The engine uses the container's model context for inference. The container's
    /// serial access guarantee applies during engine creation but not during token
    /// generation — the engine manages its own concurrency via actor isolation.
    ///
    /// ## Example
    /// ```swift
    /// let modelContainer = try await ModelFactory.shared.loadContainer(
    ///     configuration: modelConfig)
    ///
    /// let engine = try await modelContainer.makeBatchEngine(maxBatchSize: 4)
    ///
    /// // Submit multiple requests concurrently
    /// async let stream1 = engine.generate(input: input1, parameters: params)
    /// async let stream2 = engine.generate(input: input2, parameters: params)
    ///
    /// // Both streams produce text simultaneously
    /// ```
    ///
    /// - Parameters:
    ///   - maxBatchSize: Maximum concurrent sequences. Defaults to 8.
    ///   - memoryPurgeInterval: Steps between GPU memory cache purges. Defaults to 256.
    /// - Returns: A ``BatchEngine`` instance ready to accept requests.
    public func makeBatchEngine(
        maxBatchSize: Int = 8,
        memoryPurgeInterval: Int = 256
    ) async -> BatchEngine {
        // Capture the cache coordinator outside the perform closure.
        // CacheCoordinator is Sendable so this is safe.
        let coordinator = self.cacheCoordinator

        // Build the engine inside the serial access container so ModelContext
        // (which is non-Sendable) doesn't cross isolation boundaries.
        return await perform { context in
            nonisolated(unsafe) let ctx = context
            return BatchEngine(
                context: ctx,
                maxBatchSize: maxBatchSize,
                memoryPurgeInterval: memoryPurgeInterval,
                cacheCoordinator: coordinator
            )
        }
    }
}
