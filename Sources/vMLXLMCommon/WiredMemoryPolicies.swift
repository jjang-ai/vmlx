// Copyright © 2025 Apple Inc.

import Foundation
import MLX

private func recommendedWorkingSetBytes() -> Int? {
    #if canImport(Metal)
        GPU.maxRecommendedWorkingSetBytes()
    #else
        nil
    #endif
}

/// Sum policy: `baseline + sum(activeSizes)`, optionally capped.
///
/// This is the most common policy for inference workloads. Each ticket adds to
/// the wired limit, making the total proportional to concurrent demand.
///
/// If `cap` is nil, the policy clamps to `GPU.maxRecommendedWorkingSetBytes()`
/// when available. If a cap is provided, both limit computation and admission
/// respect it.
///
/// ### Example
/// ```swift
/// let policy = WiredSumPolicy(cap: 12 * 1024 * 1024 * 1024)
/// let ticket = policy.ticket(size: kvBytes, kind: .active)
/// try await ticket.withWiredLimit {
///     // run inference
/// }
/// ```
public struct WiredSumPolicy: WiredMemoryPolicy, Hashable, Sendable {
    /// Optional absolute cap (bytes) for the computed limit.
    public let cap: Int?

    /// Creates a sum policy with an optional cap in bytes.
    public init(cap: Int? = nil) {
        self.cap = cap
    }

    /// Computes the desired limit for the current active set.
    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        let sum = activeSizes.reduce(0, +)
        return clamp(baseline + sum)
    }

    /// Admission is denied if the projected limit would exceed the cap.
    public func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
        let projected = baseline + activeSizes.reduce(0, +) + max(0, newSize)
        return clamp(projected) == projected
    }

    private func clamp(_ value: Int) -> Int {
        if let cap {
            return min(value, max(0, cap))
        }
        if let maxBytes = recommendedWorkingSetBytes() {
            return min(value, maxBytes)
        }
        return value
    }
}

/// Max policy: `max(baseline, max(activeSizes))`.
///
/// This policy tracks the single largest active ticket. It is useful when you
/// want the limit to scale with the largest in-flight request rather than the
/// sum of concurrent work.
///
/// ### Example
/// ```swift
/// let policy = WiredMaxPolicy()
/// let ticket = policy.ticket(size: kvBytes, kind: .active)
/// try await ticket.withWiredLimit {
///     // run inference
/// }
/// ```
public struct WiredMaxPolicy: WiredMemoryPolicy, Hashable, Sendable {
    /// Creates a max policy.
    public init() {}

    /// Computes the desired limit for the current active set.
    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        let maxSize = activeSizes.max() ?? 0
        return max(baseline, maxSize)
    }
}

/// Fixed policy: limit is constant while any ticket is active.
///
/// This policy ignores the active sizes and applies a fixed limit any time at
/// least one ticket is active. It is useful when you want a deterministic cap.
///
/// ### Example
/// ```swift
/// let policy = WiredFixedPolicy(limit: 8 * 1024 * 1024 * 1024)
/// let ticket = policy.ticket(size: kvBytes, kind: .active)
/// try await ticket.withWiredLimit {
///     // run inference
/// }
/// ```
public struct WiredFixedPolicy: WiredMemoryPolicy, Hashable, Sendable {
    /// Fixed limit in bytes to apply while any ticket is active.
    public let bytes: Int

    /// Creates a fixed policy with the given limit in bytes.
    public init(limit: Int) {
        self.bytes = limit
    }

    /// Computes the desired limit for the current active set.
    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        bytes
    }
}

/// Budget policy: `baseline + baseBytes + sum(activeSizes)`, optionally capped.
///
/// This policy is useful when you want to bake a learned or precomputed budget
/// (for example, weights + workspace) into the limit calculation while still
/// accounting for active tickets.
///
/// - Note: Pass a stable `id` if you intend to recreate the policy and keep it
///   grouped with existing tickets.
///
/// ### Example
/// ```swift
/// let base = weightsBytes + workspaceBytes
/// let policy = WiredBudgetPolicy(baseBytes: base)
/// let ticket = policy.ticket(size: kvBytes, kind: .active)
/// try await ticket.withWiredLimit {
///     // run inference
/// }
/// ```
public struct WiredBudgetPolicy: WiredMemoryPolicy, Hashable, Sendable {
    /// Stable policy identifier used for grouping.
    public let identifier: UUID
    /// Base budget in bytes (e.g. weights + workspace).
    public let baseBytes: Int
    /// Optional absolute cap (bytes) for the computed limit.
    public let cap: Int?

    /// Creates a budget policy with an optional cap and stable id.
    public init(
        baseBytes: Int,
        cap: Int? = nil,
        id: UUID = UUID()
    ) {
        self.baseBytes = max(0, baseBytes)
        self.cap = cap
        self.identifier = id
    }

    public static func == (lhs: WiredBudgetPolicy, rhs: WiredBudgetPolicy) -> Bool {
        lhs.identifier == rhs.identifier
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(identifier)
    }

    /// Computes the desired limit for the current active set.
    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        let sum = activeSizes.reduce(0, +)
        return clamp(baseline + baseBytes + sum)
    }

    /// Admission is denied if the projected limit would exceed the cap.
    public func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
        let projected = baseline + baseBytes + activeSizes.reduce(0, +) + max(0, newSize)
        return clamp(projected) == projected
    }

    private func clamp(_ value: Int) -> Int {
        if let cap {
            return min(value, max(0, cap))
        }
        if let maxBytes = recommendedWorkingSetBytes() {
            return min(value, maxBytes)
        }
        return value
    }
}
