// GenerationLock.swift — async FIFO mutex used by `Engine.streamReal` to
// serialize MLX/Metal work across concurrent HTTP requests.
//
// Why this exists: `performStreamingGeneration` walks the model via MLX
// kernels (prefill, decode step, eval). MLX drives a single Metal
// command queue under the hood, and `_MTLCommandBuffer
// setCurrentCommandEncoder:` fatal-asserts when two Swift Tasks try to
// encode simultaneously:
//
//   failed assertion _status < MTLCommandBufferStatusCommitted at
//   line 323 in -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]
//
// Reproduced with a 3-way concurrent burst against Qwen3-0.6B-8bit on
// vmlxctl (2026-04-17). The OpenAI / Ollama / Anthropic routes all call
// `Engine.stream(request:)`, which spawns a detached Task — so the
// Engine actor's natural serialization DOES NOT apply to the generation
// itself. Wrap the generation body in `acquire()` / `release()` so only
// one request runs the MLX graph at a time; subsequent requests queue
// FIFO and resume as soon as the prior one drains.
//
// This is a decode-level mutex, NOT a batched-continuous-batching path.
// A future continuous-batching engine can replace this with real
// request fusion, but for correctness+single-GPU safety this is what
// we need today.

import Foundation

public actor GenerationLock {

    private var held: Bool = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init() {}

    /// iter-85 §163: introspection for /health scheduler honesty. Lets
    /// the /health handler report queue depth so callers that send
    /// concurrent requests see that they actually queue FIFO rather
    /// than run in parallel. Without this, a client sending two
    /// requests assumes both are decoding simultaneously when the
    /// second is really stuck behind the first.
    public var isHeld: Bool { held }
    public var waitingCount: Int { waiters.count }
    /// 1 if held + N queued waiters. Mirrors the textbook
    /// "tasks in flight OR blocked" count.
    public var inflightOrQueued: Int { (held ? 1 : 0) + waiters.count }

    public func acquire() async {
        if !held {
            held = true
            return
        }
        await withCheckedContinuation { cont in
            waiters.append(cont)
        }
    }

    public func release() {
        if let next = waiters.first {
            waiters.removeFirst()
            // `held` stays true — we just hand the baton to the next waiter.
            next.resume()
        } else {
            held = false
        }
    }

    /// Helper — acquires, runs the body, releases on both success and
    /// error paths. Prefer this over manual acquire/release so a thrown
    /// error doesn't leak the lock.
    public func withLock<T>(_ body: () async throws -> T) async rethrows -> T {
        await acquire()
        do {
            let result = try await body()
            release()
            return result
        } catch {
            release()
            throw error
        }
    }
}
