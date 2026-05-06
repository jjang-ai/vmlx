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
    // B2 §261: waiters carry an id so cancellation can remove them by
    // identity. /health reads `waiters.count`.
    //
    // Iter 144 — switched to `CheckedContinuation<Void, Error>` so
    // cancellation can resume with `CancellationError` instead of a
    // bare resume. The bare-resume path was a real bug:
    //
    //   1. Task A holds (held=true).
    //   2. Task B queued in waiters.
    //   3. Task B parent cancels → onCancel → dropWaiter removes +
    //      resumes B's continuation with success.
    //   4. Task B returns from acquire() thinking it owns the lock.
    //   5. Task.checkCancellation throws somewhere later in the body.
    //   6. Cleanup catches CancellationError, calls release().
    //   7. release() sees empty waiters → sets held=false.
    //      ← A still actually holds. Lock is corrupt.
    //   8. Next acquire() sees held=false, runs CONCURRENT with A.
    //      Metal command-queue race → process abort.
    //
    // The fix: make acquire() throwing. dropWaiter resumes with
    // `CancellationError`, which propagates out of `try await acquire()`.
    // Caller catches BEFORE calling release(), so the lock state is
    // never wrongly mutated by a task that never owned it.
    private var waiters: [(UUID, CheckedContinuation<Void, Error>)] = []

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

    public func acquire() async throws {
        if !held {
            held = true
            return
        }
        // B2 §261 + iter 144: cancel-aware wait. The continuation now
        // throws `CancellationError` on cancel rather than resuming
        // with success — see the type-level comment on `waiters` above
        // for the bug this prevents.
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                waiters.append((id, cont))
            }
        } onCancel: { [weak self] in
            Task { [weak self] in
                await self?.dropWaiter(id: id)
            }
        }
    }

    /// B2 §261 + iter 144: remove a cancelled waiter by id. If the
    /// waiter had already been dequeued and resumed (won the baton),
    /// this is a no-op. Otherwise resume with `CancellationError` so
    /// the caller's `try await acquire()` throws and they don't go on
    /// to do work or call `release()` on a lock they never owned.
    private func dropWaiter(id: UUID) {
        guard let idx = waiters.firstIndex(where: { $0.0 == id }) else { return }
        let entry = waiters.remove(at: idx)
        entry.1.resume(throwing: CancellationError())
    }

    public func release() {
        if let next = waiters.first {
            waiters.removeFirst()
            // `held` stays true — we just hand the baton to the next waiter.
            next.1.resume()
        } else {
            held = false
        }
    }

    /// Helper — acquires, runs the body, releases on both success and
    /// error paths. Prefer this over manual acquire/release so a thrown
    /// error doesn't leak the lock.
    ///
    /// Iter 144 — `acquire()` now throws on cancellation, so this
    /// helper does too. If acquire throws, body never runs and release
    /// isn't called (we never owned the lock). If body throws, we
    /// release and rethrow.
    public func withLock<T>(_ body: () async throws -> T) async throws -> T {
        try await acquire()
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
