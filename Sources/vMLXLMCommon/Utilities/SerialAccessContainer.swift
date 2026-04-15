// Copyright © 2025 Apple Inc.

/// A mutex providing exclusive access with `async` blocks.
///
/// This is used as a building block for ``SerialAccessContainer``.  Normal locks
/// do not work with `async` blocks and an `actor` does not guarantee exclusive access
/// for the duration of an `async` function.
private actor AsyncMutex {
    private var isLocked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    private func lock() async {
        if !isLocked {
            isLocked = true
            return
        }

        await withCheckedContinuation { cont in
            waiters.append(cont)
        }
    }

    private func unlock() {
        if let next = waiters.first {
            waiters.removeFirst()
            next.resume()
        } else {
            isLocked = false
        }
    }

    func withLock<T>(_ body: () async throws -> sending T) async rethrows -> sending T {
        await lock()
        defer { unlock() }
        return try await body()
    }
}

/// Provide serial exclusive access to state `<T>` to async callers.
///
/// Unlike an `actor`, this will guarantee exclusive access for the duration of the async
/// call.  This is important for things like `ModelContainer` that have to perform async
/// work but also need to prevent other callers for using _any_ of the internal state.
final class SerialAccessContainer<T>: @unchecked Sendable {

    private var value: T
    private let lock = AsyncMutex()

    public init(_ value: consuming T) {
        self.value = consume value
    }

    public func read<R>(_ body: @Sendable (T) async throws -> sending R) async rethrows -> sending R
    {
        try await lock.withLock {
            try await body(value)
        }
    }

    public func update<R>(_ body: @Sendable (inout T) async throws -> sending R) async rethrows
        -> sending R
    {
        try await lock.withLock {
            try await body(&value)
        }
    }

}

/// Internal box to wrap non-Sendable data to be transferred across
/// task boundaries.
///
/// For example, this can be used to pass non-Sendable context into a task.  Care
/// must be taken to prevent that data from being used in two threads at once --
/// this is `@unchecked Sendable`:
///
/// ```swift
/// public func generateTask(
///     input: consuming LMInput, context: consuming ModelContext,
///     iterator: consuming TokenIterator
/// ) -> (AsyncStream<Generation>, Task<Void, Never>) {
///     let context = SendableBox(context)
///     let iterator = SendableBox(iterator)
///
///     // Launch a Task to perform iteration asynchronously.
///     let task = Task {
///         let context = context.consume()
///         let iterator = iterator.consume()
/// ```
///
/// Note that the parameters are `consuming`.
///
/// Here is an example where ``UserInput`` is consumed and passed to an async block and ``LMInput``
/// is produced and handed back:
///
/// ```swift
/// public func prepare(input: consuming sending UserInput) async throws -> sending LMInput {
///     let input = SendableBox(input)
///     return try await context.read {
///         SendableBox(try await $0.processor.prepare(input: input.consume()))
///     }.consume()
/// }
/// ```
final class SendableBox<T>: @unchecked Sendable {
    private var value: T?

    init(_ value: consuming T) {
        self.value = consume value
    }

    consuming func consume() -> T {
        guard let value else {
            fatalError("value already consumed")
        }
        self.value = nil
        return value
    }
}
