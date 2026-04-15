// ModelLibraryWatcher.swift
//
// FSEvents-backed filesystem watcher that rescans `ModelLibrary` whenever a
// watched directory (HF cache + user-added dirs) changes. Replaces the old
// pull-only refresh so the Server screen model picker updates live.
//
// The `FSEventStream` C API is imported via CoreServices. The watcher owns a
// single stream for the union of watched roots and debounces bursty events
// with a 2-second timer before calling `ModelLibrary.scan(force: true)`.
//
// Start from `ModelLibrary.init` (or the first `scan()` call) and stop in the
// watcher's deinit so the stream is torn down on engine shutdown.

import Foundation
#if canImport(CoreServices)
import CoreServices
#endif

public final class ModelLibraryWatcher: @unchecked Sendable {

    /// Callback invoked (on a background queue) after the debounce window
    /// elapses. Typically wired to `ModelLibrary.scan(force: true)`.
    public typealias Rescan = @Sendable () async -> Void

    private let rescan: Rescan
    private let debounceInterval: TimeInterval
    private let queue: DispatchQueue

    #if canImport(CoreServices)
    private var stream: FSEventStreamRef? = nil
    #endif

    private var paths: [String] = []
    private var pending: DispatchWorkItem? = nil

    public init(
        paths: [URL],
        debounceInterval: TimeInterval = 2.0,
        queue: DispatchQueue = DispatchQueue(label: "ai.jangq.vmlx.modellib-watcher"),
        rescan: @escaping Rescan
    ) {
        self.rescan = rescan
        self.debounceInterval = debounceInterval
        self.queue = queue
        self.paths = paths.map { $0.path }
        start()
    }

    deinit {
        stop()
    }

    // MARK: - Lifecycle

    private func start() {
        #if canImport(CoreServices)
        guard !paths.isEmpty else { return }
        let cfPaths = paths as CFArray
        var context = FSEventStreamContext(
            version: 0,
            info: Unmanaged.passUnretained(self).toOpaque(),
            retain: nil,
            release: nil,
            copyDescription: nil
        )
        let flags: FSEventStreamCreateFlags = UInt32(
            kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagNoDefer
        )
        let callback: FSEventStreamCallback = { _, info, numEvents, _, _, _ in
            guard let info else { return }
            let me = Unmanaged<ModelLibraryWatcher>.fromOpaque(info).takeUnretainedValue()
            me.eventsReceived(count: Int(numEvents))
        }
        guard let stream = FSEventStreamCreate(
            kCFAllocatorDefault,
            callback,
            &context,
            cfPaths,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            1.0,  // latency — coarser of (this, NoDefer) wins
            flags
        ) else {
            return
        }
        self.stream = stream
        FSEventStreamSetDispatchQueue(stream, queue)
        FSEventStreamStart(stream)
        #endif
    }

    private func stop() {
        #if canImport(CoreServices)
        if let stream = self.stream {
            FSEventStreamStop(stream)
            FSEventStreamInvalidate(stream)
            FSEventStreamRelease(stream)
            self.stream = nil
        }
        #endif
        pending?.cancel()
        pending = nil
    }

    // MARK: - Debounce

    private func eventsReceived(count: Int) {
        _ = count
        pending?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            let cb = self.rescan
            Task.detached { await cb() }
        }
        pending = work
        queue.asyncAfter(deadline: .now() + debounceInterval, execute: work)
    }

    /// Test hook — trigger the debounce path manually without waiting for
    /// FSEvents to fire. Useful for deterministic unit tests.
    public func _simulateEvent() {
        eventsReceived(count: 1)
    }
}
