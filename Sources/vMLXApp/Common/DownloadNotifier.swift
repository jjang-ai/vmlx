// SPDX-License-Identifier: Apache-2.0
//
// macOS user-notification facade for download lifecycle events.
//
// Before this landed, a completed model download was silent unless
// the user happened to be staring at the Downloads window — the
// audit's #5 UX gap. Now `DownloadNotifier.notifyCompleted(job)`
// posts a `UNUserNotificationCenter` notification so the user sees
// the completion even if the app is in the background or minimized.
//
// Authorization is requested lazily on the first call. If the user
// denies, subsequent calls no-op silently (no fallback alert, no
// repeated permission prompts).

import Foundation
import vMLXEngine
#if canImport(UserNotifications)
import UserNotifications
#endif

enum DownloadNotifier {

    /// Post a "Download complete" notification for `job`. Safe to
    /// call from any queue — dispatches UNUserNotificationCenter
    /// work via its own internal threads. Silent no-op on platforms
    /// without UserNotifications (Linux builds of vMLX-core).
    static func notifyCompleted(_ job: DownloadManager.Job) {
        #if canImport(UserNotifications)
        let center = UNUserNotificationCenter.current()
        requestAuthorizationIfNeeded(center: center) { granted in
            guard granted else { return }
            let content = UNMutableNotificationContent()
            content.title = "Download complete"
            // DownloadManager.Job doesn't expose a model-friendly
            // "display name" so prefer the inferred alias if present,
            // else fall back to the job id.
            let label = job.displayName.isEmpty
                ? job.id.uuidString.prefix(8).description
                : job.displayName
            content.body = "\(label) is ready to load."
            content.sound = .default
            let req = UNNotificationRequest(
                identifier: "vmlx.download.\(job.id)",
                content: content,
                trigger: nil)
            center.add(req) { _ in /* fire-and-forget */ }
        }
        #endif
    }

    /// Post a "Download failed" notification. Surfaces errors that
    /// would otherwise be hidden in the DownloadsWindow list.
    static func notifyFailed(_ job: DownloadManager.Job, message: String) {
        #if canImport(UserNotifications)
        let center = UNUserNotificationCenter.current()
        requestAuthorizationIfNeeded(center: center) { granted in
            guard granted else { return }
            let content = UNMutableNotificationContent()
            content.title = "Download failed"
            content.body = "\(job.displayName.isEmpty ? "Model" : job.displayName): \(message)"
            content.sound = .default
            let req = UNNotificationRequest(
                identifier: "vmlx.download.\(job.id).failed",
                content: content,
                trigger: nil)
            center.add(req) { _ in }
        }
        #endif
    }

    #if canImport(UserNotifications)
    /// Ask for permission once, cache the result, and invoke the
    /// completion handler with the authorized state. We don't block
    /// on the request because that would stall the download-event
    /// queue; subsequent calls while the request is in-flight will
    /// see `authorized == false` and skip. Next notification retries.
    private static let authLock = NSLock()
    nonisolated(unsafe) private static var authDecided: Bool? = nil

    private static func requestAuthorizationIfNeeded(
        center: UNUserNotificationCenter,
        completion: @escaping @Sendable (Bool) -> Void
    ) {
        authLock.lock()
        let cached = authDecided
        authLock.unlock()
        if let cached {
            completion(cached)
            return
        }
        center.requestAuthorization(options: [.alert, .sound]) { granted, _ in
            authLock.lock()
            authDecided = granted
            authLock.unlock()
            completion(granted)
        }
    }
    #endif
}
