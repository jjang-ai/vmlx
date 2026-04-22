import Foundation

/// H1 §271 — thermal throttle observability. Wraps
/// `ProcessInfo.processInfo.thermalState` in a Sendable snapshot so
/// SwiftUI tray + chat banners can surface the state without
/// reaching into ProcessInfo directly (which isn't MainActor-safe on
/// strict concurrency).
///
/// Apple's thermal notification (`.thermalStateDidChangeNotification`)
/// posts whenever the state transitions — the tray can subscribe to
/// refresh on every change rather than polling.
///
/// Levels (per `ProcessInfo.ThermalState`):
///   - `.nominal`  — no throttling
///   - `.fair`     — mild thermal pressure, system may throttle soon
///   - `.serious`  — active throttling; decode tok/s will drop
///   - `.critical` — emergency; app should shed load / stop inference
public enum ThermalMonitor {
    public enum Level: String, Sendable, Equatable {
        case nominal, fair, serious, critical
    }

    /// Snapshot of the current process thermal state. Safe to call
    /// from any concurrency context; the underlying ProcessInfo call
    /// is thread-safe.
    public static func currentLevel() -> Level {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  return .nominal
        case .fair:     return .fair
        case .serious:  return .serious
        case .critical: return .critical
        @unknown default: return .nominal
        }
    }

    /// NotificationCenter name that fires on state transition. Tray
    /// code can subscribe and call `currentLevel()` to refresh.
    public static let didChange = ProcessInfo.thermalStateDidChangeNotification
}
