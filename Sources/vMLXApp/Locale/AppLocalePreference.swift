// AppLocalePreference.swift
// vMLXApp — §349 i18n infrastructure
//
// Persistence for the user's UI language pick. Lives in vMLXApp so the
// engine-side SettingsStore stays untouched per §349 scope constraints.
// Storage = UserDefaults via a plain key ("vmlx.uiLanguage"), readable
// by `@AppStorage` in views OR by this helper from non-View code.

import Foundation
import SwiftUI

/// UserDefaults-backed store for the app's UI language. Exposes a
/// plain `current` accessor for non-SwiftUI code (AppDelegate, command
/// handlers) while SwiftUI views should prefer `@AppStorage(Key.userDefault)`
/// so they re-render on change.
public enum AppLocalePreference {
    /// The `@AppStorage` key SwiftUI views bind to. Keep the raw string
    /// literal — hardcoding on purpose so there's one source of truth.
    public static let userDefault = "vmlx.uiLanguage"

    /// Current selection, resolved with fallback to system locale then
    /// English. Safe to call from any thread.
    public static var current: AppLocale {
        get {
            if let raw = UserDefaults.standard.string(forKey: userDefault),
               let loc = AppLocale(rawValue: raw) {
                return loc
            }
            return AppLocale.fromSystem()
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: userDefault)
        }
    }

    /// One-time seed on first launch. If the user has never picked a
    /// language, write the system-detected default so `@AppStorage`
    /// bindings observe a stable value. Called from `RootView.onAppear`.
    public static func seedIfAbsent() {
        if UserDefaults.standard.string(forKey: userDefault) == nil {
            UserDefaults.standard.set(AppLocale.fromSystem().rawValue, forKey: userDefault)
        }
    }
}

// MARK: - SwiftUI Environment wiring

private struct AppLocaleKey: EnvironmentKey {
    static let defaultValue: AppLocale = .en
}

public extension EnvironmentValues {
    /// Current UI language, injected near the root so any view can read
    /// `@Environment(\.appLocale)` and resolve L10nEntry values.
    var appLocale: AppLocale {
        get { self[AppLocaleKey.self] }
        set { self[AppLocaleKey.self] = newValue }
    }
}
