// AppLocale.swift
// vMLXApp — §349 i18n infrastructure
//
// Scope: purely visual translation layer. Lives under Sources/vMLXApp/
// only — the engine (vMLXEngine / vMLXLMCommon / vMLXLLM / vMLXVLM /
// vMLXServer) is intentionally untouched. Engine logs, API responses,
// and server-side strings remain English.
//
// Four locales, forever:
//   - en      (English)
//   - ja      (Japanese)
//   - ko      (Korean)
//   - zh-Hans (Simplified Chinese)
//
// If product ever wants a fifth language, AppLocale + L10nEntry both
// need to grow in lockstep. That's by design: the compiler will yell.

import Foundation

/// Supported UI languages for the vMLX macOS app. The engine side is
/// never localized — only user-visible strings in `Sources/vMLXApp/`.
public enum AppLocale: String, CaseIterable, Identifiable, Sendable {
    case en      = "en"
    case ja      = "ja"
    case ko      = "ko"
    case zhHans  = "zh-Hans"

    public var id: String { rawValue }

    /// Human-readable label, rendered IN the language itself so the
    /// Settings picker is usable regardless of current setting.
    public var displayName: String {
        switch self {
        case .en:     return "English"
        case .ja:     return "日本語"
        case .ko:     return "한국어"
        case .zhHans: return "简体中文"
        }
    }

    /// BCP-47 tag for SwiftUI `.environment(\.locale, ...)` injection.
    public var bcp47: String { rawValue }

    /// Best-effort mapping from the host system locale at first launch.
    /// Matches on language code; falls back to English for unsupported
    /// systems (no silent Japanese-for-everyone surprises).
    public static func fromSystem() -> AppLocale {
        let lang = Foundation.Locale.current.language.languageCode?.identifier ?? "en"
        let script = Foundation.Locale.current.language.script?.identifier
        switch lang {
        case "ja": return .ja
        case "ko": return .ko
        case "zh":
            // Treat anything that isn't explicitly Traditional as Simplified.
            // zh-Hant / zh-TW / zh-HK fall back to English here because we
            // don't ship a Traditional catalog — users can switch manually.
            if script == "Hant" { return .en }
            return .zhHans
        default:   return .en
        }
    }
}
