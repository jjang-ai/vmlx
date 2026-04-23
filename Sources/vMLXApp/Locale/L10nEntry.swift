// L10nEntry.swift
// vMLXApp — §349 i18n infrastructure
//
// Four-locale translation record. The initializer takes ALL four
// locales as explicit labeled arguments — so adding a new string to
// Strings.swift requires filling in en/ja/ko/zh. If any label is
// omitted, Swift refuses to compile the call site. That is the
// whole point of this type.
//
// Do NOT add optional parameters, default values, or a variadic
// initializer. Those are loopholes around the compile-time check.

import Foundation

/// A single user-visible string, with translations for every locale
/// vMLX ships. The four initializer labels are mandatory — that is
/// the compile-time enforcement that §349 depends on.
public struct L10nEntry: Sendable, Equatable {
    public let en: String
    public let ja: String
    public let ko: String
    public let zh: String   // zh-Hans (Simplified)

    /// All four labels are required — there are no defaults and no
    /// optional parameters. Adding a `public init(...)` overload that
    /// weakens this contract defeats §349 and must be rejected at
    /// code review.
    public init(en: String, ja: String, ko: String, zh: String) {
        self.en = en
        self.ja = ja
        self.ko = ko
        self.zh = zh
    }

    /// Resolve to the rendered string for a given locale.
    public func render(_ locale: AppLocale) -> String {
        switch locale {
        case .en:     return en
        case .ja:     return ja
        case .ko:     return ko
        case .zhHans: return zh
        }
    }

    /// Convenience for `String(format:)` call sites. Matches the shape
    /// of NSLocalizedString-style format strings — pass positional
    /// arguments to interpolate `%@`, `%d`, `%lld`, etc.
    public func render(_ locale: AppLocale, _ args: CVarArg...) -> String {
        String(format: render(locale), arguments: args)
    }
}
