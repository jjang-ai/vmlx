// L10nText.swift
// vMLXApp — §349 i18n infrastructure
//
// Sugar view for the common "render an L10nEntry with the current
// environment locale" pattern. At call sites, prefer:
//
//     L10nText(L10n.Chat.send)
//
// over:
//
//     Text(L10n.Chat.send.render(appLocale))
//
// Both work. The sugar form is shorter and harder to misuse (can't
// accidentally pass the wrong locale variable). Either way,
// `@Environment(\.appLocale)` must be populated upstream — RootView
// does that in vMLXApp.body.

import SwiftUI

public struct L10nText: View {
    @Environment(\.appLocale) private var locale: AppLocale
    private let entry: L10nEntry

    public init(_ entry: L10nEntry) {
        self.entry = entry
    }

    public var body: some View {
        Text(entry.render(locale))
    }
}
