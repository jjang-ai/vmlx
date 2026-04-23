// LanguagePicker.swift
// vMLXApp — §349 i18n infrastructure
//
// User-facing control for switching UI language. Two variants:
//
//   * `LanguagePicker()` — full card with header label and help text,
//     intended for the Settings → General panel.
//
//   * `LanguagePickerCompact` — single-row Menu picker sized for the
//     tray popover footer, next to the Appearance menu.
//
// Both variants persist through `@AppStorage(AppLocalePreference.userDefault)`
// (UserDefaults). Per §349 scope, the translation layer is purely
// visual — nothing here touches the engine-side settings store. Views
// that read `@Environment(\.appLocale)` re-evaluate automatically when
// the stored string changes, because the root scenes in vMLXApp.body
// inject the locale derived from the same @AppStorage key.

import SwiftUI

public struct LanguagePicker: View {
    @AppStorage(AppLocalePreference.userDefault)
    private var uiLanguageRaw: String = AppLocale.fromSystem().rawValue

    public init() {}

    private var selection: Binding<AppLocale> {
        Binding(
            get: { AppLocale(rawValue: uiLanguageRaw) ?? .en },
            set: { newValue in
                uiLanguageRaw = newValue.rawValue
            }
        )
    }

    public var body: some View {
        let locale = AppLocale(rawValue: uiLanguageRaw) ?? .en
        VStack(alignment: .leading, spacing: 6) {
            Text(L10n.Settings.uiLanguage.render(locale))
                .font(.headline)
            Picker("", selection: selection) {
                ForEach(AppLocale.allCases) { loc in
                    Text(loc.displayName).tag(loc)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: 260, alignment: .leading)

            Text(L10n.Settings.uiLanguageHelp.render(locale))
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Compact variant (tray footer)

/// Single-row Menu picker sized for the tray popover footer. Renders
/// the current language's endonym as the label so users see their
/// active selection without opening the menu. Same UserDefaults write
/// path as the full variant.
public struct LanguagePickerCompact: View {
    @AppStorage(AppLocalePreference.userDefault)
    private var uiLanguageRaw: String = AppLocale.fromSystem().rawValue

    public init() {}

    public var body: some View {
        let current = AppLocale(rawValue: uiLanguageRaw) ?? .en
        Menu {
            ForEach(AppLocale.allCases) { loc in
                Button(action: { apply(loc) }) {
                    HStack {
                        Text(loc.displayName)
                        if loc == current {
                            Spacer()
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "globe")
                    .font(.system(size: 11))
                Text(current.displayName)
                    .font(.system(size: 11, weight: .medium))
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help(L10n.Common.language.render(current))
    }

    private func apply(_ loc: AppLocale) {
        uiLanguageRaw = loc.rawValue
    }
}
