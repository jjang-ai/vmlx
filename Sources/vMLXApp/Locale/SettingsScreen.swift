// SPDX-License-Identifier: Apache-2.0
// §349 — Settings window.
//
// Lives under `Sources/vMLXApp/Locale/` because the only thing it
// hosts today is the language picker + the appearance mode menu. As
// product grows this will get more tabs (Cache, Network, Shortcuts…)
// and can migrate to its own folder then.
//
// Reachable three ways:
//   * macOS standard Settings scene (Cmd-,).
//   * Command menu entry (routed from the app's `.commands` block).
//   * Tray popover footer "Open Settings…" link (future — for now
//     the tray has the LanguagePickerCompact inline).

import SwiftUI
import vMLXTheme

/// Top-level Settings view. Single-tab today (General) — grows a
/// `TabView` wrapper if we add Cache / Network / Shortcuts sections
/// later.
public struct SettingsScreen: View {
    @Environment(\.appLocale) private var appLocale: AppLocale
    @AppStorage("vmlx.appearance") private var appearanceRaw: String =
        AppearanceMode.dark.rawValue

    public init() {}

    public var body: some View {
        Form {
            Section(header: Text(L10n.Settings.uiLanguage.render(appLocale))
                .font(.headline)) {
                LanguagePicker()
            }

            Section(header: Text(L10n.Settings.appearance.render(appLocale))
                .font(.headline)) {
                Picker("", selection: $appearanceRaw) {
                    ForEach(AppearanceMode.allCases) { mode in
                        Text(mode.label).tag(mode.rawValue)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(maxWidth: 320, alignment: .leading)
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 520, minHeight: 320)
        .background(Theme.Colors.background)
    }
}
