# i18n Coverage — §349

**Scope:** purely visual. Lives under `Sources/vMLXApp/Locale/` only.
Engine targets (`vMLXEngine`, `vMLXLMCommon`, `vMLXLLM`, `vMLXVLM`,
`vMLXEmbedders`, `vMLXServer`, `vMLXFlux*`, `Cmlx`, `Vendor/`) are
**off-limits**. API responses, server logs, CLI stderr, and engine
diagnostics stay in English regardless of the user's UI language
setting — that is deliberate. Translating engine surfaces is out of
scope and would complicate issue triage.

## Supported locales

Four, forever:

| Code      | Language              |
| --------- | --------------------- |
| `en`      | English               |
| `ja`      | Japanese (日本語)     |
| `ko`      | Korean (한국어)       |
| `zh-Hans` | Simplified Chinese (简体中文) |

A fifth locale requires coordinated edits in two files:

1. `Sources/vMLXApp/Locale/AppLocale.swift` — add the enum case +
   `displayName` + `fromSystem()` mapping.
2. `Sources/vMLXApp/Locale/L10nEntry.swift` — add the new label to the
   initializer. **Every call site in `Strings.swift` becomes a
   compile error until the new label is filled in.** That is the
   enforcement mechanism — do not circumvent it with optional
   parameters or default values.

## Persistence

UserDefaults key: `vmlx.uiLanguage` (see `AppLocalePreference.userDefault`).

- SwiftUI views bind via `@AppStorage(AppLocalePreference.userDefault)`.
- Non-SwiftUI code reads/writes via `AppLocalePreference.current`.
- **No `SettingsStore` / `GlobalSettings` coupling.** The engine side
  does not persist, observe, or advertise the UI language. This is
  deliberate — the translation layer is visual state owned entirely
  by the app process.

Environment injection: `vMLXApp.body` attaches
`.environment(\.appLocale, uiLocale)` to every scene root. Any child
view can read the current locale with
`@Environment(\.appLocale) private var appLocale`.

## Compile-time enforcement

`L10nEntry`'s initializer takes four labeled arguments — `en`, `ja`,
`ko`, `zh` — with no defaults and no optional parameters. Every
catalog entry in `Strings.swift` must fill in all four:

```swift
static let chatSend = L10nEntry(
    en: "Send",
    ja: "送信",
    ko: "보내기",
    zh: "发送"
)
```

Omitting any label produces a hard compile error. Reviewers should
reject any PR that:

- adds an overload of `L10nEntry.init(...)` with defaults / optionals,
- wraps catalog entries in a builder that accepts partial translations,
- writes `zh: ""` or copies the English string as a placeholder.

For placeholders, use the English string explicitly in all four slots
and tag the entry with `// TODO: translate ja/ko/zh` — that makes the
gap auditable with `git grep 'TODO: translate'`.

## Adoption plan (incremental)

The seed catalog in `Strings.swift` covers:

- top-level mode names (Chat / Server / Image / Terminal / API),
- menu-bar commands (`New Chat`, `Reopen Last Closed Chat`,
  `View`, `Command Palette…`, `Downloads`, `Undo`),
- chat composer (`Send`, `Stop`, placeholder, attach, sessions,
  settings, clear-all),
- common dialog actions (`OK`, `Cancel`, `Save`, `Delete`, `Close`,
  `Copy`, `Copied!`, `Search`, `Settings`, `Language`),
- server control (`Start Server`, `Stop Server`, `Running`, `Stopped`),
- Settings → Language card itself.

**This is not a full sweep.** `grep -rn 'Text("' Sources/vMLXApp/`
reports ~273 raw literals and `Button("` another ~104 as of the
§349 seed pass. Conversions happen opportunistically as views are
touched for other reasons, not in a big-bang rewrite. Rules for
incremental adoption:

1. When editing a vMLXApp view for any reason, convert its raw
   English strings to `L10n.*` entries on the way through.
2. New user-visible strings added to `Sources/vMLXApp/` **must** go
   through `L10n.*` — never a raw `Text("…")` literal. PR reviewers
   should grep for `Text("` in the diff and request conversion.
3. Strings that are genuinely not user-facing (debug / assert
   messages, internal log text emitted via `print`) stay English and
   do not go through the catalog.

## Review checklist

When reviewing a PR that touches `Sources/vMLXApp/`:

- [ ] Any new `Text("...")`, `Button("...")`, `Label("...")`,
  `.navigationTitle("...")`, or `NavigationLink("...")` with an
  English string literal has a matching `L10nEntry` in `Strings.swift`.
- [ ] New `L10nEntry` values have all four locales (`en`/`ja`/`ko`/`zh`)
  filled in with real translations, not stub copies.
- [ ] No edits under any engine target (`Sources/vMLXEngine/`,
  `Sources/vMLXLMCommon/`, `Sources/vMLXLLM/`, `Sources/vMLXVLM/`,
  `Sources/vMLXEmbedders/`, `Sources/vMLXServer/`, `Sources/vMLXFlux*/`,
  `Sources/Cmlx/`, `Vendor/`). If the PR claims i18n work touched
  those, something is wrong with the architecture.
- [ ] No new persistence surfaces for the UI language beyond
  `AppLocalePreference.userDefault` — specifically, `uiLanguage` must
  NOT appear in `GlobalSettings` or any `SettingsStore` surface.

## Non-goals

- Right-to-left (RTL) locales. Not shipping; would require layout
  audit across every vMLXApp view.
- User-supplied catalog dictionaries. Possible future work but not
  in §349 scope.
- Translating engine-side surfaces (server HTTP errors, CLI stderr,
  SSE event payloads, log messages). Out of scope by design — see
  "Scope" at top.
