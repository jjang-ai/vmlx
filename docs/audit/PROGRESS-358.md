# §358 — Ship-blocker sweep progress

Running log. Updated per commit. Checkboxes reflect what's **shipped**,
what's **in flight**, and what's **deferred** with honest reason.

## Shipped this batch

### Port + gateway + CORS

- [x] **CORS + rateLimit restart-required warning.** `corsOrigins` field
      now shows yellow ⟳ caption ("Restart required"). `rateLimit` is in
      the Tray and caller-scope; caption planned for Tray UI next batch.
      Files: `Sources/vMLXApp/Server/SessionConfigForm.swift`
- [x] **Gateway auto-bump on port collision.** `GatewayActor.start`
      scans the next 32 free ports when the configured port is taken
      (Ollama on 8080 is the everyday case). Records a one-shot
      `lastAutoBumpNote` the UI drains into a banner ("moved from
      8080→8081"). `requestedPort` vs `port` split so UI can show both.
      Files: `Sources/vMLXApp/Server/GatewayActor.swift`,
      `Sources/vMLXApp/vMLXApp.swift`
- [x] **Persistent gateway-status pill in tray.** `AppState.gatewayStatus`
      carries `.disabled` / `.running(bound, requested)` / `.failed(port, msg)`
      and the tray popover renders a colored pill with tooltip.
      Replaces the 3-second flashBanner-only signal.
      Files: `Sources/vMLXApp/vMLXApp.swift`, `Sources/vMLXApp/Common/TrayItem.swift`
- [x] **Duplicate-model-name banner.** `GatewayActor.registerEngine`
      now detects when a display name is claimed by a second engine
      and queues a warning; `AppState.drainGatewayDuplicateWarnings`
      surfaces it after every session start. Tells user to rename one
      session's Model alias.
      Files: `Sources/vMLXApp/Server/GatewayActor.swift`, `Sources/vMLXApp/vMLXApp.swift`

### Model folders / library scan

- [x] **User-dir path canonicalization.** `ModelLibrary.addUserDir` now
      resolves symlinks + standardizes the URL before the DB write, so
      adding "/foo/bar", "/foo/bar/", and a symlink to /foo/bar can't
      produce three rows. Fixes duplicate picker entries after a
      Finder-picker + drag-drop combo on the same folder.
      File: `Sources/vMLXEngine/Library/ModelLibrary.swift`
- [x] **Discovery toast with counts.** After a drag-drop dir-add, the
      flashBanner reports "added 3 text + 2 image models" instead of
      the generic "rescanning model library". Counts entries under the
      added dir specifically, not the whole library.
      File: `Sources/vMLXApp/vMLXApp.swift`

### Image generation

- [x] **Honest scaffold-status error message.** Replaced the
      dead-README pointer (`swift/Sources/vMLXFluxKit/README.md`, which
      doesn't exist) with a message that names the one working model
      (Z-Image Turbo), admits the rest aren't ported, and points at
      `docs/audit/OPEN-FIX-LIST.md` which does exist.
      File: `Sources/vMLXApp/Image/ImageScreen.swift`

### Infrastructure

- [x] **swift-jinja confirmed vendored.** `Vendor/Jinja/` is in tree
      with all 9 files + Package.swift path-based dep. Our patches
      (§341 `in`/`not in`, §350 inline-if-without-else) are applied in
      `Vendor/Jinja/Sources/Runtime.swift`. Gemma 4 + DeepSeek V3 +
      GLM-5.1 templates run against this copy. Safe to edit further.

## Shipped §359 (this follow-on batch)

- [x] **Model alias end-to-end**. GatewayActor new
      `registerEngine(_:canonicalName:alias:)` registers the session
      under both canonical name AND user alias. Gateway `/v1/models`
      no longer leaks the full on-disk catalog — only loaded models
      appear. `resolve()` matches alias first, canonical second,
      suffix fallback third.
- [x] **modelAlias visible in SessionCard**. `Session.aliasSnapshot`
      captured at start-time (alias is load-time-only, matching
      semantics), rendered as a "as: <alias>" pill next to the model
      name. Cleared on stop.
- [x] **modelAlias restart-required caption** in SessionConfigForm —
      parity with CORS caption.
- [x] **removeUserDir canonicalization** — matches addUserDir path
      normalization so removeDir never leaves ghost rows in user_dirs.
- [x] **FSEvents watcher lifecycle confirmed** — `rebuildWatcher()`
      nils the watcher, ARC triggers deinit, FSEventStreamStop +
      FSEventStreamRelease fire cleanly. No leak.
- [x] **Permissions audit** — vMLX.entitlements disables App Sandbox
      (Developer ID DMG only). NSOpenPanel returns regular URLs, no
      security-scoped bookmarks needed. Hardened runtime with
      `allow-jit` + `disable-library-validation` for MLX Metal kernels.
      External-drive path survival is NOT guaranteed — "missing on
      disk" badge flips on ejection (already implemented).

## In flight — will land next batch

- [ ] **Live alias re-register without Stop/Start.** Current behavior
      requires session restart to take effect (documented via the
      restart-required caption). A `reregisterSessionWithGateway(id)`
      hook triggered by the "Served as" field's onCommit could make
      alias changes live — but only for the gateway registry; the
      underlying Engine.displayName shown in API responses is still
      captured at load time, so partial live-swap would confuse more
      than it helps.
- [ ] **Model alias (nickname) end-to-end.** `s.modelAlias` is stored
      in SessionSettings and the "Served as" field in SessionConfigForm
      writes to it, but the gateway registers engines under
      `ModelEntry.displayName` (the canonical HF name), not the alias.
      Result: user typing a custom alias has no effect on what the API
      advertises in `/v1/models` or how the gateway routes requests.
      **Fix plan:** pass session alias into `gateway.registerEngine` so
      it registers under both the canonical displayName AND the alias,
      with the alias winning when a `model:` field matches.
- [ ] **SessionCard error recovery UI.** Session stays showing the
      dead port after a bind failure. Add a "Retry on next free port"
      button next to the red error state so the user can recover
      without editing the port field manually.
- [ ] **rateLimit restart-required caption in Tray.** Parity with the
      CORS caption — user should know that rate-limit edits require
      a session restart to take effect (Hummingbird captures the
      middleware config at `Server.run()` time).
- [ ] **Dir-remove flow UX.** Verify that removing a custom dir with
      an in-flight scan doesn't leak FSEvents listeners or the
      "Missing on disk" badge for models that were under the removed
      dir.

## Deferred — needs live model + time

- [ ] **Per-model Flux/SDXL `generate()` port.** Flux1 Schnell/Dev,
      Klein — each needs Python-reference sampler loop + numerical
      unit tests against reference images + Metal kernel validation.
      Not tractable in a code-only pass.
- [ ] **Z-Image Turbo real prompt conditioning.** Current implementation
      feeds zero-tensor embeddings to the transformer (ZImage.swift:111-118).
      Needs text-encoder wire-up + numerical diff against Python reference.
- [ ] **Per-session image model / ImageSession type.** Image gen
      bypasses the session framework entirely (singleton FluxBackend).
      Splitting it into a first-class session type with port / gateway
      registration is an architecture change, not a bugfix.

## Audit snapshots (read once, source of truth)

- `docs/audit/OPEN-FIX-LIST.md` — master priority-tier tracker
- Port / CORS / gateway audit — in conversation summary (Explore agent
  output 2026-04-23); findings baked into the fixes above.
- Image gen flow audit — same day; findings baked into the image
  fixes + the "deferred" list.
