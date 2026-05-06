// SPDX-License-Identifier: Apache-2.0
//
// Iter 128 — vmlx#121 / vmlx#133 fix.
//
// Symptom: "Add directory…" silently fails on macOS 26 with
//   NSCocoaErrorDomain Code=4099 — "XPC connection invalidated"
// for ad-hoc-signed builds. NSOpenPanel uses an XPC-launched picker
// process under the hood; macOS 26 tightened the entitlement check
// for the Open/Save panels and ad-hoc-signed binaries (the default
// for `xcodegen` + `xcodebuild` dev builds, and for any user who
// downloaded the source and built locally) lose the XPC link
// entirely. The `panel.runModal()` call returns `.cancel` with no
// visible UI — the user clicks the button and nothing happens.
//
// This helper:
//   1. Tries `NSOpenPanel.runModal()` first (works on every signed
//      release build, and on macOS pre-26).
//   2. Detects an XPC failure (panel returned `.cancel` AND zero
//      URLs AND less than 50ms elapsed → no UI was shown).
//   3. On detection, falls back to a manual-path text alert so the
//      user can always type/paste a directory path — productivity
//      preserved on ad-hoc builds.
//
// The 50ms threshold is empirical: real user dismissal of NSOpenPanel
// takes >300ms even when they immediately click Cancel; XPC-failed
// returns are typically <10ms because no panel UI was rendered.

#if canImport(AppKit)
import AppKit
import Foundation

public enum NSOpenPanelSafe {

    public struct PickResult {
        public let url: URL?
        public let usedFallback: Bool
        public let failureReason: String?
    }

    /// Run the supplied open panel; on probable XPC failure, prompt
    /// the user with a text-input alert so they can type a path
    /// manually. Returns the resolved URL (panel OR text input) or
    /// nil if the user cancels at every stage.
    @MainActor
    public static func pick(
        configure: (NSOpenPanel) -> Void,
        fallbackTitle: String = "Type a path",
        fallbackMessage: String = "macOS blocked the file picker (XPC error). Enter the directory path manually:",
        canChooseFiles: Bool = false
    ) -> PickResult {
        let panel = NSOpenPanel()
        configure(panel)

        let start = CFAbsoluteTimeGetCurrent()
        let response = panel.runModal()
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Happy path — picker UI fired and user picked or cancelled.
        if response == .OK, let url = panel.url {
            return PickResult(url: url, usedFallback: false, failureReason: nil)
        }

        // Probable XPC failure: cancelled with no URL AND elapsed
        // is suspiciously fast (no real UI). Real user cancels take
        // ≥300ms from button-tap → click-cancel even at 60Hz.
        let xpcSuspected = (response != .OK) && (elapsed < 0.05)

        if xpcSuspected {
            // Manual fallback alert with a text field.
            return runManualPathAlert(
                title: fallbackTitle,
                message: fallbackMessage,
                canChooseFiles: canChooseFiles)
        }

        // User legitimately cancelled.
        return PickResult(url: nil, usedFallback: false, failureReason: nil)
    }

    @MainActor
    private static func runManualPathAlert(
        title: String,
        message: String,
        canChooseFiles: Bool
    ) -> PickResult {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .informational
        let textField = NSTextField(frame: NSRect(
            x: 0, y: 0, width: 360, height: 24))
        textField.placeholderString = "/path/to/directory"
        // Allow paste from clipboard via Cmd-V.
        textField.isEditable = true
        textField.isSelectable = true
        alert.accessoryView = textField
        alert.addButton(withTitle: "Use Path")
        alert.addButton(withTitle: "Cancel")

        let resp = alert.runModal()
        guard resp == .alertFirstButtonReturn else {
            return PickResult(
                url: nil, usedFallback: true,
                failureReason: "User cancelled fallback")
        }

        let raw = textField.stringValue.trimmingCharacters(
            in: .whitespacesAndNewlines)
        guard !raw.isEmpty else {
            return PickResult(
                url: nil, usedFallback: true,
                failureReason: "Empty path")
        }

        // Resolve `~` expansion + create file URL.
        let expanded = (raw as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)

        // Verify path exists. For directory pickers, must be a
        // directory.
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: url.path, isDirectory: &isDir)
        guard exists else {
            return PickResult(
                url: nil, usedFallback: true,
                failureReason: "Path does not exist")
        }
        if !canChooseFiles && !isDir.boolValue {
            return PickResult(
                url: nil, usedFallback: true,
                failureReason: "Path is not a directory")
        }

        return PickResult(
            url: url, usedFallback: true, failureReason: nil)
    }
}

#endif
