// swift-tools-version: 5.9
// vMLX private E2E driver for the SwiftUI app via the macOS Accessibility API.
//
// Why custom: SwiftUI windows are AXGroup blobs from `osascript` System
// Events. Driving them needs the lower-level AXUIElement / AXObserver APIs
// from ApplicationServices, which AppleScript can't reach. This driver
// wraps the AX C API so the test suite can: dump tree, find elements
// by role/title/identifier, click, type, screenshot a single window
// (CGWindowListCreateImage — no screen-recording TCC required),
// wait for state, and assert.
//
// Usage:
//   swift run vmlx-axdriver dump <pid>
//   swift run vmlx-axdriver click <pid> <accessibilityIdentifier>
//   swift run vmlx-axdriver type  <pid> <accessibilityIdentifier> "text"
//   swift run vmlx-axdriver shot  <pid> <out.png>
//   swift run vmlx-axdriver wait  <pid> <accessibilityIdentifier> [timeout=10]
//
// Permissions: Terminal needs "Accessibility" in System Settings →
// Privacy & Security → Accessibility (one-time). Screenshot path uses
// CGWindowList which does NOT require Screen Recording permission for
// windows in the user session.
import PackageDescription

let package = Package(
    name: "vmlx-axdriver",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "vmlx-axdriver",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("ApplicationServices"),
                .linkedFramework("AppKit"),
                .linkedFramework("CoreGraphics"),
            ]
        )
    ]
)
