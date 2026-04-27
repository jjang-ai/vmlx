// SPDX-License-Identifier: Apache-2.0
//
// §429 — VL screenshot tool. Companion to BashTool for vision-capable
// models loaded in Terminal mode.
//
// Spec for the model:
//   • Tool name: "screenshot"
//   • Args:
//       - region: optional [x, y, width, height] to capture sub-rect
//       - delay:  optional seconds to wait before capture (default 0)
//       - target: optional "screen" | "active_window" (default "screen")
//   • Returns JSON: {path, width?, height?, saved: bool}
//
// The tool itself returns just the file path as a tool message. The
// Terminal stream loop watches for screenshot completions and, after
// the engine's tool-call iteration ends, auto-re-prompts with a
// synthetic user message that attaches the captured PNG as an
// `image_url` content part. The VL model SEES pixels on its next
// forward pass and can describe / decide.
//
// macOS permission gate:
//   • `/usr/sbin/screencapture` does NOT require a TCC entitlement
//     when invoked in interactive sessions on macOS 26 with screen-
//     recording permission already granted to the parent app
//   • First-run will prompt the user for Screen Recording access via
//     the standard macOS dialog
//   • If permission is denied, screencapture exits non-zero and the
//     tool returns saved=false with an error explanation

import Foundation

public actor ScreenshotTool {

    public struct Result: Sendable {
        public var path: URL
        public var savedBytes: Int
        public var widthHint: Int?
        public var heightHint: Int?
        public var error: String?
    }

    public struct Invocation: Sendable {
        public var region: [Int]?      // [x, y, w, h]
        public var delaySeconds: Double
        public var target: String      // "screen" | "active_window"

        public init(region: [Int]? = nil,
                    delaySeconds: Double = 0,
                    target: String = "screen") {
            self.region = region
            self.delaySeconds = delaySeconds
            self.target = target
        }
    }

    /// OpenAI function-tool schema for `screenshot`. Inject this into
    /// the request's `tools` array when Terminal mode has a VL model
    /// loaded.
    public nonisolated static var openAISchema: ChatRequest.Tool {
        let schema: JSONValue = .object([
            "type": .string("object"),
            "properties": .object([
                "region": .object([
                    "type": .string("array"),
                    "description": .string(
                        "Optional sub-rect to capture as [x, y, width, height] "
                      + "in screen pixels. Omit to capture the whole main display."),
                    "items": .object(["type": .string("integer")]),
                ]),
                "delay": .object([
                    "type": .string("number"),
                    "description": .string("Optional delay in seconds before capture (e.g. give a window time to open)."),
                ]),
                "target": .object([
                    "type": .string("string"),
                    "enum": .array([.string("screen"), .string("active_window")]),
                    "description": .string("What to capture. Default \"screen\" (full main display)."),
                ]),
            ]),
            "required": .array([]),
        ])
        return ChatRequest.Tool(
            type: "function",
            function: .init(
                name: "screenshot",
                description: "Capture the current screen (or a sub-region / active window) and save as PNG. The result is attached to your NEXT input as an image you can see.",
                parameters: schema
            )
        )
    }

    public init() {}

    /// Run a screencapture invocation and return the saved PNG path.
    /// Uses `/usr/sbin/screencapture` which is part of macOS base —
    /// no extra deps. Captures silently (`-x`) so no shutter sound.
    public func run(_ invocation: Invocation) async -> Result {
        // Stamp the file with seconds-since-epoch + UUID so concurrent
        // screenshots in different turns never collide and the model
        // can refer to a specific capture by name.
        let stamp = Int(Date().timeIntervalSince1970)
        let id = UUID().uuidString.prefix(8)
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-sc-\(stamp)-\(id).png")

        // Build screencapture argv. Flags:
        //   -x  silent (no camera sound)
        //   -t png  output PNG
        //   -T <s>  delay before capture
        //   -R x,y,w,h  region capture
        //   -W  active window capture (interactive — picks frontmost window)
        //   -o  no shadow on window capture
        var args: [String] = ["-x", "-t", "png"]
        if invocation.delaySeconds > 0 {
            args += ["-T", String(Int(invocation.delaySeconds.rounded()))]
        }
        if invocation.target == "active_window" {
            // -l <windowid> would be a specific id; we don't expose
            // window enumeration to the model, so use -W which picks
            // the focused window. -o suppresses the drop shadow which
            // some VL models read as content.
            args += ["-W", "-o"]
        } else if let r = invocation.region, r.count == 4 {
            args += ["-R", "\(r[0]),\(r[1]),\(r[2]),\(r[3])"]
        }
        args.append(outURL.path)

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        p.arguments = args

        // Capture stderr so we can surface permission-denied errors
        // back to the model in the tool result.
        let errPipe = Pipe()
        p.standardError = errPipe
        p.standardOutput = Pipe()  // discard

        do {
            try p.run()
            p.waitUntilExit()
        } catch {
            return Result(
                path: outURL, savedBytes: 0,
                widthHint: nil, heightHint: nil,
                error: "Failed to launch screencapture: \(error)"
            )
        }

        let exitCode = p.terminationStatus
        let errBytes = errPipe.fileHandleForReading.availableData
        let errText = String(data: errBytes, encoding: .utf8) ?? ""

        if exitCode != 0 {
            // Most common cause is "User has not granted screen
            // recording permission" — surface that verbatim.
            let hint = errText.isEmpty
                ? "screencapture exited \(exitCode) (likely missing Screen Recording permission — open System Settings → Privacy & Security → Screen & System Audio Recording → enable for vMLX)"
                : errText
            return Result(
                path: outURL, savedBytes: 0,
                widthHint: nil, heightHint: nil,
                error: hint
            )
        }

        let attrs = try? FileManager.default.attributesOfItem(atPath: outURL.path)
        let bytes = (attrs?[.size] as? Int) ?? 0

        // Extract dimensions from PNG header: bytes 16..23 hold
        // (width, height) as big-endian uint32 each. Cheap and lets
        // the tool return useful metadata without loading the image.
        var width: Int? = nil
        var height: Int? = nil
        if let fh = try? FileHandle(forReadingFrom: outURL) {
            defer { try? fh.close() }
            try? fh.seek(toOffset: 16)
            if let header = try? fh.read(upToCount: 8), header.count == 8 {
                let w = (UInt32(header[0]) << 24) | (UInt32(header[1]) << 16)
                      | (UInt32(header[2]) << 8)  |  UInt32(header[3])
                let h = (UInt32(header[4]) << 24) | (UInt32(header[5]) << 16)
                      | (UInt32(header[6]) << 8)  |  UInt32(header[7])
                width = Int(w)
                height = Int(h)
            }
        }

        return Result(
            path: outURL,
            savedBytes: bytes,
            widthHint: width,
            heightHint: height,
            error: nil
        )
    }
}
