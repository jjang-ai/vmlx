// SPDX-License-Identifier: Apache-2.0
//
// Chat → Markdown exporter. Converts a `ChatSession` + its `[ChatMessage]`
// list into a plain-text Markdown document suitable for pasting into
// docs, issue trackers, or archival. Format mirrors the structure shown
// in-app: header block, then turn-by-turn role sections with reasoning
// and tool-call fences inlined.
//
// Pure function — no I/O, no NSSavePanel concerns. The sidebar owns the
// save panel; this module just shapes the string.

import Foundation
import vMLXEngine

enum ChatExporter {

    /// Produces a Markdown document for `session` using `messages`.
    /// Messages are assumed to be ordered chronologically (as stored).
    static func exportToMarkdown(_ session: ChatSession,
                                 messages: [ChatMessage]) -> String {
        var out = ""
        out += "# \(session.title.isEmpty ? "Untitled chat" : session.title)\n\n"

        let dateStr = Self.dateFormatter.string(from: session.createdAt)
        let modelLabel: String = {
            if let mp = session.modelPath, !mp.isEmpty {
                return (mp as NSString).lastPathComponent
            }
            return "(unspecified)"
        }()

        out += "> Created: \(dateStr)  \n"
        out += "> Model: \(modelLabel)  \n"
        out += "> Messages: \(messages.count)\n\n"
        out += "---\n\n"

        for (idx, msg) in messages.enumerated() {
            out += renderMessage(msg)
            if idx < messages.count - 1 {
                out += "\n---\n\n"
            }
        }
        // Trailing newline for POSIX hygiene.
        if !out.hasSuffix("\n") { out += "\n" }
        return out
    }

    // MARK: - Helpers

    private static func renderMessage(_ m: ChatMessage) -> String {
        var s = ""
        let header: String
        switch m.role {
        case .user:      header = "## User"
        case .assistant: header = "## Assistant"
        case .system:    header = "## System"
        case .tool:      header = "## Tool"
        }
        s += header + "\n"

        let content = m.content.trimmingCharacters(in: .whitespacesAndNewlines)
        if !content.isEmpty {
            s += content + "\n"
        } else {
            s += "_(empty)_\n"
        }

        // iter-95 §122: note attached media so the Markdown export
        // isn't silently lossy for VL chats. We don't inline base64
        // images here (Markdown exports are for paste-into-docs, not
        // archival — inlining would blow up the file). The JSON
        // exporter below does include the base64 payloads for
        // faithful round-trip.
        if !m.imageData.isEmpty {
            s += "\n_\(m.imageData.count) image\(m.imageData.count == 1 ? "" : "s") attached (see JSON export for base64 payload)_\n"
        }
        if !m.videoPaths.isEmpty {
            s += "\n_Videos attached:_\n"
            for p in m.videoPaths {
                s += "- `\(p)`\n"
            }
        }

        if let reasoning = m.reasoning, !reasoning.isEmpty {
            s += "\n### Reasoning (collapsed in chat)\n"
            s += "```\n"
            s += reasoning
            if !reasoning.hasSuffix("\n") { s += "\n" }
            s += "```\n"
        }

        if let tcJSON = m.toolCallsJSON, !tcJSON.isEmpty {
            s += "\n### Tool calls\n"
            s += "```json\n"
            s += tcJSON
            if !tcJSON.hasSuffix("\n") { s += "\n" }
            s += "```\n"
        }
        return s
    }

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()

    // MARK: - JSON export
    //
    // Structured export that downstream tooling can parse without
    // having to scrape headings out of Markdown. Shape is stable and
    // documented — `version: 1` today. When fields are added, bump
    // the version and fall back gracefully in any consumer.
    //
    // Schema:
    //   {
    //     "version": 1,
    //     "exportedAt": "<ISO-8601>",
    //     "session": {
    //       "id": "...",
    //       "title": "...",
    //       "createdAt": "<ISO-8601>",
    //       "model": "<basename-or-unspecified>"
    //     },
    //     "messages": [
    //       {
    //         "role": "user|assistant|system|tool",
    //         "content": "...",
    //         "reasoning": "..."?,       // optional
    //         "toolCallsJSON": "..."?,   // optional raw JSON blob
    //         "imagesBase64": ["..."]?,  // iter-95: VL image payload
    //         "videoPaths": ["..."]?,    // iter-95: file:// URLs
    //         "createdAt": "<ISO-8601>"
    //       }, ...
    //     ]
    //   }
    //
    // iter-95 §122: bumped to `version: 2` because the schema now
    // carries `imagesBase64` and `videoPaths`. Consumers of v1
    // should still decode cleanly (extra keys are tolerated by
    // every JSON decoder); consumers that understand v2 can
    // round-trip VL chats faithfully. Without this, exporting a
    // chat with images to JSON was silently lossy — the user saw
    // message text but no image data on re-inspection.
    static func exportToJSON(_ session: ChatSession,
                             messages: [ChatMessage]) -> String {
        struct ExportEnvelope: Encodable {
            let version: Int
            let exportedAt: String
            let session: SessionBlock
            let messages: [MessageBlock]
        }
        struct SessionBlock: Encodable {
            let id: String
            let title: String
            let createdAt: String
            let model: String
        }
        struct MessageBlock: Encodable {
            let role: String
            let content: String
            let reasoning: String?
            let toolCallsJSON: String?
            let imagesBase64: [String]?
            let videoPaths: [String]?
            let createdAt: String
        }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        let modelLabel: String = {
            if let mp = session.modelPath, !mp.isEmpty {
                return (mp as NSString).lastPathComponent
            }
            return "(unspecified)"
        }()

        let envelope = ExportEnvelope(
            version: 2,
            exportedAt: iso.string(from: Date()),
            session: .init(
                id: session.id.uuidString,
                title: session.title.isEmpty ? "Untitled chat" : session.title,
                createdAt: iso.string(from: session.createdAt),
                model: modelLabel
            ),
            messages: messages.map { m in
                MessageBlock(
                    role: {
                        switch m.role {
                        case .user: return "user"
                        case .assistant: return "assistant"
                        case .system: return "system"
                        case .tool: return "tool"
                        }
                    }(),
                    content: m.content,
                    reasoning: m.reasoning?.isEmpty == false ? m.reasoning : nil,
                    toolCallsJSON: m.toolCallsJSON?.isEmpty == false
                        ? m.toolCallsJSON : nil,
                    imagesBase64: m.imageData.isEmpty
                        ? nil
                        : m.imageData.map { $0.base64EncodedString() },
                    videoPaths: m.videoPaths.isEmpty ? nil : m.videoPaths,
                    createdAt: iso.string(from: m.createdAt)
                )
            }
        )

        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        if let data = try? enc.encode(envelope),
           let str = String(data: data, encoding: .utf8) {
            return str + "\n"
        }
        // Fallback shouldn't fire in practice — all Encodable types
        // are plain Strings/Ints/Dates. If it ever does, return a
        // minimal JSON so the save panel doesn't error out.
        return "{\"version\": 1, \"error\": \"encode_failed\"}\n"
    }
}
