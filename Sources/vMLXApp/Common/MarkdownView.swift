import SwiftUI
import vMLXTheme
#if canImport(AppKit)
import AppKit
#endif

/// Lightweight markdown renderer. Splits the input on fenced ```code blocks```
/// and renders inline prose via SwiftUI's built-in `AttributedString(markdown:)`
/// initializer. Code blocks are pulled out and rendered as monospace cards
/// with a hover-overlay copy button (top-right). No third-party deps.
///
/// Mirrors the prose / code-block split that the React MessageBubble does via
/// `react-markdown` + `rehype-highlight` — minus syntax highlighting (we
/// deliberately don't ship a tokenizer).
struct MarkdownView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            ForEach(Array(MarkdownView.parse(text).enumerated()), id: \.offset) { _, segment in
                switch segment {
                case .prose(let s):
                    proseText(s)
                case .code(let lang, let body):
                    CodeBlockView(language: lang, code: body)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func proseText(_ s: String) -> some View {
        // SwiftUI `AttributedString(markdown:)` handles **bold**, *em*, `code`,
        // [links](…), and lists at a basic level. Falls back to plain on parse
        // errors so we never crash on weird input.
        if let attr = try? AttributedString(
            markdown: s,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            Text(attr)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Text(s)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Parser

    enum Segment {
        case prose(String)
        case code(language: String, body: String)
    }

    /// Naive fenced-block splitter — handles ```lang\n...\n``` and ```\n...\n```.
    /// Good enough for assistant output; no nested fences, no indented blocks.
    static func parse(_ text: String) -> [Segment] {
        var out: [Segment] = []
        var i = text.startIndex
        var proseStart = text.startIndex
        while i < text.endIndex {
            // Scan for ```
            if text[i...].hasPrefix("```") {
                if proseStart < i {
                    out.append(.prose(String(text[proseStart..<i])))
                }
                // Move past opening fence
                let afterFence = text.index(i, offsetBy: 3)
                // Capture optional language up to newline
                var langEnd = afterFence
                while langEnd < text.endIndex, text[langEnd] != "\n" { langEnd = text.index(after: langEnd) }
                let language = String(text[afterFence..<langEnd])
                let bodyStart = langEnd < text.endIndex ? text.index(after: langEnd) : langEnd
                // Find closing fence
                var search = bodyStart
                var bodyEnd = text.endIndex
                var closeEnd = text.endIndex
                while search < text.endIndex {
                    if text[search...].hasPrefix("```") {
                        bodyEnd = search
                        closeEnd = text.index(search, offsetBy: 3)
                        break
                    }
                    search = text.index(after: search)
                }
                let body = String(text[bodyStart..<bodyEnd])
                out.append(.code(language: language, body: body))
                i = closeEnd
                proseStart = i
            } else {
                i = text.index(after: i)
            }
        }
        if proseStart < text.endIndex {
            out.append(.prose(String(text[proseStart..<text.endIndex])))
        }
        return out
    }
}

/// Code block with hover-overlay copy button. Theme tokens only.
struct CodeBlockView: View {
    let language: String
    let code: String
    @State private var hovered = false
    @State private var copied = false

    var body: some View {
        ZStack(alignment: .topTrailing) {
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(size: 12, weight: .regular, design: .monospaced))
                    .foregroundStyle(Theme.Colors.textHigh)
                    .textSelection(.enabled)
                    .padding(Theme.Spacing.md)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.surfaceHi)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .stroke(Theme.Colors.border, lineWidth: 1)
                    )
            )

            if hovered {
                Button(action: copy) {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 11))
                        .foregroundStyle(copied ? Theme.Colors.success : Theme.Colors.textMid)
                        .padding(Theme.Spacing.xs)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                .fill(Theme.Colors.surface)
                                .overlay(
                                    RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                        .stroke(Theme.Colors.border, lineWidth: 1)
                                )
                        )
                }
                .buttonStyle(.plain)
                .padding(Theme.Spacing.sm)
                .help(copied ? "Copied" : "Copy code")
            }
        }
        .onHover { hovered = $0 }
    }

    private func copy() {
        #if canImport(AppKit)
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(code, forType: .string)
        #endif
        copied = true
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            copied = false
        }
    }
}
