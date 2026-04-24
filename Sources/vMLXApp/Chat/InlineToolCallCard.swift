import SwiftUI
import vMLXTheme

/// Compact inline tool-call card rendered inside an assistant `MessageBubble`.
///
/// Layout: wrench icon + function name + status pill on the header row,
/// followed by a 2-line argument preview. Click-to-expand reveals the full
/// arguments JSON and, if available, the captured output / exit code.
///
/// Per `feedback_image_checklist.md` — tool cards must appear in the normal
/// chat path too, not just Terminal mode. Users running MCP through chat
/// need to see what the model called and how it finished.
struct InlineToolCallCard: View {
    let toolCall: InlineToolCall
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            header
            argumentsPreview
            if expanded { expandedBody }
        }
        .padding(Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
        .contentShape(RoundedRectangle(cornerRadius: Theme.Radius.md))
        .onTapGesture { expanded.toggle() }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Tool call \(toolCall.name), \(toolCall.status.rawValue)")
    }

    private var header: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "wrench.and.screwdriver")
                .font(.system(size: 11))
                .foregroundStyle(Theme.Colors.textMid)
            Text(toolCall.name)
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            statusPill
            Spacer()
            Image(systemName: expanded ? "chevron.up" : "chevron.down")
                .font(.system(size: 9))
                .foregroundStyle(Theme.Colors.textLow)
        }
    }

    private var statusPill: some View {
        HStack(spacing: 4) {
            if toolCall.status == .running {
                ProgressView().controlSize(.mini)
            }
            Text(statusLabel)
                .font(Theme.Typography.caption)
                .foregroundStyle(pillForeground)
        }
        .padding(.horizontal, Theme.Spacing.xs)
        .padding(.vertical, 1)
        .background(
            Capsule().fill(pillBackground)
        )
    }

    private var statusLabel: String {
        switch toolCall.status {
        case .pending: return "pending"
        case .running: return "running"
        case .done:    return "done"
        case .error:   return "error"
        }
    }

    private var pillForeground: Color {
        switch toolCall.status {
        case .pending: return Theme.Colors.textMid
        case .running: return Theme.Colors.accent
        case .done:    return Theme.Colors.success
        case .error:   return Theme.Colors.danger
        }
    }

    private var pillBackground: Color {
        pillForeground.opacity(0.15)
    }

    private var argumentsPreview: some View {
        Text(toolCall.arguments.isEmpty ? "(no arguments)" : toolCall.arguments)
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(Theme.Colors.textMid)
            .lineLimit(expanded ? nil : 2)
            .truncationMode(.tail)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    @Environment(\.appLocale) private var appLocale: AppLocale

    @ViewBuilder
    private var expandedBody: some View {
        if let out = toolCall.output, !out.isEmpty {
            Divider().background(Theme.Colors.border)
            Text(L10n.ChatUI.output.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Text(out)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.Colors.textHigh)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let code = toolCall.exitCode {
                Text("exit \(code)")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(code == 0 ? Theme.Colors.textLow : Theme.Colors.danger)
            }
        }
    }
}
