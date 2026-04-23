import SwiftUI
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif
import vMLXTheme

struct SessionsSidebar: View {
    @Bindable var vm: ChatViewModel
    @Environment(\.appLocale) private var appLocale

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: Theme.Spacing.sm) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(Theme.Colors.textLow)
                    .font(.system(size: 11))
                TextField("Search chats", text: $vm.searchQuery)
                    .textFieldStyle(.plain)
                    .font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textHigh)
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(Theme.Colors.surfaceHi)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .stroke(Theme.Colors.border, lineWidth: 1)
                    )
            )
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.top, Theme.Spacing.md)

            Button {
                vm.newSession()
            } label: {
                HStack(spacing: Theme.Spacing.sm) {
                    Image(systemName: "plus")
                    Text("New chat")
                }
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
                .frame(maxWidth: .infinity)
                .padding(.vertical, Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.accent)
                )
            }
            .buttonStyle(.plain)
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.top, Theme.Spacing.sm)

            ScrollView {
                LazyVStack(spacing: Theme.Spacing.xs) {
                    ForEach(vm.filteredSessions) { s in
                        SessionRow(
                            session: s,
                            isActive: s.id == vm.activeSessionId,
                            onSelect: { vm.selectSession(s.id) },
                            onDelete: { vm.deleteSession(s.id) },
                            onRename: { newTitle in vm.renameSession(s.id, to: newTitle) },
                            onExport: { exportSession(s) }
                        )
                    }
                }
                .padding(.horizontal, Theme.Spacing.sm)
                .padding(.top, Theme.Spacing.md)
            }

            // Footer — Clear-all button. Only enabled when there's at
            // least one chat to remove. Two-step confirm so a stray click
            // doesn't nuke the user's entire history.
            ClearAllButton(vm: vm)
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.sm)
        }
    }

    /// Opens an NSSavePanel then writes the rendered Markdown or JSON
    /// to disk. Format is chosen via the save panel's content-type
    /// filter — the user picks `.md` or `.json` from the popup and we
    /// shape the output accordingly. Messages are fetched straight
    /// from SQLite so the export reflects persisted state even if the
    /// session isn't currently selected.
    private func exportSession(_ session: ChatSession) {
        #if canImport(AppKit)
        let msgs = Database.shared.messages(for: session.id)

        let panel = NSSavePanel()
        let mdType = UTType(filenameExtension: "md") ?? .plainText
        let jsonType = UTType(filenameExtension: "json") ?? .json
        panel.allowedContentTypes = [mdType, jsonType]
        panel.nameFieldStringValue = {
            let base = session.title.isEmpty ? "chat" : session.title
            let safe = base.replacingOccurrences(of: "/", with: "-")
            return "\(safe).md"
        }()
        panel.title = "Export chat"
        panel.prompt = "Export"
        if panel.runModal() == .OK, let url = panel.url {
            // Pick format from the final filename extension — AppKit
            // honours the user's pop-up choice by rewriting the URL's
            // extension to match the selected content type.
            let ext = url.pathExtension.lowercased()
            let payload: String
            if ext == "json" {
                payload = ChatExporter.exportToJSON(session, messages: msgs)
            } else {
                payload = ChatExporter.exportToMarkdown(session, messages: msgs)
            }
            try? payload.data(using: .utf8)?.write(to: url, options: .atomic)
        }
        #endif
    }
}

private struct SessionRow: View {
    let session: ChatSession
    let isActive: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void
    let onRename: (String) -> Void
    let onExport: () -> Void

    @State private var hovered = false
    @State private var showDeleteConfirm = false
    @State private var isRenaming = false
    @State private var renameDraft = ""
    @Environment(\.appLocale) private var appLocale

    var body: some View {
        Button(action: { if !isRenaming { onSelect() } }) {
            HStack(spacing: Theme.Spacing.sm) {
                if isRenaming {
                    TextField("Chat name", text: $renameDraft)
                        .textFieldStyle(.plain)
                        .font(Theme.Typography.body)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .onSubmit { commitRename() }
                        .onExitCommand { cancelRename() }
                } else {
                    Text(session.title)
                        .font(Theme.Typography.body)
                        .foregroundStyle(isActive ? Theme.Colors.textHigh : Theme.Colors.textMid)
                        .lineLimit(1)
                }
                Spacer()
                if hovered && !isRenaming {
                    Button {
                        startRename()
                    } label: {
                        Image(systemName: "pencil")
                            .font(.system(size: 10))
                            .foregroundStyle(Theme.Colors.textLow)
                    }
                    .buttonStyle(.plain)
                    .help("Rename chat")
                    Button {
                        showDeleteConfirm = true
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(size: 10))
                            .foregroundStyle(Theme.Colors.textLow)
                    }
                    .buttonStyle(.plain)
                    .help("Delete chat")
                }
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .fill(isActive ? Theme.Colors.surfaceHi : Color.clear)
            )
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
        .contextMenu {
            Button(L10n.Common.rename.render(appLocale)) { startRename() }
            Button(L10n.Common.exportAsMarkdown.render(appLocale)) { onExport() }
            Divider()
            Button(L10n.Common.deleteChat.render(appLocale), role: .destructive) {
                showDeleteConfirm = true
            }
        }
        .confirmationDialog(
            "Delete this chat?",
            isPresented: $showDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button(L10n.Common.delete.render(appLocale), role: .destructive) { onDelete() }
            Button(L10n.Common.cancel.render(appLocale), role: .cancel) { }
        } message: {
            Text("\"\(session.title)\" and all its messages will be permanently removed. This can't be undone.")
        }
    }

    private func startRename() {
        renameDraft = session.title
        isRenaming = true
    }

    private func commitRename() {
        onRename(renameDraft)
        isRenaming = false
    }

    private func cancelRename() {
        isRenaming = false
    }
}

/// Footer button that wipes every chat after a confirmation dialog.
/// Lives at the bottom of the sidebar; disabled when there are zero chats
/// so an empty-state app doesn't show a destructive button you can't use.
private struct ClearAllButton: View {
    @Bindable var vm: ChatViewModel
    @State private var showConfirm = false
    @Environment(\.appLocale) private var appLocale

    var body: some View {
        Button {
            showConfirm = true
        } label: {
            HStack(spacing: Theme.Spacing.sm) {
                Image(systemName: "trash.slash")
                    .font(.system(size: 11, weight: .medium))
                Text("Clear all chats")
                    .font(Theme.Typography.body)
            }
            .foregroundStyle(Theme.Colors.danger)
            .frame(maxWidth: .infinity)
            .padding(.vertical, Theme.Spacing.sm)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.md)
                    .stroke(Theme.Colors.border, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(vm.sessions.isEmpty)
        .opacity(vm.sessions.isEmpty ? 0.4 : 1.0)
        .confirmationDialog(
            "Clear all chats?",
            isPresented: $showConfirm,
            titleVisibility: .visible
        ) {
            Button(L10n.Common.deleteAll.render(appLocale), role: .destructive) { vm.clearAllSessions() }
            Button(L10n.Common.cancel.render(appLocale), role: .cancel) { }
        } message: {
            Text("All \(vm.sessions.count) chats and their messages will be permanently removed. This can't be undone.")
        }
    }
}
