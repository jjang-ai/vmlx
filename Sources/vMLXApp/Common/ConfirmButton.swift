// SPDX-License-Identifier: Apache-2.0
//
// Reusable confirm-before-action button. Wraps a button in a
// `.confirmationDialog` so destructive operations never fire from a
// single stray click. macOS sheet-free: the confirmation renders as a
// lightweight popover/dialog instead of a full modal alert, keeping UX
// friction low.
//
// Used for: delete chat, delete message, clear caches, revoke API key,
// delete custom-model dir. See NO-REGRESSION-CHECKLIST.md: destructive
// ops require confirmation.

import SwiftUI
import vMLXTheme

struct ConfirmButton: View {
    let role: ButtonRole?
    let label: String
    let icon: String?
    let confirmTitle: String
    let confirmMessage: String
    let confirmAction: String
    let showsLabel: Bool
    let action: () -> Void

    @State private var showDialog = false

    init(role: ButtonRole? = .destructive,
         label: String,
         icon: String? = nil,
         confirmTitle: String,
         confirmMessage: String,
         confirmAction: String = "Delete",
         showsLabel: Bool = true,
         action: @escaping () -> Void) {
        self.role = role
        self.label = label
        self.icon = icon
        self.confirmTitle = confirmTitle
        self.confirmMessage = confirmMessage
        self.confirmAction = confirmAction
        self.showsLabel = showsLabel
        self.action = action
    }

    var body: some View {
        Button {
            showDialog = true
        } label: {
            if showsLabel {
                HStack(spacing: Theme.Spacing.xs) {
                    if let icon {
                        Image(systemName: icon)
                    }
                    Text(label)
                }
            } else if let icon {
                Image(systemName: icon)
            } else {
                Text(label)
            }
        }
        .buttonStyle(.plain)
        .help(label)
        .confirmationDialog(
            confirmTitle,
            isPresented: $showDialog,
            titleVisibility: .visible
        ) {
            Button(confirmAction, role: role) { action() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text(confirmMessage)
        }
    }
}
