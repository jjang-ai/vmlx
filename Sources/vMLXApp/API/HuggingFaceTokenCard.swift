import SwiftUI
import vMLXEngine
import vMLXTheme

/// HuggingFace access token entry, validation, and keychain storage.
///
/// Stored in the macOS Keychain via `HuggingFaceAuth.shared`. The card
/// shows a masked field (toggleable with a show/hide eye), a Test button
/// that hits `/api/whoami-v2` to verify the token, a "Signed in as @user"
/// banner when validated, and a link to `huggingface.co/settings/tokens`
/// for users who don't have a token yet.
struct HuggingFaceTokenCard: View {
    @ObservedObject var auth = HuggingFaceAuth.shared

    @State private var fieldValue: String = ""
    @State private var isRevealed: Bool = false
    @State private var saving: Bool = false
    /// O7 §293 — focuses the text field when the Downloads CTA
    /// posts `.vmlxFocusHuggingFaceTokenField`. SwiftUI's `@FocusState`
    /// programmatic focus handles the keyboard landing; the
    /// `highlightedAt` pulse briefly flashes the border so the user
    /// can see where they've been brought.
    @FocusState private var tokenFieldFocused: Bool
    @State private var highlightedAt: Date? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            headerRow
            descriptionRow
            fieldRow
            statusRow
            instructionsRow
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .stroke(highlightTint, lineWidth: highlightedAt != nil ? 2 : 1)
        )
        .task { await initialLoad() }
        .onReceive(NotificationCenter.default
            .publisher(for: .vmlxFocusHuggingFaceTokenField)
        ) { _ in
            tokenFieldFocused = true
            highlightedAt = Date()
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                if let ts = highlightedAt, Date().timeIntervalSince(ts) >= 1.9 {
                    highlightedAt = nil
                }
            }
        }
    }

    /// Border tint when the CTA just landed us here — fades after 2s.
    private var highlightTint: Color {
        highlightedAt != nil ? Theme.Colors.accent : Theme.Colors.border
    }

    // MARK: - Rows

    private var headerRow: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "key.horizontal.fill")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Theme.Colors.accent)
            Text("HuggingFace access token")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
            if auth.hasToken {
                Label("Stored", systemImage: "lock.shield.fill")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.success)
            } else {
                Label("Not set", systemImage: "lock.open")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        }
    }

    private var descriptionRow: some View {
        Text("Required to download gated or private repos. Stored in your macOS Keychain — vMLX never writes it to disk in plaintext.")
            .font(Theme.Typography.caption)
            .foregroundStyle(Theme.Colors.textMid)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var fieldRow: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Group {
                if isRevealed {
                    TextField("hf_...", text: $fieldValue)
                        .textFieldStyle(.plain)
                        .focused($tokenFieldFocused)
                } else {
                    SecureField("hf_...", text: $fieldValue)
                        .textFieldStyle(.plain)
                        .focused($tokenFieldFocused)
                }
            }
            .font(Theme.Typography.mono)
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, 10)
            .background(Theme.Colors.surfaceHi)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.md))

            Button {
                isRevealed.toggle()
            } label: {
                Image(systemName: isRevealed ? "eye.slash" : "eye")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .help(isRevealed ? "Hide token" : "Show token")

            Button {
                Task { await saveAndValidate() }
            } label: {
                if saving {
                    ProgressView().controlSize(.small)
                } else {
                    Text("Save & Test")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(fieldValue.trimmingCharacters(in: .whitespaces).isEmpty || saving)

            if auth.hasToken {
                Button(role: .destructive) {
                    auth.clear()
                    fieldValue = ""
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .help("Remove stored token")
            }
        }
    }

    @ViewBuilder
    private var statusRow: some View {
        switch auth.validation {
        case .unknown:
            if auth.hasToken {
                Label("Token stored but not verified. Click \"Save & Test\" to re-validate.", systemImage: "questionmark.circle")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            } else {
                EmptyView()
            }
        case .validating:
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text("Contacting huggingface.co/api/whoami-v2…")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        case .valid(let username):
            Label("Signed in as @\(username)", systemImage: "checkmark.seal.fill")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.success)
        case .invalid(let reason):
            Label(reason, systemImage: "exclamationmark.triangle.fill")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.danger)
        }
    }

    private var instructionsRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Text("Get a token at")
                    .foregroundStyle(Theme.Colors.textLow)
                Link("huggingface.co/settings/tokens",
                     destination: URL(string: "https://huggingface.co/settings/tokens")!)
                    .foregroundStyle(Theme.Colors.accent)
                Text("— read scope is enough.")
                    .foregroundStyle(Theme.Colors.textLow)
            }
            .font(Theme.Typography.caption)

            Text("For gated repos (e.g. Llama, Gemma), also click \"Request access\" on the model page and wait for approval.")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Actions

    private func initialLoad() async {
        auth.loadFromKeychain()
        // We never prefill the text field with the stored token — the user
        // opted in to Keychain storage precisely so the secret isn't
        // floating in UI state. If they want to see it, they type it again.
    }

    private func saveAndValidate() async {
        saving = true
        defer { saving = false }
        _ = await auth.save(token: fieldValue, validate: true)
        // On success, clear the field so the token isn't sitting in memory
        // any longer than needed.
        if case .valid = auth.validation {
            fieldValue = ""
        }
    }
}
