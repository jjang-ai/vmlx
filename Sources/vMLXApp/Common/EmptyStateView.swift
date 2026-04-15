import SwiftUI
import vMLXTheme

/// Reusable empty-state panel. Mirrors the Electron `<EmptyState />` used in
/// chat / sessions / downloads / images — big SF symbol, short title,
/// caption, optional CTA button. Theme tokens only.
struct EmptyStateView: View {
    let systemImage: String
    let title: String
    var caption: String? = nil
    var cta: (String, () -> Void)? = nil

    var body: some View {
        VStack(spacing: Theme.Spacing.md) {
            Image(systemName: systemImage)
                .font(.system(size: 42, weight: .regular))
                .foregroundStyle(Theme.Colors.textLow)
            Text(title)
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textMid)
            if let caption, !caption.isEmpty {
                Text(caption)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 360)
            }
            if let cta {
                Button(cta.0, action: cta.1)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)
                    .padding(.top, Theme.Spacing.xs)
            }
        }
        .padding(Theme.Spacing.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
