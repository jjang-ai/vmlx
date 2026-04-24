// SPDX-License-Identifier: Apache-2.0
//
// ImageGenStateView — the live "generating" row rendered at the top of
// the gallery while a job is in flight. Owns no state; drives off the
// ImageViewModel's published generation fields.
//
// Shows:
//   • Progress bar (0..steps)
//   • "Step 12 / 30" label
//   • Optional ETA (derived from elapsed / step)
//   • Optional partial-preview image
//   • Stop button

import SwiftUI
import vMLXTheme

struct ImageGenStateView: View {
    let currentStep: Int
    let totalSteps: Int
    let elapsedSeconds: Int
    let preview: Data?
    let onStop: () -> Void
    @Environment(\.appLocale) private var appLocale: AppLocale

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Spacing.md) {
            previewThumb
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                Text(L10n.ImageUI.generating.render(appLocale))
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textHigh)
                ProgressView(value: fraction)
                    .tint(Theme.Colors.accent)
                HStack {
                    Text(L10n.ImageUI.stepFormat.format(locale: appLocale, Int64(currentStep), Int64(totalSteps)))
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textMid)
                        .monospacedDigit()
                    Spacer()
                    Text(L10n.ImageUI.elapsedEtaFormat.format(locale: appLocale, Int64(elapsedSeconds), etaString as NSString))
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                        .monospacedDigit()
                    Button(L10n.Chat.stop.render(appLocale), action: onStop)
                        .buttonStyle(.bordered)
                        .tint(Theme.Colors.danger)
                }
            }
        }
        .padding(Theme.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(Theme.Colors.surfaceHi)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(Theme.Colors.accent.opacity(0.5), lineWidth: 1)
                )
        )
    }

    @ViewBuilder
    private var previewThumb: some View {
        #if canImport(AppKit)
        if let data = preview, let img = NSImage(data: data) {
            Image(nsImage: img)
                .resizable()
                .scaledToFill()
                .frame(width: 120, height: 120)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.md))
        } else {
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surface)
                .frame(width: 120, height: 120)
                .overlay(
                    Image(systemName: "sparkles")
                        .foregroundStyle(Theme.Colors.textLow)
                )
        }
        #else
        RoundedRectangle(cornerRadius: Theme.Radius.md)
            .fill(Theme.Colors.surface)
            .frame(width: 120, height: 120)
        #endif
    }

    private var fraction: Double {
        guard totalSteps > 0 else { return 0 }
        return min(1.0, Double(currentStep) / Double(totalSteps))
    }

    private var etaString: String {
        guard currentStep > 0 else { return "…" }
        let perStep = Double(elapsedSeconds) / Double(currentStep)
        let remaining = Int(perStep * Double(max(0, totalSteps - currentStep)))
        if remaining < 60 { return "\(remaining)s" }
        return "\(remaining / 60)m \(remaining % 60)s"
    }
}
