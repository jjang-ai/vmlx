// SPDX-License-Identifier: Apache-2.0
//
// ImageTopBar — header strip at the top of the Image screen. Shows the
// selected model, a live status pill, elapsed timer while generating, a
// step counter (e.g. "12/30"), and a Stop button that only appears while a
// job is in flight.
//
// Parity with Electron components/image/ImageTopBar.tsx:
//   • status dot + label (idle | generating | editing | error)
//   • elapsed seconds
//   • current/total step
//   • cancel button

import SwiftUI
import vMLXTheme

struct ImageTopBar: View {
    let selectedModel: ImageCatalogModel?
    let status: ImageScreen.Status
    let elapsedSeconds: Int
    let currentStep: Int
    let totalSteps: Int
    let onStop: () -> Void
    let onOpenSettings: () -> Void

    var body: some View {
        HStack(spacing: Theme.Spacing.md) {
            Image(systemName: "photo.on.rectangle")
                .foregroundStyle(Theme.Colors.textMid)
            Text(selectedModel?.displayName ?? "No model selected")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)

            statusPill

            if status.isActive {
                Text("\(currentStep)/\(totalSteps)")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                    .monospacedDigit()
                Text(timeString(elapsedSeconds))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .monospacedDigit()
            }

            Spacer()

            if status.isActive {
                Button(action: onStop) {
                    Label("Stop", systemImage: "stop.circle.fill")
                        .font(Theme.Typography.bodyHi)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .padding(.horizontal, Theme.Spacing.md)
                        .padding(.vertical, Theme.Spacing.xs)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                .fill(Theme.Colors.danger)
                        )
                }
                .buttonStyle(.plain)
                .keyboardShortcut(".", modifiers: .command)
            }

            Button(action: onOpenSettings) {
                Image(systemName: "slider.horizontal.3")
                    .foregroundStyle(Theme.Colors.textMid)
                    .padding(Theme.Spacing.xs)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.sm)
                            .fill(Theme.Colors.surfaceHi)
                    )
            }
            .buttonStyle(.plain)
            .help("Image settings")
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.background)
    }

    private var statusPill: some View {
        HStack(spacing: Theme.Spacing.xs) {
            Circle().fill(status.dotColor).frame(width: 6, height: 6)
            Text(status.label)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
        }
        .padding(.horizontal, Theme.Spacing.sm)
        .padding(.vertical, 3)
        .background(
            Capsule().fill(Theme.Colors.surfaceHi)
        )
    }

    private func timeString(_ sec: Int) -> String {
        if sec < 60 { return "\(sec)s" }
        let m = sec / 60, s = sec % 60
        return String(format: "%d:%02d", m, s)
    }
}
