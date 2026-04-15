// SPDX-License-Identifier: Apache-2.0
//
// ImageSettingsDrawer — popover / drawer with per-generation knobs:
// steps, guidance, seed, width/height, num_images, scheduler, and (in
// Edit mode) the reminder that strength lives on the prompt bar.
//
// Read/write through a binding to an `ImageGenSettings` struct owned by
// the parent ImageScreen's view model. Changes ALSO pushed into
// SettingsStore as global defaults so next app launch remembers the
// last-used values. Mirrors the Electron ImageSettings component.

import SwiftUI
import vMLXTheme
import vMLXEngine

struct ImageSettingsDrawer: View {
    @Binding var settings: ImageGenSettings
    let mode: ImageScreen.Tab
    let onClose: () -> Void
    let onPersist: (ImageGenSettings) -> Void  // pushed to SettingsStore

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            header
            row("Steps", value: Double(settings.steps), range: 1...80, step: 1) { v in
                settings.steps = Int(v)
            }
            row("Guidance", value: settings.guidance, range: 0...20, step: 0.1) { v in
                settings.guidance = v
            }
            row("Width", value: Double(settings.width), range: 256...2048, step: 64) { v in
                settings.width = Int(v)
            }
            row("Height", value: Double(settings.height), range: 256...2048, step: 64) { v in
                settings.height = Int(v)
            }
            row("Num images", value: Double(settings.numImages), range: 1...4, step: 1) { v in
                settings.numImages = Int(v)
            }

            HStack {
                Text("Seed")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                    .frame(width: 90, alignment: .leading)
                TextField("-1 for random", value: $settings.seed, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
                Button("Random") { settings.seed = -1 }
                    .buttonStyle(.link)
            }

            HStack {
                Text("Scheduler")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                    .frame(width: 90, alignment: .leading)
                Picker("", selection: $settings.scheduler) {
                    Text("Default").tag("default")
                    Text("DDIM").tag("ddim")
                    Text("Euler").tag("euler")
                    Text("DPM++").tag("dpmpp")
                }
                .labelsHidden()
                .frame(width: 180)
            }

            if mode == .edit {
                Text("Strength lives on the prompt bar.")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }

            HStack {
                Spacer()
                Button("Close") { onClose() }
                Button("Save as default") {
                    onPersist(settings)
                    onClose()
                }
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(Theme.Spacing.lg)
        .frame(width: 420)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
    }

    private var header: some View {
        HStack {
            Text("Image settings")
                .font(Theme.Typography.title)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
        }
    }

    private func row(
        _ label: String,
        value: Double,
        range: ClosedRange<Double>,
        step: Double,
        onChange: @escaping (Double) -> Void
    ) -> some View {
        HStack {
            Text(label)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
                .frame(width: 90, alignment: .leading)
            Slider(
                value: Binding(
                    get: { value },
                    set: { onChange($0) }
                ),
                in: range,
                step: step
            )
            Text(step >= 1 ? "\(Int(value))" : String(format: "%.2f", value))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .frame(width: 48, alignment: .trailing)
                .monospacedDigit()
        }
    }
}
