// SPDX-License-Identifier: Apache-2.0
//
// ImagePromptBar — the bottom input row on the Image screen. Handles:
//
//   • Prompt textarea
//   • Source-image upload (file picker + drag/drop) in Edit mode
//   • Strength slider (0..1) in Edit mode
//   • "Paint mask" button that opens MaskPainter as a sheet in Edit mode
//   • Generate / Edit button that calls the Engine's typed image API
//
// All settings read/write through the ImageViewModel bound by the parent
// ImageScreen. The model mode is DRIVEN BY AN EXPLICIT TAB BINDING — no
// regex or string matching on model names, per feedback_no_regex_explicit
// _settings.md.

import SwiftUI
import UniformTypeIdentifiers
import vMLXTheme
import vMLXEngine

struct ImagePromptBar: View {
    @Environment(\.appLocale) private var appLocale: AppLocale
    @Binding var prompt: String
    @Binding var sourceImage: Data?
    @Binding var maskImage: Data?
    @Binding var strength: Double
    let mode: ImageScreen.Tab
    let canSubmit: Bool
    let onSubmit: () -> Void
    let onDownloadNeeded: (() -> Void)?  // nil when selected model is downloaded
    @State private var showMaskPainter = false
    @State private var showFileImporter = false
    @State private var isDropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            if mode == .edit {
                editControls
            }

            HStack(alignment: .bottom, spacing: Theme.Spacing.sm) {
                TextField(
                    mode == .edit
                        ? "Describe the edit…"
                        : "Describe an image…",
                    text: $prompt,
                    axis: .vertical
                )
                .textFieldStyle(.plain)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
                .lineLimit(1...4)

                if let onDownloadNeeded {
                    Button {
                        onDownloadNeeded()
                    } label: {
                        Label("Download first", systemImage: "arrow.down.circle")
                            .font(Theme.Typography.bodyHi)
                            .foregroundStyle(Theme.Colors.textHigh)
                            .padding(.horizontal, Theme.Spacing.md)
                            .padding(.vertical, Theme.Spacing.sm)
                            .background(
                                RoundedRectangle(cornerRadius: Theme.Radius.md)
                                    .fill(Theme.Colors.warning)
                            )
                    }
                    .buttonStyle(.plain)
                } else {
                    Button(action: onSubmit) {
                        Text(mode == .edit ? "Edit" : "Generate")
                            .font(Theme.Typography.bodyHi)
                            .foregroundStyle(Theme.Colors.textHigh)
                            .padding(.horizontal, Theme.Spacing.md)
                            .padding(.vertical, Theme.Spacing.sm)
                            .background(
                                RoundedRectangle(cornerRadius: Theme.Radius.md)
                                    .fill(canSubmit
                                          ? Theme.Colors.accent
                                          : Theme.Colors.surfaceHi)
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSubmit)
                    .keyboardShortcut(.return, modifiers: .command)
                }
            }
        }
        .padding(Theme.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(
                            isDropTargeted ? Theme.Colors.accent : Theme.Colors.border,
                            lineWidth: isDropTargeted ? 2 : 1
                        )
                )
        )
        .padding(Theme.Spacing.lg)
        .onDrop(of: [.image, .fileURL], isTargeted: $isDropTargeted) { providers in
            guard mode == .edit, let p = providers.first else { return false }
            _ = p.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                if let data {
                    DispatchQueue.main.async { sourceImage = data }
                }
            }
            return true
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first,
               let data = try? Data(contentsOf: url) {
                sourceImage = data
            }
        }
        .sheet(isPresented: $showMaskPainter) {
            if let src = sourceImage {
                MaskPainter(
                    sourceImage: src,
                    onSave: { maskImage = $0; showMaskPainter = false },
                    onCancel: { showMaskPainter = false }
                )
            }
        }
    }

    @ViewBuilder
    private var editControls: some View {
        HStack(spacing: Theme.Spacing.md) {
            // Source image preview / pick
            Button {
                showFileImporter = true
            } label: {
                HStack(spacing: Theme.Spacing.xs) {
                    if let data = sourceImage {
                        #if canImport(AppKit)
                        if let img = NSImage(data: data) {
                            Image(nsImage: img)
                                .resizable().scaledToFill()
                                .frame(width: 36, height: 36)
                                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.sm))
                        }
                        #endif
                        Text(L10n.ImageUI.change.render(appLocale)).font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textMid)
                    } else {
                        Image(systemName: "photo.badge.plus")
                            .foregroundStyle(Theme.Colors.textMid)
                        Text(L10n.ImageUI.sourceImage.render(appLocale)).font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textMid)
                    }
                }
                .padding(.horizontal, Theme.Spacing.sm)
                .padding(.vertical, Theme.Spacing.xs)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.sm)
                        .fill(Theme.Colors.surfaceHi)
                )
            }
            .buttonStyle(.plain)

            Button {
                showMaskPainter = true
            } label: {
                Label(maskImage == nil ? "Paint mask" : "Edit mask",
                      systemImage: "paintbrush.pointed")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(sourceImage == nil
                                     ? Theme.Colors.textLow
                                     : Theme.Colors.textMid)
                    .padding(.horizontal, Theme.Spacing.sm)
                    .padding(.vertical, Theme.Spacing.xs)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.sm)
                            .fill(Theme.Colors.surfaceHi)
                    )
            }
            .buttonStyle(.plain)
            .disabled(sourceImage == nil)

            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.ImageUI.strengthFormat.format(locale: appLocale, String(format: "%.2f", strength) as NSString))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Slider(value: $strength, in: 0...1)
                    .controlSize(.small)
                    .frame(width: 180)
            }

            Spacer()
        }
    }
}
