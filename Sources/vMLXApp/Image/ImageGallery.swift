// SPDX-License-Identifier: Apache-2.0
//
// ImageGallery — full-parity grid. Renders:
//
//   • Error banner (HF 401/403 gated auth, or generic failure)
//   • Live GeneratingSkeleton (ImageGenStateView) when a job is in flight
//   • Empty state when no images yet
//   • Grid of past generations with redo buttons ALWAYS visible (not
//     hover-only) — per feedback_image_checklist.md + MEMORY "Redo buttons
//     always visible".

import SwiftUI
import vMLXTheme
import vMLXEngine

struct GeneratedImage: Identifiable, Hashable {
    let id = UUID()
    let data: Data
    let prompt: String
    let createdAt: Date
}

struct ImageGallery: View {
    let images: [GeneratedImage]
    let isGenerating: Bool
    let currentStep: Int
    let totalSteps: Int
    let elapsedSeconds: Int
    let preview: Data?
    let errorBanner: ImageErrorBanner?
    let onRedo: (GeneratedImage) -> Void
    let onDelete: (GeneratedImage) -> Void
    let onStop: () -> Void
    let onDismissError: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                if let banner = errorBanner {
                    errorView(banner)
                }

                if isGenerating {
                    ImageGenStateView(
                        currentStep: currentStep,
                        totalSteps: totalSteps,
                        elapsedSeconds: elapsedSeconds,
                        preview: preview,
                        onStop: onStop
                    )
                }

                if images.isEmpty && !isGenerating {
                    emptyState
                } else {
                    LazyVGrid(columns: [
                        GridItem(.adaptive(minimum: 220), spacing: Theme.Spacing.lg)
                    ], spacing: Theme.Spacing.lg) {
                        ForEach(images) { img in
                            ImageCard(
                                image: img,
                                onRedo: { onRedo(img) },
                                onDelete: { onDelete(img) }
                            )
                        }
                    }
                }
            }
            .padding(Theme.Spacing.lg)
        }
    }

    private var emptyState: some View {
        VStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 42))
                .foregroundStyle(Theme.Colors.textLow)
            Text("No images yet")
                .font(Theme.Typography.title)
                .foregroundStyle(Theme.Colors.textHigh)
            Text("Pick a model, type a prompt, and hit Generate.")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
        }
        .frame(maxWidth: .infinity, minHeight: 280)
    }

    private func errorView(_ banner: ImageErrorBanner) -> some View {
        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
            Image(systemName: banner.hfAuth
                  ? "lock.shield"
                  : "exclamationmark.triangle")
                .foregroundStyle(banner.hfAuth ? Theme.Colors.warning : Theme.Colors.danger)
            VStack(alignment: .leading, spacing: 2) {
                Text(banner.hfAuth
                     ? "Gated model — Hugging Face authentication required"
                     : "Image generation failed")
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.textHigh)
                Text(banner.message)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            Spacer()
            Button {
                onDismissError()
            } label: {
                Image(systemName: "xmark")
                    .foregroundStyle(Theme.Colors.textLow)
            }
            .buttonStyle(.plain)
        }
        .padding(Theme.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surfaceHi)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(
                            banner.hfAuth ? Theme.Colors.warning : Theme.Colors.danger,
                            lineWidth: 1
                        )
                )
        )
    }
}

struct ImageErrorBanner: Equatable {
    let message: String
    let hfAuth: Bool
}

private struct ImageCard: View {
    let image: GeneratedImage
    let onRedo: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            #if canImport(AppKit)
            if let nsimg = NSImage(data: image.data) {
                Image(nsImage: nsimg)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
                    .background(Theme.Colors.surfaceHi)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.md))
            }
            #endif

            Text(image.prompt)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
                .lineLimit(2)

            // Redo + delete ALWAYS visible (not hover-only) — see
            // feedback_image_checklist.md.
            HStack(spacing: Theme.Spacing.sm) {
                Button(action: onRedo) {
                    Label("Redo", systemImage: "arrow.clockwise")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .padding(.horizontal, Theme.Spacing.sm)
                        .padding(.vertical, Theme.Spacing.xs)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                .fill(Theme.Colors.surfaceHi)
                        )
                }
                .buttonStyle(.plain)

                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.Colors.textLow)
                        .padding(.horizontal, Theme.Spacing.sm)
                        .padding(.vertical, Theme.Spacing.xs)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                .fill(Theme.Colors.surfaceHi)
                        )
                }
                .buttonStyle(.plain)
                Spacer()
            }
        }
        .padding(Theme.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
    }
}
