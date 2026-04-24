// SPDX-License-Identifier: Apache-2.0
//
// ImageHistory — sidebar for the Image screen. Lists every past generation
// from `ImageHistoryStore` grouped by day. Clicking an item re-loads the
// prompt + settings into the prompt bar.
//
// Matches the Electron ImageHistory component:
//   • grouped by date (Today / Yesterday / absolute date)
//   • Gen / Edit badge per row
//   • status dot (pending/completed/failed/cancelled)
//   • click to recall

import SwiftUI
import vMLXTheme
import vMLXEngine

struct ImageHistory: View {
    let records: [ImageGenerationRecord]
    let onRecall: (ImageGenerationRecord) -> Void
    let onDelete: (ImageGenerationRecord) -> Void
    @Environment(\.appLocale) private var appLocale: AppLocale

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text(L10n.ImageUI.history.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.top, Theme.Spacing.md)

            if records.isEmpty {
                emptyState
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                        ForEach(grouped, id: \.0) { pair in
                            let (day, items) = pair
                            Text(day)
                                .font(Theme.Typography.caption)
                                .foregroundStyle(Theme.Colors.textLow)
                                .padding(.horizontal, Theme.Spacing.md)
                            ForEach(items) { r in
                                row(r)
                            }
                        }
                    }
                    .padding(.vertical, Theme.Spacing.sm)
                }
            }
            Spacer(minLength: 0)
        }
        .frame(width: 240)
        .background(Theme.Colors.surface)
    }

    private var emptyState: some View {
        VStack(spacing: Theme.Spacing.xs) {
            Image(systemName: "photo.stack")
                .foregroundStyle(Theme.Colors.textLow)
                .font(.title2)
            Text(L10n.ImageUI.noHistoryYet.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
        .frame(maxWidth: .infinity)
        .padding(Theme.Spacing.lg)
    }

    private var grouped: [(String, [ImageGenerationRecord])] {
        let cal = Calendar.current
        let today = cal.startOfDay(for: .now)
        let yesterday = cal.date(byAdding: .day, value: -1, to: today)!
        var buckets: [(String, [ImageGenerationRecord])] = []
        var current: (String, [ImageGenerationRecord]) = ("", [])
        let df = DateFormatter()
        df.dateStyle = .medium
        for r in records {
            let day = cal.startOfDay(for: r.createdAt)
            let label: String
            if day == today { label = "Today" }
            else if day == yesterday { label = "Yesterday" }
            else { label = df.string(from: r.createdAt) }
            if current.0 != label {
                if !current.1.isEmpty { buckets.append(current) }
                current = (label, [r])
            } else {
                current.1.append(r)
            }
        }
        if !current.1.isEmpty { buckets.append(current) }
        return buckets
    }

    private func row(_ r: ImageGenerationRecord) -> some View {
        Button {
            onRecall(r)
        } label: {
            HStack(alignment: .top, spacing: Theme.Spacing.sm) {
                Circle()
                    .fill(statusColor(r.status))
                    .frame(width: 6, height: 6)
                    .padding(.top, 6)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: Theme.Spacing.xs) {
                        Text(isEdit(r) ? "EDIT" : "GEN")
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textHigh)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(isEdit(r)
                                          ? Theme.Colors.accent.opacity(0.7)
                                          : Theme.Colors.surfaceHi)
                            )
                        Text(r.modelAlias)
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textMid)
                            .lineLimit(1)
                    }
                    Text(r.prompt)
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textMid)
                        .lineLimit(2)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.xs)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button("Recall prompt") { onRecall(r) }
            Divider()
            Button("Delete", role: .destructive) { onDelete(r) }
        }
    }

    private func isEdit(_ r: ImageGenerationRecord) -> Bool {
        r.sourceImagePath != nil
    }

    private func statusColor(_ s: ImageGenerationRecord.Status) -> Color {
        switch s {
        case .pending:   return Theme.Colors.accent
        case .completed: return Theme.Colors.success
        case .failed:    return Theme.Colors.danger
        case .cancelled: return Theme.Colors.textLow
        }
    }
}
