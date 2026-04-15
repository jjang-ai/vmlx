// SPDX-License-Identifier: Apache-2.0
//
// Numeric field with inline range validation. Used by
// `SessionConfigForm` to replace loose text fields / steppers that
// previously accepted any value (negative numbers, zero block sizes,
// ports out of range). Two behaviors:
//
//  1. Clamps the committed value to `range` on write.
//  2. Shows a tooltip + red border while the user is typing a value
//     outside the range, without eating the keystroke.
//
// Pure SwiftUI, theme tokens only, no regex.

import SwiftUI
import vMLXTheme

/// Type-erased numeric range validator. `Double` is used internally so
/// the same component drives int sliders (Port, maxNumSeqs, …) and
/// float sliders (Temperature, topP, …).
struct ValidatedField: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let format: String
    /// Optional cross-field invariant check, e.g. `deepSec >= softSec`.
    /// Returns an error message if invalid, nil when OK.
    let invariant: ((Double) -> String?)?

    init(title: String,
         value: Binding<Double>,
         range: ClosedRange<Double>,
         step: Double = 1,
         format: String = "%.0f",
         invariant: ((Double) -> String?)? = nil) {
        self.title = title
        self._value = value
        self.range = range
        self.step = step
        self.format = format
        self.invariant = invariant
    }

    @State private var text: String = ""
    @State private var error: String? = nil
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack {
                Text(title)
                    .font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textMid)
                Spacer()
                TextField("", text: $text)
                    .textFieldStyle(.plain)
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textHigh)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 120)
                    .focused($focused)
                    .padding(.horizontal, Theme.Spacing.sm)
                    .padding(.vertical, Theme.Spacing.xs)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.sm)
                            .fill(Theme.Colors.surfaceHi)
                            .overlay(
                                RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                    .stroke(error != nil
                                            ? Color.red.opacity(0.8)
                                            : Theme.Colors.border,
                                            lineWidth: 1)
                            )
                    )
                    .help(helpText)
                    .onAppear { text = formatted(value) }
                    .onChange(of: focused) { _, isFocused in
                        if !isFocused { commit() }
                    }
                    .onSubmit { commit() }
            }
            Slider(value: $value, in: range, step: step, onEditingChanged: { editing in
                if !editing {
                    text = formatted(value)
                    error = nil
                }
            })
            .tint(Theme.Colors.accent)

            if let error {
                Text(error)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Color.red.opacity(0.9))
            }
        }
    }

    private var helpText: String {
        "Valid range: \(formatted(range.lowerBound))–\(formatted(range.upperBound))"
    }

    private func formatted(_ v: Double) -> String {
        String(format: format, v)
    }

    private func commit() {
        guard let parsed = Double(text.trimmingCharacters(in: .whitespaces)) else {
            error = "Not a number"
            text = formatted(value)
            return
        }
        let clamped = Self.clamp(parsed, to: range)
        if clamped != parsed {
            error = "Clamped to \(formatted(range.lowerBound))–\(formatted(range.upperBound))"
        } else if let inv = invariant, let msg = inv(clamped) {
            error = msg
            value = clamped
            text = formatted(clamped)
            return
        } else {
            error = nil
        }
        value = clamped
        text = formatted(clamped)
    }

    /// Public static so tests can drive the clamp logic without building a view.
    static func clamp(_ v: Double, to range: ClosedRange<Double>) -> Double {
        min(max(v, range.lowerBound), range.upperBound)
    }

    /// Pure validation helper used by tests. Returns a tuple of
    /// `(clampedValue, errorMessage?)` matching the semantics of
    /// `commit()` above.
    static func validate(_ v: Double,
                         range: ClosedRange<Double>,
                         invariant: ((Double) -> String?)? = nil) -> (Double, String?) {
        let clamped = clamp(v, to: range)
        if clamped != v {
            return (clamped, "out of range")
        }
        if let inv = invariant, let msg = inv(clamped) {
            return (clamped, msg)
        }
        return (clamped, nil)
    }
}
