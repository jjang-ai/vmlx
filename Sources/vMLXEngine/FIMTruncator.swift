// SPDX-License-Identifier: Apache-2.0
//
// §405 — FIM (Fill-In-the-Middle) completion truncation.
//
// Mirrors the Python `truncate_fim` heuristic in
// `jang-tools/jang_tools/dsv4/bench_humaneval.py`. Trims decode-loop
// tail tokens that pollute FIM completions when a code-model keeps
// generating past the natural end of a function body. Without this
// guard, DSV4 (and other code models running FIM) emits content like:
//
//     def has_close_elements(...):
//         for i in ...:
//             if abs(...) < threshold:
//                 return True
//         return False
//
//     # Example
//     print(has_close_elements([1.0, 2.0], 0.5))      ← decode tail
//     # Test
//     assert has_close_elements(...) == True           ← decode tail
//     return True                                      ← decode loop
//     return False
//     return True
//     ...
//
// The truncation rules (in priority order, applied per line):
//
//   1. SECTION_END marker (`# Example`, `# Test`, `# Usage`,
//      `if __name__`, …) — drop this line and STOP.
//   2. EOS_TOKEN appearing inside the line — truncate the line at the
//      marker, append the truncated portion, STOP.
//   3. Top-level non-indented line that is NOT `import`/`from` — the
//      function body has ended, STOP (drop this line too).
//   4. Decode-loop detection: if the current trimmed line has appeared
//      ≥2 times in the previous 12 lines of `out`, we're entering the
//      third instance of a cycle. Truncate `out` to the start of the
//      cycle (the FIRST repeat position) and STOP.
//
// After the loop, trailing blank lines are popped and any EOS tokens
// that escaped per-line detection are stripped from the full text.
//
// Pure-Swift, allocation-light. No tokenizer required — operates on
// the assembled completion `String`. Apply at the OpenAI
// `/v1/completions` non-stream finalize step (or a streaming finalize
// hook) when the request opts in via `truncate_fim: true`.

import Foundation

public enum FIMTruncator {

    /// EOS-like tokens that DSV4 + DeepSeek-family models occasionally
    /// emit as text instead of as a real EOS token id. Order doesn't
    /// matter — we take the leftmost match per line.
    private static let eosTokens: [String] = [
        "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>",  // <｜end▁of▁sentence｜>
        "<\u{FF5C}end\u{25C1}of\u{25C1}sentence\u{FF5C}>",  // <｜end◁of◁sentence｜>
        "<|end|>",
        "<|e|>",
        "</s>",
    ]

    /// Markers that delimit the next section (Python convention from
    /// HumanEval-style code prompts). Drop the whole line and stop.
    private static let sectionEnd: [String] = [
        "# Example", "# Test", "# example", "# test",
        "# Usage", "# usage", "if __name__",
    ]

    /// Apply FIM truncation to a completion text. Returns the trimmed
    /// completion. No-op (returns input) for empty strings.
    public static func truncate(_ text: String) -> String {
        guard !text.isEmpty else { return text }

        let lines = text.components(separatedBy: "\n")
        var out: [String] = []
        out.reserveCapacity(lines.count)

        for ln in lines {
            // Rule 1: section-end marker → drop + stop.
            if sectionEnd.contains(where: { ln.contains($0) }) {
                break
            }

            // Rule 2: EOS token inside the line → truncate at marker,
            // append the prefix, stop. Use the leftmost marker found.
            var eosCutPosition: String.Index? = nil
            for tok in eosTokens {
                if let r = ln.range(of: tok) {
                    if eosCutPosition == nil || r.lowerBound < eosCutPosition! {
                        eosCutPosition = r.lowerBound
                    }
                }
            }
            if let cut = eosCutPosition {
                out.append(String(ln[..<cut]))
                break
            }

            // Rule 3: top-level non-indented line ends the function
            // body. Imports are an exception (they're legitimate
            // FIM completions for some prompts) — keep walking.
            if !ln.isEmpty,
               let first = ln.first, first != " ", first != "\t"
            {
                let stripped = ln.trimmingCharacters(in: .whitespaces)
                if stripped.hasPrefix("import ") || stripped.hasPrefix("from ") {
                    out.append(ln)
                    continue
                }
                // Anything else at column 0 = function ended.
                break
            }

            // Rule 4: decode-loop detection. If this line's stripped
            // form has appeared ≥2 prior times within the last 12
            // lines of `out`, we're entering the 3rd cycle iteration
            // — truncate to the first repeat boundary and stop.
            let s = ln.trimmingCharacters(in: .whitespaces)
            if !s.isEmpty {
                var positions: [Int] = []
                for (i, o) in out.enumerated()
                where o.trimmingCharacters(in: .whitespaces) == s {
                    positions.append(i)
                }
                if positions.count >= 2,
                   let last = positions.last,
                   last >= out.count - 12
                {
                    // Cycle confirmed. Keep everything before the FIRST
                    // repeat (positions.last == start of the recurring
                    // segment). Drop the rest.
                    out = Array(out.prefix(last))
                    break
                }
            }

            out.append(ln)
        }

        // Trailing-blank strip.
        while let last = out.last,
              last.trimmingCharacters(in: .whitespaces).isEmpty
        {
            out.removeLast()
        }

        var result = out.joined(separator: "\n")

        // Final-pass EOS strip: catches markers that snuck through (e.g.
        // appearing only on a line we already accepted).
        for marker in eosTokens {
            if let r = result.range(of: marker) {
                result = String(result[..<r.lowerBound])
            }
        }

        return result
    }
}
