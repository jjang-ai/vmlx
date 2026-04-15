// SPDX-License-Identifier: Apache-2.0
//
// Minimal `multipart/form-data` parser for `/v1/images/edits` and
// `/v1/audio/transcriptions` routes — the two OpenAI endpoints whose
// wire format is multipart instead of JSON.
//
// We implement just enough of RFC 7578 to extract fields by name:
// - boundary parsing from the Content-Type header
// - per-part Content-Disposition → name + optional filename
// - per-part Content-Type (for files)
// - per-part body (raw bytes)
//
// Does NOT implement:
// - nested multipart (multipart/mixed-in-multipart/form-data)
// - quoted-printable / base64 transfer encoding (OpenAI clients send raw)
// - streamed parsing for huge uploads (we assume the whole body fits
//   in memory — the caller already bounded it via `collectBody(upTo:)`)
//
// Good enough for image edits up to 64 MB, which is the practical
// upper bound for Flux / Qwen-Image-Edit inputs.

import Foundation

/// One parsed part from a multipart/form-data body.
public struct MultipartPart: Sendable {
    /// `name` attribute from `Content-Disposition: form-data`.
    public let name: String
    /// `filename` attribute (when the part is a file upload).
    public let filename: String?
    /// Value of the part's `Content-Type` header (e.g. "image/png").
    public let contentType: String?
    /// Raw body bytes for the part. For text fields, `String(data:encoding:.utf8)`
    /// recovers the value.
    public let body: Data
}

public enum MultipartFormParser {

    /// Extract the `boundary=` token from a `Content-Type: multipart/form-data`
    /// header value. The boundary may be quoted; we strip both variants.
    public static func boundary(from contentType: String) -> String? {
        let parts = contentType.split(separator: ";").map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        for part in parts {
            if part.lowercased().hasPrefix("boundary=") {
                let raw = String(part.dropFirst("boundary=".count))
                if raw.hasPrefix("\"") && raw.hasSuffix("\"") && raw.count >= 2 {
                    return String(raw.dropFirst().dropLast())
                }
                return raw
            }
        }
        return nil
    }

    /// Parse a full multipart/form-data body and return one `MultipartPart`
    /// per discovered part. Returns an empty array if the boundary
    /// markers aren't found or the body is malformed.
    ///
    /// Parser structure (per RFC 7578 §4):
    ///
    ///     --<boundary>\r\n
    ///     Header: Value\r\n
    ///     Content-Disposition: form-data; name="image"; filename="cat.png"\r\n
    ///     Content-Type: image/png\r\n
    ///     \r\n
    ///     <raw bytes...>\r\n
    ///     --<boundary>\r\n
    ///     ...
    ///     --<boundary>--\r\n
    public static func parse(body: Data, boundary: String) -> [MultipartPart] {
        let delim = Data("--\(boundary)".utf8)
        let crlf = Data("\r\n".utf8)
        let headerEnd = Data("\r\n\r\n".utf8)

        // Find the first boundary marker.
        guard var cursor = body.range(of: delim)?.upperBound else {
            return []
        }

        var parts: [MultipartPart] = []
        while cursor < body.endIndex {
            // After a boundary marker we expect either `\r\n` (another
            // part follows) or `--\r\n` (final marker).
            guard cursor + 1 < body.endIndex else { break }
            let nextTwo = body[cursor..<min(cursor + 2, body.endIndex)]
            if nextTwo == Data("--".utf8) {
                // Final boundary marker.
                break
            }
            // Skip the CRLF after the boundary.
            guard nextTwo == crlf else { break }
            cursor += 2

            // Find header/body separator.
            guard let headerEndRange = body.range(
                of: headerEnd, in: cursor..<body.endIndex
            ) else { break }
            let headerBytes = body[cursor..<headerEndRange.lowerBound]
            let bodyStart = headerEndRange.upperBound

            // Find the next boundary marker to delimit this part's body.
            guard let nextBoundary = body.range(
                of: delim, in: bodyStart..<body.endIndex
            ) else { break }
            // Strip the trailing CRLF that sits between the body and the
            // next `--boundary` marker.
            var bodyEnd = nextBoundary.lowerBound
            if bodyEnd >= body.index(body.startIndex, offsetBy: 2),
               body[bodyEnd - 2..<bodyEnd] == crlf
            {
                bodyEnd -= 2
            }
            let partBody = body[bodyStart..<bodyEnd]

            // Parse headers — we only care about Content-Disposition and
            // Content-Type. One header per `\r\n`-terminated line.
            let headerString = String(data: headerBytes, encoding: .utf8) ?? ""
            var partName: String?
            var partFilename: String?
            var partContentType: String?
            for line in headerString.split(separator: "\r\n", omittingEmptySubsequences: false) {
                let lower = line.lowercased()
                if lower.hasPrefix("content-disposition:") {
                    (partName, partFilename) = parseContentDisposition(String(line))
                } else if lower.hasPrefix("content-type:") {
                    partContentType = String(line.dropFirst("content-type:".count))
                        .trimmingCharacters(in: .whitespaces)
                }
            }

            if let name = partName {
                parts.append(MultipartPart(
                    name: name,
                    filename: partFilename,
                    contentType: partContentType,
                    body: Data(partBody)
                ))
            }

            // Advance past this part's body + the boundary marker.
            cursor = nextBoundary.upperBound
        }
        return parts
    }

    // MARK: - Content-Disposition

    /// Parse `Content-Disposition: form-data; name="image"; filename="cat.png"`
    /// into `(name, filename)`. Handles quoted and unquoted values.
    /// Returns `(nil, nil)` when the name attribute is missing.
    private static func parseContentDisposition(
        _ line: String
    ) -> (name: String?, filename: String?) {
        // Drop the header name itself.
        let body = line
            .drop(while: { $0 != ":" })
            .dropFirst()   // colon
            .trimmingCharacters(in: .whitespaces)
        // Split on `;` but preserve quoted values.
        var params: [String: String] = [:]
        var current = ""
        var inQuotes = false
        for ch in body {
            if ch == "\"" { inQuotes.toggle(); current.append(ch); continue }
            if ch == ";" && !inQuotes {
                processDispositionParam(current, into: &params)
                current = ""
                continue
            }
            current.append(ch)
        }
        if !current.isEmpty {
            processDispositionParam(current, into: &params)
        }
        return (params["name"], params["filename"])
    }

    private static func processDispositionParam(
        _ raw: String, into params: inout [String: String]
    ) {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard let eq = trimmed.firstIndex(of: "=") else { return }
        let key = trimmed[..<eq].lowercased()
        var value = String(trimmed[trimmed.index(after: eq)...])
        if value.hasPrefix("\"") && value.hasSuffix("\"") && value.count >= 2 {
            value = String(value.dropFirst().dropLast())
        }
        params[key] = value
    }
}
