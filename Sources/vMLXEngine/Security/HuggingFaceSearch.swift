import Foundation

/// HuggingFace Hub model search (§250).
///
/// Wraps `huggingface.co/api/models` so the Downloads window can show a
/// search UI instead of forcing users to type `org/repo` by hand. Purely
/// additive — the existing "add by URL" paste flow keeps working.
///
/// The hub API returns a flat list of models matching a `search` term,
/// optionally filtered by tags (e.g. `mlx`, `safetensors`, `quantized`)
/// and sorted by `downloads` or `likes`. We expose a small typed result
/// so the UI can render a table without re-parsing free-form JSON.
///
/// Auth: when a token is configured in `HuggingFaceAuth`, it's forwarded
/// so gated + private repos show up for authenticated users. Without a
/// token the call still works for public models.
///
/// Rate limits: HF's public API throttles unauthenticated traffic at
/// ~100 req/min per IP. We surface `.rateLimited` as a distinct error
/// so the caller can show a 60s backoff banner instead of a generic
/// "search failed".
public enum HuggingFaceSearchError: Error, LocalizedError {
    case badURL
    case network(String)
    case badStatus(Int)
    case decodeFailed(String)
    case rateLimited

    public var errorDescription: String? {
        switch self {
        case .badURL: return "Invalid search query"
        case .network(let m): return "Network error: \(m)"
        case .badStatus(let c): return "HuggingFace API returned \(c)"
        case .decodeFailed(let m): return "Failed to parse response: \(m)"
        case .rateLimited: return "Rate-limited by HuggingFace — retry in a minute"
        }
    }
}

/// A single result row suitable for a picker. All fields are optional
/// where HF's API can return null.
public struct HuggingFaceSearchResult: Sendable, Hashable, Identifiable {
    public var id: String { modelId }
    /// Full `org/repo` identifier — the handle the DownloadManager wants.
    public let modelId: String
    /// Total download count (lifetime). Used for sorting.
    public let downloads: Int
    public let likes: Int
    public let lastModified: Date?
    /// Tag list (e.g. `["mlx", "safetensors", "4-bit"]`) for quick
    /// family / format badges in the UI.
    public let tags: [String]
    /// Whether the repo is marked gated. Surfaces a "Request access"
    /// CTA before attempting a download — otherwise the first shard
    /// download would fail with 401 and leave a half-downloaded dir.
    public let gated: Bool
    /// Pipeline tag (e.g. `text-generation`, `image-to-image`).
    public let pipeline: String?
}

public struct HuggingFaceSearch {
    /// `session` is injectable for tests; defaults to the shared URL session.
    public let session: URLSession
    /// Optional HF token; pass nil for unauthenticated calls.
    public let token: String?

    public init(session: URLSession = .shared, token: String? = nil) {
        self.session = session
        self.token = token
    }

    /// Search the HF Hub. Caps at 50 results to keep payloads small; UI
    /// should paginate by refining the query rather than scrolling.
    ///
    /// - Parameters:
    ///   - query: free-text search. Empty string returns popular models.
    ///   - filters: repo-level filters (e.g. `["mlx"]`, `["text-to-image"]`).
    ///   - sort: `downloads` (default), `likes`, `modified`, `created`.
    ///   - limit: clamped to [1, 50].
    public func search(
        query: String,
        filters: [String] = [],
        sort: String = "downloads",
        limit: Int = 30
    ) async throws -> [HuggingFaceSearchResult] {
        var comps = URLComponents(string: "https://huggingface.co/api/models")!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "search", value: query),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: String(max(1, min(limit, 50)))),
            // `full=true` asks HF to include the siblings/gated/tags
            // fields we need. Without it the response is a bare row.
            URLQueryItem(name: "full", value: "true"),
        ]
        for f in filters where !f.isEmpty {
            items.append(URLQueryItem(name: "filter", value: f))
        }
        comps.queryItems = items
        guard let url = comps.url else { throw HuggingFaceSearchError.badURL }

        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = 10
        if let token, !token.isEmpty {
            req.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let data: Data
        let resp: URLResponse
        do {
            (data, resp) = try await session.data(for: req)
        } catch {
            throw HuggingFaceSearchError.network(error.localizedDescription)
        }

        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if status == 429 { throw HuggingFaceSearchError.rateLimited }
        guard (200..<300).contains(status) else {
            throw HuggingFaceSearchError.badStatus(status)
        }

        guard let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else {
            let snippet = String(data: data.prefix(160), encoding: .utf8) ?? ""
            throw HuggingFaceSearchError.decodeFailed(snippet)
        }

        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        var out: [HuggingFaceSearchResult] = []
        out.reserveCapacity(rows.count)
        for row in rows {
            guard let id = row["modelId"] as? String ?? row["id"] as? String else { continue }
            let downloads = (row["downloads"] as? Int) ?? 0
            let likes = (row["likes"] as? Int) ?? 0
            let tags = (row["tags"] as? [String]) ?? []
            let gated = Self.parseGated(row["gated"])
            let pipeline = row["pipeline_tag"] as? String
            var lastMod: Date? = nil
            if let s = row["lastModified"] as? String {
                lastMod = fmt.date(from: s)
                    ?? ISO8601DateFormatter().date(from: s)
            }
            out.append(HuggingFaceSearchResult(
                modelId: id,
                downloads: downloads,
                likes: likes,
                lastModified: lastMod,
                tags: tags,
                gated: gated,
                pipeline: pipeline
            ))
        }
        return out
    }

    /// HF returns `gated` as either the literal bool `false` or the string
    /// `"auto"` / `"manual"`. Normalize to a single bool.
    private static func parseGated(_ raw: Any?) -> Bool {
        if let b = raw as? Bool { return b }
        if let s = raw as? String { return s != "false" }
        return false
    }
}
