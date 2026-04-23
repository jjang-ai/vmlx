//
//  CORSAllowlistMiddleware.swift
//  vMLXServer
//
//  §331 — enforces a CORS origin allowlist.
//
//  Hummingbird's built-in `CORSMiddleware.AllowOrigin` has four cases:
//    - `.all`              → header echoes `*`
//    - `.custom(String)`   → header echoes the fixed value (single origin)
//    - `.originBased`      → header echoes whatever `Origin` was on the
//                            request — NO allowlist gating
//    - `.none`             → header stays absent
//
//  Prior to this middleware, `Server.resolveAllowOrigin(_:)` collapsed
//  a 2+-entry allowlist (`["example.com", "app.com"]`) to `.originBased`,
//  which means any browser origin received `Access-Control-Allow-Origin:
//  <their-origin>` headers back — equivalent to `.all` for practical
//  purposes, since the CORS security boundary is client-side.
//
//  This middleware wraps the allowlist contract: intercept every
//  request, and if the `Origin` header is set but NOT in the allowlist,
//  strip it before passing to the next middleware (including
//  Hummingbird's CORSMiddleware). Hummingbird's middleware skips the
//  CORS headers entirely when `Origin` is absent, so a rejected origin
//  simply gets no CORS headers — the browser then fails the preflight /
//  blocks the response body, which is the correct behavior for a
//  CORS-restricted server.
//
//  For preflight `OPTIONS` requests from disallowed origins, we return
//  403 Forbidden so the browser's error surface matches what a proper
//  allowlist server would emit (rather than a silently-missing Allow-
//  Origin header that produces a generic browser error).
//

import Foundation
import HTTPTypes
import Hummingbird

/// Wraps Hummingbird's built-in CORSMiddleware with strict allowlist
/// enforcement for the multi-origin case. Install this BEFORE the
/// Hummingbird CORSMiddleware in the router chain — we filter the
/// `Origin` header, then let Hummingbird handle the standard CORS
/// response shape for allowed requests.
public struct CORSAllowlistMiddleware<Context: RequestContext>: RouterMiddleware {
    public let allowedOrigins: Set<String>

    public init(allowedOrigins: [String]) {
        // Normalize: drop empty + lowercase scheme+host for case-
        // insensitive match. Origin header values are URL-shaped
        // (e.g. `https://example.com`), and `Origin` comparison in
        // browsers is case-sensitive on path but case-insensitive on
        // scheme+host. Strip trailing slashes since some clients send
        // `https://example.com/` on POST redirects.
        self.allowedOrigins = Set(
            allowedOrigins
                .filter { !$0.isEmpty }
                .map { $0.lowercased().trimmingCharacters(
                    in: CharacterSet(charactersIn: "/")) }
        )
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard let rawOrigin = request.headers[.origin], !rawOrigin.isEmpty
        else {
            // No Origin header → same-origin request, let it through.
            return try await next(request, context)
        }
        let normalized = rawOrigin.lowercased().trimmingCharacters(
            in: CharacterSet(charactersIn: "/"))
        if allowedOrigins.contains(normalized) {
            return try await next(request, context)
        }
        // Origin is set but NOT in the allowlist.
        // For preflight OPTIONS: explicit 403 so the browser surfaces
        // a clear CORS-denied error instead of a generic
        // "missing Allow-Origin" failure.
        if request.method == .options {
            return Response(
                status: .forbidden,
                headers: [.contentType: "application/json; charset=utf-8"],
                body: .init(byteBuffer: .init(string:
                    #"{"error":{"message":"origin not permitted","type":"cors_error"}}"#))
            )
        }
        // For non-preflight: strip Origin so the downstream
        // CORSMiddleware skips header emission entirely — the
        // browser's same-origin fetch will still receive the body,
        // but cross-origin fetches will see no Allow-Origin and
        // block the response client-side. This is the standard
        // "origin not allowed" outcome for CORS-restricted servers.
        // `Request.head` is `let`, so we rebuild it from a mutable
        // copy of the underlying HTTPRequest whose headerFields is
        // settable. Request(head:body:) takes a HTTPRequest + the
        // existing RequestBody, preserving the body stream.
        var head = request.head
        head.headerFields[.origin] = nil
        let stripped = Request(head: head, body: request.body)
        return try await next(stripped, context)
    }
}
