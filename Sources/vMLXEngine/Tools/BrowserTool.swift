// SPDX-License-Identifier: Apache-2.0
//
// §431 — Headless / hidden-window WKWebView browser tool for VL agents.
//
// Why this exists:
//   `bash` + `open https://x` is useless agency — it just hands the
//   URL to the user's default browser and returns. The model can't
//   SEE the page, can't click, can't type. VL agents need a real
//   browser they can drive iteratively, with each action returning
//   a fresh screenshot so the next forward pass has the visual
//   ground truth.
//
// Design:
//   • One persistent `WKWebView` per engine (cookies / history / DOM
//     persist across calls) hosted in an off-screen `NSWindow`.
//   • Five primary actions: open, click, type, scroll, screenshot.
//     Plus eval, back, forward, reload, close for completeness.
//   • Every action ends with `takeSnapshot()` + `recordScreenshot`
//     on the engine actor — the same rendezvous §429 uses, so the
//     Terminal screen's auto-continue path attaches the PNG to the
//     model's next user turn without any extra wiring.
//   • Headless mode: window is positioned far off-screen and never
//     `makeKeyAndOrderFront`-ed. The view still lays out + renders
//     because it's in a window hierarchy — `takeSnapshot` works.
//   • Click / type uses JS injection (`elementFromPoint(x, y).click()`,
//     `document.activeElement.value += …`) instead of NSEvent posting.
//     Simpler, faster, and works the same way browser-automation
//     libraries (Puppeteer / Playwright) do under the hood.
//
// Limitations:
//   • Single web view per engine — multi-tab agents would need a
//     separate session-keyed map. Not worth it for v1.
//   • No file-upload / no download wiring — agents can shell out to
//     `curl` for that.
//   • Screenshot path piggy-backs on the §429 rendezvous, so a
//     `screenshot` tool call mid-browser-session will also drain
//     into the same VL re-prompt batch. That's by design.

#if canImport(WebKit) && canImport(AppKit)
import Foundation
import WebKit
import AppKit
import vMLXLMCommon

@MainActor
public final class BrowserTool {

    public struct Invocation: Sendable {
        public var action: String           // open|click|type|scroll|screenshot|eval|back|forward|reload|close
        public var url: String?
        public var x: Double?               // viewport pixels for click
        public var y: Double?
        public var selector: String?        // CSS selector for click/type — beats x,y when present
        public var text: String?            // type
        public var script: String?          // eval
        public var deltaY: Double?          // scroll
        public var width: Int = 1280
        public var height: Int = 800
        public var visible: Bool = false    // off-screen by default; true = bring window front (debug)
        public var navigationTimeoutSeconds: Double = 20

        public init(
            action: String,
            url: String? = nil,
            x: Double? = nil, y: Double? = nil,
            selector: String? = nil,
            text: String? = nil,
            script: String? = nil,
            deltaY: Double? = nil,
            width: Int = 1280, height: Int = 800,
            visible: Bool = false,
            navigationTimeoutSeconds: Double = 20
        ) {
            self.action = action
            self.url = url
            self.x = x; self.y = y
            self.selector = selector
            self.text = text
            self.script = script
            self.deltaY = deltaY
            self.width = width
            self.height = height
            self.visible = visible
            self.navigationTimeoutSeconds = navigationTimeoutSeconds
        }
    }

    public struct Result: Sendable {
        public var screenshotPath: URL?
        public var widthHint: Int?
        public var heightHint: Int?
        public var pageURL: String?
        public var pageTitle: String?
        public var evalResult: String?
        public var error: String?
    }

    public init() {}

    private var window: NSWindow?
    private var webView: WKWebView?
    /// Most recent navigation `WKNavigation` so we can `await` it via
    /// the `WKNavigationDelegate` shim. Single-flight is fine for v1.
    private var pendingNavigation: CheckedContinuation<Void, Error>?
    private let navDelegate = NavDelegate()

    public func run(_ invocation: Invocation) async -> Result {
        ensureWebView(width: invocation.width, height: invocation.height)
        let action = invocation.action.lowercased()
        var result = Result()

        switch action {
        case "open", "navigate":
            guard let urlString = invocation.url,
                  let url = URL(string: urlString) else {
                result.error = "missing or invalid 'url'"
                return await finalizeWith(result, snapshot: false)
            }
            do {
                try await navigate(to: url, timeout: invocation.navigationTimeoutSeconds)
            } catch {
                result.error = "navigation failed: \(error)"
            }

        case "click":
            if let sel = invocation.selector, !sel.isEmpty {
                let js = """
                (function(){
                  const el = document.querySelector(\(jsString(sel)));
                  if (!el) return 'NOT_FOUND';
                  el.scrollIntoView({block:'center'});
                  el.click();
                  return 'OK';
                })();
                """
                let r = await evalJS(js)
                if let s = r as? String, s != "OK" {
                    result.error = "click selector \(sel): \(s)"
                }
            } else if let x = invocation.x, let y = invocation.y {
                let js = """
                (function(){
                  const el = document.elementFromPoint(\(x), \(y));
                  if (!el) return 'NO_ELEMENT_AT_POINT';
                  el.click();
                  return 'OK';
                })();
                """
                let r = await evalJS(js)
                if let s = r as? String, s != "OK" {
                    result.error = "click \(x),\(y): \(s)"
                }
            } else {
                result.error = "click needs either selector OR (x,y)"
            }

        case "type":
            guard let text = invocation.text else {
                result.error = "type needs 'text'"
                return await finalizeWith(result, snapshot: true)
            }
            let target = invocation.selector ?? "document.activeElement"
            let selectorExpr: String
            if let sel = invocation.selector, !sel.isEmpty {
                selectorExpr = "document.querySelector(\(jsString(sel)))"
            } else {
                selectorExpr = "document.activeElement"
            }
            let js = """
            (function(){
              const el = \(selectorExpr);
              if (!el) return 'NO_TARGET';
              if (el.focus) el.focus();
              if ('value' in el) {
                el.value = (el.value || '') + \(jsString(text));
              } else {
                el.textContent = (el.textContent || '') + \(jsString(text));
              }
              el.dispatchEvent(new Event('input', {bubbles:true}));
              el.dispatchEvent(new Event('change', {bubbles:true}));
              return 'OK';
            })();
            """
            _ = target  // appease unused-let if optimization removes path
            let r = await evalJS(js)
            if let s = r as? String, s != "OK" {
                result.error = "type: \(s)"
            }

        case "scroll":
            let dy = invocation.deltaY ?? 400
            let js = "window.scrollBy({top: \(dy), behavior: 'instant'}); 'OK';"
            _ = await evalJS(js)

        case "back":
            webView?.goBack()
            try? await Task.sleep(nanoseconds: 500_000_000)
        case "forward":
            webView?.goForward()
            try? await Task.sleep(nanoseconds: 500_000_000)
        case "reload":
            webView?.reload()
            try? await Task.sleep(nanoseconds: 500_000_000)

        case "eval":
            guard let script = invocation.script else {
                result.error = "eval needs 'script'"
                return await finalizeWith(result, snapshot: false)
            }
            let r = await evalJS(script)
            result.evalResult = stringifyJSResult(r)

        case "close":
            tearDown()
            return result  // no snapshot — view is gone

        case "screenshot":
            break  // snapshot at end

        default:
            result.error = "unknown action '\(invocation.action)'"
        }

        // Bring window forward if user asked for visible debug.
        if invocation.visible, let window {
            window.setFrame(NSRect(x: 100, y: 100,
                                   width: CGFloat(invocation.width),
                                   height: CGFloat(invocation.height)),
                            display: true)
            window.orderFront(nil)
        }

        return await finalizeWith(result, snapshot: action != "close")
    }

    /// Take a snapshot + populate `pageURL` / `pageTitle` / dimensions.
    private func finalizeWith(_ partial: Result, snapshot: Bool) async -> Result {
        var r = partial
        if let webView {
            r.pageURL = webView.url?.absoluteString
            r.pageTitle = webView.title
        }
        if snapshot, let webView {
            do {
                let path = try await takeSnapshot(of: webView)
                r.screenshotPath = path
                if let dims = readPNGDimensions(path) {
                    r.widthHint = dims.0
                    r.heightHint = dims.1
                }
            } catch {
                if r.error == nil {
                    r.error = "snapshot failed: \(error)"
                }
            }
        }
        return r
    }

    // MARK: - WebView lifecycle

    private func ensureWebView(width: Int, height: Int) {
        if webView != nil { return }
        let cfg = WKWebViewConfiguration()
        cfg.websiteDataStore = .nonPersistent()  // fresh cookies per engine
        let frame = NSRect(x: 0, y: 0, width: width, height: height)
        let view = WKWebView(frame: frame, configuration: cfg)
        view.navigationDelegate = navDelegate
        navDelegate.owner = self

        // Off-screen window so layout + rendering happen but the user
        // never sees it. Borderless + utility style so it doesn't show
        // in Mission Control even if accidentally moved on-screen.
        let window = NSWindow(
            contentRect: NSRect(x: -50000, y: -50000,
                                 width: CGFloat(width),
                                 height: CGFloat(height)),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.contentView?.frame = frame
        window.alphaValue = 0.0
        window.collectionBehavior = [.transient, .ignoresCycle]
        window.orderOut(nil)
        // Force initial layout so subsequent takeSnapshot doesn't
        // come back with a black/empty image.
        view.layoutSubtreeIfNeeded()

        self.window = window
        self.webView = view
    }

    private func tearDown() {
        webView?.stopLoading()
        webView?.navigationDelegate = nil
        webView = nil
        window?.close()
        window = nil
    }

    // MARK: - Navigation

    private func navigate(to url: URL, timeout: Double) async throws {
        guard let webView else { throw NSError(domain: "BrowserTool", code: 1) }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            self.pendingNavigation = cont
            webView.load(URLRequest(url: url))
            // Bound the wait — some pages never finish loading; we
            // settle on whatever we have at the timeout.
            Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                if let self, let pending = self.pendingNavigation {
                    self.pendingNavigation = nil
                    pending.resume()  // partial-load is fine for screenshots
                }
            }
        }
    }

    fileprivate func navigationFinished(error: Error?) {
        guard let pending = self.pendingNavigation else { return }
        self.pendingNavigation = nil
        if let error {
            pending.resume(throwing: error)
        } else {
            pending.resume()
        }
    }

    // MARK: - JS eval

    private func evalJS(_ js: String) async -> Any? {
        guard let webView else { return nil }
        return await withCheckedContinuation { (cont: CheckedContinuation<Any?, Never>) in
            webView.evaluateJavaScript(js) { result, _ in
                cont.resume(returning: result)
            }
        }
    }

    private func stringifyJSResult(_ value: Any?) -> String {
        guard let value else { return "null" }
        if let s = value as? String { return s }
        if let n = value as? NSNumber { return n.stringValue }
        if let dict = value as? [String: Any],
           let data = try? JSONSerialization.data(withJSONObject: dict),
           let s = String(data: data, encoding: .utf8) {
            return s
        }
        if let arr = value as? [Any],
           let data = try? JSONSerialization.data(withJSONObject: arr),
           let s = String(data: data, encoding: .utf8) {
            return s
        }
        return String(describing: value)
    }

    private func jsString(_ s: String) -> String {
        // Conservative JSON-style string quoting — covers " \ \n \r \t.
        var out = "\""
        for ch in s {
            switch ch {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:   out.append(ch)
            }
        }
        out += "\""
        return out
    }

    // MARK: - Snapshot

    private func takeSnapshot(of webView: WKWebView) async throws -> URL {
        let img: NSImage = try await withCheckedThrowingContinuation { cont in
            let cfg = WKSnapshotConfiguration()
            cfg.afterScreenUpdates = true
            webView.takeSnapshot(with: cfg) { image, error in
                if let image { cont.resume(returning: image); return }
                cont.resume(throwing: error
                    ?? NSError(domain: "BrowserTool", code: 2,
                               userInfo: [NSLocalizedDescriptionKey: "snapshot returned nil"]))
            }
        }
        guard let tiff = img.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:])
        else {
            throw NSError(domain: "BrowserTool", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "PNG encode failed"])
        }
        let stamp = Int(Date().timeIntervalSince1970)
        let id = UUID().uuidString.prefix(8)
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-br-\(stamp)-\(id).png")
        try png.write(to: path)
        return path
    }

    private func readPNGDimensions(_ url: URL) -> (Int, Int)? {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? fh.close() }
        try? fh.seek(toOffset: 16)
        guard let header = try? fh.read(upToCount: 8), header.count == 8 else { return nil }
        let w = (UInt32(header[0]) << 24) | (UInt32(header[1]) << 16)
              | (UInt32(header[2]) << 8)  |  UInt32(header[3])
        let h = (UInt32(header[4]) << 24) | (UInt32(header[5]) << 16)
              | (UInt32(header[6]) << 8)  |  UInt32(header[7])
        return (Int(w), Int(h))
    }

    // MARK: - OpenAI tool schema

    public nonisolated static var openAISchema: ChatRequest.Tool {
        let schema: JSONValue = .object([
            "type": .string("object"),
            "properties": .object([
                "action": .object([
                    "type": .string("string"),
                    "enum": .array([
                        .string("open"), .string("click"), .string("type"),
                        .string("scroll"), .string("screenshot"), .string("eval"),
                        .string("back"), .string("forward"), .string("reload"),
                        .string("close"),
                    ]),
                    "description": .string(
                        "Browser action. 'open' loads a URL. 'click' uses selector OR (x,y). "
                      + "'type' inserts text into selector OR document.activeElement. "
                      + "Each action returns a fresh PNG screenshot you'll see on your next turn."),
                ]),
                "url": .object([
                    "type": .string("string"),
                    "description": .string("Required for 'open'. Full URL including https://."),
                ]),
                "selector": .object([
                    "type": .string("string"),
                    "description": .string("CSS selector for click/type. Preferred over (x,y) when known."),
                ]),
                "x": .object([
                    "type": .string("number"),
                    "description": .string("Viewport X (pixels) for click. Use only when no selector available."),
                ]),
                "y": .object([
                    "type": .string("number"),
                    "description": .string("Viewport Y (pixels) for click."),
                ]),
                "text": .object([
                    "type": .string("string"),
                    "description": .string("Text to type. Press Enter by including \\n at end."),
                ]),
                "script": .object([
                    "type": .string("string"),
                    "description": .string("JS for 'eval'. Last expression's value is returned as text."),
                ]),
                "delta_y": .object([
                    "type": .string("number"),
                    "description": .string("Scroll delta in pixels (positive = down). Default 400."),
                ]),
                "visible": .object([
                    "type": .string("boolean"),
                    "description": .string("Bring browser window on-screen for debugging. Default false."),
                ]),
            ]),
            "required": .array([.string("action")]),
        ])
        return ChatRequest.Tool(
            type: "function",
            function: .init(
                name: "browser",
                description: "Headless browser. open URL → see page → click / type / scroll → see results. "
                           + "Each action auto-attaches a PNG on your next input so you can SEE the page.",
                parameters: schema
            )
        )
    }
}

private final class NavDelegate: NSObject, WKNavigationDelegate, @unchecked Sendable {
    weak var owner: BrowserTool?

    @MainActor
    func webView(_ webView: WKWebView,
                 didFinish navigation: WKNavigation!) {
        owner?.navigationFinished(error: nil)
    }

    @MainActor
    func webView(_ webView: WKWebView,
                 didFail navigation: WKNavigation!,
                 withError error: Error) {
        owner?.navigationFinished(error: error)
    }

    @MainActor
    func webView(_ webView: WKWebView,
                 didFailProvisionalNavigation navigation: WKNavigation!,
                 withError error: Error) {
        owner?.navigationFinished(error: error)
    }
}

#endif
