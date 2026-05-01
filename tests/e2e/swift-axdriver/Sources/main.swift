// Driver for the SwiftUI vMLX app via the macOS Accessibility API.
import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

// MARK: - AX helpers

func axApp(pid: pid_t) -> AXUIElement {
    AXUIElementCreateApplication(pid)
}

func ensureTrust() {
    let opts: NSDictionary = [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: true]
    if !AXIsProcessTrustedWithOptions(opts) {
        FileHandle.standardError.write(Data("axdriver: not yet trusted — grant Accessibility to Terminal in System Settings, then re-run.\n".utf8))
    }
}

func attr(_ el: AXUIElement, _ name: String) -> CFTypeRef? {
    var v: CFTypeRef?
    return AXUIElementCopyAttributeValue(el, name as CFString, &v) == .success ? v : nil
}

func childrenOf(_ el: AXUIElement) -> [AXUIElement] {
    (attr(el, kAXChildrenAttribute) as? [AXUIElement]) ?? []
}

func roleOf(_ el: AXUIElement) -> String { (attr(el, kAXRoleAttribute) as? String) ?? "?" }
func titleOf(_ el: AXUIElement) -> String { (attr(el, kAXTitleAttribute) as? String) ?? "" }
func valueOf(_ el: AXUIElement) -> String {
    let v = attr(el, kAXValueAttribute)
    if let s = v as? String { return s }
    return ""
}
func identOf(_ el: AXUIElement) -> String {
    (attr(el, kAXIdentifierAttribute) as? String) ?? ""
}
func descOf(_ el: AXUIElement) -> String {
    (attr(el, kAXDescriptionAttribute) as? String) ?? ""
}

func walk(_ el: AXUIElement, depth: Int = 0, visit: (AXUIElement, Int) -> Bool) {
    if !visit(el, depth) { return }
    for c in childrenOf(el) {
        walk(c, depth: depth + 1, visit: visit)
    }
}

// MARK: - Commands

func cmdDump(pid: pid_t) {
    let app = axApp(pid: pid)
    walk(app) { el, depth in
        let r = roleOf(el)
        let t = titleOf(el)
        let v = valueOf(el)
        let i = identOf(el)
        let d = descOf(el)
        let pad = String(repeating: "  ", count: depth)
        var line = "\(pad)\(r)"
        if !t.isEmpty { line += " title=\"\(t.prefix(60))\"" }
        if !i.isEmpty { line += " id=\"\(i.prefix(40))\"" }
        if !d.isEmpty { line += " desc=\"\(d.prefix(40))\"" }
        if !v.isEmpty && r != "AXGroup" { line += " value=\"\(v.prefix(40))\"" }
        print(line)
        return true
    }
}

func find(pid: pid_t, predicate: (AXUIElement) -> Bool) -> AXUIElement? {
    let app = axApp(pid: pid)
    var hit: AXUIElement?
    walk(app) { el, _ in
        if predicate(el) { hit = el; return false }
        return true
    }
    return hit
}

func cmdClick(pid: pid_t, ident: String) -> Int32 {
    guard let el = find(pid: pid, predicate: { identOf($0) == ident || titleOf($0) == ident }) else {
        FileHandle.standardError.write(Data("axdriver: no element matched \"\(ident)\"\n".utf8))
        return 2
    }
    let r = AXUIElementPerformAction(el, kAXPressAction as CFString)
    if r != .success {
        FileHandle.standardError.write(Data("axdriver: AXPressAction failed: \(r.rawValue)\n".utf8))
        return 3
    }
    return 0
}

func cmdType(pid: pid_t, ident: String, text: String) -> Int32 {
    guard let el = find(pid: pid, predicate: { identOf($0) == ident || titleOf($0) == ident }) else {
        FileHandle.standardError.write(Data("axdriver: no element matched \"\(ident)\"\n".utf8))
        return 2
    }
    // Focus first
    AXUIElementSetAttributeValue(el, kAXFocusedAttribute as CFString, true as CFTypeRef)
    let r = AXUIElementSetAttributeValue(el, kAXValueAttribute as CFString, text as CFTypeRef)
    return r == .success ? 0 : 4
}

func cmdShot(pid: pid_t, outPath: String) -> Int32 {
    let info = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as! [[String: Any]]
    let candidates = info.filter { ($0[kCGWindowOwnerPID as String] as? pid_t) == pid }
    guard let first = candidates.first, let wid = first[kCGWindowNumber as String] as? CGWindowID else {
        FileHandle.standardError.write(Data("axdriver: no on-screen window for pid=\(pid)\n".utf8))
        return 2
    }
    guard let cg = CGWindowListCreateImage(.null, [.optionIncludingWindow], wid, [.bestResolution, .boundsIgnoreFraming]) else {
        FileHandle.standardError.write(Data("axdriver: CGWindowListCreateImage failed (Screen Recording perm may be required on macOS 14+)\n".utf8))
        return 3
    }
    let rep = NSBitmapImageRep(cgImage: cg)
    guard let png = rep.representation(using: .png, properties: [:]) else { return 4 }
    do { try png.write(to: URL(fileURLWithPath: outPath)) } catch { return 5 }
    print("wrote \(outPath) (\(cg.width)x\(cg.height))")
    return 0
}

func cmdWait(pid: pid_t, ident: String, timeoutSec: Double = 10) -> Int32 {
    let deadline = Date().addingTimeInterval(timeoutSec)
    while Date() < deadline {
        if find(pid: pid, predicate: { identOf($0) == ident || titleOf($0) == ident }) != nil {
            return 0
        }
        Thread.sleep(forTimeInterval: 0.25)
    }
    FileHandle.standardError.write(Data("axdriver: timed out waiting for \"\(ident)\"\n".utf8))
    return 6
}

func cmdGrep(pid: pid_t, needle: String) {
    let app = axApp(pid: pid)
    walk(app) { el, _ in
        let blob = "\(roleOf(el)) \(titleOf(el)) \(identOf(el)) \(descOf(el)) \(valueOf(el))"
        if blob.localizedCaseInsensitiveContains(needle) {
            print("\(roleOf(el)) title=\"\(titleOf(el))\" id=\"\(identOf(el))\" value=\"\(valueOf(el).prefix(80))\"")
        }
        return true
    }
}

// MARK: - Main

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("""
    usage:
      vmlx-axdriver dump <pid>
      vmlx-axdriver grep <pid> <needle>
      vmlx-axdriver click <pid> <id-or-title>
      vmlx-axdriver type  <pid> <id-or-title> "text"
      vmlx-axdriver shot  <pid> <out.png>
      vmlx-axdriver wait  <pid> <id-or-title> [timeout=10]
    """)
    exit(1)
}

ensureTrust()
let cmd = args[1]
guard let pid = pid_t(args[2]) else { print("bad pid"); exit(1) }
let rest = Array(args.dropFirst(3))

switch cmd {
case "dump":  cmdDump(pid: pid)
case "grep":  guard rest.count >= 1 else { exit(1) }; cmdGrep(pid: pid, needle: rest[0])
case "click": guard rest.count >= 1 else { exit(1) }; exit(cmdClick(pid: pid, ident: rest[0]))
case "type":  guard rest.count >= 2 else { exit(1) }; exit(cmdType(pid: pid, ident: rest[0], text: rest[1]))
case "shot":  guard rest.count >= 1 else { exit(1) }; exit(cmdShot(pid: pid, outPath: rest[0]))
case "wait":
    guard rest.count >= 1 else { exit(1) }
    let to = rest.count >= 2 ? (Double(rest[1]) ?? 10) : 10
    exit(cmdWait(pid: pid, ident: rest[0], timeoutSec: to))
default:
    print("unknown command: \(cmd)"); exit(1)
}
