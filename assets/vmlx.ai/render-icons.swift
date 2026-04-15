#!/usr/bin/env swift
// Render the vMLX app icon: simple "vMLX" wordmark, SF Pro Display Black,
// white on indigo, squircle-masked at 1024×1024.
//
//   swift assets/vmlx.ai/render-icons.swift

import AppKit
import CoreText

let accent      = NSColor(srgbRed: 0x5E/255, green: 0x6A/255, blue: 0xD2/255, alpha: 1)
let surfaceDark = NSColor(srgbRed: 0x08/255, green: 0x09/255, blue: 0x0A/255, alpha: 1)
let pureWhite   = NSColor.white

func render(name: String, _ draw: (CGContext, CGRect) -> Void) {
    let size = 1024
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: size, height: size,
        bitsPerComponent: 8, bytesPerRow: 0, space: cs,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return }
    let rect = CGRect(x: 0, y: 0, width: size, height: size)
    draw(ctx, rect)
    guard let img = ctx.makeImage() else { return }
    let bitmap = NSBitmapImageRep(cgImage: img)
    let url = URL(fileURLWithPath: "assets/vmlx.ai/preview/\(name).png")
    try? bitmap.representation(using: .png, properties: [:])?.write(to: url)
    print("rendered \(name).png")
}

func drawWordmark(_ ctx: CGContext, _ r: CGRect, bg: NSColor) {
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)

    bg.setFill()
    NSBezierPath(roundedRect: r, xRadius: r.width * 0.225, yRadius: r.width * 0.225).fill()

    let font = NSFont.systemFont(ofSize: r.width * 0.30, weight: .black)
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: pureWhite,
    ]
    let line = CTLineCreateWithAttributedString(
        NSAttributedString(string: "vMLX", attributes: attrs))
    let b = CTLineGetBoundsWithOptions(line, .useGlyphPathBounds)

    ctx.saveGState()
    ctx.translateBy(x: r.midX - b.midX, y: r.midY - b.midY)
    ctx.textPosition = .zero
    CTLineDraw(line, ctx)
    ctx.restoreGState()

    NSGraphicsContext.restoreGraphicsState()
}

render(name: "icon-vmlx-indigo") { ctx, r in drawWordmark(ctx, r, bg: accent) }
render(name: "icon-vmlx-dark")   { ctx, r in drawWordmark(ctx, r, bg: surfaceDark) }

print("done — \(NSDate())")
