import SwiftUI
#if canImport(CoreImage)
import CoreImage
import CoreImage.CIFilterBuiltins
#endif
#if canImport(AppKit)
import AppKit
#endif

/// Pure-SwiftUI QR code renderer built on `CIFilter.qrCodeGenerator()`.
/// Used by `APIScreen` to show a phone-scannable endpoint URL when the
/// user enables LAN binding. Apple framework only — no third-party deps.
struct QRCodeView: View {
    let text: String
    var size: CGFloat = 160

    var body: some View {
        Group {
            if let img = makeImage() {
                Image(nsImage: img)
                    .interpolation(.none)
                    .resizable()
                    .frame(width: size, height: size)
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.2))
                    .frame(width: size, height: size)
                    .overlay(Text("QR"))
            }
        }
    }

    private func makeImage() -> NSImage? {
        #if canImport(CoreImage)
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(text.utf8)
        filter.correctionLevel = "M"
        guard let output = filter.outputImage else { return nil }
        // Scale to requested px size.
        let scale = size / max(1, output.extent.width)
        let scaled = output.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(scaled, from: scaled.extent) else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: size, height: size))
        #else
        return nil
        #endif
    }
}
