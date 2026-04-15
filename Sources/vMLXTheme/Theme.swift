import SwiftUI

/// Linear-inspired theme with explicit dark + light palettes. Single source
/// of truth for vMLX visual design.
///
/// **Color scheme awareness**: every color token is now a `Color` built with
/// a SwiftUI dynamic provider, so the same token resolves to the dark or
/// light palette automatically when `.preferredColorScheme(.light/.dark)`
/// flips at the Scene level. Previously every token was a hardcoded dark
/// hex which silently overrode the system color scheme — this is what
/// broke the menu-bar Appearance picker live-test 2026-04-15.
///
/// Hex sources:
///   Dark  — Linear web app inspector (Apr 2026)
///   Light — derived: invert lightness, preserve hue, raise text contrast
public enum Theme {

    // MARK: Colors
    public enum Colors {
        public static let background = dynamic(dark: 0x08090A, light: 0xFFFFFF)
        public static let surface    = dynamic(dark: 0x101112, light: 0xF7F7F8)
        public static let surfaceHi  = dynamic(dark: 0x17181B, light: 0xEDEDEF)
        public static let border     = dynamic(dark: 0x1F2023, light: 0xE0E0E4)
        public static let borderHi   = dynamic(dark: 0x2C2D31, light: 0xCFD0D4)

        public static let textHigh   = dynamic(dark: 0xF7F8F8, light: 0x0E0F11)
        public static let textMid    = dynamic(dark: 0x8A8F98, light: 0x52555C)
        public static let textLow    = dynamic(dark: 0x62666D, light: 0x84878E)

        // Accent colors stay the same across both schemes — Linear indigo
        // reads cleanly on either background, and identity colors should
        // be invariant.
        public static let accent     = Color(hex: 0x5E6AD2)
        public static let accentHi   = Color(hex: 0x7A86E8)

        public static let success    = Color(hex: 0x4CB782)
        public static let warning    = Color(hex: 0xF2994A)
        public static let danger     = Color(hex: 0xEB5757)
    }

    // MARK: Spacing
    public enum Spacing {
        public static let xs: CGFloat = 4
        public static let sm: CGFloat = 8
        public static let md: CGFloat = 12
        public static let lg: CGFloat = 16
        public static let xl: CGFloat = 24
        public static let xxl: CGFloat = 32
    }

    // MARK: Radius
    public enum Radius {
        public static let sm: CGFloat = 4
        public static let md: CGFloat = 6
        public static let lg: CGFloat = 8
        public static let xl: CGFloat = 12
    }

    // MARK: Typography
    public enum Typography {
        public static let title    = Font.system(size: 18, weight: .semibold, design: .default)
        public static let body     = Font.system(size: 13, weight: .regular, design: .default)
        public static let bodyHi   = Font.system(size: 13, weight: .medium,  design: .default)
        public static let caption  = Font.system(size: 11, weight: .regular, design: .default)
        public static let mono     = Font.system(size: 12, weight: .regular, design: .monospaced)
    }
}

extension Color {
    /// Initialize a Color from a 0xRRGGBB integer literal.
    public init(hex: UInt32, alpha: Double = 1.0) {
        let r = Double((hex >> 16) & 0xFF) / 255.0
        let g = Double((hex >>  8) & 0xFF) / 255.0
        let b = Double( hex        & 0xFF) / 255.0
        self.init(.sRGB, red: r, green: g, blue: b, opacity: alpha)
    }
}

/// Build a `Color` that resolves to two different sRGB tuples based on
/// the active `userInterfaceStyle` (light or dark). The closure form lets
/// us avoid `Color(NSColor(...))` ceremony and works across macOS 14+
/// without needing an asset catalog. Used by `Theme.Colors` so the same
/// token name (`background`, `surface`, `textHigh`) automatically picks
/// the right shade when the user flips the Appearance menu in the tray.
@inline(__always)
private func dynamic(dark: UInt32, light: UInt32) -> Color {
    #if canImport(AppKit)
    return Color(nsColor: NSColor(name: nil) { appearance in
        let isDark = appearance.bestMatch(
            from: [.darkAqua, .vibrantDark, .accessibilityHighContrastDarkAqua]
        ) != nil
        let hex = isDark ? dark : light
        let r = CGFloat((hex >> 16) & 0xFF) / 255.0
        let g = CGFloat((hex >>  8) & 0xFF) / 255.0
        let b = CGFloat( hex        & 0xFF) / 255.0
        return NSColor(srgbRed: r, green: g, blue: b, alpha: 1)
    })
    #else
    return Color(hex: dark)
    #endif
}
