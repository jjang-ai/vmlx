# vMLX Logo & Branding

## Files Created

### Logo Assets
- **`resources/logo.svg`** - Master SVG logo (512x512)
  - Blue gradient background (Apple-style)
  - "vMLX" text in SF Pro Display style
  - Subtle chip icon accent (representing Apple Silicon/MLX)

### App Icons
- **`resources/icon.png`** - Main app icon (1024x1024)
  - Used by electron-builder for macOS .app bundle
  - Automatically generates .icns during build

### Favicons
- **`public/favicon.svg`** - SVG favicon (32x32 optimized)
- **`public/favicon-32.png`** - PNG fallback (32x32)
- **`public/favicon-16.png`** - PNG fallback (16x16)

## Design Details

### Color Palette
- Background gradient: `#1e3a8a` → `#3b82f6` (blue-700 to blue-500)
- Text: White with subtle gradient (`#ffffff` → `#e0e7ff`)
- Accent chip: Light blue tones (`#60a5fa`, `#93c5fd`, `#dbeafe`)

### Typography
- Font: SF Pro Display / -apple-system fallback
- Weight: 600 (Semibold)
- Letter spacing: -2 (tight, Apple-style)

### Visual Theme
- Rounded square background (115px border radius on 512px = ~22.5%)
- Minimalist chip icon above text (represents MLX/Apple Silicon)
- Clean, modern Apple aesthetic
- Matches macOS Big Sur+ icon style

## Regenerating Icons

To regenerate icons from the SVG source:

```bash
npm run icons
```

This will:
1. Convert `resources/logo.svg` → `resources/icon.png` (1024x1024)
2. Generate favicon sizes (32px, 16px)

## App Bundle

The app will display as **"vMLX"** in:
- macOS menu bar
- Dock
- Application folder
- Finder

Product name is set in `package.json`:
```json
"build": {
  "productName": "vMLX",
  ...
}
```

## Notes

- Icon follows macOS Big Sur+ design guidelines
- SVG favicon for modern browsers
- PNG fallbacks for older browsers
- 1024x1024 source ensures crisp display on Retina displays
- electron-builder auto-generates .icns with all required sizes
