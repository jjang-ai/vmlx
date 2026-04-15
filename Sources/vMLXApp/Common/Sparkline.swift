import SwiftUI
import vMLXTheme

/// Lightweight sparkline for live metrics (tok/s history, latency, etc.).
///
/// - Accepts a flat `[Double]` and scales to fit the container.
/// - Smooths with quadratic curves between midpoints of adjacent samples
///   (a.k.a. "monotone-ish smoothing") for a Linear-style organic feel without
///   a third-party charting dep.
/// - Optional exponential-moving-average pre-filter (`smoothingAlpha`) keeps
///   the line from zapping vertically when tok/s jumps from 0 → 45 on the
///   first decoded token of a request. 0 disables, 1 = raw.
/// - Optional gradient baseline fill (accent → clear) under the curve.
/// - Empty / single-sample inputs render a dashed guide line at mid-height so
///   the view never flashes to zero-height when a stream first arrives.
public struct Sparkline: View {
    public let values: [Double]
    public let color: Color
    public let showsBaseline: Bool
    public let lineWidth: CGFloat
    public let smoothingAlpha: Double
    public let showsGradientFill: Bool

    public init(
        values: [Double],
        color: Color = Theme.Colors.accent,
        showsBaseline: Bool = true,
        lineWidth: CGFloat = 1.5,
        smoothingAlpha: Double = 0.3,
        showsGradientFill: Bool = true
    ) {
        self.values = values
        self.color = color
        self.showsBaseline = showsBaseline
        self.lineWidth = lineWidth
        self.smoothingAlpha = smoothingAlpha
        self.showsGradientFill = showsGradientFill
    }

    public var body: some View {
        GeometryReader { geo in
            let smoothed = smoothedValues()
            ZStack {
                if smoothed.count < 2 {
                    // Dashed empty-state guide at mid-height.
                    Path { p in
                        let y = geo.size.height / 2
                        p.move(to: CGPoint(x: 0, y: y))
                        p.addLine(to: CGPoint(x: geo.size.width, y: y))
                    }
                    .stroke(
                        Theme.Colors.border,
                        style: StrokeStyle(lineWidth: 1, dash: [3, 3])
                    )
                } else {
                    if showsBaseline {
                        Path { p in
                            p.move(to: CGPoint(x: 0, y: geo.size.height - 0.5))
                            p.addLine(to: CGPoint(x: geo.size.width, y: geo.size.height - 0.5))
                        }
                        .stroke(Theme.Colors.border, lineWidth: 1)
                    }
                    if showsGradientFill {
                        fillPath(values: smoothed, in: geo.size)
                            .fill(
                                LinearGradient(
                                    colors: [color.opacity(0.35), color.opacity(0.0)],
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                    }
                    linePath(values: smoothed, in: geo.size)
                        .stroke(color, style: StrokeStyle(
                            lineWidth: lineWidth,
                            lineCap: .round,
                            lineJoin: .round
                        ))
                }
            }
            .animation(.easeOut(duration: 0.25), value: smoothed)
        }
    }

    /// Exponential moving average: `s[i] = α*v[i] + (1-α)*s[i-1]`. α∈[0,1].
    /// α=1 returns the raw input (no smoothing). α=0.3 is our default — cuts
    /// high-frequency spikes (first-token zaps) without lagging the tail.
    private func smoothedValues() -> [Double] {
        guard smoothingAlpha < 1.0, !values.isEmpty else { return values }
        let α = max(0.0, smoothingAlpha)
        var out = [Double]()
        out.reserveCapacity(values.count)
        var prev = values[0]
        out.append(prev)
        for i in 1..<values.count {
            let next = α * values[i] + (1 - α) * prev
            out.append(next)
            prev = next
        }
        return out
    }

    private func points(values: [Double], in size: CGSize) -> [CGPoint] {
        let minV = values.min() ?? 0
        let maxV = values.max() ?? 1
        let range = max(maxV - minV, 0.0001)
        let n = values.count
        let dx = size.width / CGFloat(max(n - 1, 1))
        return values.enumerated().map { i, v in
            let norm = (v - minV) / range
            let y = size.height - (CGFloat(norm) * (size.height - 2)) - 1
            return CGPoint(x: CGFloat(i) * dx, y: y)
        }
    }

    private func linePath(values: [Double], in size: CGSize) -> Path {
        var p = Path()
        let pts = points(values: values, in: size)
        guard pts.count >= 2 else { return p }
        p.move(to: pts[0])
        for i in 0..<(pts.count - 1) {
            let a = pts[i]
            let b = pts[i + 1]
            let mid = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
            p.addQuadCurve(to: mid, control: a)
            p.addQuadCurve(to: b, control: b)
        }
        return p
    }

    private func fillPath(values: [Double], in size: CGSize) -> Path {
        var p = linePath(values: values, in: size)
        let pts = points(values: values, in: size)
        guard let last = pts.last, let first = pts.first else { return p }
        p.addLine(to: CGPoint(x: last.x, y: size.height))
        p.addLine(to: CGPoint(x: first.x, y: size.height))
        p.closeSubpath()
        return p
    }
}
