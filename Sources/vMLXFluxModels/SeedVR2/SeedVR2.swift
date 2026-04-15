import Foundation
import vMLXFluxKit

// SeedVR2 — super-resolution / upscale. Python source:
// `mflux.models.seedvr2.variants.upscale.seedvr2.SeedVR2`.

public final class SeedVR2: ImageUpscaler, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "seedvr2",
            displayName: "SeedVR2 Upscaler",
            kind: .imageUpscale,
            defaultSteps: 10,
            defaultGuidance: 0.0,
            loader: { path, quant in
                _ = SeedVR2._register
                return try SeedVR2(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
    }

    public func upscale(_ request: UpscaleRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "SeedVR2.upscale — port from mflux/models/seedvr2/variants/upscale/seedvr2.py"))
        }
    }
}
