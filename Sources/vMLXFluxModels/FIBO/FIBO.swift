import Foundation
import vMLXFluxKit

// FIBO — Python source: `mflux.models.fibo.variants.txt2img.fibo.FIBO`.

public final class FIBO: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "fibo",
            displayName: "FIBO",
            kind: .imageGen,
            defaultSteps: 20,
            defaultGuidance: 3.5,
            loader: { path, quant in
                _ = FIBO._register
                return try FIBO(modelPath: path, quantize: quant)
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

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "FIBO.generate — port from mflux/models/fibo/variants/txt2img/fibo.py"))
        }
    }
}
