import Foundation
import vMLXFluxKit

// FLUX.2 Klein family — second-generation single-encoder Flux. Python
// source: `mflux.models.flux2.variants.txt2img.flux2_klein.Flux2Klein`
// and `flux2_klein_edit.Flux2KleinEdit`.

public final class Flux2Klein: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux2-klein",
            displayName: "FLUX.2 Klein",
            kind: .imageGen,
            defaultSteps: 28,
            defaultGuidance: 3.5,
            supportsLoRA: false,
            loader: { path, quant in
                _ = Flux2Klein._register
                return try Flux2Klein(modelPath: path, quantize: quant)
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
                "Flux2Klein.generate — port from mflux/models/flux2/variants/txt2img/flux2_klein.py"))
        }
    }
}

public final class Flux2KleinEdit: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux2-klein-edit",
            displayName: "FLUX.2 Klein Edit",
            kind: .imageEdit,
            defaultSteps: 28,
            defaultGuidance: 3.5,
            loader: { path, quant in
                _ = Flux2KleinEdit._register
                return try Flux2KleinEdit(modelPath: path, quantize: quant)
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

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "Flux2KleinEdit.edit — port from mflux/models/flux2/variants/edit/flux2_klein_edit.py"))
        }
    }
}
