import Foundation
import vMLXFluxKit

// Qwen-Image (gen) + Qwen-Image-Edit — Alibaba's image model family.
// Python source: `mflux.models.qwen.variants.{txt2img.qwen_image, edit.qwen_image_edit}`.

public final class QwenImage: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "qwen-image",
            displayName: "Qwen-Image",
            kind: .imageGen,
            defaultSteps: 30,
            defaultGuidance: 4.0,
            loader: { path, quant in
                _ = QwenImage._register
                return try QwenImage(modelPath: path, quantize: quant)
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
                "QwenImage.generate — port from mflux/models/qwen/variants/txt2img/qwen_image.py"))
        }
    }
}

public final class QwenImageEdit: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "qwen-image-edit",
            displayName: "Qwen-Image-Edit",
            kind: .imageEdit,
            defaultSteps: 30,
            defaultGuidance: 4.0,
            loader: { path, quant in
                _ = QwenImageEdit._register
                return try QwenImageEdit(modelPath: path, quantize: quant)
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
                "QwenImageEdit.edit — port from mflux/models/qwen/variants/edit/qwen_image_edit.py"))
        }
    }
}
