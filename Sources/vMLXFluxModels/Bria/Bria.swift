import Foundation
@preconcurrency import MLX
import vMLXFluxKit

// MARK: - Bria
//
// Bria is referenced in the Track 1 spec
// (`docs/superpowers/specs/2026-04-27-swift-image-edit-video-port-design.md`)
// as one of the seven generation heads to port. However, the upstream
// reference repo `filipstrand/mflux` (cloned 2026-04-27 to /tmp/mflux-ref)
// does NOT contain a `models/bria/` package. The only `bria` reference
// in mflux is `models/fibo/model/fibo_transformer/bria_fibo_timesteps.py`
// — a timestep utility shared between FIBO and Bria, not a full Bria
// pipeline.
//
// Per spec rule #3 ("NO model substitution") and rule #6 (placeholder
// discipline), Bria is registered as a `notImplemented` stub. The
// registry surface stays present so downstream consumers can list it
// without crashing; calling `generate` returns a clear error pointing
// at the documentation gap.
//
// Status: documented gap, **not** a Swift-side bug.

public final class Bria: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "bria",
            displayName: "Bria",
            kind: .imageGen,
            defaultSteps: 28,
            defaultGuidance: 3.5,
            isPlaceholder: true,
            loader: { path, quant in
                _ = Bria._register
                return try Bria(modelPath: path, quantize: quant)
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
                "Bria — no upstream mflux/bria/ package exists in filipstrand/mflux as of 2026-04-27. The spec references it but the reference Python source for a faithful port doesn't ship in mflux. Track 1 documents this as a gap rather than substituting a similar model. To unblock: (a) confirm an authoritative Bria-MLX reference, or (b) drop `bria` from the registry."))
        }
    }
}
