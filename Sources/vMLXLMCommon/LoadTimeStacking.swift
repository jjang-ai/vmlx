// Copyright © 2026 Jinho Jang. All rights reserved.

import MLX

/// Stack load-time per-expert tensors and immediately materialize the result.
///
/// MLX stack operations are lazy. During model `sanitize(...)`, that can leave
/// a retained graph pointing at every per-expert input tensor until the final
/// model eval. Large MoE/JANGTQ bundles can have tens of thousands of
/// per-expert tensors, so the lazy graph doubles the peak footprint and can
/// crash before the model finishes loading. Use this helper for weight-loading
/// restacks, not for small runtime stacks in forward passes.
public func loadTimeMaterializedStacked(_ arrays: [MLXArray], axis: Int = 0) -> MLXArray {
    let result = MLX.stacked(arrays, axis: axis)
    MLX.eval(result)
    MLX.Memory.clearCache()
    return result
}
