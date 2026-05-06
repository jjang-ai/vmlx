// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressCanonicalMmapAdvisor — Swift bridge to the canonical MLX
// safetensors mmap registry.
//
// The auxiliary JangPressMmapTier maps model files independently and can
// advise those parallel views. Once Engine.load enables
// VMLX_MMAP_SAFETENSORS, however, the real model tensors are canonical
// MLX no-copy buffers backed by mmap'd safetensors ranges. This bridge
// sends router-aware WILLNEED/DONTNEED advice to those canonical ranges.

import Cmlx
import Foundation

public enum JangPressCanonicalMmapAdvisor {
    @discardableResult
    public static func adviseRouted(
        _ advice: JangPressAdvice,
        percent: Int
    ) -> Int64 {
        let clamped = Int32(max(0, min(100, percent)))
        return mlx_safetensors_mmap_advise_routed(advice.rawValue, clamped)
    }

    @discardableResult
    public static func adviseExperts(
        _ advice: JangPressAdvice,
        pairs: [(layer: Int, expert: Int)]
    ) -> Int64 {
        guard !pairs.isEmpty else { return 0 }

        var layers = [Int32]()
        var experts = [Int32]()
        layers.reserveCapacity(pairs.count)
        experts.reserveCapacity(pairs.count)

        for pair in pairs {
            guard pair.layer >= 0, pair.expert >= 0 else { continue }
            guard pair.layer <= Int(Int32.max),
                  pair.expert <= Int(Int32.max)
            else { continue }
            layers.append(Int32(pair.layer))
            experts.append(Int32(pair.expert))
        }
        guard !layers.isEmpty else { return 0 }

        return layers.withUnsafeBufferPointer { layerBuf in
            experts.withUnsafeBufferPointer { expertBuf in
                guard let layerBase = layerBuf.baseAddress,
                      let expertBase = expertBuf.baseAddress
                else { return 0 }
                return mlx_safetensors_mmap_advise_experts(
                    advice.rawValue,
                    layerBase,
                    expertBase,
                    Int64(layers.count))
            }
        }
    }
}
