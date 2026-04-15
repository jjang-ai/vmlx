// Dynamic slice operations with MLXArray start positions.
// These use DynamicSlice/DynamicSliceUpdate primitives which support
// compile() tracing (the start positions are part of the computation graph).

import Cmlx
import Foundation
import MLX

/// Read a slice of `src` starting at dynamic `start` positions on given `axes`.
///
/// Unlike the subscript `src[from..<to]` which uses Int indices (baked into
/// the graph as constants), this takes an MLXArray start position so the
/// compile tracer can track it through the graph.
///
/// - Parameters:
///   - src: Source array
///   - start: MLXArray with start indices (one per axis in `axes`)
///   - axes: Which axes to slice along
///   - sliceSize: Size of the slice on each axis
///   - stream: Stream for the operation
/// - Returns: Sliced array
public func dynamicSlice(
    _ src: MLXArray,
    start: MLXArray,
    axes: [Int32],
    sliceSize: [Int32],
    stream: StreamOrDevice = .default
) -> MLXArray {
    // Use the C++ level slice(array, array_start, axes, slice_size) overload.
    // This creates a DynamicSlice primitive which has output_shapes support.
    var result = mlx_array_new()
    var axesInt = axes.map { Int32($0) }
    var sizes = sliceSize.map { Int32($0) }
    let rc = mlx_slice_dynamic(
        &result,
        src.ctx,
        start.ctx,
        &axesInt,
        axesInt.count,
        &sizes,
        sizes.count,
        stream.ctx)
    if rc != 0 {
        fatalError("[dynamicSlice] mlx_slice_dynamic failed with rc=\(rc)")
    }
    return MLXArray(result)
}

/// Update a slice of `src` at dynamic `start` positions on given `axes`.
public func dynamicSliceUpdate(
    _ src: MLXArray,
    update: MLXArray,
    start: MLXArray,
    axes: [Int32],
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    var axesInt = axes.map { Int32($0) }
    let rc = mlx_slice_update_dynamic(
        &result,
        src.ctx,
        update.ctx,
        start.ctx,
        &axesInt,
        axesInt.count,
        stream.ctx)
    if rc != 0 {
        fatalError("[dynamicSliceUpdate] mlx_slice_update_dynamic failed with rc=\(rc)")
    }
    return MLXArray(result)
}
