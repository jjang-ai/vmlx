//
//  InterpolationUtils.swift
//  mlx-swift-lm
//
//  Bicubic and Nearest interpolation using Metal kernels.
//  Port of the Python MLX interpolation kernels.
//  https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/kernels.py
//

import Foundation
import MLX

// MARK: - Kernel Sources

private func makeBicubicInterpolationKernel() -> MLXFast.MLXFastKernel {
    let header = """
        // Bicubic kernel function
        float cubic_kernel(float x) {
            float absx = fabs(x);
            float absx2 = absx * absx;
            float absx3 = absx2 * absx;

            const float a = -0.5f;

            if (absx <= 1.0f) {
                return (a + 2.0f) * absx3 - (a + 3.0f) * absx2 + 1.0f;
            } else if (absx < 2.0f) {
                return a * absx3 - 5.0f * a * absx2 + 8.0f * a * absx - 4.0f * a;
            }
            return 0.0f;
        }

        // Antialiased bicubic kernel - scales the support region for downsampling
        float cubic_kernel_antialias(float x, float scale) {
            // When downsampling, we need to integrate over a wider region
            // This matches PyTorch's antialiasing behavior
            return cubic_kernel(x / scale);
        }
        """

    let source = """
        // Get thread position
        uint x_out = thread_position_in_grid.x;
        uint y_out = thread_position_in_grid.y;
        uint bc_idx = thread_position_in_grid.z;

        // Extract dimensions
        int batch_size = dims[0];
        int channels = dims[1];
        int in_h = dims[2];
        int in_w = dims[3];
        int out_h = dims[4];
        int out_w = dims[5];

        // Extract parameters
        float scale_h = params[0];
        float scale_w = params[1];
        bool align_corners = params[2] > 0.5f;
        bool use_antialias = params[3] > 0.5f;
        float filter_scale_h = params[4];
        float filter_scale_w = params[5];
        float support = params[6];

        // Check bounds
        if (x_out >= (uint)out_w || y_out >= (uint)out_h || bc_idx >= (uint)(batch_size * channels))
            return;

        // Calculate batch and channel indices
        int c = bc_idx % channels;
        int b = bc_idx / channels;

        // Calculate input coordinates
        float x_in, y_in;

        if (align_corners && out_w > 1 && out_h > 1) {
            x_in = float(x_out) * (in_w - 1) / (out_w - 1);
            y_in = float(y_out) * (in_h - 1) / (out_h - 1);
        } else {
            // PyTorch's default coordinate mapping
            x_in = ((float(x_out) + 0.5f) / float(out_w)) * float(in_w) - 0.5f;
            y_in = ((float(y_out) + 0.5f) / float(out_h)) * float(in_h) - 0.5f;
        }

        // Calculate the support region based on antialiasing
        float support_h = use_antialias ? support * filter_scale_h : support;
        float support_w = use_antialias ? support * filter_scale_w : support;

        // Calculate the range of input pixels to sample
        int y_start = int(floor(y_in - support_h)) + 1;
        int y_end = int(floor(y_in + support_h)) + 1;
        int x_start = int(floor(x_in - support_w)) + 1;
        int x_end = int(floor(x_in + support_w)) + 1;

        // Clamp to valid range
        y_start = max(0, y_start);
        y_end = min(in_h, y_end);
        x_start = max(0, x_start);
        x_end = min(in_w, x_end);

        // Perform bicubic interpolation with antialiasing
        float result = 0.0f;
        float weight_sum = 0.0f;

        for (int y_pos = y_start; y_pos < y_end; y_pos++) {
            float dy = float(y_pos) - y_in;
            float wy = use_antialias ?
                cubic_kernel_antialias(dy, filter_scale_h) :
                cubic_kernel(dy);

            for (int x_pos = x_start; x_pos < x_end; x_pos++) {
                float dx = float(x_pos) - x_in;
                float wx = use_antialias ?
                    cubic_kernel_antialias(dx, filter_scale_w) :
                    cubic_kernel(dx);

                float weight = wy * wx;

                // Calculate input tensor offset
                int input_offset = ((b * channels + c) * in_h + y_pos) * in_w + x_pos;

                // Add weighted contribution
                result += input[input_offset] * weight;
                weight_sum += weight;
            }
        }

        // Normalize by weight sum
        if (weight_sum > 1e-8f) {
            result /= weight_sum;
        }

        // Calculate output tensor offset
        int output_offset = ((b * channels + c) * out_h + y_out) * out_w + x_out;

        // Assign the result to output
        output[output_offset] = result;
        """

    return MLXFast.metalKernel(
        name: "bicubic_interpolation_antialias",
        inputNames: ["input", "dims", "params"],
        outputNames: ["output"],
        source: source,
        header: header
    )
}

private func makeNearestInterpolationKernel() -> MLXFast.MLXFastKernel {
    let source = """
        uint x_out = thread_position_in_grid.x;
        uint y_out = thread_position_in_grid.y;
        uint bc_idx = thread_position_in_grid.z;

        int batch_size = dims[0];
        int channels = dims[1];
        int in_h = dims[2];
        int in_w = dims[3];
        int out_h = dims[4];
        int out_w = dims[5];

        if (x_out >= (uint)out_w || y_out >= (uint)out_h || bc_idx >= (uint)(batch_size * channels))
            return;

        int c = bc_idx % channels;
        int b = bc_idx / channels;

        // PyTorch's coordinate calculation for nearest neighbor
        // This matches: torch.nn.functional.interpolate(..., mode='nearest')
        float scale_h = float(in_h) / float(out_h);
        float scale_w = float(in_w) / float(out_w);

        // PyTorch uses floor for nearest neighbor coordinate mapping
        int y_in = int(floor(float(y_out) * scale_h));
        int x_in = int(floor(float(x_out) * scale_w));

        // Clamp to bounds
        y_in = max(0, min(y_in, in_h - 1));
        x_in = max(0, min(x_in, in_w - 1));

        int input_offset = ((b * channels + c) * in_h + y_in) * in_w + x_in;
        int output_offset = ((b * channels + c) * out_h + y_out) * out_w + x_out;

        output[output_offset] = input[input_offset];
        """

    return MLXFast.metalKernel(
        name: "nearest_interpolation",
        inputNames: ["input", "dims"],
        outputNames: ["output"],
        source: source
    )
}

// MARK: - Kernel Manager

/// Manages Metal kernels for interpolation operations.
private final class InterpolationKernelManager: Sendable {
    static let shared = InterpolationKernelManager()

    let bicubicKernel: MLXFast.MLXFastKernel
    let nearestKernel: MLXFast.MLXFastKernel

    private init() {
        bicubicKernel = makeBicubicInterpolationKernel()
        nearestKernel = makeNearestInterpolationKernel()
    }
}

// MARK: - Helper Functions

/// Calculate optimal threadgroup dimensions based on output dimensions.
private func getOptimalThreadgroup(outW: Int, outH: Int) -> (Int, Int, Int) {
    // Maximum threadgroup size for most Metal GPUs
    let maxThreadsPerGroup = 1024
    let maxThreadsPerDim = 1024

    // Default threadgroup for 2D workloads
    let defaultThreadgroup = (32, 32, 1)

    // Don't create threadgroups larger than the work dimensions
    let maxWidth = min(maxThreadsPerDim, outW)
    let maxHeight = min(maxThreadsPerDim, outH)

    guard maxWidth > 0 && maxHeight > 0 else {
        return defaultThreadgroup
    }

    // Find largest power of 2 that fits within our dimensions
    var width = 1 << (Int.bitWidth - maxWidth.leadingZeroBitCount - 1)
    if width > maxWidth {
        width = width / 2
    }

    var height = 1 << (Int.bitWidth - maxHeight.leadingZeroBitCount - 1)
    if height > maxHeight {
        height = height / 2
    }

    // Ensure we don't exceed maximum threads per threadgroup
    while width * height > maxThreadsPerGroup {
        // Reduce the larger dimension first
        if width >= height {
            width = width / 2
        } else {
            height = height / 2
        }
    }

    // Ensure minimum size for efficiency
    width = max(8, width)
    height = max(8, height)

    return (width, height, 1)
}

// MARK: - Public Interpolation Functions

/// Bicubic interpolation using MLX Metal kernel.
///
/// - Parameters:
///   - x: Input tensor of shape [B, C, H, W]
///   - size: Target output size (outH, outW). Either this or scaleFactor must be provided.
///   - scaleFactor: Scale factor as a single value or tuple (scaleH, scaleW). Either this or size must be provided.
///   - alignCorners: Whether to align corners during interpolation.
///   - antialias: Whether to apply antialiasing (useful when downsampling).
/// - Returns: Interpolated tensor of shape [B, C, outH, outW]
public func bicubicInterpolate(
    _ x: MLXArray,
    size: (Int, Int)? = nil,
    scaleFactor: (Float, Float)? = nil,
    alignCorners: Bool = false,
    antialias: Bool = false
) -> MLXArray {
    // Get input dimensions
    precondition(x.ndim == 4, "Input must be 4D tensor [B, C, H, W]")
    let batchSize = x.dim(0)
    let channels = x.dim(1)
    let inH = x.dim(2)
    let inW = x.dim(3)

    // Calculate output dimensions
    let outH: Int
    let outW: Int
    let scaleH: Float
    let scaleW: Float

    if let size = size {
        outH = size.0
        outW = size.1
        scaleH = Float(outH) / Float(inH)
        scaleW = Float(outW) / Float(inW)
    } else if let scaleFactor = scaleFactor {
        scaleH = scaleFactor.0
        scaleW = scaleFactor.1
        outH = Int(Float(inH) * scaleH)
        outW = Int(Float(inW) * scaleW)
    } else {
        fatalError("Either size or scaleFactor must be specified")
    }

    // Calculate antialiasing parameters
    // PyTorch uses support = 2.0 for bicubic when antialiasing
    let support: Float = 2.0
    let antialiasFlag: Float = (antialias && (scaleH < 1.0 || scaleW < 1.0)) ? 1.0 : 0.0

    // When downsampling with antialias, PyTorch expands the filter support
    let filterScaleH: Float = (antialias && scaleH < 1.0) ? (1.0 / scaleH) : 1.0
    let filterScaleW: Float = (antialias && scaleW < 1.0) ? (1.0 / scaleW) : 1.0

    // Create parameters tensor
    let params = MLXArray(
        [
            scaleH, scaleW, alignCorners ? 1.0 : 0.0, antialiasFlag, filterScaleH, filterScaleW,
            support,
        ]
    ).asType(.float32)

    // Create dimensions tensor
    let dims = MLXArray([batchSize, channels, inH, inW, outH, outW]).asType(.int32)

    // Reshape input tensor to 1D for kernel processing
    var xFlat = x.reshaped(-1)

    // Convert to float32 for processing if needed
    let inputDtype = x.dtype
    if inputDtype != .float32 {
        xFlat = xFlat.asType(.float32)
    }

    // Get optimal threadgroup
    let threadgroup = getOptimalThreadgroup(outW: outW, outH: outH)

    // Run the kernel
    let outputs = InterpolationKernelManager.shared.bicubicKernel(
        [xFlat, dims, params],
        grid: (outW, outH, batchSize * channels),
        threadGroup: threadgroup,
        outputShapes: [[batchSize * channels * outH * outW]],
        outputDTypes: [.float32]
    )

    // Reshape output back to 4D tensor
    var result = outputs[0].reshaped(batchSize, channels, outH, outW)

    // Convert back to original dtype
    if inputDtype != .float32 {
        result = result.asType(inputDtype)
    }

    return result
}

/// Bicubic interpolation with a single scale factor.
///
/// - Parameters:
///   - x: Input tensor of shape [B, C, H, W]
///   - scaleFactor: Single scale factor applied to both dimensions.
///   - alignCorners: Whether to align corners during interpolation.
///   - antialias: Whether to apply antialiasing.
/// - Returns: Interpolated tensor of shape [B, C, outH, outW]
public func bicubicInterpolate(
    _ x: MLXArray,
    scaleFactor: Float,
    alignCorners: Bool = false,
    antialias: Bool = false
) -> MLXArray {
    bicubicInterpolate(
        x,
        scaleFactor: (scaleFactor, scaleFactor),
        alignCorners: alignCorners,
        antialias: antialias
    )
}

/// Nearest neighbor interpolation using MLX Metal kernel.
///
/// - Parameters:
///   - x: Input tensor of shape [B, C, H, W]
///   - size: Target output size (outH, outW). Either this or scaleFactor must be provided.
///   - scaleFactor: Scale factor as a single value or tuple (scaleH, scaleW). Either this or size must be provided.
/// - Returns: Interpolated tensor of shape [B, C, outH, outW]
public func nearestInterpolate(
    _ x: MLXArray,
    size: (Int, Int)? = nil,
    scaleFactor: (Float, Float)? = nil
) -> MLXArray {
    // Get input dimensions
    precondition(x.ndim == 4, "Input must be 4D tensor [B, C, H, W]")
    let batchSize = x.dim(0)
    let channels = x.dim(1)
    let inH = x.dim(2)
    let inW = x.dim(3)

    // Calculate output dimensions
    let outH: Int
    let outW: Int

    if let size = size {
        outH = size.0
        outW = size.1
    } else if let scaleFactor = scaleFactor {
        outH = Int(Float(inH) * scaleFactor.0)
        outW = Int(Float(inW) * scaleFactor.1)
    } else {
        fatalError("Either size or scaleFactor must be specified")
    }

    // Create dimensions tensor
    let dims = MLXArray([batchSize, channels, inH, inW, outH, outW]).asType(.int32)

    // Reshape input tensor to 1D for kernel processing
    var xFlat = x.reshaped(-1)

    // Convert to float32 for processing if needed
    let inputDtype = x.dtype
    if inputDtype != .float32 {
        xFlat = xFlat.asType(.float32)
    }

    // Get optimal threadgroup
    let threadgroup = getOptimalThreadgroup(outW: outW, outH: outH)

    // Run the kernel
    let outputs = InterpolationKernelManager.shared.nearestKernel(
        [xFlat, dims],
        grid: (outW, outH, batchSize * channels),
        threadGroup: threadgroup,
        outputShapes: [[batchSize * channels * outH * outW]],
        outputDTypes: [.float32]
    )

    // Reshape output back to 4D tensor
    var result = outputs[0].reshaped(batchSize, channels, outH, outW)

    // Convert back to original dtype
    if inputDtype != .float32 {
        result = result.asType(inputDtype)
    }

    return result
}

/// Nearest neighbor interpolation with a single scale factor.
///
/// - Parameters:
///   - x: Input tensor of shape [B, C, H, W]
///   - scaleFactor: Single scale factor applied to both dimensions.
/// - Returns: Interpolated tensor of shape [B, C, outH, outW]
public func nearestInterpolate(
    _ x: MLXArray,
    scaleFactor: Float
) -> MLXArray {
    nearestInterpolate(x, scaleFactor: (scaleFactor, scaleFactor))
}
