/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_ENUMS_PRIVATE_H
#define MLX_ENUMS_PRIVATE_H

#include "mlx/c/array.h"
#include "mlx/c/compile.h"
#include "mlx/mlx.h"

namespace {
inline mlx_compile_mode mlx_compile_mode_to_c(mlx::core::CompileMode type) {
  static mlx_compile_mode map[] = {
      MLX_COMPILE_MODE_DISABLED,
      MLX_COMPILE_MODE_NO_SIMPLIFY,
      MLX_COMPILE_MODE_NO_FUSE,
      MLX_COMPILE_MODE_ENABLED};
  return map[(int)type];
}
inline mlx::core::CompileMode mlx_compile_mode_to_cpp(mlx_compile_mode type) {
  static mlx::core::CompileMode map[] = {
      mlx::core::CompileMode::disabled,
      mlx::core::CompileMode::no_simplify,
      mlx::core::CompileMode::no_fuse,
      mlx::core::CompileMode::enabled};
  return map[(int)type];
}
inline mlx_dtype mlx_dtype_to_c(mlx::core::Dtype type) {
  static mlx_dtype map[] = {
      MLX_BOOL,
      MLX_UINT8,
      MLX_UINT16,
      MLX_UINT32,
      MLX_UINT64,
      MLX_INT8,
      MLX_INT16,
      MLX_INT32,
      MLX_INT64,
      MLX_FLOAT16,
      MLX_FLOAT32,
      MLX_FLOAT64,
      MLX_BFLOAT16,
      MLX_COMPLEX64,
  };
  return map[(int)type.val()];
}
inline mlx::core::Dtype mlx_dtype_to_cpp(mlx_dtype type) {
  static mlx::core::Dtype map[] = {
      mlx::core::bool_,
      mlx::core::uint8,
      mlx::core::uint16,
      mlx::core::uint32,
      mlx::core::uint64,
      mlx::core::int8,
      mlx::core::int16,
      mlx::core::int32,
      mlx::core::int64,
      mlx::core::float16,
      mlx::core::float32,
      mlx::core::float64,
      mlx::core::bfloat16,
      mlx::core::complex64,
  };
  return map[(int)type];
}
inline mlx_device_type mlx_device_type_to_c(
    mlx::core::Device::DeviceType type) {
  static mlx_device_type map[] = {MLX_CPU, MLX_GPU};
  return map[(int)type];
}
inline mlx::core::Device::DeviceType mlx_device_type_to_cpp(
    mlx_device_type type) {
  static mlx::core::Device::DeviceType map[] = {
      mlx::core::Device::DeviceType::cpu, mlx::core::Device::DeviceType::gpu};
  return map[(int)type];
}
} // namespace

#endif
