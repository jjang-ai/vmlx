/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/fft.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/fft.h"

extern "C" int mlx_fft_fft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::fft(mlx_array_get_(a), n, axis, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_fft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::fft2(
            mlx_array_get_(a),
            mlx::core::Shape(n, n + n_num),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_fftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::fftn(
            mlx_array_get_(a),
            mlx::core::Shape(n, n + n_num),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_fftshift(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::fftshift(
            mlx_array_get_(a),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_ifft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::ifft(mlx_array_get_(a), n, axis, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_ifft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::ifft2(
            mlx_array_get_(a),
            mlx::core::Shape(n, n + n_num),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_ifftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::ifftn(
            mlx_array_get_(a),
            mlx::core::Shape(n, n + n_num),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_ifftshift(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::ifftshift(
            mlx_array_get_(a),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_irfft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::irfft(mlx_array_get_(a), n, axis, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_irfft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::irfft2(
            mlx_array_get_(a),
            mlx::core::Shape(n, n + n_num),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_irfftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::irfftn(
            mlx_array_get_(a),
            mlx::core::Shape(n, n + n_num),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_rfft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::rfft(mlx_array_get_(a), n, axis, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_rfft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::rfft2(
            mlx_array_get_(a),
            mlx::core::Shape(n, n + n_num),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fft_rfftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fft::rfftn(
            mlx_array_get_(a),
            mlx::core::Shape(n, n + n_num),
            std::vector<int>(axes, axes + axes_num),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
