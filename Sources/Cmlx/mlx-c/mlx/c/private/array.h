/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_ARRAY_PRIVATE_H
#define MLX_ARRAY_PRIVATE_H

#include "mlx/c/array.h"
#include "mlx/mlx.h"

inline mlx_array mlx_array_new_() {
  return mlx_array({nullptr});
}

inline mlx_array mlx_array_new_(const mlx::core::array& s) {
  return mlx_array({new mlx::core::array(s)});
}

inline mlx_array mlx_array_new_(mlx::core::array&& s) {
  return mlx_array({new mlx::core::array(std::move(s))});
}

inline mlx_array& mlx_array_set_(mlx_array& d, const mlx::core::array& s) {
  if (d.ctx) {
    *static_cast<mlx::core::array*>(d.ctx) = s;
  } else {
    d.ctx = new mlx::core::array(s);
  }
  return d;
}

inline mlx_array& mlx_array_set_(mlx_array& d, mlx::core::array&& s) {
  if (d.ctx) {
    *static_cast<mlx::core::array*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new mlx::core::array(std::move(s));
  }
  return d;
}

inline mlx::core::array& mlx_array_get_(mlx_array d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_array");
  }
  return *static_cast<mlx::core::array*>(d.ctx);
}

inline void mlx_array_free_(mlx_array d) {
  if (d.ctx) {
    delete static_cast<mlx::core::array*>(d.ctx);
  }
}

#endif
