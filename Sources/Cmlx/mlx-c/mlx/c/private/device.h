/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_DEVICE_PRIVATE_H
#define MLX_DEVICE_PRIVATE_H

#include "mlx/c/device.h"
#include "mlx/mlx.h"

inline mlx_device mlx_device_new_() {
  return mlx_device({nullptr});
}

inline mlx_device mlx_device_new_(const mlx::core::Device& s) {
  return mlx_device({new mlx::core::Device(s)});
}

inline mlx_device mlx_device_new_(mlx::core::Device&& s) {
  return mlx_device({new mlx::core::Device(std::move(s))});
}

inline mlx_device& mlx_device_set_(mlx_device& d, const mlx::core::Device& s) {
  if (d.ctx) {
    *static_cast<mlx::core::Device*>(d.ctx) = s;
  } else {
    d.ctx = new mlx::core::Device(s);
  }
  return d;
}

inline mlx_device& mlx_device_set_(mlx_device& d, mlx::core::Device&& s) {
  if (d.ctx) {
    *static_cast<mlx::core::Device*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new mlx::core::Device(std::move(s));
  }
  return d;
}

inline mlx::core::Device& mlx_device_get_(mlx_device d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_device");
  }
  return *static_cast<mlx::core::Device*>(d.ctx);
}

inline void mlx_device_free_(mlx_device d) {
  if (d.ctx) {
    delete static_cast<mlx::core::Device*>(d.ctx);
  }
}

using mlx_device_info_cpp =
    std::unordered_map<std::string, std::variant<std::string, size_t>>;

inline mlx_device_info mlx_device_info_new_() {
  return mlx_device_info({nullptr});
}

inline mlx_device_info mlx_device_info_new_(const mlx_device_info_cpp& s) {
  return mlx_device_info({new mlx_device_info_cpp(s)});
}

inline mlx_device_info mlx_device_info_new_(mlx_device_info_cpp&& s) {
  return mlx_device_info({new mlx_device_info_cpp(std::move(s))});
}

inline mlx_device_info& mlx_device_info_set_(
    mlx_device_info& d,
    const mlx_device_info_cpp& s) {
  if (d.ctx) {
    *static_cast<mlx_device_info_cpp*>(d.ctx) = s;
  } else {
    d.ctx = new mlx_device_info_cpp(s);
  }
  return d;
}

inline mlx_device_info& mlx_device_info_set_(
    mlx_device_info& d,
    mlx_device_info_cpp&& s) {
  if (d.ctx) {
    *static_cast<mlx_device_info_cpp*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new mlx_device_info_cpp(std::move(s));
  }
  return d;
}

inline mlx_device_info_cpp& mlx_device_info_get_(mlx_device_info d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_device_info");
  }
  return *static_cast<mlx_device_info_cpp*>(d.ctx);
}

inline void mlx_device_info_free_(mlx_device_info d) {
  if (d.ctx) {
    delete static_cast<mlx_device_info_cpp*>(d.ctx);
  }
}

#endif
