/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_STRING_PRIVATE_H
#define MLX_STRING_PRIVATE_H

#include "mlx/c/string.h"
#include "mlx/mlx.h"

inline mlx_string mlx_string_new_() {
  return mlx_string({nullptr});
}

inline mlx_string mlx_string_new_(const std::string& s) {
  return mlx_string({new std::string(s)});
}

inline mlx_string mlx_string_new_(std::string&& s) {
  return mlx_string({new std::string(std::move(s))});
}

inline mlx_string& mlx_string_set_(mlx_string& d, const std::string& s) {
  if (d.ctx) {
    *static_cast<std::string*>(d.ctx) = s;
  } else {
    d.ctx = new std::string(s);
  }
  return d;
}

inline mlx_string& mlx_string_set_(mlx_string& d, std::string&& s) {
  if (d.ctx) {
    *static_cast<std::string*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new std::string(std::move(s));
  }
  return d;
}

inline std::string& mlx_string_get_(mlx_string d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_string");
  }
  return *static_cast<std::string*>(d.ctx);
}

inline void mlx_string_free_(mlx_string d) {
  if (d.ctx) {
    delete static_cast<std::string*>(d.ctx);
  }
}

#endif
