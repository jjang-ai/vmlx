/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_VECTOR_PRIVATE_H
#define MLX_VECTOR_PRIVATE_H

#include "mlx/c/vector.h"
#include "mlx/mlx.h"

inline mlx_vector_array mlx_vector_array_new_() {
  return mlx_vector_array({nullptr});
}

inline mlx_vector_array mlx_vector_array_new_(
    const std::vector<mlx::core::array>& s) {
  return mlx_vector_array({new std::vector<mlx::core::array>(s)});
}

inline mlx_vector_array mlx_vector_array_new_(
    std::vector<mlx::core::array>&& s) {
  return mlx_vector_array({new std::vector<mlx::core::array>(std::move(s))});
}

inline mlx_vector_array& mlx_vector_array_set_(
    mlx_vector_array& d,
    const std::vector<mlx::core::array>& s) {
  if (d.ctx) {
    *static_cast<std::vector<mlx::core::array>*>(d.ctx) = s;
  } else {
    d.ctx = new std::vector<mlx::core::array>(s);
  }
  return d;
}

inline mlx_vector_array& mlx_vector_array_set_(
    mlx_vector_array& d,
    std::vector<mlx::core::array>&& s) {
  if (d.ctx) {
    *static_cast<std::vector<mlx::core::array>*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new std::vector<mlx::core::array>(std::move(s));
  }
  return d;
}

inline std::vector<mlx::core::array>& mlx_vector_array_get_(
    mlx_vector_array d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_vector_array");
  }
  return *static_cast<std::vector<mlx::core::array>*>(d.ctx);
}

inline void mlx_vector_array_free_(mlx_vector_array d) {
  if (d.ctx) {
    delete static_cast<std::vector<mlx::core::array>*>(d.ctx);
  }
}

inline mlx_vector_vector_array mlx_vector_vector_array_new_() {
  return mlx_vector_vector_array({nullptr});
}

inline mlx_vector_vector_array mlx_vector_vector_array_new_(
    const std::vector<std::vector<mlx::core::array>>& s) {
  return mlx_vector_vector_array(
      {new std::vector<std::vector<mlx::core::array>>(s)});
}

inline mlx_vector_vector_array mlx_vector_vector_array_new_(
    std::vector<std::vector<mlx::core::array>>&& s) {
  return mlx_vector_vector_array(
      {new std::vector<std::vector<mlx::core::array>>(std::move(s))});
}

inline mlx_vector_vector_array& mlx_vector_vector_array_set_(
    mlx_vector_vector_array& d,
    const std::vector<std::vector<mlx::core::array>>& s) {
  if (d.ctx) {
    *static_cast<std::vector<std::vector<mlx::core::array>>*>(d.ctx) = s;
  } else {
    d.ctx = new std::vector<std::vector<mlx::core::array>>(s);
  }
  return d;
}

inline mlx_vector_vector_array& mlx_vector_vector_array_set_(
    mlx_vector_vector_array& d,
    std::vector<std::vector<mlx::core::array>>&& s) {
  if (d.ctx) {
    *static_cast<std::vector<std::vector<mlx::core::array>>*>(d.ctx) =
        std::move(s);
  } else {
    d.ctx = new std::vector<std::vector<mlx::core::array>>(std::move(s));
  }
  return d;
}

inline std::vector<std::vector<mlx::core::array>>& mlx_vector_vector_array_get_(
    mlx_vector_vector_array d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_vector_vector_array");
  }
  return *static_cast<std::vector<std::vector<mlx::core::array>>*>(d.ctx);
}

inline void mlx_vector_vector_array_free_(mlx_vector_vector_array d) {
  if (d.ctx) {
    delete static_cast<std::vector<std::vector<mlx::core::array>>*>(d.ctx);
  }
}

inline mlx_vector_int mlx_vector_int_new_() {
  return mlx_vector_int({nullptr});
}

inline mlx_vector_int mlx_vector_int_new_(const std::vector<int>& s) {
  return mlx_vector_int({new std::vector<int>(s)});
}

inline mlx_vector_int mlx_vector_int_new_(std::vector<int>&& s) {
  return mlx_vector_int({new std::vector<int>(std::move(s))});
}

inline mlx_vector_int& mlx_vector_int_set_(
    mlx_vector_int& d,
    const std::vector<int>& s) {
  if (d.ctx) {
    *static_cast<std::vector<int>*>(d.ctx) = s;
  } else {
    d.ctx = new std::vector<int>(s);
  }
  return d;
}

inline mlx_vector_int& mlx_vector_int_set_(
    mlx_vector_int& d,
    std::vector<int>&& s) {
  if (d.ctx) {
    *static_cast<std::vector<int>*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new std::vector<int>(std::move(s));
  }
  return d;
}

inline std::vector<int>& mlx_vector_int_get_(mlx_vector_int d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_vector_int");
  }
  return *static_cast<std::vector<int>*>(d.ctx);
}

inline void mlx_vector_int_free_(mlx_vector_int d) {
  if (d.ctx) {
    delete static_cast<std::vector<int>*>(d.ctx);
  }
}

inline mlx_vector_string mlx_vector_string_new_() {
  return mlx_vector_string({nullptr});
}

inline mlx_vector_string mlx_vector_string_new_(
    const std::vector<std::string>& s) {
  return mlx_vector_string({new std::vector<std::string>(s)});
}

inline mlx_vector_string mlx_vector_string_new_(std::vector<std::string>&& s) {
  return mlx_vector_string({new std::vector<std::string>(std::move(s))});
}

inline mlx_vector_string& mlx_vector_string_set_(
    mlx_vector_string& d,
    const std::vector<std::string>& s) {
  if (d.ctx) {
    *static_cast<std::vector<std::string>*>(d.ctx) = s;
  } else {
    d.ctx = new std::vector<std::string>(s);
  }
  return d;
}

inline mlx_vector_string& mlx_vector_string_set_(
    mlx_vector_string& d,
    std::vector<std::string>&& s) {
  if (d.ctx) {
    *static_cast<std::vector<std::string>*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new std::vector<std::string>(std::move(s));
  }
  return d;
}

inline std::vector<std::string>& mlx_vector_string_get_(mlx_vector_string d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_vector_string");
  }
  return *static_cast<std::vector<std::string>*>(d.ctx);
}

inline void mlx_vector_string_free_(mlx_vector_string d) {
  if (d.ctx) {
    delete static_cast<std::vector<std::string>*>(d.ctx);
  }
}

#endif
