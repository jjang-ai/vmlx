/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_CLOSURE_PRIVATE_H
#define MLX_CLOSURE_PRIVATE_H

#include "mlx/c/closure.h"
#include "mlx/mlx.h"

inline mlx_closure mlx_closure_new_() {
  return mlx_closure({nullptr});
}

inline mlx_closure mlx_closure_new_(
    const std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>& s) {
  return mlx_closure({new std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>(s)});
}

inline mlx_closure mlx_closure_new_(
    std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>&& s) {
  return mlx_closure({new std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>(std::move(s))});
}

inline mlx_closure& mlx_closure_set_(
    mlx_closure& d,
    const std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>& s) {
  if (d.ctx) {
    *static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>*>(d.ctx) = s;
  } else {
    d.ctx = new std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>(s);
  }
  return d;
}

inline mlx_closure& mlx_closure_set_(
    mlx_closure& d,
    std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>&& s) {
  if (d.ctx) {
    *static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>(std::move(s));
  }
  return d;
}

inline std::function<
    std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>&
mlx_closure_get_(mlx_closure d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_closure");
  }
  return *static_cast<std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>*>(d.ctx);
}

inline void mlx_closure_free_(mlx_closure d) {
  if (d.ctx) {
    delete static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&)>*>(d.ctx);
  }
}

inline mlx_closure_kwargs mlx_closure_kwargs_new_() {
  return mlx_closure_kwargs({nullptr});
}

inline mlx_closure_kwargs mlx_closure_kwargs_new_(
    const std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>& s) {
  return mlx_closure_kwargs({new std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&,
      const std::unordered_map<std::string, mlx::core::array>&)>(s)});
}

inline mlx_closure_kwargs mlx_closure_kwargs_new_(
    std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>&& s) {
  return mlx_closure_kwargs({new std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&,
      const std::unordered_map<std::string, mlx::core::array>&)>(
      std::move(s))});
}

inline mlx_closure_kwargs& mlx_closure_kwargs_set_(
    mlx_closure_kwargs& d,
    const std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>& s) {
  if (d.ctx) {
    *static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>*>(d.ctx) = s;
  } else {
    d.ctx = new std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>(s);
  }
  return d;
}

inline mlx_closure_kwargs& mlx_closure_kwargs_set_(
    mlx_closure_kwargs& d,
    std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>&& s) {
  if (d.ctx) {
    *static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>*>(d.ctx) =
        std::move(s);
  } else {
    d.ctx = new std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>(
        std::move(s));
  }
  return d;
}

inline std::function<std::vector<mlx::core::array>(
    const std::vector<mlx::core::array>&,
    const std::unordered_map<std::string, mlx::core::array>&)>&
mlx_closure_kwargs_get_(mlx_closure_kwargs d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_closure_kwargs");
  }
  return *static_cast<std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&,
      const std::unordered_map<std::string, mlx::core::array>&)>*>(d.ctx);
}

inline void mlx_closure_kwargs_free_(mlx_closure_kwargs d) {
  if (d.ctx) {
    delete static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::unordered_map<std::string, mlx::core::array>&)>*>(d.ctx);
  }
}

inline mlx_closure_value_and_grad mlx_closure_value_and_grad_new_() {
  return mlx_closure_value_and_grad({nullptr});
}

inline mlx_closure_value_and_grad mlx_closure_value_and_grad_new_(
    const std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>& s) {
  return mlx_closure_value_and_grad({new std::function<
      std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
          const std::vector<mlx::core::array>&)>(s)});
}

inline mlx_closure_value_and_grad mlx_closure_value_and_grad_new_(
    std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>&& s) {
  return mlx_closure_value_and_grad({new std::function<
      std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
          const std::vector<mlx::core::array>&)>(std::move(s))});
}

inline mlx_closure_value_and_grad& mlx_closure_value_and_grad_set_(
    mlx_closure_value_and_grad& d,
    const std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>& s) {
  if (d.ctx) {
    *static_cast<std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>*>(d.ctx) = s;
  } else {
    d.ctx = new std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>(s);
  }
  return d;
}

inline mlx_closure_value_and_grad& mlx_closure_value_and_grad_set_(
    mlx_closure_value_and_grad& d,
    std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>&& s) {
  if (d.ctx) {
    *static_cast<std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>(std::move(s));
  }
  return d;
}

inline std::function<
    std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
        const std::vector<mlx::core::array>&)>&
mlx_closure_value_and_grad_get_(mlx_closure_value_and_grad d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_closure_value_and_grad");
  }
  return *static_cast<std::function<
      std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
          const std::vector<mlx::core::array>&)>*>(d.ctx);
}

inline void mlx_closure_value_and_grad_free_(mlx_closure_value_and_grad d) {
  if (d.ctx) {
    delete static_cast<std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>(
            const std::vector<mlx::core::array>&)>*>(d.ctx);
  }
}

inline mlx_closure_custom mlx_closure_custom_new_() {
  return mlx_closure_custom({nullptr});
}

inline mlx_closure_custom mlx_closure_custom_new_(
    const std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>& s) {
  return mlx_closure_custom({new std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&,
      const std::vector<mlx::core::array>&,
      const std::vector<mlx::core::array>&)>(s)});
}

inline mlx_closure_custom mlx_closure_custom_new_(
    std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>&& s) {
  return mlx_closure_custom({new std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&,
      const std::vector<mlx::core::array>&,
      const std::vector<mlx::core::array>&)>(std::move(s))});
}

inline mlx_closure_custom& mlx_closure_custom_set_(
    mlx_closure_custom& d,
    const std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>& s) {
  if (d.ctx) {
    *static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>*>(d.ctx) = s;
  } else {
    d.ctx = new std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>(s);
  }
  return d;
}

inline mlx_closure_custom& mlx_closure_custom_set_(
    mlx_closure_custom& d,
    std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>&& s) {
  if (d.ctx) {
    *static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>(std::move(s));
  }
  return d;
}

inline std::function<std::vector<mlx::core::array>(
    const std::vector<mlx::core::array>&,
    const std::vector<mlx::core::array>&,
    const std::vector<mlx::core::array>&)>&
mlx_closure_custom_get_(mlx_closure_custom d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_closure_custom");
  }
  return *static_cast<std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&,
      const std::vector<mlx::core::array>&,
      const std::vector<mlx::core::array>&)>*>(d.ctx);
}

inline void mlx_closure_custom_free_(mlx_closure_custom d) {
  if (d.ctx) {
    delete static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&)>*>(d.ctx);
  }
}

inline mlx_closure_custom_jvp mlx_closure_custom_jvp_new_() {
  return mlx_closure_custom_jvp({nullptr});
}

inline mlx_closure_custom_jvp mlx_closure_custom_jvp_new_(
    const std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>& s) {
  return mlx_closure_custom_jvp(
      {new std::function<std::vector<mlx::core::array>(
          const std::vector<mlx::core::array>&,
          const std::vector<mlx::core::array>&,
          const std::vector<int>&)>(s)});
}

inline mlx_closure_custom_jvp mlx_closure_custom_jvp_new_(
    std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>&& s) {
  return mlx_closure_custom_jvp(
      {new std::function<std::vector<mlx::core::array>(
          const std::vector<mlx::core::array>&,
          const std::vector<mlx::core::array>&,
          const std::vector<int>&)>(std::move(s))});
}

inline mlx_closure_custom_jvp& mlx_closure_custom_jvp_set_(
    mlx_closure_custom_jvp& d,
    const std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>& s) {
  if (d.ctx) {
    *static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>*>(d.ctx) = s;
  } else {
    d.ctx = new std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>(s);
  }
  return d;
}

inline mlx_closure_custom_jvp& mlx_closure_custom_jvp_set_(
    mlx_closure_custom_jvp& d,
    std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>&& s) {
  if (d.ctx) {
    *static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>(std::move(s));
  }
  return d;
}

inline std::function<std::vector<mlx::core::array>(
    const std::vector<mlx::core::array>&,
    const std::vector<mlx::core::array>&,
    const std::vector<int>&)>&
mlx_closure_custom_jvp_get_(mlx_closure_custom_jvp d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_closure_custom_jvp");
  }
  return *static_cast<std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&,
      const std::vector<mlx::core::array>&,
      const std::vector<int>&)>*>(d.ctx);
}

inline void mlx_closure_custom_jvp_free_(mlx_closure_custom_jvp d) {
  if (d.ctx) {
    delete static_cast<std::function<std::vector<mlx::core::array>(
        const std::vector<mlx::core::array>&,
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>*>(d.ctx);
  }
}

inline mlx_closure_custom_vmap mlx_closure_custom_vmap_new_() {
  return mlx_closure_custom_vmap({nullptr});
}

inline mlx_closure_custom_vmap mlx_closure_custom_vmap_new_(
    const std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<int>>(
            const std::vector<mlx::core::array>&,
            const std::vector<int>&)>& s) {
  return mlx_closure_custom_vmap({new std::function<
      std::pair<std::vector<mlx::core::array>, std::vector<int>>(
          const std::vector<mlx::core::array>&, const std::vector<int>&)>(s)});
}

inline mlx_closure_custom_vmap mlx_closure_custom_vmap_new_(
    std::function<std::pair<std::vector<mlx::core::array>, std::vector<int>>(
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>&& s) {
  return mlx_closure_custom_vmap({new std::function<
      std::pair<std::vector<mlx::core::array>, std::vector<int>>(
          const std::vector<mlx::core::array>&, const std::vector<int>&)>(
      std::move(s))});
}

inline mlx_closure_custom_vmap& mlx_closure_custom_vmap_set_(
    mlx_closure_custom_vmap& d,
    const std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<int>>(
            const std::vector<mlx::core::array>&,
            const std::vector<int>&)>& s) {
  if (d.ctx) {
    *static_cast<std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<int>>(
            const std::vector<mlx::core::array>&, const std::vector<int>&)>*>(
        d.ctx) = s;
  } else {
    d.ctx = new std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<int>>(
            const std::vector<mlx::core::array>&, const std::vector<int>&)>(s);
  }
  return d;
}

inline mlx_closure_custom_vmap& mlx_closure_custom_vmap_set_(
    mlx_closure_custom_vmap& d,
    std::function<std::pair<std::vector<mlx::core::array>, std::vector<int>>(
        const std::vector<mlx::core::array>&,
        const std::vector<int>&)>&& s) {
  if (d.ctx) {
    *static_cast<std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<int>>(
            const std::vector<mlx::core::array>&, const std::vector<int>&)>*>(
        d.ctx) = std::move(s);
  } else {
    d.ctx = new std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<int>>(
            const std::vector<mlx::core::array>&, const std::vector<int>&)>(
        std::move(s));
  }
  return d;
}

inline std::function<std::pair<std::vector<mlx::core::array>, std::vector<int>>(
    const std::vector<mlx::core::array>&,
    const std::vector<int>&)>&
mlx_closure_custom_vmap_get_(mlx_closure_custom_vmap d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_closure_custom_vmap");
  }
  return *static_cast<
      std::function<std::pair<std::vector<mlx::core::array>, std::vector<int>>(
          const std::vector<mlx::core::array>&, const std::vector<int>&)>*>(
      d.ctx);
}

inline void mlx_closure_custom_vmap_free_(mlx_closure_custom_vmap d) {
  if (d.ctx) {
    delete static_cast<std::function<
        std::pair<std::vector<mlx::core::array>, std::vector<int>>(
            const std::vector<mlx::core::array>&, const std::vector<int>&)>*>(
        d.ctx);
  }
}

#endif
