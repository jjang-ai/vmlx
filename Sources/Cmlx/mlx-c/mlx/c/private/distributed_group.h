/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_DISTRIBUTED_GROUP_PRIVATE_H
#define MLX_DISTRIBUTED_GROUP_PRIVATE_H

#include "mlx/c/distributed_group.h"
#include "mlx/mlx.h"

inline mlx_distributed_group mlx_distributed_group_new_() {
  return mlx_distributed_group({nullptr});
}

inline mlx_distributed_group mlx_distributed_group_new_(
    const mlx::core::distributed::Group& s) {
  return mlx_distributed_group({new mlx::core::distributed::Group(s)});
}

inline mlx_distributed_group mlx_distributed_group_new_(
    mlx::core::distributed::Group&& s) {
  return mlx_distributed_group(
      {new mlx::core::distributed::Group(std::move(s))});
}

inline mlx_distributed_group& mlx_distributed_group_set_(
    mlx_distributed_group& d,
    const mlx::core::distributed::Group& s) {
  if (d.ctx) {
    *static_cast<mlx::core::distributed::Group*>(d.ctx) = s;
  } else {
    d.ctx = new mlx::core::distributed::Group(s);
  }
  return d;
}

inline mlx_distributed_group& mlx_distributed_group_set_(
    mlx_distributed_group& d,
    mlx::core::distributed::Group&& s) {
  if (d.ctx) {
    *static_cast<mlx::core::distributed::Group*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new mlx::core::distributed::Group(std::move(s));
  }
  return d;
}

inline mlx::core::distributed::Group& mlx_distributed_group_get_(
    mlx_distributed_group d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_distributed_group");
  }
  return *static_cast<mlx::core::distributed::Group*>(d.ctx);
}

inline void mlx_distributed_group_free_(mlx_distributed_group d) {
  if (d.ctx) {
    delete static_cast<mlx::core::distributed::Group*>(d.ctx);
  }
}

#endif
