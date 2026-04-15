/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>

#include "mlx/c/distributed_group.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_distributed_group_rank(mlx_distributed_group group) {
  try {
    return mlx_distributed_group_get_(group).rank();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" int mlx_distributed_group_size(mlx_distributed_group group) {
  try {
    return mlx_distributed_group_get_(group).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" mlx_distributed_group
mlx_distributed_group_split(mlx_distributed_group group, int color, int key) {
  try {
    return mlx_distributed_group_new_(
        mlx_distributed_group_get_(group).split(color, key));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_distributed_group_new_();
  }
}

extern "C" bool mlx_distributed_is_available(const char* bk) {
  try {
    if (bk) {
      return mlx::core::distributed::is_available(bk);
    } else {
      return mlx::core::distributed::is_available();
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return false;
  }
}

extern "C" mlx_distributed_group mlx_distributed_init(
    bool strict,
    const char* bk) {
  try {
    if (bk) {
      return mlx_distributed_group_new_(
          mlx::core::distributed::init(strict, bk));
    } else {
      return mlx_distributed_group_new_(mlx::core::distributed::init(strict));
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_distributed_group_new_();
  }
}
