/* Copyright © 2023-2024 Apple Inc. */

#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_version(mlx_string* str_) {
  try {
    mlx_string_set_(*str_, mlx::core::version());
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
