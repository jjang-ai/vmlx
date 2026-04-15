/* Copyright © 2023-2024 Apple Inc. */

#include "mlx/c/error.h"

#include <memory>
#include <vector>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

static void mlx_error_handler_default_(const char* msg, void* data) {
  printf("MLX error: %s\n", msg);
  exit(-1);
}

static std::shared_ptr<void> mlx_error_handler_data_ = nullptr;
static mlx_error_handler_func mlx_error_handler_ = mlx_error_handler_default_;

extern "C" void mlx_set_error_handler(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*)) {
  if (dtor) {
    mlx_error_handler_data_ = std::shared_ptr<void>(data, dtor);
  } else {
    mlx_error_handler_data_ = nullptr;
  }
  if (handler) {
    mlx_error_handler_ = handler;
  } else {
    mlx_error_handler_ = mlx_error_handler_default_;
  }
}

extern "C" void
_mlx_error(const char* file, const int line, const char* fmt, ...) {
  va_list args, args_copy;
  va_start(args, fmt);

  // compute total size
  va_copy(args_copy, args);
  int size = vsnprintf(nullptr, 0, fmt, args_copy);
  va_end(args_copy);
  int size_loc = snprintf(nullptr, 0, " at %s:%d", file, line);

  std::vector<char> msg(size + size_loc + 1); // \0 at the end
  size = vsnprintf(msg.data(), size + 1, fmt, args);
  snprintf(msg.data() + size, size_loc + 1, " at %s:%d", file, line);
  va_end(args);

  mlx_error_handler_(msg.data(), mlx_error_handler_data_.get());
}
