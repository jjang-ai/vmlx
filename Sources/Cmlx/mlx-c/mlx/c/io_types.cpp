/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>

#include "mlx/c/device.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" mlx_io_reader mlx_io_reader_new(void* desc, mlx_io_vtable vtable) {
  try {
    return mlx_io_reader_new_(desc, vtable);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_io_reader({nullptr});
  }
}

extern "C" int mlx_io_reader_free(mlx_io_reader io) {
  try {
    mlx_io_reader_free_(io);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_reader_descriptor(void** desc_, mlx_io_reader io) {
  try {
    *desc_ = mlx_io_reader_get_(io)->desc;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_reader_tostring(mlx_string* str_, mlx_io_reader io) {
  try {
    mlx_string_set_(*str_, mlx_io_reader_get_(io)->label());
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_io_writer mlx_io_writer_new(void* desc, mlx_io_vtable vtable) {
  try {
    return mlx_io_writer_new_(desc, vtable);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_io_writer({nullptr});
  }
}

extern "C" int mlx_io_writer_free(mlx_io_writer io) {
  try {
    mlx_io_writer_free_(io);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_writer_descriptor(void** desc_, mlx_io_writer io) {
  try {
    *desc_ = mlx_io_writer_get_(io)->desc;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_writer_tostring(mlx_string* str_, mlx_io_writer io) {
  try {
    mlx_string_set_(*str_, mlx_io_writer_get_(io)->label());
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
