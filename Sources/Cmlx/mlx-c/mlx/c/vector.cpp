/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/vector.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" mlx_vector_array mlx_vector_array_new(void) {
  try {
    return mlx_vector_array_new_({});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_array_new_();
  }
}

extern "C" int mlx_vector_array_set(
    mlx_vector_array* vec,
    const mlx_vector_array src) {
  try {
    mlx_vector_array_set_(*vec, mlx_vector_array_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_array_free(mlx_vector_array vec) {
  try {
    mlx_vector_array_free_(vec);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_vector_array mlx_vector_array_new_data(
    const mlx_array* data,
    size_t size) {
  try {
    auto vec = mlx_vector_array_new();
    for (size_t i = 0; i < size; i++) {
      mlx_vector_array_get_(vec).push_back(mlx_array_get_(data[i]));
    }
    return vec;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_array_new_();
  }
}

extern "C" mlx_vector_array mlx_vector_array_new_value(const mlx_array val) {
  try {
    return mlx_vector_array_new_({mlx_array_get_(val)});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_array_new_();
  }
}

extern "C" int mlx_vector_array_set_data(
    mlx_vector_array* vec_,
    const mlx_array* data,
    size_t size) {
  try {
    std::vector<mlx::core::array> cpp_arrs;
    for (size_t i = 0; i < size; i++) {
      cpp_arrs.push_back(mlx_array_get_(data[i]));
    }
    mlx_vector_array_set_(*vec_, cpp_arrs);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_array_set_value(
    mlx_vector_array* vec_,
    const mlx_array val) {
  try {
    mlx_vector_array_set_(
        *vec_, std::vector<mlx::core::array>({mlx_array_get_(val)}));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_array_append_data(
    mlx_vector_array vec,
    const mlx_array* data,
    size_t size) {
  try {
    for (size_t i = 0; i < size; i++) {
      mlx_vector_array_get_(vec).push_back(mlx_array_get_(data[i]));
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_array_append_value(
    mlx_vector_array vec,
    const mlx_array value) {
  try {
    mlx_vector_array_get_(vec).push_back(mlx_array_get_(value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int
mlx_vector_array_get(mlx_array* res, const mlx_vector_array vec, size_t index) {
  try {
    mlx_array_set_(*res, mlx_vector_array_get_(vec).at(index));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" size_t mlx_vector_array_size(mlx_vector_array vec) {
  try {
    return mlx_vector_array_get_(vec).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" mlx_vector_vector_array mlx_vector_vector_array_new(void) {
  try {
    return mlx_vector_vector_array_new_({});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_vector_array_new_();
  }
}

extern "C" int mlx_vector_vector_array_set(
    mlx_vector_vector_array* vec,
    const mlx_vector_vector_array src) {
  try {
    mlx_vector_vector_array_set_(*vec, mlx_vector_vector_array_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_vector_array_free(mlx_vector_vector_array vec) {
  try {
    mlx_vector_vector_array_free_(vec);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_vector_vector_array mlx_vector_vector_array_new_data(
    const mlx_vector_array* data,
    size_t size) {
  try {
    auto vec = mlx_vector_vector_array_new();
    for (size_t i = 0; i < size; i++) {
      mlx_vector_vector_array_get_(vec).push_back(
          mlx_vector_array_get_(data[i]));
    }
    return vec;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_vector_array_new_();
  }
}

extern "C" mlx_vector_vector_array mlx_vector_vector_array_new_value(
    const mlx_vector_array val) {
  try {
    return mlx_vector_vector_array_new_({mlx_vector_array_get_(val)});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_vector_array_new_();
  }
}

extern "C" int mlx_vector_vector_array_set_data(
    mlx_vector_vector_array* vec_,
    const mlx_vector_array* data,
    size_t size) {
  try {
    std::vector<std::vector<mlx::core::array>> cpp_arrs;
    for (size_t i = 0; i < size; i++) {
      cpp_arrs.push_back(mlx_vector_array_get_(data[i]));
    }
    mlx_vector_vector_array_set_(*vec_, cpp_arrs);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_vector_array_set_value(
    mlx_vector_vector_array* vec_,
    const mlx_vector_array val) {
  try {
    mlx_vector_vector_array_set_(
        *vec_,
        std::vector<std::vector<mlx::core::array>>(
            {mlx_vector_array_get_(val)}));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_vector_array_append_data(
    mlx_vector_vector_array vec,
    const mlx_vector_array* data,
    size_t size) {
  try {
    for (size_t i = 0; i < size; i++) {
      mlx_vector_vector_array_get_(vec).push_back(
          mlx_vector_array_get_(data[i]));
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_vector_array_append_value(
    mlx_vector_vector_array vec,
    const mlx_vector_array value) {
  try {
    mlx_vector_vector_array_get_(vec).push_back(mlx_vector_array_get_(value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_vector_array_get(
    mlx_vector_array* res,
    const mlx_vector_vector_array vec,
    size_t index) {
  try {
    mlx_vector_array_set_(*res, mlx_vector_vector_array_get_(vec).at(index));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" size_t mlx_vector_vector_array_size(mlx_vector_vector_array vec) {
  try {
    return mlx_vector_vector_array_get_(vec).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" mlx_vector_int mlx_vector_int_new(void) {
  try {
    return mlx_vector_int_new_({});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_int_new_();
  }
}

extern "C" int mlx_vector_int_set(
    mlx_vector_int* vec,
    const mlx_vector_int src) {
  try {
    mlx_vector_int_set_(*vec, mlx_vector_int_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_int_free(mlx_vector_int vec) {
  try {
    mlx_vector_int_free_(vec);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_vector_int mlx_vector_int_new_data(int* data, size_t size) {
  try {
    auto vec = mlx_vector_int_new();
    for (size_t i = 0; i < size; i++) {
      mlx_vector_int_get_(vec).push_back(data[i]);
    }
    return vec;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_int_new_();
  }
}

extern "C" mlx_vector_int mlx_vector_int_new_value(int val) {
  try {
    return mlx_vector_int_new_({val});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_int_new_();
  }
}

extern "C" int
mlx_vector_int_set_data(mlx_vector_int* vec_, int* data, size_t size) {
  try {
    std::vector<int> cpp_arrs;
    for (size_t i = 0; i < size; i++) {
      cpp_arrs.push_back(data[i]);
    }
    mlx_vector_int_set_(*vec_, cpp_arrs);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_int_set_value(mlx_vector_int* vec_, int val) {
  try {
    mlx_vector_int_set_(*vec_, std::vector<int>({val}));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int
mlx_vector_int_append_data(mlx_vector_int vec, int* data, size_t size) {
  try {
    for (size_t i = 0; i < size; i++) {
      mlx_vector_int_get_(vec).push_back(data[i]);
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_int_append_value(mlx_vector_int vec, int value) {
  try {
    mlx_vector_int_get_(vec).push_back(value);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int
mlx_vector_int_get(int* res, const mlx_vector_int vec, size_t index) {
  try {
    *res = mlx_vector_int_get_(vec).at(index);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" size_t mlx_vector_int_size(mlx_vector_int vec) {
  try {
    return mlx_vector_int_get_(vec).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" mlx_vector_string mlx_vector_string_new(void) {
  try {
    return mlx_vector_string_new_({});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_string_new_();
  }
}

extern "C" int mlx_vector_string_set(
    mlx_vector_string* vec,
    const mlx_vector_string src) {
  try {
    mlx_vector_string_set_(*vec, mlx_vector_string_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_string_free(mlx_vector_string vec) {
  try {
    mlx_vector_string_free_(vec);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_vector_string mlx_vector_string_new_data(
    const char** data,
    size_t size) {
  try {
    auto vec = mlx_vector_string_new();
    for (size_t i = 0; i < size; i++) {
      mlx_vector_string_get_(vec).push_back(data[i]);
    }
    return vec;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_string_new_();
  }
}

extern "C" mlx_vector_string mlx_vector_string_new_value(const char* val) {
  try {
    return mlx_vector_string_new_({val});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_string_new_();
  }
}

extern "C" int mlx_vector_string_set_data(
    mlx_vector_string* vec_,
    const char** data,
    size_t size) {
  try {
    std::vector<std::string> cpp_arrs;
    for (size_t i = 0; i < size; i++) {
      cpp_arrs.push_back(data[i]);
    }
    mlx_vector_string_set_(*vec_, cpp_arrs);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_string_set_value(
    mlx_vector_string* vec_,
    const char* val) {
  try {
    mlx_vector_string_set_(*vec_, std::vector<std::string>({val}));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_string_append_data(
    mlx_vector_string vec,
    const char** data,
    size_t size) {
  try {
    for (size_t i = 0; i < size; i++) {
      mlx_vector_string_get_(vec).push_back(data[i]);
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_string_append_value(
    mlx_vector_string vec,
    const char* value) {
  try {
    mlx_vector_string_get_(vec).push_back(value);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int
mlx_vector_string_get(char** res, const mlx_vector_string vec, size_t index) {
  try {
    *res = mlx_vector_string_get_(vec).at(index).data();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" size_t mlx_vector_string_size(mlx_vector_string vec) {
  try {
    return mlx_vector_string_get_(vec).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
