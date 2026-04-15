/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>

#include "mlx/c/device.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_device_tostring(mlx_string* str_, mlx_device dev) {
  try {
    std::ostringstream os;
    os << mlx_device_get_(dev);
    std::string str = os.str();
    mlx_string_set_(*str_, str);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_device mlx_device_new(void) {
  return mlx_device_new_();
}

extern "C" mlx_device mlx_device_new_type(mlx_device_type type, int index) {
  try {
    auto cpp_type = mlx_device_type_to_cpp(type);
    return mlx_device_new_(mlx::core::Device(cpp_type, index));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_device_new_();
  }
}

extern "C" int mlx_device_set(mlx_device* dev, const mlx_device src) {
  try {
    mlx_device_set_(*dev, mlx_device_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_device_get_index(int* index, mlx_device dev) {
  try {
    *index = mlx_device_get_(dev).index;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_device_get_type(mlx_device_type* type, mlx_device dev) {
  try {
    *type = mlx_device_type_to_c(mlx_device_get_(dev).type);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" bool mlx_device_equal(mlx_device lhs, mlx_device rhs) {
  return mlx_device_get_(lhs) == mlx_device_get_(rhs);
}

extern "C" int mlx_get_default_device(mlx_device* dev) {
  try {
    mlx_device_set_(*dev, mlx::core::default_device());
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_set_default_device(mlx_device dev) {
  try {
    mlx::core::set_default_device(mlx_device_get_(dev));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_device_free(mlx_device dev) {
  try {
    mlx_device_free_(dev);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_device_is_available(bool* avail, mlx_device dev) {
  try {
    *avail = mlx::core::is_available(mlx_device_get_(dev));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_device_count(int* count, mlx_device_type type) {
  try {
    auto cpp_type = mlx_device_type_to_cpp(type);
    *count = mlx::core::device_count(cpp_type);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_device_info mlx_device_info_new(void) {
  return mlx_device_info_new_();
}

extern "C" int mlx_device_info_get(mlx_device_info* info, mlx_device dev) {
  try {
    const auto& cpp_info = mlx::core::device_info(mlx_device_get_(dev));
    mlx_device_info_set_(*info, cpp_info);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_device_info_free(mlx_device_info info) {
  try {
    mlx_device_info_free_(info);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int
mlx_device_info_has_key(bool* exists, mlx_device_info info, const char* key) {
  try {
    const auto& cpp_info = mlx_device_info_get_(info);
    *exists = cpp_info.find(key) != cpp_info.end();
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_device_info_is_string(
    bool* is_string,
    mlx_device_info info,
    const char* key) {
  try {
    const auto& cpp_info = mlx_device_info_get_(info);
    auto it = cpp_info.find(key);
    if (it == cpp_info.end()) {
      return 2;
    }
    *is_string = std::holds_alternative<std::string>(it->second);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_device_info_get_string(
    const char** value,
    mlx_device_info info,
    const char* key) {
  try {
    const auto& cpp_info = mlx_device_info_get_(info);
    auto it = cpp_info.find(key);
    if (it == cpp_info.end()) {
      return 2;
    }
    if (!std::holds_alternative<std::string>(it->second)) {
      return 2;
    }
    *value = std::get<std::string>(it->second).c_str();
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int
mlx_device_info_get_size(size_t* value, mlx_device_info info, const char* key) {
  try {
    const auto& cpp_info = mlx_device_info_get_(info);
    auto it = cpp_info.find(key);
    if (it == cpp_info.end()) {
      return 2;
    }
    if (!std::holds_alternative<size_t>(it->second)) {
      return 2;
    }
    *value = std::get<size_t>(it->second);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_device_info_get_keys(
    mlx_vector_string* keys,
    mlx_device_info info) {
  try {
    const auto& cpp_info = mlx_device_info_get_(info);
    std::vector<std::string> key_vec;
    key_vec.reserve(cpp_info.size());
    for (const auto& kv : cpp_info) {
      key_vec.push_back(kv.first);
    }
    mlx_vector_string_set_(*keys, std::move(key_vec));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
