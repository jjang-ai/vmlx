/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>

#include "mlx/c/array.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/string.h"

extern "C" size_t mlx_dtype_size(mlx_dtype dtype) {
  return mlx_dtype_to_cpp(dtype).size();
}

extern "C" int mlx_array_tostring(mlx_string* str_, const mlx_array arr) {
  try {
    std::ostringstream os;
    os << mlx_array_get_(arr);
    std::string str = os.str();
    mlx_string_set_(*str_, str);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_array_free(mlx_array arr) {
  try {
    mlx_array_free_(arr);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_array mlx_array_new(void) {
  try {
    return mlx_array_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}

extern "C" int mlx_array_set(mlx_array* arr, const mlx_array src) {
  try {
    mlx_array_set_(*arr, mlx_array_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_set_bool(mlx_array* arr, bool val) {
  try {
    mlx_array_set_(*arr, mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_array mlx_array_new_bool(bool val) {
  try {
    return mlx_array_new_(mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" int mlx_array_set_int(mlx_array* arr, int val) {
  try {
    mlx_array_set_(*arr, mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_array mlx_array_new_int(int val) {
  try {
    return mlx_array_new_(mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" int mlx_array_set_float32(mlx_array* arr, float val) {
  try {
    mlx_array_set_(*arr, mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_set_float(mlx_array* arr, float val) {
  return mlx_array_set_float32(arr, val);
}
extern "C" int mlx_array_set_float64(mlx_array* arr, double val) {
  try {
    mlx_array_set_(*arr, mlx::core::array(val, mlx::core::float64));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_set_double(mlx_array* arr, double val) {
  return mlx_array_set_float64(arr, val);
}
extern "C" mlx_array mlx_array_new_float32(float val) {
  try {
    return mlx_array_new_(mlx::core::array(val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" mlx_array mlx_array_new_float(float val) {
  return mlx_array_new_float32(val);
}
extern "C" mlx_array mlx_array_new_float64(double val) {
  try {
    return mlx_array_new_(mlx::core::array(val, mlx::core::float64));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" mlx_array mlx_array_new_double(double val) {
  return mlx_array_new_float64(val);
}
extern "C" int
mlx_array_set_complex(mlx_array* arr, float real_val, float imag_val) {
  try {
    std::complex<float> cpp_val(real_val, imag_val);
    mlx_array_set_(*arr, mlx::core::array(cpp_val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_array mlx_array_new_complex(float real_val, float imag_val) {
  try {
    std::complex<float> cpp_val(real_val, imag_val);
    return mlx_array_new_(mlx::core::array(cpp_val));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" int mlx_array_set_data(
    mlx_array* arr,
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype) {
  try {
    mlx::core::Shape cpp_shape(shape, shape + dim);
    mlx::core::Dtype cpp_dtype = mlx_dtype_to_cpp(dtype);
    switch (cpp_dtype) {
      case mlx::core::bool_:
        mlx_array_set_(
            *arr, mlx::core::array((bool*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::uint8:
        mlx_array_set_(
            *arr, mlx::core::array((uint8_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::uint16:
        mlx_array_set_(
            *arr, mlx::core::array((uint16_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::uint32:
        mlx_array_set_(
            *arr, mlx::core::array((uint32_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::uint64:
        mlx_array_set_(
            *arr, mlx::core::array((uint64_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::int8:
        mlx_array_set_(
            *arr, mlx::core::array((int8_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::int16:
        mlx_array_set_(
            *arr, mlx::core::array((int16_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::int32:
        mlx_array_set_(
            *arr, mlx::core::array((int32_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::int64:
        mlx_array_set_(
            *arr, mlx::core::array((int64_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::float16:
        mlx_array_set_(
            *arr,
            mlx::core::array(
                (mlx::core::float16_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::float32:
        mlx_array_set_(
            *arr, mlx::core::array((float*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::float64:
        mlx_array_set_(
            *arr, mlx::core::array((double*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::bfloat16:
        mlx_array_set_(
            *arr,
            mlx::core::array(
                (mlx::core::bfloat16_t*)data, cpp_shape, cpp_dtype));
        break;
      case mlx::core::complex64:
        mlx_array_set_(
            *arr,
            mlx::core::array(
                (mlx::core::complex64_t*)data, cpp_shape, cpp_dtype));
        break;
      default:
        mlx_error("unknown data type");
        return 1;
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_array mlx_array_new_data(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype) {
  try {
    mlx_array arr = mlx_array_new_();
    if (mlx_array_set_data(&arr, data, shape, dim, dtype)) {
      return mlx_array_();
    }
    return arr;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" mlx_array mlx_array_new_data_managed_payload(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void* payload,
    void (*dtor)(void*)) {
  try {
    mlx::core::Shape cpp_shape(shape, shape + dim);
    mlx::core::Dtype cpp_dtype = mlx_dtype_to_cpp(dtype);
    std::function<void(void*)> cpp_deleter = [dtor, payload](void*) {
      dtor(payload);
    };
    return mlx_array_new_(
        mlx::core::array(data, cpp_shape, cpp_dtype, cpp_deleter));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_();
  }
}
extern "C" mlx_array mlx_array_new_data_managed(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void (*dtor)(void*)) {
  return mlx_array_new_data_managed_payload(
      data, shape, dim, dtype, data, dtor);
}

extern "C" size_t mlx_array_itemsize(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).itemsize();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" size_t mlx_array_size(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" size_t mlx_array_nbytes(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).nbytes();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" size_t mlx_array_ndim(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).ndim();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" const int* mlx_array_shape(const mlx_array arr) {
  try {
    return (int*)mlx_array_get_(arr).shape().data();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const size_t* mlx_array_strides(const mlx_array arr) {
  try {
    return (size_t*)mlx_array_get_(arr).strides().data();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" int mlx_array_dim(const mlx_array arr, int dim) {
  try {
    return mlx_array_get_(arr).shape(dim);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
extern "C" mlx_dtype mlx_array_dtype(const mlx_array arr) {
  try {
    return mlx_dtype_to_c(mlx_array_get_(arr).dtype());
  } catch (std::exception& e) {
    mlx_error(e.what());
    return MLX_BOOL;
  }
}

extern "C" int mlx_array_eval(mlx_array arr) {
  try {
    mlx_array_get_(arr).eval();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_array_item_bool(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<bool>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_uint8(uint8_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<uint8_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_uint16(uint16_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<uint16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_uint32(uint32_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<uint32_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_uint64(uint64_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<uint64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_int8(int8_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<int8_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_int16(int16_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<int16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_int32(int32_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<int32_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_int64(int64_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<int64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_float32(float* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<float>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_float64(double* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<double>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_array_item_complex64(
    mlx_complex64_t* res,
    const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<mlx_complex64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

#ifdef HAS_FLOAT16
extern "C" int mlx_array_item_float16(float16_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<float16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
#endif

#ifdef HAS_BFLOAT16
extern "C" int mlx_array_item_bfloat16(bfloat16_t* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).item<bfloat16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
#endif

extern "C" const bool* mlx_array_data_bool(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<bool>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const uint8_t* mlx_array_data_uint8(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<uint8_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const uint16_t* mlx_array_data_uint16(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<uint16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const uint32_t* mlx_array_data_uint32(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<uint32_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const uint64_t* mlx_array_data_uint64(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<uint64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int8_t* mlx_array_data_int8(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<int8_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int16_t* mlx_array_data_int16(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<int16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int32_t* mlx_array_data_int32(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<int32_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const int64_t* mlx_array_data_int64(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<int64_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const float* mlx_array_data_float32(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<float>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const double* mlx_array_data_float64(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<double>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
extern "C" const mlx_complex64_t* mlx_array_data_complex64(
    const mlx_array arr) {
  try {
    // std::complex<float> and mlx_complex64_t have the same memory layout
    return reinterpret_cast<const mlx_complex64_t*>(
        mlx_array_get_(arr).data<std::complex<float>>());
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}

#ifdef HAS_FLOAT16
extern "C" const float16_t* mlx_array_data_float16(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<float16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
#endif

#ifdef HAS_BFLOAT16
extern "C" const bfloat16_t* mlx_array_data_bfloat16(const mlx_array arr) {
  try {
    return mlx_array_get_(arr).data<bfloat16_t>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }
}
#endif

extern "C" int _mlx_array_is_available(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).is_available();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int _mlx_array_wait(const mlx_array arr) {
  try {
    mlx_array_get_(arr).wait();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int _mlx_array_is_contiguous(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).flags().contiguous;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int _mlx_array_is_row_contiguous(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).flags().row_contiguous;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int _mlx_array_is_col_contiguous(bool* res, const mlx_array arr) {
  try {
    *res = mlx_array_get_(arr).flags().col_contiguous;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
