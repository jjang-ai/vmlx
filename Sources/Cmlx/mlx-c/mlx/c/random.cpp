/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/random.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/random.h"

extern "C" int mlx_random_bernoulli(
    mlx_array* res,
    const mlx_array p,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::bernoulli(
            mlx_array_get_(p),
            mlx::core::Shape(shape, shape + shape_num),
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_bits(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    int width,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::bits(
            mlx::core::Shape(shape, shape + shape_num),
            width,
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_categorical_shape(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::categorical(
            mlx_array_get_(logits),
            axis,
            mlx::core::Shape(shape, shape + shape_num),
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_categorical_num_samples(
    mlx_array* res,
    const mlx_array logits_,
    int axis,
    int num_samples,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::categorical(
            mlx_array_get_(logits_),
            axis,
            num_samples,
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_categorical(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::categorical(
            mlx_array_get_(logits),
            axis,
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_gumbel(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::gumbel(
            mlx::core::Shape(shape, shape + shape_num),
            mlx_dtype_to_cpp(dtype),
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_key(mlx_array* res, uint64_t seed) {
  try {
    mlx_array_set_(*res, mlx::core::random::key(seed));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_laplace(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::laplace(
            mlx::core::Shape(shape, shape + shape_num),
            mlx_dtype_to_cpp(dtype),
            loc,
            scale,
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_multivariate_normal(
    mlx_array* res,
    const mlx_array mean,
    const mlx_array cov,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::multivariate_normal(
            mlx_array_get_(mean),
            mlx_array_get_(cov),
            mlx::core::Shape(shape, shape + shape_num),
            mlx_dtype_to_cpp(dtype),
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_normal_broadcast(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array loc /* may be null */,
    const mlx_array scale /* may be null */,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::normal(
            mlx::core::Shape(shape, shape + shape_num),
            mlx_dtype_to_cpp(dtype),
            (loc.ctx ? std::make_optional(mlx_array_get_(loc)) : std::nullopt),
            (scale.ctx ? std::make_optional(mlx_array_get_(scale))
                       : std::nullopt),
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_normal(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::normal(
            mlx::core::Shape(shape, shape + shape_num),
            mlx_dtype_to_cpp(dtype),
            loc,
            scale,
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_permutation(
    mlx_array* res,
    const mlx_array x,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::permutation(
            mlx_array_get_(x),
            axis,
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_permutation_arange(
    mlx_array* res,
    int x,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::permutation(
            x,
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_randint(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::randint(
            mlx_array_get_(low),
            mlx_array_get_(high),
            mlx::core::Shape(shape, shape + shape_num),
            mlx_dtype_to_cpp(dtype),
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_seed(uint64_t seed) {
  try {
    mlx::core::random::seed(seed);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_split_num(
    mlx_array* res,
    const mlx_array key,
    int num,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::split(mlx_array_get_(key), num, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_split(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array key,
    const mlx_stream s) {
  try {
    {
      auto [tpl_0, tpl_1] =
          mlx::core::random::split(mlx_array_get_(key), mlx_stream_get_(s));
      mlx_array_set_(*res_0, tpl_0);
      mlx_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_truncated_normal(
    mlx_array* res,
    const mlx_array lower,
    const mlx_array upper,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::truncated_normal(
            mlx_array_get_(lower),
            mlx_array_get_(upper),
            mlx::core::Shape(shape, shape + shape_num),
            mlx_dtype_to_cpp(dtype),
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_random_uniform(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::random::uniform(
            mlx_array_get_(low),
            mlx_array_get_(high),
            mlx::core::Shape(shape, shape + shape_num),
            mlx_dtype_to_cpp(dtype),
            (key.ctx ? std::make_optional(mlx_array_get_(key)) : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
