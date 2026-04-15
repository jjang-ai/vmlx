/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/transforms.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/transforms.h"

extern "C" int mlx_async_eval(const mlx_vector_array outputs) {
  try {
    mlx::core::async_eval(mlx_vector_array_get_(outputs));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_checkpoint(mlx_closure* res, const mlx_closure fun) {
  try {
    mlx_closure_set_(*res, mlx::core::checkpoint(mlx_closure_get_(fun)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_custom_function(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp /* may be null */,
    const mlx_closure_custom_jvp fun_jvp /* may be null */,
    const mlx_closure_custom_vmap fun_vmap /* may be null */) {
  try {
    mlx_closure_set_(
        *res,
        mlx::core::custom_function(
            mlx_closure_get_(fun),
            (fun_vjp.ctx ? std::make_optional(mlx_closure_custom_get_(fun_vjp))
                         : std::nullopt),
            (fun_jvp.ctx
                 ? std::make_optional(mlx_closure_custom_jvp_get_(fun_jvp))
                 : std::nullopt),
            (fun_vmap.ctx
                 ? std::make_optional(mlx_closure_custom_vmap_get_(fun_vmap))
                 : std::nullopt)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_custom_vjp(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp) {
  try {
    mlx_closure_set_(
        *res,
        mlx::core::custom_vjp(
            mlx_closure_get_(fun), mlx_closure_custom_get_(fun_vjp)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_eval(const mlx_vector_array outputs) {
  try {
    mlx::core::eval(mlx_vector_array_get_(outputs));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_jvp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array tangents) {
  try {
    {
      auto [tpl_0, tpl_1] = mlx::core::jvp(
          mlx_closure_get_(fun),
          mlx_vector_array_get_(primals),
          mlx_vector_array_get_(tangents));
      mlx_vector_array_set_(*res_0, tpl_0);
      mlx_vector_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_value_and_grad(
    mlx_closure_value_and_grad* res,
    const mlx_closure fun,
    const int* argnums,
    size_t argnums_num) {
  try {
    mlx_closure_value_and_grad_set_(
        *res,
        mlx::core::value_and_grad(
            mlx_closure_get_(fun),
            std::vector<int>(argnums, argnums + argnums_num)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_vjp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array cotangents) {
  try {
    {
      auto [tpl_0, tpl_1] = mlx::core::vjp(
          mlx_closure_get_(fun),
          mlx_vector_array_get_(primals),
          mlx_vector_array_get_(cotangents));
      mlx_vector_array_set_(*res_0, tpl_0);
      mlx_vector_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
