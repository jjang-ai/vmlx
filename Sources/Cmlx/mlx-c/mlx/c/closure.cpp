/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/closure.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" mlx_closure mlx_closure_new(void) {
  try {
    return mlx_closure_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_new_();
  }
}

extern "C" int mlx_closure_set(mlx_closure* cls, const mlx_closure src) {
  try {
    mlx_closure_set_(*cls, mlx_closure_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_closure_free(mlx_closure cls) {
  try {
    mlx_closure_free_(cls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_closure mlx_closure_new_func(
    int (*fun)(mlx_vector_array*, const mlx_vector_array)) {
  try {
    auto cpp_closure = [fun](const std::vector<mlx::core::array>& cpp_input) {
      auto input = mlx_vector_array_new_();
      mlx_vector_array_set_(input, cpp_input);
      auto res = mlx_vector_array_new_();
      auto status = fun(&res, input);
      mlx_vector_array_free(input);
      if (status) {
        mlx_vector_array_free(res);
        throw std::runtime_error("mlx_closure returned a non-zero value");
      }
      auto cpp_res = mlx_vector_array_get_(res);
      mlx_vector_array_free(res);
      return cpp_res;
    };
    return mlx_closure_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_new_();
  }
}

extern "C" mlx_closure mlx_closure_new_func_payload(
    int (*fun)(mlx_vector_array*, const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*)) {
  try {
    std::shared_ptr<void> cpp_payload = nullptr;
    if (dtor) {
      cpp_payload = std::shared_ptr<void>(payload, dtor);
    } else {
      cpp_payload = std::shared_ptr<void>(payload, [](void*) {});
    }
    auto cpp_closure =
        [fun, cpp_payload](const std::vector<mlx::core::array>& cpp_input) {
          auto input = mlx_vector_array_new_();
          mlx_vector_array_set_(input, cpp_input);
          auto res = mlx_vector_array_new_();
          auto status = fun(&res, input, cpp_payload.get());
          mlx_vector_array_free(input);
          if (status) {
            mlx_vector_array_free(res);
            throw std::runtime_error("mlx_closure returned a non-zero value");
          }
          auto cpp_res = mlx_vector_array_get_(res);
          mlx_vector_array_free(res);
          return cpp_res;
        };
    return mlx_closure_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_new_();
  }
}

extern "C" int mlx_closure_apply(
    mlx_vector_array* res,
    mlx_closure cls,
    const mlx_vector_array input) {
  try {
    mlx_vector_array_set_(
        *res, mlx_closure_get_(cls)(mlx_vector_array_get_(input)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_closure mlx_closure_new_unary(
    int (*fun)(mlx_array*, const mlx_array)) {
  try {
    auto cpp_closure = [fun](const std::vector<mlx::core::array>& cpp_input) {
      if (cpp_input.size() != 1) {
        throw std::runtime_error("closure: expected unary input");
      }
      auto input = mlx_array_new_(cpp_input[0]);
      auto res = mlx_array_new_();
      auto status = fun(&res, input);
      if (status) {
        mlx_array_free_(res);
        throw std::runtime_error("mlx_closure returned a non-zero value");
      }
      mlx_array_free(input);
      std::vector<mlx::core::array> cpp_res = {mlx_array_get_(res)};
      mlx_array_free(res);
      return cpp_res;
    };
    return mlx_closure_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_new_();
  }
}

extern "C" mlx_closure_kwargs mlx_closure_kwargs_new(void) {
  try {
    return mlx_closure_kwargs_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_kwargs_new_();
  }
}

extern "C" int mlx_closure_kwargs_set(
    mlx_closure_kwargs* cls,
    const mlx_closure_kwargs src) {
  try {
    mlx_closure_kwargs_set_(*cls, mlx_closure_kwargs_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_closure_kwargs_free(mlx_closure_kwargs cls) {
  try {
    mlx_closure_kwargs_free_(cls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_closure_kwargs mlx_closure_kwargs_new_func(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_map_string_to_array)) {
  try {
    auto cpp_closure =
        [fun](
            const std::vector<mlx::core::array>& cpp_input_0,
            const std::unordered_map<std::string, mlx::core::array>&
                cpp_input_1) {
          auto input_0 = mlx_vector_array_new_();
          mlx_vector_array_set_(input_0, cpp_input_0);
          auto input_1 = mlx_map_string_to_array_new_();
          mlx_map_string_to_array_set_(input_1, cpp_input_1);
          auto res = mlx_vector_array_new_();
          auto status = fun(&res, input_0, input_1);
          mlx_vector_array_free(input_0);
          mlx_map_string_to_array_free(input_1);
          if (status) {
            mlx_vector_array_free(res);
            throw std::runtime_error(
                "mlx_closure_kwargs returned a non-zero value");
          }
          auto cpp_res = mlx_vector_array_get_(res);
          mlx_vector_array_free(res);
          return cpp_res;
        };
    return mlx_closure_kwargs_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_kwargs_new_();
  }
}

extern "C" mlx_closure_kwargs mlx_closure_kwargs_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_map_string_to_array,
        void*),
    void* payload,
    void (*dtor)(void*)) {
  try {
    std::shared_ptr<void> cpp_payload = nullptr;
    if (dtor) {
      cpp_payload = std::shared_ptr<void>(payload, dtor);
    } else {
      cpp_payload = std::shared_ptr<void>(payload, [](void*) {});
    }
    auto cpp_closure =
        [fun, cpp_payload](
            const std::vector<mlx::core::array>& cpp_input_0,
            const std::unordered_map<std::string, mlx::core::array>&
                cpp_input_1) {
          auto input_0 = mlx_vector_array_new_();
          mlx_vector_array_set_(input_0, cpp_input_0);
          auto input_1 = mlx_map_string_to_array_new_();
          mlx_map_string_to_array_set_(input_1, cpp_input_1);
          auto res = mlx_vector_array_new_();
          auto status = fun(&res, input_0, input_1, cpp_payload.get());
          mlx_vector_array_free(input_0);
          mlx_map_string_to_array_free(input_1);
          if (status) {
            mlx_vector_array_free(res);
            throw std::runtime_error(
                "mlx_closure_kwargs returned a non-zero value");
          }
          auto cpp_res = mlx_vector_array_get_(res);
          mlx_vector_array_free(res);
          return cpp_res;
        };
    return mlx_closure_kwargs_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_kwargs_new_();
  }
}

extern "C" int mlx_closure_kwargs_apply(
    mlx_vector_array* res,
    mlx_closure_kwargs cls,
    const mlx_vector_array input_0,
    const mlx_map_string_to_array input_1) {
  try {
    mlx_vector_array_set_(
        *res,
        mlx_closure_kwargs_get_(cls)(
            mlx_vector_array_get_(input_0),
            mlx_map_string_to_array_get_(input_1)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_closure_value_and_grad mlx_closure_value_and_grad_new(void) {
  try {
    return mlx_closure_value_and_grad_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_value_and_grad_new_();
  }
}

extern "C" int mlx_closure_value_and_grad_set(
    mlx_closure_value_and_grad* cls,
    const mlx_closure_value_and_grad src) {
  try {
    mlx_closure_value_and_grad_set_(*cls, mlx_closure_value_and_grad_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_closure_value_and_grad_free(mlx_closure_value_and_grad cls) {
  try {
    mlx_closure_value_and_grad_free_(cls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_closure_value_and_grad mlx_closure_value_and_grad_new_func(
    int (*fun)(mlx_vector_array*, mlx_vector_array*, const mlx_vector_array)) {
  try {
    auto cpp_closure = [fun](const std::vector<mlx::core::array>& cpp_input) {
      auto input = mlx_vector_array_new_();
      mlx_vector_array_set_(input, cpp_input);
      auto res_0 = mlx_vector_array_new_();
      auto res_1 = mlx_vector_array_new_();
      ;
      auto status = fun(&res_0, &res_1, input);
      mlx_vector_array_free(input);
      if (status) {
        mlx_vector_array_free(res_0);
        mlx_vector_array_free(res_1);
        ;
        throw std::runtime_error(
            "mlx_closure_value_and_grad returned a non-zero value");
      }
      auto cpp_res = std::make_pair(
          mlx_vector_array_get_(res_0), mlx_vector_array_get_(res_1));
      mlx_vector_array_free(res_0);
      mlx_vector_array_free(res_1);
      ;
      return cpp_res;
    };
    return mlx_closure_value_and_grad_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_value_and_grad_new_();
  }
}

extern "C" mlx_closure_value_and_grad
mlx_closure_value_and_grad_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_array*,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*)) {
  try {
    std::shared_ptr<void> cpp_payload = nullptr;
    if (dtor) {
      cpp_payload = std::shared_ptr<void>(payload, dtor);
    } else {
      cpp_payload = std::shared_ptr<void>(payload, [](void*) {});
    }
    auto cpp_closure =
        [fun, cpp_payload](const std::vector<mlx::core::array>& cpp_input) {
          auto input = mlx_vector_array_new_();
          mlx_vector_array_set_(input, cpp_input);
          auto res_0 = mlx_vector_array_new_();
          auto res_1 = mlx_vector_array_new_();
          ;
          auto status = fun(&res_0, &res_1, input, cpp_payload.get());
          mlx_vector_array_free(input);
          if (status) {
            mlx_vector_array_free(res_0);
            mlx_vector_array_free(res_1);
            ;
            throw std::runtime_error(
                "mlx_closure_value_and_grad returned a non-zero value");
          }
          auto cpp_res = std::make_pair(
              mlx_vector_array_get_(res_0), mlx_vector_array_get_(res_1));
          mlx_vector_array_free(res_0);
          mlx_vector_array_free(res_1);
          ;
          return cpp_res;
        };
    return mlx_closure_value_and_grad_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_value_and_grad_new_();
  }
}

extern "C" int mlx_closure_value_and_grad_apply(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    mlx_closure_value_and_grad cls,
    const mlx_vector_array input) {
  try {
    {
      auto [tpl_0, tpl_1] =
          mlx_closure_value_and_grad_get_(cls)(mlx_vector_array_get_(input));
      mlx_vector_array_set_(*res_0, tpl_0);
      mlx_vector_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_closure_custom mlx_closure_custom_new(void) {
  try {
    return mlx_closure_custom_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_new_();
  }
}

extern "C" int mlx_closure_custom_set(
    mlx_closure_custom* cls,
    const mlx_closure_custom src) {
  try {
    mlx_closure_custom_set_(*cls, mlx_closure_custom_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_closure_custom_free(mlx_closure_custom cls) {
  try {
    mlx_closure_custom_free_(cls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_closure_custom mlx_closure_custom_new_func(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const mlx_vector_array)) {
  try {
    auto cpp_closure = [fun](
                           const std::vector<mlx::core::array>& cpp_input_0,
                           const std::vector<mlx::core::array>& cpp_input_1,
                           const std::vector<mlx::core::array>& cpp_input_2) {
      auto input_0 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_0, cpp_input_0);
      auto input_1 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_1, cpp_input_1);
      auto input_2 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_2, cpp_input_2);
      auto res = mlx_vector_array_new_();
      auto status = fun(&res, input_0, input_1, input_2);
      mlx_vector_array_free(input_0);
      mlx_vector_array_free(input_1);
      mlx_vector_array_free(input_2);
      if (status) {
        mlx_vector_array_free(res);
        throw std::runtime_error(
            "mlx_closure_custom returned a non-zero value");
      }
      auto cpp_res = mlx_vector_array_get_(res);
      mlx_vector_array_free(res);
      return cpp_res;
    };
    return mlx_closure_custom_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_new_();
  }
}

extern "C" mlx_closure_custom mlx_closure_custom_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*)) {
  try {
    std::shared_ptr<void> cpp_payload = nullptr;
    if (dtor) {
      cpp_payload = std::shared_ptr<void>(payload, dtor);
    } else {
      cpp_payload = std::shared_ptr<void>(payload, [](void*) {});
    }
    auto cpp_closure = [fun, cpp_payload](
                           const std::vector<mlx::core::array>& cpp_input_0,
                           const std::vector<mlx::core::array>& cpp_input_1,
                           const std::vector<mlx::core::array>& cpp_input_2) {
      auto input_0 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_0, cpp_input_0);
      auto input_1 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_1, cpp_input_1);
      auto input_2 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_2, cpp_input_2);
      auto res = mlx_vector_array_new_();
      auto status = fun(&res, input_0, input_1, input_2, cpp_payload.get());
      mlx_vector_array_free(input_0);
      mlx_vector_array_free(input_1);
      mlx_vector_array_free(input_2);
      if (status) {
        mlx_vector_array_free(res);
        throw std::runtime_error(
            "mlx_closure_custom returned a non-zero value");
      }
      auto cpp_res = mlx_vector_array_get_(res);
      mlx_vector_array_free(res);
      return cpp_res;
    };
    return mlx_closure_custom_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_new_();
  }
}

extern "C" int mlx_closure_custom_apply(
    mlx_vector_array* res,
    mlx_closure_custom cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const mlx_vector_array input_2) {
  try {
    mlx_vector_array_set_(
        *res,
        mlx_closure_custom_get_(cls)(
            mlx_vector_array_get_(input_0),
            mlx_vector_array_get_(input_1),
            mlx_vector_array_get_(input_2)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_closure_custom_jvp mlx_closure_custom_jvp_new(void) {
  try {
    return mlx_closure_custom_jvp_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_jvp_new_();
  }
}

extern "C" int mlx_closure_custom_jvp_set(
    mlx_closure_custom_jvp* cls,
    const mlx_closure_custom_jvp src) {
  try {
    mlx_closure_custom_jvp_set_(*cls, mlx_closure_custom_jvp_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_closure_custom_jvp_free(mlx_closure_custom_jvp cls) {
  try {
    mlx_closure_custom_jvp_free_(cls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_closure_custom_jvp mlx_closure_custom_jvp_new_func(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const int*,
        size_t _num)) {
  try {
    auto cpp_closure = [fun](
                           const std::vector<mlx::core::array>& cpp_input_0,
                           const std::vector<mlx::core::array>& cpp_input_1,
                           const std::vector<int>& cpp_input_2) {
      auto input_0 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_0, cpp_input_0);
      auto input_1 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_1, cpp_input_1);
      const int* input_2 = nullptr;
      size_t input_2_num = 0;
      input_2 = cpp_input_2.data();
      input_2_num = cpp_input_2.size();
      auto res = mlx_vector_array_new_();
      auto status = fun(&res, input_0, input_1, input_2, input_2_num);
      mlx_vector_array_free(input_0);
      mlx_vector_array_free(input_1);
      ;
      if (status) {
        mlx_vector_array_free(res);
        throw std::runtime_error(
            "mlx_closure_custom_jvp returned a non-zero value");
      }
      auto cpp_res = mlx_vector_array_get_(res);
      mlx_vector_array_free(res);
      return cpp_res;
    };
    return mlx_closure_custom_jvp_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_jvp_new_();
  }
}

extern "C" mlx_closure_custom_jvp mlx_closure_custom_jvp_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*)) {
  try {
    std::shared_ptr<void> cpp_payload = nullptr;
    if (dtor) {
      cpp_payload = std::shared_ptr<void>(payload, dtor);
    } else {
      cpp_payload = std::shared_ptr<void>(payload, [](void*) {});
    }
    auto cpp_closure = [fun, cpp_payload](
                           const std::vector<mlx::core::array>& cpp_input_0,
                           const std::vector<mlx::core::array>& cpp_input_1,
                           const std::vector<int>& cpp_input_2) {
      auto input_0 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_0, cpp_input_0);
      auto input_1 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_1, cpp_input_1);
      const int* input_2 = nullptr;
      size_t input_2_num = 0;
      input_2 = cpp_input_2.data();
      input_2_num = cpp_input_2.size();
      auto res = mlx_vector_array_new_();
      auto status =
          fun(&res, input_0, input_1, input_2, input_2_num, cpp_payload.get());
      mlx_vector_array_free(input_0);
      mlx_vector_array_free(input_1);
      ;
      if (status) {
        mlx_vector_array_free(res);
        throw std::runtime_error(
            "mlx_closure_custom_jvp returned a non-zero value");
      }
      auto cpp_res = mlx_vector_array_get_(res);
      mlx_vector_array_free(res);
      return cpp_res;
    };
    return mlx_closure_custom_jvp_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_jvp_new_();
  }
}

extern "C" int mlx_closure_custom_jvp_apply(
    mlx_vector_array* res,
    mlx_closure_custom_jvp cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const int* input_2,
    size_t input_2_num) {
  try {
    mlx_vector_array_set_(
        *res,
        mlx_closure_custom_jvp_get_(cls)(
            mlx_vector_array_get_(input_0),
            mlx_vector_array_get_(input_1),
            std::vector<int>(input_2, input_2 + input_2_num)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_closure_custom_vmap mlx_closure_custom_vmap_new(void) {
  try {
    return mlx_closure_custom_vmap_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_vmap_new_();
  }
}

extern "C" int mlx_closure_custom_vmap_set(
    mlx_closure_custom_vmap* cls,
    const mlx_closure_custom_vmap src) {
  try {
    mlx_closure_custom_vmap_set_(*cls, mlx_closure_custom_vmap_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_closure_custom_vmap_free(mlx_closure_custom_vmap cls) {
  try {
    mlx_closure_custom_vmap_free_(cls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_closure_custom_vmap mlx_closure_custom_vmap_new_func(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_int*,
        const mlx_vector_array,
        const int*,
        size_t _num)) {
  try {
    auto cpp_closure = [fun](
                           const std::vector<mlx::core::array>& cpp_input_0,
                           const std::vector<int>& cpp_input_1) {
      auto input_0 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_0, cpp_input_0);
      const int* input_1 = nullptr;
      size_t input_1_num = 0;
      input_1 = cpp_input_1.data();
      input_1_num = cpp_input_1.size();
      auto res_0 = mlx_vector_array_new_();
      auto res_1 = mlx_vector_int_new_();
      ;
      auto status = fun(&res_0, &res_1, input_0, input_1, input_1_num);
      mlx_vector_array_free(input_0);
      ;
      if (status) {
        mlx_vector_array_free(res_0);
        mlx_vector_int_free(res_1);
        ;
        throw std::runtime_error(
            "mlx_closure_custom_vmap returned a non-zero value");
      }
      auto cpp_res = std::make_pair(
          mlx_vector_array_get_(res_0), mlx_vector_int_get_(res_1));
      mlx_vector_array_free(res_0);
      mlx_vector_int_free(res_1);
      ;
      return cpp_res;
    };
    return mlx_closure_custom_vmap_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_vmap_new_();
  }
}

extern "C" mlx_closure_custom_vmap mlx_closure_custom_vmap_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_int*,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*)) {
  try {
    std::shared_ptr<void> cpp_payload = nullptr;
    if (dtor) {
      cpp_payload = std::shared_ptr<void>(payload, dtor);
    } else {
      cpp_payload = std::shared_ptr<void>(payload, [](void*) {});
    }
    auto cpp_closure = [fun, cpp_payload](
                           const std::vector<mlx::core::array>& cpp_input_0,
                           const std::vector<int>& cpp_input_1) {
      auto input_0 = mlx_vector_array_new_();
      mlx_vector_array_set_(input_0, cpp_input_0);
      const int* input_1 = nullptr;
      size_t input_1_num = 0;
      input_1 = cpp_input_1.data();
      input_1_num = cpp_input_1.size();
      auto res_0 = mlx_vector_array_new_();
      auto res_1 = mlx_vector_int_new_();
      ;
      auto status =
          fun(&res_0, &res_1, input_0, input_1, input_1_num, cpp_payload.get());
      mlx_vector_array_free(input_0);
      ;
      if (status) {
        mlx_vector_array_free(res_0);
        mlx_vector_int_free(res_1);
        ;
        throw std::runtime_error(
            "mlx_closure_custom_vmap returned a non-zero value");
      }
      auto cpp_res = std::make_pair(
          mlx_vector_array_get_(res_0), mlx_vector_int_get_(res_1));
      mlx_vector_array_free(res_0);
      mlx_vector_int_free(res_1);
      ;
      return cpp_res;
    };
    return mlx_closure_custom_vmap_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_custom_vmap_new_();
  }
}

extern "C" int mlx_closure_custom_vmap_apply(
    mlx_vector_array* res_0,
    mlx_vector_int* res_1,
    mlx_closure_custom_vmap cls,
    const mlx_vector_array input_0,
    const int* input_1,
    size_t input_1_num) {
  try {
    {
      auto [tpl_0, tpl_1] = mlx_closure_custom_vmap_get_(cls)(
          mlx_vector_array_get_(input_0),
          std::vector<int>(input_1, input_1 + input_1_num));
      mlx_vector_array_set_(*res_0, tpl_0);
      mlx_vector_int_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
