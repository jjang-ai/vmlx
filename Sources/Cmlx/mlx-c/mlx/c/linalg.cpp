/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/linalg.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/linalg.h"

extern "C" int mlx_linalg_cholesky(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::cholesky(
            mlx_array_get_(a), upper, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_cholesky_inv(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::cholesky_inv(
            mlx_array_get_(a), upper, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_cross(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::cross(
            mlx_array_get_(a), mlx_array_get_(b), axis, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_eig(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) {
  try {
    {
      auto [tpl_0, tpl_1] =
          mlx::core::linalg::eig(mlx_array_get_(a), mlx_stream_get_(s));
      mlx_array_set_(*res_0, tpl_0);
      mlx_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_eigh(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s) {
  try {
    {
      auto [tpl_0, tpl_1] = mlx::core::linalg::eigh(
          mlx_array_get_(a), std::string(UPLO), mlx_stream_get_(s));
      mlx_array_set_(*res_0, tpl_0);
      mlx_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int
mlx_linalg_eigvals(mlx_array* res, const mlx_array a, const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::eigvals(mlx_array_get_(a), mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_eigvalsh(
    mlx_array* res,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::eigvalsh(
            mlx_array_get_(a), std::string(UPLO), mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int
mlx_linalg_inv(mlx_array* res, const mlx_array a, const mlx_stream s) {
  try {
    mlx_array_set_(
        *res, mlx::core::linalg::inv(mlx_array_get_(a), mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int
mlx_linalg_lu(mlx_vector_array* res, const mlx_array a, const mlx_stream s) {
  try {
    mlx_vector_array_set_(
        *res, mlx::core::linalg::lu(mlx_array_get_(a), mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_lu_factor(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) {
  try {
    {
      auto [tpl_0, tpl_1] =
          mlx::core::linalg::lu_factor(mlx_array_get_(a), mlx_stream_get_(s));
      mlx_array_set_(*res_0, tpl_0);
      mlx_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_norm(
    mlx_array* res,
    const mlx_array a,
    double ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::norm(
            mlx_array_get_(a),
            ord,
            (axis ? std::make_optional(std::vector<int>(axis, axis + axis_num))
                  : std::nullopt),
            keepdims,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_norm_matrix(
    mlx_array* res,
    const mlx_array a,
    const char* ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::norm(
            mlx_array_get_(a),
            std::string(ord),
            (axis ? std::make_optional(std::vector<int>(axis, axis + axis_num))
                  : std::nullopt),
            keepdims,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_norm_l2(
    mlx_array* res,
    const mlx_array a,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::norm(
            mlx_array_get_(a),
            (axis ? std::make_optional(std::vector<int>(axis, axis + axis_num))
                  : std::nullopt),
            keepdims,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int
mlx_linalg_pinv(mlx_array* res, const mlx_array a, const mlx_stream s) {
  try {
    mlx_array_set_(
        *res, mlx::core::linalg::pinv(mlx_array_get_(a), mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_qr(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) {
  try {
    {
      auto [tpl_0, tpl_1] =
          mlx::core::linalg::qr(mlx_array_get_(a), mlx_stream_get_(s));
      mlx_array_set_(*res_0, tpl_0);
      mlx_array_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_solve(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::solve(
            mlx_array_get_(a), mlx_array_get_(b), mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_solve_triangular(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool upper,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::solve_triangular(
            mlx_array_get_(a), mlx_array_get_(b), upper, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_svd(
    mlx_vector_array* res,
    const mlx_array a,
    bool compute_uv,
    const mlx_stream s) {
  try {
    mlx_vector_array_set_(
        *res,
        mlx::core::linalg::svd(
            mlx_array_get_(a), compute_uv, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_linalg_tri_inv(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::linalg::tri_inv(
            mlx_array_get_(a), upper, mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
