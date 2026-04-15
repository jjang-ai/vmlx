import argparse
import regex
import string
import type_private_generator as tpg

parser = argparse.ArgumentParser("MLX C vector code generator", add_help=False)
parser.add_argument("--implementation", default=False, action="store_true")
parser.add_argument("--private", default=False, action="store_true")
args = parser.parse_args()


def replace_match_parenthesis(string, keyword, fun):
    pattern = regex.compile(keyword + r"(\((?:[^()]++|(?1))++\))")
    res = []
    pos = 0
    for m in pattern.finditer(string):
        res.append(string[pos : m.start()])
        res.append(fun(m[1][1:-1]))
        pos = m.end()
    res.append(string[pos:])
    return "".join(res)


decl_code = r"""
/**
 * A vector of SCTYPE.
 */
typedef struct mlx_vector_SCTYPE_ {
  void* ctx;
} mlx_vector_SCTYPE;
mlx_vector_SCTYPE mlx_vector_SCTYPE_new(void);
int mlx_vector_SCTYPE_set(mlx_vector_SCTYPE* vec, const mlx_vector_SCTYPE src);
int mlx_vector_SCTYPE_free(mlx_vector_SCTYPE vec);
mlx_vector_SCTYPE mlx_vector_SCTYPE_new_data(CTYPE* data, size_t size);
mlx_vector_SCTYPE mlx_vector_SCTYPE_new_value(CTYPE val);
int mlx_vector_SCTYPE_set_data(
    mlx_vector_SCTYPE* vec,
    CTYPE* data,
    size_t size);
int mlx_vector_SCTYPE_set_value(mlx_vector_SCTYPE* vec, CTYPE val);
int mlx_vector_SCTYPE_append_data(
    mlx_vector_SCTYPE vec,
    CTYPE* data,
    size_t size);
int mlx_vector_SCTYPE_append_value(mlx_vector_SCTYPE vec, CTYPE val);
size_t mlx_vector_SCTYPE_size(mlx_vector_SCTYPE vec);
int mlx_vector_SCTYPE_get(
    RETURN_CTYPE res,
    const mlx_vector_SCTYPE vec,
    size_t idx);
"""

impl_code = r"""
extern "C" mlx_vector_SCTYPE mlx_vector_SCTYPE_new(void) {
  try {
    return mlx_vector_SCTYPE_new_({});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_SCTYPE_new_();
  }
}

extern "C" int mlx_vector_SCTYPE_set(
    mlx_vector_SCTYPE* vec,
    const mlx_vector_SCTYPE src) {
  try {
    mlx_vector_SCTYPE_set_(*vec, mlx_vector_SCTYPE_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_SCTYPE_free(mlx_vector_SCTYPE vec) {
  try {
    mlx_vector_SCTYPE_free_(vec);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_vector_SCTYPE mlx_vector_SCTYPE_new_data(
    CTYPE* data,
    size_t size) {
  try {
    auto vec = mlx_vector_SCTYPE_new();
    for (size_t i = 0; i < size; i++) {
      mlx_vector_SCTYPE_get_(vec).push_back(C_TO_CPP(data[i]));
    }
    return vec;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_SCTYPE_new_();
  }
}

extern "C" mlx_vector_SCTYPE mlx_vector_SCTYPE_new_value(CTYPE val) {
  try {
    return mlx_vector_SCTYPE_new_({C_TO_CPP(val)});
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_vector_SCTYPE_new_();
  }
}

extern "C" int
mlx_vector_SCTYPE_set_data(mlx_vector_SCTYPE* vec_, CTYPE* data, size_t size) {
  try {
    std::vector<CPPTYPE> cpp_arrs;
    for (size_t i = 0; i < size; i++) {
      cpp_arrs.push_back(C_TO_CPP(data[i]));
    }
    mlx_vector_SCTYPE_set_(*vec_, cpp_arrs);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_SCTYPE_set_value(mlx_vector_SCTYPE* vec_, CTYPE val) {
  try {
    mlx_vector_SCTYPE_set_(*vec_, std::vector<CPPTYPE>({C_TO_CPP(val)}));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int
mlx_vector_SCTYPE_append_data(mlx_vector_SCTYPE vec, CTYPE* data, size_t size) {
  try {
    for (size_t i = 0; i < size; i++) {
      mlx_vector_SCTYPE_get_(vec).push_back(C_TO_CPP(data[i]));
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_SCTYPE_append_value(
    mlx_vector_SCTYPE vec,
    CTYPE value) {
  try {
    mlx_vector_SCTYPE_get_(vec).push_back(C_TO_CPP(value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_vector_SCTYPE_get(
    RETURN_CTYPE res,
    const mlx_vector_SCTYPE vec,
    size_t index) {
  try {
    C_ASSIGN(res, mlx_vector_SCTYPE_get_(vec).at(index));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" size_t mlx_vector_SCTYPE_size(mlx_vector_SCTYPE vec) {
  try {
    return mlx_vector_SCTYPE_get_(vec).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}
"""


def generate(
    code,
    cpptype,
    ctype,
    sctype,
    rctype=None,
    c_to_cpp=lambda s: s + "->ctx",
    c_assign=lambda d, s: "(*" + d + ")->ctx = " + s,
):
    if code is None:
        return tpg.generate("mlx_vector_" + sctype, "std::vector<" + cpptype + ">")

    if rctype is None:
        rctype = ctype.replace("const ", "") + "*"

    def c_assign_wrap(s):
        d, s = s.split(",")
        return c_assign(d, s)

    code = replace_match_parenthesis(code, "C_ASSIGN", c_assign_wrap)
    code = replace_match_parenthesis(code, "C_TO_CPP", c_to_cpp)
    code = code.replace("RETURN_CTYPE", rctype)
    code = code.replace("SCTYPE", sctype)
    code = code.replace("CPPTYPE", cpptype)
    code = code.replace("CTYPE", ctype)
    return code


decl_begin = r"""/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_VECTOR_H
#define MLX_VECTOR_H

#include "mlx/c/array.h"
#include "mlx/c/string.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_vector Vectors
 * MLX vector objects.
 */
/**@{*/
"""

decl_end = """
/**@}*/

#ifdef __cplusplus
}
#endif

#endif
"""

impl_begin = r"""/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/vector.h"
"""

impl_end = """
"""

priv_begin = r"""/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_VECTOR_PRIVATE_H
#define MLX_VECTOR_PRIVATE_H

#include "mlx/c/vector.h"
#include "mlx/mlx.h"
"""

priv_end = r"""
#endif
"""

if args.implementation:
    code = impl_code
    begin = impl_begin
    end = impl_end
elif args.private:
    code = None
    begin = priv_begin
    end = priv_end
else:
    code = decl_code
    begin = decl_begin
    end = decl_end

print(begin)
print(
    generate(
        code,
        "mlx::core::array",
        "const mlx_array",
        "array",
        "mlx_array*",
        lambda s: "mlx_array_get_(" + s + ")",
        lambda d, s: "mlx_array_set_(*" + d + ", " + s + ")",
    )
)
print(
    generate(
        code,
        "std::vector<mlx::core::array>",
        "const mlx_vector_array",
        "vector_array",
        "mlx_vector_array*",
        lambda s: "mlx_vector_array_get_(" + s + ")",
        lambda d, s: "mlx_vector_array_set_(*" + d + ", " + s + ")",
    )
)
print(
    generate(
        code,
        "int",
        "int",
        "int",
        "int*",
        lambda s: s,
        lambda d, s: "*" + d + " = " + s,
    )
)
# print(
#     generate(
#         code,
#         "std::vector<int>",
#         "const mlx_vector_int",
#         "vector_int",
#     )
# )
print(
    generate(
        code,
        "std::string",
        "const char*",
        "string",
        "char**",
        lambda s: s,
        lambda d, s: "*" + d + " = " + s + ".data()",
    )
)
print(end)
