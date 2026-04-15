import argparse
import regex
import string
import mlxtypes as mt
import type_private_generator as tpg

parser = argparse.ArgumentParser("MLX C closure code generator", add_help=False)
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
typedef struct NAME_ {
  void* ctx;
} NAME;
NAME NAME_new(void);
int NAME_free(NAME cls);
NAME NAME_new_func(int (*fun)(RCARGS_UNNAMED, CARGS_UNNAMED));
NAME NAME_new_func_payload(
    int (*fun)(RCARGS_UNNAMED, CARGS_UNNAMED, void*),
    void* payload,
    void (*dtor)(void*));
int NAME_set(NAME *cls, const NAME src);
int NAME_apply(RCARGS, NAME cls, CARGS);
"""


def generate(code, name, rcpptype, cpptypes):
    rcpparg = mt.cpptypes[rcpptype]["cpp"].replace("@", "")
    cppargs = ", ".join([mt.cpptypes[cpptype]["cpp_arg"]("") for cpptype in cpptypes])

    if code is None:
        return tpg.generate(name, "std::function<" + rcpparg + "(" + cppargs + ")>")

    cargs_untyped = []
    cargs = []
    cppargs_type_name = []
    cppargs_to_cargs = []
    cargs_free = []
    cargs_ctx = []
    for i in range(len(cpptypes)):
        cpptype = mt.cpptypes[cpptypes[i]]
        cpparg = cpptype["cpp"]
        suffix = "_" + str(i) if len(cpptypes) > 1 else ""
        cargs_untyped.append(cpptype["c_arg"]("input" + suffix, untyped=True))
        cargs.append(cpptype["c_arg"]("input" + suffix))
        cppargs_type_name.append(cpptype["cpp_arg"]("cpp_input" + suffix))
        cargs_free.append(cpptype["free"]("input" + suffix) + ";")
        cargs_ctx.append(cpptype["c_to_cpp"]("input" + suffix))
        cppargs_to_cargs.append(cpptype["c_new"]("input" + suffix) + ";")
        cppargs_to_cargs.append(
            cpptype["c_assign_from_cpp"](
                "input" + suffix, "cpp_input" + suffix, returned=False
            )
            + ";"
        )

    rcargs_new = mt.cpptypes[rcpptype]["c_new"]("res") + ";"
    rcargs_free = mt.cpptypes[rcpptype]["free"]("res") + ";"
    rcargs_to_cpp = "auto cpp_res = " + mt.cpptypes[rcpptype]["c_to_cpp"]("res") + ";"

    cargs_untyped = ", ".join(cargs_untyped)
    cargs = ", ".join(cargs)
    cppargs_type_name = ", ".join(cppargs_type_name)
    cppargs_to_cargs = "\n".join(cppargs_to_cargs)
    cargs_free = "\n".join(cargs_free)
    cargs_ctx = ", ".join(cargs_ctx)
    cargs_unnamed = " ".join(
        [mt.cpptypes[cpptype]["c_arg"]("") for cpptype in cpptypes]
    )
    rcargs_unnamed = mt.cpptypes[rcpptype]["c_return_arg"]("")
    rcargs = mt.cpptypes[rcpptype]["c_return_arg"]("res")
    rcargs_untyped = mt.cpptypes[rcpptype]["c_return_arg"]("res", untyped=True)

    code = code.replace("RCARGS_UNTYPED", rcargs_untyped)
    code = code.replace("RCARGS_UNNAMED", rcargs_unnamed)
    code = code.replace("CPPARGS_TYPE_NAME", cppargs_type_name)
    code = code.replace("CPPARGS_TO_CARGS", cppargs_to_cargs)
    code = code.replace("RCARGS_NEW", rcargs_new)
    code = code.replace("RCARGS_FREE", rcargs_free)
    code = code.replace("RCARGS_TO_CPP", rcargs_to_cpp)
    code = code.replace("CARGS_UNTYPED", cargs_untyped)
    code = code.replace("CARGS_CTX", cargs_ctx)
    code = code.replace("CARGS_FREE", cargs_free)
    code = code.replace("RCPPARG", rcpparg)
    code = code.replace(
        "CARGS_UNNAMED",
        ", ".join([mt.cpptypes[cpptype]["c_arg"]("") for cpptype in cpptypes]),
    )

    code = code.replace(
        "ASSIGN_CLS_TO_RCARGS",
        mt.cpptypes[rcpptype]["c_assign_from_cpp"](
            "res", "NAME_get_(cls)(" + cargs_ctx + ")", returned=True
        )
        + ";",
    )

    code = code.replace("CPPARGS", cppargs)
    code = code.replace("NAME", name)
    code = code.replace("RCARGS", rcargs)
    code = code.replace("CARGS", cargs)

    return code


impl_code = r"""
extern "C" NAME NAME_new(void) {
  try {
    return NAME_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return NAME_new_();
  }
}

extern "C" int NAME_set(NAME *cls, const NAME src) {
  try {
    NAME_set_(*cls, NAME_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int NAME_free(NAME cls) {
  try {
    NAME_free_(cls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" NAME NAME_new_func(int (*fun)(RCARGS_UNNAMED, CARGS_UNNAMED)) {
  try {
    auto cpp_closure = [fun](CPPARGS_TYPE_NAME) {
      CPPARGS_TO_CARGS
      RCARGS_NEW
      auto status = fun(RCARGS_UNTYPED, CARGS_UNTYPED);
      CARGS_FREE
      if(status) {
        RCARGS_FREE
        throw std::runtime_error("NAME returned a non-zero value");
      }
      RCARGS_TO_CPP
      RCARGS_FREE
      return cpp_res;
    };
    return NAME_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return NAME_new_();
  }
}

extern "C" NAME NAME_new_func_payload(
    int (*fun)(RCARGS_UNNAMED, CARGS_UNNAMED, void*),
    void* payload,
    void (*dtor)(void*)) {
  try {
    std::shared_ptr<void> cpp_payload = nullptr;
    if (dtor) {
      cpp_payload = std::shared_ptr<void>(payload, dtor);
    } else {
      cpp_payload = std::shared_ptr<void>(payload, [](void*) {});
    }
    auto cpp_closure = [fun, cpp_payload](CPPARGS_TYPE_NAME) {
      CPPARGS_TO_CARGS
      RCARGS_NEW
      auto status = fun(RCARGS_UNTYPED, CARGS_UNTYPED, cpp_payload.get());
      CARGS_FREE
      if(status) {
        RCARGS_FREE
        throw std::runtime_error("NAME returned a non-zero value");
      }
      RCARGS_TO_CPP
      RCARGS_FREE
      return cpp_res;
    };
    return NAME_new_(cpp_closure);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return NAME_new_();
  }
}

extern "C" int NAME_apply(RCARGS, NAME cls, CARGS) {
  try {
    ASSIGN_CLS_TO_RCARGS
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
"""

priv_code = None

decl_begin = r"""/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_CLOSURE_H
#define MLX_CLOSURE_H

#include "mlx/c/array.h"
#include "mlx/c/map.h"
#include "mlx/c/optional.h"
#include "mlx/c/stream.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_closure Closures
 * MLX closure objects.
 */
/**@{*/
"""

decl_end = r"""
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

#include "mlx/c/closure.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
"""

impl_end = """
"""

priv_begin = r"""/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_CLOSURE_PRIVATE_H
#define MLX_CLOSURE_PRIVATE_H

#include "mlx/c/closure.h"
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
    code = priv_code
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
        "mlx_closure",
        "std::vector<mlx::core::array>",
        ["std::vector<mlx::core::array>"],
    )
)
if args.implementation:
    print(
        r"""
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
      if(status) {
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
"""
    )
elif args.private:
    pass
else:
    print(
        r"""
mlx_closure mlx_closure_new_unary(int (*fun)(mlx_array*, const mlx_array));
    """
    )
print(
    generate(
        code,
        "mlx_closure_kwargs",
        "std::vector<mlx::core::array>",
        [
            "std::vector<mlx::core::array>",
            "std::unordered_map<std::string, mlx::core::array>",
        ],
    )
)
print(
    generate(
        code,
        "mlx_closure_value_and_grad",
        "std::pair<std::vector<mlx::core::array>, std::vector<mlx::core::array>>",
        ["std::vector<mlx::core::array>"],
    )
)
print(
    generate(
        code,
        "mlx_closure_custom",
        "std::vector<mlx::core::array>",
        ["std::vector<mlx::core::array>"] * 3,
    )
)
print(
    generate(
        code,
        "mlx_closure_custom_jvp",
        "std::vector<mlx::core::array>",
        [
            "std::vector<mlx::core::array>",
            "std::vector<mlx::core::array>",
            "std::vector<int>",
        ],
    )
)
print(
    generate(
        code,
        "mlx_closure_custom_vmap",
        "std::pair<std::vector<mlx::core::array>, @std::vector<int>>",
        ["std::vector<mlx::core::array>", "std::vector<int>"],
    )
)
if args.private:
    print(
        """
     """
    )

print(end)
