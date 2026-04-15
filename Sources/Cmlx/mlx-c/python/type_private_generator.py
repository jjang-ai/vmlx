ctor_copy_code = """
inline CTYPE CTYPE_new_(const CPPTYPE& s) {
  return CTYPE({new CPPTYPE(s)});
}
"""

ctor_code = """
inline CTYPE CTYPE_new_() {
  return CTYPE({nullptr});
}
CTOR_COPY_CODE
inline CTYPE CTYPE_new_(CPPTYPE&& s) {
  return CTYPE({new CPPTYPE(std::move(s))});
}
"""

set_copy_code = """
inline CTYPE& CTYPE_set_(CTYPE& d, const CPPTYPE& s) {
  if (d.ctx) {
    *static_cast<CPPTYPE*>(d.ctx) = s;
  } else {
    d.ctx = new CPPTYPE(s);
  }
  return d;
}

inline CTYPE& CTYPE_set_(CTYPE& d, CPPTYPE&& s) {
  if (d.ctx) {
    *static_cast<CPPTYPE*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new CPPTYPE(std::move(s));
  }
  return d;
}
"""

set_no_copy_code = """
inline CTYPE& CTYPE_set_(CTYPE& d, CPPTYPE&& s) {
  if (d.ctx) {
    delete static_cast<CPPTYPE*>(d.ctx);
  }
  d.ctx = new CPPTYPE(std::move(s));
  return d;
}
"""

code = """
SET_CODE

inline CPPTYPE& CTYPE_get_(CTYPE d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty CTYPE");
  }
  return *static_cast<CPPTYPE*>(d.ctx);
}

inline void CTYPE_free_(CTYPE d) {
  if (d.ctx) {
    delete static_cast<CPPTYPE*>(d.ctx);
  }
}
"""


def generate(
    ctype, cpptype, ctor=True, no_copy=False, code=code, ctor_code=ctor_code, using=None
):
    if using:
        print(" ".join(["using", using, " = ", cpptype, ";"]))

    if ctor:
        code = ctor_code + code
    if no_copy:
        code = code.replace("CTOR_COPY_CODE", "")
        code = code.replace("SET_CODE", set_no_copy_code)
    else:
        code = code.replace("CTOR_COPY_CODE", ctor_copy_code)
        code = code.replace("SET_CODE", set_copy_code)

    code = code.replace("CTYPE", ctype)
    code = code.replace("CPPTYPE", using if using else cpptype)
    return code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MLX C private type generator", add_help=False)
    parser.add_argument("--ctype", type=str)
    parser.add_argument("--cpptype", type=str)
    parser.add_argument("--no-copy", default=False, action="store_true")
    parser.add_argument("--include", default="", type=str)
    parser.add_argument("--using", default="", type=str)
    args = parser.parse_args()

    if args.include:
        short_ctype = args.include
    else:
        short_ctype = args.ctype.replace("mlx_", "")
    print("/* Copyright © 2023-2024 Apple Inc.                   */")
    print("/*                                                    */")
    print("/* This file is auto-generated. Do not edit manually. */")
    print("/*                                                    */")
    print()
    print("#ifndef MLX_" + short_ctype.upper() + "_PRIVATE_H")
    print("#define MLX_" + short_ctype.upper() + "_PRIVATE_H")
    print()
    print('#include "mlx/c/' + short_ctype + '.h"')
    print('#include "mlx/mlx.h"')
    ctypes = args.ctype.split(";")
    cpptypes = args.cpptype.split(";")
    usings = args.using.split(";")
    for i in range(len(ctypes)):
        print(
            generate(
                ctypes[i],
                cpptypes[i],
                no_copy=args.no_copy,
                using=usings[i] if args.using else None,
            )
        )
    print("#endif")
