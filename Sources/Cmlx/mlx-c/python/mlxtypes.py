import string

types = []

for t in [
    ["mlx_array", "mlx::core::array", "array"],
    ["mlx_vector_int", "@std::vector<int>", "@std::vector<int>"],
    ["mlx_vector_string", "std::vector<std::string>", "std::vector<std::string>"],
    ["mlx_vector_array", "std::vector<mlx::core::array>", "std::vector<array>"],
    [
        "mlx_stream",
        "mlx::core::Stream",
    ],
    [
        "mlx_map_string_to_array",
        "std::unordered_map<std::string, mlx::core::array>",
        "std::unordered_map<std::string, array>",
    ],
    [
        "mlx_map_string_to_string",
        "std::unordered_map<std::string, std::string>",
        "std::unordered_map<std::string, std::string>",
    ],
    [
        "mlx_stream",
        "mlx::core::Stream",
        "StreamOrDevice",
    ],
    [
        "mlx_distributed_group",
        "mlx::core::distributed::Group",
        "Group",
    ],
    [
        "mlx_closure",
        "std::function<std::vector<array>(std::vector<array>)>",
    ],
    [
        "mlx_closure_value_and_grad",
        "std::function<std::pair<std::vector<array>, std::vector<array>>(const std::vector<array>&)>",
        "ValueAndGradFn",
    ],
    [
        "mlx_closure_custom",
        "std::function<std::vector<mlx::core::array>(std::vector<mlx::core::array>,std::vector<mlx::core::array>,std::vector<mlx::core::array>)>",
        "std::function<std::vector<array>(std::vector<array>,std::vector<array>,std::vector<array>)>",
    ],
    [
        "mlx_closure_custom_jvp",
        "std::function<std::vector<mlx::core::array>(std::vector<mlx::core::array>,std::vector<mlx::core::array>,std::vector<int>)>",
        "std::function<std::vector<array>(std::vector<array>,std::vector<array>,std::vector<int>)>",
    ],
    [
        "mlx_closure_custom_vmap",
        "std::function<std::pair<std::vector<mlx::core::array>, std::vector<int>>(std::vector<mlx::core::array>,std::vector<int>)>",
        "std::function<std::pair<std::vector<array>, std::vector<int>>(std::vector<array>,std::vector<int>)>",
    ],
]:
    if len(t) == 2:
        ctype, cpptype = t
        alt = None
    else:
        ctype, cpptype, alt = t
    types.append(
        {
            "c": ctype,
            "cpp": cpptype,
            "alt": alt,
            "free": lambda s, ctype=ctype: ctype + "_free(" + s + ")",
            "cpp_to_c": lambda s, ctype=ctype: ctype + "_new_(" + s + ")",
            "c_to_cpp": lambda s, ctype=ctype: ctype + "_get_(" + s + ")",
            "return": lambda s: "RETURN_MLX_C_PTR(" + s + ")",
            "c_assign_from_cpp": lambda d, s, returned=True, ctype=ctype: ctype
            + "_set_("
            + ("*" if returned else "")
            + d
            + ", "
            + s
            + ")",
            "c_arg": lambda s, untyped=False, ctype=ctype: (
                s if untyped else ("const " + ctype + " " + s).strip()
            ),
            "c_return_arg": lambda s, untyped=False, ctype=ctype: (
                ("&" if untyped else ctype + "* ") + s
            ).strip(),
            "c_new": lambda s, ctype=ctype: "auto " + s + " = " + ctype + "_new_()",
            "cpp_arg": lambda s, cpptype=cpptype: (
                "const " + cpptype.replace("@", "") + "& " + s
            ).strip(),
        }
    )


def find_cpp_type(cpp_type):
    for t in types:
        if t["cpp"] == cpp_type:
            return t
    raise RuntimeError("Could not find type " + cpp_type)


def register_raw_vector_type(cpptype, alt=None):
    types.append(
        {
            # "c": "mlx_vector_" + cpptype, # DEBUG: ONLY FOR RETURN?
            "alt": alt,  # "alt": "std::vector<" + cpptype + ">", # DEBUG: ONLY FOR RETURN?
            "cpp": "std::vector<" + cpptype + ">",
            "free": lambda s: "",
            "c_to_cpp": lambda s, cpptype=cpptype: "std::vector<"
            + cpptype
            + ">("
            + s
            + ", "
            + s
            + " + "
            + s
            + "_num"
            + ")",
            "c_assign_from_cpp": lambda d, s, returned=True: d
            + " = "
            + s
            + ".data(); "
            + d
            + "_num = "
            + s
            + ".size()",
            "c_arg": lambda s, untyped=False, cpptype=cpptype: (
                (s + ", " + s + "_num")
                if untyped
                else ("const " + cpptype + "* " + s + ", size_t " + s + "_num").strip()
            ),
            "c_new": lambda s, cpptype=cpptype: "const "
            + cpptype
            + "* "
            + s
            + "= nullptr;  size_t "
            + s
            + "_num = 0",
            # "c_return_arg": lambda s, untyped=False, ctype=ctype: (
            #     ("" if untyped else ctype + " ") + s
            # ).strip(),
            # "c_new": lambda s, ctype=ctype: "auto " + s + " = new " + ctype + "_()",
            "cpp_arg": lambda s, cpptype=cpptype: (
                "const std::vector<" + cpptype + ">& " + s
            ).strip(),
        }
    )


def register_small_vector_type(etype, cpptype, alt=None):
    types.append(
        {
            # "c": "mlx_vector_" + cpptype, # DEBUG: ONLY FOR RETURN?
            "alt": alt,
            "cpp": cpptype,
            "free": lambda s: "",
            "c_to_cpp": lambda s, cpptype=cpptype: cpptype
            + "("
            + s
            + ", "
            + s
            + " + "
            + s
            + "_num"
            + ")",
            "c_assign_from_cpp": lambda d, s, returned=True: d
            + " = "
            + s
            + ".data(); "
            + d
            + "_num = "
            + s
            + ".size()",
            "c_arg": lambda s, untyped=False, etype=etype: (
                (s + ", " + s + "_num")
                if untyped
                else ("const " + etype + "* " + s + ", size_t " + s + "_num").strip()
            ),
            "c_new": lambda s, etype=etype: "const "
            + etype
            + "* "
            + s
            + "= nullptr;  size_t "
            + s
            + "_num = 0",
            # "c_return_arg": lambda s, untyped=False, ctype=ctype: (
            #     ("" if untyped else ctype + " ") + s
            # ).strip(),
            # "c_new": lambda s, ctype=ctype: "auto " + s + " = new " + ctype + "_()",
            "cpp_arg": lambda s, cpptype=cpptype: (
                "const " + cpptype + "& " + s
            ).strip(),
        }
    )


register_small_vector_type("int", "mlx::core::Shape", "Shape")
register_small_vector_type("int64_t", "mlx::core::Strides", "Strides")
register_raw_vector_type("int")
register_raw_vector_type("size_t")
register_raw_vector_type("uint64_t")


def register_optional_raw_vector_type(cpptype):
    cpp = "std::optional<std::vector<" + cpptype + ">>"

    def free(s):
        return ""

    def c_to_cpp(s):
        return "".join(
            [
                "(",
                s,
                "? std::make_optional(std::vector<",
                cpptype,
                ">(",
                s,
                ", ",
                s,
                " + ",
                s,
                "_num))",
                " : std::nullopt)",
            ]
        )

    def c_assign_from_cpp(d, s, returned=True):
        return "".join(
            [
                "if(",
                s,
                ".has_value()) {",
                d,
                " = ",
                s,
                ".data();",
                d,
                "_num = ",
                s,
                ".size();",
                "} else {",
                d,
                " = nullptr;",
                d,
                "_num = 0;",
                "}",
            ]
        )

    def c_arg(s, untyped=False):
        if untyped:
            return "".join([s, ", ", s, "_num"])
        else:
            return "".join(
                ["const ", cpptype, "*", s, "/* may be null */", ", size_t ", s, "_num"]
            )

    types.append(
        {
            "cpp": cpp,
            "free": free,
            "c_to_cpp": c_to_cpp,
            "c_assign_from_cpp": c_assign_from_cpp,
            "c_arg": c_arg,
        }
    )


register_optional_raw_vector_type("int")

# "c_arg": lambda s, untyped=False, cpptype=cpptype: (s + ", " + s + "_num")
#         if untyped
#         else ("const " + cpptype + "* " + s + ", size_t " + s + "_num").strip(),
#         "c_new": lambda s, cpptype=cpptype: "const "
#         + cpptype
#         + "* "
#         + s
#         + "= nullptr;  size_t "
#         + s
#         + "_num = 0",
#         # "c_return_arg": lambda s, untyped=False, ctype=ctype: (
#         #     ("" if untyped else ctype + " ") + s
#         # ).strip(),
#         # "c_new": lambda s, ctype=ctype: "auto " + s + " = new " + ctype + "_()",
#         "cpp_arg": lambda s, cpptype=cpptype: (
#             "const std::vector<" + cpptype + ">& " + s
#         ).strip()
#     }


def register_return_tuple_type(cpp_types, alts=[]):
    n = len(cpp_types)
    c_types = []
    alt_types = []
    c_to_cpps = []
    for cpp_type in cpp_types:
        typedef = find_cpp_type(cpp_type)
        c_types.append(typedef["c"])
        alt_types.append(typedef["alt"])
        c_to_cpps.append(typedef["c_to_cpp"])
    cpp_make_tuple = "std::make_pair" if n == 2 else "std::tie"
    cpp_tuple = "std::pair" if n == 2 else "std::tuple"
    types.append(
        {
            "cpp": cpp_tuple + "<" + ", ".join(cpp_types) + ">",
            "alt": [cpp_tuple + "<" + ", ".join(alt_types) + ">"] + alts,
            "c_to_cpp": lambda s: cpp_make_tuple
            + "("
            + ", ".join([c_to_cpps[i](s + "_" + str(i)) for i in range(n)])
            + ")",
            "c_return_arg": lambda s, untyped=False: ", ".join(
                [
                    ("&" if untyped else c_types[i] + "*")
                    + (" " + s + "_" + str(i) if s else "")
                    for i in range(n)
                ]
            ),
            "c_new": lambda s: "\n".join(
                [
                    "auto " + s + "_" + str(i) + " = " + ctype + "_new_();"
                    for i, ctype in enumerate(c_types)
                ]
            ),
            "free": lambda s: "\n".join(
                [
                    ctype + "_free(" + s + "_" + str(i) + ");"
                    for i, ctype in enumerate(c_types)
                ]
            ),
            "c_assign_from_cpp": lambda d, s, returned=True: "{ auto ["
            + ", ".join(["tpl_" + str(i) for i in range(n)])
            + "] = "
            + s
            + ";"
            + "\n".join(
                [
                    c_types[i]
                    + "_set_("
                    + ("*" if returned else "")
                    + d
                    + "_"
                    + str(i)
                    + ","
                    + "tpl_"
                    + str(i)
                    + ");"
                    for i in range(n)
                ]
            )
            + "}",
        }
    )


register_return_tuple_type(["mlx::core::array", "mlx::core::array"])
register_return_tuple_type(["mlx::core::array", "mlx::core::array", "mlx::core::array"])
register_return_tuple_type(
    ["std::vector<mlx::core::array>", "std::vector<mlx::core::array>"]
)
register_return_tuple_type(["std::vector<mlx::core::array>", "@std::vector<int>"])
register_return_tuple_type(
    [
        "std::unordered_map<std::string, mlx::core::array>",
        "std::unordered_map<std::string, std::string>",
    ],
    ["SafetensorsLoad"],
)

types.append(
    {
        "cpp": "void",
        "c_return_arg": lambda s: "",
        "c_assign_from_cpp": lambda d, s: s,
    }
)

types.append(
    {
        "cpp": "mlx::core::Dtype",
        "c": "mlx_dtype",
        "alt": "Dtype",
        "c_to_cpp": lambda s: "mlx_dtype_to_cpp(" + s + ")",
        "c_arg": lambda s, untyped=False: s if untyped else "mlx_dtype " + s,
        "c_return_arg": lambda s, untyped=False: s if untyped else "mlx_dtype* " + s,
        "c_new": lambda s: "mlx_dtype " + s,
        "free": lambda s: "",
        "c_assign_from_cpp": lambda d, s: d
        + " = "
        + "mlx_dtype_to_c((int)(("
        + s
        + ").val))",
    }
)

types.append(
    {
        "cpp": "mlx::core::CompileMode",
        "alt": "CompileMode",
        "c_to_cpp": lambda s: "mlx_compile_mode_to_cpp(" + s + ")",
        "c_arg": lambda s, untyped=False: s if untyped else "mlx_compile_mode " + s,
        "c_return_arg": lambda s, untyped=False: (
            s if untyped else "mlx_compile_mode* " + s
        ),
        "c_new": lambda s: "mlx_dtype " + s,
        "free": lambda s: "",
        "c_assign_from_cpp": lambda d, s: d
        + " = "
        + "mlx_compile_mode_to_c((int)(("
        + s
        + ").val))",
    }
)

types.append(
    {
        "cpp": "std::string",
        "alt": "std::string",
        "c_to_cpp": lambda s: "std::string(" + s + ")",
        "c_arg": lambda s, untyped=False: s if untyped else "const char* " + s,
        "c_return_arg": lambda s, untyped=False: s if untyped else "char** " + s,
        # "c_new": lambda s: "char* " + s,
        # "free": lambda s: "",
        "c_assign_from_cpp": lambda d, s: d + " = " + s + ".c_str()",
    }
)

types.append(
    {
        "cpp": "std::shared_ptr<io::Reader>",
        "c_to_cpp": lambda s: "mlx_io_reader_get_(" + s + ")",
        "c_arg": lambda s, untyped=False: s if untyped else "mlx_io_reader " + s,
    }
)

types.append(
    {
        "cpp": "std::shared_ptr<io::Writer>",
        "c_to_cpp": lambda s: "mlx_io_writer_get_(" + s + ")",
        "c_arg": lambda s, untyped=False: s if untyped else "mlx_io_writer " + s,
    }
)

for ctype in ["int", "size_t", "float", "double", "bool", "uint64_t", "uintptr_t"]:
    types.append(
        {
            "c": ctype,
            "cpp": ctype,
            "alt": None,
            "free": lambda s: "",
            "cpp_to_c": lambda s: s,
            "c_to_cpp": lambda s: s,
            "return": lambda s: "return" + s,
            "c_arg": lambda s, ctype=ctype: (ctype + " " + s).strip(),
            "cpp_arg": lambda s, ctype=ctype: (ctype + " " + s).strip(),
            "c_return_arg": lambda s, ctype=ctype: ctype + "* " + s,
            "c_assign_from_cpp": lambda d, s: "*" + d + " = " + s,
        }
    )
types[-1]["alt"] = "std::uintptr_t"

for cpptype in ["float", "int", "mlx::core::Dtype"]:
    typedef = find_cpp_type(cpptype)
    ctype = typedef["c"]
    alttype = typedef["alt"]

    opt_ctype = "mlx_optional_" + ctype.replace("mlx_", "")
    opt_cpptype = "std::optional<" + cpptype + ">"
    c_to_cpp = typedef["c_to_cpp"] if "c_to_cpp" in typedef else lambda s: s
    cpp_to_c = typedef["cpp_to_c"] if "cpp_to_c" in typedef else lambda s: s
    types.append(
        {
            "c": opt_ctype,
            "cpp": opt_cpptype,
            "alt": "std::optional<" + alttype + ">" if alttype else None,
            "free": lambda s: "",
            "cpp_to_c": lambda s, opt_ctype=opt_ctype, cpp_to_c=cpp_to_c: "("
            + s
            + ".has_value() ? "
            + opt_ctype
            + "_"
            + "({"
            + cpp_to_c(s + ".value()")
            + ", true}) : "
            + opt_ctype
            + "_({0, false}))",
            "c_to_cpp": lambda s, cpptype=cpptype, c_to_cpp=c_to_cpp: "("
            + s
            + ".has_value ? std::make_optional<"
            + cpptype
            + ">("
            + c_to_cpp(s + ".value")
            + ") : std::nullopt)",
            "return": lambda s: "return" + s,
            "c_arg": lambda s, opt_ctype=opt_ctype: (opt_ctype + " " + s).strip(),
            "cpp_arg": lambda s, opt_cpptype=opt_cpptype: (opt_cpptype + s).strip(),
        }
    )

types.append(
    {
        "cpp": "std::pair<int, int>",
        "alt": "std::pair<int, int>",
        "c_to_cpp": lambda s: "std::make_pair(" + s + "_0, " + s + "_1)",
        "c_arg": lambda s, untyped=False: (
            (s + "_0, " + s + "_1") if untyped else ("int " + s + "_0, int " + s + "_1")
        ),
        "c_return_arg": lambda s, untyped=False: (
            (s + "_0, " + s + "_1")
            if untyped
            else ("int* " + s + "_0, int* " + s + "_1")
        ),
        # "c_new": lambda s: "char* " + s,
        # "free": lambda s: "",
        "c_assign_from_cpp": lambda d, s: "std::tie(" + d + "_0, " + d + "_1) = " + s,
    }
)

types.append(
    {
        "cpp": "std::tuple<int, int, int>",
        "alt": "std::tuple<int, int, int>",
        "c_to_cpp": lambda s: "std::make_tuple(" + s + "_0, " + s + "_1," + s + "_2)",
        "c_arg": lambda s, untyped=False: (
            (s + "_0, " + s + "_1, " + s + "_2")
            if untyped
            else ("int " + s + "_0, int " + s + "_1, int " + s + "_2")
        ),
        "c_return_arg": lambda s, untyped=False: (
            (s + "_0, " + s + "_1, " + s + "_2")
            if untyped
            else ("int* " + s + "_0, int* " + s + "_1, int " + s + "_2")
        ),
        # "c_new": lambda s: "char* " + s,
        # "free": lambda s: "",
        "c_assign_from_cpp": lambda d, s: "std::tie("
        + d
        + "_0, "
        + d
        + "_1, "
        + d
        + "_2) = "
        + s,
    }
)


def register_optional_type(cpptype):
    typedef = find_cpp_type(cpptype)
    opt_t = {}
    for k in typedef:
        opt_t[k] = typedef[k]

    def c_arg(s):
        return "".join(
            [
                typedef["c_arg"](s),
                " /* may be null */",
            ]
        )

    def c_to_cpp(s):
        return (
            "("
            + s
            + ".ctx ? std::make_optional("
            + typedef["c_to_cpp"](s)
            + ") : std::nullopt)"
        )

    def c_assign_from_cpp(d, s):
        return "(" + s + ".has_value() ? " + s + ".value() : nullptr)"

    opt_t["cpp"] = "std::optional<" + typedef["cpp"] + ">"
    opt_t["alt"] = "std::optional<" + typedef["alt"] + ">"
    opt_t["c_arg"] = c_arg
    opt_t["c_to_cpp"] = c_to_cpp
    opt_t["c_assign_from_cpp"] = c_assign_from_cpp

    types.append(opt_t)


register_optional_type("mlx::core::array")
register_optional_type("mlx::core::distributed::Group")
register_optional_type(
    "std::function<std::vector<mlx::core::array>(std::vector<mlx::core::array>,std::vector<mlx::core::array>,std::vector<mlx::core::array>)>"
)
register_optional_type(
    "std::function<std::vector<mlx::core::array>(std::vector<mlx::core::array>,std::vector<mlx::core::array>,std::vector<int>)>"
)
register_optional_type(
    "std::function<std::pair<std::vector<mlx::core::array>, std::vector<int>>(std::vector<mlx::core::array>,std::vector<int>)>"
)

ctypes = {}
cpptypes = {}
alttypes = {}
for t in types:
    if "c" in t:
        ctype = t["c"]
        ctypes[ctype] = t

    if "cpp" in t:
        cpptype = t["cpp"]
        cpptypes[cpptype] = t

    if "alt" in t:
        alts = t["alt"]
        if alts is not None:
            if isinstance(alts, str):
                alts = [alts]
            for alttype in alts:
                alttypes[alttype] = t
