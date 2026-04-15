# Copyright © 2023-2024 Apple Inc.

import cxxheaderparser
from cxxheaderparser.simple import parse_string
import argparse
import os
import re

parser = argparse.ArgumentParser("MLX C bindings generator", add_help=False)
parser.add_argument("--header", type=str)
parser.add_argument("--implementation", default=False, action="store_true")
parser.add_argument("--language", default="C", type=str)
parser.add_argument("--docstring", default="", type=str)
parser.add_argument("--headername", default="", type=str)
args = parser.parse_args()

if args.headername:
    headername = args.headername
else:
    headername = os.path.basename(args.header)
    if headername.endswith(".h"):
        headername = headername[:-2]
    else:
        raise RuntimeError("are you sure you are providing a header?")


def getname(t):
    if type(t) == cxxheaderparser.types.TemplateArgument:
        return getname(t.arg)
    elif type(t) == cxxheaderparser.types.Reference:
        return getname(t.ref_to)
    elif type(t) == cxxheaderparser.types.MoveReference:
        return getname(t.moveref_to)
    elif type(t) == cxxheaderparser.types.PQName:
        res = []
        for s in t.segments:
            res.append(getname(s))
        return "::".join(res)
    elif type(t) == cxxheaderparser.types.FundamentalSpecifier:
        return t.name
    elif type(t) == cxxheaderparser.types.NameSpecifier:
        res = t.name
        if t.specialization is not None:
            res += getname(t.specialization)
        return res
    elif type(t) == cxxheaderparser.types.Type:
        return getname(t.typename)
    elif type(t) == cxxheaderparser.types.TemplateSpecialization:
        res = []
        for s in t.args:
            res.append(getname(s))
        return "<" + ", ".join(res) + ">"
    elif type(t) == cxxheaderparser.types.FunctionType:
        return_t = getname(t.return_type)
        params_t = []
        for p in t.parameters:
            params_t.append(getname(p.type))
        res = return_t + "(" + ",".join(params_t) + ")"
        return res
    elif type(t) == cxxheaderparser.types.Pointer:
        # circumvents parser crashing on pointers
        res = "*(" + getname(t.ptr_to) + ")"
        return res

    raise RuntimeError("unsupported type: " + str(t))


def get_default_value(d):
    if d is None:
        return d
    res = []
    for tok in d.tokens:
        res.append(tok.value)
    return "".join(res)


funcs = {}
enums = {}


def preprocess_header(content):
    """Simple preprocessor that strips MLX_API macro without resolving includes."""
    # Remove MLX_API macro (appears as MLX_API or MLX_API followed by space)
    content = re.sub(r"\bMLX_API\s*", "", content)
    return content


for header in args.header.split(";"):
    # Read file and preprocess to strip MLX_API
    with open(header, "r") as f:
        content = f.read()
    content = preprocess_header(content)
    Z = parse_string(content)

    def process_namespace(l, namespace, funcs, enums):
        namespace = namespace.lstrip("::")
        for e in l.enums:
            name = getname(e.typename)
            values = [v.name for v in e.values]
            enums[namespace + "::" + name] = {
                "name": name,
                "namespace": namespace,
                "values": values,
            }

        for f in l.functions:
            name = getname(f.name)
            if name.startswith("operator"):
                continue
            params_t = []
            params_name = []
            params_default = []
            return_t = getname(f.return_type)
            if return_t == "Stream":  # unsupported
                continue
            for p in f.parameters:
                params_t.append(getname(p.type))
                params_name.append(p.name)
                params_default.append(get_default_value(p.default))
            func = {
                "name": name,
                "params_t": params_t,
                "params_name": params_name,
                "return_t": return_t,
                "namespace": namespace,
                "params_default": params_default,
            }
            ns_name = namespace + "::" + name
            if ns_name in funcs:
                funcs[ns_name].append(func)
            else:
                funcs[ns_name] = [func]

        for subnamespace in l.namespaces:
            process_namespace(
                l.namespaces[subnamespace],
                namespace + "::" + subnamespace,
                funcs,
                enums,
            )

    process_namespace(Z.namespace, "", funcs, enums)

if args.language == "C":
    from c import generate
else:
    raise RuntimeError("Unsupported language")

generate(
    funcs,
    enums,
    header,
    headername,
    args.implementation,
    args.docstring,
)
