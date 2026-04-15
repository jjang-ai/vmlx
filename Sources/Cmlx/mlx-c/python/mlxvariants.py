import sys


def _pretty_string_def(d):
    txt = []
    txt.append(d["return_t"])
    txt.append(d["namespace"] + "::" + d["name"])
    txt.append("(")
    args = []
    for i in range(len(d["params_t"])):
        args.append(d["params_t"][i] + " " + (d["params_name"][i] or ""))
    txt.append(", ".join(args))
    txt.append(")")
    return " ".join(txt)


def _make_variant_suffixes(name, defs, variants):
    if len(defs) > 1:
        print("OVL", file=sys.stderr)
    if name in variants:
        variants = variants[name]
        if len(variants) != len(defs):
            print("function overloads length:", len(defs), file=sys.stderr)
            for i, d in enumerate(defs):
                print(i, _pretty_string_def(d), file=sys.stderr)
            print("namings length:", len(variants), file=sys.stderr)
            for i, v in enumerate(variants):
                print(i, v, file=sys.stderr)
            raise RuntimeError("function overloads and namings do not match")
        for i, d in enumerate(defs):
            print("OVL", i, _pretty_string_def(d), " -> ", variants[i], file=sys.stderr)
        newdefs = []
        for i, d in enumerate(defs):
            v = variants[i]
            if v is not None:
                # do we need to specify variant name?
                if v != "":
                    d["variant"] = v
                newdefs.append(d)
        return newdefs
    else:
        if len(defs) > 1:
            for i, d in enumerate(defs):
                print(
                    "OVL",
                    i,
                    _pretty_string_def(d),
                    " -> ",
                    "" if i == 0 else "None",
                    file=sys.stderr,
                )
        return [defs[0]]


def mlx_core(name, defs):
    variants = {
        "arange": ["", None, None, None, None, None, None, None, None],
        "eye": ["", None, None, None, None],
        "tri": ["", None],
        "flatten": ["", None],
        "squeeze": ["axes", "axis", ""],
        "expand_dims": ["axes", ""],
        "slice": ["", None, "dynamic", None],
        "slice_update": ["", None, "dynamic"],
        "split": ["", "sections", None, None],
        "concatenate": ["axis", ""],
        "stack": ["axis", ""],
        "repeat": ["axis", ""],
        "transpose": ["axes", None, ""],
        "all": ["axes", "axis", "", None],
        "any": ["axes", "axis", "", None],
        "sum": ["axes", "axis", "", None],
        "mean": ["axes", "axis", "", None],
        "var": ["axes", "axis", "", None],
        "std": ["axes", "axis", "", None],
        "prod": ["axes", "axis", "", None],
        "max": ["axes", "axis", "", None],
        "min": ["axes", "axis", "", None],
        "argmax": ["axis", "", None],
        "argmin": ["axis", "", None],
        "load": ["reader", ""],
        "load_safetensors": ["reader", ""],
        "pad": ["", None, None, "symmetric"],
        "save": ["writer", ""],
        "save_safetensors": ["writer", ""],
        "gather": ["", "single"],
        "scatter": ["", "single"],
        "scatter_add": ["", "single"],
        "scatter_min": ["", "single"],
        "scatter_prod": ["", "single"],
        "scatter_max": ["", "single"],
        "argpartition": ["axis", ""],
        "partition": ["axis", ""],
        "argsort": ["axis", ""],
        "sort": ["axis", ""],
        "topk": ["axis", ""],
        "take": ["axis", None, "", None],
        "roll": [None, None, "axis", "axes", None, ""],
        "logsumexp": ["axes", "axis", "", None],
        "softmax": ["axes", "axis", ""],
        "tensordot": ["", "axis"],
        "array_equal": ["", None],
        "round": ["", None],
        "trace": ["", None, None],
        "export_function": [None, "", "kwargs"],
    }
    return _make_variant_suffixes(name, defs, variants)


def mlx_core_linalg(name, defs):
    variants = {"norm": ["", None, "matrix", None, "l2", None]}
    return _make_variant_suffixes(name, defs, variants)


def mlx_core_random(name, defs):
    variants = {
        "categorical": ["shape", "num_samples", ""],
        "permutation": ["", "arange"],
        "split": ["num", ""],
        "uniform": ["", None, None, None],
        "normal": ["broadcast", "", None, None, None],
    }
    return _make_variant_suffixes(name, defs, variants)


def mlx_core_fast(name, defs):
    variants = {
        "rope": ["", "dynamic"],
    }
    return _make_variant_suffixes(name, defs, variants)


def mlx_core_detail(name, defs):
    if name not in [
        "compile",
        "compile_clear_cache",
        "compile_erase",
        "vmap_replace",
        "vmap_trace",
    ]:
        defs = []

    return defs
