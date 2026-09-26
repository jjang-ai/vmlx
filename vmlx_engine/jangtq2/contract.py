"""CPU-only validation of the supported JANGTQ2 checkpoint contract."""
import math
import struct

CUBIC_PARAMS = {
    2: (0.8929999999999999, 0.05065),
    3: (0.47124999999999995, 0.011599999999999997),
    4: (0.2405, 0.002100000000000002),
}
ROTATIONS = ("none", "hadamard32")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _fail(message):
    raise ValueError("jangtq2: " + message)


def _f32(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail("codebook values must be finite numbers")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error):
        _fail("codebook value is outside float32 range")


def validate_format(config):
    """Return whether this is v2; ordinary/v1 declarations retain their route."""
    declaration = config.get("jangtq")
    entries = config.get("quantization")
    has_v2 = isinstance(entries, dict) and any(
        isinstance(v, dict) and v.get("mode") == "jangtq2" for v in entries.values()
    )
    if declaration is None:
        if has_v2:
            _fail("jangtq2 entries require a format declaration")
        return False
    if not isinstance(declaration, dict):
        _fail("format declaration must be an object")
    version = declaration.get("version")
    if type(version) is not int:
        _fail("version must be an integer")
    if version == 1 and not has_v2:
        return False
    if version != 2:
        _fail("unsupported format version")
    for key, expected in (("packing", "lsb-bitstream"),
                          ("scale_dtype", "float16"),
                          ("codebook_family", "odd-cubic")):
        if declaration.get(key) != expected:
            _fail("unsupported " + key)
    if declaration.get("rotation", "none") not in ROTATIONS:
        _fail("unsupported default rotation")
    books = declaration.get("codebooks")
    if not isinstance(books, dict) or set(books) != {"2", "3", "4"}:
        _fail("codebooks must declare supported widths 2, 3 and 4")
    for bits, (alpha, beta) in CUBIC_PARAMS.items():
        book = books[str(bits)]
        if not isinstance(book, dict):
            _fail("codebook must be an object")
        for key, expected in (("alpha", alpha), ("beta", beta)):
            value = book.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
                _fail("codebook coefficient differs from runtime")
        levels = book.get("levels")
        if not isinstance(levels, list) or len(levels) != 1 << bits:
            _fail("codebook levels have wrong length")
        for index, level in enumerate(levels):
            u = index - ((1 << bits) - 1) / 2
            if _f32(level) != _f32(u * (alpha + beta * u * u)):
                _fail("codebook level differs from runtime")
    return True


def canonical_path(path):
    for prefix in ("language_model.model.", "model.language_model.", "language_model."):
        if path.startswith(prefix):
            return "model." + path[len(prefix):]
    return path


def projection_contract(config):
    if not validate_format(config):
        return {}
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        _fail("per-module quantization is required")
    stacks = {}
    default_rotation = config["jangtq"].get("rotation", "none")
    for path, entry in quant.items():
        if not isinstance(entry, dict) or entry.get("mode") != "jangtq2":
            continue
        if not isinstance(path, str) or ".switch_mlp." not in path:
            _fail("unrecognized routed projection path")
        path = canonical_path(path)
        stack, projection = path.rsplit(".", 1)
        if projection not in PROJECTIONS or not stack.endswith(".switch_mlp"):
            _fail("unrecognized routed projection")
        bits = entry.get("bits")
        rotation = entry.get("rotation", default_rotation)
        if type(bits) is not int or bits not in CUBIC_PARAMS or rotation not in ROTATIONS:
            _fail("unsupported projection bits or rotation")
        normalized = {"bits": bits, "rotation": rotation}
        prior = stacks.setdefault(stack, {}).get(projection)
        if prior is not None and prior != normalized:
            _fail("conflicting routed projection aliases")
        stacks[stack][projection] = normalized
    if not stacks:
        _fail("no routed projections declared")
    for stack, projections in stacks.items():
        if set(projections) != set(PROJECTIONS):
            _fail("incomplete routed stack " + stack)
        if projections["gate_proj"] != projections["up_proj"]:
            _fail("gate/up bits or rotation differs")
    return stacks


def alias_runtime_quant_keys(quant, prefix="language_model."):
    """Validate all aliases before adding any; never overwrite a conflict."""
    if not isinstance(quant, dict):
        _fail("quantization must be an object")
    additions = {}
    for key, value in quant.items():
        if not isinstance(value, dict) or value.get("mode") == "jangtq2":
            continue
        if isinstance(key, str) and (key.startswith("model.") or key == "lm_head"):
            alias = prefix + key
            if alias in quant and quant[alias] != value:
                _fail("conflicting quantization alias " + alias)
            if alias not in quant:
                additions[alias] = dict(value)
    quant.update(additions)
    return len(additions)
