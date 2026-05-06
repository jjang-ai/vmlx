# mlx_lm vendored patches

Each `*.patched.py` file in this directory is a **controlled fork** of an
upstream `mlx_lm` model file (Apache-2.0, attribution preserved in-file).

`panel/scripts/bundle-python.sh` copies these into the bundled site-packages
after installing `mlx-lm` from PyPI, replacing the upstream copies with our
patched versions. The script hard-fails if any patch source is missing or if
the post-install file does not contain the expected sentinel
(`q_sdpa.*astype(mx.float32)`).

## Why these patches exist

All three are MLA family L==1 decode fixes: cast `q_nope` / `k` / `v` /
`mask` to fp32 before `scaled_dot_product_attention` so the absorb branch
does not produce logit drift on bfloat16 weights. Upstream mlx_lm carried
this fix in some 0.31.x releases but regressed it in 0.31.3, so we ship the
fix ourselves.

| File | Upstream module | Sentinel marker |
|------|-----------------|-----------------|
| `deepseek_v3.patched.py` | `mlx_lm.models.deepseek_v3` | `JANG fast fix` comment + `q_sdpa = q_nope.astype(mx.float32)` |
| `deepseek_v32.patched.py` | `mlx_lm.models.deepseek_v32` | `q_sdpa = q_nope.astype(mx.float32)` |
| `mistral4.patched.py` | `mlx_lm.models.mistral4` | `q_sdpa = q_nope.astype(mx.float32)` |

## Updating these patches

1. `pip install --upgrade mlx-lm` in a clean venv.
2. Diff the upstream file against the corresponding `*.patched.py` here.
3. Re-apply the fp32 SDPA cast on top of the new upstream version.
4. Save back here and re-run `bundle-python.sh`.

## Why not a runtime monkeypatch?

Mutating third-party module classes at runtime is fragile (import-order
dependent, breaks signed-bundle integrity checks, depends on
`is_patched()` heuristics that drift). A build-time controlled fork is
deterministic, signs cleanly, and produces the same artifact every build.
