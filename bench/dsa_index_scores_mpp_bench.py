"""DSA indexer pool scores: fp32 simdgroup-MMA kernel vs the M5 matrix-unit kernel (H=32, D=128).

Times the median of 5 runs per shape, over --rounds rounds (other GPU work on the machine shows up as spread):

  fp32   dsa_pool_index_scores on fp32 operands (today's DeepSeek-V4.1 call)
  mpp    dsa_pool_index_scores_mpp on bf16 operands
  wrap   dsa_pool_index_scores(..., bf16_operands=True) on fp32 operands (includes the bf16 casts)

    PYTHONPATH=. python bench/dsa_index_scores_mpp_bench.py [--rounds 3]
"""
import argparse
import time

import mlx.core as mx

from vmlx_engine.metal.dsa_index_scores import dsa_pool_index_scores
from vmlx_engine.metal.dsa_index_scores_mpp import available, dsa_pool_index_scores_mpp

H, D = 32, 128
SHAPES = [(2048, 25_000), (2048, 50_000), (2048, 100_000), (256, 1_000_000)]


def median_ms(fn, n=5):
    mx.eval(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    print(mx.device_info()["device_name"], "MLX", mx.__version__, "matrix units:", available())
    res = {}
    for S, T in SHAPES:
        mx.random.seed(0)
        q = mx.random.normal((1, S, H, D)).astype(mx.bfloat16)
        k = mx.random.normal((1, T, D)).astype(mx.bfloat16)
        w = mx.random.normal((1, S, H)).astype(mx.bfloat16).astype(mx.float32)
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        mx.eval(q, k, w, qf, kf)
        runs = {
            "fp32": lambda: dsa_pool_index_scores(qf, kf, w, 1.0),
            "mpp": lambda: dsa_pool_index_scores_mpp(q, k, w, 1.0),
            "wrap": lambda: dsa_pool_index_scores(qf, kf, w, 1.0, bf16_operands=True),
        }
        for r in range(args.rounds):
            for name, fn in runs.items():
                res.setdefault((S, T, name), []).append(median_ms(fn))
        del q, k, w, qf, kf, runs
        mx.clear_cache()
    print(f"{'S':>5} {'T':>8}  {'fp32 ms':>13}  {'mpp ms':>13}  {'wrap ms':>13}  {'speedup':>11}  mpp TFLOPS")
    for S, T in SHAPES:
        f, m, wr = (res[(S, T, n)] for n in ("fp32", "mpp", "wrap"))
        rng = lambda v: f"{min(v):5.1f}-{max(v):5.1f}"  # noqa: E731
        sp = sorted(a / b for a, b in zip(f, m))
        fl = 2 * S * T * H * D
        print(f"{S:>5} {T:>8}  {rng(f):>13}  {rng(m):>13}  {rng(wr):>13}  {sp[0]:4.2f}-{sp[-1]:4.2f}x"
              f"  {fl / max(m) / 1e9:5.0f}-{fl / min(m) / 1e9:5.0f}")


if __name__ == "__main__":
    main()
