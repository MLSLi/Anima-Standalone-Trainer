#!/usr/bin/env python3
"""3D RoPE Benchmark — Triton fused vs PyTorch reference (triton.testing.do_bench).

Matches Anima's actual rope_emb format: [N, 1, 1, D] with H-aware indexing.
"""

import sys; sys.path.insert(0, '.')
import torch, triton.testing
from triton_ops.lora.fused_rope_3d import rope_ref_anima, fused
from triton_ops.config import detect_gpu, print_gpu_info


CONFIGS = [
    # (label,            B, H,    N, D)
    ("N=64  (tiny)",     1, 16,   64, 128),
    ("N=256 (256²)",     1, 16,  256, 128),
    ("N=484 (362²)",     1, 16,  484, 128),
    ("N=1024(512²)",     1, 16, 1024, 128),
    ("N=2304(768²)",     1, 16, 2304, 128),
    ("N=1024,B=2",       2, 16, 1024, 128),
    ("N=1024,B=4",       4, 16, 1024, 128),
    ("N=1024,B=8",       8, 16, 1024, 128),
]


def benchmark(warmup_ms=50, rep_ms=200):
    gpu = detect_gpu()
    print_gpu_info(gpu)
    print()
    print(f"{'Config':<20s} {'PyTorch ref':>12s} {'Triton fused':>12s} {'speedup':>8s}")
    print("-" * 54)

    dev = torch.device("cuda"); dtype = torch.bfloat16

    for label, B, H, N, D in CONFIGS:
        q = torch.randn(B, N, H, D, dtype=dtype, device=dev) * 0.1
        k = torch.randn(B, N, H, D, dtype=dtype, device=dev) * 0.1
        r = torch.randn(N, 1, 1, D, device=dev) * 0.1  # rope_emb [N,1,1,D]

        fused(q, k, r[:N])  # warmup

        t_ref = triton.testing.do_bench(
            lambda: rope_ref_anima(q, k, r), warmup=warmup_ms, rep=rep_ms)
        t_tri = triton.testing.do_bench(
            lambda: fused(q, k, r[:N]), warmup=warmup_ms, rep=rep_ms)

        print(f"{label:<20s} {t_ref*1e3:>11.0f}us {t_tri*1e3:>11.0f}us {t_ref/t_tri:>7.2f}x")

    # Detailed stats for Anima 512²
    print(f"\n--- N=1024 (Anima 512²) detailed ---")
    N = 1024
    q = torch.randn(1, N, 16, 128, dtype=dtype, device=dev) * 0.1
    k = torch.randn(1, N, 16, 128, dtype=dtype, device=dev) * 0.1
    r = torch.randn(N, 1, 1, 128, device=dev) * 0.1
    fused(q, k, r[:N])  # warmup

    for label, fn in [
        ("PyTorch ref", lambda: rope_ref_anima(q, k, r)),
        ("Triton fused", lambda: fused(q, k, r[:N])),
    ]:
        all_m = triton.testing.do_bench(fn, warmup=50, rep=500, return_mode='all')
        m = torch.tensor(all_m)
        print(f"  {label}:")
        print(f"    mean={m.mean()*1e3:.0f}us  median={m.median()*1e3:.0f}us  std={m.std()*1e3:.0f}us")
        print(f"    n={len(all_m)}  P25={torch.quantile(m,0.25)*1e3:.0f}us  P75={torch.quantile(m,0.75)*1e3:.0f}us")


if __name__ == "__main__":
    benchmark()
