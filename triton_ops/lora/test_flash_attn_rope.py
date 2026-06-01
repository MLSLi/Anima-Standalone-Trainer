#!/usr/bin/env python3
"""RoPE + Attention fusion — accuracy & benchmark.

Tests: (1) fused_rope + SDPA end-to-end, (2) Triton FlashAttention standalone.
"""

import sys; sys.path.insert(0, '.')
import torch, triton.testing, math
from triton_ops.lora.flash_attn_rope import (
    attn_rope_ref, attn_rope_triton, flash_attn_triton,
)
from triton_ops.lora.fused_rope_3d import fused as fused_rope, rope_ref_anima
from triton_ops.config import detect_gpu, print_gpu_info

DEVICE = torch.device("cuda"); DTYPE = torch.bfloat16
WARMUP, REP = 50, 200


def test_accuracy():
    print("=" * 60)
    print("ACCURACY TESTS")
    print("=" * 60)

    all_ok = True

    # Test 1: fused_rope + SDPA vs reference RoPE + SDPA
    print("\n--- Test 1: attn_rope_ref vs attn_rope_triton ---")
    for B, H, N, D in [(1, 16, 256, 128), (1, 16, 1024, 128), (2, 16, 1024, 128)]:
        q = torch.randn(B, N, H, D, dtype=DTYPE, device=DEVICE) * 0.1
        k = torch.randn(B, N, H, D, dtype=DTYPE, device=DEVICE) * 0.1
        v = torch.randn(B, N, H, D, dtype=DTYPE, device=DEVICE) * 0.1
        r = torch.randn(N, 1, 1, D, device=DEVICE) * 0.1

        with torch.no_grad():
            o_ref = attn_rope_ref(q, k, v, r)
            o_tri = attn_rope_triton(q, k, v, r)

        ok = torch.allclose(o_tri.float(), o_ref.float(), rtol=1e-2, atol=1e-3)
        err = (o_tri.float() - o_ref.float()).abs().max().item()
        print(f"  B={B} H={H} N={N:>4} D={D}: {'✅' if ok else '❌'} err={err:.6f}")
        if not ok: all_ok = False

    # Test 2: Triton FlashAttention vs cuDNN SDPA
    print("\n--- Test 2: flash_attn_triton vs F.scaled_dot_product_attention ---")
    for B, H, N, D in [(1, 4, 256, 128), (1, 4, 1024, 128)]:
        q = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE) * 0.1
        k = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE) * 0.1
        v = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE) * 0.1
        sm = 1.0 / math.sqrt(D)

        with torch.no_grad():
            o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm)
            o_tri = flash_attn_triton(q, k, v, sm_scale=sm)

        ok = torch.allclose(o_tri.float(), o_ref.float(), rtol=1e-2, atol=1e-3)
        err = (o_tri.float() - o_ref.float()).abs().max().item()
        print(f"  B={B} H={H} N={N:>4} D={D}: {'✅' if ok else '❌'} err={err:.6f}")
        if not ok: all_ok = False

    return all_ok


def benchmark():
    print("\n" + "=" * 60)
    print("BENCHMARKS (triton.testing.do_bench)")
    print("=" * 60)

    for label, B, H, N, D in [
        ("N=256", 1, 16, 256, 128),
        ("N=1024 (512²)", 1, 16, 1024, 128),
        ("N=2304 (768²)", 1, 16, 2304, 128),
        ("N=1024, B=2", 2, 16, 1024, 128),
        ("N=1024, B=4", 4, 16, 1024, 128),
    ]:
        q = torch.randn(B, N, H, D, dtype=DTYPE, device=DEVICE) * 0.1
        k = torch.randn(B, N, H, D, dtype=DTYPE, device=DEVICE) * 0.1
        v = torch.randn(B, N, H, D, dtype=DTYPE, device=DEVICE) * 0.1
        r = torch.randn(N, 1, 1, D, device=DEVICE) * 0.1

        # Warmup
        attn_rope_triton(q, k, v, r)

        # Benchmark: RoPE + SDPA (separate)
        t_ref = triton.testing.do_bench(
            lambda: attn_rope_ref(q, k, v, r), warmup=WARMUP, rep=REP)
        # Benchmark: fused_rope + SDPA
        t_tri = triton.testing.do_bench(
            lambda: attn_rope_triton(q, k, v, r), warmup=WARMUP, rep=REP)

        print(f"  {label:<18s} ref={t_ref*1e6:8.0f}us  tri={t_tri*1e6:8.0f}us  speedup={t_ref/t_tri:.2f}x")

    # Also benchmark attention-only Triton vs cuDNN
    print("\n--- FlashAttention standalone (no RoPE) ---")
    for B, H, N, D in [(1, 4, 256, 128), (1, 4, 1024, 128)]:
        q = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE) * 0.1
        k = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE) * 0.1
        v = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE) * 0.1
        sm = 1.0 / math.sqrt(D)

        flash_attn_triton(q, k, v, sm)  # warmup
        t_ref = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm),
            warmup=WARMUP, rep=REP)
        t_tri = triton.testing.do_bench(
            lambda: flash_attn_triton(q, k, v, sm), warmup=WARMUP, rep=REP)
        print(f"  N={N}: cuDNN={t_ref*1e6:.0f}us  Triton={t_tri*1e6:.0f}us  speedup={t_ref/t_tri:.2f}x")


if __name__ == "__main__":
    gpu = detect_gpu()
    print_gpu_info(gpu)

    ok = test_accuracy()
    print(f"\nAccuracy: {'✅ ALL PASSED' if ok else '❌ FAILURES'}")

    benchmark()
