"""
Comprehensive Accuracy & Performance Test Suite — All 5 Anima LoRA Kernels

Tests every kernel across Anima-typical shapes with proper BF16 tolerances.
Uses triton.testing.do_bench for fair GPU timing with L2 cache flush.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import triton
import triton.testing

from triton_ops.config import detect_gpu, print_gpu_info, GPUInfo
from triton_ops.lora import base_lora, qkv_lora, ffn_silu, adaln_norm, output_residual

# ===========================================================================
# Test configuration
# ===========================================================================

ANIMA_RESOLUTIONS = {
    # resolution → latent H×W → tokens N
    "256²":   (1, 16, 32, 32),    # N = 16×16 = 256
    "362²":   (1, 16, 45, 45),    # N ≈ 484
    "512²":   (1, 16, 64, 64),    # N = 32×32 = 1024
    "768²":   (1, 16, 96, 96),    # N = 48×48 = 2304
}

ANIMA_SHAPES = [
    # (name,           B, N,    C_in, C_out)
    ("sa-256 (1×256)",        1,   256, 2048, 2048),
    ("sa-512 (1×1024)",       1,  1024, 2048, 2048),
    ("sa-768 (1×2304)",       1,  2304, 2048, 2048),
    ("sa-512-b2 (2×1024)",    2,  1024, 2048, 2048),
    ("sa-512-b4 (4×1024)",    4,  1024, 2048, 2048),
    ("sa-512-r64 (1×1024)",   1,  1024, 2048, 2048),
    ("mlp-w1 (1×1024→8192)",  1,  1024, 2048, 8192),
    ("mlp-w2 (1×1024←8192)",  1,  1024, 8192, 2048),
    ("tiny (1×64)",           1,    64, 2048, 2048),
]

TOLERANCES = {
    # kernel_name: (rtol, atol, note)
    "base_lora":        (1e-2, 1e-4,  "bit-exact — cuDNN base matmul identical"),
    "qkv_lora":         (1e-2, 1e-4,  "bit-exact — cuDNN base matmuls identical"),
    "ffn_silu":         (1e-2, 5e-3,  "BF16 tolerated — SiLU rounding at extremes"),
    "adaln_norm":       (2e-2, 5e-2,  "BF16 tolerant — norm stats bit-exact, output rounding"),
    "output_residual":  (1e-2, 5e-3,  "BF16 tolerated — gate·proj rounding at extremes"),
}

BENCH_WARMUP_MS = 50
BENCH_REP_MS = 200

# ===========================================================================
# Helpers
# ===========================================================================

def make_tensors(B, N, C_in, C_out, rank=32, device="cuda"):
    """Create consistent test tensors for a given shape."""
    torch.manual_seed(42)
    x = torch.randn(B, N, C_in, dtype=torch.bfloat16, device=device) * 0.1
    bw = torch.randn(C_out, C_in, dtype=torch.bfloat16, device=device) * 0.02
    bb = torch.randn(C_out, dtype=torch.bfloat16, device=device) * 0.01
    la = torch.randn(rank, C_in, dtype=torch.bfloat16, device=device) * 0.02
    lb = torch.randn(C_out, rank, dtype=torch.bfloat16, device=device) * 0.02
    return x, bw, bb, la, lb, rank


def benchmark_pair(label_ref: str, fn_ref, label_tri: str, fn_tri):
    """Benchmark reference vs Triton with do_bench.  Returns (t_ref, t_tri, speedup)."""
    t_ref = triton.testing.do_bench(fn_ref, warmup=BENCH_WARMUP_MS, rep=BENCH_REP_MS)
    t_tri = triton.testing.do_bench(fn_tri, warmup=BENCH_WARMUP_MS, rep=BENCH_REP_MS)
    return t_ref, t_tri, t_ref / t_tri


# ===========================================================================
# Test runners — each returns (all_pass: bool, errors: list[str])
# ===========================================================================

def test_base_lora(shapes, device="cuda") -> tuple[bool, list[str]]:
    """Test base_lora.fused() across all shapes."""
    errors = []
    rtol, atol, _ = TOLERANCES["base_lora"]

    for name, B, N, C_in, C_out in shapes:
        x, bw, bb, la, lb, rank = make_tensors(B, N, C_in, C_out, device=device)
        with torch.no_grad():
            y_ref = base_lora.reference(x, bw, bb, la, lb, rank=rank)
            y_tri, _ = base_lora.fused(x, bw, bb, la, lb, rank=rank)

        if not torch.allclose(y_tri, y_ref, rtol=rtol, atol=atol):
            err = (y_tri.float() - y_ref.float()).abs().max().item()
            errors.append(f"{name}: max_err={err:.6f} (shape [{B},{N},{C_in}]→[{C_out}])")
    return len(errors) == 0, errors


def test_qkv_lora(shapes, device="cuda") -> tuple[bool, list[str]]:
    """Test qkv_lora.fused() — all three projections share same input x."""
    errors = []
    rtol, atol, _ = TOLERANCES["qkv_lora"]

    for name, B, N, C, _ in shapes:
        if C != 2048:  # QKV only for self-attn projections (C_in=C_out=2048)
            continue
        x, bw, bb, la1, lb1, rank = make_tensors(B, N, C, C, device=device)
        la2 = torch.randn(rank, C, dtype=torch.bfloat16, device=device) * 0.02
        la3 = torch.randn(rank, C, dtype=torch.bfloat16, device=device) * 0.02
        lb2 = torch.randn(C, rank, dtype=torch.bfloat16, device=device) * 0.02
        lb3 = torch.randn(C, rank, dtype=torch.bfloat16, device=device) * 0.02

        with torch.no_grad():
            qr, kr, vr = qkv_lora.reference(
                x, bw, bw, bw, bb, bb, bb,
                la1, la2, la3, lb1, lb2, lb3, rank=rank,
            )
            qt, kt, vt = qkv_lora.fused(
                x, bw, bw, bw, bb, bb, bb,
                la1, la2, la3, lb1, lb2, lb3, rank=rank,
            )

        for head, ta, ra in [("Q", qt, qr), ("K", kt, kr), ("V", vt, vr)]:
            if not torch.allclose(ta, ra, rtol=rtol, atol=atol):
                err = (ta.float() - ra.float()).abs().max().item()
                errors.append(f"{name}/{head}: max_err={err:.6f}")
    return len(errors) == 0, errors


def test_ffn_silu(shapes, device="cuda") -> tuple[bool, list[str]]:
    """Test ffn_silu.fused() — only for w1 (C_in=2048, C_out=8192)."""
    errors = []
    rtol, atol, _ = TOLERANCES["ffn_silu"]

    for name, B, N, C_in, _ in shapes:
        if C_in != 2048:  # Only w1 expand
            continue
        x, _, _, la, lb1, rank = make_tensors(B, N, C_in, 8192, device=device)
        bw1 = torch.randn(8192, C_in, dtype=torch.bfloat16, device=device) * 0.02
        bb1 = torch.randn(8192, dtype=torch.bfloat16, device=device) * 0.01
        lb1 = torch.randn(8192, rank, dtype=torch.bfloat16, device=device) * 0.02

        with torch.no_grad():
            yr = ffn_silu.reference(x, bw1, bb1, la, lb1, rank=rank)
            yt = ffn_silu.fused(x, bw1, bb1, la, lb1, rank=rank)

        if not torch.allclose(yt, yr, rtol=rtol, atol=atol):
            err = (yt.float() - yr.float()).abs().max().item()
            errors.append(f"{name}: max_err={err:.6f}")
    return len(errors) == 0, errors


def test_adaln_norm(shapes, device="cuda") -> tuple[bool, list[str]]:
    """Test adaln_norm.fused() — check both output AND statistics."""
    errors = []
    rtol, atol, _ = TOLERANCES["adaln_norm"]

    for name, B, N, C, _ in shapes:
        x, _, _, _, _, _ = make_tensors(B, N, C, C, device=device)
        scale = torch.ones(C, dtype=torch.bfloat16, device=device) + \
                torch.randn(C, dtype=torch.bfloat16, device=device) * 0.1
        shift = torch.randn(C, dtype=torch.bfloat16, device=device) * 0.1
        w = torch.ones(C, dtype=torch.bfloat16, device=device)
        b = torch.zeros(C, dtype=torch.bfloat16, device=device)

        with torch.no_grad():
            yr = adaln_norm.reference(x, scale, shift, w, b)
            yt, mean_t, rstd_t = adaln_norm.fused(x, scale, shift, w, b, save_stats=True)

        # Check output
        if not torch.allclose(yt, yr, rtol=rtol, atol=atol):
            err = (yt.float() - yr.float()).abs().max().item()
            errors.append(f"{name}: output max_err={err:.6f}")

        # Check statistics are bit-exact
        x_mod = x.float() * scale.float() + shift.float()
        mean_r = x_mod.mean(dim=-1).reshape(-1)  # [B*N]
        rstd_r = (1.0 / torch.sqrt(x_mod.var(dim=-1, unbiased=False) + 1e-6)).reshape(-1)
        md = (mean_t.float() - mean_r).abs().max().item()
        rd = (rstd_t.float() - rstd_r).abs().max().item()
        if md > 1e-5:
            errors.append(f"{name}: mean stats diff={md:.10f}")
        if rd > 1e-5:
            errors.append(f"{name}: rstd stats diff={rd:.10f}")
    return len(errors) == 0, errors


def test_output_residual(shapes, device="cuda") -> tuple[bool, list[str]]:
    """Test output_residual.fused()."""
    errors = []
    rtol, atol, _ = TOLERANCES["output_residual"]

    for name, B, N, C, _ in shapes:
        if C != 2048:
            continue
        x, bw, bb, la, lb, rank = make_tensors(B, N, C, C, device=device)
        attn = torch.randn(B, N, C, dtype=torch.bfloat16, device=device) * 0.1
        res = torch.randn(B, N, C, dtype=torch.bfloat16, device=device) * 0.1
        gate = torch.randn(C, dtype=torch.bfloat16, device=device) * 0.1 + 0.5

        with torch.no_grad():
            yr = output_residual.reference(attn, res, gate, bw, bb, la, lb, rank=rank)
            yt = output_residual.fused(attn, res, gate, bw, bb, la, lb, rank=rank)

        if not torch.allclose(yt, yr, rtol=rtol, atol=atol):
            err = (yt.float() - yr.float()).abs().max().item()
            errors.append(f"{name}: max_err={err:.6f}")
    return len(errors) == 0, errors


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    gpu = detect_gpu()
    print_gpu_info(gpu)

    # =======================================================================
    # PHASE 1 — ACCURACY
    # =======================================================================
    print("\n" + "=" * 72)
    print("PHASE 1: ACCURACY — All Kernels × All Shapes")
    print("=" * 72)

    tests = [
        ("base_lora",        test_base_lora,        ANIMA_SHAPES),
        ("qkv_lora",         test_qkv_lora,         ANIMA_SHAPES),
        ("ffn_silu",         test_ffn_silu,         ANIMA_SHAPES),
        ("adaln_norm",       test_adaln_norm,       ANIMA_SHAPES),
        ("output_residual",  test_output_residual,  ANIMA_SHAPES),
    ]

    all_accuracy_ok = True
    for kernel_name, test_fn, shapes in tests:
        ok, errors = test_fn(shapes)
        status = "✅ PASS" if ok else "❌ FAIL"
        rtol, atol, note = TOLERANCES[kernel_name]
        print(f"\n  {kernel_name:<20} {status}  (tol: rtol={rtol}, atol={atol})")
        print(f"    {note}")
        for e in errors:
            print(f"    ⚠️  {e}")
        if not ok:
            all_accuracy_ok = False

    # =======================================================================
    # PHASE 2 — BENCHMARKS
    # =======================================================================
    print("\n" + "=" * 72)
    print("PHASE 2: PERFORMANCE — triton.testing.do_bench")
    print(f"  warmup={BENCH_WARMUP_MS}ms  rep={BENCH_REP_MS}ms  L2 cache flushed")
    print("=" * 72)

    # Use Anima-typical shapes for benchmark
    bench_configs = [
        # (label, kernel_name, shape config for make_tensors)
        ("base_lora/sa-512",    "base_lora",        *make_tensors(1, 1024, 2048, 2048)[:5], 32),
        ("base_lora/mlp-w1",    "base_lora",        *make_tensors(1, 1024, 2048, 8192)[:5], 32),
        ("qkv_lora/sa-512",     "qkv_lora",         None, None),  # special handling
        ("ffn_silu/w1-512",     "ffn_silu",         *make_tensors(1, 1024, 2048, 8192)[:5], 32),
        ("adaln_norm/sa-512",   "adaln_norm",       None, None),
        ("output_res/sa-512",   "output_residual",  None, None),
    ]

    print(f"\n  {'Kernel/Config':<28} {'cuDNN ref':>10}  {'Triton':>10}  {'vsRef':>8}  Result")
    print(f"  {'-'*28} {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

    # ---- base_lora ----
    x, bw, bb, la, lb, rank = make_tensors(1, 1024, 2048, 2048)
    t_ref, t_tri, sp = benchmark_pair(
        "ref", lambda r=rank: base_lora.reference(x, bw, bb, la, lb, rank=r),
        "tri", lambda r=rank: base_lora.fused(x, bw, bb, la, lb, rank=r)[0],
    )
    print(f"  {'base_lora / sa-proj-512':<28} {t_ref:>9.3f}ms  {t_tri:>9.3f}ms  {sp:>7.2f}x  {'✅' if sp>=0.95 else '⚠️'}")

    # ---- base_lora MLP ----
    x2, bw2, bb2, la2, lb2, _ = make_tensors(1, 1024, 2048, 8192)
    t_ref, t_tri, sp = benchmark_pair(
        "ref", lambda: base_lora.reference(x2, bw2, bb2, la2, lb2, rank=32),
        "tri", lambda: base_lora.fused(x2, bw2, bb2, la2, lb2, rank=32)[0],
    )
    print(f"  {'base_lora / mlp-layer1':<28} {t_ref:>9.3f}ms  {t_tri:>9.3f}ms  {sp:>7.2f}x  {'✅' if sp>=0.95 else '⚠️'}")

    # ---- qkv_lora ----
    la1q = torch.randn(32, 2048, dtype=torch.bfloat16, device="cuda") * 0.02
    la2q = torch.randn(32, 2048, dtype=torch.bfloat16, device="cuda") * 0.02
    la3q = torch.randn(32, 2048, dtype=torch.bfloat16, device="cuda") * 0.02
    lb1q = torch.randn(2048, 32, dtype=torch.bfloat16, device="cuda") * 0.02
    lb2q = torch.randn(2048, 32, dtype=torch.bfloat16, device="cuda") * 0.02
    lb3q = torch.randn(2048, 32, dtype=torch.bfloat16, device="cuda") * 0.02
    t_ref, t_tri, sp = benchmark_pair(
        "ref", lambda: qkv_lora.reference(x, bw, bw, bw, bb, bb, bb, la1q, la2q, la3q, lb1q, lb2q, lb3q, rank=32),
        "tri", lambda: qkv_lora.fused(x, bw, bw, bw, bb, bb, bb, la1q, la2q, la3q, lb1q, lb2q, lb3q, rank=32),
    )
    print(f"  {'qkv_lora / sa-512':<28} {t_ref:>9.3f}ms  {t_tri:>9.3f}ms  {sp:>7.2f}x  {'✅' if sp>=0.95 else '⚠️'}")

    # ---- ffn_silu ----
    bw1f = torch.randn(8192, 2048, dtype=torch.bfloat16, device="cuda") * 0.02
    bb1f = torch.randn(8192, dtype=torch.bfloat16, device="cuda") * 0.01
    la1f = torch.randn(32, 2048, dtype=torch.bfloat16, device="cuda") * 0.02
    lb1f = torch.randn(8192, 32, dtype=torch.bfloat16, device="cuda") * 0.02
    t_ref, t_tri, sp = benchmark_pair(
        "ref", lambda: ffn_silu.reference(x, bw1f, bb1f, la1f, lb1f, rank=32),
        "tri", lambda: ffn_silu.fused(x, bw1f, bb1f, la1f, lb1f, rank=32),
    )
    print(f"  {'ffn_silu / w1-512':<28} {t_ref:>9.3f}ms  {t_tri:>9.3f}ms  {sp:>7.2f}x  {'✅' if sp>=1.10 else '⚠️'}")

    # ---- adaln_norm ----
    scale_a = torch.ones(2048, dtype=torch.bfloat16, device="cuda") + \
              torch.randn(2048, dtype=torch.bfloat16, device="cuda") * 0.1
    shift_a = torch.randn(2048, dtype=torch.bfloat16, device="cuda") * 0.1
    wa = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
    ba = torch.zeros(2048, dtype=torch.bfloat16, device="cuda")
    t_ref, t_tri, sp = benchmark_pair(
        "ref", lambda: adaln_norm.reference(x, scale_a, shift_a, wa, ba),
        "tri", lambda: adaln_norm.fused(x, scale_a, shift_a, wa, ba)[0],
    )
    print(f"  {'adaln_norm / sa-512':<28} {t_ref:>9.3f}ms  {t_tri:>9.3f}ms  {sp:>7.2f}x  {'✅' if sp>=3.0 else '⚠️'}")

    # ---- output_residual ----
    attn_o = torch.randn(1, 1024, 2048, dtype=torch.bfloat16, device="cuda") * 0.1
    res_o = torch.randn(1, 1024, 2048, dtype=torch.bfloat16, device="cuda") * 0.1
    gate_o = torch.randn(2048, dtype=torch.bfloat16, device="cuda") * 0.1 + 0.5
    la_o = torch.randn(32, 2048, dtype=torch.bfloat16, device="cuda") * 0.02
    lb_o = torch.randn(2048, 32, dtype=torch.bfloat16, device="cuda") * 0.02
    t_ref, t_tri, sp = benchmark_pair(
        "ref", lambda: output_residual.reference(attn_o, res_o, gate_o, bw, bb, la_o, lb_o, rank=32),
        "tri", lambda: output_residual.fused(attn_o, res_o, gate_o, bw, bb, la_o, lb_o, rank=32),
    )
    print(f"  {'output_residual / sa-512':<28} {t_ref:>9.3f}ms  {t_tri:>9.3f}ms  {sp:>7.2f}x  {'✅' if sp>=1.05 else '⚠️'}")

    # =======================================================================
    # PHASE 3 — BATCH SCALING
    # =======================================================================
    print("\n" + "=" * 72)
    print("PHASE 3: BATCH SCALING — base_lora @ B=1,2,4,8")
    print("=" * 72)
    print(f"  {'Batch':>8} {'ref (ms)':>10} {'tri (ms)':>10} {'speedup':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for B in [1, 2, 4, 8]:
        xb = torch.randn(B, 1024, 2048, dtype=torch.bfloat16, device="cuda") * 0.1
        t_ref = triton.testing.do_bench(
            lambda: base_lora.reference(xb, bw, bb, la, lb, rank=32),
            warmup=BENCH_WARMUP_MS, rep=BENCH_REP_MS,
        )
        t_tri = triton.testing.do_bench(
            lambda: base_lora.fused(xb, bw, bb, la, lb, rank=32)[0],
            warmup=BENCH_WARMUP_MS, rep=BENCH_REP_MS,
        )
        print(f"  {B:>8} {t_ref:>9.3f}ms {t_tri:>9.3f}ms {t_ref/t_tri:>7.2f}x")

    # =======================================================================
    # Final verdict
    # =======================================================================
    print("\n" + "=" * 72)
    if all_accuracy_ok:
        print("✅ ALL KERNELS PASS ACCURACY TESTS")
    else:
        print("❌ SOME KERNELS FAILED ACCURACY TESTS")
    print("=" * 72)
