#!/usr/bin/env python3
"""Anima LoRA operator benchmark with realistic DiT block shapes.

Compares the original PyTorch/cuDNN implementation against Triton paths using
the same tensors.  Shapes follow ``library/anima_models.py``:

    x:    [B, T, H, W, D]
    flat: [B, T*H*W, D]
    qkv:  D=2048, heads=16, head_dim=128

The benchmark uses Triton's do_bench with warmup=128 and rep=1024.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import triton.testing

from triton_ops.lora import adaln_norm, base_lora, output_residual, qkv_lora
from triton_ops.lora.fused_rope_3d import clear_rope_cache, fused as fused_rope, rope_ref_anima


DTYPE = torch.bfloat16
WARMUP = 128
REP = 1024
RANK = 32
ALPHA = 16.0


CASES = [
    # label, B, T, H, W, D, heads, head_dim
    ("512_like", 1, 1, 32, 32, 2048, 16, 128),
    ("768_like", 1, 1, 48, 48, 2048, 16, 128),
]


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def mem_mib() -> tuple[int, int]:
    free, total = torch.cuda.mem_get_info()
    return int((total - free) / 2**20), int(free / 2**20)


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def max_err_tuple(xs, ys) -> float:
    return max(max_err(x, y) for x, y in zip(xs, ys))


def ok_tuple(xs, ys, rtol=2e-2, atol=8e-3) -> bool:
    return all(torch.allclose(x.float(), y.float(), rtol=rtol, atol=atol) for x, y in zip(xs, ys))


def randn(*shape, scale=0.02):
    return torch.randn(*shape, device="cuda", dtype=DTYPE) * scale


def bench_adaln(label: str, B: int, T: int, H: int, W: int, D: int) -> None:
    cleanup()
    used0, free0 = mem_mib()
    x = randn(B, T, H, W, D, scale=0.1)
    scale = randn(B, T, D, scale=0.1)
    shift = randn(B, T, D, scale=0.1)
    scale_5d = scale.reshape(B, T, 1, 1, D)
    shift_5d = shift.reshape(B, T, 1, 1, D)

    ref = lambda: torch.nn.functional.layer_norm(x, (D,), None, None, 1e-6) * (1 + scale_5d) + shift_5d
    opt = lambda: adaln_norm.fused_anima_modulated(x, scale_5d, shift_5d, 1e-6)

    with torch.no_grad():
        y_ref = ref()
        y_opt = opt()
    ok = torch.allclose(y_opt.float(), y_ref.float(), rtol=2e-2, atol=5e-2)
    err = max_err(y_opt, y_ref)
    t_ref = bench(ref)
    t_opt = bench(opt)
    used1, free1 = mem_mib()
    print(f"{label:10s} adaln_two_pass   acc={'PASS' if ok else 'FAIL'} err={err:.6f} ref={t_ref:.4f}ms opt={t_opt:.4f}ms speedup={t_ref/t_opt:.2f}x mem={used0}->{used1}MiB free={free0}->{free1}MiB")


def bench_rope(label: str, B: int, N: int, heads: int, head_dim: int) -> None:
    cleanup()
    clear_rope_cache()
    used0, free0 = mem_mib()
    q = randn(B, N, heads, head_dim, scale=0.1)
    k = randn(B, N, heads, head_dim, scale=0.1)
    rope = torch.randn(N, 1, 1, head_dim, device="cuda", dtype=torch.float32) * 0.1

    ref = lambda: rope_ref_anima(q, k, rope)
    opt = lambda: fused_rope(q, k, rope)

    with torch.no_grad():
        y_ref = ref()
        y_opt = opt()
    ok = ok_tuple(y_opt, y_ref, rtol=0.0, atol=1e-2)
    err = max_err_tuple(y_opt, y_ref)
    opt()  # warm cos/sin cache before timing steady-state training behavior
    t_ref = bench(ref)
    t_opt = bench(opt)
    used1, free1 = mem_mib()
    print(f"{label:10s} rope_cached      acc={'PASS' if ok else 'FAIL'} err={err:.6f} ref={t_ref:.4f}ms opt={t_opt:.4f}ms speedup={t_ref/t_opt:.2f}x mem={used0}->{used1}MiB free={free0}->{free1}MiB")


def bench_qkv(label: str, B: int, N: int, D: int) -> None:
    cleanup()
    used0, free0 = mem_mib()
    x = randn(B, N, D, scale=0.1)
    bw = [randn(D, D) for _ in range(3)]
    bb = [None, None, None]
    la = [randn(RANK, D) for _ in range(3)]
    lb = [randn(D, RANK) for _ in range(3)]
    bw_qkv = torch.cat(bw, dim=0).contiguous()

    ref = lambda: qkv_lora.reference(x, *bw, *bb, *la, *lb, alpha=ALPHA, rank=RANK)
    opt = lambda: qkv_lora.fused_packed(x, bw_qkv, None, *la, *lb, alpha=ALPHA, rank=RANK)

    with torch.no_grad():
        y_ref = ref()
        y_opt = opt()
    ok = ok_tuple(y_opt, y_ref)
    err = max_err_tuple(y_opt, y_ref)
    t_ref = bench(ref)
    t_opt = bench(opt)
    used1, free1 = mem_mib()
    print(f"{label:10s} self_qkv_lora   acc={'PASS' if ok else 'FAIL'} err={err:.6f} ref={t_ref:.4f}ms opt={t_opt:.4f}ms speedup={t_ref/t_opt:.2f}x mem={used0}->{used1}MiB free={free0}->{free1}MiB")


def bench_base_lora(label: str, B: int, N: int, in_d: int, out_d: int, name: str) -> None:
    cleanup()
    used0, free0 = mem_mib()
    x = randn(B, N, in_d, scale=0.1)
    bw = randn(out_d, in_d)
    la = randn(RANK, in_d)
    lb = randn(out_d, RANK)

    ref = lambda: base_lora.reference(x, bw, None, la, lb, alpha=ALPHA, rank=RANK)
    opt = lambda: base_lora.fused(x, bw, None, la, lb, alpha=ALPHA, rank=RANK)[0]

    with torch.no_grad():
        y_ref = ref()
        y_opt = opt()
    ok = torch.allclose(y_opt.float(), y_ref.float(), rtol=2e-2, atol=8e-3)
    err = max_err(y_opt, y_ref)
    t_ref = bench(ref)
    t_opt = bench(opt)
    used1, free1 = mem_mib()
    print(f"{label:10s} {name:15s} acc={'PASS' if ok else 'FAIL'} err={err:.6f} ref={t_ref:.4f}ms opt={t_opt:.4f}ms speedup={t_ref/t_opt:.2f}x mem={used0}->{used1}MiB free={free0}->{free1}MiB")


def bench_output_residual(label: str, B: int, N: int, D: int) -> None:
    cleanup()
    used0, free0 = mem_mib()
    attn = randn(B, N, D, scale=0.1)
    residual = randn(B, N, D, scale=0.1)
    gate = randn(D, scale=0.1) + 0.5
    bw = randn(D, D)
    la = randn(RANK, D)
    lb = randn(D, RANK)

    ref = lambda: output_residual.reference(attn, residual, gate, bw, None, la, lb, alpha=ALPHA, rank=RANK)
    opt = lambda: output_residual.fused(attn, residual, gate, bw, None, la, lb, alpha=ALPHA, rank=RANK)

    with torch.no_grad():
        y_ref = ref()
        y_opt = opt()
    ok = torch.allclose(y_opt.float(), y_ref.float(), rtol=2e-2, atol=8e-3)
    err = max_err(y_opt, y_ref)
    t_ref = bench(ref)
    t_opt = bench(opt)
    used1, free1 = mem_mib()
    print(f"{label:10s} output_residual acc={'PASS' if ok else 'FAIL'} err={err:.6f} ref={t_ref:.4f}ms opt={t_opt:.4f}ms speedup={t_ref/t_opt:.2f}x mem={used0}->{used1}MiB free={free0}->{free1}MiB")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} total={props.total_memory / 2**30:.2f}GB")
    print(f"do_bench warmup={WARMUP} rep={REP}")
    print("shape source: Anima block [B,T,H,W,D] -> [B,T*H*W,D], D=2048 heads=16 head_dim=128")

    torch.manual_seed(20260706)
    with torch.no_grad():
        for label, B, T, H, W, D, heads, head_dim in CASES:
            N = T * H * W
            print(f"\nCASE {label}: B={B} T={T} H={H} W={W} N={N} D={D}")
            bench_adaln(label, B, T, H, W, D)
            bench_rope(label, B, N, heads, head_dim)
            bench_qkv(label, B, N, D)
            bench_base_lora(label, B, N, D, D, "base_lora_DxD")
            bench_output_residual(label, B, N, D)
            # GPT2FeedForward in Anima is GELU, so this uses base LoRA for its two real linear shapes.
            bench_base_lora(label, B, N, D, 4 * D, "mlp_layer1")
            bench_base_lora(label, B, N, 4 * D, D, "mlp_layer2")
            cleanup()


if __name__ == "__main__":
    main()
