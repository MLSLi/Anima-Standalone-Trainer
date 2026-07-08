#!/usr/bin/env python3
"""Safe accuracy and performance checks for optimized Triton paths.

The shapes are intentionally modest for 8 GB laptop GPUs.  They compare
optimized paths against the original PyTorch/cuDNN references.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import triton.testing

from triton_ops.lora import adaln_norm, qkv_lora
from triton_ops.lora.fused_rope_3d import clear_rope_cache, fused as fused_rope, rope_ref_anima


WARMUP_MS = 20
REP_MS = 80
DTYPE = torch.bfloat16


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS)


def _max_err(xs, ys):
    return max((x.float() - y.float()).abs().max().item() for x, y in zip(xs, ys))


def _allclose(xs, ys, rtol=2e-2, atol=8e-3):
    return all(torch.allclose(x.float(), y.float(), rtol=rtol, atol=atol) for x, y in zip(xs, ys))


def _print_gpu():
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {props.name}  total={total / 2**30:.2f}GB  free={free / 2**30:.2f}GB")


def bench_qkv_packed(device="cuda"):
    torch.manual_seed(123)
    B, N, C, rank = 1, 256, 512, 32
    x = torch.randn(B, N, C, dtype=DTYPE, device=device) * 0.1

    bw_q = torch.randn(C, C, dtype=DTYPE, device=device) * 0.02
    bw_k = torch.randn(C, C, dtype=DTYPE, device=device) * 0.02
    bw_v = torch.randn(C, C, dtype=DTYPE, device=device) * 0.02
    bb_q = torch.randn(C, dtype=DTYPE, device=device) * 0.01
    bb_k = torch.randn(C, dtype=DTYPE, device=device) * 0.01
    bb_v = torch.randn(C, dtype=DTYPE, device=device) * 0.01

    la_q = torch.randn(rank, C, dtype=DTYPE, device=device) * 0.02
    la_k = torch.randn(rank, C, dtype=DTYPE, device=device) * 0.02
    la_v = torch.randn(rank, C, dtype=DTYPE, device=device) * 0.02
    lb_q = torch.randn(C, rank, dtype=DTYPE, device=device) * 0.02
    lb_k = torch.randn(C, rank, dtype=DTYPE, device=device) * 0.02
    lb_v = torch.randn(C, rank, dtype=DTYPE, device=device) * 0.02

    bw_qkv = torch.cat([bw_q, bw_k, bw_v], dim=0).contiguous()
    bb_qkv = torch.cat([bb_q, bb_k, bb_v], dim=0).contiguous()

    ref_fn = lambda: qkv_lora.reference(
        x, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v,
        la_q, la_k, la_v, lb_q, lb_k, lb_v, rank=rank,
    )
    opt_fn = lambda: qkv_lora.fused_packed(
        x, bw_qkv, bb_qkv, la_q, la_k, la_v, lb_q, lb_k, lb_v, rank=rank,
    )

    with torch.no_grad():
        ref = ref_fn()
        opt = opt_fn()
    ok = _allclose(opt, ref)
    err = _max_err(opt, ref)

    torch.cuda.synchronize()
    t_ref = _bench(ref_fn)
    t_opt = _bench(opt_fn)
    print(
        f"qkv_packed B={B} N={N} C={C} r={rank}: "
        f"acc={'PASS' if ok else 'FAIL'} max_err={err:.6f} "
        f"cudnn_ref={t_ref:.4f}ms opt={t_opt:.4f}ms speedup={t_ref / t_opt:.2f}x"
    )


def bench_rope_cached(device="cuda"):
    torch.manual_seed(321)
    B, N, H, D = 1, 512, 8, 64
    q = torch.randn(B, N, H, D, dtype=DTYPE, device=device) * 0.1
    k = torch.randn(B, N, H, D, dtype=DTYPE, device=device) * 0.1
    rope_emb = torch.randn(N, 1, 1, D, dtype=torch.float32, device=device) * 0.1

    ref_fn = lambda: rope_ref_anima(q, k, rope_emb)
    opt_fn = lambda: fused_rope(q, k, rope_emb)

    clear_rope_cache()
    with torch.no_grad():
        ref = ref_fn()
        opt = opt_fn()
    ok = _allclose(opt, ref, rtol=0.0, atol=1e-2)
    err = _max_err(opt, ref)

    opt_fn()  # warm the cos/sin cache before timing steady-state training behavior
    torch.cuda.synchronize()
    t_ref = _bench(ref_fn)
    t_opt = _bench(opt_fn)
    print(
        f"rope_cached B={B} N={N} H={H} D={D}: "
        f"acc={'PASS' if ok else 'FAIL'} max_err={err:.6f} "
        f"torch_ref={t_ref:.4f}ms opt={t_opt:.4f}ms speedup={t_ref / t_opt:.2f}x"
    )


def bench_adaln_two_pass(device="cuda"):
    torch.manual_seed(456)
    B, N, C = 1, 512, 2048
    x = torch.randn(B, N, C, dtype=DTYPE, device=device) * 0.1
    scale = torch.ones(C, dtype=DTYPE, device=device) + torch.randn(C, dtype=DTYPE, device=device) * 0.1
    shift = torch.randn(C, dtype=DTYPE, device=device) * 0.1
    weight = torch.ones(C, dtype=DTYPE, device=device)
    bias = torch.zeros(C, dtype=DTYPE, device=device)

    ref_fn = lambda: adaln_norm.reference(x, scale, shift, weight, bias)
    opt_fn = lambda: adaln_norm.fused_two_pass(x, scale, shift, weight, bias)[0]

    with torch.no_grad():
        ref = ref_fn()
        opt = opt_fn()
    ok = torch.allclose(opt.float(), ref.float(), rtol=2e-2, atol=5e-2)
    err = (opt.float() - ref.float()).abs().max().item()

    torch.cuda.synchronize()
    t_ref = _bench(ref_fn)
    t_opt = _bench(opt_fn)
    print(
        f"adaln_two_pass B={B} N={N} C={C}: "
        f"acc={'PASS' if ok else 'FAIL'} max_err={err:.6f} "
        f"torch_ref={t_ref:.4f}ms opt={t_opt:.4f}ms speedup={t_ref / t_opt:.2f}x"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    _print_gpu()
    torch.cuda.empty_cache()
    with torch.no_grad():
        bench_qkv_packed()
        torch.cuda.empty_cache()
        bench_rope_cached()
        torch.cuda.empty_cache()
        bench_adaln_two_pass()


if __name__ == "__main__":
    main()
