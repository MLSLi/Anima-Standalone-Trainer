#!/usr/bin/env python3
"""Comprehensive accuracy & benchmark test for ALL 6 Triton kernels.

Tests: base_lora, qkv_lora, ffn_silu, adaln_norm, output_residual, fused_rope_3d.
No source modifications — imports kernels as-is and validates against PyTorch.
"""

import sys; sys.path.insert(0, '.')
import torch, triton.testing
from triton_ops.config import detect_gpu, print_gpu_info
from triton_ops.lora import base_lora, qkv_lora, ffn_silu, adaln_norm, output_residual
from triton_ops.lora.fused_rope_3d import rope_ref_anima, fused as fused_rope

import logging; logging.basicConfig(level=logging.WARNING)

WARMUP, REP = 50, 200  # ms
DEVICE = torch.device("cuda")


def make_base_lora_inputs(B, N, C_in, C_out, rank=32):
    t = torch.randn; bf = torch.bfloat16
    x = t(B, N, C_in, dtype=bf, device=DEVICE) * 0.1
    bw = t(C_out, C_in, dtype=bf, device=DEVICE) * 0.02
    bb = t(C_out, dtype=bf, device=DEVICE) * 0.01
    la = t(rank, C_in, dtype=bf, device=DEVICE) * 0.02
    lb = t(C_out, rank, dtype=bf, device=DEVICE) * 0.02
    return x, bw, bb, la, lb


def make_qkv_inputs(B, N, C, rank=32):
    x, bw, bb, la1, lb1 = make_base_lora_inputs(B, N, C, C, rank)
    t = torch.randn; bf = torch.bfloat16
    return (x, bw, bw, bw, bb, bb, bb,
            la1, t(rank, C, dtype=bf, device=DEVICE)*0.02, t(rank, C, dtype=bf, device=DEVICE)*0.02,
            lb1, t(C, rank, dtype=bf, device=DEVICE)*0.02, t(C, rank, dtype=bf, device=DEVICE)*0.02)


def make_ffn_inputs(B, N, C_in=2048, C_out=8192, rank=32):
    x, _, _, la, lb1 = make_base_lora_inputs(B, N, C_in, C_out, rank)
    t = torch.randn; bf = torch.bfloat16
    bw1 = t(C_out, C_in, dtype=bf, device=DEVICE) * 0.02
    bb1 = t(C_out, dtype=bf, device=DEVICE) * 0.01
    return x, bw1, bb1, la, t(C_out, rank, dtype=bf, device=DEVICE) * 0.02


def make_adaln_inputs(B, N, C):
    t = torch.randn; bf = torch.bfloat16
    x = t(B, N, C, dtype=bf, device=DEVICE) * 0.1
    scale = torch.ones(C, dtype=bf, device=DEVICE) + t(C, dtype=bf, device=DEVICE) * 0.1
    shift = t(C, dtype=bf, device=DEVICE) * 0.1
    w = torch.ones(C, dtype=bf, device=DEVICE)
    b = torch.zeros(C, dtype=bf, device=DEVICE)
    return x, scale, shift, w, b


def make_output_res_inputs(B, N, C, rank=32):
    t = torch.randn; bf = torch.bfloat16
    attn = t(B, N, C, dtype=bf, device=DEVICE) * 0.1
    res = t(B, N, C, dtype=bf, device=DEVICE) * 0.1
    gate = t(C, dtype=bf, device=DEVICE) * 0.1 + 0.5
    x, bw, bb, la, lb = make_base_lora_inputs(B, N, C, C, rank)
    return attn, res, gate, bw, bb, la, lb


def make_rope_inputs(B, H, N, D=128, rot=128):
    assert rot == D, "Anima RoPE uses full head_dim frequencies"
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=DEVICE) * 0.1
    k = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=DEVICE) * 0.1
    rope_emb = torch.randn(N, 1, 1, D, dtype=torch.float32, device=DEVICE) * 0.1
    return q, k, rope_emb


# ============================================================================
# Accuracy Tests
# ============================================================================

def test_base_lora():
    print("\n--- base_lora ---")
    ok_all = True
    for B,N,C_in,C_out,rank in [(1,1024,2048,2048,32),(1,1024,2048,8192,32),(2,1024,2048,2048,32)]:
        x,bw,bb,la,lb = make_base_lora_inputs(B,N,C_in,C_out,rank)
        with torch.no_grad():
            yr = base_lora.reference(x,bw,bb,la,lb,rank=rank)
            yt,_ = base_lora.fused(x,bw,bb,la,lb,rank=rank)
        ok = torch.allclose(yt.float(), yr.float(), rtol=1e-2, atol=5e-3)
        e = (yt.float()-yr.float()).abs().max().item()
        print(f"  {B=} N={N} {C_in}->{C_out} r={rank}: {'✅' if ok else '❌'} err={e:.6f}")
        if not ok: ok_all = False
    return ok_all


def test_qkv_lora():
    print("\n--- qkv_lora ---")
    ok_all = True
    for B,N,C,rank in [(1,1024,2048,32),(2,1024,2048,32)]:
        xq,bwq,bwk,bwv,bbq,bbk,bbv,laq,lak,lav,lbq,lbk,lbv = make_qkv_inputs(B,N,C,rank)
        with torch.no_grad():
            qr,kr,vr = qkv_lora.reference(xq,bwq,bwk,bwv,bbq,bbk,bbv,laq,lak,lav,lbq,lbk,lbv,rank=rank)
            qt,kt,vt = qkv_lora.fused(xq,bwq,bwk,bwv,bbq,bbk,bbv,laq,lak,lav,lbq,lbk,lbv,rank=rank)
        o = all(torch.allclose(a.float(),b.float(),rtol=1e-2,atol=5e-3) for a,b in [(qt,qr),(kt,kr),(vt,vr)])
        e = max((a.float()-b.float()).abs().max().item() for a,b in [(qt,qr),(kt,kr),(vt,vr)])
        print(f"  {B=} N={N} C={C} r={rank}: {'✅' if o else '❌'} err={e:.6f}")
        if not o: ok_all = False
    return ok_all


def test_ffn_silu():
    print("\n--- ffn_silu ---")
    ok_all = True
    for B,N,rk in [(1,1024,32),(2,1024,32)]:
        x,bw1,bb1,la,lb1 = make_ffn_inputs(B,N,rank=rk)
        with torch.no_grad():
            yr = ffn_silu.reference(x,bw1,bb1,la,lb1,rank=rk)
            yt = ffn_silu.fused(x,bw1,bb1,la,lb1,rank=rk)
        ok = torch.allclose(yt.float(), yr.float(), rtol=1e-2, atol=5e-3)
        e = (yt.float()-yr.float()).abs().max().item()
        print(f"  {B=} N={N} 2048->8192 r={rk}: {'✅' if ok else '❌'} err={e:.6f}")
        if not ok: ok_all = False
    return ok_all


def test_adaln_norm():
    print("\n--- adaln_norm ---")
    ok_all = True
    for B,N in [(1,1024),(2,1024)]:
        x,sc,sh,w,b = make_adaln_inputs(B,N,2048)
        with torch.no_grad():
            yr = adaln_norm.reference(x,sc,sh,w,b)
            yt,_,_ = adaln_norm.fused(x,sc,sh,w,b)
        ok = torch.allclose(yt.float(), yr.float(), rtol=2e-2, atol=5e-2)
        e = (yt.float()-yr.float()).abs().max().item()
        print(f"  {B=} N={N} C=2048: {'✅' if ok else '❌'} err={e:.6f}")
        if not ok: ok_all = False
    return ok_all


def test_output_residual():
    print("\n--- output_residual ---")
    ok_all = True
    for B,N,rank in [(1,1024,32),(2,1024,32)]:
        attn,res,gate,bw,bb,la,lb = make_output_res_inputs(B,N,2048,rank)
        with torch.no_grad():
            yr = output_residual.reference(attn,res,gate,bw,bb,la,lb,rank=rank)
            yt = output_residual.fused(attn,res,gate,bw,bb,la,lb,rank=rank)
        ok = torch.allclose(yt.float(), yr.float(), rtol=1e-2, atol=5e-3)
        e = (yt.float()-yr.float()).abs().max().item()
        print(f"  {B=} N={N} C=2048 r={rank}: {'✅' if ok else '❌'} err={e:.6f}")
        if not ok: ok_all = False
    return ok_all


def test_fused_rope():
    print("\n--- fused_rope_3d ---")
    ok_all = True
    for B,H,N,D,rot in [(1,16,1024,128,128),(2,16,1024,128,128),(1,16,256,128,128)]:
        q,k,rope_emb = make_rope_inputs(B,H,N,D,rot)
        with torch.no_grad():
            qr,kr = rope_ref_anima(q,k,rope_emb)
            qt,kt = fused_rope(q,k,rope_emb)
        o = torch.allclose(qt.float(),qr.float(),rtol=0.0,atol=0.01) and \
            torch.allclose(kt.float(),kr.float(),rtol=0.0,atol=0.01)
        eq = (qt.float()-qr.float()).abs().max().item()
        ek = (kt.float()-kr.float()).abs().max().item()
        print(f"  {B=} H={H} N={N} D={D}: {'✅' if o else '❌'} q_err={eq:.6f} k_err={ek:.6f}")
        if not o: ok_all = False
    return ok_all


# ============================================================================
# Benchmarks
# ============================================================================

def bench(name, fn_ref, fn_tri):
    """Benchmark with do_bench, return (ref_ms, tri_ms, speedup)."""
    # Warmup
    for _ in range(3): fn_ref(); fn_tri()
    torch.cuda.synchronize()
    t_ref = triton.testing.do_bench(fn_ref, warmup=WARMUP, rep=REP)
    t_tri = triton.testing.do_bench(fn_tri, warmup=WARMUP, rep=REP)
    sp = t_ref / t_tri
    print(f"  {name:<25s} ref={t_ref*1e3:7.1f}us  tri={t_tri*1e3:7.1f}us  speedup={sp:.2f}x")
    return t_ref, t_tri, sp


def run_benchmarks():
    print("\n" + "=" * 60)
    print("BENCHMARKS (triton.testing.do_bench)")
    print("=" * 60)

    # base_lora @ 512²
    x1024, bw, bb, la, lb = make_base_lora_inputs(1, 1024, 2048, 2048)
    bench("base_lora (SA-proj)",
          lambda: base_lora.reference(x1024, bw, bb, la, lb),
          lambda: base_lora.fused(x1024, bw, bb, la, lb)[0])

    # base_lora @ MLP
    xm, bwm, bbm, lam, lbm = make_base_lora_inputs(1, 1024, 2048, 8192)
    bench("base_lora (MLP-w1)",
          lambda: base_lora.reference(xm, bwm, bbm, lam, lbm),
          lambda: base_lora.fused(xm, bwm, bbm, lam, lbm)[0])

    # qkv_lora
    xq,bwq,bwk,bwv,bbq,bbk,bbv,laq,lak,lav,lbq,lbk,lbv = make_qkv_inputs(1, 1024, 2048, 32)
    bench("qkv_lora (SA)",
          lambda: qkv_lora.reference(xq,bwq,bwk,bwv,bbq,bbk,bbv,laq,lak,lav,lbq,lbk,lbv),
          lambda: qkv_lora.fused(xq,bwq,bwk,bwv,bbq,bbk,bbv,laq,lak,lav,lbq,lbk,lbv))

    # ffn_silu
    xf, bwf, bbf, laf, lbf = make_ffn_inputs(1, 1024)
    bench("ffn_silu (MLP W1+SiLU)",
          lambda: ffn_silu.reference(xf, bwf, bbf, laf, lbf),
          lambda: ffn_silu.fused(xf, bwf, bbf, laf, lbf))

    # adaln_norm
    xa, sca, sha, wa, ba = make_adaln_inputs(1, 1024, 2048)
    bench("adaln_norm",
          lambda: adaln_norm.reference(xa, sca, sha, wa, ba),
          lambda: adaln_norm.fused(xa, sca, sha, wa, ba)[0])

    # output_residual
    ao, ro, go, bwo, bbo, lao, lbo = make_output_res_inputs(1, 1024, 2048, 32)
    bench("output_residual",
          lambda: output_residual.reference(ao, ro, go, bwo, bbo, lao, lbo),
          lambda: output_residual.fused(ao, ro, go, bwo, bbo, lao, lbo))

    # fused_rope
    qr, kr, rope_emb = make_rope_inputs(1, 16, 1024, 128, 128)
    bench("fused_rope (Q+K)",
          lambda: rope_ref_anima(qr, kr, rope_emb),
          lambda: fused_rope(qr, kr, rope_emb))


if __name__ == "__main__":
    gpu = detect_gpu()
    print_gpu_info(gpu)

    print("=" * 60)
    print("ACCURACY TESTS")
    print("=" * 60)

    results = [
        ("base_lora",        test_base_lora()),
        ("qkv_lora",         test_qkv_lora()),
        ("ffn_silu",         test_ffn_silu()),
        ("adaln_norm",       test_adaln_norm()),
        ("output_residual",  test_output_residual()),
        ("fused_rope_3d",    test_fused_rope()),
    ]

    print("\n" + "=" * 60)
    acc_ok = all(r[1] for r in results)
    print(f"ACCURACY: {'✅ ALL PASSED' if acc_ok else '❌ SOME FAILED'}")
    for name, ok in results:
        print(f"  {name:<25s} {'✅' if ok else '❌'}")

    run_benchmarks()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
