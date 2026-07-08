"""Accuracy and performance tests for ``base_lora``."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import triton.testing
from triton_ops.config import detect_gpu, print_gpu_info
from triton_ops.lora.base_lora import fused, reference


_TEST_CONFIGS = [
    # (name,            B, N,    C_in, C_out, rank)
    ("sa-proj-256",     1,   256, 2048, 2048, 32),
    ("sa-proj-512",     1,  1024, 2048, 2048, 32),
    ("sa-proj-768",     1,  2304, 2048, 2048, 32),
    ("sa-proj-512-b2",  2,  1024, 2048, 2048, 32),
    ("sa-proj-512-r64", 1,  1024, 2048, 2048, 64),
    ("mlp-layer1-512",  1,  1024, 2048, 8192, 32),
    ("mlp-layer2-512",  1,  1024, 8192, 2048, 32),
    ("tiny",            1,    64, 2048, 2048, 32),
]


def _make_inputs(config, device="cuda"):
    _, B, N, C_in, C_out, rank = config
    torch.manual_seed(42)
    x = torch.randn(B, N, C_in, dtype=torch.bfloat16, device=device) * 0.1
    bw = torch.randn(C_out, C_in, dtype=torch.bfloat16, device=device) * 0.02
    bb = torch.randn(C_out, dtype=torch.bfloat16, device=device) * 0.01
    la = torch.randn(rank, C_in, dtype=torch.bfloat16, device=device) * 0.02
    lb = torch.randn(C_out, rank, dtype=torch.bfloat16, device=device) * 0.02
    return x, bw, bb, la, lb


def test_accuracy() -> bool:
    """Verify Triton output matches PyTorch reference."""
    all_ok = True
    for cfg in _TEST_CONFIGS:
        x, bw, bb, la, lb = _make_inputs(cfg)
        with torch.no_grad():
            y_ref = reference(x, bw, bb, la, lb, rank=cfg[5])
            y_tri, _ = fused(x, bw, bb, la, lb, rank=cfg[5])
        ok = torch.allclose(y_tri, y_ref, rtol=1e-2, atol=1e-4)
        err = (y_tri.float() - y_ref.float()).abs().max().item()
        status = "✅" if ok else "❌"
        print(f"  {cfg[0]:<20} {status}  max_err={err:.6f}")
        if not ok:
            all_ok = False
    return all_ok


def benchmark(warmup_ms: int = 50, rep_ms: int = 200) -> None:
    """Benchmark with ``triton.testing.do_bench``."""
    print(f"  {'Config':<20} {'cuDNN ref':>10}  {'Triton':>10}  {'vsRef':>8}")
    print(f"  {'-'*20} {'-'*10}  {'-'*10}  {'-'*8}")

    for cfg in _TEST_CONFIGS:
        x, bw, bb, la, lb = _make_inputs(cfg)
        rank = cfg[5]

        t_ref = triton.testing.do_bench(
            lambda: reference(x, bw, bb, la, lb, rank=rank),
            warmup=warmup_ms, rep=rep_ms,
        )
        t_tri = triton.testing.do_bench(
            lambda: fused(x, bw, bb, la, lb, rank=rank),
            warmup=warmup_ms, rep=rep_ms,
        )
        print(f"  {cfg[0]:<20} {t_ref:>9.3f}ms  {t_tri:>9.3f}ms  {t_ref/t_tri:>7.2f}x")


if __name__ == "__main__":
    gpu = detect_gpu()
    print_gpu_info(gpu)
    print()

    print("=== Accuracy ===")
    ok = test_accuracy()
    print(f"\n  {'All passed' if ok else 'FAILURES'}")

    print("\n=== Benchmark (do_bench) ===")
    benchmark()
