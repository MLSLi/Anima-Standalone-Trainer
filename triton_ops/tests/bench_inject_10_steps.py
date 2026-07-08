#!/usr/bin/env python3
"""Run 10 isolated Anima Block forward steps for baseline or Triton-injected path."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from library.anima_models import Block
import triton_inject


class FakeLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int = 32, alpha: float = 16.0):
        super().__init__()
        self.lora_name = "fake"
        self.lora_dim = rank
        self._triton_rank = rank
        self.scale = alpha / rank
        self.multiplier = 1.0
        self.split_dims = None
        self.dropout = None
        self.rank_dropout = None
        self.module_dropout = None
        self.ggpo_sigma = None
        self.ggpo_beta = None
        self._base_weight = linear.weight.data
        self._base_bias = linear.bias.data if linear.bias is not None else None
        self.org_forward = linear.forward
        self.lora_down = torch.nn.Linear(
            linear.in_features, rank, bias=False, device=linear.weight.device, dtype=linear.weight.dtype
        )
        self.lora_up = torch.nn.Linear(
            rank, linear.out_features, bias=False, device=linear.weight.device, dtype=linear.weight.dtype
        )
        torch.nn.init.normal_(self.lora_down.weight, std=0.02)
        torch.nn.init.normal_(self.lora_up.weight, std=0.02)

    def forward(self, x):
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale


def wrap_lora(linear: torch.nn.Linear) -> FakeLoRA:
    lora = FakeLoRA(linear)
    linear.forward = lora.forward
    return lora


def build_case(dtype: torch.dtype):
    torch.manual_seed(20260706)
    D = 2048
    heads = 16
    B, T, H, W = 1, 1, 16, 16
    N = T * H * W
    block = Block(x_dim=D, context_dim=D, num_heads=heads).cuda().to(dtype).eval()

    loras = []
    for linear in (
        block.self_attn.q_proj,
        block.self_attn.k_proj,
        block.self_attn.v_proj,
        block.self_attn.output_proj,
        block.cross_attn.q_proj,
        block.cross_attn.k_proj,
        block.cross_attn.v_proj,
        block.cross_attn.output_proj,
        block.mlp.layer1,
        block.mlp.layer2,
    ):
        loras.append(wrap_lora(linear))

    x = torch.randn(B, T, H, W, D, device="cuda", dtype=dtype) * 0.1
    emb = torch.randn(B, T, D, device="cuda", dtype=dtype) * 0.1
    ctx = torch.randn(B, N, D, device="cuda", dtype=dtype) * 0.1
    rope = torch.randn(N, 1, 1, D // heads, device="cuda", dtype=torch.float32) * 0.1
    dit = types.SimpleNamespace(blocks=[block])
    return block, dit, loras, x, emb, ctx, rope


def sync() -> None:
    torch.cuda.synchronize()


def run(mode: str, steps: int, warmup: int, backward_probe: bool) -> dict:
    dtype = torch.bfloat16
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    block, dit, loras, x, emb, ctx, rope = build_case(dtype)
    params = [p for lora in loras for p in lora.parameters()]
    optimizer = torch.optim.SGD(params, lr=1e-4)

    if mode == "optimized":
        counts = triton_inject.inject_optimized(dit, network=None, rank=32, alpha=16)
    else:
        counts = {}

    sync()
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        y = block._forward(x, emb, ctx, rope_emb_L_1_1_D=rope)
        loss = y.float().mean()
        loss.backward()
        optimizer.step()
        del y, loss
    sync()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_alloc = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()
    t0 = time.perf_counter()
    checksum = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        y = block._forward(x, emb, ctx, rope_emb_L_1_1_D=rope)
        loss = y.float().mean()
        checksum += float(loss.item())
        loss.backward()
        optimizer.step()
        del y, loss
    sync()
    elapsed = time.perf_counter() - t0
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    end_alloc = torch.cuda.memory_allocated()
    end_reserved = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()

    backward_status = "ok"
    if backward_probe:
        grad_ok = all(p.grad is not None for p in params)
        backward_status = "ok" if grad_ok else "missing parameter gradients"

    return {
        "mode": mode,
        "counts": counts,
        "steps": steps,
        "warmup": warmup,
        "elapsed_ms": elapsed * 1000.0,
        "avg_ms": elapsed * 1000.0 / steps,
        "checksum": checksum,
        "start_alloc_mib": start_alloc / 2**20,
        "end_alloc_mib": end_alloc / 2**20,
        "peak_alloc_mib": peak_alloc / 2**20,
        "start_reserved_mib": start_reserved / 2**20,
        "end_reserved_mib": end_reserved / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
        "nvidia_used_mib": (total - free) / 2**20,
        "nvidia_free_mib": free / 2**20,
        "backward_status": backward_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backward-probe", action="store_true")
    args = parser.parse_args()
    result = run(args.mode, args.steps, args.warmup, args.backward_probe)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
