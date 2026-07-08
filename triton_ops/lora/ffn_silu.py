"""
Fused FFN W1 + SiLU activation (BF16).

Anima MLP layer-1 (expand projection)::

    hidden = base_w1(x) + (alpha/r) * lora_B_w1(lora_A_w1(x))
    output = SiLU(hidden)              # SiLU = x * sigmoid(x)

Fusion saves one HBM round-trip for the large intermediate tensor
``hidden [*, 4*features]`` (16 MB/sample at 512², bf16).

Used once per AnimaBlock (28 calls/forward).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from triton_ops.config import MAX_GRID_SIZE


# ============================================================================
# Reference (pytorch)
# ============================================================================

def reference(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> torch.Tensor:
    """Reference: base matmul + LoRA matmuls + separate SiLU."""
    scale = alpha / rank
    *batch_dims, in_features = x.shape
    out_features = base_weight.shape[0]
    xf = x.reshape(-1, in_features)
    base_out = torch.nn.functional.linear(xf, base_weight, base_bias)
    h = torch.nn.functional.linear(xf, lora_a)
    lora_out = torch.nn.functional.linear(h, lora_b)
    hidden = base_out + lora_out * scale
    return torch.nn.functional.silu(hidden).reshape(*batch_dims, out_features)


# ============================================================================
# Triton kernel
# ============================================================================

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_C": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_C": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_C": 64}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_C": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_K": 256, "BLOCK_C": 128}, num_warps=4, num_stages=1),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["total_rows", "in_features", "out_features", "rank"])
@triton.jit
def _kernel(
    x_ptr,
    base_out_ptr,
    lora_a_ptr,
    lora_b_ptr,
    out_ptr,
    lora_scale: tl.constexpr,
    total_rows,
    in_features,
    out_features,
    rank: tl.constexpr,
    stride_x_0,
    stride_base_0,
    stride_out_0,
    stride_la_0,
    stride_lb_0,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """LoRA path + SiLU in SRAM.  Grid-stride loop for SM occupancy.

    Phase A — ``h = x @ lora_A^T`` accumulated in SRAM.
    Phase B — ``hidden = base_out + lora_scale * h @ lora_B^T``,
              then ``SiLU(hidden)`` before store.
    """
    num_row_blocks = tl.cdiv(total_rows, BLOCK_M)
    pid = tl.program_id(0)
    rank_offs = tl.arange(0, rank)

    for block_idx in range(pid, num_row_blocks, tl.num_programs(0)):
        row_start = block_idx * BLOCK_M
        row_offs = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offs < total_rows

        x_base = x_ptr + row_offs[:, None] * stride_x_0
        bo_base = base_out_ptr + row_offs[:, None] * out_features
        out_base = out_ptr + row_offs[:, None] * stride_out_0

        # ---- Phase A: h = x @ lora_A^T ----
        h = tl.zeros([BLOCK_M, rank], dtype=tl.float32)

        for k_start in range(0, in_features, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < in_features

            x_block = tl.load(
                x_base + k_offs[None, :],
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            la = tl.load(
                lora_a_ptr + rank_offs[:, None] * stride_la_0 + k_offs[None, :],
                mask=k_mask[None, :], other=0.0,
            ).to(tl.float32)
            h += tl.dot(x_block, tl.trans(la))

        # ---- Phase B: base + lora → SiLU → store ----
        for c_start in range(0, out_features, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < out_features

            base_block = tl.load(
                bo_base + c_offs[None, :],
                mask=row_mask[:, None] & c_mask[None, :], other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                lora_b_ptr + c_offs[:, None] * stride_lb_0 + rank_offs[None, :],
                mask=c_mask[:, None], other=0.0,
            ).to(tl.float32)

            hidden = base_block + lora_scale * tl.dot(h, tl.trans(lb))

            # SiLU:  x / (1 + exp(-x)),  with clamp for bf16 safety
            hidden_silu = hidden / (1.0 + tl.math.exp(tl.minimum(-hidden, 80.0)))

            tl.store(
                out_base + c_offs[None, :],
                hidden_silu.to(tl.bfloat16),
                mask=row_mask[:, None] & c_mask[None, :],
            )


# ============================================================================
# Public API
# ============================================================================

def fused(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> torch.Tensor:
    """Fused FFN W1 + LoRA + SiLU forward.

    Args:
        x:           ``[*, in_features]`` bf16.
        base_weight: ``[out_features, in_features]`` (typically ``[8192, 2048]``).
        base_bias:   ``[out_features]`` or ``None``.
        lora_a:      ``[rank, in_features]``.
        lora_b:      ``[out_features, rank]``.
        alpha:       LoRA scaling (default ``16``).
        rank:        LoRA rank (default ``32``).

    Returns:
        ``hidden_silu`` — ``[*, out_features]`` bf16.
    """
    *batch_dims, in_features = x.shape
    total_rows = 1
    for d in batch_dims:
        total_rows *= d

    out_features = base_weight.shape[0]
    lora_scale = alpha / rank
    x_flat = x.reshape(total_rows, in_features).contiguous()

    # cuDNN handles the base matmul
    base_out = torch.nn.functional.linear(x_flat, base_weight, base_bias)

    out_flat = torch.empty(total_rows, out_features, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (min(triton.cdiv(total_rows, meta["BLOCK_M"]), MAX_GRID_SIZE),)
    _kernel[grid](
        x_flat, base_out,
        lora_a.contiguous(), lora_b.contiguous(),
        out_flat,
        lora_scale=lora_scale,
        total_rows=total_rows,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        stride_x_0=x_flat.stride(0),
        stride_base_0=base_out.stride(0),
        stride_out_0=out_flat.stride(0),
        stride_la_0=lora_a.stride(0),
        stride_lb_0=lora_b.stride(0),
    )

    return out_flat.reshape(*batch_dims, out_features)
