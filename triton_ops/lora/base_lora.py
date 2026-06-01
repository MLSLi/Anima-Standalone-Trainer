"""
Fused Base Linear + LoRA (BF16).

Computes ``y = base_linear(x) + (alpha/r) * lora_B(lora_A(x))`` in one
fused kernel.  cuDNN handles the compute-bound base matmul; Triton fuses
the memory-bound LoRA down- and up-projections so that the intermediate
hidden state ``h`` never leaves SRAM.

Applicable to every ``nn.Linear`` layer with a LoRA adapter in the Anima
DiT architecture (self-attention, cross-attention, MLP, output
projections, and ``x_embedder`` / ``final_layer``).
"""

from __future__ import annotations

from typing import Optional, Tuple

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
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Reference implementation using three separate cuDNN matmuls."""
    scale = alpha / rank
    x_2d = x.reshape(-1, x.shape[-1])
    x_lora = torch.nn.functional.dropout(x_2d, p=dropout_p, training=dropout_p > 0)
    base_out = torch.nn.functional.linear(x_2d, base_weight, base_bias)
    h = torch.nn.functional.linear(x_lora, lora_a)
    lora_out = torch.nn.functional.linear(h, lora_b)
    *batch_dims, _ = x.shape
    out_features = base_weight.shape[0]
    return (base_out + lora_out * scale).reshape(*batch_dims, out_features)


# ============================================================================
# Triton kernel
# ============================================================================

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_C": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_C": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_C": 64}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_C": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_K": 256, "BLOCK_C": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 256, "BLOCK_C": 64}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["total_rows", "in_features", "out_features"])
@triton.jit
def _kernel(
    # Input buffers (bf16)
    x_ptr,                     # [total_rows, in_features]
    base_out_ptr,              # [total_rows, out_features]  — cuDNN pre-computed
    lora_a_ptr,                # [rank, in_features]
    lora_b_ptr,                # [out_features, rank]
    # Output buffers
    out_ptr,                   # [total_rows, out_features]
    h_save_ptr,                # [total_rows, rank] or null — for backward
    # Scalars
    lora_scale: tl.constexpr,
    total_rows,
    in_features,
    out_features,
    rank: tl.constexpr,        # LoRA rank — constexpr for tl.zeros shape
    # Strides
    stride_x_0,
    stride_base_0,
    stride_out_0,
    stride_la_0,
    stride_lb_0,
    # Tiles (auto-tuned)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Split-design LoRA fusion kernel with grid-stride loop.

    Phase A — *x* loaded once, ``h = x @ lora_A^T`` accumulated in SRAM.
    Phase B — per *out_features* block: ``base + lora_scale * h @ lora_B^T``.
    """
    num_row_blocks = tl.cdiv(total_rows, BLOCK_M)
    pid = tl.program_id(0)
    rank_offs = tl.arange(0, rank)  # power-of-2

    # Grid-stride: each program may handle multiple row blocks
    for block_idx in range(pid, num_row_blocks, tl.num_programs(0)):
        row_start = block_idx * BLOCK_M
        row_offs = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offs < total_rows

        x_base = x_ptr + row_offs[:, None] * stride_x_0
        bo_base = base_out_ptr + row_offs[:, None] * out_features
        out_base = out_ptr + row_offs[:, None] * out_features

        # ---------------------------------------------------------------
        # Phase A: h = x @ lora_A^T   [BLOCK_M, rank] in SRAM
        # ---------------------------------------------------------------
        h = tl.zeros([BLOCK_M, rank], dtype=tl.float32)

        for k_start in range(0, in_features, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < in_features

            x_block = tl.load(
                x_base + k_offs[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            la_block = tl.load(
                lora_a_ptr + rank_offs[:, None] * stride_la_0 + k_offs[None, :],
                mask=k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            h += tl.dot(x_block, tl.trans(la_block))

        # Optionally save h for backward-pass recomputation
        if h_save_ptr is not None:
            h_save_base = h_save_ptr + row_offs[:, None] * rank
            tl.store(
                h_save_base + rank_offs[None, :],
                h.to(tl.bfloat16),
                mask=row_mask[:, None],
            )

        # ---------------------------------------------------------------
        # Phase B: lora_out = lora_scale * h @ lora_B^T + base_out → store
        # ---------------------------------------------------------------
        for c_start in range(0, out_features, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < out_features

            base_block = tl.load(
                bo_base + c_offs[None, :],
                mask=row_mask[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            lb_block = tl.load(
                lora_b_ptr + c_offs[:, None] * stride_lb_0 + rank_offs[None, :],
                mask=c_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            lora_out = lora_scale * tl.dot(h, tl.trans(lb_block))
            result = base_block + lora_out

            tl.store(
                out_base + c_offs[None, :],
                result.to(tl.bfloat16),
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
    save_h: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused base-linear + LoRA forward.

    Args:
        x:           ``[*, in_features]`` bf16 input.
        base_weight: ``[out_features, in_features]`` bf16 frozen weight.
        base_bias:   ``[out_features]`` bf16 frozen bias (or ``None``).
        lora_a:      ``[rank, in_features]`` bf16 trainable down-projection.
        lora_b:      ``[out_features, rank]`` bf16 trainable up-projection.
        alpha:       LoRA scaling factor (default ``16``).
        rank:        LoRA rank (default ``32``).
        save_h:      If ``True``, return the LoRA hidden state for backward.

    Returns:
        ``(y, h)`` where *y* is ``[*, out_features]`` and *h* is ``None``
        unless ``save_h=True``.
    """
    *batch_dims, in_features = x.shape
    total_rows = 1
    for d in batch_dims:
        total_rows *= d

    out_features = base_weight.shape[0]
    lora_scale = alpha / rank
    device = x.device

    # Flatten batch dimensions
    x_flat = x.reshape(total_rows, in_features).contiguous()

    # cuDNN handles the base matmul
    base_out_flat = torch.nn.functional.linear(x_flat, base_weight, base_bias)

    # Allocate outputs
    out_flat = torch.empty(total_rows, out_features, dtype=torch.bfloat16, device=device)
    h_flat = (
        torch.empty(total_rows, rank, dtype=torch.bfloat16, device=device)
        if save_h
        else None
    )

    grid = lambda meta: (min(triton.cdiv(total_rows, meta["BLOCK_M"]), MAX_GRID_SIZE),)
    _kernel[grid](
        x_flat,
        base_out_flat,
        lora_a.contiguous(),
        lora_b.contiguous(),
        out_flat,
        h_flat if save_h else None,
        lora_scale=lora_scale,
        total_rows=total_rows,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        stride_x_0=x_flat.stride(0),
        stride_base_0=base_out_flat.stride(0),
        stride_out_0=out_flat.stride(0),
        stride_la_0=lora_a.stride(0),
        stride_lb_0=lora_b.stride(0),
    )

    y = out_flat.reshape(*batch_dims, out_features)
    if save_h:
        h = h_flat.reshape(total_rows, rank)
        return y, h
    return y, None
