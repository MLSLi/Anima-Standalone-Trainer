"""
Fused AdaLN modulation + LayerNorm (BF16).

Every AnimaBlock calls this twice (56 calls/forward).
Fusion substitutes 5–6 PyTorch kernel launches with a single Triton kernel.

Numerics: three-pass FP32 accumulation for correctness.
Final ``FP32→BF16`` store may differ from PyTorch rounding by ≤0.016.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ============================================================================
# Reference (pytorch)
# ============================================================================

def reference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PyTorch reference: modulation + LayerNorm + affine."""
    x_mod = x * scale + shift
    mean = x_mod.mean(dim=-1, keepdim=True)
    var = x_mod.var(dim=-1, keepdim=True, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)
    return (x_mod - mean) * rstd * weight + bias


# ============================================================================
# Triton kernel
# ============================================================================

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_C": 1024}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["features"])
@triton.jit
def _kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    total_rows,
    features,
    eps: tl.constexpr,
    stride_row,
    BLOCK_C: tl.constexpr,
):
    """Per-row AdaLN + LayerNorm with three-pass FP32 accumulation.

    One program per row — grid size = ``total_rows``.
    """
    row_idx = tl.program_id(0)
    if row_idx >= total_rows:
        return

    row_offset = row_idx * stride_row

    # ---- Pass 1: sum → mean ----
    acc_sum = tl.zeros([], dtype=tl.float32)
    acc_count = tl.zeros([], dtype=tl.float32)

    for c_start in range(0, features, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < features

        x_val = tl.load(x_ptr + row_offset + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        s_val = tl.load(scale_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        sh_val = tl.load(shift_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        x_mod = x_val * s_val + sh_val
        acc_sum += tl.sum(tl.where(c_mask, x_mod, 0.0))
        acc_count += tl.sum(c_mask.to(tl.float32))

    mean = acc_sum / acc_count

    # ---- Pass 2: sum of squared deviations → rstd ----
    acc_m2 = tl.zeros([], dtype=tl.float32)

    for c_start in range(0, features, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < features

        x_val = tl.load(x_ptr + row_offset + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        s_val = tl.load(scale_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        sh_val = tl.load(shift_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        x_mod = x_val * s_val + sh_val
        diff = x_mod - mean
        acc_m2 += tl.sum(tl.where(c_mask, diff * diff, 0.0))

    var = acc_m2 / acc_count
    rstd = 1.0 / tl.sqrt(var + eps)

    # Save statistics for backward pass
    if mean_ptr is not None:
        tl.store(mean_ptr + row_idx, mean)
        tl.store(rstd_ptr + row_idx, rstd)

    # ---- Pass 3: normalize → affine → store ----
    for c_start in range(0, features, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < features

        x_val = tl.load(x_ptr + row_offset + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        s_val = tl.load(scale_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        sh_val = tl.load(shift_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        w_val = tl.load(weight_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        b_val = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        x_mod = x_val * s_val + sh_val
        x_norm = (x_mod - mean) * rstd
        y = x_norm * w_val + b_val

        tl.store(out_ptr + row_offset + c_offs, y.to(tl.bfloat16), mask=c_mask)


# ============================================================================
# Public API
# ============================================================================

def fused(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    save_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused AdaLN modulation + LayerNorm.

    Args:
        x:      ``[*, features]`` bf16.
        scale:  ``[features]`` bf16  (``1 + adaln_modulation``).
        shift:  ``[features]`` bf16.
        weight: ``[features]`` bf16  (LayerNorm gamma).
        bias:   ``[features]`` bf16  (LayerNorm beta).
        eps:    LayerNorm epsilon (default ``1e-6``).
        save_stats:  Return mean and rstd for backward.

    Returns:
        ``(y, mean, rstd)`` — *y* ``[*, features]`` bf16.
    """
    *batch_dims, features = x.shape
    total_rows = 1
    for d in batch_dims:
        total_rows *= d

    x_flat = x.reshape(total_rows, features).contiguous()
    out_flat = torch.empty(total_rows, features, dtype=torch.bfloat16, device=x.device)

    mean_out = (
        torch.empty(total_rows, dtype=torch.float32, device=x.device)
        if save_stats
        else x.new_empty(1)
    )
    rstd_out = (
        torch.empty(total_rows, dtype=torch.float32, device=x.device)
        if save_stats
        else x.new_empty(1)
    )

    _kernel[(total_rows,)](
        x_flat,
        scale, shift, weight, bias,
        out_flat,
        mean_out if save_stats else None,
        rstd_out if save_stats else None,
        total_rows,
        features,
        eps,
        x_flat.stride(0),
    )

    y = out_flat.reshape(*batch_dims, features)
    if save_stats:
        return y, mean_out.reshape(total_rows), rstd_out.reshape(total_rows)
    return y, None, None
