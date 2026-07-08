"""
Fused Output Projection + Residual + AdaLN Gate (BF16).

Anima residual connection after every sublayer::

    x = residual + gate * (base_linear(attn_out) + lora_scale * lora(attn_out))

Fusion applies the LoRA output projection, AdaLN *gate* scaling, and
residual addition in one kernel — saving two element-wise HBM round-trips.

Used 3× per AnimaBlock (after self-attention, cross-attention, MLP),
totalling 84 calls per forward pass.
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
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> torch.Tensor:
    """Reference: base + lora → gate → residual."""
    scale = alpha / rank
    xf = attn_out.reshape(-1, attn_out.shape[-1])
    base_out = torch.nn.functional.linear(xf, base_weight, base_bias)
    h = torch.nn.functional.linear(xf, lora_a)
    lora_out = torch.nn.functional.linear(h, lora_b)
    proj = (base_out + lora_out * scale).reshape_as(attn_out)
    return residual + gate * proj


# ============================================================================
# Triton kernel
# ============================================================================

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_C": 64}, num_warps=4, num_stages=2),
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
    residual_ptr,
    lora_a_ptr,
    lora_b_ptr,
    gate_ptr,
    out_ptr,
    lora_scale: tl.constexpr,
    total_rows,
    features,
    rank: tl.constexpr,
    stride_x_0,
    stride_lb_0,                # stride for dim 0 of lora_b (= rank)
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """LoRA output-proj + gate + residual in one pass.

    Phase A — ``h = x @ lora_A^T``.
    Phase B — ``out = residual + gate * (base_out + lora_scale * h @ lora_B^T)``.
    """
    num_row_blocks = tl.cdiv(total_rows, BLOCK_M)
    pid = tl.program_id(0)
    rank_offs = tl.arange(0, rank)
    stride_la_0 = features  # lora_a: [rank, features] → stride_0 = features

    for block_idx in range(pid, num_row_blocks, tl.num_programs(0)):
        row_start = block_idx * BLOCK_M
        row_offs = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offs < total_rows

        x_base = x_ptr + row_offs[:, None] * stride_x_0
        bo_base = base_out_ptr + row_offs[:, None] * features
        res_base = residual_ptr + row_offs[:, None] * stride_x_0
        out_base = out_ptr + row_offs[:, None] * stride_x_0

        # ---- Phase A: h = x @ lora_A^T ----
        h = tl.zeros([BLOCK_M, rank], dtype=tl.float32)

        for k_start in range(0, features, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < features

            x_block = tl.load(
                x_base + k_offs[None, :],
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            la = tl.load(
                lora_a_ptr + rank_offs[:, None] * stride_la_0 + k_offs[None, :],
                mask=k_mask[None, :], other=0.0,
            ).to(tl.float32)
            h += tl.dot(x_block, tl.trans(la))

        # ---- Phase B: residual + gate * (base + lora) ----
        for c_start in range(0, features, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < features

            base_block = tl.load(
                bo_base + c_offs[None, :],
                mask=row_mask[:, None] & c_mask[None, :], other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                lora_b_ptr + c_offs[:, None] * stride_lb_0 + rank_offs[None, :],
                mask=c_mask[:, None], other=0.0,
            ).to(tl.float32)

            proj = base_block + lora_scale * tl.dot(h, tl.trans(lb))

            g = tl.load(gate_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)
            res_block = tl.load(
                res_base + c_offs[None, :],
                mask=row_mask[:, None] & c_mask[None, :], other=0.0,
            ).to(tl.float32)

            tl.store(
                out_base + c_offs[None, :],
                (res_block + g[None, :] * proj).to(tl.bfloat16),
                mask=row_mask[:, None] & c_mask[None, :],
            )


# ============================================================================
# Public API
# ============================================================================

def fused(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> torch.Tensor:
    """Fused output projection + residual + gate.

    Args:
        attn_out:    ``[*, features]`` bf16  sublayer output.
        residual:    ``[*, features]`` bf16  original *x*.
        gate:        ``[features]``   bf16  AdaLN gate.
        base_weight: ``[features, features]``.
        base_bias:   ``[features]`` or ``None``.
        lora_a:      ``[rank, features]``.
        lora_b:      ``[features, rank]``.
        alpha:       LoRA scaling (default ``16``).
        rank:        LoRA rank (default ``32``).

    Returns:
        ``y`` — ``[*, features]`` bf16.
    """
    *batch_dims, features = attn_out.shape
    total_rows = 1
    for d in batch_dims:
        total_rows *= d

    lora_scale = alpha / rank
    xf = attn_out.reshape(total_rows, features).contiguous()
    res_flat = residual.reshape(total_rows, features).contiguous()

    # cuDNN handles the base matmul
    base_out = torch.nn.functional.linear(xf, base_weight, base_bias)

    out_flat = torch.empty(total_rows, features, dtype=torch.bfloat16, device=attn_out.device)

    grid = lambda meta: (min(triton.cdiv(total_rows, meta["BLOCK_M"]), MAX_GRID_SIZE),)
    _kernel[grid](
        xf, base_out, res_flat,
        lora_a.contiguous(), lora_b.contiguous(),
        gate.contiguous(), out_flat,
        lora_scale=lora_scale,
        total_rows=total_rows,
        features=features,
        rank=rank,
        stride_x_0=xf.stride(0),
        stride_lb_0=lora_b.stride(0),
    )

    return out_flat.reshape(*batch_dims, features)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["total_rows", "features", "rank"])
@triton.jit
def _kernel_anima_gate(
    x_ptr,
    base_out_ptr,
    residual_ptr,
    lora_a_ptr,
    lora_b_ptr,
    gate_ptr,
    out_ptr,
    lora_scale: tl.constexpr,
    total_rows,
    in_features,
    out_features,
    spatial_rows: tl.constexpr,
    rank: tl.constexpr,
    stride_x_0,
    stride_res_0,
    stride_out_0,
    stride_lb_0,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Anima residual: ``residual + gate[B,T,D] * LoRAProj(x)``."""
    num_row_blocks = tl.cdiv(total_rows, BLOCK_M)
    pid = tl.program_id(0)
    rank_offs = tl.arange(0, rank)
    stride_la_0 = in_features

    for block_idx in range(pid, num_row_blocks, tl.num_programs(0)):
        row_start = block_idx * BLOCK_M
        row_offs = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offs < total_rows

        x_base = x_ptr + row_offs[:, None] * stride_x_0
        bo_base = base_out_ptr + row_offs[:, None] * out_features
        res_base = residual_ptr + row_offs[:, None] * stride_res_0
        out_base = out_ptr + row_offs[:, None] * stride_out_0
        gate_base = (row_offs // spatial_rows)[:, None] * out_features

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
            proj = base_block + lora_scale * tl.dot(h, tl.trans(lb))
            gate = tl.load(
                gate_ptr + gate_base + c_offs[None, :],
                mask=row_mask[:, None] & c_mask[None, :], other=0.0,
            ).to(tl.float32)
            res_block = tl.load(
                res_base + c_offs[None, :],
                mask=row_mask[:, None] & c_mask[None, :], other=0.0,
            ).to(tl.float32)
            tl.store(
                out_base + c_offs[None, :],
                (res_block + gate * proj).to(tl.bfloat16),
                mask=row_mask[:, None] & c_mask[None, :],
            )


def fused_anima_gate(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> torch.Tensor:
    """Fused output projection + Anima per-time gate + residual.

    ``x`` and ``residual`` are ``[B,T,H,W,D]``. ``gate`` is broadcast-shaped
    ``[B,T,1,1,D]`` or contiguous ``[B,T,D]`` equivalent.
    """
    B, T, H, W, in_features = x.shape
    out_features = residual.shape[-1]
    total_rows = B * T * H * W
    spatial_rows = H * W
    lora_scale = alpha / rank

    xf = x.reshape(total_rows, in_features).contiguous()
    res_flat = residual.reshape(total_rows, out_features).contiguous()
    gate_flat = gate.reshape(B * T, out_features).contiguous()
    base_out = torch.nn.functional.linear(xf, base_weight, base_bias)
    out_flat = torch.empty(total_rows, out_features, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (min(triton.cdiv(total_rows, meta["BLOCK_M"]), MAX_GRID_SIZE),)
    _kernel_anima_gate[grid](
        xf, base_out, res_flat,
        lora_a.contiguous(), lora_b.contiguous(),
        gate_flat, out_flat,
        lora_scale=lora_scale,
        total_rows=total_rows,
        in_features=in_features,
        out_features=out_features,
        spatial_rows=spatial_rows,
        rank=rank,
        stride_x_0=xf.stride(0),
        stride_res_0=res_flat.stride(0),
        stride_out_0=out_flat.stride(0),
        stride_lb_0=lora_b.stride(0),
    )

    return out_flat.reshape(B, T, H, W, out_features)


class _FusedAnimaGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, gate, base_weight, base_bias, lora_a, lora_b, alpha: float, rank: int):
        ctx.has_bias = base_bias is not None
        bias = base_bias if base_bias is not None else x.new_empty(0)
        ctx.save_for_backward(x, residual, gate, base_weight, bias, lora_a, lora_b)
        ctx.alpha = alpha
        ctx.rank = rank
        return fused_anima_gate(x, residual, gate, base_weight, base_bias, lora_a, lora_b, alpha=alpha, rank=rank)

    @staticmethod
    def backward(ctx, grad_out):
        x, residual, gate, base_weight, bias, lora_a, lora_b = ctx.saved_tensors
        base_bias = bias if ctx.has_bias else None
        with torch.enable_grad():
            x_r = x.detach().requires_grad_(ctx.needs_input_grad[0])
            residual_r = residual.detach().requires_grad_(ctx.needs_input_grad[1])
            gate_r = gate.detach().requires_grad_(ctx.needs_input_grad[2])
            bw_r = base_weight.detach().requires_grad_(ctx.needs_input_grad[3])
            bb_r = base_bias.detach().requires_grad_(ctx.needs_input_grad[4]) if base_bias is not None else None
            la_r = lora_a.detach().requires_grad_(ctx.needs_input_grad[5])
            lb_r = lora_b.detach().requires_grad_(ctx.needs_input_grad[6])
            B, T, H, W, in_features = x_r.shape
            out_features = residual_r.shape[-1]
            xf = x_r.reshape(-1, in_features)
            base_out = torch.nn.functional.linear(xf, bw_r, bb_r)
            h = torch.nn.functional.linear(xf, la_r)
            lora_out = torch.nn.functional.linear(h, lb_r)
            proj = (base_out + lora_out * (ctx.alpha / ctx.rank)).reshape(B, T, H, W, out_features)
            y = residual_r + gate_r * proj
        inputs = tuple(
            v for v in (x_r, residual_r, gate_r, bw_r, bb_r, la_r, lb_r)
            if v is not None and v.requires_grad
        )
        grad_inputs = torch.autograd.grad(y, inputs, grad_out, allow_unused=True)
        grads = []
        idx = 0
        for v in (x_r, residual_r, gate_r, bw_r, bb_r, la_r, lb_r):
            if v is None or not v.requires_grad:
                grads.append(None)
            else:
                grads.append(grad_inputs[idx])
                idx += 1
        return (*grads, None, None)


def fused_anima_gate_autograd(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> torch.Tensor:
    """Autograd-compatible Anima output residual wrapper."""
    return _FusedAnimaGateFn.apply(x, residual, gate, base_weight, base_bias, lora_a, lora_b, alpha, rank)
