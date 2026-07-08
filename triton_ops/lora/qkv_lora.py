"""
Batched Q/K/V LoRA fusion for Self-Attention (BF16).

Anima self-attention applies three independent linear + LoRA projections
(``q_proj``, ``k_proj``, ``v_proj``), all taking the *same* input *x*::

    q, k, v = [base(x) + (alpha/r)*lora_B(lora_A(x)) for each head]

Fusion loads *x* once, accumulates all three LoRA hidden states in SRAM,
and writes three output channels in a single kernel launch.

Applicable to every ``self_attn`` block in the Anima DiT (28 calls/forward).
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
    bw_q: torch.Tensor, bw_k: torch.Tensor, bw_v: torch.Tensor,
    bb_q: Optional[torch.Tensor], bb_k: Optional[torch.Tensor], bb_v: Optional[torch.Tensor],
    la_q: torch.Tensor, la_k: torch.Tensor, la_v: torch.Tensor,
    lb_q: torch.Tensor, lb_k: torch.Tensor, lb_v: torch.Tensor,
    alpha: float = 16.0, rank: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: three separate LoRA chains for Q, K, V."""
    scale = alpha / rank
    xf = x.reshape(-1, x.shape[-1])

    def _path(bw, bb, la, lb):
        b = torch.nn.functional.linear(xf, bw, bb)
        h = torch.nn.functional.linear(xf, la)
        return (b + torch.nn.functional.linear(h, lb) * scale).reshape_as(x)

    return (_path(bw_q, bb_q, la_q, lb_q),
            _path(bw_k, bb_k, la_k, lb_k),
            _path(bw_v, bb_v, la_v, lb_v))


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


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["total_rows", "features", "rank"])
@triton.jit
def _kernel(
    x_ptr,
    base_q_ptr, base_k_ptr, base_v_ptr,
    la_q_ptr, la_k_ptr, la_v_ptr,
    lb_q_ptr, lb_k_ptr, lb_v_ptr,
    out_q_ptr, out_k_ptr, out_v_ptr,
    lora_scale: tl.constexpr,
    total_rows,
    features,                   # C = 2048
    rank: tl.constexpr,
    stride_x_0,
    stride_base_0,
    stride_out_0,
    stride_la_0,
    stride_lb_0,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Three-head LoRA fusion.  *x* loaded once, three independent
    ``tl.dot`` calls share the same *x* block."""
    num_row_blocks = tl.cdiv(total_rows, BLOCK_M)
    pid = tl.program_id(0)
    rank_offs = tl.arange(0, rank)

    for block_idx in range(pid, num_row_blocks, tl.num_programs(0)):
        row_start = block_idx * BLOCK_M
        row_offs = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offs < total_rows

        x_base = x_ptr + row_offs[:, None] * stride_x_0

        # ---- Phase A: h_q, h_k, h_v (in SRAM) ----
        h_q = tl.zeros([BLOCK_M, rank], dtype=tl.float32)
        h_k = tl.zeros([BLOCK_M, rank], dtype=tl.float32)
        h_v = tl.zeros([BLOCK_M, rank], dtype=tl.float32)

        for k_start in range(0, features, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < features

            x_block = tl.load(
                x_base + k_offs[None, :],
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)

            # Q
            la_q = tl.load(
                la_q_ptr + rank_offs[:, None] * stride_la_0 + k_offs[None, :],
                mask=k_mask[None, :], other=0.0,
            ).to(tl.float32)
            h_q += tl.dot(x_block, tl.trans(la_q))
            # K
            la_k = tl.load(
                la_k_ptr + rank_offs[:, None] * stride_la_0 + k_offs[None, :],
                mask=k_mask[None, :], other=0.0,
            ).to(tl.float32)
            h_k += tl.dot(x_block, tl.trans(la_k))
            # V
            la_v = tl.load(
                la_v_ptr + rank_offs[:, None] * stride_la_0 + k_offs[None, :],
                mask=k_mask[None, :], other=0.0,
            ).to(tl.float32)
            h_v += tl.dot(x_block, tl.trans(la_v))

        # ---- Phase B: output per features block ----
        bq_base = base_q_ptr + row_offs[:, None] * stride_base_0
        bk_base = base_k_ptr + row_offs[:, None] * stride_base_0
        bv_base = base_v_ptr + row_offs[:, None] * stride_base_0
        oq_base = out_q_ptr + row_offs[:, None] * stride_out_0
        ok_base = out_k_ptr + row_offs[:, None] * stride_out_0
        ov_base = out_v_ptr + row_offs[:, None] * stride_out_0

        for c_start in range(0, features, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < features

            # Q channel
            bq = tl.load(bq_base + c_offs[None, :],
                         mask=row_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            lb_q = tl.load(lb_q_ptr + c_offs[:, None] * stride_lb_0 + rank_offs[None, :],
                           mask=c_mask[:, None], other=0.0).to(tl.float32)
            tl.store(oq_base + c_offs[None, :],
                     (bq + lora_scale * tl.dot(h_q, tl.trans(lb_q))).to(tl.bfloat16),
                     mask=row_mask[:, None] & c_mask[None, :])
            # K channel
            bk = tl.load(bk_base + c_offs[None, :],
                         mask=row_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            lb_k = tl.load(lb_k_ptr + c_offs[:, None] * stride_lb_0 + rank_offs[None, :],
                           mask=c_mask[:, None], other=0.0).to(tl.float32)
            tl.store(ok_base + c_offs[None, :],
                     (bk + lora_scale * tl.dot(h_k, tl.trans(lb_k))).to(tl.bfloat16),
                     mask=row_mask[:, None] & c_mask[None, :])
            # V channel
            bv = tl.load(bv_base + c_offs[None, :],
                         mask=row_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            lb_v = tl.load(lb_v_ptr + c_offs[:, None] * stride_lb_0 + rank_offs[None, :],
                           mask=c_mask[:, None], other=0.0).to(tl.float32)
            tl.store(ov_base + c_offs[None, :],
                     (bv + lora_scale * tl.dot(h_v, tl.trans(lb_v))).to(tl.bfloat16),
                     mask=row_mask[:, None] & c_mask[None, :])


# ============================================================================
# Public API
# ============================================================================

def fused(
    x: torch.Tensor,
    base_w_q: torch.Tensor, base_w_k: torch.Tensor, base_w_v: torch.Tensor,
    base_b_q: Optional[torch.Tensor],
    base_b_k: Optional[torch.Tensor],
    base_b_v: Optional[torch.Tensor],
    lora_a_q: torch.Tensor, lora_a_k: torch.Tensor, lora_a_v: torch.Tensor,
    lora_b_q: torch.Tensor, lora_b_k: torch.Tensor, lora_b_v: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Q/K/V LoRA forward.

    All Q/K/V projections share the same input *x* and have shape
    ``[features, features]``.

    Returns:
        ``(q, k, v)`` — each ``[*, features]`` bf16.
    """
    *batch_dims, features = x.shape
    total_rows = 1
    for d in batch_dims:
        total_rows *= d

    lora_scale = alpha / rank
    x_flat = x.reshape(total_rows, features).contiguous()

    # cuDNN handles base matmuls
    base_q = torch.nn.functional.linear(x_flat, base_w_q, base_b_q)
    base_k = torch.nn.functional.linear(x_flat, base_w_k, base_b_k)
    base_v = torch.nn.functional.linear(x_flat, base_w_v, base_b_v)

    out_q = torch.empty(total_rows, features, dtype=torch.bfloat16, device=x.device)
    out_k = torch.empty(total_rows, features, dtype=torch.bfloat16, device=x.device)
    out_v = torch.empty(total_rows, features, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (min(triton.cdiv(total_rows, meta["BLOCK_M"]), MAX_GRID_SIZE),)
    _kernel[grid](
        x_flat,
        base_q, base_k, base_v,
        lora_a_q.contiguous(), lora_a_k.contiguous(), lora_a_v.contiguous(),
        lora_b_q.contiguous(), lora_b_k.contiguous(), lora_b_v.contiguous(),
        out_q, out_k, out_v,
        lora_scale=lora_scale,
        total_rows=total_rows,
        features=features,
        rank=rank,
        stride_x_0=x_flat.stride(0),
        stride_base_0=base_q.stride(0),
        stride_out_0=out_q.stride(0),
        stride_la_0=lora_a_q.stride(0),
        stride_lb_0=lora_b_q.stride(0),
    )

    return (out_q.reshape(*batch_dims, features),
            out_k.reshape(*batch_dims, features),
            out_v.reshape(*batch_dims, features))


def fused_packed(
    x: torch.Tensor,
    base_w_qkv: torch.Tensor,
    base_b_qkv: Optional[torch.Tensor],
    lora_a_q: torch.Tensor, lora_a_k: torch.Tensor, lora_a_v: torch.Tensor,
    lora_b_q: torch.Tensor, lora_b_k: torch.Tensor, lora_b_v: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Q/K/V LoRA with one packed cuDNN base projection.

    ``base_w_qkv`` is ``torch.cat([Wq, Wk, Wv], dim=0)`` and
    ``base_b_qkv`` is the matching concatenated bias, or ``None``.
    """
    *batch_dims, features = x.shape
    total_rows = 1
    for d in batch_dims:
        total_rows *= d

    lora_scale = alpha / rank
    x_flat = x.reshape(total_rows, features).contiguous()

    base_qkv = torch.nn.functional.linear(x_flat, base_w_qkv, base_b_qkv)
    base_q, base_k, base_v = base_qkv.split(features, dim=1)

    out_q = torch.empty(total_rows, features, dtype=torch.bfloat16, device=x.device)
    out_k = torch.empty(total_rows, features, dtype=torch.bfloat16, device=x.device)
    out_v = torch.empty(total_rows, features, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (min(triton.cdiv(total_rows, meta["BLOCK_M"]), MAX_GRID_SIZE),)
    _kernel[grid](
        x_flat,
        base_q, base_k, base_v,
        lora_a_q.contiguous(), lora_a_k.contiguous(), lora_a_v.contiguous(),
        lora_b_q.contiguous(), lora_b_k.contiguous(), lora_b_v.contiguous(),
        out_q, out_k, out_v,
        lora_scale=lora_scale,
        total_rows=total_rows,
        features=features,
        rank=rank,
        stride_x_0=x_flat.stride(0),
        stride_base_0=base_q.stride(0),
        stride_out_0=out_q.stride(0),
        stride_la_0=lora_a_q.stride(0),
        stride_lb_0=lora_b_q.stride(0),
    )

    return (out_q.reshape(*batch_dims, features),
            out_k.reshape(*batch_dims, features),
            out_v.reshape(*batch_dims, features))


class _FusedPackedQKVFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base_w_qkv, base_b_qkv, la_q, la_k, la_v, lb_q, lb_k, lb_v, alpha: float, rank: int):
        ctx.has_bias = base_b_qkv is not None
        bias = base_b_qkv if base_b_qkv is not None else x.new_empty(0)
        ctx.save_for_backward(x, base_w_qkv, bias, la_q, la_k, la_v, lb_q, lb_k, lb_v)
        ctx.alpha = alpha
        ctx.rank = rank
        return fused_packed(x, base_w_qkv, base_b_qkv, la_q, la_k, la_v, lb_q, lb_k, lb_v, alpha=alpha, rank=rank)

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        x, base_w_qkv, bias, la_q, la_k, la_v, lb_q, lb_k, lb_v = ctx.saved_tensors
        base_b_qkv = bias if ctx.has_bias else None
        features = x.shape[-1]
        with torch.enable_grad():
            x_r = x.detach().requires_grad_(ctx.needs_input_grad[0])
            bw_r = base_w_qkv.detach().requires_grad_(ctx.needs_input_grad[1])
            bb_r = base_b_qkv.detach().requires_grad_(ctx.needs_input_grad[2]) if base_b_qkv is not None else None
            laq_r = la_q.detach().requires_grad_(ctx.needs_input_grad[3])
            lak_r = la_k.detach().requires_grad_(ctx.needs_input_grad[4])
            lav_r = la_v.detach().requires_grad_(ctx.needs_input_grad[5])
            lbq_r = lb_q.detach().requires_grad_(ctx.needs_input_grad[6])
            lbk_r = lb_k.detach().requires_grad_(ctx.needs_input_grad[7])
            lbv_r = lb_v.detach().requires_grad_(ctx.needs_input_grad[8])
            bw_q, bw_k, bw_v = bw_r.split(features, dim=0)
            if bb_r is None:
                bb_q = bb_k = bb_v = None
            else:
                bb_q, bb_k, bb_v = bb_r.split(features, dim=0)
            outputs = reference(
                x_r, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v,
                laq_r, lak_r, lav_r, lbq_r, lbk_r, lbv_r,
                alpha=ctx.alpha, rank=ctx.rank,
            )
        inputs = tuple(
            v for v in (x_r, bw_r, bb_r, laq_r, lak_r, lav_r, lbq_r, lbk_r, lbv_r)
            if v is not None and v.requires_grad
        )
        grad_inputs = torch.autograd.grad(outputs, inputs, (grad_q, grad_k, grad_v), allow_unused=True)
        grads = []
        idx = 0
        for v in (x_r, bw_r, bb_r, laq_r, lak_r, lav_r, lbq_r, lbk_r, lbv_r):
            if v is None or not v.requires_grad:
                grads.append(None)
            else:
                grads.append(grad_inputs[idx])
                idx += 1
        return (*grads, None, None)


def fused_packed_autograd(
    x: torch.Tensor,
    base_w_qkv: torch.Tensor,
    base_b_qkv: Optional[torch.Tensor],
    lora_a_q: torch.Tensor, lora_a_k: torch.Tensor, lora_a_v: torch.Tensor,
    lora_b_q: torch.Tensor, lora_b_k: torch.Tensor, lora_b_v: torch.Tensor,
    alpha: float = 16.0,
    rank: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd-compatible packed QKV: Triton forward, PyTorch recompute backward."""
    return _FusedPackedQKVFn.apply(
        x, base_w_qkv, base_b_qkv,
        lora_a_q, lora_a_k, lora_a_v,
        lora_b_q, lora_b_k, lora_b_v,
        alpha, rank,
    )
