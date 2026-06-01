"""
Fused RoPE + Attention benchmark — measures end-to-end speedup.

Strategy: our fused_rope outputs to memory, then SDPA runs.
The FULL fusion (RoPE inside FlashAttention) requires rewriting the
entire attention kernel with backward pass — beyond current scope.

This module benchmarks: fused_rope + SDPA vs PyTorch RoPE + SDPA.
"""
import torch
import triton
import triton.language as tl
import math
from triton_ops.lora.fused_rope_3d import fused as fused_rope, rope_ref_anima


def attn_rope_ref(q, k, v, rope_emb, sm_scale=None):
    """PyTorch reference: RoPE on Q and K, then SDPA."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    qr, kr = rope_ref_anima(q, k, rope_emb)
    return torch.nn.functional.scaled_dot_product_attention(qr, kr, v, scale=sm_scale)


def attn_rope_triton(q, k, v, rope_emb, sm_scale=None):
    """Triton-accelerated: fused RoPE on Q+K, then cuDNN SDPA."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    qr, kr = fused_rope(q, k, rope_emb[:q.shape[1]])
    return torch.nn.functional.scaled_dot_product_attention(qr, kr, v, scale=sm_scale)


# ============================================================================
# Attempt at full fusion: RoPE + FlashAttention in one kernel
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=1),
    ],
    key=['N', 'D'],
)
@triton.jit
def _flash_attn_kernel(
    q_ptr, k_ptr, v_ptr,
    o_ptr,
    N, H, D: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FlashAttention tile (no RoPE).  Validates attention-only kernel."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b_idx = pid_bh // H; h_idx = pid_bh % H
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < N

    row_stride = N * D
    q_base = q_ptr + b_idx * H * row_stride + h_idx * row_stride
    k_base = k_ptr + b_idx * H * row_stride + h_idx * row_stride
    v_base = v_ptr + b_idx * H * row_stride + h_idx * row_stride
    o_base = o_ptr + b_idx * H * row_stride + h_idx * row_stride

    q = tl.load(q_base + m_offs[:, None] * D + tl.arange(0, D)[None, :],
                mask=m_mask[:, None], other=0.0).to(tl.float32)

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N); n_mask = n_offs < N
        k = tl.load(k_base + n_offs[:, None] * D + tl.arange(0, D)[None, :],
                     mask=n_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_base + n_offs[:, None] * D + tl.arange(0, D)[None, :],
                     mask=n_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc_o = alpha[:, None] * acc_o + tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new

    acc_o = acc_o / l_i[:, None]
    tl.store(o_base + m_offs[:, None] * D + tl.arange(0, D)[None, :],
             acc_o.to(tl.bfloat16), mask=m_mask[:, None])


def flash_attn_triton(q, k, v, sm_scale=None):
    """Triton FlashAttention (attention-only, no RoPE).  For validation."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    grid = (B * H, triton.cdiv(N, 64))
    _flash_attn_kernel[grid](q, k, v, o, N, H, D=D, sm_scale=sm_scale)
    return o
