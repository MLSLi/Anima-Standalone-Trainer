"""
Fused 3D RoPE — applies rotary embeddings to Q and K in one Triton kernel.

Matches Anima's actual ``_apply_rotary_pos_emb_base``:
  rope_emb: [N, 1, 1, D]  (full head_dim, pre-expanded)
  cos = cos(rope_emb.transpose(0,1)) → [1, N, 1, D]
  Rotates ALL D elements via half-dim swap.
"""
import torch
import triton
import triton.language as tl


# ============================================================================
# PyTorch reference — matches model's _apply_rotary_pos_emb_base exactly
# ============================================================================

def rope_ref_anima(q, k, rope_emb):
    """Reference: apply Anima's actual RoPE to Q and K.

    rope_emb: [N, 1, 1, D] — pre-expanded frequencies matching head_dim.
    """
    freqs = rope_emb.squeeze(1).squeeze(1)  # [N, D]
    cos_ = torch.cos(freqs).to(dtype=q.dtype)  # [N, D]
    sin_ = torch.sin(freqs).to(dtype=q.dtype)  # [N, D]

    def _rot_half(x):
        d2 = x.shape[-1] // 2
        return torch.cat([-x[..., d2:], x[..., :d2]], dim=-1)

    # Broadcast: cos/sin [N, D] apply per-token, shared across batch and heads
    q_rot = q * cos_.unsqueeze(0).unsqueeze(2) + _rot_half(q) * sin_.unsqueeze(0).unsqueeze(2)
    k_rot = k * cos_.unsqueeze(0).unsqueeze(2) + _rot_half(k) * sin_.unsqueeze(0).unsqueeze(2)
    return q_rot, k_rot


# ============================================================================
# Triton kernel
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_N': 32}, num_warps=4),
    ],
    key=['total_tokens'],
)
@triton.jit
def _fused_rope_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,           # [N, D] — squeezed from [N, 1, 1, D]
    out_q_ptr, out_k_ptr,
    total_tokens,
    N, H_nheads,                # N=sequence length, H_nheads=num_heads
    D: tl.constexpr,
    D_half: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused RoPE: half-dim rotation on ALL D elements.

    Layout: q_2d is [B*H*N, D] from q[B, N, H, D].reshape.
    Row i corresponds to: n = i // H, h = i % H.
    cos/sin are per-position N, so position index = i // H_nheads.
    """
    pid = tl.program_id(0)
    tok_start = pid * BLOCK_N
    tok_offs = tok_start + tl.arange(0, BLOCK_N)
    tok_mask = tok_offs < total_tokens

    # Position index: q_2d row i has token position n = i // H_nheads
    pos_offs = (tok_offs // H_nheads % N)[:, None] * D
    row_off = tok_offs[:, None] * D
    d_first = tl.arange(0, D_half)
    d_second = D_half + tl.arange(0, D_half)

    cos_first = tl.load(cos_ptr + pos_offs + d_first[None, :],
                        mask=tok_mask[:, None], other=0.0).to(tl.float32)
    cos_second = tl.load(cos_ptr + pos_offs + d_second[None, :],
                         mask=tok_mask[:, None], other=0.0).to(tl.float32)
    sin_first = tl.load(sin_ptr + pos_offs + d_first[None, :],
                        mask=tok_mask[:, None], other=0.0).to(tl.float32)
    sin_second = tl.load(sin_ptr + pos_offs + d_second[None, :],
                         mask=tok_mask[:, None], other=0.0).to(tl.float32)

    q_first = tl.load(q_ptr + row_off + d_first[None, :],
                      mask=tok_mask[:, None], other=0.0).to(tl.float32)
    q_second = tl.load(q_ptr + row_off + d_second[None, :],
                       mask=tok_mask[:, None], other=0.0).to(tl.float32)
    k_first = tl.load(k_ptr + row_off + d_first[None, :],
                      mask=tok_mask[:, None], other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + row_off + d_second[None, :],
                       mask=tok_mask[:, None], other=0.0).to(tl.float32)

    q_first_rot = q_first * cos_first - q_second * sin_first
    q_second_rot = q_second * cos_second + q_first * sin_second
    k_first_rot = k_first * cos_first - k_second * sin_first
    k_second_rot = k_second * cos_second + k_first * sin_second

    tl.store(out_q_ptr + row_off + d_first[None, :],
             q_first_rot.to(tl.bfloat16), mask=tok_mask[:, None])
    tl.store(out_q_ptr + row_off + d_second[None, :],
             q_second_rot.to(tl.bfloat16), mask=tok_mask[:, None])
    tl.store(out_k_ptr + row_off + d_first[None, :],
             k_first_rot.to(tl.bfloat16), mask=tok_mask[:, None])
    tl.store(out_k_ptr + row_off + d_second[None, :],
             k_second_rot.to(tl.bfloat16), mask=tok_mask[:, None])


# ============================================================================
# Public API
# ============================================================================

def fused(q, k, cos, sin=None):
    """Fused Anima RoPE for Q and K.

    q,k:   [B, N, H, D]  bf16  (model's compute_qkv output format)
    cos:   [N, D] or [N, 1, 1, D] — rope_emb or pre-computed cos
    sin:   optional — if None, cos is rope_emb and sin is auto-computed
    """
    if sin is None:
        freqs = cos.squeeze(1).squeeze(1)  # [N, 1, 1, D] → [N, D]
        cos_val = torch.cos(freqs).to(dtype=q.dtype)
        sin_val = torch.sin(freqs).to(dtype=q.dtype)
    else:
        cos_val = cos; sin_val = sin

    if cos_val.ndim == 4:
        cos_val = cos_val.squeeze(1).squeeze(1)
    if sin_val.ndim == 4:
        sin_val = sin_val.squeeze(1).squeeze(1)

    B, N_q, H, D = q.shape
    total = B * H * N_q
    q_2d = q.reshape(total, D).contiguous()
    k_2d = k.reshape(total, D).contiguous()
    out_q = torch.empty_like(q_2d); out_k = torch.empty_like(k_2d)

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_N']),)
    _fused_rope_kernel[grid](
        q_2d, k_2d, cos_val.contiguous(), sin_val.contiguous(), out_q, out_k,
        total, N_q, H, D=D, D_half=D//2,
    )
    return out_q.reshape_as(q), out_k.reshape_as(k)
