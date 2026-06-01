"""
Triton-accelerated operators for Anima LoRA training.

Kernels:
    base_lora   — Fused Base Linear + LoRA
    qkv_lora    — Batched Q/K/V LoRA for self-attention
    ffn_silu    — Fused FFN W1 with SiLU activation
    adaln_norm  — Fused AdaLN modulation + LayerNorm
    output_residual — Fused Output Projection + Residual + Gate

All kernels target BF16 training on NVIDIA Ada Lovelace GPUs (SM 8.9+).
Each kernel follows a "split design" pattern: cuDNN handles compute-bound GEMM,
Triton fuses memory-bound LoRA paths and element-wise operations.

Usage::

    from triton_ops.lora import base_lora, qkv_lora

    # Replace: y = base_linear(x) + lora_scale * lora_B(lora_A(x))
    y = base_lora.fused(x, base_w, base_b, lora_a, lora_b, alpha=16, rank=32)
"""

__version__ = "1.0.0"
__author__ = "Anima Triton Ops"

from triton_ops.lora import (  # noqa: F401
    base_lora,
    qkv_lora,
    ffn_silu,
    adaln_norm,
    output_residual,
)
