"""
Anima LoRA training kernels.

Each module exposes a ``fused()`` function (the Triton-accelerated path)
and a ``reference()`` function (pure PyTorch, for validation).
"""

from triton_ops.lora import (  # noqa: F401
    base_lora,
    qkv_lora,
    ffn_silu,
    adaln_norm,
    output_residual,
)
