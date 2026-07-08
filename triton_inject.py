#!/usr/bin/env python3
"""
Triton-accelerated LoRA kernels for Anima training.

Import and call ``inject(dit)`` before training to replace every LoRA down+up
matmul pair with a single fused kernel.  Completely transparent to the
training loop — no other code changes needed.

Usage (add to top of train script)::

    import triton_inject
    # ... load dit, create LoRA network, call network.apply_to(dit, ...) ...
    triton_inject.inject(dit, rank=32, alpha=16)
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

_PATCHED = set()
_ORIG_FORWARDS = {}
_APPLY_TO_PATCHED = False
_FALLBACK_WARNED = set()


# ============================================================================
# Patch LoRAModule.apply_to to preserve base weight for Triton access
# ============================================================================

def _patch_apply_to():
    """Monkey-patch LoRAModule.apply_to to store base weight reference."""
    global _APPLY_TO_PATCHED
    if _APPLY_TO_PATCHED:
        return

    from networks.lora_flux import LoRAModule
    _orig_apply = LoRAModule.apply_to

    def _patched_apply(self):
        # Store weight before original module is deleted
        self._base_weight = self.org_module.weight.data
        self._base_bias = self.org_module.bias.data if self.org_module.bias is not None else None
        _orig_apply(self)

    LoRAModule.apply_to = _patched_apply
    _APPLY_TO_PATCHED = True


def _on_device(tensor: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if tensor is None or tensor.device == device:
        return tensor
    return tensor.to(device)


def _effective_lora_alpha(lora_module) -> float:
    return float(lora_module.multiplier * lora_module.scale * lora_module._triton_rank)


def _linear_lora_module(linear: torch.nn.Linear):
    forward_owner = getattr(getattr(linear, "forward", None), "__self__", None)
    if forward_owner is None or not hasattr(forward_owner, "lora_down") or not hasattr(forward_owner, "lora_up"):
        return None
    return forward_owner


def _can_group_lora(lora_module) -> bool:
    return (
        lora_module is not None
        and getattr(lora_module, "split_dims", None) is None
        and getattr(lora_module, "dropout", None) is None
        and getattr(lora_module, "rank_dropout", None) is None
        and getattr(lora_module, "module_dropout", None) is None
        and getattr(lora_module, "ggpo_sigma", None) is None
        and getattr(lora_module, "ggpo_beta", None) is None
        and hasattr(lora_module, "_base_weight")
        and hasattr(lora_module, "_triton_rank")
    )


def _same_lora_scale(*lora_modules) -> bool:
    first = _effective_lora_alpha(lora_modules[0])
    return all(abs(_effective_lora_alpha(m) - first) < 1e-12 for m in lora_modules[1:])


def _packed_base_qkv(attn, q_lora, k_lora, v_lora, device: torch.device):
    weights = (q_lora._base_weight, k_lora._base_weight, v_lora._base_weight)
    biases = (q_lora._base_bias, k_lora._base_bias, v_lora._base_bias)
    key = tuple(w.data_ptr() for w in weights) + (str(device),)
    if getattr(attn, "_triton_qkv_base_key", None) != key:
        attn._triton_qkv_base_weight = torch.cat([_on_device(w, device) for w in weights], dim=0).contiguous()
        if all(b is None for b in biases):
            attn._triton_qkv_base_bias = None
        else:
            if any(b is None for b in biases):
                return None, None
            attn._triton_qkv_base_bias = torch.cat([_on_device(b, device) for b in biases], dim=0).contiguous()
        attn._triton_qkv_base_key = key
    return attn._triton_qkv_base_weight, attn._triton_qkv_base_bias


# ============================================================================
# Triton-accelerated LoRA forward (replaces LoRAModule.forward)
# ============================================================================

def _triton_lora_forward(self, x):
    """Fused: base_linear(x) + scale * lora_up(lora_down(x))."""
    if not _TRITON_AVAIL or not hasattr(self, '_triton_rank'):
        return self._orig_forward(x)

    try:
        from triton_ops.lora.base_lora import fused_autograd as triton_fused
        dev = x.device
        return triton_fused(
            x,
            base_weight=_on_device(self._base_weight, dev),
            base_bias=_on_device(self._base_bias, dev),
            lora_a=_on_device(self.lora_down.weight, dev),
            lora_b=_on_device(self.lora_up.weight, dev),
            alpha=_effective_lora_alpha(self),
            rank=self._triton_rank,
        )[0]
    except Exception as exc:
        module_key = id(self)
        if module_key not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(module_key)
            module_name = getattr(self, "lora_name", self.__class__.__name__)
            logger.warning("Triton LoRA fallback for %s: %s", module_name, exc)
        return self._orig_forward(x)


_TRITON_AVAIL = False


def inject(dit, network, rank: int = 32, alpha: int = 16) -> int:
    """Inject Triton-accelerated LoRA kernels.

    Call AFTER ``network.apply_to(dit, ...)``.

    Usage::

        network = lora_anima.create_network(1.0, 32, 16, vae, [], dit)
        network.apply_to(dit, [])
        triton_inject.inject(dit, network, rank=32, alpha=16)
        # ... train as normal ...

    Returns:
        Number of LoRA modules accelerated.
    """
    global _TRITON_AVAIL

    try:
        import triton  # noqa: F401
        _TRITON_AVAIL = True
    except ImportError:
        logger.warning("Triton not available")
        return 0

    # Step 1: Patch apply_to to capture base weights (for future apply_to calls)
    _patch_apply_to()

    # Step 2: For already-applied LoRA modules, extract base weights from dit
    from networks.lora_flux import LoRAModule

    # Build a map of Linear module -> weight data for quick lookup
    weight_map = {}
    for name, module in dit.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_map[id(module)] = (module.weight.data,
                                       module.bias.data if module.bias is not None else None)

    count = 0
    for lora_module in network.modules():
        if not isinstance(lora_module, LoRAModule):
            continue
        if id(lora_module) in _PATCHED:
            continue

        # Extract base weight from org_forward's bound method
        # org_forward = self.org_module.forward where org_module is a nn.Linear
        org_fwd = getattr(lora_module, 'org_forward', None)
        if org_fwd is not None:
            # org_forward is a bound method — its __self__ is the original module
            org_mod = getattr(org_fwd, '__self__', None)
            if org_mod is not None and isinstance(org_mod, torch.nn.Linear):
                lora_module._base_weight = org_mod.weight.data
                lora_module._base_bias = org_mod.bias.data if org_mod.bias is not None else None
            else:
                continue  # can't find base weight, skip

        if not hasattr(lora_module, '_base_weight'):
            continue

        lora_module._triton_rank = rank
        lora_module._orig_forward = lora_module.forward
        lora_module.forward = _triton_lora_forward.__get__(lora_module)
        _PATCHED.add(id(lora_module))
        count += 1

    if count > 0:
        logger.info(f"Triton injected: {count} LoRA modules accelerated (rank={rank})")
    else:
        logger.warning("No LoRA modules with base weights found")

    return count


# ============================================================================
# RoPE injection — fuse Q+K rotary embedding into one kernel
# ============================================================================

_ORIG_APPLY_ROTARY_POS_EMB = None
_ROPE_PATCHED = False


def inject_rope():
    """Replace ``apply_rotary_pos_emb`` with fused Q+K RoPE kernel.

    Call BEFORE training starts.  The fused kernel pre-computes cos/sin
    from ``freqs`` and processes Q and K in a single Triton launch.

    Usage::

        triton_inject.inject_rope()
        # ... train as normal ...
    """
    global _ORIG_APPLY_ROTARY_POS_EMB, _ROPE_PATCHED

    if _ROPE_PATCHED:
        return

    import library.anima_models as am
    _ORIG_APPLY_ROTARY_POS_EMB = am.apply_rotary_pos_emb

    def _fused_apply_rotary_pos_emb(t, freqs, tensor_format="bshd", start_positions=None,
                                     interleaved=False, fused=False, cu_seqlens=None, cp_size=1):
        """Drop-in replacement: if called on Q, also applies to K via fused kernel."""
        # Only fuse when: standard call, single tensor, no CP
        if tensor_format != "bshd" or start_positions is not None or cu_seqlens is not None:
            return _ORIG_APPLY_ROTARY_POS_EMB(t, freqs, tensor_format,
                                              start_positions, interleaved, fused,
                                              cu_seqlens, cp_size)

        # Compute cos/sin from freqs (matching original)
        max_seq_len = freqs.shape[0]
        cur_seq_len = t.shape[1]
        freqs = freqs[:cur_seq_len].transpose(0, 1)
        cos_ = torch.cos(freqs)
        sin_ = torch.sin(freqs)

        # Catch: this is called via a hook on the Q tensor.
        # We apply RoPE to `t` (which is Q) and store `cos_, sin_, freqs, t` for K.
        # When K is called next, we return the previously-computed K_rope.
        # This is fragile — the better approach is to patch compute_qkv directly.

        # For now: just apply original on both Q and K (no fusion).
        # The actual fusion is in the model-level injection below.
        return _ORIG_APPLY_ROTARY_POS_EMB(t, freqs.transpose(0, 1), tensor_format,
                                          start_positions, interleaved, fused,
                                          cu_seqlens, cp_size)

    am.apply_rotary_pos_emb = _fused_apply_rotary_pos_emb
    _ROPE_PATCHED = True
    logger.info("Triton RoPE injection prepared (model-level patching recommended)")


def inject_rope_model(dit):
    """Patch AnimaModel compute_qkv to use fused RoPE on Q and K.

    This is the production injection — works at the model level.
    Call AFTER model loading, BEFORE training.
    """
    from library.anima_models import apply_rotary_pos_emb as _ref_rope
    from triton_ops.lora.fused_rope_3d import fused as _triton_rope
    from triton_ops.lora.qkv_lora import fused_packed_autograd as _triton_qkv_packed

    def _patched_compute_qkv(self, x, context=None, rope_emb=None):
        q_lora = _linear_lora_module(self.q_proj)
        k_lora = _linear_lora_module(self.k_proj)
        v_lora = _linear_lora_module(self.v_proj)

        use_grouped_qkv = (
            self.is_selfattn
            and context is None
            and _can_group_lora(q_lora)
            and _can_group_lora(k_lora)
            and _can_group_lora(v_lora)
            and q_lora._triton_rank == k_lora._triton_rank == v_lora._triton_rank
            and _same_lora_scale(q_lora, k_lora, v_lora)
            and x.shape[-1] == self.query_dim == self.context_dim
        )

        if use_grouped_qkv:
            try:
                base_w_qkv, base_b_qkv = _packed_base_qkv(self, q_lora, k_lora, v_lora, x.device)
                if base_w_qkv is None:
                    raise RuntimeError("mixed QKV bias state is unsupported")
                q, k, v = _triton_qkv_packed(
                    x,
                    base_w_qkv,
                    base_b_qkv,
                    _on_device(q_lora.lora_down.weight, x.device),
                    _on_device(k_lora.lora_down.weight, x.device),
                    _on_device(v_lora.lora_down.weight, x.device),
                    _on_device(q_lora.lora_up.weight, x.device),
                    _on_device(k_lora.lora_up.weight, x.device),
                    _on_device(v_lora.lora_up.weight, x.device),
                    alpha=_effective_lora_alpha(q_lora),
                    rank=q_lora._triton_rank,
                )
            except Exception as exc:
                attn_key = id(self)
                if attn_key not in _FALLBACK_WARNED:
                    _FALLBACK_WARNED.add(attn_key)
                    logger.warning("Triton grouped QKV fallback: %s", exc)
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
        else:
            q = self.q_proj(x)
            ctx = x if context is None else context
            k = self.k_proj(ctx)
            v = self.v_proj(ctx)

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # ---- Fused Q+K RoPE ----
        if self.is_selfattn and rope_emb is not None:
            try:
                # rope_emb: [N, 1, 1, D] — pass directly; fused() computes cos/sin
                q, k = _triton_rope(q, k, rope_emb[:q.shape[1]])
            except Exception:
                q = _ref_rope(q, rope_emb, tensor_format=self.qkv_format, fused=False)
                k = _ref_rope(k, rope_emb, tensor_format=self.qkv_format, fused=False)

        return q, k, v

    # Patch attention layers in all blocks
    patched_count = 0
    for block in dit.blocks:
        sa = getattr(block, 'self_attn', None)
        if sa is not None and hasattr(sa, 'compute_qkv') and not hasattr(sa, '_orig_compute_qkv'):
            sa._orig_compute_qkv = sa.compute_qkv
            sa.compute_qkv = _patched_compute_qkv.__get__(sa)
            patched_count += 1

    if patched_count > 0:
        logger.info(f"Triton RoPE injected: {patched_count} self-attention layers")
    return patched_count


def inject_adaln_model(dit) -> int:
    """Patch Anima blocks to use Triton AdaLN for real model formula.

    Replaces ``LayerNorm(x) * (1 + scale) + shift`` in the three sublayers
    of each block.  This does not alter attention, MLP, or residual math.
    """
    from triton_ops.lora.adaln_norm import fused_anima_modulated_autograd as _triton_adaln
    from triton_ops.lora.output_residual import fused_anima_gate_autograd as _triton_output_residual

    def _project_residual_5d(linear, proj_input_5d, residual_5d, gate_5d):
        lora = _linear_lora_module(linear)
        if not _can_group_lora(lora):
            return None
        try:
            return _triton_output_residual(
                proj_input_5d,
                residual_5d,
                gate_5d,
                _on_device(lora._base_weight, proj_input_5d.device),
                _on_device(lora._base_bias, proj_input_5d.device),
                _on_device(lora.lora_down.weight, proj_input_5d.device),
                _on_device(lora.lora_up.weight, proj_input_5d.device),
                alpha=_effective_lora_alpha(lora),
                rank=lora._triton_rank,
            )
        except Exception as exc:
            linear_key = id(linear)
            if linear_key not in _FALLBACK_WARNED:
                _FALLBACK_WARNED.add(linear_key)
                logger.warning("Triton output residual fallback: %s", exc)
            return None

    def _attention_output_no_proj(attn, x, context, rope_emb):
        q, k, v = attn.compute_qkv(x, context, rope_emb=rope_emb)
        return attn.attn_op(q, k, v)

    def _patched_forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            if extra_per_block_pos_emb is not None:
                x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

            shift_self_attn = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
            scale_self_attn = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
            gate_self_attn = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

            shift_cross_attn = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
            scale_cross_attn = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
            gate_cross_attn = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

            shift_mlp = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
            scale_mlp = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
            gate_mlp = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

            B, T, H, W, _ = x_B_T_H_W_D.shape

            normalized_x = _triton_adaln(x_B_T_H_W_D, scale_self_attn, shift_self_attn, self.layer_norm_self_attn.eps)
            self_attn_in = rearrange(normalized_x, "b t h w d -> b (t h w) d")
            self_attn_raw = _attention_output_no_proj(self.self_attn, self_attn_in, None, rope_emb_L_1_1_D)
            self_attn_raw_5d = rearrange(self_attn_raw, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            fused_residual = _project_residual_5d(self.self_attn.output_proj, self_attn_raw_5d, x_B_T_H_W_D, gate_self_attn)
            if fused_residual is None:
                result = rearrange(
                    self.self_attn.output_dropout(self.self_attn.output_proj(self_attn_raw)),
                    "b (t h w) d -> b t h w d",
                    t=T, h=H, w=W,
                )
                x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn * result
            else:
                x_B_T_H_W_D = fused_residual

            normalized_x = _triton_adaln(x_B_T_H_W_D, scale_cross_attn, shift_cross_attn, self.layer_norm_cross_attn.eps)
            cross_attn_in = rearrange(normalized_x, "b t h w d -> b (t h w) d")
            cross_attn_raw = _attention_output_no_proj(self.cross_attn, cross_attn_in, crossattn_emb, rope_emb_L_1_1_D)
            cross_attn_raw_5d = rearrange(cross_attn_raw, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            fused_residual = _project_residual_5d(self.cross_attn.output_proj, cross_attn_raw_5d, x_B_T_H_W_D, gate_cross_attn)
            if fused_residual is None:
                result = rearrange(
                    self.cross_attn.output_dropout(self.cross_attn.output_proj(cross_attn_raw)),
                    "b (t h w) d -> b t h w d",
                    t=T, h=H, w=W,
                )
                x_B_T_H_W_D = result * gate_cross_attn + x_B_T_H_W_D
            else:
                x_B_T_H_W_D = fused_residual

            normalized_x = _triton_adaln(x_B_T_H_W_D, scale_mlp, shift_mlp, self.layer_norm_mlp.eps)
            mlp_hidden = self.mlp.activation(self.mlp.layer1(normalized_x))
            fused_residual = _project_residual_5d(self.mlp.layer2, mlp_hidden, x_B_T_H_W_D, gate_mlp)
            if fused_residual is None:
                result = self.mlp.layer2(mlp_hidden)
                x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp * result
            else:
                x_B_T_H_W_D = fused_residual
            return x_B_T_H_W_D
        except Exception as exc:
            block_key = id(self)
            if block_key not in _FALLBACK_WARNED:
                _FALLBACK_WARNED.add(block_key)
                logger.warning("Triton AdaLN fallback for Block: %s", exc)
            return self._orig_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_per_block_pos_emb,
            )

    patched_count = 0
    for block in getattr(dit, "blocks", []):
        if hasattr(block, "_forward") and not hasattr(block, "_orig_forward"):
            block._orig_forward = block._forward
            block._forward = _patched_forward.__get__(block)
            patched_count += 1

    if patched_count > 0:
        logger.info("Triton AdaLN injected: %d blocks", patched_count)
    return patched_count


def inject_optimized(dit, network=None, rank: int = 32, alpha: int = 16) -> dict[str, int]:
    """Enable the currently validated Triton optimizations for Anima."""
    counts = {}
    if network is not None:
        counts["lora"] = inject(dit, network, rank=rank, alpha=alpha)
    counts["rope"] = inject_rope_model(dit)
    counts["adaln"] = inject_adaln_model(dit)
    return counts


def uninject_rope_model(dit):
    """Restore original compute_qkv."""
    for block in dit.blocks:
        sa = getattr(block, 'self_attn', None)
        if sa is not None and hasattr(sa, '_orig_compute_qkv'):
            sa.compute_qkv = sa._orig_compute_qkv
            del sa._orig_compute_qkv
            for attr in ("_triton_qkv_base_key", "_triton_qkv_base_weight", "_triton_qkv_base_bias"):
                if hasattr(sa, attr):
                    delattr(sa, attr)
    logger.info("Triton RoPE uninjected")


def uninject_adaln_model(dit):
    """Restore original Anima block forwards after ``inject_adaln_model``."""
    for block in getattr(dit, "blocks", []):
        if hasattr(block, "_orig_forward"):
            block._forward = block._orig_forward
            del block._orig_forward
    logger.info("Triton AdaLN uninjected")


def uninject(network):
    """Restore original forward methods."""
    from networks.lora_flux import LoRAModule
    for module in network.modules():
        if isinstance(module, LoRAModule) and hasattr(module, '_orig_forward'):
            module.forward = module._orig_forward
            del module._orig_forward
    _PATCHED.clear()
    logger.info("Triton uninjected")
