"""Shared configuration and GPU detection utilities."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUInfo:
    """Immutable GPU capability snapshot."""

    name: str
    vram_gb: float
    compute_capability: tuple[int, int]
    sm_count: int
    max_shared_memory: int  # bytes per SM
    max_threads_per_sm: int

    @property
    def is_ada_lovelace(self) -> bool:
        """Ada Lovelace (SM 8.9) — RTX 40 series."""
        return self.compute_capability >= (8, 9)

    @property
    def smem_kb(self) -> int:
        """Usable shared memory per SM in KB."""
        return self.max_shared_memory // 1024


def detect_gpu(device_index: int = 0) -> GPUInfo:
    """Return GPU capabilities for the given device index.

    Raises:
        RuntimeError: If CUDA is unavailable or no GPU at index.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    props = torch.cuda.get_device_properties(device_index)
    return GPUInfo(
        name=props.name,
        vram_gb=props.total_memory / (1024**3),
        compute_capability=(props.major, props.minor),
        sm_count=props.multi_processor_count,
        max_shared_memory=props.shared_memory_per_block_optin,
        max_threads_per_sm=props.max_threads_per_multi_processor,
    )


def print_gpu_info(gpu: GPUInfo) -> None:
    """Pretty-print GPU configuration."""
    print("=" * 64)
    print(f"Device:     {gpu.name}")
    print(f"VRAM:       {gpu.vram_gb:.1f} GB")
    print(f"SM:         {gpu.compute_capability[0]}.{gpu.compute_capability[1]}  "
          f"({gpu.sm_count} SMs, {gpu.max_threads_per_sm} threads/SM)")
    print(f"Shared mem: {gpu.smem_kb} KB/SM")
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.version.cuda}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Autotune defaults — conservative for laptop GPUs (8 GB, ~100 KB smem)
# ---------------------------------------------------------------------------
AUTOTUNE_WARMUP_MS = 50
AUTOTUNE_REP_MS = 200
MAX_GRID_SIZE = 256
