"""
HybridTransport: auto-selects P2P or SHM transport based on GPU topology.

If ALL GPU pairs in the process group have P2P access → CudaPeerTransport (fast path).
If ANY pair lacks P2P (typical for consumer GPUs) → ShmTransport (host-staged path).

Both transports expose the same interface:
  publish_tensor(tensor, sync)
  fetch_chunk(peer, dst, src_offset_bytes, size_bytes, sync)
  wait_for_peer(peer)
  invalidate_cache()
  sync_peer_stream(peer)
  enable_peer_access(devices)
  set_session(session_id)   [ShmTransport only, no-op for P2P]
  is_shm: bool
  streams: dict[int, Stream]
"""

import logging

import torch

from . import cuda_ipc
from .transport import CudaPeerTransport
from .shm_transport import ShmTransport

logger = logging.getLogger("cuda_direct.hybrid_transport")


def _all_pairs_have_p2p(device_count: int) -> bool:
    """Return True iff every ordered pair of CUDA devices can do P2P."""
    for i in range(device_count):
        for j in range(device_count):
            if i == j:
                continue
            try:
                if not cuda_ipc.can_access_peer(i, j):
                    logger.info("No P2P between device %d and device %d — using SHM transport", i, j)
                    return False
            except RuntimeError:
                logger.info("P2P check failed for device %d↔%d — using SHM transport", i, j)
                return False
    return True


def _ipc_sharing_works() -> bool:
    """Return True iff the active CUDA allocator supports IPC handle sharing.

    cudaMallocAsync (PyTorch's stream-ordered allocator, default in newer
    torch+CUDA builds) does NOT support cudaIpcGetMemHandle / _share_cuda_().
    Probing here lets HybridTransport fall back to SHM automatically instead
    of crashing mid-run with "cudaMallocAsync does not yet support shareIpcHandle".
    """
    try:
        t = torch.empty(1, dtype=torch.float32, device="cuda")
        t.untyped_storage()._share_cuda_()
        del t
        return True
    except Exception as e:
        logger.info(
            "IPC sharing probe failed (%s) — CUDA allocator does not support "
            "cudaIpcGetMemHandle; falling back to SHM transport", e
        )
        return False


class HybridTransport:
    """
    Wrapper that delegates to either CudaPeerTransport or ShmTransport.

    Instantiate via HybridTransport.create() which probes P2P availability.
    All attribute accesses are forwarded to the inner transport so callers
    do not need to know which backend is active.
    """

    def __init__(self, inner: CudaPeerTransport | ShmTransport):
        self._inner = inner

    @classmethod
    def create(cls, rank: int, world_size: int,
               peer_device_ids: list[int] | None = None) -> "HybridTransport":
        """
        Probe GPU topology and return a HybridTransport wrapping the best transport.

        Selection order:
          1. Hardware P2P check — cudaDeviceCanAccessPeer for all pairs
          2. Allocator IPC check — _share_cuda_() probe to detect cudaMallocAsync
          Both must pass for CudaPeerTransport; otherwise ShmTransport is used.

        Called once per process during ProcessGroupCudaDirect.__init__.
        """
        device_count = torch.cuda.device_count()
        has_p2p = _all_pairs_have_p2p(device_count)
        ipc_ok   = _ipc_sharing_works() if has_p2p else False

        if has_p2p and ipc_ok:
            inner = CudaPeerTransport(rank, world_size)
            logger.info(
                "HybridTransport: P2P + IPC OK — using CudaPeerTransport  "
                "rank=%d  world=%d", rank, world_size
            )
        else:
            inner = ShmTransport(rank, world_size)
            reason = "no P2P" if not has_p2p else "allocator IPC unsupported (cudaMallocAsync?)"
            logger.info(
                "HybridTransport: %s — using ShmTransport (host-staged)  "
                "rank=%d  world=%d", reason, rank, world_size
            )
        return cls(inner)

    # ---------------------------------------------------------------
    #  Transparent delegation
    # ---------------------------------------------------------------

    def __getattr__(self, name: str):
        # Called only when normal attribute lookup fails, i.e. attr is on inner
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value):
        if name == "_inner":
            super().__setattr__(name, value)
        else:
            setattr(self._inner, name, value)
