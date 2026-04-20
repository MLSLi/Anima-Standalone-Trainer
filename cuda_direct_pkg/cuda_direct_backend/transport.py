"""
GPU-GPU data movement using IPC handles and cudaMemcpyAsync over dedicated streams.
"""

import logging
import torch
from . import cuda_ipc

logger = logging.getLogger("cuda_direct.transport")


class CudaPeerTransport:
    is_shm: bool = False
    prefer_ring: bool = False

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.cuda.current_device()

        # Dedicated CUDA streams for each peer
        self.streams = {
            peer: torch.cuda.Stream()
            for peer in range(world_size)
            if peer != rank
        }
        logger.debug("rank=%d  created %d peer stream(s)  device=%d",
                     rank, len(self.streams), self.device)

        # Cache for opened IPC handles: handle_bytes -> local_ptr
        self.ipc_cache: dict[bytes, int] = {}
        self._ptr_to_handle: dict[int, bytes] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def get_or_open_ipc_handle(self, peer_rank: int, handle_bytes: bytes,
                                peer_device_id: int) -> int:
        cache_key = handle_bytes
        if cache_key in self.ipc_cache:
            self._cache_hits += 1
            ptr = self.ipc_cache[cache_key]
            logger.debug("IPC cache HIT   peer=%d  ptr=%s  (hits=%d misses=%d)",
                         peer_rank, hex(ptr), self._cache_hits, self._cache_misses)
            return ptr

        self._cache_misses += 1
        logger.debug("IPC cache MISS  peer=%d  device=%d  (hits=%d misses=%d)",
                     peer_rank, peer_device_id, self._cache_hits, self._cache_misses)

        local_ptr = cuda_ipc.open_ipc_handle(handle_bytes, peer_device_id)
        self.ipc_cache[cache_key] = local_ptr
        self._ptr_to_handle[local_ptr] = cache_key
        logger.debug("Opened IPC handle  peer=%d  device=%d  mapped_ptr=%s",
                     peer_rank, peer_device_id, hex(local_ptr))
        return local_ptr

    def invalidate_cache(self):
        """Close all cached IPC handles.

        Call before each collective when FSDP may have called storage.resize_(0),
        which invalidates previously shared pointers.
        """
        n = len(self.ipc_cache)
        if n == 0:
            return
        errors = 0
        for handle_bytes, local_ptr in self.ipc_cache.items():
            try:
                cuda_ipc.close_ipc_handle(local_ptr)
            except RuntimeError as e:
                errors += 1
                logger.warning("Failed to close IPC handle ptr=%s: %s",
                               hex(local_ptr), e)
        self.ipc_cache.clear()
        self._ptr_to_handle.clear()
        # Also clear the IPC handle creation cache (get_ipc_handle)
        # since tensors may have been reallocated at the same data_ptr
        cuda_ipc.invalidate_ipc_handle_cache()
        logger.debug("IPC cache invalidated  closed=%d  errors=%d", n, errors)

    def copy_tensor(self, size_bytes: int, dst_ptr: int, src_ptr: int, peer_rank: int):
        """Enqueue an async D2D copy on the stream dedicated to peer_rank."""
        if peer_rank not in self.streams:
            raise RuntimeError(
                f"cuda_direct_backend: copy_tensor called with peer_rank={peer_rank} "
                f"but no stream exists for that peer (rank={self.rank}, "
                f"world_size={self.world_size}). "
                f"Valid peers: {list(self.streams.keys())}"
            )
        if dst_ptr == 0 or src_ptr == 0:
            raise RuntimeError(
                f"cuda_direct_backend: copy_tensor received a NULL pointer — "
                f"src={hex(src_ptr)}  dst={hex(dst_ptr)}  size={size_bytes}  "
                f"peer={peer_rank}. This usually means an IPC handle was stale "
                f"or cudaIpcOpenMemHandle failed silently."
            )
        stream = self.streams[peer_rank]
        logger.debug("copy_tensor  peer=%d  src=%s  dst=%s  size=%.3fMB",
                     peer_rank, hex(src_ptr), hex(dst_ptr), size_bytes / 1e6)
        cuda_ipc.memcpy_async(dst_ptr, src_ptr, size_bytes, stream.cuda_stream)

    def sync_peer_stream(self, peer_rank: int):
        """CPU-block until the peer's dedicated stream has drained."""
        if peer_rank not in self.streams:
            raise RuntimeError(
                f"cuda_direct_backend: sync_peer_stream called with unknown "
                f"peer_rank={peer_rank} (rank={self.rank})"
            )
        self.streams[peer_rank].synchronize()
        logger.debug("sync_peer_stream done  peer=%d", peer_rank)

    def enable_peer_access(self, peer_device_ids: list[int]):
        """Enable CUDA P2P access to each device in the list."""
        for dev_id in peer_device_ids:
            if dev_id == self.device:
                continue
            can = cuda_ipc.can_access_peer(self.device, dev_id)
            if can:
                cuda_ipc.enable_peer_access(dev_id)
                logger.info("rank=%d  P2P enabled  local_device=%d  peer_device=%d",
                            self.rank, self.device, dev_id)
            else:
                logger.warning(
                    "rank=%d  P2P NOT available  local_device=%d  peer_device=%d — "
                    "transfers will fall back to system memory path (slower). "
                    "GPUs may be on different PCIe root complexes.",
                    self.rank, self.device, dev_id
                )

    def set_session(self, session_id: str) -> None:
        """No-op for P2P transport (session ID not needed)."""
        pass

    def publish_tensor(self, tensor: "torch.Tensor", sync) -> None:
        """Write this rank's IPC handle + sub-allocation offset to sync."""
        handle, offset = cuda_ipc.get_ipc_handle_and_offset(tensor)
        sync.write_tensor_meta(tensor.numel(), 0, tensor.device.index, handle,
                               ipc_offset=offset)

    def fetch_chunk(self, peer_rank: int, dst_tensor: "torch.Tensor",
                    src_offset_bytes: int, size_bytes: int, sync) -> None:
        """Open peer's IPC handle (cached), apply sub-allocation offset, then DMA."""
        _, _, dev_idx, handle, ipc_offset = sync.read_tensor_meta(peer_rank)
        peer_base = self.get_or_open_ipc_handle(peer_rank, handle, dev_idx)
        self.copy_tensor(size_bytes, dst_tensor.data_ptr(),
                         peer_base + ipc_offset + src_offset_bytes, peer_rank)

    def wait_for_peer(self, peer_rank: int) -> None:
        """Make the current CUDA stream wait for the last DMA from peer to finish."""
        import torch
        torch.cuda.current_stream().wait_stream(self.streams[peer_rank])

    def cleanup(self):
        self.invalidate_cache()
        logger.debug("rank=%d  transport cleanup done  cache_hits=%d  cache_misses=%d",
                     self.rank, self._cache_hits, self._cache_misses)
