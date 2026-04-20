"""
Collective operations: AllReduce, Broadcast, AllGather, ReduceScatter, Barrier.

Two transport modes are supported:

  P2P (CudaPeerTransport, is_shm=False):
    Workstation/datacenter GPUs with CUDA P2P access.
    Direct GPU-GPU DMA via IPC handles — live pointer access.
    world_size=2  → direct (all-to-all) algorithms
    world_size>=3 → ring algorithms

  SHM (ShmTransport, is_shm=True):
    Consumer GPUs (RTX 3000/4000/5000) without P2P.
    Data staged through OS shared memory + pinned host buffers.
    Always uses "publish-once, pull-all" algorithms (ring requires live
    pointer access which SHM snapshots cannot provide).
"""

import logging
import time
import torch
import torch.distributed
from . import cuda_ipc
from .sync import SharedMemorySync

logger = logging.getLogger("cuda_direct.collectives")


class CollectivesImpl:
    def __init__(self, rank: int, world_size: int, transport,
                 sync: SharedMemorySync, ring_order: list[int] | None = None):
        self.rank = rank
        self.world_size = world_size
        self.transport = transport
        self.sync = sync

        # Persistent scratch buffer cache: (collective, peer) -> Tensor
        self._buf_cache: dict[tuple[str, int], torch.Tensor] = {}

        # Ring topology (P2P ring algorithms)
        if ring_order is not None and len(ring_order) == world_size:
            self.ring_order = ring_order
        else:
            self.ring_order = list(range(world_size))

        self._ring_pos = self.ring_order.index(self.rank)
        self._ring_prev = self.ring_order[(self._ring_pos - 1) % world_size]
        self._ring_next = self.ring_order[(self._ring_pos + 1) % world_size]

        # SHM transport uses its own publish-once/pull-all algorithms.
        # P2P uses ring (world_size>=3) or direct (world_size=2).
        self._is_shm = getattr(transport, 'is_shm', False)
        self._use_ring = (not self._is_shm) and (world_size >= 3)

        if self._is_shm:
            algo = "shm"
        elif self._use_ring:
            algo = "ring"
        else:
            algo = "direct"
        logger.info("CollectivesImpl ready  rank=%d  world=%d  algo=%s  ring_order=%s",
                    rank, world_size, algo, self.ring_order)

    def _get_peer_buffer(self, collective: str, peer: int, like: torch.Tensor) -> torch.Tensor:
        """Return a cached GPU scratch buffer matching `like`, or allocate a new one."""
        key = (collective, peer)
        buf = self._buf_cache.get(key)
        if buf is not None and buf.shape == like.shape and buf.dtype == like.dtype and buf.device == like.device:
            return buf
        buf = torch.empty_like(like)
        self._buf_cache[key] = buf
        return buf

    def _exchange_handles(self, tensor: torch.Tensor):
        """Write IPC handle + offset for tensor to sync (P2P direct algorithms only)."""
        handle, offset = cuda_ipc.get_ipc_handle_and_offset(tensor)
        self.sync.write_tensor_meta(tensor.numel(), 0, tensor.device.index, handle,
                                    ipc_offset=offset)

    # ---------------------------------------------------------------
    #  ReduceOp helpers
    # ---------------------------------------------------------------

    _BITWISE_OPS = frozenset([
        torch.distributed.ReduceOp.BAND,
        torch.distributed.ReduceOp.BOR,
        torch.distributed.ReduceOp.BXOR,
    ])

    _INTEGER_DTYPES = frozenset([
        # Standard integers
        torch.bool,
        torch.uint8, torch.uint16, torch.uint32, torch.uint64,
        torch.int8, torch.int16, torch.int32, torch.int64,
        # Sub-byte integers (PyTorch 2.3+)
        torch.uint1, torch.uint2, torch.uint3, torch.uint4,
        torch.uint5, torch.uint6, torch.uint7,
        torch.int1, torch.int2, torch.int3, torch.int4,
        torch.int5, torch.int6, torch.int7,
    ])

    _FLOATING_DTYPES = frozenset([
        # Standard floats
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
        # Float8 variants (PyTorch 2.1+)
        torch.float8_e4m3fn, torch.float8_e4m3fnuz,
        torch.float8_e5m2, torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
        # Complex floats
        torch.complex32, torch.complex64, torch.complex128,
    ])

    def _normalize_op(self, op):
        if op is None:
            return torch.distributed.ReduceOp.SUM
        return op

    def _validate_op_dtype(self, tensor: torch.Tensor, op):
        dtype = tensor.dtype
        op_name = str(op)

        # 1. AVG is only supported for floating-point tensors
        if op == torch.distributed.ReduceOp.AVG and dtype not in self._FLOATING_DTYPES:
            raise TypeError(
                f"cuda_direct_backend: {op_name} requires a floating-point tensor, "
                f"got dtype={dtype} (rank={self.rank})."
            )

        # 2. Bitwise ops require integer or bool tensors
        # Use any(op == x for x in ...) instead of `op in ...` — see _apply_op comment
        if any(op == x for x in self._BITWISE_OPS) and dtype not in self._INTEGER_DTYPES:
            raise TypeError(
                f"cuda_direct_backend: {op_name} requires an integer or bool tensor, "
                f"got dtype={dtype} (rank={self.rank})."
            )

        # 3. Numeric SUM/PRODUCT on bool is non-standard (acts as logical OR/AND)
        if (op == torch.distributed.ReduceOp.SUM or op == torch.distributed.ReduceOp.PRODUCT) and dtype == torch.bool:
            raise TypeError(
                f"cuda_direct_backend: Numerical {op_name} is not supported for bool tensors. "
                "Use bitwise ops (BAND/BOR) instead. (rank={self.rank})."
            )

    def _apply_op(self, dst: torch.Tensor, src: torch.Tensor, op):
        """Accumulate src into dst in-place."""
        # IMPORTANT: Do NOT use `op in (SUM, AVG)` here.
        # When op comes from AllreduceOptions.reduceOp, it is a ReduceOp wrapper
        # whose __eq__ is asymmetric: `wrapper == SUM` is True but `SUM == wrapper`
        # is False.  The `in` operator calls the tuple element's __eq__ (reverse),
        # which fails.  Explicit `op == X` calls the wrapper's __eq__ (forward).
        if op == torch.distributed.ReduceOp.SUM or op == torch.distributed.ReduceOp.AVG:
            dst.add_(src)
        elif op == torch.distributed.ReduceOp.MAX:
            torch.maximum(dst, src, out=dst)
        elif op == torch.distributed.ReduceOp.MIN:
            torch.minimum(dst, src, out=dst)
        elif op == torch.distributed.ReduceOp.PRODUCT:
            dst.mul_(src)
        elif op == torch.distributed.ReduceOp.BAND:
            dst.bitwise_and_(src)
        elif op == torch.distributed.ReduceOp.BOR:
            dst.bitwise_or_(src)
        elif op == torch.distributed.ReduceOp.BXOR:
            dst.bitwise_xor_(src)
        else:
            raise ValueError(
                f"cuda_direct_backend: unsupported ReduceOp {op}  rank={self.rank}"
            )

    def _finalize_avg(self, tensor: torch.Tensor, op, denom: int):
        if op == torch.distributed.ReduceOp.AVG:
            tensor.div_(denom)

    # ---------------------------------------------------------------
    #  AllReduce
    # ---------------------------------------------------------------

    def _validate_tensors(self, tensors, op_name):
        """Hardware Guard: Ensure all tensors are on the correct GPU device."""
        if not tensors:
            return
            
        first_device = tensors[0].device
        for t in tensors:
            if not t.is_cuda:
                raise TypeError(
                    f"cuda_direct_backend: {op_name} expects CUDA tensors, "
                    f"got device={t.device}"
                )
            if t.device != first_device:
                raise ValueError(
                    f"cuda_direct_backend: {op_name} tensors must all be on the same "
                    f"GPU device. Found {first_device} and {t.device}"
                )

    def allreduce(self, tensor_list, op):
        self._validate_tensors(tensor_list, "allreduce")
        if len(tensor_list) != 1:
            raise ValueError(
                f"cuda_direct_backend: allreduce expects exactly 1 tensor, "
                f"got {len(tensor_list)}."
            )
        tensor = tensor_list[0]
        
        if not tensor.is_contiguous():
            temp_contiguous = tensor.contiguous()
            self._allreduce_dispatch(temp_contiguous, op)
            tensor.copy_(temp_contiguous)
            return

        self._allreduce_dispatch(tensor, op)

    def _allreduce_dispatch(self, tensor, op):
        """Internal dispatch for allreduce on a physically contiguous tensor."""
        if self._is_shm:
            self._shm_allreduce(tensor, op)
        elif self._use_ring:
            self._ring_allreduce(tensor, op)
        else:
            self._direct_allreduce(tensor, op)

    def _direct_allreduce(self, tensor, op):
        """All-to-all allreduce for 2 GPUs (P2P only)."""
        op = self._normalize_op(op)
        self._validate_op_dtype(tensor, op)

        self._exchange_handles(tensor)
        self.sync.barrier()

        peer_ptrs = []
        for p in range(self.world_size):
            if p != self.rank:
                numel, _, dev_idx, handle, ipc_off = self.sync.read_tensor_meta(p)
                if numel != tensor.numel():
                    raise RuntimeError(
                        f"cuda_direct allreduce size mismatch: "
                        f"rank={self.rank} numel={tensor.numel()}, peer={p} numel={numel}"
                    )
                ptr = self.transport.get_or_open_ipc_handle(p, handle, dev_idx) + ipc_off
                peer_ptrs.append((p, ptr))

        size_bytes = tensor.numel() * tensor.element_size()
        peer_buffers = {}
        for p, ptr in peer_ptrs:
            peer_buffers[p] = self._get_peer_buffer("allreduce", p, tensor)
            self.transport.copy_tensor(size_bytes, peer_buffers[p].data_ptr(), ptr, p)

        current_stream = torch.cuda.current_stream()
        for p, ptr in peer_ptrs:
            current_stream.wait_stream(self.transport.streams[p])
            self._apply_op(tensor, peer_buffers[p], op)

        self._finalize_avg(tensor, op, self.world_size)
        current_stream.synchronize()
        self.sync.barrier()

    def _ring_allreduce(self, tensor, op):
        """Ring allreduce: scatter-reduce then allgather (P2P, world_size >= 3)."""
        op = self._normalize_op(op)
        self._validate_op_dtype(tensor, op)

        N = self.world_size
        numel = tensor.numel()
        chunk_size = (numel + N - 1) // N
        elem_bytes = tensor.element_size()

        self.transport.publish_tensor(tensor, self.sync)
        self.sync.barrier()

        chunk_numel = min(chunk_size, numel)
        scratch = self._get_peer_buffer("ring_ar", self._ring_prev,
                                        tensor.narrow(0, 0, chunk_numel))
        pos = self._ring_pos

        # === Phase 1: Scatter-Reduce (N-1 steps) ===
        for k in range(N - 1):
            recv_idx = (pos - k - 1) % N
            src_start = recv_idx * chunk_size
            actual = min(chunk_size, numel - src_start)
            if actual <= 0:
                self.sync.barrier()
                continue

            src_offset = src_start * elem_bytes
            copy_bytes = actual * elem_bytes
            scratch_view = scratch.narrow(0, 0, actual)

            self.transport.fetch_chunk(self._ring_prev, scratch_view,
                                       src_offset, copy_bytes, self.sync)
            self.transport.wait_for_peer(self._ring_prev)

            self._apply_op(tensor.narrow(0, src_start, actual), scratch_view, op)
            torch.cuda.current_stream().synchronize()
            self.sync.barrier()

        # AVG: divide once after scatter-reduce, before allgather propagates
        if op == torch.distributed.ReduceOp.AVG:
            tensor.div_(N)

        # === Phase 2: AllGather (N-1 steps) ===
        for k in range(N - 1):
            recv_idx = (pos - k) % N
            src_start = recv_idx * chunk_size
            actual = min(chunk_size, numel - src_start)
            if actual <= 0:
                self.sync.barrier()
                continue

            src_offset = src_start * elem_bytes
            copy_bytes = actual * elem_bytes
            dst_view = tensor.narrow(0, src_start, actual)

            self.transport.fetch_chunk(self._ring_prev, dst_view,
                                       src_offset, copy_bytes, self.sync)
            self.transport.wait_for_peer(self._ring_prev)
            torch.cuda.current_stream().synchronize()
            self.sync.barrier()

        self.sync.barrier()

    def _shm_allreduce(self, tensor, op):
        op = self._normalize_op(op)
        self._validate_op_dtype(tensor, op)

        N = self.world_size
        size_bytes = tensor.numel() * tensor.element_size()
        pos = self._ring_pos

        # Pre-publish barrier: ensure all ranks' invalidate_cache() has completed
        # before any rank calls _ensure_my_shm (SHM regrowth).  On Windows,
        # SharedMemory(create=True) silently opens the OLD mapping if a peer
        # still has it open — this barrier prevents that race.
        self.sync.barrier()

        self.transport.publish_tensor(tensor, self.sync)
        self.sync.barrier()  # all ranks have published

        # Build peer order once so both loops use the same sequence
        peer_order = [self.ring_order[(pos - 1 - k) % N] for k in range(N - 1)]

        # Issue ALL fetches in parallel — each peer has its own H2D stream and
        # its own scratch buffer, so there is no contention between them.
        for peer_rank in peer_order:
            scratch = self._get_peer_buffer("shm_ar", peer_rank, tensor)
            self.transport.fetch_chunk(peer_rank, scratch, 0, size_bytes, self.sync)

        for peer_rank in peer_order:
            self.transport.wait_for_peer(peer_rank)
            scratch = self._get_peer_buffer("shm_ar", peer_rank, tensor)
            self._apply_op(tensor, scratch, op)

        self._finalize_avg(tensor, op, N)
        torch.cuda.current_stream().synchronize()
        self.sync.barrier()  # protect SHM from being overwritten before all reads done

    def allreduce_coalesced(self, tensor_list, op):
        """Coalesced allreduce for DDP bucketed gradient sync."""
        total_mb = sum(t.numel() * t.element_size() for t in tensor_list) / 1e6
        logger.debug("allreduce_coalesced  rank=%d  buckets=%d  total=%.3fMB  op=%s",
                     self.rank, len(tensor_list), total_mb, op)

        if len(tensor_list) == 1:
            return self.allreduce(tensor_list, op)

        flat = torch.cat([t.contiguous().view(-1) for t in tensor_list])
        self.allreduce([flat], op)

        offset = 0
        for t in tensor_list:
            numel = t.numel()
            t.copy_(flat[offset:offset + numel].view_as(t))
            offset += numel

    # ---------------------------------------------------------------
    #  Broadcast
    # ---------------------------------------------------------------

    def broadcast(self, tensor_list, root_rank):
        self._validate_tensors(tensor_list, "broadcast")
        if len(tensor_list) != 1:
            raise ValueError(
                f"cuda_direct_backend: broadcast expects exactly 1 tensor, "
                f"got {len(tensor_list)}."
            )
        tensor = tensor_list[0]

        # Pre-publish barrier: see _shm_allreduce comment on SHM regrowth race.
        self.sync.barrier()

        if self.rank == root_rank:
            send_tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
            self.transport.publish_tensor(send_tensor, self.sync)
            self.sync.barrier()
            torch.cuda.current_stream().synchronize()
            self.sync.barrier()
        else:
            self.sync.barrier()
            size_bytes = tensor.numel() * tensor.element_size()
            
            if not tensor.is_contiguous():
                recv_tensor = torch.empty_like(tensor).contiguous()
                self.transport.fetch_chunk(root_rank, recv_tensor, 0, size_bytes, self.sync)
                self.transport.wait_for_peer(root_rank)
                torch.cuda.current_stream().synchronize()
                tensor.copy_(recv_tensor)
            else:
                self.transport.fetch_chunk(root_rank, tensor, 0, size_bytes, self.sync)
                self.transport.wait_for_peer(root_rank)
                torch.cuda.current_stream().synchronize()
                
            self.sync.barrier()

    # ---------------------------------------------------------------
    #  AllGather
    # ---------------------------------------------------------------

    def _allgather_base(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        self._validate_tensors([output_tensor, input_tensor], "allgather")
        size_mb = input_tensor.numel() * input_tensor.element_size() / 1e6
        if self._is_shm:
            algo = "shm"
        elif self._use_ring:
            algo = "ring"
        else:
            algo = "direct"
        logger.debug("_allgather_base  rank=%d  input_shape=%s  size=%.3fMB  algo=%s",
                     self.rank, input_tensor.shape, size_mb, algo)

        if not output_tensor.is_contiguous():
            raise ValueError(
                f"cuda_direct _allgather_base requires contiguous output_tensor "
                f"(rank={self.rank}, shape={output_tensor.shape})"
            )
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # N-dimensional slicing mathematically crashes custom offsets unless flattened first
        output_tensor = output_tensor.view(-1)
        input_tensor = input_tensor.view(-1)

        if self._is_shm:
            self._shm_allgather_base(output_tensor, input_tensor)
        elif self._use_ring:
            self._ring_allgather_base(output_tensor, input_tensor)
        else:
            self._direct_allgather_base(output_tensor, input_tensor)

    def _direct_allgather_base(self, output_tensor, input_tensor):
        """All-to-all allgather for 2 GPUs (P2P only)."""
        handle, ipc_off = cuda_ipc.get_ipc_handle_and_offset(output_tensor)
        self.sync.write_tensor_meta(output_tensor.numel(), 0, output_tensor.device.index,
                                    handle, ipc_offset=ipc_off)

        input_numel = input_tensor.numel()
        start_idx = self.rank * input_numel

        if input_numel > 0:
            output_tensor[start_idx : start_idx + input_numel].copy_(input_tensor)

        self.sync.barrier()

        if input_numel == 0:
            self.sync.barrier()
            return

        bytes_per_elem = output_tensor.element_size()
        total_send_bytes = input_numel * bytes_per_elem

        peer_ptrs = []
        for p in range(self.world_size):
            if p != self.rank:
                numel, _, dev_idx, peer_handle, peer_ipc_off = self.sync.read_tensor_meta(p)
                ptr = self.transport.get_or_open_ipc_handle(p, peer_handle, dev_idx) + peer_ipc_off
                peer_ptrs.append((p, ptr))

        src_base = input_tensor.data_ptr()

        chunk_size_bytes = 64 * 1024 * 1024
        if total_send_bytes < chunk_size_bytes:
            chunk_size_bytes = total_send_bytes
        num_chunks = (total_send_bytes + chunk_size_bytes - 1) // chunk_size_bytes

        for p, _ in peer_ptrs:
            self.sync.write_tail(p, 0)

        for c in range(num_chunks):
            offset = c * chunk_size_bytes
            size = min(chunk_size_bytes, total_send_bytes - offset)
            expected_tail = offset + size

            for p, ptr in peer_ptrs:
                dst_offset = (self.rank * total_send_bytes) + offset
                self.transport.copy_tensor(size, ptr + dst_offset, src_base + offset, p)

            for p, _ in peer_ptrs:
                self.transport.sync_peer_stream(p)
                self.sync.write_tail(p, expected_tail)

            for p, _ in peer_ptrs:
                while self.sync.read_tail(p) < expected_tail:
                    time.sleep(0.000_05)

    def _ring_allgather_base(self, output_tensor, input_tensor):
        """Ring allgather for P2P (world_size >= 3)."""
        N = self.world_size
        input_numel = input_tensor.numel()
        elem_bytes = output_tensor.element_size()
        segment_bytes = input_numel * elem_bytes

        start_idx = self.rank * input_numel
        if input_numel > 0:
            output_tensor[start_idx : start_idx + input_numel].copy_(input_tensor)

        self.transport.publish_tensor(output_tensor, self.sync)

        # Commit the local shard copy before signaling the barrier.
        # After the barrier, peers fetch_chunk from our output_tensor via
        # IPC — the DMA has no visibility into our stream ordering, so the
        # copy_ must be physically committed to device memory first.
        # This is lightweight (~0.1 ms for a local GPU copy) because the
        # worker stream has no dependency on the caller's compute stream
        # when dispatched via _allgather_work.
        torch.cuda.current_stream().synchronize()

        self.sync.barrier()

        if input_numel == 0:
            self.sync.barrier()
            return

        pos = self._ring_pos

        for k in range(N - 1):
            source_rank = self.ring_order[(pos - 1 - k) % N]
            src_offset = source_rank * segment_bytes
            dst_view = output_tensor.narrow(0, source_rank * input_numel, input_numel)

            self.transport.fetch_chunk(self._ring_prev, dst_view,
                                       src_offset, segment_bytes, self.sync)
            self.transport.wait_for_peer(self._ring_prev)
            torch.cuda.current_stream().synchronize()
            self.sync.barrier()

    def _shm_allgather_base(self, output_tensor, input_tensor):
        """
        SHM allgather: each rank publishes its input shard, then pulls all others.

        Parallel fetch: all peer chunks are issued simultaneously using
        per-peer pinned buffers and H2D streams, then synchronized once.
        """
        N = self.world_size
        input_numel = input_tensor.numel()
        segment_bytes = input_numel * output_tensor.element_size()
        pos = self._ring_pos

        # Place own shard in output
        start_idx = self.rank * input_numel
        if input_numel > 0:
            output_tensor[start_idx : start_idx + input_numel].copy_(input_tensor)

        # Pre-publish barrier: see _shm_allreduce comment on SHM regrowth race.
        self.sync.barrier()

        # Publish input shard (peers will pull it)
        self.transport.publish_tensor(input_tensor, self.sync)
        self.sync.barrier()  # all shards published

        if input_numel == 0:
            self.sync.barrier()
            return

        # Issue ALL fetch_chunk calls in parallel (each peer has its own
        # pinned recv buffer and H2D stream, so no contention)
        for k in range(N - 1):
            peer_rank = self.ring_order[(pos - 1 - k) % N]
            dst_view = output_tensor.narrow(0, peer_rank * input_numel, input_numel)
            self.transport.fetch_chunk(peer_rank, dst_view, 0, segment_bytes, self.sync)

        # Single sync point: wait for ALL H2D transfers to finish
        if hasattr(self.transport, 'wait_all_peers'):
            self.transport.wait_all_peers()
        else:
            # Fallback for non-SHM transports
            for k in range(N - 1):
                peer_rank = self.ring_order[(pos - 1 - k) % N]
                self.transport.wait_for_peer(peer_rank)

        torch.cuda.current_stream().synchronize()
        self.sync.barrier()  # protect SHM before next publish

    # ---------------------------------------------------------------
    #  ReduceScatter
    # ---------------------------------------------------------------

    def _reduce_scatter_base(self, output_tensor: torch.Tensor,
                             input_tensor: torch.Tensor, op=None):
        self._validate_tensors([output_tensor, input_tensor], "reduce_scatter")
        size_mb = input_tensor.numel() * input_tensor.element_size() / 1e6
        if self._is_shm:
            algo = "shm"
        elif self._use_ring:
            algo = "ring"
        else:
            algo = "direct"
        logger.debug("_reduce_scatter_base  rank=%d  op=%s  input_shape=%s  "
                     "size=%.3fMB  algo=%s",
                     self.rank, op, input_tensor.shape, size_mb, algo)

        if not output_tensor.is_contiguous():
            raise ValueError(
                f"cuda_direct _reduce_scatter_base requires contiguous output_tensor "
                f"(rank={self.rank}, shape={output_tensor.shape})"
            )
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # N-dimensional slicing mathematically crashes custom offsets unless flattened first
        output_tensor = output_tensor.view(-1)
        input_tensor = input_tensor.view(-1)

        if self._is_shm:
            self._shm_reduce_scatter_base(output_tensor, input_tensor, op)
        elif self._use_ring:
            self._ring_reduce_scatter_base(output_tensor, input_tensor, op)
        else:
            self._direct_reduce_scatter_base(output_tensor, input_tensor, op)

    def _direct_reduce_scatter_base(self, output_tensor, input_tensor, op):
        """All-to-all reduce-scatter for 2 GPUs (P2P only)."""
        handle, ipc_off = cuda_ipc.get_ipc_handle_and_offset(input_tensor)
        self.sync.write_tensor_meta(input_tensor.numel(), 0, input_tensor.device.index,
                                    handle, ipc_offset=ipc_off)
        self.sync.barrier()

        output_numel = output_tensor.numel()
        start_idx = self.rank * output_numel

        peer_ptrs = []
        for p in range(self.world_size):
            if p != self.rank:
                numel, _, dev_idx, peer_handle, peer_ipc_off = self.sync.read_tensor_meta(p)
                if numel != input_tensor.numel():
                    raise RuntimeError(
                        f"cuda_direct reduce_scatter size mismatch: "
                        f"rank={self.rank} numel={input_tensor.numel()}, peer={p} numel={numel}"
                    )
                ptr = self.transport.get_or_open_ipc_handle(p, peer_handle, dev_idx) + peer_ipc_off
                peer_ptrs.append((p, ptr))

        bytes_per_elem = output_tensor.element_size()
        segment_bytes = output_numel * bytes_per_elem
        read_offset = start_idx * bytes_per_elem

        if output_numel > 0:
            output_tensor.copy_(input_tensor[start_idx : start_idx + output_numel])

        op = self._normalize_op(op)
        self._validate_op_dtype(output_tensor, op)

        current_stream = torch.cuda.current_stream()
        peer_buffers = {}
        for p, ptr in peer_ptrs:
            peer_buffers[p] = self._get_peer_buffer("reduce_scatter", p, output_tensor)
            self.transport.copy_tensor(segment_bytes, peer_buffers[p].data_ptr(),
                                       ptr + read_offset, p)

        for p, ptr in peer_ptrs:
            current_stream.wait_stream(self.transport.streams[p])
            self._apply_op(output_tensor, peer_buffers[p], op)

        self._finalize_avg(output_tensor, op, self.world_size)
        current_stream.synchronize()

    def _ring_reduce_scatter_base(self, output_tensor, input_tensor, op):
        """Ring-ordered reduce-scatter (P2P, world_size >= 3)."""
        N = self.world_size
        output_numel = output_tensor.numel()
        elem_bytes = output_tensor.element_size()
        segment_bytes = output_numel * elem_bytes
        start_idx = self.rank * output_numel
        read_offset = start_idx * elem_bytes

        self.transport.publish_tensor(input_tensor, self.sync)
        self.sync.barrier()

        if output_numel > 0:
            output_tensor.copy_(input_tensor[start_idx : start_idx + output_numel])

        op = self._normalize_op(op)
        self._validate_op_dtype(output_tensor, op)

        pos = self._ring_pos
        stream = torch.cuda.current_stream()
        peer_order = [self.ring_order[(pos - 1 - k) % N] for k in range(N - 1)]

        # Issue all fetches in parallel — each targets a different peer stream
        for peer_rank in peer_order:
            scratch = self._get_peer_buffer("ring_rs", peer_rank, output_tensor)
            self.transport.fetch_chunk(peer_rank, scratch,
                                       read_offset, segment_bytes, self.sync)

        # Wait per-peer then reduce — all DMAs already in flight
        for peer_rank in peer_order:
            self.transport.wait_for_peer(peer_rank)
            scratch = self._get_peer_buffer("ring_rs", peer_rank, output_tensor)
            self._apply_op(output_tensor, scratch, op)

        self._finalize_avg(output_tensor, op, N)
        stream.synchronize()

    def _shm_reduce_scatter_base(self, output_tensor, input_tensor, op):
        """
        SHM reduce-scatter: publish full input once, pull my segment from all N-1 peers.
        Safe for SHM because we always read the same fixed offset from each peer's
        constant input_tensor snapshot.

        Parallel fetch: all N-1 segment DMAs are issued simultaneously. Each peer's
        segment is small (input/N bytes), so per-peer scratch buffers are cheap.
        """
        N = self.world_size
        output_numel = output_tensor.numel()
        elem_bytes = output_tensor.element_size()
        segment_bytes = output_numel * elem_bytes
        start_idx = self.rank * output_numel
        read_offset = start_idx * elem_bytes
        pos = self._ring_pos

        # Pre-publish barrier: see _shm_allreduce comment on SHM regrowth race.
        self.sync.barrier()

        self.transport.publish_tensor(input_tensor, self.sync)
        self.sync.barrier()  # all full tensors published

        if output_numel > 0:
            output_tensor.copy_(input_tensor[start_idx : start_idx + output_numel])

        op = self._normalize_op(op)
        self._validate_op_dtype(output_tensor, op)

        peer_order = [self.ring_order[(pos - 1 - k) % N] for k in range(N - 1)]

        # Issue all segment fetches in parallel (different peers, different H2D streams)
        for peer_rank in peer_order:
            scratch = self._get_peer_buffer("shm_rs", peer_rank, output_tensor)
            self.transport.fetch_chunk(peer_rank, scratch,
                                       read_offset, segment_bytes, self.sync)

        # Wait per-peer and reduce — DMAs already in flight so waits are near-instant
        for peer_rank in peer_order:
            self.transport.wait_for_peer(peer_rank)
            scratch = self._get_peer_buffer("shm_rs", peer_rank, output_tensor)
            self._apply_op(output_tensor, scratch, op)

        self._finalize_avg(output_tensor, op, N)
        torch.cuda.current_stream().synchronize()
        # No extra barrier needed: process_group._reduce_scatter_base calls barrier() after

    # ---------------------------------------------------------------
    #  AllToAll helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _prefix_sums(sizes: list) -> list:
        """Return exclusive prefix sums: [0, sizes[0], sizes[0]+sizes[1], ...]"""
        offsets = [0] * len(sizes)
        for i in range(1, len(sizes)):
            offsets[i] = offsets[i - 1] + sizes[i - 1]
        return offsets

    # ---------------------------------------------------------------
    #  AllToAll
    # ---------------------------------------------------------------

    def all_to_all_single(self, output_tensor, input_tensor,
                          output_split_sizes=None, input_split_sizes=None):
        """All-to-all: rank r sends its p-th chunk to rank p, receives its r-th chunk from rank p.

        Equal splits: output_split_sizes and input_split_sizes are both None (or omitted).
            Each rank sends/receives exactly numel/world_size elements to/from every peer.

        Unequal splits: one or both split-size lists are supplied.
            input_split_sizes[p]  — number of elements this rank sends to rank p.
            output_split_sizes[p] — number of elements this rank receives from rank p.
            output_split_sizes[p] must equal rank p's input_split_sizes[self.rank].

        Protocol (pull model, same as equal-split):
          1. Each rank publishes its input tensor (IPC handle / SHM region) and its
             input_split_sizes array into the shared metadata slot.
          2. After a barrier, each rank reads every peer's split sizes, computes the
             source byte offset as prefix_sum(peer_input_split_sizes[:self.rank]), and
             issues a DMA copy of the correct slice into the local output buffer.
        """
        self._validate_tensors([output_tensor, input_tensor], "all_to_all")

        N = self.world_size
        size_mb = input_tensor.numel() * input_tensor.element_size() / 1e6
        if self._is_shm:
            algo = "shm"
        elif self._use_ring:
            algo = "ring"
        else:
            algo = "direct"

        if not output_tensor.is_contiguous():
            raise ValueError(
                f"cuda_direct all_to_all requires contiguous output_tensor "
                f"(rank={self.rank}, shape={output_tensor.shape})"
            )
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # ── Unequal-splits path ─────────────────────────────────────────────
        if output_split_sizes is not None or input_split_sizes is not None:
            total_in  = input_tensor.numel()
            total_out = output_tensor.numel()

            in_splits  = list(input_split_sizes)  if input_split_sizes  is not None \
                         else [total_in  // N] * N
            out_splits = list(output_split_sizes) if output_split_sizes is not None \
                         else [total_out // N] * N

            if len(in_splits) != N:
                raise ValueError(
                    f"cuda_direct all_to_all: input_split_sizes length {len(in_splits)} "
                    f"!= world_size {N}  (rank={self.rank})"
                )
            if len(out_splits) != N:
                raise ValueError(
                    f"cuda_direct all_to_all: output_split_sizes length {len(out_splits)} "
                    f"!= world_size {N}  (rank={self.rank})"
                )
            if any(s < 0 for s in in_splits):
                raise ValueError(
                    f"cuda_direct all_to_all: negative value in input_split_sizes "
                    f"{in_splits}  (rank={self.rank})"
                )
            if any(s < 0 for s in out_splits):
                raise ValueError(
                    f"cuda_direct all_to_all: negative value in output_split_sizes "
                    f"{out_splits}  (rank={self.rank})"
                )
            if sum(in_splits) != total_in:
                raise ValueError(
                    f"cuda_direct all_to_all: sum(input_split_sizes)={sum(in_splits)} "
                    f"!= input numel {total_in}  (rank={self.rank})"
                )
            if sum(out_splits) != total_out:
                raise ValueError(
                    f"cuda_direct all_to_all: sum(output_split_sizes)={sum(out_splits)} "
                    f"!= output numel {total_out}  (rank={self.rank})"
                )

            logger.debug(
                "all_to_all_single  rank=%d  UNEQUAL  input_shape=%s  size=%.3fMB  "
                "algo=%s  in_splits=%s  out_splits=%s",
                self.rank, input_tensor.shape, size_mb, algo, in_splits, out_splits,
            )

            output_tensor = output_tensor.view(-1)
            input_tensor  = input_tensor.view(-1)

            if self._is_shm:
                self._shm_alltoallv(output_tensor, input_tensor, out_splits, in_splits)
            elif self._use_ring:
                self._ring_alltoallv(output_tensor, input_tensor, out_splits, in_splits)
            else:
                self._direct_alltoallv(output_tensor, input_tensor, out_splits, in_splits)
            return

        # ── Equal-splits path ───────────────────────────────────────────────
        logger.debug("all_to_all_single  rank=%d  EQUAL  input_shape=%s  size=%.3fMB  algo=%s",
                     self.rank, input_tensor.shape, size_mb, algo)

        total = input_tensor.numel()
        if total != output_tensor.numel():
            raise ValueError(
                f"cuda_direct all_to_all: input numel ({total}) != output numel "
                f"({output_tensor.numel()}) for equal split (rank={self.rank})"
            )
        if total % N != 0:
            raise ValueError(
                f"cuda_direct all_to_all: numel {total} not divisible by "
                f"world_size {N} (rank={self.rank})"
            )

        output_tensor = output_tensor.view(-1)
        input_tensor = input_tensor.view(-1)

        if self._is_shm:
            self._shm_all_to_all(output_tensor, input_tensor)
        elif self._use_ring:
            self._ring_all_to_all(output_tensor, input_tensor)
        else:
            self._direct_all_to_all(output_tensor, input_tensor)

    def _direct_all_to_all(self, output_tensor, input_tensor):
        """P2P all-to-all for 2 GPUs."""
        N = self.world_size
        chunk_numel = input_tensor.numel() // N
        elem_bytes = input_tensor.element_size()
        chunk_bytes = chunk_numel * elem_bytes

        # Copy own chunk (rank's data to itself)
        own = self.rank * chunk_numel
        output_tensor[own : own + chunk_numel].copy_(
            input_tensor[own : own + chunk_numel])

        # Exchange IPC handles
        handle, ipc_off = cuda_ipc.get_ipc_handle_and_offset(input_tensor)
        self.sync.write_tensor_meta(
            input_tensor.numel(), 0, input_tensor.device.index, handle,
            ipc_offset=ipc_off)
        self.sync.barrier()

        # Read chunk[self.rank] from each peer → output[peer]
        for p in range(N):
            if p == self.rank:
                continue
            numel, _, dev_idx, peer_handle, peer_ipc_off = self.sync.read_tensor_meta(p)
            ptr = self.transport.get_or_open_ipc_handle(p, peer_handle, dev_idx) + peer_ipc_off
            src_offset = self.rank * chunk_bytes
            dst_view = output_tensor.narrow(0, p * chunk_numel, chunk_numel)
            self.transport.copy_tensor(
                chunk_bytes, dst_view.data_ptr(), ptr + src_offset, p)

        current_stream = torch.cuda.current_stream()
        for p in range(N):
            if p != self.rank:
                current_stream.wait_stream(self.transport.streams[p])
        current_stream.synchronize()
        self.sync.barrier()

    def _ring_all_to_all(self, output_tensor, input_tensor):
        """P2P all-to-all for 3+ GPUs. Parallel fetch from all peers."""
        N = self.world_size
        chunk_numel = input_tensor.numel() // N
        chunk_bytes = chunk_numel * input_tensor.element_size()

        own = self.rank * chunk_numel
        output_tensor[own : own + chunk_numel].copy_(
            input_tensor[own : own + chunk_numel])

        self.transport.publish_tensor(input_tensor, self.sync)
        self.sync.barrier()

        pos = self._ring_pos
        read_offset = self.rank * chunk_bytes

        # Issue all fetches in parallel — each peer has its own DMA stream
        for k in range(N - 1):
            peer_rank = self.ring_order[(pos - 1 - k) % N]
            dst_view = output_tensor.narrow(0, peer_rank * chunk_numel, chunk_numel)
            self.transport.fetch_chunk(
                peer_rank, dst_view, read_offset, chunk_bytes, self.sync)

        for k in range(N - 1):
            peer_rank = self.ring_order[(pos - 1 - k) % N]
            self.transport.wait_for_peer(peer_rank)

        torch.cuda.current_stream().synchronize()
        self.sync.barrier()

    def _shm_all_to_all(self, output_tensor, input_tensor):
        """SHM all-to-all: publish input once, each rank pulls its chunks from all peers."""
        N = self.world_size
        chunk_numel = input_tensor.numel() // N
        chunk_bytes = chunk_numel * input_tensor.element_size()
        pos = self._ring_pos

        own = self.rank * chunk_numel
        output_tensor[own : own + chunk_numel].copy_(
            input_tensor[own : own + chunk_numel])

        # Pre-publish barrier: see _shm_allreduce comment on SHM regrowth race.
        self.sync.barrier()
        self.transport.publish_tensor(input_tensor, self.sync)
        self.sync.barrier()

        read_offset = self.rank * chunk_bytes

        # Issue all fetches in parallel (per-peer H2D streams)
        for k in range(N - 1):
            peer_rank = self.ring_order[(pos - 1 - k) % N]
            dst_view = output_tensor.narrow(0, peer_rank * chunk_numel, chunk_numel)
            self.transport.fetch_chunk(
                peer_rank, dst_view, read_offset, chunk_bytes, self.sync)

        if hasattr(self.transport, 'wait_all_peers'):
            self.transport.wait_all_peers()
        else:
            for k in range(N - 1):
                peer_rank = self.ring_order[(pos - 1 - k) % N]
                self.transport.wait_for_peer(peer_rank)

        torch.cuda.current_stream().synchronize()
        self.sync.barrier()

    # ---------------------------------------------------------------
    #  AllToAllV  (unequal splits)
    # ---------------------------------------------------------------
    #
    # Protocol mirrors the equal-split variants with one addition: each rank
    # also publishes its input_split_sizes array into the free tail of the
    # metadata slot (bytes [88:]).  After the barrier every rank reads each
    # peer's split sizes, computes the source byte offset as
    #
    #     src_offset = prefix_sum(peer_input_split_sizes[:self.rank]) * elem_bytes
    #
    # and issues a DMA copy of exactly output_split_sizes[peer] elements into
    # the correct position of the local output buffer.
    #
    # write_split_sizes / read_split_sizes live alongside write_tensor_meta in
    # the same 256-byte slot but occupy non-overlapping byte ranges, so the
    # two writes can happen in any order and both are visible after the barrier.

    def _direct_alltoallv(self, output_tensor, input_tensor,
                          output_split_sizes, input_split_sizes):
        """P2P all-to-all with unequal splits for 2 GPUs."""
        N = self.world_size
        elem_bytes = input_tensor.element_size()
        send_offsets = self._prefix_sums(input_split_sizes)
        recv_offsets = self._prefix_sums(output_split_sizes)

        # Self-copy: rank → rank chunk requires no network
        own_size = input_split_sizes[self.rank]
        if own_size > 0:
            output_tensor[recv_offsets[self.rank] : recv_offsets[self.rank] + own_size].copy_(
                input_tensor[send_offsets[self.rank] : send_offsets[self.rank] + own_size])

        # Publish IPC handle for own input tensor + split sizes.
        # write_split_sizes writes bytes [88:]; write_tensor_meta writes bytes [0:88].
        # The two byte ranges do not overlap, so order is irrelevant, but writing
        # splits first keeps split data stable before the IPC handle is visible.
        handle, ipc_off = cuda_ipc.get_ipc_handle_and_offset(input_tensor)
        self.sync.write_split_sizes(input_split_sizes)
        self.sync.write_tensor_meta(input_tensor.numel(), 0,
                                    input_tensor.device.index, handle,
                                    ipc_offset=ipc_off)
        self.sync.barrier()

        current_stream = torch.cuda.current_stream()
        for p in range(N):
            if p == self.rank:
                continue
            recv_size = output_split_sizes[p]
            if recv_size == 0:
                continue

            _numel, _, dev_idx, peer_handle, peer_ipc_off = self.sync.read_tensor_meta(p)
            peer_splits       = self.sync.read_split_sizes(p, N)
            peer_send_offsets = self._prefix_sums(peer_splits)

            # Byte offset into peer's input where our data starts
            src_offset_bytes = peer_send_offsets[self.rank] * elem_bytes
            copy_bytes       = recv_size * elem_bytes
            dst_ptr          = output_tensor.data_ptr() + recv_offsets[p] * elem_bytes

            ptr = self.transport.get_or_open_ipc_handle(p, peer_handle, dev_idx) + peer_ipc_off
            self.transport.copy_tensor(copy_bytes, dst_ptr, ptr + src_offset_bytes, p)

        for p in range(N):
            if p != self.rank and output_split_sizes[p] > 0:
                current_stream.wait_stream(self.transport.streams[p])

        current_stream.synchronize()
        self.sync.barrier()

    def _ring_alltoallv(self, output_tensor, input_tensor,
                        output_split_sizes, input_split_sizes):
        """P2P all-to-all with unequal splits for 3+ GPUs. Parallel fetch from all peers."""
        N = self.world_size
        elem_bytes = input_tensor.element_size()
        send_offsets = self._prefix_sums(input_split_sizes)
        recv_offsets = self._prefix_sums(output_split_sizes)
        pos = self._ring_pos

        # Self-copy
        own_size = input_split_sizes[self.rank]
        if own_size > 0:
            output_tensor[recv_offsets[self.rank] : recv_offsets[self.rank] + own_size].copy_(
                input_tensor[send_offsets[self.rank] : send_offsets[self.rank] + own_size])

        # Publish split sizes then IPC handle (via publish_tensor).
        # publish_tensor writes bytes [0:88]; write_split_sizes writes bytes [88:].
        self.sync.write_split_sizes(input_split_sizes)
        self.transport.publish_tensor(input_tensor, self.sync)
        # Ensure own data is committed to device memory before barrier so peers'
        # DMA reads see the correct contents.
        torch.cuda.current_stream().synchronize()
        self.sync.barrier()

        # Issue all peer fetches in parallel — each uses its own DMA stream
        for k in range(N - 1):
            peer_rank = self.ring_order[(pos - 1 - k) % N]
            recv_size = output_split_sizes[peer_rank]
            if recv_size == 0:
                continue

            peer_splits       = self.sync.read_split_sizes(peer_rank, N)
            peer_send_offsets = self._prefix_sums(peer_splits)
            src_offset_bytes  = peer_send_offsets[self.rank] * elem_bytes
            copy_bytes        = recv_size * elem_bytes
            dst_view          = output_tensor.narrow(0, recv_offsets[peer_rank], recv_size)

            self.transport.fetch_chunk(peer_rank, dst_view,
                                       src_offset_bytes, copy_bytes, self.sync)

        for k in range(N - 1):
            peer_rank = self.ring_order[(pos - 1 - k) % N]
            if output_split_sizes[peer_rank] > 0:
                self.transport.wait_for_peer(peer_rank)

        torch.cuda.current_stream().synchronize()
        self.sync.barrier()

    def _shm_alltoallv(self, output_tensor, input_tensor,
                       output_split_sizes, input_split_sizes):
        """SHM all-to-all with unequal splits."""
        N = self.world_size
        elem_bytes = input_tensor.element_size()
        send_offsets = self._prefix_sums(input_split_sizes)
        recv_offsets = self._prefix_sums(output_split_sizes)
        pos = self._ring_pos

        # Self-copy
        own_size = input_split_sizes[self.rank]
        if own_size > 0:
            output_tensor[recv_offsets[self.rank] : recv_offsets[self.rank] + own_size].copy_(
                input_tensor[send_offsets[self.rank] : send_offsets[self.rank] + own_size])

        # Pre-publish barrier: prevents SHM regrowth race (see _shm_allreduce comment).
        self.sync.barrier()

        # Write split sizes (bytes [88:]) before publish_tensor (bytes [0:88]).
        # Both are visible to peers after the barrier below.
        self.sync.write_split_sizes(input_split_sizes)
        self.transport.publish_tensor(input_tensor, self.sync)
        self.sync.barrier()  # all ranks have published input data and split sizes

        # Issue all peer fetches in parallel (per-peer H2D streams, no contention)
        for k in range(N - 1):
            peer_rank = self.ring_order[(pos - 1 - k) % N]
            recv_size = output_split_sizes[peer_rank]
            if recv_size == 0:
                continue

            peer_splits       = self.sync.read_split_sizes(peer_rank, N)
            peer_send_offsets = self._prefix_sums(peer_splits)
            src_offset_bytes  = peer_send_offsets[self.rank] * elem_bytes
            copy_bytes        = recv_size * elem_bytes
            dst_view          = output_tensor.narrow(0, recv_offsets[peer_rank], recv_size)

            self.transport.fetch_chunk(peer_rank, dst_view,
                                       src_offset_bytes, copy_bytes, self.sync)

        if hasattr(self.transport, 'wait_all_peers'):
            self.transport.wait_all_peers()
        else:
            for k in range(N - 1):
                peer_rank = self.ring_order[(pos - 1 - k) % N]
                if output_split_sizes[peer_rank] > 0:
                    self.transport.wait_for_peer(peer_rank)

        torch.cuda.current_stream().synchronize()
        self.sync.barrier()

    # ---------------------------------------------------------------
    #  Barrier
    # ---------------------------------------------------------------

    def barrier(self):
        self.sync.barrier()
