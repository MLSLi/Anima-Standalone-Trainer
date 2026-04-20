"""
Cross-process synchronization using shared memory.

No TCP/IP — all coordination happens through OS shared memory (mmap-backed).
Each rank reads/writes to fixed slots in shared memory to synchronize
collective operations.
"""

import atexit
import ctypes
import logging
import struct
import time
from multiprocessing import shared_memory

logger = logging.getLogger("cuda_direct.sync")

# Layout constants
_SLOT_SIZE = 256            # bytes per rank metadata slot
_BARRIER_COUNTER_SIZE = 4   # int32 per rank arrival flag
_BARRIER_PHASE_SIZE = 4     # int32 global phase
_IPC_HANDLE_SIZE = 64       # cudaIpcMemHandle_t size

# Byte layout within each metadata slot:
#   [0:8]   numel (int64)
#   [8:12]  dtype_code (int32)
#   [12:16] device_index (int32)
#   [16:80] ipc_handle (64 bytes)
#   [80:88] ipc_offset (int64) — byte offset from the IPC base ptr to the tensor data
#           Required because PyTorch's caching allocator sub-allocates from large
#           cudaMalloc blocks; cudaIpcGetMemHandle gives the block's handle, not the
#           tensor's handle.  The receiver adds this offset after opening the handle.
#   [88:]   split sizes (int32 each) — used by variable all_to_all
_TENSOR_META_SIZE = 88      # bytes consumed by write_tensor_meta (was 80)
_SPLIT_SIZES_OFFSET = _TENSOR_META_SIZE
_SPLIT_SIZES_MAX = (_SLOT_SIZE - _TENSOR_META_SIZE) // 4  # max world_size supported (42)

# How long to spin before emitting a slow-barrier warning (seconds)
_BARRIER_WARN_AFTER = 5.0
# Interval between slow-barrier log lines so they don't flood the log
_BARRIER_WARN_INTERVAL = 10.0


def _shm_name(session_id: str, purpose: str) -> str:
    return f"cuda_direct_{session_id}_{purpose}"


class SharedBarrier:
    """
    Lock-free sense-reversing barrier using shared memory.

    Layout:
      [0:4]              phase (uint32)  — tracks current sense value (0 or 1)
      [4 : 4+4*N]        per-rank arrival flag (uint32 each)

    Protocol (sense-reversing):
      Each rank keeps a local sense bit that flips every call (0→1→0→1...).
      The expected slot value for barrier N alternates, so a stale slot value
      from barrier N-1 is always the WRONG value for barrier N — eliminating
      the race where rank 0 mistakes a stale arrival for a new one.

      1. Flip local sense → this is the expected slot value this barrier
      2. Write expected value to my slot
      3. Spin until all slots == expected
      4. Rank 0 writes phase = expected; others wait for phase == expected
      (No explicit slot reset needed — the next barrier's write handles it)
    """

    def __init__(self, rank: int, world_size: int, session_id: str,
                 create: bool = False, timeout: float = 300.0):
        self._rank = rank
        self._world_size = world_size
        self._name = _shm_name(session_id, "barrier")
        self._timeout = timeout
        total_size = _BARRIER_PHASE_SIZE + _BARRIER_COUNTER_SIZE * world_size

        if create:
            self._shm = shared_memory.SharedMemory(
                name=self._name, create=True, size=total_size
            )
            self._shm.buf[:total_size] = b'\x00' * total_size
            logger.debug("SharedBarrier created  name=%s  world_size=%d",
                         self._name, world_size)
        else:
            self._shm = shared_memory.SharedMemory(name=self._name, create=False)
            logger.debug("SharedBarrier attached  name=%s", self._name)

        self._total_size = total_size
        self._barrier_count = 0   # incremented each time this rank passes a barrier
        self._sense = 0           # local sense bit — flips each barrier call
        atexit.register(self._cleanup)

    def _cleanup(self):
        try:
            self._shm.close()
            if self._rank == 0:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
        except Exception:
            pass

    def _read_phase(self) -> int:
        return struct.unpack_from('I', self._shm.buf, 0)[0]

    def _write_phase(self, phase: int):
        struct.pack_into('I', self._shm.buf, 0, phase)

    def _read_slot(self, rank: int) -> int:
        offset = _BARRIER_PHASE_SIZE + rank * _BARRIER_COUNTER_SIZE
        return struct.unpack_from('I', self._shm.buf, offset)[0]

    def _write_slot(self, rank: int, value: int):
        offset = _BARRIER_PHASE_SIZE + rank * _BARRIER_COUNTER_SIZE
        struct.pack_into('I', self._shm.buf, offset, value)

    def wait(self, timeout: float | None = None):
        """Block until all ranks have called wait().

        Protocol (sense-reversing, rank-0-gated):

          Each rank keeps a local sense bit that flips every call (0→1→0→1...).
          The expected slot value for barrier N alternates, so a stale slot value
          from barrier N-1 is always the WRONG value for barrier N.

          1. Flip local sense → this is the expected slot value this barrier
          2. Write expected value to MY slot
          3. RANK 0 ONLY: spin until ALL slots == expected
          4. Rank 0 writes phase = expected; others wait for phase == expected

        Why only rank 0 checks all slots (critical correctness point):
          If all ranks checked all slots, rank 0 could exit the barrier, enter the
          NEXT barrier, and overwrite slot[0] with the next expected value BEFORE
          slower ranks finish checking slot[0] in the current barrier.  Those ranks
          would then see slot[0] != current_expected and spin forever — deadlock.
          By having only rank 0 check all slots (rank 0 is never in two barriers at
          once), and non-rank-0 ranks only waiting on the phase signal, this race
          is eliminated entirely.
        """
        if timeout is None:
            timeout = self._timeout

        self._sense = 1 - self._sense
        expected = self._sense
        deadline = time.monotonic() + timeout
        self._barrier_count += 1

        # All ranks signal arrival by writing to their own slot.
        self._write_slot(self._rank, expected)
        logger.debug("barrier #%d  rank=%d  arrived  expected=%d",
                     self._barrier_count, self._rank, expected)

        if self._rank == 0:
            # Rank 0: wait for ALL other ranks to write expected, then signal.
            # Reading other ranks' slots is safe here because rank 0 cannot
            # be in two barriers simultaneously — it only reaches the next
            # barrier after exiting this one.
            next_warn = time.monotonic() + _BARRIER_WARN_AFTER
            while True:
                missing = [r for r in range(self._world_size)
                           if self._read_slot(r) != expected]
                if not missing:
                    break
                now = time.monotonic()
                if now > deadline:
                    raise TimeoutError(
                        f"cuda_direct_backend: barrier #{self._barrier_count} timed out "
                        f"after {timeout:.1f}s — rank=0 is waiting for ranks "
                        f"{missing} to arrive. "
                        f"Possible causes: crashed rank, exception in collective op, "
                        f"mismatched collective call order, or insufficient timeout "
                        f"(current: {timeout}s)."
                    )
                if now > next_warn:
                    elapsed = now - (deadline - timeout)
                    logger.debug(
                        "barrier #%d  rank=0  slow — waited %.1fs, still waiting for "
                        "ranks %s  (timeout in %.1fs)",
                        self._barrier_count, elapsed, missing, deadline - now
                    )
                    next_warn = now + _BARRIER_WARN_INTERVAL
                time.sleep(0.0001)
            self._write_phase(expected)

        else:
            # Non-rank-0: DO NOT read other ranks' slots.
            # Rank 0 may already have entered the next barrier and overwritten
            # slot[0] with the next expected value.  Reading slot[0] here would
            # see the wrong value and loop forever.  Instead, wait only for the
            # phase signal that rank 0 writes after it has verified all arrivals.
            next_warn = time.monotonic() + _BARRIER_WARN_AFTER
            while self._read_phase() != expected:
                now = time.monotonic()
                if now > deadline:
                    raise TimeoutError(
                        f"cuda_direct_backend: barrier #{self._barrier_count} phase-flip "
                        f"timed out after {timeout:.1f}s — rank={self._rank} waiting for "
                        f"rank 0 to signal completion. Rank 0 may have crashed or hung."
                    )
                if now > next_warn:
                    elapsed = now - (deadline - timeout)
                    logger.debug(
                        "barrier #%d  rank=%d  waiting for rank-0 phase flip  "
                        "elapsed=%.1fs",
                        self._barrier_count, self._rank, elapsed
                    )
                    next_warn = now + _BARRIER_WARN_INTERVAL
                time.sleep(0.0001)

        logger.debug("barrier #%d  rank=%d  done  sense=%d",
                     self._barrier_count, self._rank, expected)


class SharedMemorySync:
    """
    Manages shared memory regions for cross-process synchronization.

    Provides:
    - Barrier synchronization
    - Metadata slots for exchanging tensor info and IPC handles
    - Tail/head control blocks for chunked transfer progress tracking
    """

    def __init__(self, rank: int, world_size: int, session_id: str,
                 create: bool = False, timeout: float = 300.0):
        self._rank = rank
        self._world_size = world_size
        self._session_id = session_id
        self._timeout = timeout

        self._barrier = SharedBarrier(rank, world_size, session_id, create=create, timeout=timeout)

        # Metadata exchange: [seq_num (8 bytes)] + [world_size * _SLOT_SIZE]
        meta_size = 8 + world_size * _SLOT_SIZE
        meta_name = _shm_name(session_id, "meta")

        if create:
            self._meta_shm = shared_memory.SharedMemory(
                name=meta_name, create=True, size=meta_size
            )
            self._meta_shm.buf[:meta_size] = b'\x00' * meta_size
            logger.debug("SharedMemorySync meta created  name=%s  size=%d",
                         meta_name, meta_size)
        else:
            self._meta_shm = shared_memory.SharedMemory(
                name=meta_name, create=False
            )
            logger.debug("SharedMemorySync meta attached  name=%s", meta_name)

        self._meta_size = meta_size
        self._seq_num = 0

        # Control block: world_size × world_size × 64 bytes
        # [src_rank][dst_rank] → { tail(8), head(8), status(4), pad(44) }
        _CTRL_BLOCK_SIZE = 64
        ctrl_size = world_size * world_size * _CTRL_BLOCK_SIZE
        ctrl_name = _shm_name(session_id, "ctrl")

        if create:
            self._ctrl_shm = shared_memory.SharedMemory(
                name=ctrl_name, create=True, size=ctrl_size
            )
            self._ctrl_shm.buf[:ctrl_size] = b'\x00' * ctrl_size
            logger.debug("SharedMemorySync ctrl created  name=%s  size=%d",
                         ctrl_name, ctrl_size)
        else:
            self._ctrl_shm = shared_memory.SharedMemory(
                name=ctrl_name, create=False
            )
            logger.debug("SharedMemorySync ctrl attached  name=%s", ctrl_name)

        self._ctrl_size = ctrl_size

        atexit.register(self._cleanup_meta)
        atexit.register(self._cleanup_ctrl)

    def _cleanup_meta(self):
        try:
            self._meta_shm.close()
            if self._rank == 0:
                try:
                    self._meta_shm.unlink()
                except FileNotFoundError:
                    pass
        except Exception:
            pass

    def _cleanup_ctrl(self):
        try:
            self._ctrl_shm.close()
            if self._rank == 0:
                try:
                    self._ctrl_shm.unlink()
                except FileNotFoundError:
                    pass
        except Exception:
            pass

    def barrier(self, timeout: float | None = None):
        self._barrier.wait(timeout)

    def _slot_offset(self, rank: int) -> int:
        return 8 + rank * _SLOT_SIZE

    def write_slot(self, data: bytes):
        if len(data) > _SLOT_SIZE:
            raise ValueError(
                f"cuda_direct_backend: metadata slot overflow — "
                f"data size {len(data)} exceeds slot size {_SLOT_SIZE} bytes "
                f"(rank={self._rank}). Increase _SLOT_SIZE if larger payloads are needed."
            )
        offset = self._slot_offset(self._rank)
        self._meta_shm.buf[offset:offset + len(data)] = data

    def read_slot(self, rank: int, size: int) -> bytes:
        if rank < 0 or rank >= self._world_size:
            raise ValueError(
                f"cuda_direct_backend: read_slot called with invalid rank={rank} "
                f"(world_size={self._world_size})"
            )
        offset = self._slot_offset(rank)
        return bytes(self._meta_shm.buf[offset:offset + size])

    def write_ipc_handle(self, handle_bytes: bytes):
        if len(handle_bytes) != _IPC_HANDLE_SIZE:
            raise ValueError(
                f"cuda_direct_backend: IPC handle must be exactly {_IPC_HANDLE_SIZE} "
                f"bytes, got {len(handle_bytes)}"
            )
        self.write_slot(handle_bytes)

    def read_ipc_handle(self, rank: int) -> bytes:
        return self.read_slot(rank, _IPC_HANDLE_SIZE)

    def write_tensor_meta(self, numel: int, dtype_code: int, device_index: int,
                          ipc_handle: bytes = b'', ipc_offset: int = 0):
        """Write tensor metadata + optional IPC handle to this rank's slot.

        Layout:
          [0:8]   numel (int64)
          [8:12]  dtype_code (int32)
          [12:16] device_index (int32)
          [16:80] ipc_handle (64 bytes, optional)
          [80:88] ipc_offset (int64) — byte offset from IPC base to tensor data
        """
        meta = struct.pack('<qii', numel, dtype_code, device_index)
        if ipc_handle:
            meta += ipc_handle
        else:
            meta += b'\x00' * _IPC_HANDLE_SIZE
        meta += struct.pack('<q', ipc_offset)
        self.write_slot(meta)
        logger.debug("write_tensor_meta  rank=%d  numel=%d  device=%d  ipc_offset=%d",
                     self._rank, numel, device_index, ipc_offset)

    def read_tensor_meta(self, rank: int):
        """Read tensor metadata from a rank's slot.

        Returns: (numel, dtype_code, device_index, ipc_handle_bytes, ipc_offset)
        """
        data = self.read_slot(rank, _TENSOR_META_SIZE)
        numel, dtype_code, device_index = struct.unpack_from('<qii', data, 0)
        ipc_handle = data[16:16 + _IPC_HANDLE_SIZE]
        ipc_offset, = struct.unpack_from('<q', data, 16 + _IPC_HANDLE_SIZE)
        logger.debug("read_tensor_meta  from_rank=%d  numel=%d  device=%d  ipc_offset=%d",
                     rank, numel, device_index, ipc_offset)
        return numel, dtype_code, device_index, ipc_handle, ipc_offset

    def next_seq(self) -> int:
        self._seq_num += 1
        return self._seq_num

    def _ctrl_offset(self, src_rank: int, dst_rank: int) -> int:
        return (src_rank * self._world_size + dst_rank) * 64

    def write_tail(self, dst_rank: int, tail: int):
        """Publish sender progress (bytes transferred so far) to dst_rank."""
        offset = self._ctrl_offset(self._rank, dst_rank)
        struct.pack_into('<Q', self._ctrl_shm.buf, offset, tail)

    def read_tail(self, src_rank: int) -> int:
        """Read sender progress published by src_rank."""
        offset = self._ctrl_offset(src_rank, self._rank)
        return struct.unpack_from('<Q', self._ctrl_shm.buf, offset)[0]

    def write_head(self, src_rank: int, head: int):
        """Write consumer progress for data from src_rank."""
        offset = self._ctrl_offset(src_rank, self._rank)
        struct.pack_into('<Q', self._ctrl_shm.buf, offset + 8, head)

    def read_head(self, dst_rank: int) -> int:
        """Read consumer progress as seen by dst_rank."""
        offset = self._ctrl_offset(self._rank, dst_rank)
        return struct.unpack_from('<Q', self._ctrl_shm.buf, offset + 8)[0]

    def write_status(self, dst_rank: int, status: int):
        offset = self._ctrl_offset(self._rank, dst_rank)
        struct.pack_into('<I', self._ctrl_shm.buf, offset + 16, status)

    def read_status(self, src_rank: int) -> int:
        offset = self._ctrl_offset(src_rank, self._rank)
        return struct.unpack_from('<I', self._ctrl_shm.buf, offset + 16)[0]

    def write_split_sizes(self, split_sizes: list) -> None:
        """Write per-peer split size array to bytes [80:] of this rank's metadata slot.
        Maximum supported world_size is _SPLIT_SIZES_MAX (44 for the default
        _SLOT_SIZE of 256).
        """
        n = len(split_sizes)
        if n > _SPLIT_SIZES_MAX:
            raise ValueError(
                f"cuda_direct_backend: world_size={n} exceeds the maximum number of "
                f"split sizes that fit in the metadata slot ({_SPLIT_SIZES_MAX}). "
                f"Increase _SLOT_SIZE from {_SLOT_SIZE} to at least "
                f"{_TENSOR_META_SIZE + n * 4} bytes to support this world size."
            )
        data = struct.pack(f'<{n}I', *split_sizes)
        offset = self._slot_offset(self._rank) + _SPLIT_SIZES_OFFSET
        self._meta_shm.buf[offset:offset + len(data)] = data

    def read_split_sizes(self, rank: int, n: int) -> list:
        """Read n split sizes (int32) from bytes [80:] of a rank's metadata slot."""
        if n > _SPLIT_SIZES_MAX:
            raise ValueError(
                f"cuda_direct_backend: cannot read {n} split sizes from slot "
                f"(max {_SPLIT_SIZES_MAX} for _SLOT_SIZE={_SLOT_SIZE})."
            )
        offset = self._slot_offset(rank) + _SPLIT_SIZES_OFFSET
        data = bytes(self._meta_shm.buf[offset:offset + n * 4])
        return list(struct.unpack_from(f'<{n}I', data))
