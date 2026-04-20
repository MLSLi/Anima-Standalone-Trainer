"""
CUDA IPC utilities using ctypes to access cudart API.
Allows passing GPU device pointers between processes on Windows.
"""

import ctypes
import logging
import os
import torch

logger = logging.getLogger("cuda_direct.ipc")

_cudart = None

# Try platform-appropriate library names
if os.name == "nt":
    _lib_names = ["cudart64_12.dll", "cudart64_110.dll", "cudart64_11.dll"]
    _lib_glob = "cudart64_*.dll"
else:
    _lib_names = ["libcudart.so.12", "libcudart.so.11.0", "libcudart.so"]
    _lib_glob = "libcudart.so*"

for lib_name in _lib_names:
    try:
        _cudart = ctypes.cdll.LoadLibrary(lib_name)
        logger.debug("Loaded CUDA runtime: %s", lib_name)
        break
    except OSError:
        pass

if _cudart is None:
    # Fallback: search inside torch's lib directory
    import glob
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    for lib_path in glob.glob(os.path.join(torch_lib_dir, _lib_glob)):
        try:
            _cudart = ctypes.cdll.LoadLibrary(lib_path)
            logger.debug("Loaded CUDA runtime from torch/lib: %s", lib_path)
            break
        except OSError:
            pass

if _cudart is None:
    raise RuntimeError(
        "cuda-direct-backend: Could not load CUDA runtime library. "
        f"Searched for: {_lib_names} and in "
        f"{os.path.join(os.path.dirname(torch.__file__), 'lib')}"
    )

cudaSuccess = 0
cudaIpcMemLazyEnablePeerAccess = 1
cudaErrorPeerAccessAlreadyEnabled = 704
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3

# Human-readable names for the most common CUDA error codes.
# Full list: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
_CUDA_ERROR_NAMES = {
    1:   "cudaErrorInvalidValue",
    2:   "cudaErrorMemoryAllocation",
    3:   "cudaErrorInitializationError",
    4:   "cudaErrorCudartUnloading",
    6:   "cudaErrorProfilerDisabled",
    35:  "cudaErrorInsufficientDriver",
    36:  "cudaErrorCallRequiresNewerDriver",
    37:  "cudaErrorInvalidSurface",
    46:  "cudaErrorDuplicateSurfaceName",
    47:  "cudaErrorAllDevicesSupportP2P",  # not an error but useful
    48:  "cudaErrorInvalidKernelImage",
    49:  "cudaErrorDeviceUninitilialized",
    50:  "cudaErrorMapBufferObjectFailed",
    51:  "cudaErrorUnmapBufferObjectFailed",
    52:  "cudaErrorArrayIsMapped",
    53:  "cudaErrorAlreadyMapped",
    54:  "cudaErrorNoKernelImageForDevice",
    55:  "cudaErrorAlreadyAcquired",
    56:  "cudaErrorNotYetReady",
    57:  "cudaErrorOperatingSystem",
    58:  "cudaErrorInvalidResourceHandle",
    59:  "cudaErrorIllegalState",
    60:  "cudaErrorSymbolNotFound",
    63:  "cudaErrorNotReady",
    65:  "cudaErrorIllegalAddress",
    74:  "cudaErrorInvalidPtx",
    75:  "cudaErrorInvalidGraphicsContext",
    76:  "cudaErrorNvlinkUncorrectable",
    77:  "cudaErrorJitCompilerNotFound",
    78:  "cudaErrorInvalidSource",
    79:  "cudaErrorFileNotFound",
    80:  "cudaErrorSharedObjectSymbolNotFound",
    81:  "cudaErrorSharedObjectInitFailed",
    82:  "cudaErrorOperatingSystem2",
    100: "cudaErrorInvalidDevice",
    101: "cudaErrorStartupFailure",
    200: "cudaErrorInvalidKernelImage2",
    201: "cudaErrorDeviceNotLicensed",  # also cudaErrorInvalidDevice on some versions
    205: "cudaErrorUnsupportedLimit",
    206: "cudaErrorPeerAccessUnsupported",
    207: "cudaErrorInvalidPtx2",
    208: "cudaErrorInvalidGraphicsContext2",
    214: "cudaErrorHostMemoryAlreadyRegistered",
    215: "cudaErrorHostMemoryNotRegistered",
    216: "cudaErrorHardwareStackError",
    217: "cudaErrorIllegalInstruction",
    218: "cudaErrorMisalignedAddress",
    219: "cudaErrorInvalidAddressSpace",
    220: "cudaErrorInvalidPc",
    221: "cudaErrorLaunchFailure",
    222: "cudaErrorCooperativeLaunchTooLarge",
    300: "cudaErrorNotPermitted",
    301: "cudaErrorNotSupported",
    302: "cudaErrorSystemNotReady",
    303: "cudaErrorSystemDriverMismatch",
    304: "cudaErrorCompatNotSupportedOnDevice",
    400: "cudaErrorStreamCaptureUnsupported",
    401: "cudaErrorStreamCaptureInvalidated",
    402: "cudaErrorStreamCaptureMerge",
    403: "cudaErrorStreamCaptureUnmatched",
    404: "cudaErrorStreamCaptureUnjoined",
    405: "cudaErrorStreamCaptureIsolation",
    406: "cudaErrorStreamCaptureImplicit",
    407: "cudaErrorCapturedEvent",
    408: "cudaErrorStreamCaptureWrongThread",
    409: "cudaErrorTimeout",
    410: "cudaErrorGraphExecUpdateFailure",
    500: "cudaErrorUnknown",
    700: "cudaErrorAssert",
    701: "cudaErrorTooManyPeers",
    702: "cudaErrorHostMemoryAlreadyRegistered2",
    703: "cudaErrorHostMemoryNotRegistered2",
    704: "cudaErrorPeerAccessAlreadyEnabled",
    705: "cudaErrorPeerAccessNotEnabled",
    708: "cudaErrorSetOnActiveProcess",
    709: "cudaErrorContextIsDestroyed",
    710: "cudaErrorDeviceAlreadyInUse",
    712: "cudaErrorProfilerDisabled2",
    713: "cudaErrorProfilerNotInitialized",
    714: "cudaErrorProfilerAlreadyStarted",
    715: "cudaErrorProfilerAlreadyStopped",
    800: "cudaErrorAssert2",
    900: "cudaErrorExternalDevice",
    999: "cudaErrorApiFailureBase",
}

# Extra hints for errors commonly seen in IPC / P2P workflows
_CUDA_ERROR_HINTS = {
    1: (
        "cudaErrorInvalidValue usually means a NULL or out-of-bounds pointer was passed "
        "to cudaMemcpyAsync. In IPC mode this often happens when cudaIpcOpenMemHandle "
        "returned a stale/invalid handle (e.g. the peer tensor was reallocated) or when "
        "pointer arithmetic on an IPC-mapped address goes out of the allocation bounds."
    ),
    201: (
        "Error 201 from cudaIpcOpenMemHandle means the handle was created by the same "
        "CUDA context (same process). IPC handles are strictly cross-process — you cannot "
        "open your own handle. This happens in single-GPU multi-process simulations."
    ),
    206: (
        "cudaErrorPeerAccessUnsupported: the two GPUs cannot do direct P2P transfers. "
        "They are likely on different PCIe root complexes (different NUMA nodes). "
        "Consider setting NCCL_P2P_DISABLE=1 or using a CPU-bridge fallback."
    ),
    704: (
        "cudaErrorPeerAccessAlreadyEnabled: cudaDeviceEnablePeerAccess was called more "
        "than once for the same peer. This is harmless and already handled."
    ),
    409: (
        "cudaErrorTimeout: a CUDA operation did not complete in time. "
        "Check for deadlocks in barrier synchronization."
    ),
}


def _cuda_error_str(code: int) -> str:
    name = _CUDA_ERROR_NAMES.get(code, f"cudaError<{code}>")
    hint = _CUDA_ERROR_HINTS.get(code)
    if hint:
        return f"{name} (code {code}): {hint}"
    return f"{name} (code {code})"


def check_cuda_error(res: int, func_name: str, **context):
    """Raise RuntimeError with a human-readable message if res != cudaSuccess.

    Args:
        res:       Return value from a cudart function.
        func_name: Name of the cudart function that was called.
        **context: Extra key=value pairs included in the error message
                   (e.g. src_ptr=0x..., dst_ptr=0x..., size=1234).
    """
    if res == cudaSuccess:
        return
    ctx_str = "  ".join(f"{k}={v}" for k, v in context.items()) if context else ""
    error_msg = (
        f"cuda_direct_backend: {func_name} failed — {_cuda_error_str(res)}"
        + (f"\n  Context: {ctx_str}" if ctx_str else "")
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


class CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 64)]

_cudart.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(CudaIpcMemHandle), ctypes.c_void_p]
_cudart.cudaIpcGetMemHandle.restype = ctypes.c_int

# cudaGetLastError clears the per-thread CUDA error state and returns it.
# Call this after any failed CUDA API to prevent the stale error from being
# picked up later by PyTorch's StorageImpl destructor (which calls a CUDA sync
# check on tensor free), causing a deferred SIGABRT through c10_cuda_check.
_cudart.cudaGetLastError.argtypes = []
_cudart.cudaGetLastError.restype = ctypes.c_int

_cudart.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), CudaIpcMemHandle, ctypes.c_uint]
_cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int

_cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
_cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int

_cudart.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
_cudart.cudaDeviceEnablePeerAccess.restype = ctypes.c_int

_cudart.cudaDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_cudart.cudaDeviceCanAccessPeer.restype = ctypes.c_int

_cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
_cudart.cudaMemcpyAsync.restype = ctypes.c_int

_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int

_cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudart.cudaHostUnregister.restype = ctypes.c_int

_cudart.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
_cudart.cudaHostGetDevicePointer.restype = ctypes.c_int

# cudaHostRegister flags
cudaHostRegisterPortable = 0x01   # mapping visible to all CUDA contexts (cross-process)
cudaHostRegisterMapped   = 0x02   # map host address into device address space


# IPC handle cache: data_ptr -> (handle_bytes, ipc_offset_bytes)
#
# IMPORTANT: cudaIpcGetMemHandle requires the **base pointer** of a cudaMalloc
# allocation.  PyTorch's caching allocator sub-allocates from large cudaMalloc
# blocks, so a tensor's data_ptr() is almost never the block's base address.
# Passing a sub-allocation pointer to cudaIpcGetMemHandle returns
# cudaErrorInvalidValue.
#
# We therefore use storage._share_cuda_() (the same mechanism PyTorch's
# multiprocessing uses) which returns the IPC handle for the underlying
# cudaMalloc block *plus* the byte offset from that block to the tensor's data.
# The receiver opens the handle (gets the base pointer) then adds the offset.
#
# FSDP/DDP typically reuse the same tensor buffers across steps, so caching
# avoids repeated _share_cuda_() calls (~0.01 ms each).
_ipc_handle_cache: dict[int, tuple[bytes, int]] = {}
_ipc_cache_hits: int = 0
_ipc_cache_misses: int = 0


def invalidate_ipc_handle_cache() -> None:
    """Clear the IPC handle cache. Called before each collective by
    ProcessGroupCudaDirect.invalidate_cache() to handle tensor
    reallocation between forward/backward passes."""
    global _ipc_cache_hits, _ipc_cache_misses
    n = len(_ipc_handle_cache)
    if n:
        logger.debug("IPC handle cache invalidated  entries=%d  hits=%d  misses=%d",
                     n, _ipc_cache_hits, _ipc_cache_misses)
    _ipc_handle_cache.clear()
    _ipc_cache_hits = 0
    _ipc_cache_misses = 0


def get_ipc_handle_and_offset(tensor: torch.Tensor) -> tuple[bytes, int]:
    """Return (ipc_handle_bytes, byte_offset) for a CUDA tensor.

    Uses storage._share_cuda_() so that sub-allocations from PyTorch's caching
    allocator work correctly.  The returned handle is for the underlying
    cudaMalloc block; byte_offset is how far into that block the tensor's data
    starts.  The receiver must open the handle and add byte_offset before any
    DMA operation.

    Results are cached by data_ptr() so repeated calls within one collective
    are free.
    """
    global _ipc_cache_hits, _ipc_cache_misses
    if not tensor.is_cuda:
        raise ValueError(
            f"cuda_direct_backend: get_ipc_handle_and_offset requires a CUDA tensor, "
            f"got device='{tensor.device}'"
        )
    ptr = tensor.data_ptr()

    cached = _ipc_handle_cache.get(ptr)
    if cached is not None:
        _ipc_cache_hits += 1
        return cached

    _ipc_cache_misses += 1
    try:
        storage = tensor.untyped_storage()
        # _share_cuda_() -> (device, handle_bytes, storage_size, storage_offset_bytes,
        #                     ref_counter_handle, ref_counter_offset,
        #                     event_handle, event_sync_required)
        _, handle_bytes, _, storage_offset_bytes, *_ = storage._share_cuda_()
    except Exception as exc:
        raise RuntimeError(
            f"cuda_direct_backend: storage._share_cuda_() failed for tensor "
            f"ptr={hex(ptr)} device={tensor.device} numel={tensor.numel()} "
            f"dtype={tensor.dtype}: {exc}"
        ) from exc

    # Tensor may be a storage slice (e.g. a parameter view).
    tensor_byte_offset = tensor.storage_offset() * tensor.element_size()
    total_offset = storage_offset_bytes + tensor_byte_offset

    result = (handle_bytes, total_offset)
    _ipc_handle_cache[ptr] = result
    logger.debug(
        "get_ipc_handle_and_offset OK  ptr=%s  storage_offset=%d  tensor_offset=%d  "
        "total_offset=%d  numel=%d  dtype=%s",
        hex(ptr), storage_offset_bytes, tensor_byte_offset, total_offset,
        tensor.numel(), tensor.dtype,
    )
    return result


def get_ipc_handle(tensor: torch.Tensor) -> bytes:
    """Compatibility wrapper"""
    handle, _offset = get_ipc_handle_and_offset(tensor)
    return handle


def open_ipc_handle(handle_bytes: bytes, device_index: int) -> int:
    handle = CudaIpcMemHandle()
    ctypes.memmove(ctypes.addressof(handle.internal), handle_bytes, 64)
    out_ptr = ctypes.c_void_p()
    res = _cudart.cudaIpcOpenMemHandle(
        ctypes.byref(out_ptr), handle,
        ctypes.c_uint(cudaIpcMemLazyEnablePeerAccess)
    )
    check_cuda_error(res, "cudaIpcOpenMemHandle", device_index=device_index)
    ptr_val = out_ptr.value
    logger.debug("cudaIpcOpenMemHandle OK  device=%d  mapped_ptr=%s",
                 device_index, hex(ptr_val) if ptr_val else "NULL")
    if ptr_val is None or ptr_val == 0:
        raise RuntimeError(
            f"cuda_direct_backend: cudaIpcOpenMemHandle returned a NULL pointer "
            f"for device_index={device_index}. The IPC handle may be stale or the "
            f"peer tensor was freed/reallocated since the handle was created."
        )
    return ptr_val


def close_ipc_handle(ptr_val: int):
    if ptr_val == 0:
        return
    res = _cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr_val))
    if res != cudaSuccess:
        # Log but don't raise — close is best-effort cleanup
        logger.warning("cudaIpcCloseMemHandle failed for ptr=%s — %s",
                       hex(ptr_val), _cuda_error_str(res))


def can_access_peer(device_id: int, peer_device_id: int) -> bool:
    can_access = ctypes.c_int()
    res = _cudart.cudaDeviceCanAccessPeer(ctypes.byref(can_access), device_id, peer_device_id)
    check_cuda_error(res, "cudaDeviceCanAccessPeer",
                     device=device_id, peer=peer_device_id)
    result = can_access.value == 1
    logger.debug("cudaDeviceCanAccessPeer  device=%d  peer=%d  result=%s",
                 device_id, peer_device_id, result)
    return result


def enable_peer_access(peer_device_id: int):
    res = _cudart.cudaDeviceEnablePeerAccess(peer_device_id, 0)
    if res == cudaSuccess:
        logger.debug("cudaDeviceEnablePeerAccess OK  peer=%d", peer_device_id)
        return
    if res == cudaErrorPeerAccessAlreadyEnabled:
        logger.debug("cudaDeviceEnablePeerAccess already enabled for peer=%d", peer_device_id)
        return
    check_cuda_error(res, "cudaDeviceEnablePeerAccess", peer=peer_device_id)


def host_register(host_ptr: int, size: int) -> bool:
    """
    Register a host memory region with CUDA so the GPU DMA engine can access it
    directly without a staging buffer (zero-copy).

    Flags: cudaHostRegisterPortable | cudaHostRegisterMapped
      - Portable: mapping is visible to all CUDA contexts (required for cross-process SHM)
      - Mapped:   maps the address into the device address space

    Returns True on success, False if registration failed (OOM, permissions, etc.).
    Callers should fall back to the double-bounce path on False.
    """
    if size == 0:
        return False
    flags = ctypes.c_uint(cudaHostRegisterPortable | cudaHostRegisterMapped)
    res = _cudart.cudaHostRegister(ctypes.c_void_p(host_ptr), ctypes.c_size_t(size), flags)
    if res == cudaSuccess:
        logger.debug("cudaHostRegister OK  ptr=%s  size=%d", hex(host_ptr), size)
        return True
    # 214 = already registered — treat as success
    if res == 214:
        logger.debug("cudaHostRegister already registered  ptr=%s", hex(host_ptr))
        return True
    logger.warning("cudaHostRegister failed (code %d) — falling back to double-bounce  ptr=%s  size=%d",
                   res, hex(host_ptr), size)
    return False


def host_unregister(host_ptr: int) -> None:
    """Unregister a previously registered host memory region. Best-effort."""
    if host_ptr == 0:
        return
    res = _cudart.cudaHostUnregister(ctypes.c_void_p(host_ptr))
    if res != cudaSuccess and res != 215:  # 215 = not registered (already cleaned up)
        logger.warning("cudaHostUnregister failed (code %d)  ptr=%s", res, hex(host_ptr))


def host_get_device_pointer(host_ptr: int) -> int:
    """
    Return the device-mapped address for a registered host pointer.
    Returns 0 if the lookup fails.
    This device pointer can be used directly in cudaMemcpyAsync as src or dst.
    """
    dev_ptr = ctypes.c_void_p(0)
    res = _cudart.cudaHostGetDevicePointer(ctypes.byref(dev_ptr), ctypes.c_void_p(host_ptr), 0)
    if res != cudaSuccess:
        logger.warning("cudaHostGetDevicePointer failed (code %d)  host=%s", res, hex(host_ptr))
        return 0
    val = dev_ptr.value or 0
    logger.debug("cudaHostGetDevicePointer  host=%s  dev=%s", hex(host_ptr), hex(val))
    return val


def memcpy_d2h_async(dst_host_ptr: int, src_gpu_ptr: int, size: int, stream_ptr: int):
    """Async DMA: GPU → pinned host memory."""
    if size == 0:
        return
    res = _cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst_host_ptr),
        ctypes.c_void_p(src_gpu_ptr),
        ctypes.c_size_t(size),
        cudaMemcpyDeviceToHost,
        ctypes.c_void_p(stream_ptr)
    )
    check_cuda_error(res, "cudaMemcpyAsync D2H",
                     dst=hex(dst_host_ptr) if dst_host_ptr else "NULL",
                     src=hex(src_gpu_ptr) if src_gpu_ptr else "NULL",
                     size_bytes=size)


def memcpy_h2d_async(dst_gpu_ptr: int, src_host_ptr: int, size: int, stream_ptr: int):
    """Async DMA: pinned host memory → GPU."""
    if size == 0:
        return
    res = _cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst_gpu_ptr),
        ctypes.c_void_p(src_host_ptr),
        ctypes.c_size_t(size),
        cudaMemcpyHostToDevice,
        ctypes.c_void_p(stream_ptr)
    )
    check_cuda_error(res, "cudaMemcpyAsync H2D",
                     dst=hex(dst_gpu_ptr) if dst_gpu_ptr else "NULL",
                     src=hex(src_host_ptr) if src_host_ptr else "NULL",
                     size_bytes=size)


def memcpy_async(dst_ptr: int, src_ptr: int, size: int, stream_ptr: int):
    if size == 0:
        logger.debug("memcpy_async skipped — size=0")
        return
    res = _cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(size),
        cudaMemcpyDeviceToDevice,
        ctypes.c_void_p(stream_ptr)
    )
    check_cuda_error(res, "cudaMemcpyAsync",
                     dst=hex(dst_ptr) if dst_ptr else "NULL",
                     src=hex(src_ptr) if src_ptr else "NULL",
                     size_bytes=size,
                     size_mb=f"{size/1e6:.2f}")
    logger.debug("cudaMemcpyAsync queued  src=%s  dst=%s  size=%.2fMB",
                 hex(src_ptr), hex(dst_ptr), size / 1e6)
