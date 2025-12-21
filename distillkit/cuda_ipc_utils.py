"""CUDA IPC utilities for zero-copy tensor sharing between processes."""

import hashlib
from typing import Any

import torch


def serialize_cuda_tensor(tensor: torch.Tensor) -> tuple[Any, dict[str, Any]]:
    """
    Serialize a CUDA tensor for IPC transfer via shared memory.

    CRITICAL: The tensor MUST remain alive in the sender process until the
    receiver has finished using it. The sender should maintain a reference
    to the tensor (e.g., in a cache) until the receiver confirms receipt or
    a timeout expires.

    Args:
        tensor: CUDA tensor to serialize

    Returns:
        (ipc_handle, metadata) tuple where:
        - ipc_handle: CUDA IPC handle (can be pickled and sent via queue)
        - metadata: Dict with shape, dtype, device, stride, etc.

    Raises:
        ValueError: If tensor is not on a CUDA device
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA device for IPC")

    # Get CUDA IPC handle from tensor storage
    # This creates a handle that can be shared across processes
    ipc_handle = tensor.storage()._share_cuda_()

    metadata = {
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "device": tensor.device.index,
        "stride": tensor.stride(),
        "storage_offset": tensor.storage_offset(),
        "numel": tensor.numel(),
    }

    return ipc_handle, metadata


def deserialize_cuda_tensor(
    ipc_handle: Any, metadata: dict[str, Any], target_device: torch.device | None = None
) -> torch.Tensor:
    """
    Reconstruct a tensor from a CUDA IPC handle.

    The reconstructed tensor shares memory with the sender's tensor - this is
    zero-copy transfer. Modifications to the reconstructed tensor will be
    visible to the sender (though in our use case we only read).

    Args:
        ipc_handle: CUDA IPC handle from sender
        metadata: Tensor metadata (shape, dtype, etc.)
        target_device: Device to place tensor on (uses metadata device if None)

    Returns:
        Reconstructed tensor sharing memory with sender

    Raises:
        RuntimeError: If IPC handle is invalid or tensor reconstruction fails
    """
    # Determine target device
    if target_device is None:
        target_device = torch.device(f"cuda:{metadata['device']}")

    # Calculate storage size in bytes
    dtype_size = torch._utils._element_size(metadata["dtype"])
    storage_size = metadata["numel"] * dtype_size

    # Reconstruct storage from IPC handle
    # This is the critical zero-copy operation - creates a storage object
    # that references the same GPU memory as the sender's tensor
    storage = torch.UntypedStorage._new_shared_cuda(
        device=target_device,
        handle=ipc_handle,
        size=storage_size,
        dtype=metadata["dtype"],
    )

    # Create typed storage
    typed_storage = torch.storage.TypedStorage(
        wrap_storage=storage,
        dtype=metadata["dtype"],
    )

    # Reconstruct tensor from storage with original layout
    tensor = torch.tensor(
        [],
        dtype=metadata["dtype"],
        device=target_device,
    ).set_(
        typed_storage,
        storage_offset=metadata["storage_offset"],
        size=metadata["shape"],
        stride=metadata["stride"],
    )

    return tensor


def compute_batch_hash(input_ids: torch.Tensor) -> str:
    """
    Compute a deterministic hash for a batch of input_ids for caching.

    Args:
        input_ids: Input token IDs tensor [batch_size, seq_len]

    Returns:
        Hexadecimal hash string
    """
    # Convert to bytes (handles variable batch sizes and sequence lengths)
    # Use CPU tensor to ensure consistent hashing regardless of device
    data = input_ids.cpu().numpy().tobytes()

    # Use BLAKE2b for speed (faster than SHA256, cryptographic strength not needed)
    # 16-byte digest is sufficient for cache key collision resistance
    return hashlib.blake2b(data, digest_size=16).hexdigest()
