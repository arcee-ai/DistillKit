"""Tests for CUDA IPC utilities."""

import pytest
import torch

from distillkit.cuda_ipc_utils import (
    compute_batch_hash,
    deserialize_cuda_tensor,
    serialize_cuda_tensor,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ipc_round_trip():
    """Test that we can serialize and deserialize a tensor via CUDA IPC."""
    device = torch.device("cuda:0")

    # Create test tensor
    original = torch.randn(4, 128, 32, dtype=torch.float32, device=device)

    # Serialize
    handle, metadata = serialize_cuda_tensor(original)

    # Deserialize
    reconstructed = deserialize_cuda_tensor(handle, metadata, target_device=device)

    # Verify they share memory (zero-copy)
    assert reconstructed.data_ptr() == original.data_ptr()

    # Verify values match
    assert torch.allclose(reconstructed, original)

    # Verify metadata
    assert metadata["shape"] == tuple(original.shape)
    assert metadata["dtype"] == original.dtype
    assert metadata["device"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ipc_different_dtypes():
    """Test CUDA IPC with different dtypes."""
    device = torch.device("cuda:0")

    for dtype in [torch.float16, torch.float32, torch.bfloat16, torch.long]:
        original = torch.randn(2, 64, dtype=dtype, device=device)
        handle, metadata = serialize_cuda_tensor(original)
        reconstructed = deserialize_cuda_tensor(handle, metadata, target_device=device)

        assert reconstructed.dtype == original.dtype
        assert torch.allclose(
            reconstructed.float(), original.float(), rtol=1e-3, atol=1e-5
        )


def test_cuda_ipc_non_cuda_tensor_fails():
    """Test that serializing non-CUDA tensor raises error."""
    cpu_tensor = torch.randn(4, 8)

    with pytest.raises(ValueError, match="must be on CUDA device"):
        serialize_cuda_tensor(cpu_tensor)


def test_batch_hash_determinism():
    """Test that batch hashing is deterministic."""
    input_ids = torch.randint(0, 1000, (4, 128))

    hash1 = compute_batch_hash(input_ids)
    hash2 = compute_batch_hash(input_ids)

    assert hash1 == hash2


def test_batch_hash_different_for_different_inputs():
    """Test that different inputs produce different hashes."""
    input_ids1 = torch.randint(0, 1000, (4, 128))
    input_ids2 = torch.randint(0, 1000, (4, 128))

    hash1 = compute_batch_hash(input_ids1)
    hash2 = compute_batch_hash(input_ids2)

    # With high probability, these should be different
    assert hash1 != hash2


def test_batch_hash_device_agnostic():
    """Test that batch hash is same regardless of device."""
    input_ids_cpu = torch.randint(0, 1000, (4, 128))

    hash_cpu = compute_batch_hash(input_ids_cpu)

    if torch.cuda.is_available():
        input_ids_gpu = input_ids_cpu.cuda()
        hash_gpu = compute_batch_hash(input_ids_gpu)
        assert hash_cpu == hash_gpu
