"""vLLM-based online signal source with CUDA IPC communication."""

import logging
import multiprocessing as mp
import uuid
from typing import Any

import torch
from typing_extensions import override

from distillkit.cuda_ipc_utils import compute_batch_hash, deserialize_cuda_tensor
from distillkit.signals import SignalSource, SparseSignal
from distillkit.vllm_server import TeacherRequest, TeacherResponse

LOG = logging.getLogger(__name__)


class VLLMOnlineSignalSource(SignalSource):
    """
    Signal source using vLLM teacher server with CUDA IPC.

    Communicates with a separate vLLM teacher server process via multiprocessing
    queues. Tensors are transferred via CUDA IPC for zero-copy performance.

    The signal source maintains a local cache of teacher outputs to handle cache
    hits without needing to contact the server. This provides natural "prefetching"
    behavior when batches repeat (e.g., during evaluation).
    """

    def __init__(
        self,
        vocab_size: int,
        top_k: int,
        server_process: mp.Process,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        shutdown_event: mp.Event,
        request_timeout_sec: float = 60.0,
    ):
        """
        Initialize vLLM online signal source.

        Args:
            vocab_size: Vocabulary size for signal
            top_k: Number of top-k logprobs (must match server config)
            server_process: Teacher server process handle
            request_queue: Queue for sending requests to server
            response_queue: Queue for receiving responses from server
            shutdown_event: Event to signal server shutdown
            request_timeout_sec: Timeout for waiting on server response
        """
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.server_process = server_process
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.shutdown_event = shutdown_event
        self.request_timeout_sec = request_timeout_sec

        # Local cache: batch_hash -> (sparse_ids, sparse_values)
        # These are cloned copies, not CUDA IPC tensors
        self.cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @override
    def supports_hidden_states(self) -> bool:
        """vLLM does not easily expose hidden states."""
        return False

    @override
    def get_signal(
        self, batch: dict[str, Any], return_hidden_states: bool = False
    ) -> SparseSignal:
        """
        Get teacher signal for a batch.

        Args:
            batch: Batch dict with 'input_ids' and optionally 'attention_mask'
            return_hidden_states: Whether to return hidden states (not supported)

        Returns:
            SparseSignal with top-k logprobs

        Raises:
            RuntimeError: If hidden states requested or server fails
        """
        if return_hidden_states:
            raise RuntimeError("vLLM teacher source does not support hidden states")

        input_ids = batch["input_ids"]

        # Check cache
        batch_hash = compute_batch_hash(input_ids)
        if batch_hash in self.cache:
            self.cache_hits += 1
            sparse_ids, sparse_values = self.cache[batch_hash]

            return SparseSignal(
                sparse_ids=sparse_ids,
                sparse_values=sparse_values,
                log_values=True,  # vLLM returns logprobs
                generation_temperature=1.0,
                hidden_states=None,
                vocab_size=self.vocab_size,
            )

        # Cache miss - request from server
        self.cache_misses += 1

        # Create request
        request = TeacherRequest(
            request_id=str(uuid.uuid4()),
            input_ids=input_ids.cpu().tolist(),
            batch_hash=batch_hash,
            attention_mask=(
                batch["attention_mask"].cpu().tolist()
                if "attention_mask" in batch
                else None
            ),
        )

        # Send request
        self.request_queue.put(request)

        # Wait for response (blocking)
        try:
            response: TeacherResponse = self.response_queue.get(
                timeout=self.request_timeout_sec
            )
        except mp.queues.Empty:
            # Check if server crashed
            if not self.server_process.is_alive():
                raise RuntimeError(
                    "Teacher server process died. Check server logs for details."
                )
            raise RuntimeError(
                f"Teacher server timeout after {self.request_timeout_sec}s. "
                f"Server may be overloaded or crashed."
            )

        if not response.success:
            raise RuntimeError(
                f"Teacher inference failed: {response.error_message}"
            )

        # Reconstruct tensors from CUDA IPC handles
        device = input_ids.device

        sparse_ids = deserialize_cuda_tensor(
            response.sparse_ids_handle,
            response.sparse_ids_metadata,
            target_device=device,
        )

        sparse_values = deserialize_cuda_tensor(
            response.sparse_values_handle,
            response.sparse_values_metadata,
            target_device=device,
        )

        # Clone tensors for cache (break IPC link)
        # This is important - we don't want to keep IPC handles alive indefinitely
        sparse_ids_cached = sparse_ids.clone()
        sparse_values_cached = sparse_values.clone()

        # Update cache with cloned tensors
        self._update_cache(batch_hash, sparse_ids_cached, sparse_values_cached)

        return SparseSignal(
            sparse_ids=sparse_ids,
            sparse_values=sparse_values,
            log_values=True,
            generation_temperature=1.0,
            hidden_states=None,
            vocab_size=self.vocab_size,
        )

    def _update_cache(
        self, batch_hash: str, sparse_ids: torch.Tensor, sparse_values: torch.Tensor
    ):
        """
        Update local cache with teacher outputs.

        Simple LRU: limit number of cache entries. Could be more sophisticated
        (e.g., based on memory usage) but this is sufficient for now.
        """
        # Limit cache size to prevent unbounded memory growth
        MAX_CACHE_ENTRIES = 100

        if len(self.cache) >= MAX_CACHE_ENTRIES:
            # Evict oldest (first inserted, since dict is ordered in Python 3.7+)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[batch_hash] = (sparse_ids, sparse_values)

    def shutdown(self):
        """
        Shutdown teacher server gracefully.

        Should be called when training completes to clean up resources.
        """
        hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0.0
        )
        LOG.info(
            f"vLLM teacher stats: {self.cache_hits} hits, {self.cache_misses} misses "
            f"(hit rate: {hit_rate:.1%})"
        )

        LOG.info("Shutting down teacher server...")
        self.shutdown_event.set()

        # Wait for server to exit gracefully
        self.server_process.join(timeout=5.0)

        if self.server_process.is_alive():
            LOG.warning("Teacher server did not shutdown gracefully, terminating")
            self.server_process.terminate()
            self.server_process.join(timeout=2.0)

        if self.server_process.is_alive():
            LOG.error("Teacher server did not terminate, killing")
            self.server_process.kill()

        LOG.info("Teacher server shutdown complete")
