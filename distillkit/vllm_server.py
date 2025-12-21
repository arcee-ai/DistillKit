"""vLLM teacher server for online distillation with CUDA IPC."""

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

try:
    import vllm
    from vllm.logprobs import FlatLogprobs, PromptLogprobs
except ImportError:
    raise ImportError(
        "vLLM must be installed to use the vLLM teacher. "
        "Install with: pip install vllm"
    )

from distillkit.cuda_ipc_utils import serialize_cuda_tensor

LOG = logging.getLogger(__name__)


@dataclass
class TeacherRequest:
    """Request format for teacher inference."""

    request_id: str  # Unique ID for tracking
    input_ids: list[list[int]]  # Batch of sequences (ragged)
    batch_hash: str  # Cache key
    attention_mask: list[list[int]] | None = None


@dataclass
class TeacherResponse:
    """Response format from teacher server."""

    request_id: str
    batch_hash: str

    # CUDA IPC handles for zero-copy transfer
    sparse_ids_handle: Any
    sparse_values_handle: Any

    # Metadata for tensor reconstruction
    sparse_ids_metadata: dict[str, Any]
    sparse_values_metadata: dict[str, Any]

    # Signal metadata
    vocab_size: int
    top_k: int

    # Error handling
    success: bool = True
    error_message: str | None = None


def process_prompt_logprobs(
    prompt_logprobs: PromptLogprobs, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract top-k logprobs from vLLM output.

    Adapted from sample_logits_vllm.py::process_prompt_logprobs() but returns
    tensors on the specified GPU device instead of CPU.

    Args:
        prompt_logprobs: vLLM prompt logprobs output
        k: Number of top-k logprobs
        device: CUDA device to place tensors on

    Returns:
        (top_indices, top_values) tuple of tensors on GPU
    """
    # Fast path: FlatLogprobs (modern vLLM)
    if isinstance(prompt_logprobs, FlatLogprobs):
        # Skip first position if empty (first token has no logprobs)
        start_pos = 0
        if len(prompt_logprobs) > 0:
            first_start = prompt_logprobs.start_indices[0]
            first_end = prompt_logprobs.end_indices[0]
            if first_end - first_start == 0:
                start_pos = 1

        num_prompt_tokens = len(prompt_logprobs) - start_pos
        if num_prompt_tokens <= 0:
            return torch.empty((0, 0), dtype=torch.long, device=device), torch.empty(
                (0, 0), dtype=torch.float32, device=device
            )

        # Allocate on target GPU device directly
        top_indices = torch.zeros((num_prompt_tokens, k), dtype=torch.long, device=device)
        top_values = torch.full(
            (num_prompt_tokens, k),
            fill_value=float("-inf"),
            dtype=torch.float32,
            device=device,
        )

        # Build index arrays for vectorized assignment
        seq_ids = []
        rank_ids = []
        token_ids_to_copy = []
        logprobs_to_copy = []

        for pos_id in range(start_pos, len(prompt_logprobs)):
            seq_id = pos_id - start_pos
            start_idx = prompt_logprobs.start_indices[pos_id]
            end_idx = prompt_logprobs.end_indices[pos_id]

            for i in range(start_idx, end_idx):
                rank = prompt_logprobs.ranks[i]
                if rank is None or rank > k:
                    # None: actual prompt token when not in top-k
                    # rank > k: truncate to k values requested
                    continue
                seq_ids.append(seq_id)
                rank_ids.append(rank - 1)  # ranks are 1-indexed
                token_ids_to_copy.append(prompt_logprobs.token_ids[i])
                logprobs_to_copy.append(prompt_logprobs.logprobs[i])

        # Vectorized assignment
        if seq_ids:
            seq_idx_tensor = torch.tensor(seq_ids, dtype=torch.long, device=device)
            rank_idx_tensor = torch.tensor(rank_ids, dtype=torch.long, device=device)
            top_indices[seq_idx_tensor, rank_idx_tensor] = torch.tensor(
                token_ids_to_copy, dtype=torch.long, device=device
            )
            top_values[seq_idx_tensor, rank_idx_tensor] = torch.tensor(
                logprobs_to_copy, dtype=torch.float32, device=device
            )

        return top_indices, top_values

    # Slow path: legacy list format (older vLLM versions)
    else:
        valid_logprobs = [lp for lp in prompt_logprobs]
        if valid_logprobs and (
            valid_logprobs[0] is None or len(valid_logprobs[0]) < 1
        ):
            valid_logprobs.pop(0)

        if not valid_logprobs:
            return torch.empty((0, 0), dtype=torch.long, device=device), torch.empty(
                (0, 0), dtype=torch.float32, device=device
            )

        num_prompt_tokens = len(valid_logprobs)
        top_indices = torch.zeros((num_prompt_tokens, k), dtype=torch.long, device=device)
        top_values = torch.full(
            (num_prompt_tokens, k),
            fill_value=float("-inf"),
            dtype=torch.float32,
            device=device,
        )

        for seq_id, logprobs in enumerate(valid_logprobs):
            if logprobs is None:
                continue
            for tok_id, logprob in logprobs.items():
                if logprob.rank is None or logprob.rank > k:
                    continue
                top_indices[seq_id, logprob.rank - 1] = tok_id
                top_values[seq_id, logprob.rank - 1] = logprob.logprob

        return top_indices, top_values


class VLLMTeacherServer:
    """
    Teacher server running vLLM on dedicated GPUs.

    Runs in a separate process spawned via multiprocessing. Processes teacher
    inference requests from training ranks and returns CUDA IPC handles for
    zero-copy tensor transfer.

    GPU isolation: The server sets CUDA_VISIBLE_DEVICES to its assigned GPUs
    before initializing CUDA, ensuring it doesn't interfere with student training.
    """

    def __init__(
        self,
        config: dict[str, Any],
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        ready_event: mp.Event,
        shutdown_event: mp.Event,
    ):
        """
        Initialize teacher server.

        Args:
            config: Serialized TeacherVLLMConfig dict
            request_queue: Queue for receiving inference requests
            response_queue: Queue for sending responses
            ready_event: Event to signal when server is ready
            shutdown_event: Event to signal shutdown request
        """
        self.config = config
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event

        self.llm: vllm.LLM | None = None
        self.sampling_params: vllm.SamplingParams | None = None
        self.vocab_size: int = 0

        # Tensor cache to keep tensors alive for IPC
        # Maps batch_hash -> (sparse_ids, sparse_values, timestamp)
        self.tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        self.cache_size_limit = config["cache_size_mb"] * 1024 * 1024

    def run(self):
        """Main server loop. This is the entry point for the server process."""
        try:
            self._initialize_vllm()
            self.ready_event.set()
            LOG.info("vLLM teacher server ready and listening for requests")

            while not self.shutdown_event.is_set():
                try:
                    # Non-blocking get with timeout to check shutdown event
                    request = self.request_queue.get(timeout=0.1)
                    response = self._process_request(request)
                    self.response_queue.put(response)
                except mp.queues.Empty:
                    continue
                except Exception as e:
                    LOG.error(f"Error processing request: {e}", exc_info=True)
                    # Send error response
                    error_response = TeacherResponse(
                        request_id="unknown",
                        batch_hash="",
                        sparse_ids_handle=None,
                        sparse_values_handle=None,
                        sparse_ids_metadata={},
                        sparse_values_metadata={},
                        vocab_size=self.vocab_size,
                        top_k=self.config["top_k"],
                        success=False,
                        error_message=str(e),
                    )
                    self.response_queue.put(error_response)

        except Exception as e:
            LOG.error(f"Fatal error in teacher server: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()

    def _initialize_vllm(self):
        """
        Initialize vLLM engine on teacher GPUs.

        CRITICAL: Sets CUDA_VISIBLE_DEVICES before any CUDA initialization to
        isolate teacher from student GPUs. This only works because we use
        spawn context for multiprocessing, which starts a fresh Python process.
        """
        # Set CUDA_VISIBLE_DEVICES to teacher GPUs BEFORE importing torch.cuda
        teacher_gpus = self.config["teacher_gpu_ids"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, teacher_gpus))
        LOG.info(f"Teacher server using GPUs: {teacher_gpus}")

        # Now safe to initialize vLLM
        self.llm = vllm.LLM(
            model=self.config["model_path"],
            dtype=self.config.get("dtype"),
            quantization=self.config.get("quantization"),
            trust_remote_code=self.config["trust_remote_code"],
            tensor_parallel_size=self.config["tensor_parallel_size"],
            gpu_memory_utilization=self.config["gpu_memory_utilization"],
            max_logprobs=self.config["top_k"],
            logprobs_mode="raw_logprobs",  # More efficient, unnormalized
            max_model_len=self.config.get("max_model_len"),
        )

        # Configure sampling parameters for logprob extraction
        self.sampling_params = vllm.SamplingParams(
            temperature=1.0,  # Unbiased
            top_p=1.0,
            min_p=0.0,
            top_k=-1,  # No sampling truncation
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            prompt_logprobs=self.config["top_k"],  # Extract k logprobs per token
            logprobs=self.config["top_k"],
            flat_logprobs=True,  # Use optimized format
            max_tokens=1,  # vLLM requires generating at least 1 token
            detokenize=False,
            skip_special_tokens=False,
        )

        # Get vocab size from model
        self.vocab_size = self.llm.llm_engine.model_config.get_vocab_size()
        LOG.info(f"Teacher model vocab size: {self.vocab_size}")

    def _process_request(self, request: TeacherRequest) -> TeacherResponse:
        """
        Process a single teacher inference request.

        Args:
            request: Teacher inference request

        Returns:
            Response with CUDA IPC handles or error
        """
        try:
            # Check cache first
            if request.batch_hash in self.tensor_cache:
                cached_ids, cached_values, _ = self.tensor_cache[request.batch_hash]
                LOG.debug(f"Cache hit for batch {request.batch_hash}")
                return self._create_response_from_tensors(
                    request, cached_ids, cached_values
                )

            LOG.debug(
                f"Processing request {request.request_id} "
                f"(batch_size={len(request.input_ids)})"
            )

            # Run vLLM inference
            outputs = self.llm.generate(
                [{"prompt_token_ids": seq} for seq in request.input_ids],
                sampling_params=self.sampling_params,
            )

            # Determine batch dimensions
            batch_size = len(outputs)
            max_seq_len = max(len(seq) for seq in request.input_ids)
            k = self.config["top_k"]

            # After CUDA_VISIBLE_DEVICES remapping, teacher GPU is always cuda:0
            # from the teacher process's perspective
            device = torch.device("cuda:0")

            # Allocate output tensors on teacher GPU
            sparse_ids = torch.zeros(
                (batch_size, max_seq_len, k), dtype=torch.long, device=device
            )
            sparse_values = torch.full(
                (batch_size, max_seq_len, k),
                fill_value=float("-inf"),
                dtype=torch.float32,
                device=device,
            )

            # Extract logprobs for each sequence in batch
            for batch_idx, output in enumerate(outputs):
                seq_ids, seq_values = process_prompt_logprobs(
                    output.prompt_logprobs, k, device
                )
                seq_len = seq_ids.shape[0]
                sparse_ids[batch_idx, :seq_len] = seq_ids
                sparse_values[batch_idx, :seq_len] = seq_values

            # Cache tensors (must stay alive for IPC)
            self._cache_tensors(request.batch_hash, sparse_ids, sparse_values)

            # Create response with IPC handles
            return self._create_response_from_tensors(request, sparse_ids, sparse_values)

        except Exception as e:
            LOG.error(f"Error processing request {request.request_id}: {e}", exc_info=True)
            return TeacherResponse(
                request_id=request.request_id,
                batch_hash=request.batch_hash,
                sparse_ids_handle=None,
                sparse_values_handle=None,
                sparse_ids_metadata={},
                sparse_values_metadata={},
                vocab_size=self.vocab_size,
                top_k=self.config["top_k"],
                success=False,
                error_message=str(e),
            )

    def _create_response_from_tensors(
        self,
        request: TeacherRequest,
        sparse_ids: torch.Tensor,
        sparse_values: torch.Tensor,
    ) -> TeacherResponse:
        """Create response with CUDA IPC handles from tensors."""
        ids_handle, ids_meta = serialize_cuda_tensor(sparse_ids)
        values_handle, values_meta = serialize_cuda_tensor(sparse_values)

        return TeacherResponse(
            request_id=request.request_id,
            batch_hash=request.batch_hash,
            sparse_ids_handle=ids_handle,
            sparse_values_handle=values_handle,
            sparse_ids_metadata=ids_meta,
            sparse_values_metadata=values_meta,
            vocab_size=self.vocab_size,
            top_k=self.config["top_k"],
            success=True,
        )

    def _cache_tensors(
        self, batch_hash: str, sparse_ids: torch.Tensor, sparse_values: torch.Tensor
    ):
        """
        Cache tensors to keep them alive for CUDA IPC.

        CRITICAL: Tensors must remain alive in sender process until receiver
        finishes using them. We use LRU eviction based on cache size limit.
        """
        # Evict old entries if cache exceeds size limit
        while self._get_cache_size() > self.cache_size_limit and self.tensor_cache:
            # Evict oldest (earliest timestamp)
            oldest_key = min(
                self.tensor_cache.keys(), key=lambda k: self.tensor_cache[k][2]
            )
            LOG.debug(f"Evicting cache entry: {oldest_key}")
            del self.tensor_cache[oldest_key]

        self.tensor_cache[batch_hash] = (sparse_ids, sparse_values, time.time())

    def _get_cache_size(self) -> int:
        """Estimate cache size in bytes."""
        total = 0
        for ids, values, _ in self.tensor_cache.values():
            total += ids.numel() * ids.element_size()
            total += values.numel() * values.element_size()
        return total

    def _cleanup(self):
        """Cleanup resources on shutdown."""
        LOG.info("Shutting down teacher server")
        self.tensor_cache.clear()
        if self.llm is not None:
            del self.llm
        torch.cuda.empty_cache()
