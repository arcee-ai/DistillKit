# Copyright 2025 Arcee AI
"""Online vLLM server signal source."""

import asyncio
import logging
import os
from typing import Any

import aiohttp
import torch
from typing_extensions import override

from distillkit.signals.base import SignalSource, SparseSignal

LOG = logging.getLogger(__name__)


class VLLMSignalSource(SignalSource):
    """Teacher signal source using vLLM server API."""

    base_url: str
    model: str
    top_k: int
    vocab_size: int
    timeout: float
    max_retries: int
    api_key: str | None

    def __init__(
        self,
        base_url: str,
        model: str,
        top_k: int,
        vocab_size: int,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.top_k = top_k
        self.vocab_size = vocab_size
        self.timeout = timeout
        self.max_retries = max_retries

        LOG.info(f"VLLMSignalSource initialized with vocab_size={vocab_size}")
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_name = api_key[2:-1]
            api_key = os.getenv(env_name)
        self.api_key = api_key

        # Validate server connection at initialization
        asyncio.run(self._validate_server_connection_init())

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @override
    def supports_hidden_states(self) -> bool:
        return False

    @override
    def get_signal(
        self, batch: dict[str, Any], return_hidden_states: bool = False
    ) -> SparseSignal:
        if return_hidden_states:
            raise RuntimeError("vLLM signal source does not support hidden states")

        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]
        sequences = [input_ids[i].tolist() for i in range(batch_size)]

        results = asyncio.run(self._get_logprobs_batch(sequences))

        # Unpack results
        all_sparse_ids = [r[0] for r in results]
        all_sparse_values = [r[1] for r in results]
        sparse_ids_tensor = torch.stack(all_sparse_ids)
        sparse_values_tensor = torch.stack(all_sparse_values)

        return SparseSignal(
            sparse_ids=sparse_ids_tensor.to(input_ids.device, non_blocking=True),
            sparse_values=sparse_values_tensor.to(input_ids.device, non_blocking=True),
            log_values=True,  # vLLM returns log probabilities
            generation_temperature=1.0,
            hidden_states=None,
            vocab_size=self.vocab_size,
        )

    async def _get_logprobs_batch(
        self, sequences: list[list[int]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get logprobs for a batch of sequences (creates session for this batch)."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        async with aiohttp.ClientSession(
            connector=connector, headers=self._get_headers()
        ) as session:
            results = await self._get_logprobs_parallel(session, sequences)
        return results

    async def _validate_server_connection_init(self) -> None:
        """Validate vLLM server is reachable at initialization."""
        timeout = aiohttp.ClientTimeout(total=5.0)
        try:
            async with aiohttp.ClientSession(headers=self._get_headers()) as session:
                async with session.get(
                    f"{self.base_url}/v1/models", timeout=timeout
                ) as response:
                    response.raise_for_status()
                    models = await response.json()
                    available_models = [m["id"] for m in models.get("data", [])]
                    if self.model not in available_models:
                        LOG.warning(
                            f"Requested model {self.model} not found. Available: {available_models}"
                        )
                    else:
                        LOG.info(f"Found requested model {self.model}")
        except aiohttp.ClientError as e:
            raise RuntimeError(
                f"Failed to connect to vLLM server at {self.base_url}. "
                f"Please ensure the server is running. Error: {e}"
            ) from e

    async def _get_logprobs_parallel(
        self, session: aiohttp.ClientSession, sequences: list[list[int]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get logprobs for multiple sequences in parallel."""
        # Create tasks for all sequences
        tasks = [self._get_logprobs_for_sequence(session, seq) for seq in sequences]

        # Run all requests concurrently
        results = await asyncio.gather(*tasks)

        return results

    async def _get_logprobs_for_sequence(
        self, session: aiohttp.ClientSession, input_ids: list[int]
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Get top-k logprobs for a single sequence from vLLM (async)."""

        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.model,
            "prompt": input_ids,
            "max_tokens": 1,  # vLLM requires at least 1 output token
            "temperature": 1.0,
            "top_p": 1.0,
            "logprobs": self.top_k,  # Get completion logprobs for the generated token
            "prompt_logprobs": self.top_k,
            "return_tokens_as_token_ids": True,
            "echo": False,
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        # Make request with retries and exponential backoff
        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=payload, timeout=timeout) as response:
                    response.raise_for_status()
                    result = await response.json()
                    break
            except aiohttp.ClientError as e:
                LOG.debug(
                    f"vLLM API request #{attempt + 1}/{self.max_retries} failed: {e}",
                    exc_info=True,
                )
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"vLLM API request failed: {e}")
                # Exponential backoff
                await asyncio.sleep(2**attempt)

        # Parse response
        completion = result["choices"][0]
        prompt_logprobs = completion["prompt_logprobs"]
        generated_logprobs = completion["logprobs"]["top_logprobs"]
        items = []
        for tok_str, lp in generated_logprobs[0].items():
            assert tok_str.startswith("token_id:")
            tok_id = int(tok_str[len("token_id:") :])
            items.append((lp, tok_id))
        items.sort(reverse=True)
        completion_logprob = {
            tok_id: {"rank": idx + 1, "logprob": logprob}
            for idx, (logprob, tok_id) in enumerate(items)
        }

        # Combine prompt and completion logprobs to get full sequence
        # prompt_logprobs[i] = P(input_ids[i] | input_ids[:i])
        # We want P(output[i] | input_ids[:i+1]) which is:
        #   - For i=0 to len-2: prompt_logprobs[i+1] (next token given prefix)
        #   - For i=len-1: completion_logprob (generated token)

        logprobs = prompt_logprobs[1:] + [completion_logprob]

        # Convert to tensors
        # logprobs is a list of dicts: [{token_id: {"logprob": ..., "rank": ...}, ...}, ...]
        seq_len = len(logprobs)
        sparse_ids = torch.full(
            (seq_len, self.top_k),
            fill_value=-1,
            dtype=torch.long,
            device="cpu",
        )
        sparse_values = torch.full(
            (seq_len, self.top_k), fill_value=float("-inf"), device="cpu"
        )

        for pos, logprob_dict in enumerate(logprobs):
            for tok_id, lp in logprob_dict.items():
                rank = lp.get("rank", None)
                if rank is None or rank > self.top_k:
                    # rank is 1-indexed
                    # truncate to the top k that we want
                    # also can be None for prompt token if vllm included it despite not being in top N
                    continue
                sparse_ids[pos, rank - 1] = int(tok_id)
                sparse_values[pos, rank - 1] = lp["logprob"]

        return sparse_ids, sparse_values
