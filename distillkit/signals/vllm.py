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
    _connector: aiohttp.TCPConnector
    _session: aiohttp.ClientSession

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
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_name = api_key[2:-1]
            api_key = os.getenv(env_name)
        self.api_key = api_key

        # Create persistent connector and session
        self._connector = aiohttp.TCPConnector()

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._session = aiohttp.ClientSession(
            connector=self._connector,
            headers=headers,
        )

        asyncio.run(self._validate_server_connection())

    async def _cleanup(self):
        await self._session.close()
        await self._connector.close()

    def __del__(self):
        try:
            asyncio.get_event_loop().run_until_complete(self._cleanup())
        except RuntimeError:
            # Event loop may be closed during interpreter shutdown
            pass

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

        results = asyncio.run(self._get_logprobs_parallel(sequences))

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

    async def _validate_server_connection(self) -> None:
        """Validate vLLM server is reachable at initialization."""
        timeout = aiohttp.ClientTimeout(total=5.0)
        try:
            async with self._session.get(
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
        self, sequences: list[list[int]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get logprobs for multiple sequences in parallel."""
        # Create tasks for all sequences
        tasks = [self._get_logprobs_for_sequence(seq) for seq in sequences]

        # Run all requests concurrently
        results = await asyncio.gather(*tasks)

        return results

    async def _get_logprobs_for_sequence(
        self, input_ids: list[int]
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Get top-k logprobs for a single sequence from vLLM (async)."""

        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.model,
            "prompt": input_ids,
            "max_tokens": 1,  # vLLM requires at least 1 output token
            "temperature": 1.0,
            "top_p": 1.0,
            "logprobs": self.top_k,
            "prompt_logprobs": self.top_k,
            "return_tokens_as_token_ids": True,
            "echo": False,
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        # Make request with retries and exponential backoff
        for attempt in range(self.max_retries):
            try:
                async with self._session.post(
                    url, json=payload, timeout=timeout
                ) as response:
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
        prompt_logprobs = result["choices"][0]["prompt_logprobs"]
        if len(prompt_logprobs) > 0 and prompt_logprobs[0] is None:
            # Expected - vLLM returns first prompt logprob as None by convention
            prompt_logprobs = prompt_logprobs[1:]

        # Convert to tensors
        # prompt_logprobs is a list of dicts: [{token_id: logprob, ...}, ...]
        seq_len = len(prompt_logprobs)
        sparse_ids = torch.full(
            (seq_len, self.top_k),
            fill_value=-1,
            dtype=torch.long,
            device="cpu",
        )
        sparse_values = torch.full(
            (seq_len, self.top_k), fill_value=float("-inf"), device="cpu"
        )

        for pos, logprob_dict in enumerate(prompt_logprobs):
            for tok_id, lp in logprob_dict.items():
                rank = lp.get("rank", None)
                if rank is None or rank > self.top_k:
                    # rank is 1-indexed
                    # truncate to the top k that we want
                    # also can be None for prompt token if vllm included it despite not being in top N
                    continue
                sparse_ids[pos, rank - 1] = tok_id
                sparse_values[pos, rank - 1] = lp["logprob"]

        return sparse_ids, sparse_values
