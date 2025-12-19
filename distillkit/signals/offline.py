# Copyright 2025 Arcee AI
"""Offline signal source using pre-captured compressed logits."""

from typing import Any

import torch
from typing_extensions import override

from distillkit.compression import LogprobCompressor
from distillkit.signals.base import SignalSource, SparseSignal


class OfflineSignalSource(SignalSource):
    compressor: LogprobCompressor
    preapplied_temperature: float
    vocab_size: int
    log_values: bool

    def __init__(
        self,
        compressor: LogprobCompressor,
        vocab_size: int,
        preapplied_temperature: float = 1.0,
        log_values: bool = True,
    ):
        self.compressor = compressor
        self.vocab_size = vocab_size
        self.preapplied_temperature = preapplied_temperature
        self.log_values = log_values

    @override
    def supports_hidden_states(self) -> bool:
        return False

    @override
    def get_signal(
        self, batch: dict[str, Any], return_hidden_states: bool = False
    ) -> SparseSignal:
        if return_hidden_states:
            raise RuntimeError(
                "Hidden states requested but signal source is precomputed logits"
            )
        with torch.no_grad():
            sparse_ids, sparse_values = self.compressor.decompress_to_sparse(batch)
        return SparseSignal(
            sparse_ids=sparse_ids,
            sparse_values=sparse_values,
            log_values=self.log_values,
            generation_temperature=self.preapplied_temperature,
            hidden_states=None,
            vocab_size=self.vocab_size,
        )
