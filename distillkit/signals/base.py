# Copyright 2025 Arcee AI
"""Base classes and types for teacher signal sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch


@dataclass
class TeacherSignalBase:
    generation_temperature: float
    hidden_states: tuple[torch.Tensor, ...] | None
    vocab_size: int


@dataclass
class DenseSignal(TeacherSignalBase):
    logits: torch.Tensor


@dataclass
class SparseSignal(TeacherSignalBase):
    sparse_ids: torch.LongTensor
    sparse_values: torch.Tensor
    log_values: bool  # if True, values are logprobs


TeacherSignal: TypeAlias = SparseSignal | DenseSignal


class SignalSource(ABC):
    @abstractmethod
    def supports_hidden_states(self) -> bool: ...

    @abstractmethod
    def get_signal(
        self, batch: dict[str, Any], return_hidden_states: bool = False
    ) -> TeacherSignal: ...
