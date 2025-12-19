# Copyright 2025 Arcee AI
"""Teacher signal sources for distillation."""

from distillkit.signals.base import (
    DenseSignal,
    SignalSource,
    SparseSignal,
    TeacherSignal,
    TeacherSignalBase,
)
from distillkit.signals.offline import OfflineSignalSource
from distillkit.signals.online import OnlineSignalSource
from distillkit.signals.vllm import VLLMSignalSource

__all__ = [
    # Base types
    "TeacherSignalBase",
    "DenseSignal",
    "SparseSignal",
    "TeacherSignal",
    "SignalSource",
    # Signal sources
    "OfflineSignalSource",
    "OnlineSignalSource",
    "VLLMSignalSource",
]
