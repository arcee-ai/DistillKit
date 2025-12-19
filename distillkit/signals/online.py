# Copyright 2025 Arcee AI
"""Online signal source using HuggingFace model inference."""

from typing import Any

import torch
import transformers
from typing_extensions import override

from distillkit.signals.base import (
    DenseSignal,
    SignalSource,
    SparseSignal,
    TeacherSignal,
)


class OnlineSignalSource(SignalSource):
    teacher_model: transformers.PreTrainedModel
    vocab_size: int
    sparsify_top_k: int | None
    teacher_kwargs: dict[str, Any]

    def __init__(
        self,
        teacher_model: transformers.PreTrainedModel,
        vocab_size: int,
        sparsify_top_k: int | None = None,
        teacher_kwargs: dict[str, Any] | None = None,
    ):
        self.teacher_model = teacher_model.eval()
        self.vocab_size = vocab_size
        self.sparsify_top_k = sparsify_top_k
        self.teacher_kwargs = teacher_kwargs or {}

        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    @override
    def supports_hidden_states(self) -> bool:
        return True

    @override
    def get_signal(
        self, batch: dict[str, Any], return_hidden_states: bool = False
    ) -> TeacherSignal:
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                output_hidden_states=return_hidden_states,
                **self.teacher_kwargs,
            )

        real_vocab_size = teacher_outputs.logits.shape[-1]
        vocab_size = min(real_vocab_size, self.vocab_size)

        if self.sparsify_top_k is not None:
            logprobs = torch.log_softmax(
                teacher_outputs.logits[..., :vocab_size], dim=-1
            )
            values, indices = torch.topk(logprobs, self.sparsify_top_k, dim=-1)
            return SparseSignal(
                sparse_ids=indices,
                sparse_values=values,
                log_values=True,
                generation_temperature=1.0,
                hidden_states=teacher_outputs.hidden_states,
                vocab_size=vocab_size,
            )

        return DenseSignal(
            logits=teacher_outputs.logits[..., :vocab_size],
            hidden_states=teacher_outputs.hidden_states,
            generation_temperature=1.0,
            vocab_size=vocab_size,
        )
