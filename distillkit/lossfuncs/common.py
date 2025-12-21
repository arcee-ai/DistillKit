# Copyright 2024 Charles O. Goddard
import math
from abc import ABC, abstractmethod
from typing import Callable

import torch
from transformers.modeling_outputs import CausalLMOutput

from distillkit.configuration import MissingProbabilityHandling
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.signals import TeacherSignal


def get_target_logprobs(
    values_in: torch.Tensor,
    log_target: bool,
    distillation_temperature: float,
    target_generation_temperature: float,
    missing: MissingProbabilityHandling,
) -> torch.Tensor:
    temperature_change = not math.isclose(
        distillation_temperature, target_generation_temperature
    )
    if log_target:
        if (
            not temperature_change
            and missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM
        ):
            # if we are not changing the temperature and don't need to
            # renormalize to sum to 1 as in the ZERO case, we can just
            # return the input values
            return values_in
        # we want to divide the target logits by temperature
        # but unfortunately, we only have topk logprobs
        # general approach: convert to probs, apply temperature, renormalize
        alpha = target_generation_temperature / distillation_temperature

        # compute total mass of unscaled target probs
        max_in, _ = values_in.max(dim=-1, keepdim=True)
        lse_in = torch.logsumexp(values_in - max_in, dim=-1, keepdim=True) + max_in
        target_sum = lse_in.exp()
        leftover = (1.0 - target_sum).clamp(0, 1)

        alpha_values_in = alpha * values_in
        # We need sum( p_i^alpha ). In log space: sum( exp(alpha * log p_i) ).
        max_alpha, _ = alpha_values_in.max(dim=-1, keepdim=True)
        alpha_lse = (
            torch.logsumexp(alpha_values_in - max_alpha, dim=-1, keepdim=True)
            + max_alpha
        )
        alpha_sum = alpha_lse.exp()

        if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
            # assume the remaining probability mass is distributed uniformly
            # over the missing token indices in both the teacher and student
            # distributions (symmetrically) - this lets us act as though
            # instead of a N-dimensional distribution we have a k+1 dimensional
            # distribution, where the k+1th dimension is the probability of all
            # other tokens.
            leftover_alpha = leftover.pow(alpha)
            divisor = alpha_sum + leftover_alpha
            final_lse = divisor.log()
        else:
            # assume the missing tokens have zero probability
            final_lse = alpha_lse
        # final log-probabilities are alpha * log p_i - log(Z)
        return alpha_values_in - final_lse
    else:
        # we have logits, praise be
        if temperature_change:
            logits = values_in * (
                target_generation_temperature / distillation_temperature
            )
        else:
            logits = values_in
        sparse_max = torch.max(logits, dim=-1, keepdim=True).values
        sparse_lse = (
            torch.logsumexp(
                (logits - sparse_max).to(torch.float32), dim=-1, keepdim=True
            )
            + sparse_max
        ).to(values_in.dtype)
        return logits - sparse_lse


def get_logprobs(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    eps: float = 1e-6,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    log_target: bool = True,
    distillation_temperature: float = 1.0,
    target_generation_temperature: float = 1.0,
    student_generation_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, vocab_size = logits.shape
    assert target_ids.shape[:-1] == (batch_size, seq_len), (
        f"Target ids shape {target_ids.shape[:-1]} does not match logits shape "
        f"{logits.shape}"
    )
    assert target_values.shape == target_ids.shape, (
        f"Target values shape {target_values.shape} does not match target ids shape "
        f"{target_ids.shape}"
    )
    assert distillation_temperature > eps, (
        f"Temperature must be positive and non-zero, got {distillation_temperature}"
    )
    out_dtype = logits.dtype

    if (not log_target) and (missing != MissingProbabilityHandling.ZERO):
        raise ValueError(
            "For log_target=False (teacher inputs are logits), "
            "MissingProbabilityHandling.SYMMETRIC_UNIFORM is ill-defined. "
            "The teacher distribution is only over the provided sparse logits. "
            "Use MissingProbabilityHandling.ZERO."
        )

    if not math.isclose(distillation_temperature, student_generation_temperature):
        logits = logits * (student_generation_temperature / distillation_temperature)
    student_lse = torch.logsumexp(logits.to(torch.float32), dim=-1, keepdim=True).to(
        out_dtype
    )
    sparse_student_logprobs = logits.gather(-1, target_ids) - student_lse
    del student_lse, logits

    with torch.no_grad():
        sparse_target_logprobs = get_target_logprobs(
            target_values.to(torch.float32),
            log_target=log_target,
            distillation_temperature=distillation_temperature,
            target_generation_temperature=target_generation_temperature,
            missing=missing,
        ).to(out_dtype)
    del target_values

    return sparse_student_logprobs, sparse_target_logprobs


def accumulate_over_chunks(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None,
    chunk_length: int | None,
    fn: Callable,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Accumulate the result of a function over chunks of the input tensors.
    Args:
        logits (torch.Tensor): The logits tensor.
        target_ids (torch.LongTensor): The target IDs tensor.
        target_values (torch.Tensor): The target values tensor.
        chunk_size (int | None): The size of each chunk. If None, the entire sequence is used.
        fn (Callable): The function to apply to each chunk.
        *args: Additional arguments to pass to the function.
        **kwargs: Additional keyword arguments to pass to the function.
    Returns:
        torch.Tensor: The accumulated result.
    """
    seq_len = logits.shape[1]
    if chunk_length is None:
        chunk_length = seq_len

    total = 0.0

    for start_idx in range(0, seq_len, chunk_length):
        if mask is not None:
            cur_mask = mask[:, start_idx : start_idx + chunk_length]
        else:
            cur_mask = None
        end_idx = min(start_idx + chunk_length, seq_len)
        total += fn(
            logits[:, start_idx:end_idx],
            target_ids[:, start_idx:end_idx],
            target_values[:, start_idx:end_idx],
            cur_mask,
            *args,
            **kwargs,
        )
    return total


class LossFunctionBase(ABC):
    @classmethod
    @abstractmethod
    def name(cls) -> str: ...

    def requires_hidden_states(self) -> bool:
        return False

    @abstractmethod
    def __init__(self, **kwargs) -> None: ...

    @abstractmethod
    def __call__(
        self,
        student_outputs: CausalLMOutput,
        signal: TeacherSignal,
        mask: torch.Tensor | None = None,
        hidden_state_mapping: HiddenStateMapping | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor: ...
