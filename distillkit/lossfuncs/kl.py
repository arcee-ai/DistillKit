# Copyright 2024 Charles O. Goddard

import torch
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.common import (
    LossFunctionBase,
    MissingProbabilityHandling,
    accumulate_over_chunks,
    get_logprobs,
)
from distillkit.signals import DenseSignal, TeacherSignal


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(x)) in a numerically stable way."""
    mask = x > -1
    result = torch.empty_like(x)
    result[mask] = torch.log(-torch.expm1(x[mask]))
    result[~mask] = torch.log1p(-torch.exp(x[~mask]))
    return result


def log_missing_prob(sparse_logprobs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    log_prob_sum = torch.logsumexp(sparse_logprobs.float(), dim=-1)
    log_prob_sum = log_prob_sum.clamp(max=-eps)
    return log1mexp(log_prob_sum)


def sparse_kl_div_inner(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    log_target: bool = True,
    temperature: float = 1.0,
    target_generation_temperature: float = 1.0,
    student_generation_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the KL divergence between a dense set of predictions and a sparse set of target logits.

    See `sparse_kl_div` for details.
    """
    batch_size, seq_len, vocab_size = logits.shape
    out_dtype = logits.dtype
    sparse_student_logprobs, sparse_target_logprobs = get_logprobs(
        logits,
        target_ids,
        target_values,
        eps=eps,
        missing=missing,
        log_target=log_target,
        distillation_temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )

    # Terms for non-zero target probabilities
    teacher_sparse_probs = torch.exp(sparse_target_logprobs.float())
    inner_sum = torch.sum(
        teacher_sparse_probs
        * (sparse_target_logprobs - sparse_student_logprobs.float()),
        dim=-1,
    )

    # Compute the contribution of missing logits to KL divergence
    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        # if the teacher's logprobs don't sum to 1, we assume the remaining
        # probability mass in *both* the teacher and student is distributed
        # uniformly over the token indices missing from the teacher's distribution

        log_teacher_missing = log_missing_prob(sparse_target_logprobs, eps=eps)
        log_student_missing = log_missing_prob(sparse_student_logprobs, eps=eps)
        log_ratio = log_teacher_missing - log_student_missing

        teacher_missing_prob = torch.exp(log_teacher_missing)

        missing_kl = torch.where(
            teacher_missing_prob > eps,
            teacher_missing_prob * log_ratio,
            torch.zeros_like(teacher_missing_prob),
        )
    else:
        # in this case we assume zero probability mass for missing tokens
        # in the teacher distribution, and thus zero contribution to KL divergence
        missing_kl = None

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        inner_sum *= mask
        if missing_kl is not None:
            missing_kl *= mask

    if missing_kl is not None:
        inner_sum += missing_kl

    return torch.sum(inner_sum).to(out_dtype)


def sparse_kl_div(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    log_target: bool = True,
    temperature: float = 1.0,
    target_generation_temperature: float = 1.0,
    student_generation_temperature: float = 1.0,
    chunk_length: int | None = None,
) -> torch.Tensor:
    """Compute the KL divergence between a dense set of predictions and a sparse set of target logits.

    Uses a chunked approach to avoid memory issues with large sequences.

    Args:
        logits: Dense tensor of predictions.
        target_ids: Tensor of indices for target logits.
        target_values: Tensor of values for target logits or log probabilities.
        mask: Optional boolean mask tensor. True indicates tokens to include, False to exclude.
        eps: Small value to prevent numerical instability.
        missing: How to handle missing probabilities in the target distribution. If ZERO, missing
            probabilities are assumed to be zero. If SYMMETRIC_UNIFORM, missing probabilities are
            assumed to be distributed uniformly over the missing tokens in both the teacher and
            student distributions.
        log_target: Whether the target values are already log probabilities.
        temperature: Temperature to apply to the distributions.
        target_generation_temperature: Temperature already applied to the target logits/logprobs.
        student_generation_temperature: Temperature already applied to the student logits.
        chunk_length: Number of tokens per chunk. If None, the entire sequence is processed at once.
    """
    return accumulate_over_chunks(
        logits,
        target_ids,
        target_values,
        mask,
        chunk_length,
        sparse_kl_div_inner,
        eps=eps,
        missing=missing,
        log_target=log_target,
        temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )


def dense_kl_div(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the KL divergence between a dense set of predictions and a dense set of target logits.

    Args:
        logits: Dense tensor of predictions (Student).
        target_logits: Dense tensor of target logits (Teacher).
        mask: Optional boolean mask tensor. True indicates tokens to include, False to exclude.
        temperature: Temperature to apply to the distributions.
    """
    out_dtype = logits.dtype

    student_logprobs = torch.log_softmax(logits.float() / temperature, dim=-1)
    teacher_logprobs = torch.log_softmax(target_logits.float() / temperature, dim=-1)

    kl_per_element = torch.nn.functional.kl_div(
        input=student_logprobs,
        target=teacher_logprobs,
        reduction="none",
        log_target=True,
    )
    # Sum over the vocabulary dimension (dim=-1) to get KL per token
    kl_per_token = torch.sum(kl_per_element, dim=-1)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        kl_per_token = kl_per_token * mask.float()

    return torch.sum(kl_per_token).to(out_dtype)


class KLDLoss(LossFunctionBase):
    temperature: float
    missing: MissingProbabilityHandling
    chunk_length: int | None

    @override
    @classmethod
    def name(cls) -> str:
        return "kl"

    @override
    def __init__(
        self,
        temperature: float,
        missing_probability_handling: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
        sparse_chunk_length: int | None = None,
    ) -> None:
        self.temperature = temperature
        self.missing = missing_probability_handling
        self.chunk_length = sparse_chunk_length

    @override
    def __call__(
        self,
        student_outputs: CausalLMOutput,
        signal: TeacherSignal,
        mask: torch.Tensor | None = None,
        hidden_state_mapping: HiddenStateMapping | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if num_items_in_batch is None:
            if mask is not None:
                num_items_in_batch = mask.float().sum()
            else:
                num_items_in_batch = (
                    student_outputs.logits.shape[0] * student_outputs.logits.shape[1]
                )
        if isinstance(signal, DenseSignal):
            res = dense_kl_div(
                student_outputs.logits,
                signal.logits,
                mask=mask,
                temperature=self.temperature,
            )
        else:
            res = sparse_kl_div(
                logits=student_outputs.logits,
                target_ids=signal.sparse_ids,
                target_values=signal.sparse_values,
                mask=mask,
                missing=self.missing,
                log_target=signal.log_values,
                temperature=self.temperature,
                target_generation_temperature=signal.generation_temperature,
                chunk_length=self.chunk_length,
            )
        return res * (self.temperature**2) / num_items_in_batch
