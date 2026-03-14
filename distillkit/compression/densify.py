import torch

from distillkit.missing_probability import MissingProbabilityHandling


def densify(
    top_indices: torch.LongTensor,
    top_values: torch.Tensor,
    vocab_size: int,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    renormalize: bool = False,
    fill_value: float = -float("inf"),
) -> torch.Tensor:
    """Expand a sparse set of logits to a dense tensor.

    Fills missing logits with -inf."""

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        # compute total probability mass
        log_total_mass = torch.logsumexp(top_values, dim=-1, keepdim=True)
        missing = 1 - log_total_mass.exp()
        # equally spread missing mass over all missing tokens
        fill_value = torch.log(
            (missing / (vocab_size - top_values.shape[-1])).clamp(min=1e-8)
        )
    elif missing == MissingProbabilityHandling.ZERO:
        # just fill with -inf (or whatever fill_value is)
        pass

    expanded_logits = (
        torch.zeros(
            tuple(top_indices.shape[:-1]) + (vocab_size,),
            device=top_indices.device,
            dtype=top_values.dtype,
        )
        + fill_value
    )
    expanded_logits.scatter_(-1, top_indices, top_values)
    if renormalize:
        expanded_logits = torch.log_softmax(expanded_logits, dim=-1)
    return expanded_logits
