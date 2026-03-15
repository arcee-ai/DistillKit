from enum import Enum

import torch
from pydantic import BaseModel, Field, model_validator


class SpecialTerm(Enum):
    SQRT = "sqrt"
    EXP = "exp"


class TermDtype(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    def bit_width(self) -> int:
        if self == TermDtype.FLOAT16:
            return 16
        elif self == TermDtype.BFLOAT16:
            return 16
        elif self == TermDtype.FLOAT32:
            return 32
        elif self == TermDtype.FLOAT64:
            return 64
        else:
            raise ValueError(f"Unsupported dtype: {self}")

    def dtype(self) -> torch.dtype:
        if self == TermDtype.FLOAT16:
            return torch.float16
        elif self == TermDtype.BFLOAT16:
            return torch.bfloat16
        elif self == TermDtype.FLOAT32:
            return torch.float32
        elif self == TermDtype.FLOAT64:
            return torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {self}")


class QuantizationBin(BaseModel):
    scale_dtype: TermDtype = Field(
        ...,
        description="Data type for the scale value.",
    )
    element_bits: int = Field(
        ...,
        description="Number of bits per element.",
        gt=0,
        le=64,
    )
    num_elements: int = Field(
        ...,
        description="Number of elements in the bin.",
        gt=0,
    )

    # note that element_bits can be anything from 1 to 64
    # *could* constrain it to 1-8, 16, 32, 64
    # but would be better not to


class DistributionQuantizationConfig(BaseModel):
    d: int = Field(
        ...,
        description="Dimension of the unquantized distribution.",
        gt=0,
    )
    k: int = Field(
        ...,
        description="Number of non-zero values in the quantized distribution.",
        gt=0,
    )
    exact_k: int = Field(
        ...,
        description="Number of top values to store unquantized.",
        ge=0,
    )
    exact_dtype: TermDtype = Field(
        TermDtype.FLOAT32,
        description="Data type for the top `exact_k` values.",
    )
    polynomial_terms: list[SpecialTerm | int] | None = Field(
        None,
        description="Terms to use in the polynomial approximation. Integer values represent power terms, "
        "SpecialTerm values represent special non-polynomial terms.",
    )
    term_dtype: TermDtype = Field(
        TermDtype.FLOAT32,
        description="Data type for the polynomial terms.",
    )
    residual_bins: list[QuantizationBin] = Field(
        ...,
        description="List of bins for quantized residuals.",
    )
    delta_encoding: bool = Field(True, description="Whether to use delta encoding.")
    error_diffusion: bool = Field(
        False,
        description="Whether to use error diffusion.",
    )
    normalize_t: bool = Field(
        default=False, description="Map t to [0,1] instead of [0,k]."
    )

    @model_validator(mode="after")
    def check_valid(self) -> "DistributionQuantizationConfig":
        if self.k < self.exact_k:
            raise ValueError("k must be >= exact_k")
        if (
            self.exact_k < self.k
            and (not self.polynomial_terms)
            and (not self.residual_bins)
        ):
            raise ValueError(
                "If exact_k < k, at least one of polynomial_terms or residual_bins must be provided."
            )
        approx_terms = self.k - self.exact_k
        bin_elems = sum([bin.num_elements for bin in self.residual_bins])
        if bin_elems > approx_terms:
            raise ValueError(
                "Sum of num_elements in residual_bins must be <= k - exact_k"
            )
        return self

    def logprob_bits(self) -> int:
        res = 0
        res += self.exact_k * self.exact_dtype.bit_width()
        res += len(self.polynomial_terms or []) * self.term_dtype.bit_width()
        for bin in self.residual_bins:
            bin_bits = bin.scale_dtype.bit_width() + bin.element_bits * bin.num_elements
            if bin_bits % 8 != 0:
                bin_bits += 8 - (bin_bits % 8)
            res += bin_bits
        if res % 8 != 0:
            res += 8 - (res % 8)
        return res

    def total_bits(self) -> int:
        lpb = self.logprob_bits()
        vocab_index_bits = int(
            torch.log2(torch.tensor(self.d, dtype=torch.float32)).ceil().item()
        )
        index_bytes = (self.k * vocab_index_bits + 7) // 8
        total = lpb + index_bytes * 8
        return total


class LegacyLogitCompressionConfig(BaseModel):
    """Configuration for legacy polynomial logit compression.

    Args:
        k (int): Total number of logits per token
        exact_k (int): Number of exact logits to keep
        polynomial_degree (int): Degree of the polynomial to approximate the remaining logits
        invert_polynomial (bool): Whether to invert the polynomial terms
        with_sqrt_term (bool): Whether to include a square root term in the polynomial
        term_dtype (str): Data type for the polynomial terms (float16, bfloat16, float32, float64)
    """

    k: int
    exact_k: int
    polynomial_degree: int
    vocab_size: int
    invert_polynomial: bool = True
    with_sqrt_term: bool = False
    term_dtype: str = "float32"
