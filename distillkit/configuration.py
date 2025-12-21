# Copyright 2025 Arcee AI
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypeAlias

from distillkit.compression.config import (
    DistributionQuantizationConfig,
    LegacyLogitCompressionConfig,
)


class LossFunction(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    KL = "kl"
    JSD = "jsd"
    TVD = "tvd"
    HINGE = "hinge"
    LOGISTIC_RANKING = "logistic_ranking"
    HIDDEN_STATE_COSINE = "hs_cosine"
    HIDDEN_STATE_MSE = "hs_mse"


class MissingProbabilityHandling(Enum):
    ZERO = "zero"
    SYMMETRIC_UNIFORM = "symmetric_uniform"


class LossFunctionConfig(BaseModel):
    function: LossFunction = Field(
        ...,
        description="Type of loss function to use.",
    )
    weight: float = Field(
        ...,
        description="Weight for the loss function.",
    )
    temperature: float | None = Field(
        default=None,
        description="Temperature for loss, if applicable.",
    )
    missing_probability_handling: MissingProbabilityHandling | None = Field(
        default=None,
        description="Missing probability handling mode for sparse divergence functions.",
    )
    sparse_chunk_length: int | None = Field(
        default=None,
        description="Chunk length for sparse divergence functions. None to disable chunking.",
    )
    margin: float | None = Field(
        default=None,
        description="Margin for hinge loss, if applicable.",
    )


class HfRepoDataset(BaseModel):
    repo_id: str = Field(
        description="Hugging Face repository ID of the dataset.",
    )
    revision: str | None = Field(
        default=None,
        description="Revision of the dataset to use.",
    )
    config_name: str | None = Field(
        default=None,
        description="Configuration name of the dataset.",
    )
    split: str | None = Field(
        default=None,
        description="Split of the dataset to use.",
    )


class LocalDataset(BaseModel):
    disk_path: str = Field(
        description="Path to the local dataset or dataset dict directory.",
    )
    split: str | None = Field(
        default=None,
        description="Split of the dataset to use.",
    )


DatasetPath: TypeAlias = HfRepoDataset | LocalDataset


class DatasetConfiguration(BaseModel):
    train_dataset: DatasetPath = Field(
        description="Dataset to use for training.",
    )
    eval_dataset: DatasetPath | None = Field(
        default=None,
        description="Dataset to use for evaluation.",
    )
    seed: int | None = Field(
        default=42,
        description="Random seed for shuffling datasets.",
    )
    num_samples: int | None = Field(
        default=None,
        description="Number of samples to use from the dataset.",
    )
    num_eval_samples: int | None = Field(
        default=None,
        description="Number of samples to use from the evaluation dataset.",
    )
    eos_label_token_ids: list[int] | None = Field(
        default=None,
        description="List of token IDs to replace with EOS token IDs in the labels.",
    )
    prepared_dataset_path: str | None = Field(
        default=None,
        description="Path to store prepared dataset.",
    )
    prepacked: bool = Field(
        default=False,
        description="Assume dataset is pretokenized and packed, skip TRL packing.",
    )


class TeacherModelConfig(BaseModel):
    kind: Literal["hf"] = "hf"

    path: str
    kwargs: dict[str, Any] | None = None

    top_k: int | None = None


class TeacherDatasetConfig(BaseModel):
    kind: Literal["dataset"] = "dataset"
    legacy_logit_compression: LegacyLogitCompressionConfig | None = Field(
        default=None,
        description="Legacy logit compression configuration. Must match configuration used to capture logits.",
    )
    logprob_compressor: DistributionQuantizationConfig | None = Field(
        default=None,
        description="Logit compression configuration. Must match configuration used to capture logits.",
    )


class TeacherVLLMConfig(BaseModel):
    kind: Literal["vllm"] = "vllm"

    model_path: str = Field(
        description="HuggingFace model path for vLLM teacher.",
    )
    top_k: int = Field(
        description="Number of top-k logprobs to return (sparse signal). "
        "vLLM has linear performance degradation with number of logprobs, "
        "so use moderate values (32-128 recommended).",
    )
    teacher_gpu_ids: list[int] = Field(
        description="GPU IDs for teacher server (e.g., [3] or [3, 4] for tensor parallelism). "
        "These GPUs must be visible to the process but separate from student training GPUs.",
    )

    tensor_parallel_size: int = Field(
        default=1,
        description="vLLM tensor parallelism degree. Should match len(teacher_gpu_ids).",
    )
    dtype: str | None = Field(
        default="bfloat16",
        description="Model dtype (auto, bfloat16, float16, float32).",
    )
    quantization: str | None = Field(
        default=None,
        description="vLLM quantization method (awq, gptq, squeezellm, etc.).",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        description="Fraction of GPU memory to use for vLLM (0.0-1.0).",
    )
    max_model_len: int | None = Field(
        default=None,
        description="Maximum sequence length for vLLM. None uses model's default.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code when loading model.",
    )

    cache_size_mb: int = Field(
        default=2048,
        description="Maximum cache size in MB for teacher server tensor cache.",
    )
    request_timeout_sec: float = Field(
        default=60.0,
        description="Timeout in seconds for teacher inference requests.",
    )


class DistillationRunConfig(BaseModel):
    project_name: str = Field(
        default="distillkit",
        description="Project name for logging.",
    )
    train_model: str = Field(
        description="Model to train.",
        alias="model",
    )
    dataset: DatasetConfiguration
    teacher: TeacherModelConfig | TeacherDatasetConfig | TeacherVLLMConfig = Field(
        ..., discriminator="kind"
    )
    sequence_length: int = Field(
        description="Sequence length for training.",
    )
    output_path: str = Field(
        description="Path to save the model.",
    )
    resize_embeddings_to_multiple_of: int | None = Field(
        default=None,
        description="Resize embeddings to a multiple of this value.",
    )
    use_flash_attention: bool = Field(
        default=True,
        description="Use flash attention for training.",
    )

    loss_functions: list[LossFunctionConfig] = Field(
        description="List of loss functions to use for distillation.",
        default_factory=lambda: [
            LossFunctionConfig(
                function=LossFunction.CROSS_ENTROPY,
                weight=0.5,
            ),
            LossFunctionConfig(
                function=LossFunction.KL,
                weight=0.5,
                temperature=1.0,
                missing_probability_handling=MissingProbabilityHandling.ZERO,
            ),
        ],
    )
    layer_mapping: list[tuple[int, int]] | Literal["all"] | None = Field(
        default=None,
        description='List of (student_layer_idx, teacher_layer_idx) pairs (or "all" for a complete one-to-one mapping.)',
    )
    force_hidden_state_projection: bool = Field(
        default=False,
        description="Use linear layers to project between teacher and student hidden states even if sizes are equal.",
    )
    functionary_packing: bool = Field(
        default=False,
        description="Use functionary's packing code. Requires flash attention and may not be compatible with all models.",
    )
    training_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments for the trainer.",
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments for the model.",
    )
    model_auto_class: str | None = Field(
        default="AutoModelForCausalLM",
        description="Auto class for the model.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code when loading the model.",
    )
    frozen_modules: list[str] | None = Field(
        default=None,
        description="List of modules to freeze during training.",
    )
    frozen_res: list[str] | None = Field(
        default=None,
        description="List of regular expressions matching names of parameters to freeze during training.",
    )
