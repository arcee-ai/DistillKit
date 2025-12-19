from typing import Tuple
import click
import numpy as np
import transformers

try:
    import vllm
except ImportError:
    raise ImportError("VLLM must be installed to use this script.")
import logging

import pyarrow
import torch
import tqdm
import yaml

from distillkit.compression import DistributionQuantizationConfig, LogprobCompressor
from distillkit.sample_common import (
    StreamingParquetWriter,
    load_preprocess_data,
)

from vllm.logprobs import PromptLogprobs, FlatLogprobs, Logprob, LogprobsOnePosition


@click.command("sample-logits")
@click.option("--model", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--configuration", type=str, default=None)
@click.option("--output", type=str, required=True)
@click.option("--split", type=str, default="train")
@click.option("--samples", type=int, default=None)
@click.option("--seed", type=int, default=42)
@click.option("--tokenizer", type=str, default=None)
@click.option("--apply-chat-template/--no-apply-chat-template", default=False)
@click.option("--max-seq-len", type=int, default=1024)
@click.option("--max-model-len", type=int, default=None)
@click.option("--tensor-parallel-size", type=int, default=1)
@click.option("--pipeline-parallel-size", type=int, default=1)
@click.option("--dtype", type=str, default=None)
@click.option("--quantization", type=str, default=None)
@click.option("--trust-remote-code/--no-trust-remote-code", default=False)
@click.option("--gpu-memory-utilization", type=float, default=0.9)
@click.option("--compression-config", type=str, required=True)
@click.option("--macrobatch-size", type=int, default=256)
def sample_logits(
    model: str,
    dataset: str,
    configuration: str | None,
    split: str,
    output: str,
    samples: int | None,
    tokenizer: str | None,
    apply_chat_template: bool,
    seed: int,
    max_seq_len: int,
    max_model_len: int | None,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    dtype: str | None,
    quantization: str | None,
    trust_remote_code: bool,
    gpu_memory_utilization: float,
    compression_config: str,
    macrobatch_size: int,
):
    logging.basicConfig(level=logging.INFO)

    tok = transformers.AutoTokenizer.from_pretrained(
        tokenizer or model, trust_remote_code=trust_remote_code
    )

    # load compression config
    with open(compression_config, "r") as f:
        compression_config = yaml.safe_load(f)
    compression_config = DistributionQuantizationConfig.model_validate(
        compression_config
    )
    k = compression_config.k

    logging.info(f"Loading and preprocessing data from {dataset} ({split})")
    ds = load_preprocess_data(
        dataset=dataset,
        configuration=configuration,
        split=split,
        samples=samples,
        seed=seed,
        max_seq_len=max_seq_len + 1,
        tokenizer=tok,
        add_extra_pad_token=True,
        apply_chat_template=apply_chat_template,
    )

    llm = vllm.LLM(
        model=model,
        tokenizer=tokenizer,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_logprobs=k,
        logprobs_mode="raw_logprobs",
        max_model_len=max_model_len,
        distributed_executor_backend="ray",
    )

    compressor = LogprobCompressor(
        config=compression_config,
    )

    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1,
        min_p=0,
        top_k=-1,
        frequency_penalty=0,
        presence_penalty=0,
        repetition_penalty=1,
        prompt_logprobs=k,
        logprobs=k,
        flat_logprobs=True,
        max_tokens=1,
        detokenize=False,
        skip_special_tokens=False,
    )

    logging.info(f"Generating logits for {len(ds)} samples")
    schema = pyarrow.schema(
        [
            pyarrow.field("input_ids", pyarrow.list_(pyarrow.uint64())),
            pyarrow.field(
                "compressed_logprobs", pyarrow.list_(pyarrow.list_(pyarrow.uint8()))
            ),
            pyarrow.field(
                "bytepacked_indices", pyarrow.list_(pyarrow.list_(pyarrow.uint8()))
            ),
        ]
    )
    with StreamingParquetWriter(
        output,
        schema=schema,
        file_max_rows=macrobatch_size,
        queue_maxsize=macrobatch_size * 2,
    ) as writer:
        for i0 in tqdm.tqdm(range(0, len(ds), macrobatch_size), desc="Logit Batches"):
            input_ids = ds.select(range(i0, min(i0 + macrobatch_size, len(ds))))[
                "input_ids"
            ]
            input_ids = [x[:max_seq_len] for x in input_ids]
            for idx, req_out in enumerate(
                llm.generate(
                    [{"prompt_token_ids": x} for x in input_ids],
                    sampling_params=sampling_params,
                )
            ):
                top_indices, top_values = process_prompt_logprobs(
                    req_out.prompt_logprobs, k=k + 1
                )
                top_indices.unsqueeze_(0)
                top_values.unsqueeze_(0)

                row_out = compressor.compress_from_sparse(
                    # skip first token, which is always the prompt token
                    top_indices[..., 1 : k + 1],
                    top_values[..., 1 : k + 1],
                )

                input_ids_list = input_ids[idx][:max_seq_len]
                compressed_logprobs_list = (
                    row_out["compressed_logprobs"].squeeze(0).tolist()
                )
                bytepacked_indices_list = (
                    row_out["bytepacked_indices"].squeeze(0).tolist()
                )

                writer.write(
                    {
                        "input_ids": input_ids_list,
                        "compressed_logprobs": compressed_logprobs_list,
                        "bytepacked_indices": bytepacked_indices_list,
                    }
                )

    logging.info(f"Logits saved to {output}")
    del llm


def process_prompt_logprobs(
    prompt_logprobs: PromptLogprobs, k: int
) -> tuple[torch.LongTensor, torch.Tensor]:
    valid_logprobs = [lp for lp in prompt_logprobs if lp is not None]

    if not valid_logprobs:
        return torch.empty((0, 0), dtype=torch.long, device="cuda"), torch.empty(
            (0, 0), dtype=torch.float32, device="cuda"
        )

    num_prompt_tokens = len(valid_logprobs)
    total_elements = num_prompt_tokens * k
    if total_elements == 0:
        return torch.empty((0, k), dtype=torch.long, device="cuda"), torch.empty(
            (0, k), dtype=torch.float32, device="cuda"
        )
    np_indices = np.empty(total_elements, dtype=np.int64)
    np_values = np.empty(total_elements, dtype=np.float32)

    current_idx = 0
    for logprobs_at_token in valid_logprobs:  # list[tuple[int, float]]
        assert len(logprobs_at_token) >= k
        for token_id, logprob in logprobs_at_token.items():
            np_indices[current_idx] = token_id
            np_values[current_idx] = logprob.logprob
            current_idx += 1

    top_indices = (
        torch.from_numpy(np_indices)
        .view(num_prompt_tokens, k)
        .to(device="cuda", non_blocking=True)
    )
    top_values = (
        torch.from_numpy(np_values)
        .view(num_prompt_tokens, k)
        .to(device="cuda", non_blocking=True)
    )

    return top_indices, top_values


if __name__ == "__main__":
    sample_logits()
