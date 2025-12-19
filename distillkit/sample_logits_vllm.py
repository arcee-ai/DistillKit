import os
import sys

import click
import transformers

try:
    import vllm
except ImportError:
    raise ImportError("VLLM must be installed to use this script.")
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
import tqdm
import yaml
from vllm.logprobs import FlatLogprobs, PromptLogprobs

from distillkit.compression import DistributionQuantizationConfig, LogprobCompressor
from distillkit.sample_common import (
    StreamingParquetWriter,
    compressed_logit_schema,
    load_preprocess_data,
)


@click.command("sample-logits")
@click.option("--model", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--dataset-configuration", type=str, default=None)
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
@click.option(
    "--enable-expert-parallel/--no-enable-expert-parallel", type=bool, default=False
)
@click.option("--dtype", type=str, default=None)
@click.option("--quantization", type=str, default=None)
@click.option("--trust-remote-code/--no-trust-remote-code", default=False)
@click.option("--gpu-memory-utilization", type=float, default=0.9)
@click.option("--compression-config", type=str, required=True)
@click.option("--macrobatch-size", type=int, default=256)
@click.option("--max-workers", type=int, default=None)
@click.option("--auto-vocab-size/--no-auto-vocab-size", type=bool, default=True)
def sample_logits(
    model: str,
    dataset: str,
    dataset_configuration: str | None,
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
    enable_expert_parallel: bool,
    dtype: str | None,
    quantization: str | None,
    trust_remote_code: bool,
    gpu_memory_utilization: float,
    compression_config: str,
    macrobatch_size: int,
    max_workers: int | None,
    auto_vocab_size: bool,
):
    logging.basicConfig(level=logging.INFO)

    tok = transformers.AutoTokenizer.from_pretrained(
        tokenizer or model, trust_remote_code=trust_remote_code
    )

    # load compression config
    with open(compression_config, "r") as f:
        cfg = DistributionQuantizationConfig.model_validate(yaml.safe_load(f))
    k = cfg.k

    tok_vocab = tok.get_vocab()
    tok_vocab_size = max(len(tok_vocab) + 1, max(tok_vocab.values()))
    if cfg.d != tok_vocab_size:
        if auto_vocab_size:
            cfg.d = tok_vocab_size
            logging.warning(
                f"Automatically set compressor vocab size to {tok_vocab_size}"
            )
        elif cfg.d < tok_vocab_size:
            logging.error("Compression config has too small vocabulary size!")
            logging.error(
                f"cfg.d: {cfg.d}, effective tokenizer vocab size: {tok_vocab_size}"
            )
            sys.exit(-1)
        elif (
            abs(cfg.d - tok_vocab_size) > 32
        ):  # allow a little wiggle room for common padding
            logging.warning(
                f"Vocabulary size in compression config ({cfg.d}) is larger than needed ({tok_vocab_size}). "
                "This will work but may consume more space than needed - double check that this is what you want."
            )

    os.makedirs(output, exist_ok=True)
    with open(
        os.path.join(output, "compression_config.yaml"), "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f)

    logging.info(f"Loading and preprocessing data from {dataset} ({split})")
    ds = load_preprocess_data(
        dataset=dataset,
        configuration=dataset_configuration,
        split=split,
        samples=samples,
        seed=seed,
        max_seq_len=max_seq_len,
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
        enable_expert_parallel=enable_expert_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        max_logprobs=k,
        logprobs_mode="raw_logprobs",
        max_model_len=max_model_len,
    )

    compressor = LogprobCompressor(
        config=cfg,
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
        max_tokens=1,  # vLLM wants at least 1 generated token
        detokenize=False,
        skip_special_tokens=False,
    )

    logging.info(f"Generating logits for {len(ds)} samples")

    def process_and_write_sample(
        req_out: vllm.RequestOutput,
        input_ids_sample: list[int],
        k: int,
        compressor: LogprobCompressor,
        writer: StreamingParquetWriter,
    ) -> None:
        """Process a single sample: extract logprobs, compress, and write to disk."""
        top_indices, top_values = process_prompt_logprobs(req_out.prompt_logprobs, k=k)
        top_indices.unsqueeze_(0)
        top_values.unsqueeze_(0)

        row_out = compressor.compress_from_sparse(
            top_indices,
            top_values,
        )

        compressed_logprobs_list = (
            row_out["compressed_logprobs"].cpu().squeeze(0).tolist()
        )
        bytepacked_indices_list = (
            row_out["bytepacked_indices"].cpu().squeeze(0).tolist()
        )

        writer.write(
            {
                "input_ids": input_ids_sample,
                "compressed_logprobs": compressed_logprobs_list,
                "bytepacked_indices": bytepacked_indices_list,
            }
        )

    try:
        with StreamingParquetWriter(
            output,
            schema=compressed_logit_schema(),
            file_max_rows=macrobatch_size,
            queue_maxsize=macrobatch_size * 2,
        ) as writer:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for i0 in tqdm.tqdm(
                    range(0, len(ds), macrobatch_size), desc="Logit Batches"
                ):
                    batch_input_ids = ds[i0 : i0 + macrobatch_size]["input_ids"]
                    # Submit CPU processing tasks to background thread
                    for idx, req_out in enumerate(
                        llm.generate(
                            [{"prompt_token_ids": x} for x in batch_input_ids],
                            sampling_params=sampling_params,
                        )
                    ):
                        future = executor.submit(
                            process_and_write_sample,
                            req_out,
                            batch_input_ids[idx],
                            k,
                            compressor,
                            writer,
                        )
                        futures.append(future)

                    # Limit writes in flight to avoid unbounded memory growth
                    while len(futures) > macrobatch_size * 2:
                        futures.pop(0).result()

                for future in futures:
                    future.result()

        logging.info(f"Logits saved to {output}")
    finally:
        del llm


def process_prompt_logprobs(
    prompt_logprobs: PromptLogprobs, k: int
) -> tuple[torch.LongTensor, torch.Tensor]:
    # Fast path: directly access FlatLogprobs data without materializing dicts
    if isinstance(prompt_logprobs, FlatLogprobs):
        # Skip first position if it's empty (first token has no logprobs)
        start_pos = 0
        if len(prompt_logprobs) > 0:
            first_start = prompt_logprobs.start_indices[0]
            first_end = prompt_logprobs.end_indices[0]
            if first_end - first_start == 0:
                start_pos = 1

        num_prompt_tokens = len(prompt_logprobs) - start_pos
        if num_prompt_tokens <= 0:
            return torch.empty((0, 0), dtype=torch.long), torch.empty(
                (0, 0), dtype=torch.float32
            )

        top_indices = torch.empty(
            (num_prompt_tokens, k), dtype=torch.long, device="cpu"
        )
        top_values = torch.full(
            (num_prompt_tokens, k),
            fill_value=float("-inf"),
            dtype=torch.float32,
            device="cpu",
        )

        # Build index arrays for vectorized assignment
        seq_ids = []
        rank_ids = []
        token_ids_to_copy = []
        logprobs_to_copy = []

        for pos_id in range(start_pos, len(prompt_logprobs)):
            seq_id = pos_id - start_pos
            start_idx = prompt_logprobs.start_indices[pos_id]
            end_idx = prompt_logprobs.end_indices[pos_id]

            for i in range(start_idx, end_idx):
                rank = prompt_logprobs.ranks[i]
                if rank is None or rank > k:
                    # None: vLLM returns the actual prompt token even when not in top-k
                    # rank > k: Truncate to only the k values we requested
                    continue
                seq_ids.append(seq_id)
                rank_ids.append(rank - 1)
                token_ids_to_copy.append(prompt_logprobs.token_ids[i])
                logprobs_to_copy.append(prompt_logprobs.logprobs[i])

        # Vectorized assignment using advanced indexing
        if seq_ids:
            seq_idx_tensor = torch.tensor(seq_ids, dtype=torch.long)
            rank_idx_tensor = torch.tensor(rank_ids, dtype=torch.long)
            top_indices[seq_idx_tensor, rank_idx_tensor] = torch.tensor(
                token_ids_to_copy, dtype=top_indices.dtype, device=top_indices.device
            )
            top_values[seq_idx_tensor, rank_idx_tensor] = torch.tensor(
                logprobs_to_copy, dtype=top_values.dtype, device=top_values.device
            )

        return top_indices, top_values

    # Slow path: handle legacy list format
    else:
        valid_logprobs = [lp for lp in prompt_logprobs]
        if valid_logprobs[0] is None or len(valid_logprobs[0]) < 1:
            valid_logprobs.pop(0)

        if not valid_logprobs:
            return torch.empty((0, 0), dtype=torch.long), torch.empty(
                (0, 0), dtype=torch.float32
            )

        num_prompt_tokens = len(valid_logprobs)

        top_indices = torch.empty((num_prompt_tokens, k), dtype=torch.long)
        top_values = torch.full(
            (num_prompt_tokens, k), fill_value=float("-inf"), dtype=torch.float32
        )
        for seq_id, logprobs in enumerate(valid_logprobs):
            assert logprobs is not None, (
                f"Missing logprobs for token at position {seq_id + 1} (expected logprobs for all non-first tokens)"
            )
            for tok_id, logprob in logprobs.items():
                if logprob.rank is None or logprob.rank > k:
                    continue
                top_indices[seq_id, logprob.rank - 1] = tok_id
                top_values[seq_id, logprob.rank - 1] = logprob.logprob

        return top_indices, top_values


if __name__ == "__main__":
    sample_logits()
