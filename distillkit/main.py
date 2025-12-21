# Copyright 2025 Arcee AI
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
from typing import Any

import click
import datasets
import torch
import transformers
import trl
import yaml
from accelerate import Accelerator

from distillkit.compression import LogprobCompressor
from distillkit.configuration import (
    DatasetConfiguration,
    DatasetPath,
    DistillationRunConfig,
    HfRepoDataset,
    LocalDataset,
    TeacherDatasetConfig,
    TeacherModelConfig,
    TeacherVLLMConfig,
)
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.monkey_patch_packing import monkey_patch_packing_for_model
from distillkit.signals import OfflineSignalSource, OnlineSignalSource, SignalSource
from distillkit.trainer import DistillationTrainer

LOG = logging.getLogger(__name__)


def _format_row(
    example: dict[str, Any], tokenizer: transformers.PreTrainedTokenizer
) -> dict[str, Any]:
    if ("input_ids" in example) or ("text" in example):
        # either pretokenized or raw completion - no formatting needed
        return {}
    elif "conversations" in example:
        conversations = example["conversations"]

        messages = []
        for conversation in conversations:
            role_map = {
                "human": "user",
                "user": "user",
                "gpt": "assistant",
                "assistant": "assistant",
                "system": "system",
            }
            role = role_map.get(conversation.get("from", ""), None)
            if role:
                messages.append(
                    {"role": role, "content": conversation.get("value", "")}
                )

        # Apply chat template to create a single string. SFTTrainer will handle tokenization.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    elif "messages" in example:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    else:
        raise RuntimeError("Expected `text`, `messages`, or `conversations` column")


def _load_dataset(
    path: DatasetPath,
    seed: int | None,
    num_samples: int | None,
    tokenizer: transformers.PreTrainedTokenizer,
    prepared_dataset_path: str | None = None,
    keep_in_memory: bool | None = None,
    prepacked: bool = False,
) -> datasets.Dataset:
    if prepared_dataset_path:
        honk = json.dumps(
            {
                "path": path.model_dump(),
                "seed": seed,
                "num_samples": num_samples,
            }
        )
        logging.info(f"Dataset spec: {honk}")
        ds_hash = hashlib.sha256(honk.encode()).hexdigest()
        full_prepared_path = os.path.join(prepared_dataset_path, f"dataset-{ds_hash}")
        if os.path.exists(full_prepared_path):
            return datasets.load_from_disk(full_prepared_path)
    else:
        full_prepared_path = None
    if isinstance(path, HfRepoDataset):
        res = datasets.load_dataset(
            path.repo_id,
            name=path.config_name,
            revision=path.revision,
            split=path.split,
            keep_in_memory=keep_in_memory,
        )
    elif isinstance(path, LocalDataset):
        res = datasets.load_from_disk(path.disk_path, keep_in_memory=keep_in_memory)
        if path.split:
            res = res[path.split]
        elif isinstance(res, datasets.DatasetDict):
            raise ValueError(
                "Dataset dict found but no split specified. Please specify a split."
            )
    else:
        raise ValueError(
            "Unsupported dataset type. Please provide a valid Hugging Face repo ID or local dataset path."
        )

    if prepacked:
        last_idx = len(res) - 1
        while len(res) >= 2 and len(res[last_idx]["input_ids"]) != len(
            res[0]["input_ids"]
        ):
            last_idx -= 1
        if last_idx <= 0:
            raise RuntimeError("Dataset config is probs wrong")
        res = res.select(range(last_idx + 1))

    if seed:
        res = res.shuffle(seed=seed)
    if num_samples:
        res = res.select(range(num_samples))
    if (
        (not prepacked)
        and ("text" not in res.column_names)
        and ("input_ids" not in res.column_names)
    ):
        res = res.map(
            _format_row,
            remove_columns=res.column_names,
            fn_kwargs={"tokenizer": tokenizer},
        )
    if full_prepared_path:
        os.makedirs(full_prepared_path, exist_ok=True)
        logging.info(
            f"Saving prepared dataset to {full_prepared_path} (hash: {ds_hash}, path: {path}, seed: {seed}, num_samples: {num_samples})"
        )
        res.save_to_disk(full_prepared_path)
        del res
        return datasets.load_from_disk(
            full_prepared_path, keep_in_memory=keep_in_memory
        )
    return res


def load_data(
    config: DatasetConfiguration,
    tokenizer: transformers.PreTrainedTokenizer,
    keep_in_memory: bool | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """
    Load the train (and optionally eval) datasets as specified in the configuration.
    """

    LOG.info(
        f"Loading datasets: {config.train_dataset} (train), {config.eval_dataset} (eval)"
    )
    ds_train = _load_dataset(
        config.train_dataset,
        config.seed,
        config.num_samples,
        tokenizer=tokenizer,
        prepared_dataset_path=config.prepared_dataset_path,
        keep_in_memory=keep_in_memory,
        prepacked=config.prepacked,
    )
    ds_eval = None
    if config.eval_dataset:
        ds_eval = _load_dataset(
            config.eval_dataset,
            config.seed,
            config.num_eval_samples,
            tokenizer=tokenizer,
            prepared_dataset_path=config.prepared_dataset_path,
            keep_in_memory=keep_in_memory,
            prepacked=config.prepacked,
        )
    return ds_train, ds_eval


def load_student_model(
    config: DistillationRunConfig,
    tokenizer_vocab_size: int,
) -> transformers.PreTrainedModel:
    if config.functionary_packing:
        monkey_patch_packing_for_model(config.train_model)
    auto_cls = getattr(transformers, config.model_auto_class, None)
    if auto_cls is None:
        raise ValueError(
            f"Model class {config.model_auto_class} not found in transformers."
        )
    LOG.info(f"Loading model {config.train_model} with class {auto_cls}")
    extra_kwargs = {"trust_remote_code": config.trust_remote_code}
    if config.use_flash_attention:
        extra_kwargs["attn_implementation"] = "flash_attention_2"
        extra_kwargs["torch_dtype"] = torch.bfloat16
    model = auto_cls.from_pretrained(
        config.train_model,
        **extra_kwargs,
        **config.model_kwargs,
    )
    LOG.info("Loaded model.")

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    if (
        model_vocab_size != tokenizer_vocab_size
        or config.resize_embeddings_to_multiple_of
    ):
        model.resize_token_embeddings(
            tokenizer_vocab_size,
            pad_to_multiple_of=config.resize_embeddings_to_multiple_of,
        )
        new_model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if new_model_vocab_size != model_vocab_size:
            LOG.info(
                f"Resized model vocab size from {model_vocab_size} to {new_model_vocab_size}"
            )

    model: transformers.PreTrainedModel
    if config.frozen_modules:
        module_set = set(config.frozen_modules)
        seen = set()
        for name, module in model.named_modules():
            if name in module_set:
                module.requires_grad_(False)
                seen.add(name)
        unseen = module_set - seen
        LOG.info(f"Froze {len(seen)} modules")
        if unseen:
            raise ValueError(f"Frozen modules not found in model: {', '.join(unseen)}")
    if config.frozen_res:
        num_frozen = 0
        frozen_res = [re.compile(s) for s in config.frozen_res]
        for name, param in model.named_parameters():
            if any(fre.search(name) for fre in frozen_res):
                param.requires_grad = False
                num_frozen += 1
        if num_frozen:
            print(f"Froze {num_frozen} tensors by regular expression")
    return model


def create_signal_source(
    config: DistillationRunConfig, vocab_size: int
) -> tuple[SignalSource, mp.Process | None]:
    """
    Create signal source for teacher.

    Returns:
        (signal_source, server_process) where server_process is None for
        offline/HF teachers, or the vLLM server process for vLLM teacher.
    """
    if isinstance(config.teacher, TeacherDatasetConfig):
        compressor = LogprobCompressor(
            config=config.teacher.logprob_compressor,
            legacy_config=config.teacher.legacy_logit_compression,
        )
        return OfflineSignalSource(compressor, vocab_size=vocab_size), None

    elif isinstance(config.teacher, TeacherModelConfig):
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.teacher.path, **(config.teacher.kwargs or {})
        )
        return (
            OnlineSignalSource(
                teacher_model, vocab_size=vocab_size, sparsify_top_k=config.teacher.top_k
            ),
            None,
        )

    elif isinstance(config.teacher, TeacherVLLMConfig):
        # Import here to avoid circular dependency and to fail fast if vLLM not installed
        from distillkit.vllm_server import VLLMTeacherServer
        from distillkit.vllm_signal_source import VLLMOnlineSignalSource

        # Use spawn context for clean CUDA isolation
        # This ensures teacher server can set CUDA_VISIBLE_DEVICES before CUDA init
        mp_ctx = mp.get_context("spawn")

        # Create IPC primitives
        request_queue = mp_ctx.Queue(maxsize=16)
        response_queue = mp_ctx.Queue(maxsize=16)
        ready_event = mp_ctx.Event()
        shutdown_event = mp_ctx.Event()

        # Serialize config for subprocess
        config_dict = {
            "model_path": config.teacher.model_path,
            "top_k": config.teacher.top_k,
            "teacher_gpu_ids": config.teacher.teacher_gpu_ids,
            "tensor_parallel_size": config.teacher.tensor_parallel_size,
            "dtype": config.teacher.dtype,
            "quantization": config.teacher.quantization,
            "gpu_memory_utilization": config.teacher.gpu_memory_utilization,
            "max_model_len": config.teacher.max_model_len,
            "trust_remote_code": config.teacher.trust_remote_code,
            "cache_size_mb": config.teacher.cache_size_mb,
        }

        # Create teacher server instance
        server = VLLMTeacherServer(
            config=config_dict,
            request_queue=request_queue,
            response_queue=response_queue,
            ready_event=ready_event,
            shutdown_event=shutdown_event,
        )

        # Start server process
        LOG.info(f"Starting vLLM teacher server on GPUs {config.teacher.teacher_gpu_ids}...")
        server_process = mp_ctx.Process(target=server.run, daemon=False)
        server_process.start()

        # Wait for server to initialize
        if not ready_event.wait(timeout=120):
            server_process.terminate()
            server_process.join()
            raise RuntimeError(
                "vLLM teacher server failed to initialize within 120 seconds. "
                "Check that the model can be loaded and GPUs are available."
            )
        LOG.info("vLLM teacher server ready")

        # Create signal source
        signal_source = VLLMOnlineSignalSource(
            vocab_size=vocab_size,
            top_k=config.teacher.top_k,
            server_process=server_process,
            request_queue=request_queue,
            response_queue=response_queue,
            shutdown_event=shutdown_event,
            request_timeout_sec=config.teacher.request_timeout_sec,
        )

        return signal_source, server_process

    else:
        raise RuntimeError("Teacher configuration invalid")


def collate_packed_batch(examples):
    # all sequences in the batch already have the same length
    # so we can directly stack them
    return {
        key: torch.tensor([example[key] for example in examples])
        for key in examples[0].keys()
    }


def load_tokenizer(config: DistillationRunConfig) -> transformers.PreTrainedTokenizer:
    if isinstance(config.teacher, TeacherModelConfig):
        src_path = config.teacher.path
        logging.info("Using teacher's tokenizer")
    else:
        src_path = config.train_model
        logging.info("Using student's tokenizer")
    return transformers.AutoTokenizer.from_pretrained(
        src_path,
        trust_remote_code=config.trust_remote_code,
    )


def do_distill(config: DistillationRunConfig, config_source: str | None = None):
    os.makedirs(config.output_path, exist_ok=True)
    if config_source is None:
        config_source = yaml.safe_dump(config.model_dump(mode="json", by_alias=True))
    with open(os.path.join(config.output_path, "distillkit_config.yaml"), "w") as f:
        f.write(config_source)

    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name

    accelerator = Accelerator()
    with accelerator.main_process_first():
        tokenizer = load_tokenizer(config)
        ds_train, ds_eval = load_data(config.dataset, tokenizer)

        tokenizer_vocab_size = max(
            len(tokenizer.get_vocab()),
            max(tokenizer.get_vocab().values()) + 1,
        )

    model = load_student_model(config, tokenizer_vocab_size)

    config_kwargs = dict(config.training_args)
    dataset_kwargs = config_kwargs.pop("dataset_kwargs", {})
    if config.dataset.prepacked:
        dataset_kwargs["skip_prepare_dataset"] = True
    max_length = config_kwargs.pop("max_length", config.sequence_length)
    training_arguments = trl.SFTConfig(
        **config_kwargs,
        max_length=max_length,
        output_dir=config.output_path,
        dataset_kwargs=dataset_kwargs,
    )

    signal_source, server_process = create_signal_source(config, tokenizer_vocab_size)

    try:
        if config.layer_mapping is not None:
            if not isinstance(signal_source, OnlineSignalSource):
                raise RuntimeError(
                    "Hidden state distillation not supported for offline teachers"
                )
            teacher_hidden_size = signal_source.teacher_model.config.hidden_size
            if config.layer_mapping == "all":
                mapping = [(i, i) for i in range(model.config.num_hidden_layers)]
            else:
                mapping = config.layer_mapping
            hsm = HiddenStateMapping(
                student=model,
                teacher_hidden_size=teacher_hidden_size,
                layer_mapping=mapping,
                force_projection=config.force_hidden_state_projection,
            )
        else:
            hsm = None
        trainer = DistillationTrainer(
            model=model,
            config=config,
            signal_source=signal_source,
            hidden_state_mapping=hsm,
            true_vocab_size=tokenizer_vocab_size,
            train_dataset=ds_train,
            eval_dataset=ds_eval,
            args=training_arguments,
            data_collator=collate_packed_batch if config.dataset.prepacked else None,
            processing_class=None if config.dataset.prepacked else tokenizer,
        )

        resume_from_checkpoint = config.training_args.get("resume_from_checkpoint", None)

        LOG.info("Starting training.")
        trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
        )
        LOG.info(f"Finished training. Saving model to {config.output_path}.")
        trainer.save_model(config.output_path)
        LOG.info("Done.")

    finally:
        # Clean up teacher server if running
        if server_process is not None:
            if hasattr(signal_source, "shutdown"):
                signal_source.shutdown()
            else:
                LOG.warning("Signal source does not have shutdown method, terminating server")
                server_process.terminate()
                server_process.join()


@click.command("distillkit-offline")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    count=True,
    help="Increase verbosity of logging. Use -vv for debug level.",
)
def main(config_path: str, verbosity: int):
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DistillationRunConfig.model_validate(config_dict)
    do_distill(config)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
