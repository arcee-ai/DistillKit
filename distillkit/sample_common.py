import json
import os
import queue
import threading
from typing import Any

import datasets
import pyarrow
import pyarrow.parquet as pq
import torch
import transformers

ROLE_MAP = {
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


def maybe_trim_bos(text: str, tokenizer: transformers.PreTrainedTokenizerBase):
    if (
        tokenizer.bos_token_id is not None
        and getattr(tokenizer, "add_bos_token", False)
        and text.startswith(tokenizer.bos_token)
    ):
        return text[len(tokenizer.bos_token) :]
    return text


def do_chat_template(row: dict, tokenizer: transformers.PreTrainedTokenizerBase):
    if "text" in row:
        return row["text"]
    elif "instruction" in row and "output" in row:
        res = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]},
            ],
            tokenize=False,
        )
        return maybe_trim_bos(res, tokenizer)
    elif "inputs" in row and "targets" in row:
        res = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["inputs"]},
                {"role": "assistant", "content": row["targets"]},
            ],
            tokenize=False,
        )
        return maybe_trim_bos(res, tokenizer)
    elif "tools" in row and "turns" in row:
        tool_defs = [json.loads(tool) for tool in row["tools"]]
        turns = []
        for turn in row["turns"]:
            turn_out = {
                key: turn[key]
                for key in ["role", "content", "tool_calls"]
                if (key in turn and turn[key] is not None)
            }
            if "content" not in turn_out:
                turn_out["content"] = ""
            if "tool_calls" in turn_out:
                old_tcs = list(turn_out["tool_calls"])
                new_tcs = []
                for tc in old_tcs:
                    tool_name = tc["name"]
                    arguments = json.loads(tc["arguments"])
                    tc_out = {
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                    new_tcs.append(tc_out)
                turn_out["tool_calls"] = new_tcs
            turns.append(turn_out)
        res = tokenizer.apply_chat_template(turns, tools=tool_defs, tokenize=False)
        return maybe_trim_bos(res, tokenizer)
    elif "conversations" in row:
        msgs = [
            {"role": ROLE_MAP[msg["from"]], "content": msg["value"]}
            for msg in row["conversations"]
        ]
        res = tokenizer.apply_chat_template(msgs, tokenize=False)
        return maybe_trim_bos(res, tokenizer)
    elif "messages" in row:
        res = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return maybe_trim_bos(res, tokenizer)
    else:
        raise ValueError("row must contain 'text' or 'conversations' or 'messages' key")


def load_preprocess_data(
    *,
    dataset: str,
    configuration: str | None,
    split: str,
    samples: int | None,
    seed: int,
    max_seq_len: int,
    tokenizer: transformers.PreTrainedTokenizerBase,
    add_extra_pad_token: bool = False,
    apply_chat_template: bool = False,
):
    ds = datasets.load_dataset(dataset, name=configuration, split=split)
    ds = ds.shuffle(seed=seed)
    if samples is not None:
        ds = ds.select(range(samples))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if apply_chat_template:
        ds = ds.map(
            lambda x: {
                "text": do_chat_template(x, tokenizer),
            },
            num_proc=64,
        )
    ds = ds.filter(lambda row: row["text"] and row["text"].strip(), num_proc=64)
    ds = ds.map(
        lambda x: {
            "input_ids": truncate_tokens(
                x["text"], tokenizer, max_seq_len, add_extra_pad_token
            )
        },
        num_proc=64,
    ).filter(lambda x: len(x["input_ids"]) > 0, num_proc=64)
    return ds


def truncate_tokens(
    text: str, tokenizer, max_seq_len: int, add_extra_pad_token: bool = False
):
    tokens: torch.Tensor = tokenizer(text, return_tensors="pt")["input_ids"][0]
    if (
        add_extra_pad_token
        and tokens.shape[0] < max_seq_len
        and tokens.shape[0] > 0
        and tokens[-1] != tokenizer.pad_token_id
    ):
        # add single padding token
        # so that we don't have to look at sampled_logprobs and can
        # just stick with prompt_logprobs
        tokens = torch.cat(
            [
                tokens,
                torch.tensor(
                    [tokenizer.pad_token_id],
                    dtype=torch.long,
                    device=tokens.device,
                ),
            ],
            dim=0,
        )
    return tokens[:max_seq_len]


class StreamingParquetWriter:
    def __init__(
        self,
        output_path: str,
        schema: pyarrow.Schema,
        file_max_rows: int,
        write_batch_size: int = 1000,
        queue_maxsize: int | None = None,
    ):
        """
        Initializes the StreamingParquetWriter.

        Args:
            output_path (str): The directory where Parquet files will be saved.
            schema (pyarrow.Schema): The schema of the Parquet files.
            file_max_rows (int): The maximum number of rows per Parquet file.
            write_batch_size (int): The number of rows to buffer in memory before writing to disk.
                                   A larger batch size can improve performance but uses more memory.
        """
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)  # Ensure output directory exists
        self.schema = schema

        if not (file_max_rows > 0):
            raise ValueError("file_max_rows must be a positive integer.")
        if not (write_batch_size > 0):
            raise ValueError("write_batch_size must be a positive integer.")

        self.file_max_rows = file_max_rows
        self.write_batch_size = write_batch_size

        self.pq_writer = None  # The actual pyarrow.parquet.ParquetWriter instance
        self._current_rows_in_physical_file = (
            0  # Tracks rows written to the currently open .parquet file
        )
        self.file_index = (
            0  # Used for naming output files (data_0.parquet, data_1.parquet, ...)
        )

        self._write_queue = queue.Queue(maxsize=queue_maxsize)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._shutdown_event = threading.Event()

    def start(self):
        self._writer_thread.start()

    def _ensure_writer_open(self):
        """Opens a new Parquet file writer if one is not already open."""
        if self.pq_writer is None:
            file_path = os.path.join(
                self.output_path, f"data_{self.file_index}.parquet"
            )
            self.pq_writer = pq.ParquetWriter(file_path, schema=self.schema)
            self._current_rows_in_physical_file = 0  # Reset row count for the new file

    def _write_batch_to_parquet(self, batch_data: list[dict[str, Any]]):
        if not batch_data:
            return

        self._ensure_writer_open()

        # Convert list of dicts to columnar for pyarrow.Table
        # This assumes batch_data is a list of row_dicts
        columnar_data = {name: [] for name in self.schema.names}
        for row_dict in batch_data:
            for name in self.schema.names:
                columnar_data[name].append(row_dict[name])

        arrays = []
        for name in self.schema.names:
            field_type = self.schema.field(name).type
            # Data from queue should be CPU numpy arrays or Python lists/values
            # If tensors were put on queue, .cpu().numpy() here
            # For example, if 'input_ids' was a tensor:
            # if name == 'input_ids' and isinstance(columnar_data[name][0], torch.Tensor):
            #    data_to_convert = [t.cpu().numpy() for t in columnar_data[name]]
            # else:
            #    data_to_convert = columnar_data[name]
            # arrays.append(pyarrow.array(data_to_convert, type=field_type))
            arrays.append(pyarrow.array(columnar_data[name], type=field_type))

        table = pyarrow.Table.from_arrays(arrays, schema=self.schema)
        self.pq_writer.write_table(table)
        self._current_rows_in_physical_file += len(batch_data)

        if self._current_rows_in_physical_file >= self.file_max_rows:
            if self.pq_writer is not None:
                self.pq_writer.close()
                self.pq_writer = None
            self.file_index += 1

    def _writer_loop(self):
        batch_to_write = []
        while not self._shutdown_event.is_set() or not self._write_queue.empty():
            try:
                # Wait for a short timeout to check shutdown_event periodically
                row_data = self._write_queue.get(timeout=0.1)
                if row_data is None:  # Sentinel for shutdown
                    self._write_queue.task_done()
                    break
                batch_to_write.append(row_data)
                self._write_queue.task_done()

                if len(batch_to_write) >= self.write_batch_size:
                    self._write_batch_to_parquet(batch_to_write)
                    batch_to_write = []
            except queue.Empty:
                continue  # Loop again to check shutdown_event or new items

        # Flush any remaining items after loop ends
        if batch_to_write:
            self._write_batch_to_parquet(batch_to_write)

        if self.pq_writer is not None:
            self.pq_writer.close()
            self.pq_writer = None

    def write(self, row_data: dict[str, Any]):
        """
        Adds a single row of data (as a dictionary) to the write queue.
        The dictionary keys should match schema names.
        Values should be CPU data (Python lists/values, or NumPy arrays).
        Tensors should be .cpu().numpy() or .tolist() BEFORE putting on queue.
        """
        # Expects row_data to be a dictionary now for clarity
        # e.g., {"input_ids": [...], "compressed_logprobs": [...], ...}
        # Conversion of tensors to lists/numpy happens *before* this call.
        self._write_queue.put(row_data)

    def close(self):
        # Signal shutdown and wait for writer thread
        if self._writer_thread.is_alive():
            self._write_queue.put(None)  # Sentinel
            self._shutdown_event.set()
            self._writer_thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()
        return False


def legacy_compressed_logit_schema() -> pyarrow.Schema:
    return pyarrow.schema(
        [
            pyarrow.field("input_ids", pyarrow.list_(pyarrow.uint64())),
            pyarrow.field(
                "packed_indices", pyarrow.list_(pyarrow.list_(pyarrow.uint64()))
            ),
            pyarrow.field(
                "exact_values", pyarrow.list_(pyarrow.list_(pyarrow.float32()))
            ),
            pyarrow.field("coeffs", pyarrow.list_(pyarrow.list_(pyarrow.float32()))),
        ]
    )


def compressed_logit_schema() -> pyarrow.Schema:
    return pyarrow.schema(
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
