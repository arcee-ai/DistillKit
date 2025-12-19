# Running vLLM Server for Online Distillation

DistillKit supports online distillation using a vLLM server as the teacher model.

## Why vLLM for Online Distillation?

Using a vLLM server for online distillation offers several advantages:

- **Larger teachers**: Run teacher models that don't fit in the same GPU as the student
- **GPU isolation**: Separate GPU memory management for teacher and student
- **Efficiency**: Leverage vLLM's optimized inference engine with PagedAttention
- **Continuous batching**: Parallel async requests enable vLLM's continuous batching for maximum throughput
- **Flexibility**: Scale teacher serving independently (tensor parallelism, multiple GPUs)
- **Shared teachers**: Multiple training jobs can share the same teacher server

## Starting the vLLM Server

### 1. Install vLLM

```bash
pip install vllm>=0.12.0
```

### 2. Launch OpenAI-Compatible Server

Start the vLLM server:

```bash
vllm serve <teacher_model_path> \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-abc123
```

### Advanced Configuration

For optimal performance, configure tensor parallelism and GPU memory:

```bash
vllm serve <teacher_model_path> \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --dtype auto
```

**Configuration Options:**

- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (e.g., 2, 4, 8)
- `--pipeline-parallel-size`: Number of pipeline stages for very large models
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.0-1.0, default 0.9)
- `--max-model-len`: Maximum sequence length the model can handle
- `--dtype`: Data type (auto, float16, bfloat16, float32)
- `--quantization`: Enable quantization (awq, gptq, fp8, etc.)

### Example: Running Qwen2.5-72B

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --port 8000
```

## Verifying the Server

```bash
curl http://localhost:8000/v1/models
```

Expected response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-72B-Instruct",
      "object": "model",
      ...
    }
  ]
}
```

## Configuring DistillKit

Once your vLLM server is running, configure DistillKit to use it:

### Configuration YAML

```yaml
model: "your-student-model"

teacher:
  kind: vllm
  base_url: "http://localhost:8000"
  model: "your-teacher-model"
  top_k: 256
  timeout: 120.0
  max_retries: 3
  api_key: "token-abc123"  # As specified in your `vllm serve` command
  # api_key: ${MY_VLLM_API_KEY}  # Also can use env vars

dataset:
  train_dataset:
    repo_id: "your-dataset"
    split: "train"

sequence_length: 2048
output_path: "./output/distilled-model"

loss_functions:
  - function: kl
    weight: 1.0
    temperature: 2.0
    missing_probability_handling: zero
    sparse_chunk_length: 512
```

### Configuration Options

**Required:**
- `kind: vllm` - Specifies vLLM teacher source
- `base_url` - Base URL of vLLM server (e.g., "http://localhost:8000")
- `model` - Name/ID of model to use
- `top_k` - Number of top-k logprobs to request (e.g., 256, 512)

**Optional:**
- `api_key` - API key if server requires authentication
- `timeout` - Request timeout in seconds (default: 120.0)
- `max_retries` - Max retries for failed requests (default: 3)

## Training

Run distillation as normal:

```bash
distillkit examples/vllm_teacher.yml
```

DistillKit will:
1. Validate connectivity to vLLM server at startup
2. Send tokenized sequences to vLLM in parallel (async) during training
3. Receive top-k logprobs for each position
4. Compute distillation loss using teacher signals

**Note**: All sequences in a training batch are sent to vLLM concurrently, allowing vLLM to leverage its continuous batching for optimal throughput.

## How It Works: Async Parallel Requests

DistillKit uses `aiohttp` for async HTTP communication with the vLLM server. For each training batch:

1. **Parallel dispatch**: All sequences in the batch are sent to vLLM concurrently using `asyncio.gather()`
2. **Connection pooling**: A shared `TCPConnector` maintains persistent connections (limit: 100 total, 50 per host)
3. **vLLM continuous batching**: vLLM's scheduler batches incoming requests dynamically for efficient GPU utilization
4. **Concurrent processing**: While vLLM processes requests, the training process waits asynchronously (non-blocking)

This design maximizes throughput by keeping the vLLM server fully utilized while minimizing training overhead.

## Performance Tips

### 1. Optimize Top-K Value

The `top_k` parameter balances accuracy and performance:

- **Lower (4-32)**: Faster inference, less bandwidth, may lose signal
- **Medium (32-128)**: Good balance for most use cases
- **Higher (128+)**: More accurate but slower and more bandwidth

### 2. Batch Size Considerations

DistillKit sends all sequences in a training batch to vLLM in parallel using async HTTP requests. This allows vLLM's continuous batching to efficiently process multiple sequences together. Training batch size affects:

- **Larger batches**: More concurrent requests to vLLM, better utilization of continuous batching
- **Smaller batches**: Less parallelism, may underutilize vLLM server capacity
- **Optimal**: Match training batch size to vLLM's capacity (monitor with `--max-num-batched-tokens`)

**Example**: With `per_device_train_batch_size: 8`, DistillKit sends 8 parallel requests to vLLM, which can batch them together for efficient inference.

### 3. Network Optimization

For best performance:

- Run vLLM server on same node as training (low latency)
- Use high-bandwidth network if server is remote
- Monitor request latency with `--logging_steps`

### 4. GPU Memory Management

Separate teacher and student GPUs:

```bash
# Terminal 1: Start vLLM on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve large-teacher \
  --tensor-parallel-size 4

# Terminal 2: Train student on GPU 4
CUDA_VISIBLE_DEVICES=4 distillkit config.yml
```
