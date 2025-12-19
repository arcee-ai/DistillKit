# DistillKit

A flexible and production-ready toolkit for knowledge distillation of large language models, supporting both online and offline distillation workflows with advanced logit compression.

DistillKit powers the training of many of Arcee's popular open-source models, including [Virtuoso](https://huggingface.co/arcee-ai/Virtuoso-Large), [SuperNova Medius](https://huggingface.co/arcee-ai/SuperNova-Medius), and [Blitz](https://huggingface.co/arcee-ai/Arcee-Blitz).

## Features

- **Online Distillation**: Real-time teacher inference during student training
- **Offline Distillation**: Train from pre-captured teacher outputs with advanced compression
- **Advanced Logit Compression**: Novel polynomial approximation + quantization + bit-packing achieving vigorous compression ratios while preserving distillation quality
- **Flexible Loss Functions**: Composable losses including KL divergence, JSD, TVD, ranking losses, and hidden state alignment
- **Sparse & Dense Support**: Efficient sparse distributions (top-k) or exact dense distributions
- **Battle-tested**: The infrastructure powering Arcee's distilled model releases
- **HuggingFace Integration**: Built on Transformers, TRL, and Accelerate

## Why DistillKit?

While online distillation is straightforward, **offline distillation at scale** requires careful engineering. Simply storing top-k token-logit pairs becomes prohibitively expensive when distilling on billions of tokens.

DistillKit's compression system is the result of months of experimentation to strike the delicate balance between storage costs, memory throughput, and distillation quality. Our approach:

1. **Polynomial approximation** of the logit distribution curve
2. **Error-diffusion quantization** of residuals to preserve quality
3. **Bit-level packing** with arbitrary bit widths (1-64 bits)

This enables practical offline distillation workflows that would otherwise be infeasible.

## Installation

```bash
git clone https://github.com/arcee-ai/distillkit.git
cd distillkit
pip install -e .
```

### Optional: Logit Capture

To capture your own teacher outputs, install the capture dependencies:

```bash
pip install -e ".[capture]"
```

For most users, we recommend starting with the pre-captured teacher datasets we provide (see [Datasets](#datasets) below).

## Quick Start

### Offline Distillation

Train a student model using pre-captured teacher outputs:

```yaml
# config.yaml
project_name: my-distillation
model: Qwen/Qwen3-8B
output_path: ./output
sequence_length: 8192

dataset:
  train_dataset:
    repo_id: arcee-ai/Qwen3-235B-Logits-Packed-8192  # Pre-captured teacher outputs
    split: train
  prepacked: true

teacher:
  kind: dataset
  logprob_compressor:
    d: 151936  # Vocabulary size
    delta_encoding: true
    error_diffusion: false
    exact_dtype: float32
    exact_k: 32
    k: 128
    polynomial_terms: [0, 1, 2]
    residual_bins: []
    term_dtype: float32

loss_functions:
  - function: cross_entropy
    weight: 0.5
  - function: kl
    weight: 0.5
    temperature: 1.0
    missing_probability_handling: zero
    sparse_chunk_length: 1024

training_args:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-6
  bf16: true
  optim: adamw_torch
  gradient_checkpointing: true
```

Run training:

```bash
distillkit config.yaml
```

### Online Distillation

For online distillation where the teacher runs alongside student training, see [`examples/afm_test.yml`](examples/afm_test.yml) for a complete configuration example.

## Core Concepts

### Knowledge Distillation for LLMs

Knowledge distillation transfers knowledge from a (potentially larger) "teacher" model to a "student" model. Instead of training only on hard labels (the correct token), the student learns from the teacher's probability distribution over tokens, which is a much richer learning signal.

**Key benefits:**
- Smaller, faster models with competitive performance
- Lower inference costs
- Easier deployment in resource-constrained environments

### Online vs Offline Distillation

**Online Distillation:**
- Teacher runs in real-time during student training
- No storage overhead
- Best when: You have sufficient VRAM for both models and dense distributions

**Offline Distillation:**
- Teacher outputs pre-captured and compressed
- Enables training multiple students from the same teacher
- Best when: VRAM-limited, reusing teacher signals, or training at large scale

**Rule of thumb:** If you can fit both teacher and student with dense distributions into VRAM, use online distillation. Otherwise, offline distillation with our compression system is the way to go.

### Sparse vs Dense Distributions

**Dense distributions** include probabilities for the full vocabulary. This is more accurate but memory-intensive.

**Sparse distributions** store only the top-k tokens and serve as a lossy, but useful and efficient, approximation of the full dense distribution. With sufficient training data, sparse distillation can achieve equivalent performance to dense.

DistillKit supports both, with automatic chunking for memory-efficient processing of long sequences.

### Logit Compression

Our compression system balances storage efficiency with distillation quality:

1. Select top-k logits from teacher output
2. Sort by log-probability, optionally apply delta encoding
3. Fit polynomial to the distribution curve
4. Quantize residuals, with optional error diffusion
5. Bitpack everything into byte vectors

There are lots of knobs you can twiddle here to reach a storage/fidelity tradeoff that works for your particular needs.

**Recommended configuration** (used at Arcee for new captures):
```yaml
logprob_compressor:
  d: <your_vocab_size_here>
  k: 128
  exact_k: 16
  exact_dtype: bfloat16
  polynomial_terms: [0, 1, 2, 3, 4, "sqrt"]
  term_dtype: float32
  residual_bins: []
  delta_encoding: false
  error_diffusion: false
```

This takes ~300 bytes/token (0.15% of uncompressed distribution size) with minimal quality loss.

If you're a little tight on storage, try the **budget pick**:
```yaml
logprob_compressor:
  d: <your_vocab_size_here>
  k: 50
  exact_k: 1
  exact_dtype: bfloat16
  polynomial_terms: [0, 1, "sqrt"]
  term_dtype: float32
  residual_bins: []
  delta_encoding: false
  error_diffusion: false
```

This weighs in at around 114 bytes per token, smaller and with better reconstruction quality than storing the top 32 logprobs in bf16.

Note that the configuration that was used to capture the logits must be reflected in the distillation configuration. Mixing and matching isn't gonna work out so hot.

## Configuration Guide

### Loss Functions

DistillKit supports composable loss functions with independent weights:

#### Distribution-Based Losses
- `kl`: Kullback-Leibler divergence (standard distillation loss)
- `jsd`: Jensen-Shannon divergence (symmetric alternative to KL)
- `tvd`: Total Variation Distance

#### Ranking Losses
- `hinge`: Hinge ranking loss
- `logistic_ranking`: Logistic ranking loss

#### Hidden State Alignment
- `hs_mse`: Mean squared error between teacher and student hidden states
- `hs_cosine`: Cosine similarity between hidden states

#### Standard
- `cross_entropy`: Standard language modeling loss

All distribution losses support both sparse and dense modes. Combine multiple losses:

```yaml
loss_functions:
  - function: cross_entropy
    weight: 0.25
  - function: kl
    weight: 0.5
    temperature: 2.0
  - function: hs_cosine
    weight: 0.25
```

### Teacher Configuration

**Offline (from dataset):**
```yaml
teacher:
  kind: dataset
  logprob_compressor:
    d: 128256
    k: 128
    exact_k: 16
    delta_encoding: true
    ...
  # or:
  legacy_logit_compression:
    vocab_size: 128256
    k: 128
    exact_k: 32
    polynomial_degree: 8
    ...
```

**Online (HuggingFace model):**
```yaml
teacher:
  kind: hf
  path: Qwen/Qwen3-8B
  kwargs: # keyword arguments passed when loading teacher model
    attn_implementation: flash_attention_2
    torch_dtype: bfloat16
```

## Advanced Topics


### Compression Deep-Dive

The compression system supports two modes:

**Legacy compression** (fully polynomial-based):
```yaml
legacy_logit_compression:
  vocab_size: 128256       # Size of teacher vocabulary
  k: 128                   # Total number of logprobs per token, exact plus approximated
  exact_k: 32              # Number of logprobs stored as floating point values
  polynomial_degree: 8     # Degree of approximating polynomial
  with_sqrt_term: false    # Include sqrt term in polynomial
  term_dtype: float32      # Precision for polynomial coefficients
  invert_polynomial: true  # Invert for better numerical properties
```

**Distribution quantization** (newer, more flexible):
```yaml
logprob_compressor:
  d: 128256                # Size of teacher vocabulary
  k: 128                   # Total number of logprobs per token, exact plus approximated
  exact_k: 16              # Number of logprobs stored as floating point values
  exact_dtype: bfloat16    # dtype for "exact" logprobs
  delta_encoding: false    # Store logprobs as deltas (not recommended)
  error_diffusion: false   # Perform error diffusion to spread quantization error across values (not recommended)
  polynomial_terms:        # List of polynomial terms used for approximating tail
    - 0
    - 1
    - 2
    - "sqrt"
  term_dtype: float32      # dtype for storage of polynomial coefficients
  residual_bins:           # Optional list of bins storing quantized residuals vs. the approximated tail
    - scale_dtype: float16 # dtype for scale factor for this bin
      element_bits: 8      # Bits/element
      num_elements: 16     # Total number of elements in this bin
    - scale_dtype: float32 # bfloat16 also works
      element_bits: 2      # Can use any number of bits <= 64
      num_elements: 64
    ...
```

### Hidden State Distillation

Align student hidden states with teacher hidden states:

```yaml
layer_mapping: all  # Or specify layer pairs
loss_functions:
  - function: hs_mse
    weight: 0.5
```

For cross-architecture distillation, hidden states are projected using learned linear mappings. You can also enable this for same-architecture distillations by setting `force_hidden_state_projection: true`.

### Capturing Teacher Outputs

To create your own offline distillation dataset:

```bash
python -m distillkit.sample_logits_vllm \
  --model meta-llama/Llama-3.1-70B \
  --dataset allenai/tulu-3-sft-mixture \
  --output ./llama3_70b_tulu_logits/ \
  --compression-config ./compression_config.yaml
```

Requires vLLM (see [Installation](#optional-logit-capture)).

### Memory Management Tips

**For long sequences:**
- Use `sparse_chunk_length` to process sequences in chunks (e.g., `1024`)
- Use DeepSpeed ZeRO Stage 1 or 2 to cram more tokens in there

**For general savings:**
- Use `optim: paged_adamw_8bit` or `optim: adamw_bnb_8bit`
- Enable Flash Attention 2: `use_flash_attention: true`
- Use bfloat16 instead of float32
- Enable `gradient_checkpointing`
- Reduce batch size, increase gradient accumulation

## Examples

- **Offline Distillation (70B → 8B)**: [`examples/llama_70b_base.yml`](examples/llama_70b_base.yml)
- **Online Distillation with Hidden States**: [`examples/afm_test.yml`](examples/afm_test.yml)
- **Multimodal Model Distillation**: [`examples/mistral3.yaml`](examples/mistral3.yaml)

## Datasets

We're releasing several pre-captured teacher datasets:

* [Qwen3-235B instruction-following](https://huggingface.co/datasets/arcee-ai/Qwen3-235B-Logits-Packed-8192): ~1.5 billion tokens of general instruct data at 8192 context length
* [DeepSeek V3/R1 synthetic mixed-mode reasoning](https://huggingface.co/datasets/arcee-ai/DeepSeek-MixedModeReasoning-Logits-Packed-16384): ~5 billion tokens captured from DeepSeek V3 and R1, with prefixes to distinguish reasoning from non-reasoning traces - 16k context length
* [DeepSeek V3 base](https://huggingface.co/datasets/arcee-ai/DeepSeek-DCLM-Logits-Packed-8192): ~1.2 billion tokens of raw completion data from DCLM captured from the DeepSeek V3 base model

## Cross-Architecture Distillation

DistillKit can be used together with [mergekit-tokensurgeon](https://github.com/arcee-ai/mergekit/blob/main/docs/tokensurgeon.md) for cross-tokenizer, cross-architecture distillation. Many Arcee models combine both tools:

1. Use tokensurgeon to adapt student embeddings to teacher's tokenizer
2. Use DistillKit to distill teacher knowledge to student
3. Optionally convert back to student's original tokenizer, maybe do some other weird merges, follow your dreams

## Training Tips

- **Start with ~0.5 cross-entropy weight**, then tune up or down depending on how high quality your dataset is
- **Distillation temperature**: `temperature: 2.0` is a good first choice
- **Missing probability handling**: Use `zero` to focus only on the teacher's most confident predictions; use `uniform` to match the teacher's uncertainty as well

## Citation

If you use DistillKit in your research, please cite:

```bibtex
@software{distillkit2024,
  title = {DistillKit: Flexible Knowledge Distillation for Large Language Models},
  author = {Goddard, Charles and Atkins, Lucas},
  year = {2024},
  publisher = {Arcee AI},
  url = {https://github.com/arcee-ai/distillkit}
}
```

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/arcee-ai/distillkit/issues)
- **Discussions**: [Arcee Discord](https://discord.gg/arceeai)

Note: DistillKit is an open-source research release. While it powers several of our production models and we'll happily address issues as bandwidth allows, community support is best-effort.

## License

DistillKit is released under the Apache License 2.0.

### Acknowledgments

- Flash Attention packing implementation adapted from [Functionary](https://github.com/MeetKai/functionary) (MIT License)
- Built on [HuggingFace Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), and [Accelerate](https://github.com/huggingface/accelerate)

---

**Built with ♥ by [Arcee AI](https://www.arcee.ai)**
