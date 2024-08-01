# DistillKit

## Overview

DistillKit is an open-source research effort in model distillation by Arcee.AI. Our goal is to provide the community with easy-to-use tools for researching, exploring, and enhancing the adoption of open-source Large Language Model (LLM) distillation methods. This release focuses on practical, effective techniques for improving model performance and efficiency.

## Features

- Logit-based Distillation (models must be the same architecture)
- Hidden States-based Distillation (models can be different architectures)
- Support for Supervised Fine-Tuning (SFT) - DPO and CPT to come at a later date.


## Installation

### Quick Install

For a quick and easy installation, you can use our setup script:

```bash
./setup.sh
```

### Manual Installation

If you prefer to install dependencies manually, follow these steps:

1. Install basic requirements:
   ```bash
   pip install torch wheel ninja packaging
   ```

2. Install Flash Attention:
   ```bash
   pip install flash-attn
   ```

3. Install DeepSpeed:
   ```bash
   pip install deepspeed
   ```

4. Install remaining requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

For simplicity, we've set the config settings directly within the training script. You can customize the configuration as follows:

```python
config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "mlabonne/FineTome-100k", # Only sharegpt format is currently supported.
        "split": "train",
        # "num_samples": , # You can pass a number here to limit the number of samples to use.
        "seed": 42
    },
    "models": {
        "teacher": "arcee-ai/Arcee-Spark",
        "student": "Qwen/Qwen2-1.5B"
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": True
    }
    # "spectrum": {
    #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Qwen-Qwen2-1.5B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
    # }
}
```

### Chat Template

If you want to use a chat template other than chatml, copy it from the model's `tokenizer_config.json`, and replace the current `chat_template` entry in the configuration.

### Spectrum Integration

You can use Spectrum to increase speed (but not memory overhead). To enable Spectrum, uncomment the "spectrum" section in the configuration and provide the path to your Spectrum YAML file. Please note that further evaluations with Spectrum are TBD.

## Usage

To launch DistillKit, use the following command:

```bash
accelerate launch distil_logits.py
```

You can replace `distil_logits.py` with whichever script you want to use.

### Advanced Configurations

If you wish to use DeepSpeed, Fully Sharded Data Parallel (FSDP), or Megatron sharding, you can set up your configuration using:

```bash
accelerate config
```

Follow the prompts to configure your desired setup.

### DeepSpeed Configurations

We provide sample DeepSpeed configuration files in the `./deepspeed_configs` directory. These configurations are shamelessly stolen from the Axolotl (thanks to Wing Lian and the Axolotl team for their excellent work!).

To use a specific DeepSpeed configuration, you can specify it when launching your script:

```bash
accelerate launch --config_file path/to/deepspeed_config.yaml distil_logits.py
```

## Distillation Methods

DistillKit supports two primary distillation methods:

1. **Logit-based Distillation**: This method transfers knowledge from a larger teacher model to a smaller student model by using both hard targets (actual labels) and soft targets (teacher logits). The soft target loss, computed using Kullback-Leibler (KL) divergence, encourages the student to mimic the teacher's output distribution. This method enhances the student model's generalization and efficiency while maintaining performance close to the teacher model.

2. **Hidden States-based Distillation**: This method involves transferring knowledge by aligning the intermediate layer representations of the student model with those of the teacher model. This process enhances the student's learning by providing richer, layer-wise guidance, improving its performance and generalization. This method allows for cross-architecture distillation, providing flexibility in model architecture choices.

## Performance and Memory Requirements

While the implementation of DistillKit is relatively straightforward, the memory requirements for distillation are higher compared to standard SFT. We are actively working on scaling DistillKit to support models larger than 70B parameters, which will involve advanced techniques and efficiency improvements.

## Experimental Results

Our experiments have shown promising results in both general-purpose and domain-specific tasks. Key findings include:

- Both logit-based and hidden states-based distillation methods show improvements over standard SFT across most benchmarks.
- Significant performance gains were observed when distilling models for domain-specific tasks.
- Using the same training dataset for distillation as was used for the teacher model can lead to higher performance gains.

For detailed results and analysis, please refer to our case studies and experimental here.

## Arcee-Labs

This release marks the debut of Arcee-Labs, a division of Arcee.ai dedicated to accelerating open-source research. Our mission is to rapidly deploy resources, models, and research findings to empower both Arcee and the wider community. In an era of increasingly frequent breakthroughs in LLM research, models, and techniques, we recognize the need for agility and adaptability. Through our efforts, we strive to significantly contribute to the advancement of open-source AI technology and support the community in keeping pace with these rapid developments.

## Future Directions

We are excited to see how the community will use and improve DistillKit. Future releases will include Continued Pre-Training (CPT) and Direct Preference Optimization (DPO) distillation methods. We welcome community contributions in the form of new distillation methods, training routine improvements, and memory optimizations.

## Contributing

We welcome contributions from the community! If you have ideas for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.

## Contact

For more information about Arcee.AI and our training platform, visit our website at [https://arcee.ai](https://arcee.ai).

For technical questions or support, please open an issue in this repository.
## Acknowledgments

While our work is ultimately quite different - this project was inspired by [Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs](https://arxiv.org/abs/2402.12030). We thank the authors for their efforts and contributions. We would like to thank the open-source community and all at arcee.ai who have helped make DistillKit possible. We're just getting started.
