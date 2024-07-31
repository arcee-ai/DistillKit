import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
import yaml

# Configuration
config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "mlabonne/FineTome-100k",
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

# Set up environment
os.environ['WANDB_PROJECT'] = config["project_name"]
accelerator = Accelerator()
device = accelerator.device

# Load and preprocess dataset
dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
dataset = dataset.shuffle(seed=config["dataset"]["seed"])
if "num_samples" in config["dataset"]:
    dataset = dataset.select(range(config["dataset"]["num_samples"]))

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Apply chat template to student tokenizer
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

def sharegpt_format(example):
    conversations = example['conversations']
    message = []
    
    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                if conversation.get('from') == 'human':
                    message.append({"role": "user", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'gpt':
                    message.append({"role": "assistant", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'system':
                    message.insert(0, {"role": "system", "content": conversation.get('value', '')})

    if not any(msg.get('role') == 'system' for msg in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return {"text": text}

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

def tokenize_function(examples):
    return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

# Optionally freeze layers of the student model based on spectrum configuration
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
    def freeze_student_spectrum(model, unfrozen_layers_file):
        with open(unfrozen_layers_file, 'r') as file:
            unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
        
        for name, param in model.named_parameters():
            if not any(layer in name for layer in unfrozen_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Apply freezing to student model
    freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
else:
    print("Spectrum configuration not found. All layers of the student model will be trainable.")

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits

class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(model.device)
        
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        student_logits, teacher_logits = pad_logits(student_logits.to(self.model.device), teacher_logits.to(self.model.device))
        
        student_logits_scaled = student_logits / config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (config["distillation"]["temperature"] ** 2) / config["tokenizer"]["max_length"]

        return config["distillation"]["alpha"] * loss_kd + (1 - config["distillation"]["alpha"]) * original_loss

# Training arguments
training_arguments = TrainingArguments(**config["training"])

# Create the custom SFT Trainer
trainer = CustomSFTTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=student_tokenizer,
    args=training_arguments,
    max_seq_length=config["tokenizer"]["max_length"],
)

# Add the teacher model to the trainer
trainer.teacher_model = teacher_model

# Prepare for distributed training
trainer = accelerator.prepare(trainer)

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])
