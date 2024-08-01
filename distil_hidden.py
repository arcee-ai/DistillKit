import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
import yaml

# Configuration
config = {
    "project_name": "distil-multilayer",
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
        "logging_steps": 2,
        "save_total_limit": 2,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.2,
        "lr_scheduler_type": "linear",
        "resume_from_checkpoint": None,
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 1.0,
        "group_by_length": False
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": True
    }
}

# Set up environment
os.environ['WANDB_PROJECT'] = config["project_name"]
accelerator = Accelerator()
device = accelerator.device

# Load and preprocess dataset
dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
dataset = dataset.shuffle(seed=config["dataset"]["seed"])

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Apply chat template to student tokenizer
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

def sharegpt_format(example):
    system = "You are a helpful assistant chatbot."
    conversations = example['conversations']
    
    message = [{"role": "system", "content": system}]
    
    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                if conversation.get('from') == 'human':
                    message.append({"role": "user", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'gpt':
                    message.append({"role": "assistant", "content": conversation.get('value', '')})
    
    text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return {"text": text}

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16 if config["training"]["bf16"] else (torch.float16 if config["training"]["fp16"] else torch.float32)}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

# Multi-layer Adaptation layer
class MultiLayerAdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers):
        return {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states

adaptation_layer = MultiLayerAdaptationLayer(
    student_model.config.hidden_size,
    teacher_model.config.hidden_size,
    student_model.config.num_hidden_layers,
    teacher_model.config.num_hidden_layers,
    dtype=torch.bfloat16
).to(device)

class HiddenTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        self.teacher_model = self.teacher_model.to(model.device)
        
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs, output_hidden_states=True)
        loss = student_outputs.loss
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, output_hidden_states=True)
        
        custom_loss = self.distillation_loss(student_outputs, teacher_outputs, inputs, loss)
        
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_outputs, teacher_outputs, inputs, original_loss):
        student_hidden_states = student_outputs.hidden_states
        teacher_hidden_states = teacher_outputs.hidden_states

        self.adaptation_layer = self.adaptation_layer.to(student_hidden_states[0].device)
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        total_loss_kd = 0
        for student_hidden, teacher_idx in self.adaptation_layer.layer_mapping.items():
            teacher_hidden = teacher_hidden_states[teacher_idx]
            
            if adapted_student_hidden_states[student_hidden].shape != teacher_hidden.shape:
                raise ValueError(f"Shape mismatch: student {adapted_student_hidden_states[student_hidden].shape} vs teacher {teacher_hidden.shape}")

            student_probs = F.softmax(adapted_student_hidden_states[student_hidden] / config["distillation"]["temperature"], dim=-1)
            teacher_probs = F.softmax(teacher_hidden / config["distillation"]["temperature"], dim=-1)

            loss_kd = F.kl_div(
                F.log_softmax(adapted_student_hidden_states[student_hidden] / config["distillation"]["temperature"], dim=-1),
                teacher_probs,
                reduction='batchmean'
            ) * (config["distillation"]["temperature"] ** 2)

            total_loss_kd += loss_kd

        avg_loss_kd = total_loss_kd / len(self.adaptation_layer.layer_mapping)
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / hidden_dim

        total_loss = config["distillation"]["alpha"] * scaled_loss_kd + (1 - config["distillation"]["alpha"]) * original_loss
        return total_loss

# Training arguments
training_arguments = TrainingArguments(**config["training"])

# Create the custom SFT Trainer
trainer = HiddenTrainer(
    model=student_model,
    train_dataset=dataset,
    max_seq_length=config["tokenizer"]["max_length"],
    tokenizer=student_tokenizer,
    dataset_text_field="text",
    args=training_arguments,
    packing=True,
)

# Add these attributes to the trainer
trainer.teacher_model = teacher_model
trainer.adaptation_layer = adaptation_layer

# Prepare for distributed training
trainer = accelerator.prepare(trainer)

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])

# Save the adaptation layer
torch.save(adaptation_layer.state_dict(), "adaptation_layer.pth")
