import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# Configuration
config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "mlabonne/FineTome-100k",
        "split": "train",
        "seed": 42
    },
    "models": {
        "teacher": "arcee-ai/Arcee-Spark"
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ 'system\nYou are a helpful assistant.\n' }}{% endif %}{{'' + message['role'] + '\n' + message['content'] + '' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'assistant\n' }}{% endif %}"
    },
    "output_dataset_path": "./dataset_with_logits"
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

# Apply chat template to teacher tokenizer
teacher_tokenizer.chat_template = config["tokenizer"]["chat_template"]

# make sure you have the same formaat as in the distillation prcess. 
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

    text = teacher_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return {"text": text}

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

def tokenize_function(examples):
    return teacher_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])

print("Dataset preparation complete. Loading models...")

# Load teacher model
teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"])
teacher_model = teacher_model.to(device)
teacher_model.eval()

def compute_logits(batch):
    inputs = teacher_tokenizer(batch["text"], return_tensors="pt", truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length").to(device)
    with torch.no_grad():
        outputs = teacher_model(**inputs)
    batch["logits"] = outputs.logits.cpu().numpy().tolist()
    return batch

# Compute logits
print("Computing teacher logits...")
dataset_with_logits = tokenized_dataset.map(compute_logits, batched=True, batch_size=8)

# Save the dataset with logits
dataset_with_logits.save_to_disk(config["output_dataset_path"])
print(f"Dataset with teacher logits saved to {config['output_dataset_path']}")
