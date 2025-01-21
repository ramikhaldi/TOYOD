import os
from datasets import Dataset
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = "<YOUR_DIRECTORY>"

# Function to load text files recursively
def load_text_files_recursively(directory):
    file_texts = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                print("Found file:", filepath)  # Print each file being read

                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()

                file_texts.append({"text": text_content})
    return file_texts

# Load raw data
raw_data = load_text_files_recursively(DATA_DIR)
dataset = Dataset.from_list(raw_data)
print(f"Loaded {len(dataset)} documents.")

train_data = dataset  # Use the entire dataset for training
print(f"Train size: {len(train_data)} | Validation size: None")

# Load tokenizer and model
model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

# Reuse eos_token as pad_token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load the model with device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",  # Automatically splits the model across available devices
    offload_folder="./offload",  # Offload layers to disk if memory is insufficient
    offload_state_dict=True,    # Reduce RAM usage further by offloading state_dict
)
# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

# Tokenize training data
train_data = train_data.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for causal language modeling
def data_collator(features):
    input_ids = [f["input_ids"] for f in features]
    labels = [f["input_ids"] for f in features]
    batch = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
    batch["labels"] = batch["input_ids"].clone()  # Align labels with input_ids
    return batch

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Adjust based on available memory
    gradient_accumulation_steps=4,  # Effective batch size = 8 * 4 = 32
    logging_strategy="steps",
    logging_steps=50,
    save_steps=2000,  # Save less frequently
    learning_rate=2e-4,
    lr_scheduler_type="cosine",  # Faster convergence
    fp16=True,
    fp16_full_eval=True,
    optim="adamw_torch",
    dataloader_pin_memory=True,
    report_to="tensorboard",
    logging_dir="./runs",
)

# Trainer setup (no validation set)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator,
)

# Train and save the model
trainer.train()
trainer.save_model("./lora-llama-final")  # Save model and tokenizer together