import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Step 1: Define Model and Dataset Paths
MODEL_NAME = "ibm-granite/granite-3.1-2b-instruct"  # Replace with your pre-trained model (e.g., GPT-2, LLaMA).
DATASET_PATH = "FT_hiped/text2.txt"  # Directory with text files.

# Step 2: Load Dataset
def load_text_data(path):
    """Load text data from a directory or file."""
    if os.path.isdir(path):
        return load_dataset("text", data_files=[os.path.join(path, f) for f in os.listdir(path)])
    else:
        return load_dataset("text", data_files=path)

dataset = load_text_data(DATASET_PATH)

# Step 3: Preprocess Data
def tokenize_function(examples):
    """Tokenize input text for the model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is aligned for causal models.
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(tokenized_dataset)
# Step 4: Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to False for causal language models like GPT.
)

# Step 5: Load Pretrained Model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# Step 6: Define Training Arguments
training_args = TrainingArguments(
   output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=1,  # Increase this to use more GPU memory
    per_device_eval_batch_size=1, # Increase this to use more GPU memory
    # gradient_accumulation_steps=16,  # Increase this to compensate
    num_train_epochs=1,
    logging_steps=100,
    fp16=True,
    report_to="none",
    gradient_checkpointing=True,    # using this saves memory but slows down training
)

# Step 7: Train the Model
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
    data_collator=data_collator,
)

trainer.train()

# Step 8: Save the Model
model.save_pretrained("./fine_tuned_model/fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model/fine_tuned_model")
