import os,torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig
from trl import SFTTrainer

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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
)

# Step 5: Load Pretrained Model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                              quantization_config=bnb_config, 
                                             torch_dtype=torch.float16)
model.eval()

# Apply qLoRA
qlora_config = LoraConfig(
    r=16,  # The rank of the Low-Rank Adaptation
    lora_alpha=32,  # Scaling factor for the adapted layers
    target_modules=["q_proj", "v_proj"],  # Layer names to apply LoRA to
    lora_dropout=0.1,
    bias="none"
)

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
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    processing_class=tokenizer,
    peft_config = qlora_config,
    data_collator=data_collator,
    # max_seq_length=max_seq_length,
)

trainer.train()

# Step 8: Save the Model
trainer.model.save_pretrained("./fine_tuned_model/ft_custom_data")
trainer.processing_class.save_pretrained("./fine_tuned_model/ft_custom_data")
