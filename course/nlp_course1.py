import torch
from transformers import  AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset

# Same as before
checkpoint = "bert-base-uncased"
model_cache_dir = './model_cache'
dataset_cache_dir = './dataset_cache'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir )
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, cache_dir=model_cache_dir )
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
raw_datasets = load_dataset("glue", "mrpc", cache_dir=dataset_cache_dir)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()