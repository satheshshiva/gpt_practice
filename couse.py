import torch
from transformers import  AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
cache_dir = './model_cache'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir )
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, cache_dir=cache_dir )
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = torch.optim.AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()