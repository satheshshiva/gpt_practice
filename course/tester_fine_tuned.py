
from transformers import  AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Same as before
checkpoint = "checkpoint-500"
model_cache_dir = './test-trainer'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir )
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, cache_dir=model_cache_dir,  num_labels=2)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
    "I hate this course",
]

pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
print(pipeline(sequences))