
from transformers import  AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Same as before
checkpoint = "./fine_tuned_model/ft_bert-base-uncased"
# model_cache_dir = ''
tokenizer = AutoTokenizer.from_pretrained(checkpoint )
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
    "I hate this course",
]

pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
print(pipeline(sequences))