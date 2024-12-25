from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import env

model_name = "mistralai/Mistral-7B-v0.1"
cache_dir = './model_cache'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name,  cache_dir=cache_dir, load_in_4bit=True, torch_dtype=torch.float16, token=env.HF_ACCESS_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir=cache_dir, token=env.HF_ACCESS_TOKEN)

input_text = "Hello, how to learn python code"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs["input_ids"], max_length=500, num_return_sequences=1)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))