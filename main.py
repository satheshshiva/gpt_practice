from transformers import AutoModelForCausalLM, AutoTokenizer
import env

model_name = "mistralai/Mistral-7B-v0.1"
cache_dir = './model_cache'

model = AutoModelForCausalLM.from_pretrained(model_name,  cache_dir=cache_dir, token=env.HF_ACCESS_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir=cache_dir, token=env.HF_ACCESS_TOKEN)


input_text = "Hello"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))