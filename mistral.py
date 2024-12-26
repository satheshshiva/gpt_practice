from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import env, torch

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = './model_cache'
device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name,  
                                             cache_dir=cache_dir, 
                                             token=env.HF_ACCESS_TOKEN,
                                             quantization_config=bnb_config, 
                                             torch_dtype=torch.bfloat16
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir=cache_dir, token=env.HF_ACCESS_TOKEN)
model.eval()

chat = [
    {"role": "system", "content": "You are a helpful AI assistant"},
    {"role": "user", "content": "Write a python code to generate fibonacci series"},
]

# chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
# a=chatbot(messages)
# print(a)


chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, 
                        max_new_tokens=500)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output)
