from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GraniteForCausalLM, pipeline
import torch

model_name = "ibm-granite/granite-3.1-8b-instruct"
cache_dir = './model_cache'
device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = GraniteForCausalLM.from_pretrained(model_name,  
                                             cache_dir=cache_dir, 
                                             device_map=device, 
                                             quantization_config=bnb_config, 
                                             torch_dtype=torch.bfloat16
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir=cache_dir)

# change input text as desired
chat = [
    { "role": "user", "content": "write a python code to generate fibonacci series" },
]
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