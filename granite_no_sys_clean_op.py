from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GraniteForCausalLM, GraniteConfig, AutoConfig, pipeline
import torch

model_name = "ibm-granite/granite-3.1-8b-instruct"
device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name,  
                                             device_map=device, 
                                             quantization_config=bnb_config, 
                                             torch_dtype=torch.bfloat16
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# change input text as desired
chat = [
    # { "role": "system", "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 25, 2024.\nYou are SecurityGPT, developed by Security Engineering Team at American Express. You are a helpful AI assistant at American Express to help American Express employees with queries related to security tools developed within American Express." },
    { "role": "user", "content": "who are you?" },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

prompt = "make it sound like a pirate said this, do not include any preamble or explanation only piratify the following: I want to employ new people in my company."
# tokenize the text
input_tokens = tokenizer(prompt, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, 
                        max_new_tokens=500, do_sample=True, top_p=0.95, temperature=0.7)
# decode output tokens into text

pr = tokenizer.decode(output[0], skip_special_tokens=True)
if '\n\n' in pr:
    print(pr.split('\n\n')[-1])
else:
   print(pr)