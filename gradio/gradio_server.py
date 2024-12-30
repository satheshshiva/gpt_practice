from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import torch
import re

model_name = "ibm-granite/granite-3.1-8b-instruct"
cache_dir = "./model_cache"
device = "cuda"  # "cuda" or "cpu"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def get_last_assistant_response(text):
    # Regular expression to find all assistant responses
    assistant_pattern = r"<\|start_of_role\|>\s*assistant\s*<\|end_of_role\|>\s*(.*?)"
    # Find all matches in the text
    matches = re.findall(assistant_pattern, text, re.DOTALL)

    # Return the last match if any, otherwise return None
    return matches[-1].strip().replace("<\|end_of_text\|>", "") if matches else "None"

model = AutoModelForCausalLM.from_pretrained(model_name,  
                                             cache_dir=cache_dir, 
                                             device_map=device, 
                                             quantization_config=bnb_config, 
                                             torch_dtype=torch.bfloat16
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir=cache_dir)
model.eval()

def predict(message, history):
    history.append({"role": "user", "content": message})
    input_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)  
    outputs = model.generate(inputs, max_new_tokens=500)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # if '\n\n' in decoded:
    #     return decoded.split('\n\n')[-1]
    # else:
    return get_last_assistant_response(decoded)

demo = gr.ChatInterface(predict, type="messages")

demo.launch(server_name="0.0.0.0")
