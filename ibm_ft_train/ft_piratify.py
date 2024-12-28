from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GraniteForCausalLM, GraniteConfig, AutoConfig, pipeline
import torch
from datasets import load_dataset
import datasets

## LOADING MODEL
model_name = "ibm-granite/granite-3.1-2b-instruct"
model_cache_dir = './model_cache'
dataset_cache_dir = './dataset_cache'
device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name,  
                                             cache_dir=model_cache_dir, 
                                             device_map=device, 
                                             quantization_config=bnb_config, 
                                             torch_dtype=torch.bfloat16
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir=model_cache_dir)
model.eval()

## DATA SET PREPARATION
dataset = load_dataset('alespalla/chatbot_instruction_prompts',
                       cache_dir=dataset_cache_dir)

def pirateify(batch):
  prompts = [f"make it sound like a pirate said this, do not include any preamble or explanation only piratify the following: {response}" for response in batch['response']]
  # Tokenize the inputs in batch and move them to GPU
  inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
  # Generate the pirate-like responses in batch
  outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.7)
  # Decode the generated tokens into text for each output in the batch
  pirate_responses = []
  for output in outputs:
    pr = tokenizer.decode(output, skip_special_tokens=True)
    if '\n\n' in pr:
      pirate_responses.append(pr.split('\n\n')[-1])
    else:
      pirate_responses.append(pr)

  # Move the outputs back to CPU (to free up GPU memory)
  inputs = inputs.to('cpu')
  outputs = outputs.to('cpu')
  # Clear the GPU cache to release any unused memory
  torch.cuda.empty_cache()
  return {
      'prompt': batch['prompt'],  # The original prompts (already a batch)
      'response': pirate_responses  # The pirate responses, generated in batch
  }


def filter_long_examples(example):
    prompt_tokens = tokenizer.tokenize(example['prompt'])
    response_tokens = tokenizer.tokenize(example['response'])  # Tokenize the response
    return len(response_tokens) <= 200 and len(prompt_tokens) <= 50

# Apply the filter to both train and test splits
train_filtered = dataset['train'].select(range(6000)).filter(filter_long_examples)
test_filtered = dataset['test'].select(range(500)).filter(filter_long_examples)

print(f"train_filtered: {len(train_filtered)} observations\ntest_filtered: {len(test_filtered)} observations")
pirate_train = train_filtered.select(range(3)).map(pirateify, batched=True, batch_size=64)
pirate_test = test_filtered.select(range(2)).map(pirateify, batched=True, batch_size=64)

# Save the new dataset
pirate_dataset = datasets.DatasetDict({
    'train': pirate_train,
    'test': pirate_test
})

pirate_dataset['train'].to_pandas().head()