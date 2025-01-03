from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from datasets import load_dataset,load_from_disk
import datasets
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import LoraConfig


## LOADING MODEL
model_name = "ibm-granite/granite-3.1-2b-instruct"
device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(model_name,  
                                            #  device_map=device, 
                                             quantization_config=bnb_config, 
                                             torch_dtype=torch.bfloat16
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()


def reduce_dataset_size():
  ## Reducing the dataset size for faster training. 
  train_filtered = pirate_dataset['train'].select(range(5))
  test_filtered = pirate_dataset['test'].select(range(2))
  # Save the new dataset
  return datasets.DatasetDict({
      'train': train_filtered,
      'test': test_filtered
  })

pirate_dataset = load_from_disk('./fine_tuned_dataset/pirate_dataset')
## comment if not needed
# pirate_dataset = reduce_dataset_size()


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"<|system|>\nYou are a helpful assistant\n<|user|>\n{example['prompt'][i]}\n<|assistant|>\n{example['response'][i]}<|endoftext|>"
        output_texts.append(text)
    return output_texts

response_template = "\n<|assistant|>\n"


response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


# Apply qLoRA
qlora_config = LoraConfig(
    r=16,  # The rank of the Low-Rank Adaptation
    lora_alpha=32,  # Scaling factor for the adapted layers
    target_modules=["q_proj", "v_proj"],  # Layer names to apply LoRA to
    lora_dropout=0.1,
    bias="none"
)

# Initialize the SFTTrainer
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=1,  # Increase this to use more GPU memory
    per_device_eval_batch_size=1, # Increase this to use more GPU memory
    # gradient_accumulation_steps=16,  # Increase this to compensate
    num_train_epochs=1,
    logging_steps=100,
    fp16=True,
    report_to="none",
    gradient_checkpointing=True,    # using this saves memory but slows down training
)

max_seq_length = 250

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=pirate_dataset['train'],
    eval_dataset=pirate_dataset['test'],
    processing_class=tokenizer,
    peft_config = qlora_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    # max_seq_length=max_seq_length,
)

## TRAINING
trainer.train()

## SAVING FINE TUNED MODEL
trainer.model.save_pretrained("./fine_tuned_model/ft_granite_pirateified_2b_qlora")
tokenizer.save_pretrained("./fine_tuned_model/ft_granite_pirateified_2b_qlora")

