from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
from datasets import load_from_disk
import datasets



## LOADING MODEL
model_name = "ibm-granite/granite-3.1-2b-instruct"
model_cache_dir = './model_cache'
device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name,  
                                             cache_dir=model_cache_dir, 
                                            #  device_map=device, 
                                             quantization_config=bnb_config, 
                                             torch_dtype=torch.bfloat16
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir=model_cache_dir)
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
pirate_dataset = reduce_dataset_size()


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Initialize the SFTTrainer
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_steps=100,
    fp16=True,
    report_to="none",
    eval_strategy="steps"
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=pirate_dataset["train"],
    eval_dataset=pirate_dataset["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    # compute_metrics=compute_metrics,
)

# Clear the GPU cache to release any unused memory
torch.cuda.empty_cache()

## TRAINING
trainer.train()

## SAVING FINE TUNED MODEL
model.save_pretrained("./fine_tuned_model/ft_granite_pirateified_normal")
tokenizer.save_pretrained("./fine_tuned_model/ft_granite_pirateified_normal")

