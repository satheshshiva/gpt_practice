vllm-2b:
	vllm serve ibm-granite/granite-3.1-2b-instruct  --dtype float16  --max-model-len 12000 --enable-lora --lora-modules piratify-2b=/home/sathesh/Documents/Codebase/AI/gpt/fine_tuned_model/ft_granite_pirateified_qlora

## Facing OOM
vllm-8b:
	vllm serve ibm-granite/granite-3.1-8b-instruct  --dtype bfloat16  --gpu-memory-utilization 0.8 --max-model-len 1000  --max-num-seqs 1