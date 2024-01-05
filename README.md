## Quick Start
__LoRA__
```bash
python train.py --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--output_dir "./tinyllama-sft-lora" \
--per_device_train_batch_size 4 \
--use_lora 
```
__QLoRA(4bit)__
```bash
python train.py --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--output_dir "./tinyllama-sft-qlora" \
--per_device_train_batch_size 4 \
--use_lora \
--use quant
```