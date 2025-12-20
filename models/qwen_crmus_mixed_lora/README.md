---
library_name: peft
license: other
base_model: Qwen/Qwen1.5-7B-Chat
tags:
- base_model:adapter:Qwen/Qwen1.5-7B-Chat
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen_crmus_mixed_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen_crmus_mixed_lora

This model is a fine-tuned version of [Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) on the crmus_cr and the crmus_mu datasets.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.5.1+cu121
- Datasets 4.0.0
- Tokenizers 0.22.1