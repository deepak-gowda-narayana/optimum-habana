# %%
# !%load_ext autoreload
# !%autoreload 2
import torch
from datasets import Dataset, load_dataset
from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiDPOTrainer
from optimum.habana.trl.trainer.dpo_config import GaudiDPOConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    DPODataCollatorWithPadding,
    apply_chat_template,
    extract_prompt,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)
# model.config.use_cache = False
# model.config.use_fused_rope = False
model.generation_config.attn_softmax_bf16 = False
model.generation_config.use_flash_attention = False
model.generation_config.flash_attention_recompute = False
model.generation_config.flash_attention_causal_mask = False
print(model)

# %%
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer.bos_token = tokenizer.eos_token
tokenizer.bos_token_id = tokenizer.eos_token_id
print(tokenizer)


# %%
def prep_dataset(dataset: Dataset):
    """Data preparation for the dataset."""
    return dataset.map(extract_prompt).map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=8,
    )


train_dataset = prep_dataset(
    load_dataset("trl-lib/ultrafeedback_binarized", split="train")
)
print(train_dataset)

eval_dataset = prep_dataset(
    load_dataset("trl-lib/ultrafeedback_binarized", split="test")
)
print(eval_dataset)

# %%
steps = 100
gaudi_config = GaudiConfig(use_fused_adam=True, use_fused_clip_norm=True)
dpo_config = GaudiDPOConfig(
    seed=123,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    logging_steps=50,
    max_steps=steps,
    save_steps=steps,
    eval_steps=steps,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    learning_rate=5e-4,
    evaluation_strategy="steps",
    output_dir="output/dpo",
    report_to="none",
    lr_scheduler_type="cosine",
    warmup_steps=100,
    optim="paged_adamw_32bit",
    bf16=True,
    remove_unused_columns=False,
    run_name="dpo_test",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    deepspeed=False,
    overwrite_output_dir=True,
    use_habana=True,
    use_lazy_mode=True,
    use_hpu_graphs_for_training=False,
    use_hpu_graphs_for_inference=True,
)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
label_pad_token_id = -100
trainer = GaudiDPOTrainer(
    model=model,
    args=dpo_config,
    gaudi_config=gaudi_config,
    peft_config=peft_config,
    beta=0.1,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_prompt_length=512,
    max_length=1024,
    label_pad_token_id=label_pad_token_id,
    # data_collator=None,
    data_collator=DPODataCollatorWithPadding(
        max_length=1024,
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=label_pad_token_id,
        is_encoder_decoder=model.config.is_encoder_decoder,
    ),
)
trainer.use_dpo_data_collator = True
train_result = trainer.train()
print(train_result)

# %%
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
