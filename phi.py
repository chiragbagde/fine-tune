#!/usr/bin/env python3
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "microsoft/phi-2"
DATASET_ID = "iamtarun/python_code_instructions_18k_alpaca"
OUTPUT_DIR = "./output_phi2"

SEQ_LEN = 512
MAX_STEPS = 1000
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 2e-4
SEED = 42

LORA_R = 16
LORA_ALPHA = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(SEED)

def format_alpaca(example):
    inst = example.get("instruction", "").strip()
    inp = (example.get("input") or "").strip()
    out = example.get("output", "").strip()
    if inp:
        text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
    return {"text": text}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading dataset and applying formatting...")
    dataset = load_dataset(DATASET_ID)
    train_dataset = dataset["train"].map(format_alpaca)

    print(f"Loading model in 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=None,
        max_seq_length=SEQ_LEN,
        dataset_text_field="text",
    )

    print(f"\nStarting training ({MAX_STEPS} steps)...")
    trainer.train()
    
    final_model_path = f"{OUTPUT_DIR}/final_model"
    print(f"\nSaving final LoRA adapter to {final_model_path}...")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    log_history = trainer.state.log_history
    with open(f"{OUTPUT_DIR}/training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)
    
    print(f"Training complete! Outputs saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
