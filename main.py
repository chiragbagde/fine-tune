#!/usr/bin/env python3
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_ID = "iamtarun/python_code_instructions_18k_alpaca"
OUTPUT_DIR = "./output_local"

SEQ_LEN = 256
MAX_STEPS = 10
BATCH_SIZE = 1
GRAD_ACCUM = 2
LR = 2e-4
SEED = 42

LORA_R = 8
LORA_ALPHA = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(SEED)

def format_alpaca(example):
    inst = example.get("instruction", "").strip()
    inp = (example.get("input") or "").strip()
    out = example.get("output", "").strip()
    if inp:
        text = f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
    else:
        text = f"Instruction: {inst}\nOutput: {out}"
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

    print(f"Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
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
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        bf16=(device == "cuda"),
        logging_steps=1,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
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

    print("\nStarting local test training (10 steps)...")
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
