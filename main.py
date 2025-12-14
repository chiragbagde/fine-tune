#!/usr/bin/env python3
"""
Mistral-7B-Instruct-v0.3 QLoRA + LoRA fine-tuning on A40 (single GPU).

Outputs:
- training.log           -> full stdout logs
- adapter_model/         -> LoRA adapter weights
- tokenizer/             -> tokenizer files
- train_metrics.json     -> Trainer metrics
- inference.txt          -> inference sample output

Runtime on A40:
- ~30–35 minutes (500 steps)
"""

import os
import sys
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# ---------------------------
# Config
# ---------------------------
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_ID = "iamtarun/python_code_instructions_18k_alpaca"

OUTPUT_DIR = "./output"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapter_model")
TOKENIZER_DIR = os.path.join(OUTPUT_DIR, "tokenizer")

LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")
INFER_FILE = os.path.join(OUTPUT_DIR, "inference.txt")

SEQ_LEN = 512
MAX_STEPS = 500
BATCH_SIZE = 2          # A40 can handle this
GRAD_ACCUM = 4          # effective batch = 8
LR = 2e-4
SEED = 42

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ---------------------------
# Logging setup
# ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_f = open(LOG_FILE, "w")
sys.stdout = log_f
sys.stderr = log_f

def log(msg):
    print(msg, flush=True)

# ---------------------------
# Helpers
# ---------------------------
def print_env():
    log(f"Timestamp: {datetime.now()}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"PyTorch: {torch.__version__}")

def format_prompt(ex):
    inst = ex.get("instruction", "").strip()
    inp = (ex.get("input") or "").strip()
    out = ex.get("output", "").strip()

    if inp:
        text = f"""### Instruction:
{inst}

### Input:
{inp}

### Response:
{out}"""
    else:
        text = f"""### Instruction:
{inst}

### Response:
{out}"""
    return {"text": text}

def tokenize_fn(tokenizer, batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
    )

# ---------------------------
# Main
# ---------------------------
def main():
    set_seed(SEED)
    print_env()

    log("\n--- CONFIG ---")
    log(f"MODEL_ID: {MODEL_ID}")
    log(f"DATASET_ID: {DATASET_ID}")
    log(f"SEQ_LEN: {SEQ_LEN}")
    log(f"MAX_STEPS: {MAX_STEPS}")
    log(f"BATCH: {BATCH_SIZE}, GRAD_ACCUM: {GRAD_ACCUM}")
    log(f"LR: {LR}")

    # Tokenizer
    log("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    log("\nLoading dataset...")
    dataset = load_dataset(DATASET_ID)
    dataset = dataset.map(format_prompt)

    log("Tokenizing...")
    tokenized = dataset.map(
        lambda b: tokenize_fn(tokenizer, b),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Model (QLoRA)
    log("\nLoading model in 4-bit (QLoRA)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # LoRA
    log("Applying LoRA...")
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Training
    log("\nStarting training...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        fp16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
    )

    train_result = trainer.train()
    log("Training finished")

    # Save artifacts
    log("\nSaving LoRA adapter and tokenizer...")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # Inference test
    log("\nRunning inference test...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    infer_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    infer_model.eval()

    prompt = """### Instruction:
Write a Python function to check if a number is prime.

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = infer_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True,
        )

    result = tokenizer.decode(out[0], skip_special_tokens=True)
    with open(INFER_FILE, "w") as f:
        f.write(result)

    log("\nInference output written to inference.txt")
    log("\nDONE ✅")

if __name__ == "__main__":
    main()
