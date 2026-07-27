#!/usr/bin/env python3
"""Fine-tune local CPU — Falcon3-10B-Instruct + tool-calling PT-BR."""
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL = "tiiuae/Falcon3-10B-Instruct"
DATASET = "data/ptbr_tools_train_large.jsonl"
OUTPUT = "adapters/f10b-ptbr-tools-local"
MAX_SEQ_LEN = 48  # Curto para economizar RAM (10B é grande)
LORA_R = 8
LORA_ALPHA = 16
STEPS = 30  # Menos steps para caber em RAM/tempo
BATCH_SIZE = 1
GRAD_ACCUM = 2

print(f"{'='*60}")
print(f"BitNet Studio — Fine-tune LOCAL 100% CPU")
print(f"Modelo: {MODEL} (Falcon 10B)")
print(f"Max seq len: {MAX_SEQ_LEN}")
print(f"Steps: {STEPS}")
print(f"{'='*60}\n")

torch.set_num_threads(4)
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"CUDA: {torch.cuda.is_available()}")

print("\n[1/5] Carregando tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print("Tokenizer OK")

print("\n[2/5] Carregando dataset...")
rows = []
with open(DATASET, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

def to_text(messages):
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(f"<|system|>\n{m['content']}")
        elif m["role"] == "user":
            parts.append(f"<|user|>\n{m['content']}")
        else:
            parts.append(f"<|assistant|>\n{m['content']}")
    return "\n".join(parts) + "\n<|assistant|>\n"

texts = [to_text(r["messages"]) for r in rows]
print(f"Dataset: {len(texts)} exemplos")

ds = Dataset.from_dict({"text": texts}).map(
    lambda b: tok(b["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False),
    batched=True, remove_columns=["text"],
)

print(f"\n[3/5] Carregando modelo {MODEL} em CPU...")
print("⚠️  Falcon 10B em CPU: ~20GB RAM, ~2-3 min/step")
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print(f"Modelo carregado em {time.time()-t0:.1f}s")

print("\n[4/5] Configurando LoRA...")
lora = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

print(f"\n[5/5] Iniciando treino ({STEPS} steps)...")
print(f"Estimativa: ~{STEPS*2} minutos (2-3 min/step)\n")

args = TrainingArguments(
    output_dir=OUTPUT + "/checkpoints",
    max_steps=STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=5e-4,
    warmup_steps=3,
    logging_steps=1,
    save_strategy="steps",
    save_steps=15,
    optim="adamw_torch",
    fp16=False,
    bf16=False,
    seed=42,
    report_to=[],
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)

t_start = time.time()
trainer.train()
t_total = time.time() - t_start

print(f"\n{'='*60}")
print(f"TREINO COMPLETO!")
print(f"Tempo total: {t_total/60:.1f} minutos")
print(f"Média por step: {t_total/STEPS:.1f} segundos")
print(f"{'='*60}")

print(f"\nSalvando adapter em: {OUTPUT}")
Path(OUTPUT).mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT)
tok.save_pretrained(OUTPUT)

print("\n✅ FINE-TUNE FALCON 10B COMPLETO!")
print(f"Adapter: {OUTPUT}")
