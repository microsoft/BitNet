"""Fine-tune GPU para Falcon3-10B-Instruct + tool-calling PT-BR.

Recomendado para: RunPod, Vast.ai, Lambda Labs (RTX 3090/4090 24GB, A100 40GB).
Executa em ~15-20 minutos para 300 steps.

Uso:
  python3 finetune_falcon10b_gpu.py

Requer: transformers, peft, datasets, accelerate, bitsandbytes
"""
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL = "tiiuae/Falcon3-10B-Instruct"
DATASET = "data/ptbr_tools_train_large.jsonl"
OUTPUT = "adapters/f10b-ptbr-tools-gpu"
MAX_SEQ_LEN = 256
LORA_R = 16
LORA_ALPHA = 32
STEPS = 300

# Detectar GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

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

rows = []
with open(DATASET, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

texts = [to_text(r["messages"]) for r in rows]
print(f"Dataset: {len(texts)} exemplos")

ds = Dataset.from_dict({"text": texts}).map(
    lambda b: tok(b["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False),
    batched=True, remove_columns=["text"]
)

if device == "cuda":
    print(f"Carregando {MODEL} em GPU (QLoRA 4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
else:
    print(f"Carregando {MODEL} em CPU (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

lora = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUTPUT + "/checkpoints",
    max_steps=STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=30,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
    fp16=False,
    bf16=(device == "cuda"),
    seed=42,
    report_to=[],
    gradient_checkpointing=(device == "cuda"),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)

print(f"Iniciando treino ({STEPS} steps)...")
if device == "cuda":
    print(f"Estimativa: ~{STEPS // 60} minutos em GPU")
else:
    print(f"⚠️  Estimativa: ~{STEPS * 50 // 60} horas em CPU")

trainer.train()

model.save_pretrained(OUTPUT)
tok.save_pretrained(OUTPUT)
print(f"Adapter salvo em: {OUTPUT}")

if device == "cuda":
    print("\nPara fazer merge e converter para GGUF:")
    print(f"  python3 merge_and_quantize.py --adapter {OUTPUT}")
