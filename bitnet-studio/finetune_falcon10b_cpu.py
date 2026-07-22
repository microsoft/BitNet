"""Fine-tune CPU para Falcon3-10B-Instruct + tool-calling PT-BR.

⚠️ AVISO: Falcon 10B fp16 consome ~20GB RAM.
Cada step em CPU leva ~40-60 minutos.
200 steps = ~5-8 dias de treino contínuo.

Recomendação: usar Google Colab Pro (A100 40GB) ou RunPod/Vast.ai (RTX 3090).
"""
import json
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
DATASET = "data/ptbr_tools_train_large.jsonl"  # 162 exemplos
OUTPUT = "adapters/f10b-ptbr-tools-cpu"
MAX_SEQ_LEN = 256
LORA_R = 8
LORA_ALPHA = 16
STEPS = 50

print(f"⚠️  Falcon 10B em CPU — cada step ~40-60 minutos")
print(f"Torch device: CPU (CUDA: {torch.cuda.is_available()})")

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

print(f"Carregando {MODEL} em CPU (fp16, ~20GB RAM)...")
print("Isso pode demorar 5-10 minutos...")
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
    warmup_steps=5,
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,
    optim="adamw_torch",
    fp16=False,
    seed=42,
    report_to=[],
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)

print("Iniciando treino Falcon 10B (isso vai demorar MUITO em CPU)...")
print(f"Estimativa: {STEPS} steps × ~50 min = ~{STEPS * 50 // 60} horas")
trainer.train()

model.save_pretrained(OUTPUT)
tok.save_pretrained(OUTPUT)
print(f"Adapter salvo em: {OUTPUT}")
print("Para converter para GGUF, use: python3 merge_and_quantize.py")
