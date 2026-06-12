#!/usr/bin/env python3
"""One-click fine-tune Falcon3-10B no Google Colab.

COPIE TODO este arquivo para UMA célula no Colab e execute.
⚠️ Requer: GPU T4 (16GB VRAM) — otimizado para evitar OOM.
"""

# ========== CÉLULA ÚNICA — COPIE TUDO ==========

# 1. Instalar
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.40.0", "peft==0.11.0", "datasets==2.19.0",
    "accelerate==0.30.0", "bitsandbytes==0.43.0", "safetensors"])

import torch, json, shutil
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 2. Dataset (usa large com 162 exemplos)
!git clone --depth 1 https://github.com/peder1981/BitNet.git /content/BitNet

DATASET = '/content/BitNet/bitnet-studio/data/ptbr_tools_train_large.jsonl'
OUTPUT = '/content/f10b-ptbr-tools-qlora'

rows = []
with open(DATASET, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

def to_text(messages):
    parts = []
    for m in messages:
        if m['role'] == 'system':
            parts.append(f"<|system|>\n{m['content']}")
        elif m['role'] == 'user':
            parts.append(f"<|user|>\n{m['content']}")
        else:
            parts.append(f"<|assistant|>\n{m['content']}")
    return '\n'.join(parts) + '\n<|assistant|>\n'

texts = [to_text(r['messages']) for r in rows]
print(f"Dataset: {len(texts)} exemplos")

# 3. Modelo QLoRA (otimizado T4)
MODEL = 'tiiuae/Falcon3-10B-Instruct'

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    device_map='auto',
    trust_remote_code=True,
    max_memory={0: '14GiB'},
)
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias='none', task_type='CAUSAL_LM',
    target_modules=['q_proj','k_proj','v_proj','o_proj'],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 4. Treino (seq curta para economizar VRAM)
ds = Dataset.from_dict({'text': texts}).map(
    lambda b: tok(b['text'], truncation=True, max_length=128, padding=False),
    batched=True, remove_columns=['text']
)

args = TrainingArguments(
    output_dir=OUTPUT + '/checkpoints',
    max_steps=300,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=30,
    logging_steps=10,
    save_strategy='steps', save_steps=50,
    optim='paged_adamw_8bit',
    fp16=False, bf16=True,
    seed=42, report_to=[],
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model, args=args, train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)

print("\n=== INICIANDO TREINO FALCON 10B ===")
print("⚠️  Se der OOM, reinicie e use o Falcon3-3B em vez disso")
trainer.train()

model.save_pretrained(OUTPUT)
tok.save_pretrained(OUTPUT)
print(f"\nAdapter salvo em: {OUTPUT}")

# 5. Download
from google.colab import files
shutil.make_archive('/content/f10b-ptbr-tools-qlora', 'zip', OUTPUT)
files.download('/content/f10b-ptbr-tools-qlora.zip')
print("Download iniciado!")
