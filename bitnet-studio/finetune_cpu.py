"""Fine-tune CPU-only piloto — Falcon3-3B + tool-calling PT-BR."""
import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL = "tiiuae/Falcon3-3B-Instruct"
DATASET = "data/ptbr_tools_train.jsonl"
OUTPUT = "adapters/f3b-ptbr-tools-cpu"
MAX_SEQ_LEN = 128
LORA_R = 8
LORA_ALPHA = 16
STEPS = 10

print(f"Torch: CPU (CUDA: {torch.cuda.is_available()})")

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

print(f"Carregando {MODEL} em CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True,
)

lora = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUTPUT + "/checkpoints",
    max_steps=STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    warmup_steps=2,
    logging_steps=2,
    save_strategy="no",
    optim="adamw_torch",
    fp16=False,
    seed=42,
    report_to=[],
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model, args=args, train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)

print("Iniciando treino...")
trainer.train()
model.save_pretrained(OUTPUT)
tok.save_pretrained(OUTPUT)
print(f"Adapter salvo em: {OUTPUT}")
