"""Falcon 3B fine-tune — versão com fix definitivo para tokenizer.

USO NO COLAB:
    !rm -rf /content/BitNet && git clone --depth 1 https://github.com/peder1981/BitNet.git /content/BitNet
    %run /content/BitNet/bitnet-studio/colab_f3b_fix.py
"""

# 1. Verificar GPU
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "\n" + "=" * 60 + "\n"
        "ERRO: GPU não detectada!\n\n"
        "O Google Colab está em modo CPU. Para corrigir:\n"
        "  1. Menu → Runtime → Change runtime type\n"
        "  2. Hardware accelerator: GPU\n"
        "  3. Save\n"
        "  4. Menu → Runtime → Restart runtime\n"
        "  5. Execute esta célula novamente\n"
        + "=" * 60
    )

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 2. Instalar dependências
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.40.0", "peft==0.11.0", "datasets==2.19.0",
    "accelerate==0.30.0", "bitsandbytes==0.43.0", "safetensors",
    "sentencepiece",
])

# FIX: Desinstalar tokenizers (biblioteca Rust que conflita com Falcon3)
print("Desinstalando tokenizers (workaround)...")
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "tokenizers"])

# Recarregar transformers sem tokenizers
import importlib
import transformers
transformers.utils.import_utils.is_tokenizers_available = lambda: False
for mod_name in list(sys.modules.keys()):
    if "transformers" in mod_name:
        del sys.modules[mod_name]
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# 3. Dataset
import json, os
from urllib.request import urlopen

DATASET_URL = "https://raw.githubusercontent.com/peder1981/BitNet/main/bitnet-studio/data/ptbr_tools_train.jsonl"
OUTPUT = "/content/f3b-ptbr-tools-qlora"

print("Baixando dataset...")
with urlopen(DATASET_URL) as resp:
    dataset_text = resp.read().decode("utf-8")

rows = []
for line in dataset_text.strip().split("\n"):
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

# 4. Tokenizer (sem tokenizers library)
MODEL = "tiiuae/Falcon3-3B-Instruct"

# Limpar cache
import shutil
hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache, ignore_errors=True)
    print("Cache limpo")

print("Carregando tokenizer (modo lento)...")
tok = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
    use_fast=False,
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Tokenizer OK!")

# 5. Modelo QLoRA
print("Carregando modelo QLoRA...")
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

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 6. Treino
ds = Dataset.from_dict({"text": texts}).map(
    lambda b: tok(b["text"], truncation=True, max_length=256, padding=False),
    batched=True, remove_columns=["text"],
)

args = TrainingArguments(
    output_dir=OUTPUT + "/checkpoints",
    max_steps=200,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    save_strategy="steps", save_steps=50,
    optim="paged_adamw_8bit",
    fp16=False, bf16=True,
    seed=42, report_to=[],
)

trainer = Trainer(
    model=model, args=args, train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)

print("\n=== INICIANDO TREINO ===")
trainer.train()

model.save_pretrained(OUTPUT)
tok.save_pretrained(OUTPUT)
print(f"\nAdapter salvo em: {OUTPUT}")

# 7. Download
from google.colab import files
shutil.make_archive("/content/f3b-ptbr-tools-qlora", "zip", OUTPUT)
files.download("/content/f3b-ptbr-tools-qlora.zip")
print("Download iniciado!")
