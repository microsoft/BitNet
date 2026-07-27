"""QLoRA 4-bit para GPUs modestas — fine-tuning PT-BR + tool-calling.

Defaults calibrados para GPU de 8-16GB:
  - NF4 4-bit base + bf16/fp16 compute
  - LoRA r=16, alpha=32, dropout=0.05 (q/k/v/o + gate/up/down)
  - batch 1 + gradient accumulation 16
  - gradient checkpointing + paged_adamw_8bit
  - sequências de 1024 tokens (suficiente para tool-calling)

Para Falcon3-10B em GPU de 8GB: use --model tiiuae/Falcon3-3B-Instruct
primeiro como piloto, depois rode o 10B com --max-seq 512 (mais lento, cabe).

Uso (módulo chamado pelo CLI):
  bitnet-studio finetune --base tiiuae/Falcon3-10B-Instruct \
      --dataset data/ptbr_tools.jsonl --out adapters/falcon10b-ptbr-tools
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("studio.qlora")


@dataclass
class QLoraConfig:
    base_model: str = "tiiuae/Falcon3-10B-Instruct"
    dataset_path: str = ""
    output_dir: str = "adapters/out"
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    seed: int = 42
    local_files_only: bool = False  # True = air-gapped (modelo já no cache HF)


def run_qlora(cfg: QLoraConfig) -> Path:
    """Treina o adapter e retorna o diretório de saída.

    Import lazy: torch/transformers/peft só são exigidos aqui (extra [train]).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    from studio.training.datasets import conversation_to_text, load_jsonl

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── dados ───────────────────────────────────────────────────────────────
    rows = load_jsonl(Path(cfg.dataset_path))
    texts = [conversation_to_text(r["messages"]) for r in rows]
    log.info("dataset: %d conversas", len(texts))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, local_files_only=cfg.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_len,
            padding=False,
        )

    ds = Dataset.from_dict({"text": texts}).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    # ── modelo 4-bit ────────────────────────────────────────────────────────
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb,
        device_map="auto",
        local_files_only=cfg.local_files_only,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ── trainer ─────────────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        seed=cfg.seed,
        report_to=[],  # sem telemetria (wandb/tensorboard desligados)
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("adapter salvo em %s", out_dir)
    return out_dir
