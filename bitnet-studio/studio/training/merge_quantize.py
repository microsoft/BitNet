"""Merge do adapter LoRA + conversão para GGUF (pipeline pós-treino).

Fluxo:
  1. merge:    base fp16 + adapter → modelo HF merged (safetensors)
  2. convert:  HF merged → GGUF fp16 (convert-hf-to-gguf do llama.cpp)
  3. quantize: GGUF fp16 → Q4_K_M (ou i2_s para 1.58bit via utils BitNet)

O resultado entra direto no registry (configs/models.yaml) e roda no
servidor CPU-only.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from studio.config import REPO_ROOT

log = logging.getLogger("studio.merge")

CONVERT_SCRIPT = REPO_ROOT / "3rdparty" / "llama.cpp" / "convert_hf_to_gguf.py"
CONVERT_SCRIPT_ALT = REPO_ROOT / "3rdparty" / "llama.cpp" / "convert-hf-to-gguf.py"
QUANTIZE_BIN = REPO_ROOT / "build" / "bin" / "llama-quantize"


def merge_adapter(base_model: str, adapter_dir: Path, out_dir: Path,
                  local_files_only: bool = False) -> Path:
    """Merge LoRA → modelo HF completo em fp16. Requer extra [train]."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("carregando base %s em fp16 (CPU ok, lento)...", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    log.info("merging adapter...")
    model = model.merge_and_unload()
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model,
                                        local_files_only=local_files_only)
    tok.save_pretrained(str(out_dir))
    log.info("modelo merged salvo em %s", out_dir)
    return out_dir


def convert_to_gguf(hf_dir: Path, out_gguf: Path,
                    dtype: str = "f16") -> Path:
    """HF → GGUF usando o conversor do llama.cpp do submodule."""
    script = CONVERT_SCRIPT if CONVERT_SCRIPT.exists() else CONVERT_SCRIPT_ALT
    if not script.exists():
        raise FileNotFoundError(
            f"conversor não encontrado ({CONVERT_SCRIPT.name}); "
            "verifique o submodule 3rdparty/llama.cpp"
        )
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", str(script), str(hf_dir),
           "--outfile", str(out_gguf), "--outtype", dtype]
    log.info("convertendo: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_gguf


def quantize_gguf(in_gguf: Path, out_gguf: Path,
                  quant: str = "Q4_K_M") -> Path:
    """GGUF fp16 → quantizado (Q4_K_M, Q5_K_M, etc.)."""
    if not QUANTIZE_BIN.exists():
        raise FileNotFoundError(
            f"llama-quantize não encontrado em {QUANTIZE_BIN}; "
            "compile com: cmake --build build -j"
        )
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(QUANTIZE_BIN), str(in_gguf), str(out_gguf), quant]
    log.info("quantizando: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_gguf


def full_pipeline(base_model: str, adapter_dir: Path, work_dir: Path,
                  model_name: str, quant: str = "Q4_K_M",
                  local_files_only: bool = False) -> Path:
    """Pipeline completo: merge → GGUF fp16 → quantizado."""
    merged = merge_adapter(base_model, adapter_dir,
                           work_dir / f"{model_name}-merged",
                           local_files_only=local_files_only)
    fp16_gguf = convert_to_gguf(merged, work_dir / f"{model_name}-f16.gguf")
    final = quantize_gguf(fp16_gguf,
                          work_dir / f"{model_name}-{quant}.gguf", quant)
    log.info("pipeline completo: %s", final)
    return final
