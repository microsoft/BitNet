"""Merge adapter + base model → GGUF Q4_K_M."""
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER = "adapters/f3b-ptbr-tools-cpu"
BASE = "tiiuae/Falcon3-3B-Instruct"
OUTPUT = "work/falcon3-3b-ptbr-tools-merged"
GGUF = "work/falcon3-3b-ptbr-tools-q4_k_m.gguf"

def main():
    print(f"Carregando modelo base: {BASE}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    print(f"Carregando adapter: {ADAPTER}")
    model = PeftModel.from_pretrained(model, ADAPTER)

    print("Fazendo merge...")
    model = model.merge_and_unload()

    print(f"Salvando modelo mergeado em: {OUTPUT}")
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT)
    tok.save_pretrained(OUTPUT)
    print("Merge completo!")

    # Converter para GGUF (requer llama.cpp)
    print(f"\nPara converter para GGUF, execute:")
    print(f"  python3 convert_hf_to_gguf.py {OUTPUT} --outfile {GGUF} --outtype q4_k_m")
    print(f"\nOu use o CLI do BitNet Studio:")
    print(f"  bitnet-studio merge --base {BASE} --adapter {ADAPTER} --name falcon3-3b-ptbr-tools")

if __name__ == "__main__":
    main()
