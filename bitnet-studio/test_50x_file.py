#!/usr/bin/env python3
"""Teste 50x do adapter 150 steps com extração JSON robusta."""
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from studio.server.tool_engine import parse_tool_call

BASE = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-local"
OUTPUT_FILE = "test_50x_results.json"
LOG_FILE = "test_50x_progress.log"

torch.set_num_threads(4)

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} | {msg}\n")
    print(msg, flush=True)

log(f"{'='*60}")
log("TESTE 50x — Adapter v1 (150 steps, 162 exemplos) + parse_tool_call robusto")
log(f"{'='*60}")

log("[1/2] Carregando modelo...")
tok = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True,
    use_fast=False,
    clean_up_tokenization_spaces=False,
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
log("Modelo carregado!\n")

class MockTool:
    def __init__(self, name):
        self.qualified_name = name
        self.description = ""
        self.input_schema = {}

TOOLS = [
    MockTool("protheus-rag__consultar_base_direta"),
    MockTool("protheus-rag__consultar_dicionario_direto"),
    MockTool("protheus-rag__consultar_base_interna"),
    MockTool("protheus-rag__buscar_reversa_direto"),
    MockTool("protheus-rag__consultar_reversa_rag"),
    MockTool("protheus-rag__mem0_search"),
    MockTool("protheus-rag__mem0_add"),
    MockTool("protheus-rag__mem0_list"),
    MockTool("protheus-rag__mem0_stats"),
    MockTool("protheus-rag__mem0_delete"),
]

def generate(prompt, max_tokens=180):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

QUESTIONS = [
    ("Como funciona a função MaFisCalc?", "protheus-rag__consultar_base_direta"),
    ("Quais campos tem a tabela SA1?", "protheus-rag__consultar_dicionario_direto"),
    ("Como funciona o faturamento no Protheus?", "protheus-rag__consultar_base_interna"),
    ("Como usar o skill reversa-scout?", "protheus-rag__buscar_reversa_direto"),
    ("Como criar um REST endpoint em TLPP?", "protheus-rag__consultar_reversa_rag"),
    ("O que sabemos sobre o cliente João Silva?", "protheus-rag__mem0_search"),
    ("Anote que o cliente prefere contato por e-mail", "protheus-rag__mem0_add"),
    ("Liste todas as memórias salvas", "protheus-rag__mem0_list"),
    ("Quantas memórias temos na base?", "protheus-rag__mem0_stats"),
    ("Apague a memória sobre o teste", "protheus-rag__mem0_delete"),
    ("Olá, como vai?", None),
    ("Quanto é 2 + 2?", None),
]

ITERATIONS = 6
TOTAL_TESTS = len(QUESTIONS) * ITERATIONS

log(f"[2/2] Executando {TOTAL_TESTS} testes ({len(QUESTIONS)} perguntas × {ITERATIONS} iterações)\n")

passed = 0
failed = 0
results = []

for iteration in range(ITERATIONS):
    log(f"--- Iteração {iteration + 1}/{ITERATIONS} ---")
    for q_text, expected_tool in QUESTIONS:
        prompt = f"<|user|>\n{q_text}\n<|assistant|>\n"
        t0 = time.time()
        response = generate(prompt)
        elapsed = time.time() - t0

        tc = parse_tool_call(response, TOOLS)
        got_tool = tc.name if tc else None

        ok = got_tool == expected_tool
        if ok:
            passed += 1
        else:
            failed += 1

        status = "PASS" if ok else "FAIL"
        log(f"  {status} | {q_text[:35]:35} | exp={expected_tool or 'None':40} | got={got_tool or 'None':40} | {elapsed:.1f}s")
        if not ok:
            log(f"    Response: {response[:150]}")

        results.append({
            "iteration": iteration + 1,
            "question": q_text,
            "expected": expected_tool,
            "got": got_tool,
            "pass": ok,
            "elapsed": round(elapsed, 2),
        })

log(f"\n{'='*60}")
log("RESULTADO FINAL")
log(f"{'='*60}")
log(f"Total: {passed + failed}")
log(f"Passaram: {passed}")
log(f"Falharam: {failed}")
log(f"Taxa de acerto: {passed/(passed+failed)*100:.1f}%")
log(f"{'='*60}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({
        "total": passed + failed,
        "passed": passed,
        "failed": failed,
        "accuracy": passed/(passed+failed),
        "results": results,
    }, f, ensure_ascii=False, indent=2)

log(f"\nResultados salvos em: {OUTPUT_FILE}")
