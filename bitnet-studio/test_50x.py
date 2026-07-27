#!/usr/bin/env python3
"""Teste 50x do adapter 150 steps com extração JSON robusta do tool_engine.py.

Verifica:
- 8 perguntas diferentes × 6 iterações = 48 testes + 2 anti-testes
- Extrai tool_call usando parse_tool_call do tool_engine.py
- Reporta taxa de acerto
"""
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Adicionar o path do studio para importar tool_engine
sys.path.insert(0, str(Path(__file__).parent))
from studio.server.tool_engine import parse_tool_call, ToolCall

BASE = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-local"

torch.set_num_threads(4)

print(f"{'='*60}")
print(f"TESTE 50x — Adapter 150 steps + parse_tool_call robusto")
print(f"{'='*60}\n")

print("[1/2] Carregando modelo...")
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
print("OK!\n")

# Mock de tools para parse_tool_call
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

def generate(prompt, max_tokens=80):
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
    ("Olá, como vai?", None),  # anti-test
    ("Quanto é 2 + 2?", None),  # anti-test
]

print(f"[2/2] Executando {len(QUESTIONS)} testes × 6 iterações = {len(QUESTIONS)*6} testes\n")

passed = 0
failed = 0
results = []

for iteration in range(6):
    print(f"--- Iteração {iteration + 1}/6 ---")
    for q_text, expected_tool in QUESTIONS:
        prompt = f"<|user|>\n{q_text}\n<|assistant|>\n"
        t0 = time.time()
        response = generate(prompt)
        elapsed = time.time() - t0

        # Usar parse_tool_call do tool_engine.py
        tc = parse_tool_call(response, TOOLS)
        got_tool = tc.name if tc else None

        ok = got_tool == expected_tool
        if ok:
            passed += 1
        else:
            failed += 1

        status = "PASS" if ok else "FAIL"
        print(f"  {status} | {q_text[:40]:40} | expected={expected_tool or 'None':45} | got={got_tool or 'None':45} | {elapsed:.1f}s")
        if not ok:
            print(f"    Response: {response[:120]}")

        results.append({
            "iteration": iteration + 1,
            "question": q_text,
            "expected": expected_tool,
            "got": got_tool,
            "pass": ok,
            "elapsed": round(elapsed, 2),
        })

print(f"\n{'='*60}")
print(f"RESULTADO FINAL")
print(f"{'='*60}")
print(f"Total: {passed + failed}")
print(f"Passaram: {passed}")
print(f"Falharam: {failed}")
print(f"Taxa de acerto: {passed/(passed+failed)*100:.1f}%")
print(f"{'='*60}")

# Salvar
with open("test_50x_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "total": passed + failed,
        "passed": passed,
        "failed": failed,
        "accuracy": passed/(passed+failed),
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\nResultados salvos em: test_50x_results.json")
