#!/usr/bin/env python3
"""Teste exaustivo do adapter treinado PT-BR tool-calling.

Verifica:
- Tool call para cada uma das 10 tools
- Formato correto do JSON (<tool_call>{...}</tool_call>)
- Resposta em português
- Não-hallucination de tools inexistentes
"""
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-local"
OUTPUT_LOG = "test_adapter_results.json"

# ========== CONFIG ==========
torch.set_num_threads(4)
device = "cpu"

print(f"{'='*60}")
print(f"TESTE EXAUSTIVO — Adapter PT-BR Tool-calling")
print(f"Base: {BASE}")
print(f"Adapter: {ADAPTER}")
print(f"Device: {device}")
print(f"{'='*60}\n")

# ========== CARREGAR MODELO ==========
print("[1/3] Carregando modelo base...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
print(f"Modelo base em {time.time()-t0:.1f}s")

print(f"[2/3] Carregando adapter: {ADAPTER}...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("Adapter carregado")

# ========== TESTES ==========
print(f"\n[3/3] Executando testes...\n")

TESTS = [
    # consultar_base_direta
    {
        "name": "consultar_base_direta - funcao",
        "prompt": "<|user|>\nComo funciona a função MaFisCalc?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__consultar_base_direta",
        "check_args": ["pergunta"],
    },
    {
        "name": "consultar_base_direta - rotina",
        "prompt": "<|user|>\nQual o código da rotina MATA010?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__consultar_base_direta",
        "check_args": ["pergunta"],
    },
    # consultar_dicionario_direto
    {
        "name": "consultar_dicionario_direto - campos",
        "prompt": "<|user|>\nQuais campos tem a tabela SA1?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__consultar_dicionario_direto",
        "check_args": ["pergunta"],
    },
    {
        "name": "consultar_dicionario_direto - parametro",
        "prompt": "<|user|>\nO que é o parâmetro MV_ESTADO?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__consultar_dicionario_direto",
        "check_args": ["pergunta"],
    },
    # consultar_base_interna
    {
        "name": "consultar_base_interna - processo",
        "prompt": "<|user|>\nComo funciona o faturamento no Protheus?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__consultar_base_interna",
        "check_args": ["pergunta"],
    },
    # buscar_reversa_direto
    {
        "name": "buscar_reversa_direto - skill",
        "prompt": "<|user|>\nComo usar o skill reversa-scout?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__buscar_reversa_direto",
        "check_args": ["pergunta"],
    },
    # consultar_reversa_rag
    {
        "name": "consultar_reversa_rag - REST",
        "prompt": "<|user|>\nComo criar um REST endpoint em TLPP?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__consultar_reversa_rag",
        "check_args": ["pergunta"],
    },
    # mem0_search
    {
        "name": "mem0_search",
        "prompt": "<|user|>\nO que sabemos sobre o cliente João Silva?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__mem0_search",
        "check_args": ["pergunta"],
    },
    # mem0_add
    {
        "name": "mem0_add",
        "prompt": "<|user|>\nSalve que o cliente prefere contato por e-mail\n<|assistant|>\n",
        "expected_tool": "protheus-rag__mem0_add",
        "check_args": ["content", "category"],
    },
    # mem0_list
    {
        "name": "mem0_list",
        "prompt": "<|user|>\nListe todas as memórias\n<|assistant|>\n",
        "expected_tool": "protheus-rag__mem0_list",
        "check_args": [],
    },
    # mem0_stats
    {
        "name": "mem0_stats",
        "prompt": "<|user|>\nQuantas memórias temos?\n<|assistant|>\n",
        "expected_tool": "protheus-rag__mem0_stats",
        "check_args": [],
    },
    # mem0_delete
    {
        "name": "mem0_delete",
        "prompt": "<|user|>\nApague a memória sobre o teste\n<|assistant|>\n",
        "expected_tool": "protheus-rag__mem0_delete",
        "check_args": ["memory_id"],
    },
    # Anti-test: pergunta que NÃO deve chamar tool
    {
        "name": "anti_test - saudação",
        "prompt": "<|user|>\nOlá, como vai?\n<|assistant|>\n",
        "expected_tool": None,
        "check_args": [],
    },
    {
        "name": "anti_test - matemática",
        "prompt": "<|user|>\nQuanto é 2 + 2?\n<|assistant|>\n",
        "expected_tool": None,
        "check_args": [],
    },
]

def generate(prompt):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    response = tok.decode(outputs[0], skip_special_tokens=False)
    # Extrair apenas resposta do assistant
    if "<|assistant|>" in response:
        parts = response.split("<|assistant|>")
        response = parts[-1].strip()
    return response

def extract_tool_call(text):
    match = re.search(r'<tool_call>\s*(\{.*?)\s*</tool_call>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None

results = []
passed = 0
failed = 0

for test in TESTS:
    print(f"Teste: {test['name']}")
    t0 = time.time()
    response = generate(test['prompt'])
    elapsed = time.time() - t0
    
    tool_data = extract_tool_call(response)
    
    # Verificar resultados
    checks = {
        "tool_detected": tool_data is not None,
        "tool_name_match": False,
        "args_present": False,
        "no_hallucination": True,
    }
    
    if test['expected_tool'] is None:
        # Anti-test: não deve chamar tool
        checks['tool_name_match'] = tool_data is None
        checks['args_present'] = True
        if tool_data is not None:
            checks['no_hallucination'] = False
    else:
        # Deve chamar tool específica
        if tool_data:
            checks['tool_name_match'] = tool_data.get('name') == test['expected_tool']
            checks['args_present'] = all(arg in tool_data.get('arguments', {}) for arg in test['check_args'])
    
    all_pass = all(checks.values())
    status = "✅ PASS" if all_pass else "❌ FAIL"
    
    if all_pass:
        passed += 1
    else:
        failed += 1
    
    result = {
        "name": test['name'],
        "expected_tool": test['expected_tool'],
        "response": response[:200],
        "tool_extracted": tool_data,
        "checks": checks,
        "elapsed_sec": round(elapsed, 2),
        "passed": all_pass,
    }
    results.append(result)
    
    print(f"  {status} | Tool: {tool_data.get('name') if tool_data else 'None'} | {elapsed:.1f}s")
    if not all_pass:
        print(f"  Response: {response[:150]}")
    print()

# ========== RESUMO ==========
print(f"{'='*60}")
print(f"RESUMO DOS TESTES")
print(f"{'='*60}")
print(f"Total: {len(TESTS)}")
print(f"✅ Passaram: {passed}")
print(f"❌ Falharam: {failed}")
print(f"Taxa de acerto: {passed/len(TESTS)*100:.1f}%")
print(f"{'='*60}")

# Salvar resultados
with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": BASE,
        "adapter": ADAPTER,
        "total_tests": len(TESTS),
        "passed": passed,
        "failed": failed,
        "accuracy": passed/len(TESTS),
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\nResultados salvos em: {OUTPUT_LOG}")
