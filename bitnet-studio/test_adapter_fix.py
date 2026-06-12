"""Teste exaustivo do adapter treinado PT-BR tool-calling (versão corrigida)."""
import json
import re
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-local"
OUTPUT_LOG = "test_adapter_results.json"

torch.set_num_threads(4)

print(f"{'='*60}")
print(f"TESTE EXAUSTIVO — Adapter PT-BR Tool-calling")
print(f"{'='*60}\n")

print("[1/3] Carregando tokenizer...")
tok = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True,
    use_fast=False,
    clean_up_tokenization_spaces=False,  # FIX: evita warning BPE
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print("Tokenizer OK")

print("\n[2/3] Carregando modelo + adapter...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print(f"Modelo carregado em {time.time()-t0:.1f}s")

TESTS = [
    ("consultar_base_direta", "Como funciona a função MaFisCalc?", "protheus-rag__consultar_base_direta", ["pergunta"]),
    ("consultar_dicionario_direto", "Quais campos tem a tabela SA1?", "protheus-rag__consultar_dicionario_direto", ["pergunta"]),
    ("consultar_base_interna", "Como funciona o faturamento no Protheus?", "protheus-rag__consultar_base_interna", ["pergunta"]),
    ("buscar_reversa_direto", "Como usar o skill reversa-scout?", "protheus-rag__buscar_reversa_direto", ["pergunta"]),
    ("consultar_reversa_rag", "Como criar um REST endpoint em TLPP?", "protheus-rag__consultar_reversa_rag", ["pergunta"]),
    ("mem0_search", "O que sabemos sobre o cliente João Silva?", "protheus-rag__mem0_search", ["pergunta"]),
    ("mem0_add", "Salve que o cliente prefere contato por e-mail", "protheus-rag__mem0_add", ["content", "category"]),
    ("mem0_list", "Liste todas as memórias", "protheus-rag__mem0_list", []),
    ("mem0_stats", "Quantas memórias temos?", "protheus-rag__mem0_stats", []),
    ("mem0_delete", "Apague a memória sobre o teste", "protheus-rag__mem0_delete", ["memory_id"]),
    ("anti_saudacao", "Olá, como vai?", None, []),
    ("anti_matematica", "Quanto é 2 + 2?", None, []),
]

def generate(prompt):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
    response = tok.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in response:
        parts = response.split("<|assistant|>")
        response = parts[-1].strip()
    return response

def extract_tool(text):
    match = re.search(r'<tool_call>\s*(\{.*?)\s*</tool_call>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            return None
    return None

print(f"\n[3/3] Executando {len(TESTS)} testes...\n")
results = []
passed = 0
failed = 0

for name, question, expected_tool, expected_args in TESTS:
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    t0 = time.time()
    response = generate(prompt)
    elapsed = time.time() - t0
    
    tool_data = extract_tool(response)
    
    ok = True
    if expected_tool is None:
        ok = tool_data is None
    else:
        if tool_data:
            ok = tool_data.get('name') == expected_tool
            if ok and expected_args:
                ok = all(a in tool_data.get('arguments', {}) for a in expected_args)
        else:
            ok = False
    
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    
    print(f"  {status} | {name} | tool={tool_data.get('name') if tool_data else 'None'} | {elapsed:.1f}s")
    
    results.append({
        "name": name,
        "expected": expected_tool,
        "got": tool_data.get('name') if tool_data else None,
        "response": response[:120],
        "elapsed": round(elapsed, 2),
        "pass": ok,
    })

print(f"\n{'='*60}")
print(f"RESULTADO: {passed}/{len(TESTS)} ({passed/len(TESTS)*100:.0f}%)")
print(f"{'='*60}")

with open(OUTPUT_LOG, "w") as f:
    json.dump({"total": len(TESTS), "passed": passed, "failed": failed, 
               "accuracy": passed/len(TESTS), "results": results}, f, indent=2)
print(f"Salvo em: {OUTPUT_LOG}")
