"""Teste 3x rápido."""
import json, sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent))
from studio.server.tool_engine import parse_tool_call

BASE = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-local"
torch.set_num_threads(4)

print("Carregando modelo...")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=False, clean_up_tokenization_spaces=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("OK!")

class MockTool:
    def __init__(self, name):
        self.qualified_name = name
        self.description = ""
        self.input_schema = {}

TOOLS = [MockTool(n) for n in [
    "protheus-rag__consultar_base_direta",
    "protheus-rag__consultar_dicionario_direto",
    "protheus-rag__mem0_search",
    "protheus-rag__mem0_add",
    "protheus-rag__mem0_list",
    "protheus-rag__mem0_delete",
]]

def generate(prompt, max_tokens=60):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3, do_sample=True, pad_token_id=tok.pad_token_id)
    text = tok.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

QUESTIONS = [
    ("Como funciona a função MaFisCalc?", "protheus-rag__consultar_base_direta"),
    ("O que sabemos sobre o cliente João Silva?", "protheus-rag__mem0_search"),
    ("Liste todas as memórias salvas", "protheus-rag__mem0_list"),
    ("Olá, como vai?", None),
]

passed = 0
failed = 0
for iteration in range(3):
    print(f"--- Iteração {iteration+1}/3 ---")
    for q_text, expected in QUESTIONS:
        prompt = f"<|user|>\n{q_text}\n<|assistant|>\n"
        response = generate(prompt)
        tc = parse_tool_call(response, TOOLS)
        got = tc.name if tc else None
        ok = got == expected
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {'PASS' if ok else 'FAIL'} | {q_text[:30]:30} | expected={expected or 'None':40} | got={got or 'None':40}")
        if not ok:
            print(f"    Response: {response[:100]}")

print(f"\nResultado: {passed}/{passed+failed} ({passed/(passed+failed)*100:.0f}%)")
