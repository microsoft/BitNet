"""Teste focado em ferramentas mem0 do adapter treinado."""
import json
import re
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
print("OK!\n")

def generate(prompt, max_tokens=80):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3,
                                   do_sample=True, pad_token_id=tok.pad_token_id)
    text = tok.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

def extract_tool(text):
    # Regex mais robusto para capturar JSON multiline
    m = re.search(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return None
    return None

MEM0_TESTS = [
    ("mem0_search", "O que sabemos sobre o cliente João Silva?", "protheus-rag__mem0_search", ["pergunta"]),
    ("mem0_add", "Anote que o cliente prefere contato por e-mail", "protheus-rag__mem0_add", ["content", "category"]),
    ("mem0_list", "Liste todas as memórias salvas", "protheus-rag__mem0_list", []),
    ("mem0_stats", "Quantas memórias temos na base?", "protheus-rag__mem0_stats", []),
    ("mem0_delete", "Apague a memória antiga sobre o teste", "protheus-rag__mem0_delete", ["memory_id"]),
]

passed = 0
for name, question, expected_tool, expected_args in MEM0_TESTS:
    print(f"Teste: {name}")
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    t0 = time.time()
    response = generate(prompt)
    elapsed = time.time() - t0
    
    print(f"  Response: {response[:120]}")
    
    tool = extract_tool(response)
    if tool:
        print(f"  Tool: {tool.get('name')}")
        print(f"  Args: {tool.get('arguments', {})}")
        ok = tool.get('name') == expected_tool
        if ok and expected_args:
            ok = all(a in tool.get('arguments', {}) for a in expected_args)
    else:
        print(f"  Tool: None (regex não capturou)")
        ok = False
    
    status = "PASS" if ok else "FAIL"
    if ok: passed += 1
    print(f"  {status} | {elapsed:.1f}s\n")

print(f"{'='*60}")
print(f"MEM0 Resultado: {passed}/{len(MEM0_TESTS)}")
print(f"{'='*60}")
