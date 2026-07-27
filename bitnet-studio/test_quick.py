"""Teste rápido do adapter treinado."""
import json, re, time, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-local"

torch.set_num_threads(4)
print("Carregando tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=False, clean_up_tokenization_spaces=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Carregando modelo + adapter...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("OK!\n")

def generate(prompt, max_tokens=60):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3, 
                                   do_sample=True, pad_token_id=tok.pad_token_id)
    text = tok.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

def extract_tool(text):
    m = re.search(r'<tool_call>\s*(\{.*?)\s*</tool_call>', text, re.DOTALL)
    return json.loads(m.group(1)) if m else None

TESTS = [
    ("Como funciona a função MaFisCalc?", "protheus-rag__consultar_base_direta"),
    ("Quais campos tem a tabela SA1?", "protheus-rag__consultar_dicionario_direto"),
    ("O que sabemos sobre o cliente João Silva?", "protheus-rag__mem0_search"),
]

passed = 0
for question, expected in TESTS:
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    t0 = time.time()
    response = generate(prompt)
    elapsed = time.time() - t0
    tool = extract_tool(response)
    ok = tool and tool.get('name') == expected
    status = "PASS" if ok else "FAIL"
    if ok: passed += 1
    print(f"{status} | {expected} | got={tool.get('name') if tool else 'None'} | {elapsed:.1f}s")
    print(f"  Response: {response[:100]}...")

print(f"\nResultado: {passed}/{len(TESTS)}")
