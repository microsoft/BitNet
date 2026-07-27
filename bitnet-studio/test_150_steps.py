"""Teste do adapter 150 steps com regex robusto."""
import json, re, time, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-local"

torch.set_num_threads(4)
print("Carregando modelo + adapter (150 steps)...")
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
    # Regex mais robusto - captura até </tool_call> ou <|user|> ou fim
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|\Z)', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            # Tentar extrair manualmente
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                return json.loads(text[start:end])
            except:
                return None
    return None

TESTS = [
    ("mem0_search", "O que sabemos sobre o cliente João Silva?", "protheus-rag__mem0_search", ["pergunta"]),
    ("mem0_add", "Anote que o cliente prefere contato por e-mail", "protheus-rag__mem0_add", ["content", "category"]),
    ("mem0_list", "Liste todas as memórias salvas", "protheus-rag__mem0_list", []),
    ("mem0_delete", "Apague a memória sobre o teste", "protheus-rag__mem0_delete", ["memory_id"]),
    ("consultar_base", "Como funciona a função MaFisCalc?", "protheus-rag__consultar_base_direta", ["pergunta"]),
    ("dicionario", "Quais campos tem a tabela SA1?", "protheus-rag__consultar_dicionario_direto", ["pergunta"]),
    ("reversa", "Como usar o skill reversa-scout?", "protheus-rag__buscar_reversa_direto", ["pergunta"]),
    ("anti", "Olá, como vai?", None, []),
]

passed = 0
for name, question, expected_tool, expected_args in TESTS:
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    t0 = time.time()
    response = generate(prompt)
    elapsed = time.time() - t0
    
    tool = extract_tool(response)
    if tool:
        ok = tool.get('name') == expected_tool
        if ok and expected_args:
            ok = all(a in tool.get('arguments', {}) for a in expected_args)
    else:
        ok = expected_tool is None
    
    status = "PASS" if ok else "FAIL"
    if ok: passed += 1
    print(f"{status} | {name} | got={tool.get('name') if tool else 'None'} | {elapsed:.1f}s")
    if not ok:
        print(f"  Response: {response[:120]}...")

print(f"\n{'='*60}")
print(f"Resultado 150 steps: {passed}/{len(TESTS)} ({passed/len(TESTS)*100:.0f}%)")
print(f"{'='*60}")
