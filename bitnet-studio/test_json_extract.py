"""Teste com extração JSON robusta."""
import json, re, time, torch
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

def generate(prompt, max_tokens=120):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3,
                                   do_sample=True, pad_token_id=tok.pad_token_id,
                                   eos_token_id=tok.eos_token_id)
    text = tok.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

def extract_tool(text):
    # Método 1: Regex com DOTALL
    m = re.search(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>', text)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass
    
    # Método 2: Encontrar primeiro { e último }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    
    # Método 3: Procurar por nome da tool
    for tool_name in ["mem0_search", "mem0_add", "mem0_list", "mem0_delete", "mem0_stats",
                      "consultar_base_direta", "consultar_dicionario_direto", "buscar_reversa_direto",
                      "consultar_base_interna", "consultar_reversa_rag"]:
        if tool_name in text:
            return {"name": f"protheus-rag__{tool_name}", "arguments": {}}
    
    return None

TESTS = [
    ("mem0_search", "O que sabemos sobre o cliente João Silva?", "protheus-rag__mem0_search"),
    ("mem0_add", "Anote que o cliente prefere contato por e-mail", "protheus-rag__mem0_add"),
    ("mem0_list", "Liste todas as memórias salvas", "protheus-rag__mem0_list"),
    ("mem0_delete", "Apague a memória sobre o teste", "protheus-rag__mem0_delete"),
    ("consultar_base", "Como funciona a função MaFisCalc?", "protheus-rag__consultar_base_direta"),
    ("dicionario", "Quais campos tem a tabela SA1?", "protheus-rag__consultar_dicionario_direto"),
    ("reversa", "Como usar o skill reversa-scout?", "protheus-rag__buscar_reversa_direto"),
    ("anti", "Olá, como vai?", None),
]

passed = 0
for name, question, expected in TESTS:
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    t0 = time.time()
    response = generate(prompt)
    elapsed = time.time() - t0
    
    tool = extract_tool(response)
    got = tool.get('name') if tool else None
    ok = got == expected
    
    status = "PASS" if ok else "FAIL"
    if ok: passed += 1
    print(f"{status} | {name} | expected={expected} | got={got} | {elapsed:.1f}s")
    if not ok:
        print(f"  Response: {response[:150]}")

print(f"\n{'='*60}")
print(f"Resultado com extração robusta: {passed}/{len(TESTS)} ({passed/len(TESTS)*100:.0f}%)")
print(f"{'='*60}")
