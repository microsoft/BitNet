"""Teste simples e rápido."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, re

BASE = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-local"

torch.set_num_threads(4)
print("Carregando...")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("OK!")

# Teste único e rápido
prompt = "<|user|>\nO que sabemos sobre o cliente João Silva?\n<|assistant|>\n"
inputs = tok(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.3, do_sample=True)
response = tok.decode(outputs[0], skip_special_tokens=False)
print(f"Response: {response}")

# Extrair tool
m = re.search(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>', response)
if m:
    print(f"Tool JSON: {m.group(1)}")
    try:
        tool = json.loads(m.group(1))
        print(f"Tool: {tool}")
    except:
        print("JSON parse error")
else:
    print("No tool call found")
