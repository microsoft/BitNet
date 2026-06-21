"""Testar se o modelo gera tool calls corretamente após fine-tune."""
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "tiiuae/Falcon3-3B-Instruct"
ADAPTER = "adapters/f3b-ptbr-tools-cpu"

def test_model():
    print(f"Carregando modelo base: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    if Path(ADAPTER).exists():
        print(f"Carregando adapter: {ADAPTER}")
        model = PeftModel.from_pretrained(model, ADAPTER)
    else:
        print(f"Adapter não encontrado em {ADAPTER}, usando modelo base")
    
    # Testar geração de tool call
    prompts = [
        "<|user|>\nComo funciona a função MaFisCalc?\n<|assistant|>\n",
        "<|user|>\nQuais campos tem a tabela SA1?\n<|assistant|>\n",
        "<|user|>\nO que sabemos sobre o cliente João Silva?\n<|assistant|>\n",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt[:50]}...")
        inputs = tok(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
            )
        
        response = tok.decode(outputs[0], skip_special_tokens=False)
        # Extrair apenas a resposta do assistant
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        print(f"Resposta: {response[:200]}")
        
        # Verificar se contém <tool_call>
        if "<tool_call>" in response:
            print("✅ Tool call detectado!")
            # Tentar extrair JSON
            match = re.search(r'<tool_call>\s*(\{.*?)\s*</tool_call>', response, re.DOTALL)
            if match:
                try:
                    tool_data = json.loads(match.group(1))
                    print(f"   Tool: {tool_data.get('name')}")
                    print(f"   Args: {tool_data.get('arguments')}")
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON inválido: {e}")
            else:
                print("   ⚠️ Formato de tool call não reconhecido")
        else:
            print("❌ Nenhum tool call detectado")

if __name__ == "__main__":
    test_model()
