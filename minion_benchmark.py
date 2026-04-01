# Intelligence Benchmark: BitNet (Sovereign Minion)

import os
import time
import subprocess
import json

MODELS = {
    "BitNet-2B-4T": "models/2B-4T/ggml-model-i2_s.gguf",
    "Llama3-8B": "models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf",
    "Falcon3-10B": "models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf"
}

PROMPTS = [
    {
        "id": "logic_math",
        "description": "Razonamiento Cero-Shot",
        "prompt": "Here is a logic puzzle: I have 3 apples. I give 1 to John and I eat 1. How many apples do I have left? Think step by step and provide the final number.",
        "expected_keywords": ["1", "one"]
    },
    {
        "id": "json_extraction",
        "description": "Extracción de Datos Estructurada",
        "prompt": "Extract the person's name and age from this text into strict JSON format with keys 'name' and 'age': 'My name is Carlos and I am 34 years old.' Output ONLY the JSON object.",
        "expected_keywords": ["Carlos", "34", "{", "}"]
    },
    {
        "id": "coding_python",
        "description": "Generación de Código Base",
        "prompt": "Write a Python function named 'reverse_string' that takes a string 's' and returns it reversed. Do not provide explanations, only the function code.",
        "expected_keywords": ["def", "reverse_string", "return", "[::-1]"]
    }
]

def check_memory():
    # Simple grab of current free memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    return int(line.split()[1]) / 1024 # MB
    except:
        return 0

def run_model(model_name, model_path, prompt_data):
    if not os.path.exists(model_path):
        return {"error": f"Model {model_path} not found"}
    
    print(f"\n[{model_name}] Ejecutando: {prompt_data['description']}")
    
    # Executing via llama-cli directly
    cmd = [
        "./build/bin/llama-cli",
        "-m", model_path,
        "-p", prompt_data["prompt"],
        "-n", "256",
        "-t", "8",
        "-c", "1024",
        "--temp", "0.2"
    ]
    
    start_mem = check_memory()
    start_time = time.time()
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    end_time = time.time()
    end_mem = check_memory()
    
    mem_diff = start_mem - end_mem if start_mem and end_mem else 0
    time_taken = end_time - start_time
    
    # Extracting the actual generated response between markers
    response = stdout
    if "generate:" in response:
        response = response.split("generate:", 1)[-1].split("\n", 1)[-1]
    if "llama_perf_sampler_print:" in response:
        response = response.split("llama_perf_sampler_print:")[0]
        
    response = response.replace(prompt_data["prompt"], "").strip()
    score = 0
    for kw in prompt_data["expected_keywords"]:
        if kw.lower() in response.lower():
            score += (100 / len(prompt_data["expected_keywords"]))
            
    return {
        "discipline": prompt_data["id"],
        "score": round(score),
        "time_s": round(time_taken, 2),
        "ram_cost_mb": round(mem_diff, 2),
        "output": response[-500:] # store last 500 chars to avoid massive logs
    }

def main():
    results = {}
    for m_name, m_path in MODELS.items():
        results[m_name] = []
        for p in PROMPTS:
            res = run_model(m_name, m_path, p)
            results[m_name].append(res)
            
    print("\n\n=== BENCHMARK RESULTS ===")
    print(json.dumps(results, indent=2))
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
