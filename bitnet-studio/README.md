# BitNet Studio

Produto final do ambiente **BitNet CPU-Universal**: serve modelos 1.58bit
em CPU, conecta MCPs "plugáveis" (ex: protheus-rag), faz fine-tuning
QLoRA em GPU modesta e exporta para GGUF / HuggingFace / Ollama.

```
┌─────────────────────────────────────────────────────────────┐
│  Web UI local (vanilla JS, zero CDN)                        │
├─────────────────────────────────────────────────────────────┤
│  API OpenAI-compatible  /v1/chat/completions                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Tool Engine  │←→│  MCP Bridge  │←→│ protheus-rag, ... │  │
│  │ (PT-BR+GBNF) │  │ (stdio RPC)  │  │ (plugável)        │  │
│  └──────┬───────┘  └──────────────┘  └───────────────────┘  │
│         ↓                                                    │
│  llama-server (build BitNet L2-L5, CPU-only, AVX2)           │
├─────────────────────────────────────────────────────────────┤
│  Training (GPU)         │  Export                            │
│  QLoRA 4-bit → merge →  │  GGUF + SHA256 / HF safetensors /  │
│  GGUF quantizado        │  Ollama Modelfile                  │
└─────────────────────────────────────────────────────────────┘
```

## Instalação

```bash
cd bitnet-studio
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # núcleo (serve/export/mcp)
pip install -e ".[train]"   # + treino QLoRA (GPU)
```

Pré-requisito: o build do repo pai (`cmake --build build -j`) — o Studio
usa `build/bin/llama-server` e `build/bin/llama-quantize`.

## Uso rápido

### Servir (CPU-only, D4)

```bash
bitnet-studio serve                 # http://127.0.0.1:8080
bitnet-studio models                # lista o registry
```

Abra `http://127.0.0.1:8080`, escolha o modelo (ex: `falcon3-10b-1.58`)
e pergunte em português. Se a pergunta precisar do Protheus, o modelo
chama o MCP `protheus-rag` automaticamente.

### Testar um MCP isoladamente

```bash
bitnet-studio mcp protheus-rag
bitnet-studio mcp protheus-rag --call consultar_base_direta \
    --args '{"pergunta": "tabela SE1 campos"}'
```

### API (OpenAI-compatible)

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "falcon3-10b-1.58",
       "messages": [{"role": "user",
         "content": "Quais campos tem a tabela SA1 do Protheus?"}]}'
```

A resposta inclui `tool_trace` com cada chamada MCP feita no loop agentic.

### Hot-plug de MCP em runtime

```bash
curl -X POST http://127.0.0.1:8080/mcp -H 'Content-Type: application/json' \
  -d '{"name": "meu-mcp", "command": "python3", "args": ["servidor.py"]}'
```

## Fine-tuning PT-BR + tools (GPU modesta)

Pipeline completo — piloto no 3B, produção no 10B:

```bash
# 1. Gerar dataset sintético de tool-calling a partir das tools reais
bitnet-studio mcp protheus-rag                  # ver tools disponíveis
bitnet-studio dataset synth data/ptbr_tools.jsonl \
    --tools-json data/tools.json --asks data/perguntas.txt -n 10

# 2. Validar
bitnet-studio dataset validate data/ptbr_tools.jsonl

# 3. QLoRA (GPU). Piloto 3B primeiro:
bitnet-studio finetune --base tiiuae/Falcon3-3B-Instruct \
    --dataset data/ptbr_tools.jsonl --out adapters/f3b-ptbr-tools

#    Produção 10B (GPU 8-16GB: reduza --max-seq se faltar VRAM):
bitnet-studio finetune --base tiiuae/Falcon3-10B-Instruct \
    --dataset data/ptbr_tools.jsonl --out adapters/f10b-ptbr-tools \
    --max-seq 512

# 4. Merge + quantizar → GGUF pronto para CPU
bitnet-studio merge --base tiiuae/Falcon3-10B-Instruct \
    --adapter adapters/f10b-ptbr-tools \
    --name falcon3-10b-ptbr-tools --workdir work/

# 5. Registrar em configs/models.yaml e servir
```

## Export para outras plataformas

```bash
bitnet-studio export gguf   --source work/falcon3-10b-ptbr-tools-Q4_K_M.gguf \
    --name falcon3-10b-ptbr-tools
bitnet-studio export ollama --source work/falcon3-10b-ptbr-tools-Q4_K_M.gguf \
    --name falcon3-10b-ptbr-tools
bitnet-studio export hf     --source work/falcon3-10b-ptbr-tools-merged \
    --name falcon3-10b-ptbr-tools
```

## Garantias D4

- Inferência 100% CPU (kernels L2-L5 do BitNet CPU-Universal)
- Servidor escuta apenas em `127.0.0.1`
- Web UI sem CDN, sem fonts externas, sem analytics
- `report_to=[]` no treino (sem wandb/telemetria)
- `--offline` em finetune/merge para ambientes air-gapped
- MCPs são subprocess locais auditáveis (stdio, sem rede no bridge)
