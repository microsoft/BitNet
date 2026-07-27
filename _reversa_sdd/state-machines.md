# Máquinas de Estado — BitNet

> Gerado pelo Reversa Detective | 2026-05-03

---

## 1. Pipeline de Setup do Ambiente

Estado da preparação do ambiente para inferência. Representado implicitamente pelo estado do filesystem e pelos artefatos gerados.

```mermaid
stateDiagram-v2
    [*] --> Não_Configurado

    Não_Configurado --> Baixando_Modelo : hf_repo fornecido\nhuggingface-cli download
    Não_Configurado --> Modelo_Local : model_dir existente

    Baixando_Modelo --> Modelo_Local : download completo
    Baixando_Modelo --> Erro : falha de rede / repo inválido

    Modelo_Local --> Gerando_Kernels : GGUF não existe\ngen_code()

    Gerando_Kernels --> Kernels_Prontos : codegen_tl1/tl2.py executado\nou preset copiado

    Kernels_Prontos --> Compilando : compile()

    Compilando --> Binários_Prontos : cmake --build bem-sucedido
    Compilando --> Erro : cmake não instalado\nou falha de compilação

    Binários_Prontos --> Convertendo : prepare_model()\nGGUF não existe

    Convertendo --> Pronto : GGUF gerado\ne válido (size > 0)
    Convertendo --> Erro : falha na conversão

    Modelo_Local --> Pronto : GGUF já existe\ne size > 0
    Pronto --> [*]
    Erro --> [*]
```

**Estados:**

| Estado | Condição no Filesystem |
|--------|----------------------|
| `Não_Configurado` | Nenhum artefato local |
| `Modelo_Local` | `model_dir/` existe com pesos HF |
| `Kernels_Prontos` | `include/bitnet-lut-kernels.h` existe |
| `Binários_Prontos` | `build/bin/llama-cli` existe |
| `Pronto` | `model_dir/ggml-model-{type}.gguf` existe e `size > 0` |

**Nota:** O sistema não persiste estado explicitamente — rederiva o estado atual verificando a existência dos artefatos. 🟡 INFERIDO

---

## 2. Ciclo de Vida da Geração de Texto (GPU)

Estados da geração em `FastGen.generate_all`.

```mermaid
stateDiagram-v2
    [*] --> Inicializando

    Inicializando --> Compilando_CUDA_Graph : build() completo\ncarregou fp16 + int2

    Compilando_CUDA_Graph --> Aguardando_Prompt : compile_prefill() + compile_generate()\nCUDA graphs capturados

    Aguardando_Prompt --> Tokenizando : prompt recebido

    Tokenizando --> Prefill : tokens prontos\npadded para prompt_length

    Prefill --> Decodificando : logits do último token\nnext_token selecionado

    Decodificando --> Decodificando : niter < gen_length\ne next_token ≠ eot_id\nkv_seqlen += 1

    Decodificando --> Finalizando : next_token == eot_id\nOU niter == gen_length

    Finalizando --> Aguardando_Prompt : trim_answer + decode\ntexto retornado

    Aguardando_Prompt --> [*] : EOFError / SIGINT
```

**Transições de estado de sampling:**

```mermaid
stateDiagram-v2
    [*] --> Greedy : use_sampling=False
    [*] --> Nucleus : use_sampling=True

    Greedy --> Próximo_Token : argmax(logits)
    Nucleus --> Softmax_Temp : logits / temp (0.7)
    Softmax_Temp --> Top_P : probs, p=0.95
    Top_P --> Próximo_Token : multinomial(probs_filtradas, 1)

    Próximo_Token --> [*]
```

---

## 3. Ciclo de Vida do Checkpoint GPU

Transições dos formatos de arquivo durante a preparação do modelo GPU.

```mermaid
stateDiagram-v2
    [*] --> HuggingFace_Safetensors : modelo HF com pesos ternários\nem safetensors

    HuggingFace_Safetensors --> Checkpoint_Unificado_PT : convert_safetensors.py\nremapeia nomes + inverte RoPE Q/K

    Checkpoint_Unificado_PT --> Modelo_FP16 : quant_weight_fp16()\nternário simulado em BF16
    Checkpoint_Unificado_PT --> Modelo_INT2 : quant_weight_int8() + convert_int2()\nternário comprimido + scales

    Modelo_FP16 --> Em_Inferência_Prefill : torch.load weights_only=True\nprefill_model.load_state_dict()
    Modelo_INT2 --> Em_Inferência_Decode : torch.load weights_only=True\ndecode_model.load_state_dict()

    Em_Inferência_Prefill --> [*] : geração concluída
    Em_Inferência_Decode --> [*] : geração concluída
```

---

## 4. Pipeline de Conversão CPU (HF → GGUF)

```mermaid
stateDiagram-v2
    [*] --> Pesos_HF : safetensors ou bin no model_dir

    Pesos_HF --> GGUF_F32 : convert-hf-to-gguf-bitnet.py\n--outtype f32\n(apenas para i2_s path)

    Pesos_HF --> GGUF_TL1 : convert-hf-to-gguf-bitnet.py\n--outtype tl1\n(ARM64 only)

    Pesos_HF --> GGUF_TL2 : convert-hf-to-gguf-bitnet.py\n--outtype tl2\n(x86_64 only)

    GGUF_F32 --> GGUF_I2S : llama-quantize I2_S\nternário packed 2-bit

    GGUF_TL1 --> Pronto_para_Inferência_CPU
    GGUF_TL2 --> Pronto_para_Inferência_CPU
    GGUF_I2S --> Pronto_para_Inferência_CPU

    Pronto_para_Inferência_CPU --> [*]
```

**Regra de roteamento:**

| Plataforma | Tipo de quantização | Path de conversão |
|------------|-------------------|------------------|
| ARM64 | `tl1` | Direto HF → TL1 GGUF |
| ARM64 | `i2_s` | HF → F32 GGUF → I2_S GGUF (2 passos) |
| x86_64 | `tl2` | Direto HF → TL2 GGUF |
| x86_64 | `i2_s` | HF → F32 GGUF → I2_S GGUF (2 passos) |

**Motivo dos 2 passos para I2_S:** O `llama-quantize` precisa de um modelo F32 como entrada; não consegue quantizar diretamente de BF16/F16. 🟡 INFERIDO
