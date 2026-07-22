# Análise de Código — BitNet

> Gerado pelo Reversa Archaeologist | 2026-05-03 | doc_level: completo

---

> ## ⚠️ ATENÇÃO — Documento parcial (2026-06-06)
>
> Este doc foi gerado em **2026-05-03 sobre o upstream** `microsoft/BitNet`, que tinha dois backends (CPU + GPU). O fork [`peder1981/BitNet`](https://github.com/peder1981/BitNet) (este) **removeu a pipeline `gpu/`** em junho/2026 e adicionou **5 níveis algébricos (L1-L5)** como pesquisa (WHT, ACDC, Tropical, HRR).
>
> **15 referências a `gpu/` neste documento** apontam para módulos **inexistentes no fork** (Módulos 4-11 deste documento, 268 linhas: `gpu/model.py`, `gpu/generate.py`, `gpu/tokenizer.py`, `gpu/pack_weight.py`, `gpu/convert_checkpoint.py`, `gpu/convert_safetensors.py`, `gpu/sample_utils.py`, `gpu/stats.py`).
>
> **Conteúdo válido** (referente ao fork atual):
> - Módulo 1: `run_inference.py` ✅
> - Módulo 2: `run_inference_server.py` ✅
> - Módulo 3: `setup_env.py` ✅
> - Módulo 12: `src/ggml-bitnet-lut.cpp` ✅
> - Módulo 13: `src/ggml-bitnet-mad.cpp` ✅
> - Módulo 14: `utils/codegen_tl1.py` ✅
> - Módulo 15: `utils/codegen_tl2.py` ✅
>
> **Para o estado arquitetural atual do fork**, veja:
> - [`architecture.md`](architecture.md) — visão geral
> - [`c4-containers.md`](c4-containers.md) e [`c4-components.md`](c4-components.md) — containers e componentes
> - [`erd-complete.md`](erd-complete.md) — entidades
> - `gap-analysis.md` (P6) — limitação conhecida (L3/L5 como arquitetura de treinamento, não validadas empiricamente)
>
> **Lacunas adicionais no fork** (não cobertas por este doc):
> - `src/ggml-bitnet-wht.cpp` (L2) — adicionado após 2026-05-03
> - `src/ggml-bitnet-fwht.cpp` (L3) — adicionado após 2026-05-03
> - `src/ggml-bitnet-tropical.cpp` (L4) — adicionado após 2026-05-03
> - `src/ggml-bitnet-hrr.cpp` (L5) — adicionado após 2026-05-03
> - `src/ggml-bitnet-dispatch.cpp` (orquestra L1-L5) — adicionado após 2026-05-03
> - `src/ggml-bitnet-kv-cache.cpp` (K_i8 cache, L4/L5) — adicionado em 2026-06-06

---

## Visão Geral do Sistema

**BitNet** é a implementação de referência da Microsoft para inferência eficiente de LLMs com quantização de 1 bit (ternária: {-1, 0, 1}). O projeto suporta dois backends de inferência:

1. **CPU** — via llama.cpp com kernels customizados (I2_S, TL1, TL2)
2. **GPU** — via PyTorch com CUDA Graphs e kernel CUDA customizado INT8×INT2

---

## Módulo 1: `run_inference.py` 🟢 CONFIRMADO

**Papel:** Ponto de entrada para inferência no modo CPU.

### Funções principais

| Função | Parâmetros | Retorno | Descrição |
|--------|-----------|---------|-----------|
| `run_inference()` | (via globals) | void | Monta e executa `llama-cli` via subprocess |
| `run_command(command, shell)` | list/str, bool | void | Wrapper subprocess com `check=True`; chama `sys.exit(1)` em falha |
| `signal_handler(sig, frame)` | int, frame | void | Captura SIGINT e termina graciosamente |

### Argumentos CLI

| Flag | Tipo | Default | Descrição |
|------|------|---------|-----------|
| `-m/--model` | str | `models/bitnet_b1_58-3B/ggml-model-i2_s.gguf` | Caminho do modelo GGUF |
| `-n/--n-predict` | int | 128 | Tokens a gerar |
| `-p/--prompt` | str | obrigatório | Prompt de entrada |
| `-t/--threads` | int | 2 | Threads de CPU |
| `-c/--ctx-size` | int | 2048 | Tamanho do contexto |
| `-temp/--temperature` | float | 0.8 | Temperatura de sampling |
| `-cnv/--conversation` | flag | False | Modo de conversa (instruct) |

**Nota crítica:** `-ngl 0` está hardcoded — GPU offload desabilitado; `-b 1` força batch size 1.

---

## Módulo 2: `run_inference_server.py` 🟢 CONFIRMADO

**Papel:** Ponto de entrada para servidor HTTP (OpenAI-compatible via llama-server).

### Diferenças em relação a `run_inference.py`

- Usa `llama-server` em vez de `llama-cli`
- Flag `-cb` (continuous batching) habilitada
- Expõe host/port configuráveis (default: `127.0.0.1:8080`)
- `n_predict` default = 4096 (vs 128 no CLI)
- Flag `-cnv` removida (não suportada pelo servidor)

---

## Módulo 3: `setup_env.py` 🟢 CONFIRMADO

**Papel:** Orquestrador do pipeline de setup — download, conversão, geração de kernels, compilação.

### Constantes de domínio

```python
SUPPORTED_HF_MODELS = {
    "1bitLLM/bitnet_b1_58-large": {"model_name": "bitnet_b1_58-large"},
    "1bitLLM/bitnet_b1_58-3B": {"model_name": "bitnet_b1_58-3B"},
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": {"model_name": "Llama3-8B-1.58-100B-tokens"},
    "tiiuae/Falcon3-*": {...},
    "microsoft/BitNet-b1.58-2B-4T": {...},
    ...  # 16 modelos no total
}

SUPPORTED_QUANT_TYPES = {
    "arm64": ["i2_s", "tl1"],
    "x86_64": ["i2_s", "tl2"]
}

COMPILER_EXTRA_ARGS = {
    "arm64": ["-DBITNET_ARM_TL1=OFF"],
    "x86_64": ["-DBITNET_X86_TL2=OFF"]
}

ARCH_ALIAS = {
    "AMD64": "x86_64", "x86_64": "x86_64", "x86": "x86_64",
    "aarch64": "arm64", "arm64": "arm64", "ARM64": "arm64"
}
```

### Pipeline de execução (função `main`)

```
setup_gguf() → gen_code() → compile() → prepare_model()
```

### Lógica de `prepare_model()`

```
if hf_repo → huggingface-cli download → model_dir/model_name/
if gguf não existe ou vazio:
    if quant_type.startswith("tl"):
        convert-hf-to-gguf-bitnet.py --outtype tl1/tl2
    else (i2s):
        convert-hf-to-gguf-bitnet.py --outtype f32
        llama-quantize f32.gguf i2s.gguf I2_S 1 [1 se quant_embd]
```

### Lógica de `gen_code()` (geração de kernels)

Seleção de parâmetros GEMM por modelo:

| Modelo | BM | BK | bm |
|--------|----|----|-----|
| bitnet_b1_58-large | 256,128,256 | 128,64,128 (TL1) / 96,192,96 (TL2) | 32,64,32 (TL1) / 32,32,32 (TL2) |
| bitnet_b1_58-3B | 160,320,320 | 64,128,64 (TL1) / 96,96,96 (TL2) | 32,64,32 (TL1) / 32,32,32 (TL2) |
| Llama3/Falcon models | 256,128,256,128 | 128,64,128,64 (TL1) / 96,96,96,96 (TL2) | 32,64,32,64 (TL1) / 32,32,32,32 (TL2) |
| BitNet-b1.58-2B-4T | igual ao 3B | igual ao 3B | igual ao 3B |

**Nota:** BitNet-b1.58-2B-4T usa mesmas config do 3B — pode ser intencionalmente compatível ou pendência de atualização. 🟡 INFERIDO

---

## Módulo 4: `gpu/model.py` 🟢 CONFIRMADO

**Papel:** Arquitetura do modelo Transformer BitNet para inferência GPU.

### Configuração padrão `ModelArgs`

```python
dim = 2560         # dimensão do modelo
n_layers = 30      # camadas transformer
n_heads = 20       # cabeças de atenção
n_kv_heads = 5     # cabeças de KV (GQA ratio = 4:1)
vocab_size = 128256 # vocabulário Llama 3
ffn_dim = 6912     # dimensão da FFN
norm_eps = 1e-5    # epsilon da RMSNorm
rope_theta = 500000.0  # frequência base do RoPE
use_kernel = False  # modo prefill usa BitLinear; decode usa BitLinearKernel
```

→ Configuração corresponde ao modelo BitNet 2B.

### Hierarquia de classes

```
nn.Module
├── BitLinear (extends nn.Linear)        — prefill: quant input → F.linear em fp16
├── BitLinearKernel (nn.Module)          — decode: quant input → CUDA kernel int8×int2
├── Attention (nn.Module)
│   ├── wqkv: BitLinear/Kernel           — Q+K+V concatenados
│   ├── wo: BitLinear/Kernel             — projeção de saída
│   └── attn_sub_norm: RMSNorm
├── FeedForward (nn.Module)
│   ├── w13: BitLinear/Kernel            — gate + up concatenados (SwiGLU-like)
│   ├── w2: BitLinear/Kernel             — down projection
│   └── ffn_sub_norm: RMSNorm
├── TransformerBlock (nn.Module)
│   ├── attention: Attention
│   ├── feed_forward: FeedForward
│   ├── attention_norm: RMSNorm
│   └── ffn_norm: RMSNorm
└── Transformer (nn.Module)
    ├── tok_embeddings: nn.Embedding
    ├── layers: ModuleList[TransformerBlock × n_layers]
    ├── norm: RMSNorm
    └── output: nn.Linear (sem bias, vocab_size saída)
```

### Algoritmo de quantização de input (BitLinear)

```python
# Per-token quantization
s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
quantized = (input * s).round().clamp(-128, 127)
# BitLinear: retorna quantized / s (simula quantização em fp16)
# BitLinearKernel: retorna int8, passa para kernel CUDA
```

### Algoritmo de atenção (GQA + RoPE + Flash Attention)

```python
xqkv = wqkv(x)  # shape: [seq, (n_heads + 2*n_kv_heads) * head_dim]
xq = xqkv[:, :n_heads*head_dim]
xk, xv = xqkv[:, n_heads*head_dim:].chunk(2, 1)

# Reshape para GQA: heads_per_group = n_heads // n_kv_heads = 4
xq = xq.view(1, seq, n_kv_heads, heads_per_group, head_dim)
xk = xk.view(1, seq, n_kv_heads, 1, head_dim)
xv = xv.view(1, seq, n_kv_heads, 1, head_dim)

# RoPE + atualização do KV cache via xformers rope_padded
xq = rope_padded(xq, xk, xv, cache_k, cache_v, attn_bias, theta)

# Flash Attention forward
output = fmha.memory_efficient_attention_forward(xq, cache_k, cache_v, attn_bias)
output = attn_sub_norm(output)  # sub-norm pós-atenção (exclusivo BitNet)
output = wo(output)
```

### Algoritmo da FFN (SwiGLU-like com squared ReLU)

```python
x13 = w13(x)                          # [seq, 2*ffn_dim]
x1, x3 = x13.chunk(2, -1)            # gate e up separados
inner = ffn_sub_norm(relu(x1)**2 * x3)  # squared relu (não SiLU)
output = w2(inner)
```

**Diferença importante:** Usa `squared_relu` em vez do `SiLU`/`GELU` típico de LLMs.

### Cache KV

```python
# shape: (1, length, n_kv_heads, heads_per_group, head_dim)
# length = max_batch * max_seq
# Expandido via .expand() para heads_per_group sem duplicar memória
```

---

## Módulo 5: `gpu/generate.py` 🟢 CONFIRMADO

**Papel:** Motor de inferência GPU com CUDA Graphs para alta performance.

### Classe `FastGen`

**Design dual-model:** Dois modelos carregados simultaneamente:
- `prefill_model`: usa `BitLinear` (fp16) — maior acurácia na fase de prefill
- `decode_model`: usa `BitLinearKernel` (int2) — máxima velocidade no auto-regressivo

### Fluxo de inicialização (`build`)

```
1. Criar ModelArgs com use_kernel=False (prefill) e True (decode)
2. Carregar pesos fp16 → prefill_model
3. Carregar pesos int2 → decode_model
4. compile_prefill() — cria CUDA graph para prefill
5. compile_generate() — cria CUDA graph para decode
```

### Compilação com CUDA Graphs (`compile_prefill`, `compile_generate`)

```
1. Alocar cache KV (gen_bsz * max_seq_length por camada)
2. Criar atenção bias estática (padded sequences)
3. Warm-up: executar modelo uma vez no stream auxiliar
4. Gravar CUDA graph: capturar kernel launches para replay
5. Retornar closure `replay(tokens, seq_lens)` que faz copy_() + graph.replay()
```

**Por que CUDA Graphs:** Elimina overhead de launch de kernels PyTorch no loop de decode, crítico para batch pequeno.

### Algoritmo de geração (`generate_all`)

```
Fase prefill:
  - Padding dos prompts para prompt_length
  - replay(tokens_padded, None) → logits[kv_seqlen-1, :]
  - Selecionar next_token via argmax ou top_p

Fase decode (loop):
  for niter in range(1, gen_length):
    kv_seqlen += 1  (incrementa contador de contexto)
    replay(next_token, kv_seqlen) → logits
    next_token = argmax(logits) ou top_p(probs, 0.95)
    if next_token == eos_id: break

Pós-processamento:
  trim_answer: trunca na posição do token EOS
```

### Parâmetros de `GenArgs`

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `gen_length` | 32 | Tokens a gerar |
| `gen_bsz` | 1 | Batch size |
| `prompt_length` | 64 | Comprimento fixo do prompt (pad/truncate) |
| `temperature` | 0.8 | Temperatura |
| `top_p` | 0.9 | Threshold nucleus sampling |

---

## Módulo 6: `gpu/tokenizer.py` 🟢 CONFIRMADO

**Papel:** Tokenizador Tiktoken com formato de diálogo Llama 3.

### Classe `Tokenizer`

- Usa BPE tiktoken com `load_tiktoken_bpe`
- `num_reserved_special_tokens = 256`
- Padrão regex Llama 3 para tokenização subword

### Tokens especiais

| Token | ID | Descrição |
|-------|----|-----------|
| `<\|begin_of_text\|>` | base + 0 | BOS |
| `<\|end_of_text\|>` | base + 1 | EOS |
| `<\|start_header_id\|>` | base + 6 | Início de cabeçalho de turno |
| `<\|end_header_id\|>` | base + 7 | Fim de cabeçalho de turno |
| `<\|eot_id\|>` | base + 9 | End of turn (stop token) |

### Algoritmo `encode`

```
TIKTOKEN_MAX_ENCODE_CHARS = 400_000    # limite de segurança pyo3
MAX_NO_WHITESPACES_CHARS = 25_000      # max chars não-espaço consecutivos

Divide texto em chunks via _split_whitespaces_or_nonwhitespaces()
→ codifica cada chunk separadamente
→ prepend BOS e/ou append EOS se solicitado
```

**Motivo da divisão:** Bug no tiktoken >400k chars pode causar PanicException via pyo3.

### Classe `ChatFormat`

Formata diálogos no formato Llama 3:

```
<|begin_of_text|>
User: {conteúdo}<|eot_id|>
Assistant: {conteúdo}<|eot_id|>
```

**Nota:** Headers usando texto plano ("User: ", "System: ") em vez dos tokens `<|start_header_id|>/<|end_header_id|>` — provável adaptação do formato original.

---

## Módulo 7: `gpu/pack_weight.py` 🟢 CONFIRMADO

**Papel:** Empacotamento e permutação de pesos int2 para layout WMMA da GPU.

### Algoritmo `convert_weight_int8_to_int2`

```
Entrada: weight tensor int8 com valores {-1, 0, +1}
Saída: weight tensor int8 comprimido (N × K/4)

1. weight += 2  → valores {1, 2, 3} (shift para não-negativo)
2. permutate_weight_fastest(weight)
   → Reordena blocos wmma_n=16 × wmma_k=32 para layout de carga WMMA
3. compress_int2_to_int8(permutated_weight)
   → Compacta 4 valores int2 por byte via bitwise OR
4. interleave_weight_int8(compressed_weight, nbits=2)
   → Reinterpreta como int32, reordena bits dentro de int32
   → shift pattern: [0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30]
5. reshape para (N, K//4)
```

### Função `B_global_16x32_to_shared_load_16x32_layout(i, j)`

Mapeamento para o layout de memória compartilhada WMMA:
```python
thread_id = i * 2 + j // 16
row = (thread_id // 16) * 8 + (thread_id % 8)
col = (j % 16) + 16 * ((thread_id % 16) // 8)
```

**Propósito:** Otimiza o acesso na shared memory para instruções `wmma::load_matrix_sync`, eliminando bank conflicts.

---

## Módulo 8: `gpu/convert_checkpoint.py` 🟢 CONFIRMADO

**Papel:** Conversão de checkpoint PyTorch unificado para formatos int2 e fp16 separados.

### Algoritmos de quantização de pesos

```python
# Quantização ternária para int2 (BitNet)
def quant_weight_int8(weight):
    s = 1.0 / weight.abs().mean().clamp_(min=1e-5)  # escala per-tensor via absmax médio
    new_weight = (weight * s).round().clamp(-1, 1).to(torch.int8)
    new_scale = (1.0 / s).to(torch.bfloat16)
    return new_weight, new_scale  # {-1, 0, +1} + escala

# Quantização simulada fp16 (para prefill)
def quant_weight_fp16(weight):
    s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    new_weight = (weight * s).round().clamp(-1, 1) / s
    return new_weight  # ternário em fp16
```

### Mapeamento de tensores

| Chave no checkpoint | Tratamento |
|--------------------|-----------|
| `*.wqkv.weight` | Divide em wq/wk/wv, quantiza separadamente, concatena; scale = [wa, wb, wc, zero] |
| `*.w13.weight` | Divide em w1/w3, quantiza separadamente; scale = [w1, w3, zero, zero] |
| `*.w2.weight`, `*.wo.weight` | Quantiza diretamente; scale = [s, zero, zero, zero] |
| Demais (embeddings, norms) | Copia sem alteração |

**Nota:** O zero padding nas scales (`zero = torch.zeros(1).to(torch.bfloat16)`) mantém tamanho fixo de 4 floats para todos os tensores — simplifica o kernel CUDA.

---

## Módulo 9: `gpu/convert_safetensors.py` 🟢 CONFIRMADO

**Papel:** Converte modelos safetensors (formato HuggingFace) para o formato interno `.pt`.

### Mapeamento de tensores HF → interno

| HF Key | Interno |
|--------|---------|
| `model.layers.{i}.self_attn.{q,k,v}_proj.weight` | `layers.{i}.attention.wqkv.weight` (concatenado) |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers.{i}.attention.wo.weight` |
| `model.layers.{i}.mlp.{gate,up}_proj.weight` | `layers.{i}.feed_forward.w13.weight` (concatenado) |
| `model.layers.{i}.mlp.down_proj.weight` | `layers.{i}.feed_forward.w2.weight` |
| `model.embed_tokens.weight` | `tok_embeddings.weight` e `output.weight` (compartilhados) |
| `model.norm.weight` | `norm.weight` |

**Inversão RoPE em Q e K:** Aplica `invert_convert_q/k` via einops para desfazer permutação do rotary embedding no formato HuggingFace.

---

## Módulo 10: `gpu/sample_utils.py` 🟢 CONFIRMADO

**Papel:** Nucleus sampling (top-p).

### Algoritmo top-p

```python
@torch.compile
def top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, descending=True)  # ordena decrescente
    probs_sum = torch.cumsum(probs_sort)                         # soma acumulada
    mask = probs_sum - probs_sort > p                            # tokens além do threshold
    probs_sort[mask] = 0.0                                       # zera tokens excluídos
    next_token = torch.multinomial(probs_sort, num_samples=1)   # amostra
    next_token = torch.gather(probs_idx, -1, next_token)         # mapeia de volta ao índice real
    return next_token
```

**Decorado com `@torch.compile`** para JIT compilation via Inductor.

---

## Módulo 11: `gpu/stats.py` 🟢 CONFIRMADO

**Papel:** Medição de performance por fase de geração (prefill vs decode).

### Classes

- **`PhaseStats`**: `name`, `tokens`, `time` → calcula `tokens/time` (TPS)
- **`Stats`**: lista de fases; `phase(name)` inicia nova fase e termina a anterior

---

## Módulo 12: `src/ggml-bitnet-lut.cpp` 🟢 CONFIRMADO

**Papel:** Implementação dos kernels LUT para CPU (TL1=ARM64, TL2=x86_64).

### Funções expostas (via `ggml-bitnet.h`)

| Função | Plataforma | Descrição |
|--------|-----------|-----------|
| `ggml_bitnet_init()` | TL1/TL2 | Aloca pool de `bitnet_tensor_extra[8192]` |
| `ggml_bitnet_free()` | TL1/TL2 | Libera pool |
| `ggml_bitnet_can_mul_mat()` | TL1 | Verifica se src1.ne[1]<=1 (batch 1) |
| `ggml_bitnet_can_mul_mat()` | TL2 | Sem restrição de batch |
| `ggml_bitnet_mul_mat_get_wsize()` | TL1 | `ne10*ne11*15 + ne11*2*sizeof(float)` + align 64 |
| `ggml_bitnet_mul_mat_get_wsize()` | TL2 | `ne10*ne11*11 + ne11*4*sizeof(float)` + align 64 |
| `ggml_bitnet_get_type_bits()` | TL1 | TL1→2bits, Q4_0→4bits |
| `ggml_bitnet_get_type_bits()` | TL2 | TL2→2bits, Q4_0→4bits |

**Diferença TL1 vs TL2:** TL1 (ARM) usa 15 bytes de workspace por entrada (LUT de ternário 3-value); TL2 (x86) usa 11 bytes.

### Pool estático

```cpp
#define GGML_BITNET_MAX_NODES 8192
static bool initialized = false;
static bitnet_tensor_extra * bitnet_tensor_extras = nullptr;
static size_t bitnet_tensor_extras_index = 0;
```

---

## Módulo 13: `src/ggml-bitnet-mad.cpp` 🟢 CONFIRMADO

**Papel:** Kernel MAD (Multiply-Add) para formato I2_S — implementação SIMD da multiplicação de matrizes ternárias.

### `QK_I2_S` — bloco de quantização

| Arquitetura | QK_I2_S |
|-------------|---------|
| x86 (AVX/SSE) | 128 |
| ARM NEON | 64 |

### Algoritmo `quantize_i2_s` (float → I2_S)

```
1. Encontrar max absoluto de todos os elementos → i2_scale
2. Para cada elemento:
   if |x| < 1e-6 → q8[i] = 1  (zero)
   else if x * scale > 0 → q8[i] = 2  (+1)
   else → q8[i] = 0  (-1)
3. Empacotar 4 valores por byte (layout dependente de arquitetura)
4. Armazenar scale (float) após os dados quantizados
```

**Mapeamento de valores:** 0→-1, 1→0, 2→+1

### Algoritmo `ggml_vec_dot_i2_i8_s_1x1` (produto escalar AVX2)

Loop interno com 128 elementos por iteração:
```
Para cada bloco de 32 grupos:
  carregar 256 bits de pesos packed (xq8_3)
  deslocar e mascarar para extrair 4 sub-grupos de 2 bits
  carregar 4 × 256 bits de ativações int8 (yq8_0..3)
  _mm256_maddubs_epi16: multiply-add unsigned×signed 8bit → 16bit
  acumular em int32 via _mm256_madd_epi16
soma horizontal → s[row]
```

---

## Módulo 14: `utils/codegen_tl1.py` 🟢 CONFIRMADO

**Papel:** Gerador de código C++ para kernels TL1 (ARM64 NEON).

**Estratégia:** Geração de código especializado com parâmetros de tiling hardcoded para cada modelo/arquitetura, eliminando overhead de runtime parameterization.

O código gerado inclui:
- Funções `per_tensor_quant` (NEON/AVX2 otimizadas)
- `Transpose_8_8` (NEON int16x8)
- Template `act_k` para unrolling do loop interno de ativação
- Funções de preprocessamento e QGEMM para cada combinação (BM, BK, bm)

---

## Módulo 15: `utils/codegen_tl2.py` 🟢 CONFIRMADO

**Papel:** Gerador de código C++ para kernels TL2 (x86_64 AVX2/AVX512).

Estrutura similar ao TL1, mas com:
- Intrínsecas AVX2 (`__m256i`, `_mm256_*`)
- Função `Transpose_8_8` via `_mm256_merge_epi32/64/si128`
- `BK2 = 32` para bloco interno de processamento x86

---

## Resumo de Algoritmos Críticos

### 1. Quantização Ternária de Pesos (BitNet 1.58-bit)

```
scale_per_tensor = 1 / mean(|W|)
W_q = round(W * scale).clamp(-1, 1)  → {-1, 0, +1}
```

Proporciona ~1.58 bits teóricos por parâmetro (log₂(3) ≈ 1.585).

### 2. Quantização de Ativações (absmax per-token)

```
scale_per_token = 127 / max(|x|, dim=-1)
x_q = round(x * scale).clamp(-128, 127)  → int8
```

### 3. Inferência dual-model (prefill/decode)

- **Prefill**: modelo fp16 com ternário simulado → melhor acurácia na entrada
- **Decode**: modelo int2 via kernel CUDA → máxima velocidade no loop token-a-token

### 4. LUT GEMM (TL1/TL2)

Em vez de multiplicações, usa lookup tables pré-computadas para os 3 valores possíveis dos pesos, tornando a operação basicamente uma operação de endereçamento de memória.

---

## Dependências entre Módulos

```
run_inference.py ──────────────────→ build/bin/llama-cli (externo)
run_inference_server.py ───────────→ build/bin/llama-server (externo)
setup_env.py → gen_code() ─────────→ codegen_tl1.py / codegen_tl2.py
setup_env.py → prepare_model() ───→ convert-hf-to-gguf-bitnet.py
setup_env.py → compile() ─────────→ cmake + src/ggml-bitnet-*.cpp

gpu/generate.py → gpu/model.py
gpu/generate.py → gpu/tokenizer.py
gpu/generate.py → gpu/sample_utils.py
gpu/generate.py → gpu/stats.py
gpu/convert_checkpoint.py → gpu/model.py (ModelArgs)
gpu/convert_checkpoint.py → gpu/pack_weight.py
```

---

## Lacunas identificadas 🔴

1. **`gpu/bitnet_kernels/`**: Código-fonte do kernel CUDA `bitlinear_int8xint2` não está no repositório (apenas `.so` binário referenciado). Impossível analisar a implementação interna do kernel GPU.
2. **`utils/convert.py`**: Não analisado nesta sessão (dependência de `convert-hf-to-gguf-bitnet.py`).
3. **`CMakeLists.txt`**: Não analisado — flags de compilação adicionais podem existir.
4. **Kernels pré-tunados** (`preset_kernels/`): Arquivos `.h` gerados com parâmetros hardcoded, não analisados em detalhe.
