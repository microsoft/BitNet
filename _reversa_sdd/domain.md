# Domínio — BitNet

> Gerado pelo Reversa Detective | 2026-05-03

---

## Glossário de Domínio

| Termo | Definição | Confiança |
|-------|-----------|-----------|
| **BitNet** | Família de LLMs com pesos quantizados em 1.58 bits (ternário: {-1, 0, +1}) desenvolvida pela Microsoft | 🟢 CONFIRMADO |
| **Quantização ternária** | Representação de pesos com apenas 3 valores: -1 (negativo), 0 (zero), +1 (positivo) — requer apenas ~1.585 bits por parâmetro (log₂(3)) | 🟢 CONFIRMADO |
| **I2_S** | Formato de quantização 2-bit signed: armazena 4 valores ternários por byte, com escala por tensor ao final. Suportado em ARM64 e x86_64 | 🟢 CONFIRMADO |
| **TL1** | Formato TernaryLUT 1 — kernel LUT (Look-Up Table) para ARM64 NEON. Mais eficiente que I2_S em ARM64 | 🟢 CONFIRMADO |
| **TL2** | Formato TernaryLUT 2 — kernel LUT para x86_64 AVX2. Mais eficiente que I2_S em x86 | 🟢 CONFIRMADO |
| **GGUF** | Formato de arquivo binário do llama.cpp para modelos quantizados. Armazena pesos, metadados e configuração | 🟢 CONFIRMADO |
| **GEMM** | General Matrix Multiplication — operação central na inferência de LLMs | 🟢 CONFIRMADO |
| **Escala per-tensor** | Fator de escala único calculado sobre o tensor inteiro: `1 / mean(|W|)` | 🟢 CONFIRMADO |
| **Escala per-token** | Fator de escala calculado por linha de ativação: `127 / max(|x|)` — diferente da escala de peso | 🟢 CONFIRMADO |
| **GQA** | Grouped Query Attention — mecanismo de atenção onde múltiplas cabeças de query compartilham uma cabeça de KV. No BitNet 2B: ratio 4:1 (20 query heads / 5 KV heads) | 🟢 CONFIRMADO |
| **RoPE** | Rotary Position Embedding — codificação de posição multiplicativa. BitNet 2B usa theta=500000 para suporte a contextos longos | 🟢 CONFIRMADO |
| **CUDA Graphs** | Mecanismo do PyTorch/CUDA que captura sequências de kernel launches para reprodução zero-overhead. Crítico no loop de decode | 🟢 CONFIRMADO |
| **Prefill** | Fase de processamento do prompt de entrada. Caracterizada por alto paralelismo; usa modelo fp16 para máxima acurácia | 🟢 CONFIRMADO |
| **Decode** | Fase de geração token-a-token. Caracterizada por batch pequeno e KV cache crescente; usa modelo int2 para máxima velocidade | 🟢 CONFIRMADO |
| **WMMA** | Warp Matrix Multiply Accumulate — instrução CUDA para multiplicação matricial em nível de warp. Exige layout específico de memória | 🟢 CONFIRMADO |
| **Sub-norm** | Normalização aplicada internamente em camadas de atenção e FFN do BitNet — diferencial arquitetural vs. Transformer padrão | 🟢 CONFIRMADO |
| **BPE** | Byte Pair Encoding — algoritmo de tokenização usado pelo Tiktoken (GPT-4/Llama 3) | 🟢 CONFIRMADO |
| **EOT** | End of Turn (`<\|eot_id\|>`) — token especial Llama 3 que sinaliza fim de turno em diálogo; funciona como stop token de geração | 🟢 CONFIRMADO |
| **Kernel codegen** | Geração dinâmica de código C++ especializado para cada combinação modelo/plataforma. Elimina overhead de parametrização em runtime | 🟢 CONFIRMADO |
| **Preset kernels** | Parâmetros GEMM pré-tunados empiricamente para modelos conhecidos (bitnet_b1_58-3B, Llama3-8B, bitnet-large) | 🟡 INFERIDO |
| **Embedding quantization** | Quantização opcional das embeddings de tokens (default: F32). Flag `--quant-embd` habilita; impacto em qualidade não documentado no código | 🟡 INFERIDO |

---

## Regras de Negócio Implícitas

### RN-001: Tensores protegidos da quantização I2 🟢 CONFIRMADO

Três categorias de tensores **nunca** são quantizados para formato ternário I2_S, TL1 ou TL2:

1. **Normalizations** (`*_norm.weight`, `norm.weight`) → sempre F32
2. **LM Head** (`lm_head.weight`) → sempre F32/F16
3. **Token Embeddings** (`embed_tokens.weight`) → F32 por default; F16 com `--quant-embd`

**Evidência no código:**
```python
# convert-hf-to-gguf-bitnet.py:795-797
suit_i2 = True
if name.endswith('lm_head.weight') or name.endswith('norm.weight') or name.endswith('embed_tokens.weight'):
    suit_i2 = False
```

**Razão implícita:** Normalizations e embeddings são camadas sensíveis à precisão numérica; quantizá-las degradaria significativamente a qualidade do modelo. 🟡 INFERIDO

---

### RN-002: Embeddings quantizadas para F16 apenas no modo TL (LUT) 🟢 CONFIRMADO

Quando o tipo de quantização é TL1 ou TL2, as embeddings são sempre quantizadas para F16 (flag `--quant-embd` passada implicitamente). Para I2_S, a quantização de embeddings é opt-in.

**Evidência:**
```python
# setup_env.py:129-130
if quant_type.startswith("tl"):
    run_command([..., "--quant-embd"], ...)  # sempre passa
```

---

### RN-003: Restrição de arquitetura em formatos de quantização 🟢 CONFIRMADO

Cada arquitetura de CPU só pode usar um subconjunto dos formatos:
- ARM64: `i2_s` ou `tl1` (não `tl2`)
- x86_64: `i2_s` ou `tl2` (não `tl1`)

**Razão:** TL1 usa intrínsecas NEON exclusivas do ARM; TL2 usa intrínsecas AVX2 exclusivas do x86.

---

### RN-004: Alinhamento obrigatório `nrow % 4 == 0` para I2_S sem ACT_PARALLEL 🟢 CONFIRMADO

O kernel de quantização `quantize_i2_s` em modo não-paralelo (que empacota 4 linhas por byte) exige que o número de linhas seja múltiplo de 4.

**Evidência:**
```cpp
// ggml-bitnet-mad.cpp:98
assert((nrow % 4) == 0 && "quantize_i2_s_1x4 requires nrow % 4 == 0");
```

---

### RN-005: GPU requer TWO modelos distintos para inferência 🟢 CONFIRMADO

O pipeline GPU carrega e mantém dois modelos Transformer em memória simultaneamente:
- `model_state_fp16.pt` → prefill (melhor qualidade, BF16)
- `model_state_int2.pt` → decode (máxima velocidade, INT2)

**Implicação operacional:** O uso de memória GPU é dobrado em relação a uma abordagem single-model. Para um modelo 2B, os dois modelos juntos ocupam mais memória do que um único modelo FP16.

---

### RN-006: Prompts são truncados/padded para comprimento fixo em GPU 🟢 CONFIRMADO

Para reutilização do CUDA Graph (que captura operações com shapes fixas), prompts são padded para `prompt_length` (default: 64 tokens). Prompts mais longos que `prompt_length` resultam em comportamento indefinido — os tokens extras são descartados silenciosamente.

**Evidência:**
```python
# generate.py:238
prompts = [prompt + [1] * (self.gen_args.prompt_length - len(prompt)) for prompt in prompts]
```

**Risco:** Usuários com prompts longos podem receber outputs incorretos sem mensagem de erro. 🔴 LACUNA — não há validação do comprimento do prompt

---

### RN-007: Clang é compilador obrigatório (histórico de decisão) 🟢 CONFIRMADO

O projeto força o uso de Clang/Clang++ via CMake:
```python
# setup_env.py:214
run_command(["cmake", ..., "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"])
```

Suporte a GCC foi adicionado posteriormente (commit `141ddfd`) mas com caveats (`-fpermissive`). Android/ARM64 também força Clang (commit `c9e752c`).

**Razão:** Intrínsecas SIMD (AVX2, NEON) têm comportamento mais previsível com Clang; GCC tem incompatibilidades com algumas extensões usadas nos kernels gerados.

---

### RN-008: GPU offload desabilitado (-ngl 0 hardcoded) 🟢 CONFIRMADO

O flag `-ngl 0` está hardcoded nos scripts de inferência CPU, desabilitando offload de camadas para GPU via llama.cpp.

**Razão:** O projeto tem uma pipeline GPU separada (`gpu/`). O llama.cpp é usado apenas para CPU. Misturar os dois criaria conflito. 🟡 INFERIDO

---

### RN-009: Batch size 1 hardcoded para inferência CPU 🟢 CONFIRMADO

`-b 1` está hardcoded em `run_inference.py`. A inferência CPU é otimizada para batch=1 (GEMV, não GEMM).

**Evidência no comentário do C++:**
```cpp
// ggml-bitnet-lut.cpp: TL1 só suporta src1->ne[1] <= 1
if (src1->ne[1] <= 1) { return true; }  // can_mul_mat restritivo
```

---

### RN-010: Ternário é encodado como {0, 1, 2} internamente 🟢 CONFIRMADO

Os valores ternários {-1, 0, +1} são armazenados como {0, 1, 2} internamente:
- 0 → -1 (negativo)
- 1 → 0 (zero)
- 2 → +1 (positivo)

Para GPU, o shift é `+2` no `pack_weight.py`:
```python
weight = weight + 2  # {-1, 0, +1} → {1, 2, 3} (evita 0 para LUT)
```

Para TL1/TL2, o shift em preprocess:
```python
weight = weight + 4  # offset para uint8 não-negativo
```

---

### RN-011: Vulnerabilidade de deserialização insegura foi conhecida e tardiamente corrigida 🟢 CONFIRMADO

`torch.load()` sem `weights_only=True` permite execução de código arbitrário via payloads maliciosos em arquivos `.pt`. Esta vulnerabilidade (CWE-502) existiu no pipeline GPU desde sua introdução (maio 2025) e foi corrigida apenas em março 2026 (PR #421, commit `eb60fc3`).

O fix foi aplicado apenas em `gpu/generate.py` e `gpu/convert_checkpoint.py`. Os scripts em `utils/` já usavam `weights_only=True` corretamente.

**Impacto:** Qualquer usuário que carregasse um checkpoint `.pt` malicioso na pipeline GPU teria código executado em sua máquina.

---

### RN-012: Regra de codificação base-3 para TL1/TL2 🟢 CONFIRMADO

Dois valores ternários consecutivos são comprimidos em um byte uint8 via codificação base-3:
```python
# convert-hf-to-gguf-bitnet.py
hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)  # × 3
lo_weight = np.split(weight, 2, axis=1)[1]
weight = hi_weight + lo_weight  # base-3: hi*3 + lo
weight = weight + 4             # offset para uint8
```

**Valores possíveis:** 9 combinações de {0,1,2}×{0,1,2} → valores 0..8, +4 offset → 4..12, armazenado em uint8.

---

### RN-013: Escala de peso usa absmax médio, não absmax máximo 🟢 CONFIRMADO

BitNet usa **absmax médio** para quantização de pesos:
```python
s = 1 / weight.abs().mean()  # médio — diferente do usual
```

Em contraste, ativações usam **absmax máximo**:
```python
s = 127 / input.abs().max()  # máximo — padrão de quantização de ativações
```

**Razão:** Usar a média produz quantização de melhor qualidade em distribuições Laplacianas (que os pesos de LLMs tipicamente seguem). O máximo seria afetado por outliers. 🟡 INFERIDO

---

### RN-014: Escape hatch para debugging de CUDA Graphs 🟢 CONFIRMADO

A variável de ambiente `NO_CUDA_GRAPHS` desabilita CUDA Graphs quando presente:
```python
# generate.py:343
tokens, use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ, ...
```

**Razão:** CUDA Graphs tornam o debugging difícil (stacks de erro não informativas). A variável é um mecanismo de fallback para desenvolvimento. 🟡 INFERIDO

---

### RN-015: `capture_error_mode="thread_local"` é workaround para crash em PyTorch ≥2.1 🟢 CONFIRMADO

```python
# generate.py:136-139
if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
    # In PyTorch 2.1+ and nightlies from late Aug 2023,
    # we can do this to maybe avoid watchdog-related crashes
    recording_kwargs["capture_error_mode"] = "thread_local"
```

**Natureza:** Workaround para um bug do watchdog CUDA em versões específicas do PyTorch. O código verifica dinamicamente a presença do parâmetro antes de usá-lo.

---

### RN-016: Identificação do modelo por fingerprint do tokenizador 🟢 CONFIRMADO

A função `get_vocab_base_pre` em `convert-hf-to-gguf-bitnet.py` identifica o pré-tokenizador pelo hash de tokens codificados, não pelo nome do modelo. Isso garante que o tipo de tokenizador correto seja gravado no GGUF:

```python
# NOTE: this function is generated by convert-hf-to-gguf-update.py
#       do not modify it manually!
# ref:  https://github.com/ggerganov/llama.cpp/pull/6920
```

**Regra:** Nunca editar os hashes manualmente — são gerados por script. Editar manualmente quebraria a identificação do tokenizador silenciosamente.

---

## Regras de Validação (Assertions)

| Regra | Localização | Condição | Consequência se violada |
|-------|-------------|----------|------------------------|
| Divisibilidade de dimensões | `gpu/model.py:204` | `dim % n_heads == 0` | AssertionError em construção do modelo |
| GQA válido | `gpu/model.py:211` | `n_heads % n_kv_heads == 0` | AssertionError em construção do modelo |
| Vocabulário positivo | `gpu/model.py:249` | `vocab_size > 0` | AssertionError em construção do modelo |
| Cache suficiente | `gpu/model.py:364` | `cache.shape[1] >= length` | AssertionError em `cache_prefix` |
| Tokenizer existe | `gpu/tokenizer.py:52` | `os.path.isfile(model_path)` | AssertionError com path |
| Input é string | `gpu/tokenizer.py:125` | `type(s) is str` | AssertionError |
| Nomes de tokenizador imutáveis | `convert-hf-to-gguf.py:307-309` | hash correto | NotImplementedError com instrução de update |
| Alinhamento de linhas I2_S | `ggml-bitnet-mad.cpp:98` | `nrow % 4 == 0` | Crash com assert (modo 1x4) |

---

## TODOs e FIXMEs com Impacto Funcional

| Arquivo | Linha | Tipo | Texto | Risco |
|---------|-------|------|-------|-------|
| `include/ggml-bitnet.h` | 30 | TODO | `add customized block types Q2_0/Q3_0` | Tipos customizados de bloco de quantização ainda não implementados |
| `convert-hf-to-gguf-bitnet.py` | 187 | TODO | `Why cant we use these float16 as-is?` | Conversão F16→F32 pode ser desnecessária, impactando performance de conversão |
| `convert.py` | 432 | FIXME | `Verify that added tokens here _cannot_ overlap with the main vocab` | Risco de colisão de IDs de tokens especiais com vocabulário base |
| `utils/generate-dummy-bitnet-model.py` | 259 | TODO | Mesma questão F16 | Mesmo risco de performance |

---

## Inferências sobre Decisões de Design Não Documentadas

### Por que `squared_relu` em vez de `SiLU`? 🟡 INFERIDO
A FFN do BitNet usa `relu(x)² × gate` em vez do `SiLU(x) × gate` do LLaMA/Mistral. O código-fonte não documenta o motivo. Provável razão: `squared_relu` é mais compatível com quantização ternária pois tem um ponto zero preciso, enquanto `SiLU` nunca é exatamente zero.

### Por que dois modelos separados para prefill/decode? 🟡 INFERIDO
O design dual-model (fp16 para prefill, int2 para decode) foi introduzido no commit inicial do branch GPU (`154c92b`). A separação sugere que a acurácia do prefill é mais crítica que a velocidade (processa o prompt apenas uma vez), enquanto o decode repete milhares de vezes justificando a máxima otimização.

### Por que BM/BK/bm são parâmetros por modelo? 🟡 INFERIDO
Os tiling parameters do GEMM afetam diretamente a utilização de cache L1/L2. Valores ótimos dependem da dimensão do modelo (dim, ffn_dim). Os valores hardcoded por modelo foram provavelmente obtidos via tuning automático (existe `utils/tune_gemm_config.py`) e depois congelados como presets.
