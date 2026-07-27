# Dicionário de Dados — BitNet

> Gerado pelo Reversa Archaeologist | 2026-05-03

---

## Estruturas de Dados Python

### `ModelArgs` (gpu/model.py)

Configuração do modelo Transformer BitNet para GPU.

| Campo | Tipo | Default | Obrigatório | Descrição |
|-------|------|---------|-------------|-----------|
| `dim` | int | 2560 | sim | Dimensão do embedding (modelo 2B) |
| `n_layers` | int | 30 | sim | Número de camadas Transformer |
| `n_heads` | int | 20 | sim | Cabeças de multi-head attention |
| `n_kv_heads` | int | 5 | sim | Cabeças de KV (GQA: ratio 4:1) |
| `vocab_size` | int | 128256 | sim | Tamanho do vocabulário (Llama 3) |
| `ffn_dim` | int | 6912 | sim | Dimensão interna da FFN |
| `norm_eps` | float | 1e-5 | sim | Epsilon para RMSNorm (estabilidade numérica) |
| `rope_theta` | float | 500000.0 | sim | Frequência base do Rotary Position Embedding |
| `use_kernel` | bool | False | sim | True → BitLinearKernel (int2); False → BitLinear (fp16) |

---

### `GenArgs` (gpu/generate.py)

Parâmetros de geração de texto.

| Campo | Tipo | Default | Obrigatório | Descrição |
|-------|------|---------|-------------|-----------|
| `gen_length` | int | 32 | sim | Número de tokens a gerar |
| `gen_bsz` | int | 1 | sim | Batch size de geração |
| `prompt_length` | int | 64 | sim | Comprimento fixo do prompt (pad/truncate) |
| `use_sampling` | bool | False | sim | Habilita top-p sampling vs argmax |
| `temperature` | float | 0.8 | sim | Temperatura de sampling |
| `top_p` | float | 0.9 | sim | Limiar nucleus sampling |

---

### `ModelArgs` (gpu/convert_safetensors.py)

Configuração para conversão de checkpoint safetensors.

| Campo | Tipo | Default | Obrigatório | Descrição |
|-------|------|---------|-------------|-----------|
| `block_size` | int | 4096 | não | Tamanho máximo de contexto |
| `vocab_size` | int | 32000 | não | Vocabulário (sobrescrito por config) |
| `n_layer` | int | 32 | não | Camadas |
| `n_head` | int | 32 | não | Cabeças de atenção |
| `dim` | int | 4096 | não | Dimensão do modelo |
| `intermediate_size` | int | None | não | Auto-calculado: `4*dim` → SwiGLU scaling |
| `n_local_heads` | int | -1 | não | GQA heads (-1 = igual a n_head) |
| `head_dim` | int | 64 | não | Auto-calculado: `dim // n_head` |
| `rope_base` | float | 10000 | não | Theta base do RoPE |
| `norm_eps` | float | 1e-5 | não | Epsilon para normas |

**Configurações por modelo:**

| Nome | n_layer | n_head | dim | vocab_size | n_local_heads | ffn_dim |
|------|---------|--------|-----|------------|---------------|---------|
| "2B" | 30 | 20 | 2560 | 128256 | 5 | 6912 |

---

### `Message` (gpu/tokenizer.py)

Mensagem de diálogo no formato TypedDict.

| Campo | Tipo | Valores | Descrição |
|-------|------|---------|-----------|
| `role` | `Role` | "system"\|"user"\|"assistant" | Papel do falante |
| `content` | str | qualquer | Conteúdo da mensagem |

---

### `PhaseStats` (gpu/stats.py)

Estatísticas de uma fase de geração.

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `name` | str | sim | Nome da fase ("prefill" ou "decode") |
| `tokens` | int | sim | Tokens gerados na fase |
| `time` | float | sim | Tempo em segundos |

---

## Estruturas de Dados C/C++

### `bitnet_tensor_extra` (include/ggml-bitnet.h)

Metadados extras para tensores quantizados BitNet.

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `lut_scales_size` | int | Tamanho do array de escalas da LUT |
| `BK` | int | Block size K (dimensão interna GEMM) |
| `n_tile_num` | int | Número de tiles no kernel |
| `qweights` | `uint8_t*` | Ponteiro para pesos quantizados (aligned) |
| `scales` | `bitnet_float_type*` | Ponteiro para escalas (float32 em x86, float32_t em ARM) |

**Notas:**
- `bitnet_float_type` = `float32_t` em ARM NEON, `float` em outros
- Pool estático: `bitnet_tensor_extras[GGML_BITNET_MAX_NODES]` (8192 entradas)
- Alocação alinhada em 64 bytes via `posix_memalign`

---

## Parâmetros de Configuração (GEMM)

### `gemm-config.h` — Parâmetros de bloco SIMD

| Define | Plataforma | Modo | Valor |
|--------|-----------|------|-------|
| `ROW_BLOCK_SIZE` | x86 AVX | ACT_PARALLEL | 4 |
| `COL_BLOCK_SIZE` | x86 AVX | ACT_PARALLEL | 128 |
| `PARALLEL_SIZE` | x86 AVX | ACT_PARALLEL | 4 |
| `ROW_BLOCK_SIZE` | ARM NEON+DOTPROD | ACT_PARALLEL | 8 |
| `COL_BLOCK_SIZE` | ARM NEON+DOTPROD | ACT_PARALLEL | 256 |
| `PARALLEL_SIZE` | ARM NEON+DOTPROD | ACT_PARALLEL | 8 |
| `ROW_BLOCK_SIZE` | ARM NEON (sem DOTPROD) | ACT_PARALLEL | 8 |
| `COL_BLOCK_SIZE` | ARM NEON (sem DOTPROD) | ACT_PARALLEL | 256 |
| `PARALLEL_SIZE` | ARM NEON (sem DOTPROD) | ACT_PARALLEL | 4 |

**Nota:** `ACT_PARALLEL` está sempre definido (otimização para ativações paralelas).

---

## Formatos de Arquivo

### GGUF (`.gguf`)

Formato binário do llama.cpp para armazenar modelos quantizados.

| Tipo | Sufixo | Arquitetura | Descrição |
|------|--------|-------------|-----------|
| I2_S | `ggml-model-i2_s.gguf` | arm64 + x86_64 | 2-bit packed, escala por tensor |
| TL1 | `ggml-model-tl1.gguf` | arm64 | LUT kernel ARM |
| TL2 | `ggml-model-tl2.gguf` | x86_64 | LUT kernel x86 |
| F32 | `ggml-model-f32.gguf` | intermediário | Float32, usado antes de quantizar |

### Checkpoints PyTorch (GPU)

| Arquivo | Conteúdo | Formato |
|---------|---------|---------|
| `model_state.pt` | Pesos originais fp16/bf16 | `torch.save` dict |
| `model_state_fp16.pt` | Pesos ternários simulados em bf16 | Gerado por `convert_checkpoint.py` |
| `model_state_int2.pt` | Pesos int2 comprimidos + scales | Gerado por `convert_checkpoint.py` |

### Nomes de chaves nos checkpoints

| Chave | Tensor | Shape aproximado (modelo 2B) |
|-------|--------|------------------------------|
| `layers.{i}.attention.wqkv.weight` | Q+K+V concatenados | (2560+512+512, 2560) |
| `layers.{i}.attention.wqkv.weight_scale` | Scales wq/wk/wv/zero | (4,) bf16 |
| `layers.{i}.attention.wo.weight` | Projeção de saída | (2560, 2560) |
| `layers.{i}.feed_forward.w13.weight` | Gate+Up concatenados | (2×6912, 2560) |
| `layers.{i}.feed_forward.w13.weight_scale` | Scales w1/w3/zero/zero | (4,) bf16 |
| `layers.{i}.feed_forward.w2.weight` | Down projection | (2560, 6912) |
| `tok_embeddings.weight` | Embeddings | (128256, 2560) |
| `output.weight` | LM head (compartilhado) | (128256, 2560) |
| `norm.weight` | RMSNorm final | (2560,) |
| `layers.{i}.attention_norm.weight` | Norm pré-atenção | (2560,) |
| `layers.{i}.ffn_norm.weight` | Norm pré-FFN | (2560,) |
| `layers.{i}.attention.attn_sub_norm.weight` | Sub-norm pós-atenção | (2560,) |
| `layers.{i}.feed_forward.ffn_sub_norm.weight` | Sub-norm interna da FFN | (6912,) |

---

## Constantes e Enums

### Tipos de quantização suportados

| Tipo | Plataforma | Método | Descrição |
|------|-----------|--------|-----------|
| `i2_s` | arm64 + x86_64 | MAD (SIMD) | 2-bit signed, escala por tensor |
| `tl1` | arm64 only | LUT (NEON) | Ternary LUT, ARM otimizado |
| `tl2` | x86_64 only | LUT (AVX2) | Ternary LUT, x86 otimizado |

### Mapeamento de arquitetura

| `platform.machine()` | Alias interno |
|---------------------|---------------|
| AMD64, x86, x86_64 | x86_64 |
| aarch64, arm64, ARM64 | arm64 |

### Tokens especiais (Tiktoken/Llama 3)

| Token | Índice relativo | Uso |
|-------|----------------|-----|
| `<\|begin_of_text\|>` | +0 | BOS — início de sequência |
| `<\|end_of_text\|>` | +1 | EOS — fim de sequência |
| `<\|start_header_id\|>` | +6 | Início de cabeçalho de role |
| `<\|end_header_id\|>` | +7 | Fim de cabeçalho de role |
| `<\|eot_id\|>` | +9 | End-of-turn (stop token de geração) |
