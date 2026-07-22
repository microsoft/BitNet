# Sessão 2025-06-05 — Tropical Attention: Dispatch + llama.cpp Integration

## Objetivo

Plugar `bitnet_op_tropical_attn` no builder do llama.cpp, substituindo
`ggml_flash_attn_ext` durante inferência real, controlado por env var.

---

## Arquivos Modificados

### `src/ggml-bitnet-dispatch.cpp`

- **`tropical_callback`** — atualizado para suportar tensores 3D multi-head com GQA:
  - Loop sobre `n_head` query heads
  - Mapeamento GQA: `kv_h = h / (n_head / n_head_kv)`
  - K quantizado uma vez por head, Q quantizado por token (escala mais precisa)
  - Layout de memória: `head-major, token-minor, dim-innermost` após cast F32

### `src/ggml-bitnet-tropical.cpp`

- **`tropical_attn_topk`** — guard contra `K_top > n_keys`:
  ```c
  const int K_actual = (K_top < n_keys) ? K_top : n_keys;
  if (K_actual <= 0) return;
  std::partial_sort(idx, idx + K_actual, idx + n_keys, ...)
  ```

- **`tropical_attention`** — usa `K_actual = min(K_top, n_keys)` em todos os loops:
  malloc, softmax loop, weighted-sum loop — todos com `K_actual` não `K_top`

### `3rdparty/llama.cpp/src/llama.cpp`

Duas inserções cirúrgicas no submodule (deliberate patch):

**1. Include condicional (após linha 29):**
```cpp
#if defined(BITNET_L4_TROPICAL)
#  include "ggml-bitnet-dispatch.h"
#endif
```

**2. Branch tropical em `llm_build_kqv` (antes de `if (cparams.flash_attn)`):**
```cpp
#if defined(BITNET_L4_TROPICAL)
    static const int bitnet_tropical_topk = []() {
        const char * e = getenv("BITNET_TROPICAL_TOPK");
        int v = e ? atoi(e) : 0;
        return (v > 0) ? v : 0;
    }();
    if (bitnet_tropical_topk > 0) {
        // kq_mask DEVE entrar no grafo para llama_set_inputs alocar seu buffer
        ggml_build_forward_expand(graph, kq_mask);

        struct ggml_tensor * v_t = ggml_view_3d(ctx, kv.v_l[il],
                n_embd_head_v, n_kv, n_head_kv,
                ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa),
                ggml_row_size(kv.v_l[il]->type, n_embd_head_v), 0);
        struct ggml_tensor * k_f32 = (k->type == GGML_TYPE_F32) ?
            k : ggml_cast(ctx, k, GGML_TYPE_F32);
        struct ggml_tensor * v_f32 = (v_t->type == GGML_TYPE_F32) ?
            v_t : ggml_cast(ctx, v_t, GGML_TYPE_F32);
        cur = bitnet_op_tropical_attn(ctx, q, k_f32, v_f32,
                                      bitnet_tropical_topk, kq_scale);
        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v * n_head, n_tokens);
    } else
#endif
    if (cparams.flash_attn) { ... }
```

---

## Bugs Corrigidos

### Bug 1: `std::partial_sort` UB — `K_top > n_keys`

- **Causa**: durante warmup llama.cpp processa 2 tokens (BOS+EOS), então n_kv=2.
  `std::partial_sort(idx, idx+32, idx+2)` → middle > last → undefined behavior → SIGSEGV
- **Fix**: `K_actual = min(K_top, n_keys)`, usar K_actual como middle

### Bug 2: Loops com `K_top` após preenchimento de apenas `K_actual` slots

- **Causa**: softmax loop e weighted-sum loop iteravam até K_top,
  mas top_idx/top_s tinham apenas K_actual entradas preenchidas
- **Fix**: malloc(K_actual), loops até K_actual

### Bug 3: `lctx.inp_KQ_mask->buffer == NULL` → SIGSEGV em `llama_set_inputs`

- **Causa**: na branch tropical, `kq_mask` não é operando de nenhuma op ggml,
  então o alocador de grafo (`ggml_backend_alloc_graph`) nunca aloca seu buffer.
  `llama_set_inputs` tenta `ggml_backend_buffer_is_host(inp_KQ_mask->buffer)` e
  dereferencia NULL (offset 0x50 = campo `buft` no struct).
- **Fix**: `ggml_build_forward_expand(graph, kq_mask)` força o tensor no grafo

---

## Resultado Final

```bash
BITNET_TROPICAL_TOPK=32 python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Hello" -n 20 -t 4
```

- ✅ Warmup: passa sem crash
- ✅ Prefill: 5.37 tok/s (2 tokens)  
- ✅ Decode: 5.21 tok/s (19 tokens)
- ⚠️ Qualidade: garbage (esperado — modelo não treinado com tropical attn)

---

## Modelo Testado

- `microsoft/BitNet-b1.58-2B-4T` via `hf download` (pré-convertido)
- Arquivo: `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`
- Arquitetura: 30 camadas, 20 heads (n_gqa=4, 5 KV heads), head_dim=128

---

## Notas de Arquitetura

### Layout de Memória da KV Cache para V

O tensor V no KV cache tem layout NÃO-contíguo com strides "invertidos":
- `ggml_view_3d(kv.v_l[il], d, n_kv, n_head_kv, nb1=n_embd_v_gqa*2, nb2=d*2)`
- nb2 < nb1: heads interleaved dentro de cada token
- Após `ggml_cast(→F32)`: output é contíguo com layout `[n_head_kv, n_kv, d]`
  (head-major), o que é exatamente o que `tropical_attention` espera para K e V

### Propagação de Defines via CMake PUBLIC

`BITNET_L4_TROPICAL` definido em `bitnet_math` como PUBLIC propaga via:
`bitnet_math → ggml (PUBLIC) → llama` — disponível ao compilar `llama.cpp`

### Env Var com Static Local

```cpp
static const int bitnet_tropical_topk = []() { ... }();
```
Inicializado uma vez por processo (thread-safe C++11). String
`BITNET_TROPICAL_TOPK` confirmada baked em `libllama.so` via `strings`.

---

## Próximos Passos

1. Treinar modelo com tropical attention (QAT) para validar qualidade real
2. Benchmark de throughput tropical vs. standard (mesma qualidade)
3. Ajuste fine-tuning da threshold K (atualmente 32, ótimo depende de d e n_ctx)
4. Integrar L5 HRR no mesmo padrão (adicionar `ggml_build_forward_expand(graph, kq_mask)`)
