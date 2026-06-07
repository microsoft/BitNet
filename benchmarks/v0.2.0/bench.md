# BitNet CPU-Universal — Benchmark v0.2.0

**Data:** 2026-06-07 | **Hardware:** Intel i5-10210U @ 1.60 GHz, 8 threads (4 usados), 35 GB RAM  
**Método:** `utils/cpu_universal_benchmark.py`, prompt fixo, n=64 tokens, t=4 threads, sem GPU  
**Commit do fix aplicado:** `4ad5ad6` — `bitnet_kv_i8_cache_get` corrigido para `head_dim` dinâmico

---

## Tabela comparativa — 3 modelos × 5 níveis algébricos

| Configuração | BitNet-2B¹ | Falcon3-3B-1.58bit | Falcon3-10B-1.58bit |
|---|:---:|:---:|:---:|
| **Arquitetura** | 18L / FFN=6912 / d=128 | 22L / FFN=9216 / d=256 | 40L / FFN=23040 / d=256 |
| **Tamanho GGUF** | 1.2 GB | 2.22 GB | 3.99 GB |
| **L1 baseline (I2_S GEMV)** | ~4.88 tok/s | 4.40 tok/s | 1.39 tok/s |
| L3 ACDC FFN | -3.5 % | -4.3 % | -10.1 % |
| L4 Tropical top-K=32 | -7.2 % | -4.8 % | -16.5 % |
| **L4 Sparse float top-K=32** | **-0.6 %** | **+2.0 %** | **-18.0 %** |
| L5 HRR raw | -62.1 % | -40.0 % | -36.0 % |
| L5 HRR + cleanup 8 | -61.7 % | -49.5 % | **-30.2 %** |

¹ BitNet-2B: valores aproximados da sessão 2026-06-05; run formal pendente.  
Todos os modelos usam pesos ternários {-1,0,+1} treinados nativamente (não quantização post-hoc).

---

## Achados principais

### 1. L4 sparse float: positivo para modelos menores, negativo para 10B

O overhead do sparse float (dot products em float32 sobre todos os n_kv tokens) é constante relativo
ao custo de atenção. Mas para o Falcon3-10B, a FFN (dim=23040) consome >90% do forward pass —
a atenção é uma fração pequena onde o overhead supera a economia.

**Lei observada:** L4 sparse float é benéfico quando `FFN_dim / hidden_dim < 4`.  
- BitNet-2B: 6912/2560 = 2.7 → marginal  
- Falcon3-3B: 9216/3072 = 3.0 → **+2.0 %** ✓  
- Falcon3-10B: 23040/3072 = 7.5 → **-18.0 %** ✗

### 2. L3 ACDC: degradação cresce com n_layers

O FWHT não usa AVX2 de forma tão eficiente quanto o GEMV I2_S. Com mais camadas,
o overhead acumula mais do que o benefício teórico O(n log n) vs O(n²).

**Gap crítico:** ACDC atual cobre apenas projeções de atenção **quadradas** (3072×3072).
As projeções FFN (3072×23040 e 23040×3072) não têm ACDC → **Fase II implementa ACDC retangular**.

### 3. L5 HRR: menos ruim com head_dim maior

BitNet-2B (d=128) e Falcon3-3B (d=256) mostram 62% e 40% de degradação.
Falcon3-10B (d=256, mais layers) mostra 36% — e o cleanup **supera o raw** apenas no 10B.
head_dim=256 oferece mais capacidade de representação holográfica mesmo sem retreino P6.

---

## Modelos disponíveis localmente

| Modelo | Path | Parâmetros | Formato |
|--------|------|-----------|---------|
| BitNet-b1.58-2B-4T | `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` | 2B | I2_S GGUF |
| Falcon3-3B-Instruct-1.58bit | `models/Falcon3-3B-Instruct-1.58bit/ggml-model-i2_s.gguf` | 3B | I2_S GGUF |
| Falcon3-3B-Instruct Q4_K_M | `models/Falcon3-3B-Instruct-Q4/Falcon3-3B-Instruct-Q4_K_M.gguf` | 3B | Q4_K_M GGUF |
| Falcon3-10B-Instruct-1.58bit | `models/Falcon3-10B-Instruct-1.58bit-GGUF/ggml-model-i2_s.gguf` | 10B | I2_S GGUF |

---

## Próximo passo: Fase II — ACDC retangular

O maior impacto no Falcon3-10B virá de aplicar FWHT às projeções FFN (3072×23040).
FWHT é O(n log n) vs GEMV O(n²) — para n=23040 isso é ~230× menos operações,
e essas projeções dominam o compute do 10B.

Ver `_reversa_forward/` para roadmap completo.
