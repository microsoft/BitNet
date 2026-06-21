# Hardware Compatibility — BitNet CPU-Universal

> Tabela canônica CPU → modo de operação suportado. **AC-13** do
> `requirements.md#6` (Critérios de Aceitação para Produto Viável).
>
> **Versão:** v0.2 — atualizado em 2026-06-09 (T020, bench v0.2.0).
> **Ancoragem:** `requirements.md#9` (persona D4 hardware-alvo),
> `docs/invariants.md` (P1-P7), `docs/theory/0[1-5]-*.md`.

---

## TL;DR

| CPU (classe) | L1 I2_S | L2 WHT | L3 ACDC rect | L4 adaptive-K | L5 HRR |
|--------------|---------|--------|-------------|--------------|--------|
| **AVX-512 (post-2018)** | ✅ baseline | ✅ | ✅ opt-in | ✅ opt-in | ✅ d≥256 |
| **AVX2 (2013-2018)** | ✅ baseline | ✅ | ✅ opt-in | ✅ opt-in | ✅ d≥256 |
| **SSE4.2 (2008-2013)** | ⚠️ fallback | ⚠️ | ⚠️ | ⚠️ | 🟡 degradado |
| **ARM64 NEON** | ✅ baseline | ✅ | ✅ opt-in | ✅ opt-in | ✅ d≥256 |
| **ARMv7 (32-bit)** | ❌ não suportado | ❌ | ❌ | ❌ | ❌ |
| **GPU (qualquer)** | ❌ proibido (NO-02) | ❌ | ❌ | ❌ | ❌ |

**Persona D4** (laptop corporativo padrão, hardware legado) **deve** caber
em pelo menos AVX2. SSE4.2 é degradação aceitável, não crash. ARMv7 e
32-bit são **fora de escopo**.

---

## Tabela detalhada por nível algébrico

### L1 I2_S (Ternary GEMM)

| CPU | Suporte | Notas |
|-----|---------|-------|
| x86_64 com AVX2+ | ✅ Baseline | SIMD principal: `_mm256_maddubs_epi16` (32 ops/cycle) |
| x86_64 só SSE4.2 | ⚠️ Fallback | Performance ~3-5× pior, mas funcional. Fallback em `src/ggml-bitnet-mad.cpp` |
| x86 sem SSE4.2 | ❌ Crash | Não testado. Persona D4 assume SSE4.2 mínimo. |
| ARM64 com NEON | ✅ Baseline | SIMD principal: `vmlaq_s8` / `vmlal_s8` (similar ops/cycle) |
| ARMv7 (32-bit) | ❌ Não suportado | Codegen TL1 requer ARMv8 NEON |
| GPU (qualquer) | ❌ Proibido | NO-02 (GPU kernels) |

**Test mínimo:** `tests/test_bitnet_common.cpp` roda em qualquer CPU
suportada. SSE4.2 fallback validado manualmente em laptop corporativo
i5-4590 (2014, Haswell).

### L2 WHT (Walsh-Hadamard)

| CPU | Suporte | Notas |
|-----|---------|-------|
| x86_64 com AVX2+ | ✅ Ótimo | `src/ggml-bitnet-wht.cpp` usa AVX2 (`_mm256_xor_si256`) |
| x86_64 só SSE4.2 | ⚠️ Fallback | Versão escalar em `src/ggml-bitnet-wht.cpp` |
| ARM64 com NEON | ✅ Ótimo | Codegen TL2 não se aplica a L2; usa butterflies NEON |
| ARMv7 | ❌ Não suportado | NEON 64-bit requerido |

**Operação chave:** Zero multiplicações (P4, apenas XOR e adição). O
L2 é o kernel mais portável — não usa FP.

### L3 ACDC (FWHT)

| CPU | Suporte | Notas |
|-----|---------|-------|
| x86_64 com AVX2+ | ✅ Ótimo | `src/ggml-bitnet-fwht.cpp` butterfly in-place |
| x86_64 só SSE4.2 | ⚠️ Fallback | Versão escalar; ~4× mais lento |
| ARM64 com NEON | ✅ Ótimo | NEON butterfly |
| ARMv7 | ❌ Não suportado | |

**ACDC rect (FFN):** `BITNET_ACDC_FFN_RECT=auto` ativa automaticamente
quando `n_ff/n_embd >= 3.0`. Compatível com qualquer CPU da tabela.
Para Falcon3-10B (+267%) é o modo mais impactante disponível.

**ACDC atenção:** é **uma arquitetura de treinamento** (P6), não uma
otimização. Sem retreino, ACDC de atenção dá garbage em BitNet-2B
(`docs/findings-cpu-universal.md#5`). Só funciona em modelos
**treinados com ACDC** (reserva técnica Q4 2029, ver `ROADMAP.md#2.1`).

### L4 sparse / Adaptive-K

| CPU | Suporte | Notas |
|-----|---------|-------|
| x86_64 com AVX2+ | ✅ Ótimo | `sparse_attention_float_adaptive` usa AVX2 |
| x86_64 só SSE4.2 | ⚠️ Fallback | Escalar; ~3× mais lento |
| ARM64 com NEON | ✅ Ótimo | NEON int8 dot product |
| ARMv7 | ❌ Não suportado | |

**Modos disponíveis (por prioridade de despacho):**
1. `BITNET_SPARSE_TOPK_ADAPTIVE=<cov>` — adaptive-K (**recomendado**, cov=0.90)
2. `BITNET_TROPICAL_TOPK=K` — tropical (max,+) + K_i8 cache
3. `BITNET_SPARSE_TOPK=K` — sparse float fixo (legado)

**Atenção:** L4 é **opt-in** (D1, AC-06). Default = atenção densa.
Usuário **assume o risco** de regressão de qualidade ao ativar.

**Benchmark empírico:** Falcon3-3B + adaptive-K cov=0.90 → +28.8% vs
baseline. BitNet-2B → −1.3% (quase neutro). Falcon3-10B → gargalo é
FFN, não atenção; usar ACDC rect em vez de L4.

### L5 HRR (Holographic Reduced Representations)

| CPU | Suporte | Notas |
|-----|---------|-------|
| x86_64 com AVX2+ | ✅ d≥256 | d=128 funciona, mas capacidade de retrieval cai |
| x86_64 só SSE4.2 | 🟡 d≥512 | FFT escalar; qualidade aceitável apenas com d grande |
| ARM64 com NEON | ✅ d≥256 | NEON FFT |
| ARMv7 | ❌ Não suportado | |

**Atenção (operational regime):** HRR retrieval quality requires `d ≥
10·N` (d = head_dim, N = context tokens). Para `d=128`, capacidade
limita a N≤12 tokens sem ruído. Para uso prático de atenção HRR:
`d ≥ 640` para N=64, ou usar **phasor keys** (inversa exata via
conjugação espectral) em vez de chaves Gaussianas aleatórias
(`docs/theory/04-fft-binding.md`).

**Atenção (P6):** HRR é **arquitetura de treinamento** (P6). Sem
retreino, HRR dá garbage em BitNet-2B.

---

## Tabela de testes em hardware mínimo

> Resultados empíricos de smoke tests em hardware mínimo (persona D4
> laptop legado). Atualizado em cada release minor.

| Hardware | CPU | RAM | Data | L1 (tok/s) | L3 rect auto (tok/s) | L4 adaptive-K (tok/s) | Notas |
|----------|-----|-----|------|------------|---------------------|----------------------|-------|
| ThinkPad T480 (2018) | i5-8350U (4c/8t, AVX2) | 16 GB | 2026-05-15 | ~5.3 | n/t | n/t | Baseline de desenvolvimento |
| **Dell Inspiron (2019)** | **i5-10210U (4c/8t, AVX2)** | **16 GB** | **2026-06-09** | **3.83** | **4.48** | **3.31** | **Hardware de referência bench v0.2.0** |
| Dell Latitude 5490 (2018) | i5-8250U (4c/8t, AVX2) | 8 GB | 2026-05-15 | ~5.0 | n/t | n/t | Persona D4 target |
| MacBook Air M1 (2020) | M1 (8c, NEON) | 8 GB | 2026-05-20 | ~6.0 | n/t | n/t | Apple Silicon |
| Lenovo ThinkPad X250 (2015) | i5-5200U (2c/4t, AVX2) | 8 GB | 2026-05-22 | ~1.6 | n/t | n/t | Limite inferior viável |
| Intel NUC 2013 (Ivy Bridge) | i3-3220 (2c/4t, SSE4.2) | 4 GB | 2026-05-25 | ~0.8 | n/t | n/t | Fallback SSE4.2 |

**Observações:**
1. **i5-5200U (Broadwell, 2015)** é o limite inferior para a persona D4
   (8 GB RAM, AVX2). Performance aceitável para "uso interativo" (< 100s
   para 200 tokens) mas não para "uso concorrente".
2. **SSE4.2 fallback** (Ivy Bridge, 2013) é viável mas ~5× mais lento
   que AVX2. Não é persona D4 primário; é "uso emergencial".
3. **ARMv7 32-bit (Raspberry Pi legacy)** está fora de escopo.
   Codegen TL1/TL2 requer ARMv8.

---

## Como contribuir (compatibilidade)

Se você testou em um hardware **não listado acima** e quer contribuir:

1. Rode o smoke test:
   ```bash
   python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
     -p "The quick brown fox" -n 200 -t 4
   ```
2. Meça wall-clock time (em segundos).
3. Reporte em issue com:
   - Modelo exato de CPU (e.g., `i5-8350U`)
   - Ano de fabricação
   - RAM
   - OS e versão
   - Wall-clock (L1 default), (L3 ACDC rect auto, se aplicável) e (L4 adaptive-K opt-in, se aplicável)
4. Adicionamos à tabela acima no próximo release.

**Não reportamos GPUs** (NO-02).

---

## Limitações conhecidas

1. **BitNet-2B + L2/L3/L5 sem retreino = garbage** (P6, reserva técnica
   Q4 2029). A compatibilidade acima assume modelo **treinado com a
   arquitetura correspondente**. Para BitNet-2B atual, apenas L1 e L4
   sparse (opt-in) funcionam.
2. **M3 (ACDC retangular) é condicional** (gate D2). A tabela assume
   L3 quadrado (1280×1280 attention). FFN shapes 2560×6912 (gate/up) e
   6912×2560 (down) ainda **não suportados** (T009, T018, T019
   gated por D2).
3. **HRR d<256 é ruidoso** (ver "Atenção operational regime" acima).
   Para d<256, prefira L4 sparse.

---

## Referências cruzadas

- **Persona D4 hardware-alvo:** `requirements.md#9` (Intel i5/i7 6ª+
  ou ARM64 com NEON, 8-16 GB RAM)
- **Níveis algébricos:** `docs/theory/06-5-levels.md` (T036) ou
  `docs/findings-cpu-universal.md#1`
- **Invariantes P1-P7:** `docs/invariants.md` (T013)
- **Decisão L4 opt-in:** `requirements.md#10` (D1) e `requirements.md#6` (AC-06)
- **P6 (Estrutura, não compressão):** `requirements.md#12` (NO-01) e
  `ROADMAP.md#2.3` (reserva técnica)
- **Benchmarks v0.1.0:** `benchmarks/v0.1.0/bench.md` (T030)
- **Benchmarks v0.2.0:** `benchmarks/v0.2.0/bench.md` (T020, bench 3 modelos × 9 configs)

---

*v0.2 — atualizado em 2026-06-09 (T020)*
*Adicionado: L4 adaptive-K, ACDC rect auto, hardware i5-10210U (ref bench v0.2.0).*
*v0.1 gerado por T016 em 2026-06-06T21:30:00Z — Tabela CPU → modo (L1/L2/L3/L4/L5) + 6 HW testados.*
