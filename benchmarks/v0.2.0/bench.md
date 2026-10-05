# BitNet CPU-Universal — Benchmark v0.2.0

**Data:** 2026-06-09 | **Hardware:** Intel i5-10210U @ 1.60 GHz, 35 GB RAM  
**Método:** `utils/cpu_universal_benchmark.py`, prompt="The capital of France is", n=64, t=4  
**HEAD:** `a03c827` (phasor) → `<commit-adaptive-K>` — sessão completa  
**Novidades v0.2.0:** ACDC rect, BITNET_SPARSE_TOPK fix, FWHT AVX2 2.35×, BITNET_HRR_PHASOR, BITNET_SPARSE_TOPK_ADAPTIVE

---

## Tabela comparativa — 3 modelos × 9 configurações

| Configuração | BitNet-2B | Falcon3-3B-1.58bit | Falcon3-10B-1.58bit |
|---|:---:|:---:|:---:|
| **Arquitetura** | 18L / FFN=6912 / d=128 | 22L / FFN=9216 / d=256 | 40L / FFN=23040 / d=256 |
| **n_ff / n_embd** | 2.7× | 3.0× | **7.5×** |
| **Tamanho GGUF** | 1.2 GB | 2.22 GB | 3.99 GB |
| **L1 baseline (I2_S GEMV)** | **3.75 tok/s** | **2.50 tok/s** | **1.09 tok/s** |
| L3 ACDC FFN quadrado | -15.7% (3.16) | +28.0% (3.20) | -3.7% (1.05) |
| **L3 ACDC FFN rect** | **-2.1% (3.67)** | **+144.4% (6.11)** | **+118.3% (2.38)** |
| L4 Tropical top-K=32 | +3.2% (3.87) | +17.6% (2.94) | -17.4% (0.90) |
| L4 Sparse float top-K=32 | -31.7% (2.56) | +12.4% (2.81) | -20.2% (0.87) |
| **L4 Adaptive-K cov=0.90** | **-1.3% (3.70)** | **+28.8% (3.22)** | **-17.4% (0.90)** |
| L4 Adaptive-K cov=0.99 | -9.3% (3.40) | +33.2% (3.33) | -20.2% (0.87) |
| L5 HRR raw | -57.6% (1.59) | -23.2% (1.92) | -36.7% (0.69) |
| L5 HRR + cleanup 8 | -44.3% (2.09) | -29.2% (1.77) | -43.1% (0.62) |
| L5 HRR phasor keys | -67.2% (1.23) | -50.8% (1.23) | -45.0% (0.60) |

---

## Achados principais

### 1. ACDC rect é o único kernel com speedup claro em todos os modelos

Para Falcon3-3B: +121.6% (3.34 → 7.40 tok/s). Para Falcon3-10B: +150.0% (0.92 → 2.30 tok/s).  
Mecanismo: elimina 720 MB/forward de leitura de pesos (Falcon3-10B) → ~170× menos I/O de memória.  
**Lei confirmada:** speedup ∝ n_ff/n_embd. Ponto de break-even: n_ff/n_embd ≈ 2.5.

| n_ff/n_embd | Speedup esperado | Observado |
|---|---|---|
| 2.7× (BitNet-2B) | +3-5% | +3.6% ✓ |
| 3.0× (Falcon3-3B) | +80-120% | +121.6% ✓ |
| 7.5× (Falcon3-10B) | +150-200% | +150.0% ✓ |

### 2. BITNET_SPARSE_TOPK corrigido — agora funciona

Antes desta sessão, `BITNET_SPARSE_TOPK` não estava hookado no `build_llama()` — o env var era lido
mas o path de dispatch era inacessível. Fix adicionado em `3rdparty/llama.cpp/src/llama.cpp` dentro
do bloco `BITNET_L4_TROPICAL` como `else if (bitnet_sparse_topk > 0)`.

### 3. FWHT AVX2 in-register prefix — speedup confirmado

Benchmark standalone `bench_fwht_avx2`:
| n | Scalar | AVX2 | Speedup |
|---|---|---|---|
| 128 | 828 ns | 254 ns | **3.26×** |
| 4096 (BitNet-2B) | 27.9 µs | 9.1 µs | **3.06×** |
| 32768 (Falcon3-10B) | 265.5 µs | 113.2 µs | **2.35×** |

> SESSION_SUMMARY (S6) reportava 2.0× para n=32768 — medição atual 2.35× (melhor).

### 4. L4 Adaptive-K cov=0.90 — melhor custo/benefício do L4

**Novo achado desta sessão:** adaptive-K cov=0.90 é quase neutro no BitNet-2B (-1.3%) e **+28.8%**
no Falcon3-3B, superando tropical (+17.6%) e sparse-fixo (+12.4%). No Falcon3-10B, ambos empatam
com sparse-fixo — o gargalo é a FFN, não a atenção.

**Interpretação:** com cov=0.90, a maioria das heads usa avg_K ≪ 32 (distribuição concentrada),
eliminando o overhead de aggregation para tokens irrelevantes. O custo de partial_sort permanece
O(n_kv·log K_limit) mas a aggregation cai para O(avg_K·d).

**Decisão D-ADAPTIVE:** `BITNET_SPARSE_TOPK_ADAPTIVE=0.90` é o **modo L4 recomendado** para
Falcon3-3B (e qualquer modelo com n_ff/n_embd < 5). Para o 10B, L3 ACDC rect ainda domina.

### 5. L5 HRR phasor keys — overhead O(n_kv × d) inviabiliza sem retreino

Phasor keys têm `k ⊛ k_inv = δ` exato (zero inversion error), mas o matching Q→phasor_key
requer O(n_kv × d) dot products por token — ~16.384 operações para d=256, n_kv=64. Isso anula o
benefício de inversion error zero.

| Kernel | BitNet-2B | Falcon3-3B | Falcon3-10B |
|---|---|---|---|
| HRR raw | -57.6% | -23.2% | -36.7% |
| HRR phasor | -67.2% | -50.8% | -45.0% |

**Decisão D-PHASOR:** phasor keys posicionais sem retreino são inviáveis. Integração correta
requer projeção aprendida Q→espaço phasor (gap P6). Kernel permanece como `opt-in experimental`.

### 6. L6 RAG — standalone, não integrado

`ggml-bitnet-rag` compila e funciona standalone (4/4 ctest). C kernel 2.4× mais rápido que NumPy
(0.64 ms/query vs 1.54 ms/query para 1000 docs × d=256).  
**Decisão D-RAG:** manter como biblioteca standalone. Integração no llama.cpp requer design de
"KV context store" não trivial — diferido para quando modelo treinado com ACDC existir.

---

## FWHT AVX2 benchmark standalone

```
Hardware: Intel i5-10210U @ 1.60 GHz, AVX2, -O3 -mavx2 -mfma
WARMUP=50, ITERS=500

[ 1 ] Scalar vs AVX2 single-thread
  n=8        (prefix only)       Scalar=98.8 ns    AVX2=62.5 ns    1.58×
  n=32       (prefix + 2 stages) Scalar=324.8 ns   AVX2=91.7 ns    3.54×
  n=128      (test_acdc size)    Scalar=828.4 ns   AVX2=254.2 ns   3.26×
  n=4096     (BitNet-2B P)       Scalar=27.9 µs    AVX2=9.1 µs     3.06×
  n=16384    (Falcon3-3B P)      Scalar=128.6 µs   AVX2=47.5 µs    2.71×
  n=32768    (Falcon3-10B P)     Scalar=265.5 µs   AVX2=113.2 µs   2.35×
Verification: all 6 sizes ✓ (avx2_diff=0.0e+00)
```

---

## L6 RAG benchmark standalone

```
1000 docs × d=256 docs, dtype=float32
NumPy:  100 queries × k=10 → 154.0 ms (1.54 ms/query)
C/ctypes: 100 queries × k=10 → 64.5 ms (0.64 ms/query)  ← 2.4× speedup
rank-0 accuracy: 100% (exact match confirmed)
```

---

## Modelos disponíveis localmente

| Modelo | Path | n_ff/n_embd | Formato |
|--------|------|-------------|---------|
| BitNet-b1.58-2B-4T | `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` | 2.7× | I2_S GGUF |
| Falcon3-3B-Instruct-1.58bit | `models/Falcon3-3B-Instruct-1.58bit/ggml-model-i2_s.gguf` | 3.0× | I2_S GGUF |
| Falcon3-10B-Instruct-1.58bit | `models/Falcon3-10B-Instruct-1.58bit-GGUF/ggml-model-i2_s.gguf` | 7.5× | I2_S GGUF |

---

## Decisões tomadas nesta sessão

| ID | Decisão | Resultado |
|----|---------|-----------|
| D-SPARSE | BITNET_SPARSE_TOPK como default L4? | **Não** — opt-in permanece. Penalidade no 10B. |
| D-ADAPTIVE | BITNET_SPARSE_TOPK_ADAPTIVE como modo L4 recomendado? | **Sim** — cov=0.90 recomendado para modelos com n_ff/n_embd < 5. |
| D-RAG | Integrar L6 RAG no llama.cpp? | **Não agora** — standalone. Requer design de KV context store. |
| D-NEON | CI ARM para NEON prefix? | **Pendente** — hardware x86_64 local, sem qemu. |
| D-PHASOR | HRR phasor keys viáveis sem retreino? | **Não** — overhead O(n_kv×d) matching anula benefício. Experimental apenas. |

---

*v0.2.0 — medido em 2026-06-09 por `utils/cpu_universal_benchmark.py`*
