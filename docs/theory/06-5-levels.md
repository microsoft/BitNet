# 06 — Os 5 Níveis Algébricos (Sumário Canônico de 1 Página)

> **Sumário consolidado** dos 5 níveis algébricos L1-L5 do BitNet CPU-Universal.
> **NÃO substitui** os docs primários em `docs/theory/0[1-5]-*.md`; é uma
> página de referência rápida com a tabela "Nível → Operação eliminada →
> Substituída por → Ganho".
>
> **Versão:** v0.1 — gerado por T036 (Fase 3: Núcleo) em 2026-06-06.
> **Ancoragem:** `docs/mathematical-foundations.md` (provas formais),
> `docs/findings-cpu-universal.md#1` (validação empírica), e
> `docs/invariants.md` (P1-P7).
>
> **AC-10 (do `requirements.md#6`):** "Documento `docs/theory/06-5-levels.md`
> resume os 5 níveis em uma página."

---

## Visão geral (TL;DR)

O BitNet CPU-Universal demonstra que **5 estruturas algébricas "esquecidas"**
eliminam operações caras em inferência de LLM, mantendo qualidade quando
o modelo é treinado com a arquitetura:

| Nível | Estrutura | Operação eliminada | Substituída por | Ganho |
|-------|-----------|--------------------|-----------------|-------|
| **L1** | Ternary quantization {-1, 0, +1} | FP32 weights (32 bits) | `quant(W) ∈ {-1,0,+1}` packed 4/byte | **20× menos memória** (1.58 bits/param) |
| **L2** | Walsh-Hadamard decomposition | Multiplicação por W | `W = H·D·H` (3 matrizes esparsas) + XOR/add | **Zero multiplicações** no kernel |
| **L3** | ACDC (Adaptive Circulant Diagonal Conv) | GEMM denso O(n²) | FWHT em circulant: `W·x = H·(d·(H·x))` | **O(n log n)** (vs O(n²)) |
| **L4** | Tropical (max,+) semiring | Softmax completo | `argmax` top-K + softmax sobre K tokens | **O(n·d + K·d)** (vs O(n²·d)) |
| **L5** | Holographic Reduced Representations (HRR) | Attention densa | `bind(q,k) = q ⊛ k` (FFT circular) + cleanup | **O(n·log d)** binding/unbinding |

**Restrição universal:** todos os níveis rodam **CPU-only** (decisão fundadora).
GPU é proibido (NO-02, persona D4 incompatível com GPU dedicado).

---

## Onde está cada nível no código

| Nível | Header | Source | Test primário | Test property (RF-01) |
|-------|--------|--------|---------------|----------------------|
| **L1 I2_S** | `include/ggml-bitnet-mad.h` | `src/ggml-bitnet-mad.cpp` | `tests/test_bitnet_common.cpp` | — (existente) |
| **L2 WHT** | `include/ggml-bitnet-wht.h` | `src/ggml-bitnet-wht.cpp` | `tests/test_wht.cpp` | — (existente) |
| **L3 ACDC** | `include/ggml-bitnet-fwht.h` | `src/ggml-bitnet-fwht.cpp` | `tests/test_acdc.cpp` | `tests/test_acdc_properties.cpp` (T005) |
| **L4 tropical** | `include/ggml-bitnet-tropical.h` | `src/ggml-bitnet-tropical.cpp` | `tests/test_tropical.cpp` | `tests/test_l4_sparse_properties.cpp` (T006) |
| **L5 HRR** | `include/ggml-bitnet-hrr.h` | `src/ggml-bitnet-hrr.cpp` | `tests/test_hrr_cleanup.cpp` + `tests/test_hrr_attention.cpp` | `tests/test_hrr_properties.cpp` (T007) |

---

## Trade-offs resumidos (1 linha por nível)

- **L1 I2_S** — Baseline, sempre funciona. Limitado pelo Shannon floor (1.58 bits/param).
- **L2 WHT** — Mostra a álgebra; **não integrado em produção** (kernel de pesquisa).
- **L3 ACDC** — Speedup teórico 100× vs GEMM, **mas exige retreino P6** (reserva Q4 2029).
- **L4 tropical** — **Único kernel que funciona com BitNet-2B sem retreino** (opt-in, D1).
- **L5 HRR** — Funciona com d≥256 e phasor keys; **d<256 é ruidoso** (capacidade).

---

## Quem precisa ler este documento

- **Novo contribuidor:** comece por este sumário, depois leia `docs/theory/0X-*.md`
  conforme o nível que te interessa. Não duplique o conteúdo aqui.
- **Usuário (persona D4):** §TL;DR e §Trade-offs. Não precisa das provas formais.
- **Mantenedor:** revise quando um nível ganha nova propriedade em
  `docs/invariants.md` ou novo test em `tests/test_*_properties.cpp`.

---

## Limitações conhecidas (P6)

L3 ACDC e L5 HRR são **arquiteturas de treinamento**, não compressões.
Aplicar `acdc_project` ou `hrr_bind` a um modelo clássico dá uma
**aproximação de ordem O(1/n)**, não uma representação fiel. Para
atingir paridade com transformer clássico, o modelo precisa ser
**treinado do zero** com a arquitetura correspondente.

Esta restrição está documentada em:
- `docs/invariants.md#p-especial` (P-estrutura)
- `ROADMAP.md#2.3` (reserva técnica P6)
- `requirements.md#12` (NO-01)

---

## Referências primárias (NÃO duplique, link)

| Nível | Doc primário | Conteúdo |
|-------|--------------|----------|
| L1 I2_S | `docs/theory/01-ternary-algebra.md` | Shannon floor, packing 4/byte, I2_S/TL1/TL2 codegen |
| L2 WHT | `docs/theory/02-wht-decomposition.md` | Hadamard decomposition, butterfly recursivo |
| L3 ACDC | `docs/theory/03-acdc-structured-layers.md` | FWHT em circulant, `acdc_forward` unnormalized |
| L4 tropical | `docs/theory/04-tropical-algebra.md` | (max,+) semiring, top-K argmax |
| L5 HRR | `docs/theory/05-holographic-memory.md` | FFT circular bind/unbind, phasor vs Gaussian keys |
| (todos) | `docs/mathematical-foundations.md` | Provas formais dos 5 níveis |
| (todos) | `docs/findings-cpu-universal.md#1` | Validação empírica (50 subtests) |
| (todos) | `docs/invariants.md` | P1-P7 canônicas |
| (todos) | `docs/decision-matrix.md` (T015) | Quando usar cada nível |

---

*v0.1 — gerado por T036 em 2026-06-06T21:45:00Z*
*Sumário canônico de 1 página. Não substitui `docs/theory/0[1-5]-*.md`.*
