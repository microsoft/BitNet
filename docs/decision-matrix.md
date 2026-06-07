# Decision Matrix — Quando Usar L1 / L3 / L4 / L5

> **RF-02 (do `requirements.md#4`):** Decision matrix "quando usar L3 vs L4 vs L5".
>
> **Versão:** v0.1 — gerado por T015 (Fase 3: Núcleo) em 2026-06-06.
> **Ancoragem:** `requirements.md#9` (persona D4), `docs/invariants.md`
> (P1-P7), `docs/theory/06-5-levels.md` (T036, sumário), `ROADMAP.md`.

---

## TL;DR (5 linhas)

| # | Cenário | Kernel | Justificativa |
|---|---------|--------|---------------|
| 1 | **BitNet-2B (atual, denso)** | **L1 I2_S** | Baseline. L2/L3/L5 dão garbage (P6). |
| 2 | **Atenção esparsa em modelo denso** | **L4 sparse float** | Único que funciona sem retreino (T006 ✅). |
| 3 | **FFN com modelo P6-ACDC** | **L3 ACDC** | 100× speedup teórico, mas requer retreino (reserva Q4 2029). |
| 4 | **Edge device, d ≥ 256, modelo P6-HRR** | **L5 HRR** | Funciona com d grande; inviável em d=128 (capacidade). |
| 5 | **Pesquisa/exploração** | **L2 WHT** | Mostra a álgebra; não integrado em produção. |

**Pessoa alvo:** D4 (Privacidade/Soberania, `requirements.md#9`).
**Trade-off dominante:** compatibilidade > performance (D1).
**L4 sparse é opt-in** (D1, AC-06). Default = atenção densa.

---

## Tabela expandida com critérios de decisão

### Linha 1: BitNet-2B (atual, denso)

| Campo | Valor |
|-------|-------|
| **Cenário** | Você tem um modelo BitNet-2B ou similar já treinado. |
| **Kernel recomendado** | **L1 I2_S** (baseline; sempre funciona). |
| **Kernel a evitar** | L2 WHT, L3 ACDC, L5 HRR (todos dão garbage sem retreino P6). |
| **L4 sparse é OK?** | **Sim, opt-in** via `BITNET_SPARSE_TOPK=32`. Pode degradar qualidade; teste antes. |
| **Justificativa** | P1 (Shannon floor) garante que L1 atinge o mínimo teórico. Modelos não foram treinados com ACDC/HRR (P6), então L2/L3/L5 não têm semântica. |
| **Performance** | Baseline L1: ~5 tok/s em i5-8350U (BitNet-2B, t=4, 200 tokens). L4 sparse: ~7 tok/s (~+44 %). |
| **Test de validação** | `tests/test_bitnet_common.cpp` (L1), `tests/test_l4_sparse_properties.cpp` (L4 opt-in). |

### Linha 2: Atenção esparsa em modelo denso

| Campo | Valor |
|-------|-------|
| **Cenário** | Você quer acelerar atenção em modelo denso (BitNet-2B, Llama, etc.) sem retreino. |
| **Kernel recomendado** | **L4 sparse float** (D1, opt-in). |
| **Por que float e não tropical?** | Sparse float elimina a conversão F32→I8, mais rápido E mais correto. Tropical int8 ainda é válido para modelos com pesos ternários. |
| **L4 sparse top-K sugerido** | K=32 (default smoke-tested). K=64 para n_keys ≥ 256. |
| **Justificativa** | L4 é o único kernel que **funciona com BitNet-2B** sem retreino (T006 validou 3/3 invariantes). |
| **Quando NÃO usar** | Se o modelo tem atenção esparsa inerente (ex: Longformer, BigBird) — não conflitar. Se o modelo tem < 32 keys (n_ctx pequeno) — overhead > ganho. |
| **Como ativar** | `BITNET_SPARSE_TOPK=32 python run_inference.py ...` ou flag CLI `--attn sparse`. |
| **Risco aceito** | Regressão de qualidade se o modelo não foi treinado para atenção esparsa. Usuário assume. |

### Linha 3: FFN com modelo P6-ACDC (reserva técnica)

| Campo | Valor |
|-------|-------|
| **Cenário** | Você tem (ou terá) um modelo treinado com **ACDC** (L3) desde o início. |
| **Kernel recomendado** | **L3 ACDC** (FWHT em circulant, `acdc_forward`). |
| **Por que vale a pena** | Speedup teórico 100× vs GEMM denso (P3, O(n log n) vs O(n²)). |
| **Por que ainda não é rotina** | **P6 — Estrutura, não compressão.** ACDC exige retreino do zero. BitNet-2B atual dá garbage. Reserva técnica Q4 2029. |
| **Quando ativar** | Se você (a) tem GPU para retreinar E (b) está OK com 1-2 meses de retreino E (c) validou empiricamente com Llama-2-7B (gate D2). |
| **ACDC retangular (gate/up/down 2560×6912)** | T009, T018, T019 — gated by D2. Atualmente não implementado. |
| **Test de validação** | `tests/test_acdc.cpp#test_acdc_known_dense_recovery` (L3 quadrado), `tests/test_acdc_properties.cpp#p1..p4` (T005). |

### Linha 4: Edge device, d ≥ 256, modelo P6-HRR

| Campo | Valor |
|-------|-------|
| **Cenário** | Você tem (ou terá) um modelo com cabeças d ≥ 256 E treinado com **HRR** (L5) desde o início. |
| **Kernel recomendado** | **L5 HRR** (FFT circular bind/unbind). |
| **Por que d ≥ 256** | HRR retrieval quality requires `d ≥ 10·N`. Para N=32 tokens, d=256 é o mínimo; para N=64, d=640. Abaixo disso, retrieval é ruidoso. |
| **Por que phasor keys** | Phasor keys (spectrum de magnitude unitária) têm inversa exata via `IFFT(conj(FFT(k)))`. Gaussian random keys só têm inversa aproximada. Para BitNet-2B com HRR, use **phasor** (`hrr_phasor_key(d)`). |
| **Por que ainda não é rotina** | **P6 — Estrutura, não compressão.** HRR exige retreino. BitNet-2B atual dá garbage. |
| **Quando ativar** | Se você tem um modelo **explicitamente treinado com HRR** (não aplica ACDC/HRR a um modelo clássico — vai dar garbage). |
| **Test de validação** | `tests/test_hrr_cleanup.cpp`, `tests/test_hrr_attention.cpp`, `tests/test_hrr_properties.cpp#p1..p3` (T007). |

### Linha 5: Pesquisa / exploração

| Campo | Valor |
|-------|-------|
| **Cenário** | Você está estudando a álgebra (Hadamard, FWHT, FFT) ou fazendo PoC. |
| **Kernel recomendado** | **L2 WHT** (Walsh-Hadamard decomposition). |
| **Quando NÃO usar** | Em produção. L2 não está integrado ao dispatch (`src/ggml-bitnet-dispatch.cpp`); só acessível via test ou script ad-hoc. |
| **Por que existe** | Mostra que a álgebra funciona. Útil para entender L3 (que é L2 com diagonal) e para visualizar a estrutura do ACDC. |
| **Test de validação** | `tests/test_wht.cpp#test_wht_perfect_reconstruction`. |

---

## Decisões transcendentais (D1, D2, D3, D4)

| Decisão | Efeito na matriz | Origem |
|---------|------------------|--------|
| **D1** — L4 sparse é opt-in, não default | Linha 2 marcada como "opt-in" | `requirements.md#10` |
| **D2** — ACDC retangular é bloqueador condicional | Linha 3 gated por D2 (T029) | `requirements.md#10` |
| **D3** — RF-06 (finetune_acdc.py) é reserva Q4 2029 | Linha 3 não pode ser ativada agora | `requirements.md#10` |
| **D4** — Persona governa tudo | Foco em "single user, single laptop, sem rede" | `requirements.md#9` |

---

## Quando NÃO usar nenhum kernel algébrico (além do L1)

Se o seu caso de uso é:
- "Roda em GPU" → **saia deste fork** (NO-02, persona incompatível).
- "Cloud server, multi-tenant" → **saia deste fork** (NO-07, persona incompatível).
- "Telemetria-rich dashboard" → **saia deste fork** (NO-06, persona incompatível).
- "Modelo proprietário de LLM de fronteira (GPT-4, Claude)" → use a API deles; este fork é para BitNet-2B e similares.

---

## Como esta matriz é mantida

- **Atualização:** este doc é atualizado quando uma decisão (D1-D4) muda, ou quando um kernel novo entra em produção.
- **Fonte canônica:** se este doc diverge de `requirements.md#10` (esclarecimentos) ou `docs/invariants.md` (P1-P7), esses dois vencem.
- **Auditoria:** T033 (Fase 5) valida que cada linha tem test verde correspondente via `verification-report.md`.

---

## Referências cruzadas

- **Persona D4 (origem):** `requirements.md#9`
- **Esclarecimentos D1-D4:** `requirements.md#10`
- **Níveis L1-L5 (sumário):** `docs/theory/06-5-levels.md` (T036)
- **Invariantes P1-P7:** `docs/invariants.md` (T013)
- **Hardware-compatibility:** `docs/hardware-compatibility.md` (T016)
- **Roadmap público:** `ROADMAP.md` (T014)
- **Examples persona D4:** `examples/{medical,legal,finance}_offline.md` (T021-T023)

---

*v0.1 — gerado por T015 em 2026-06-06T22:00:00Z*
*5 linhas (BitNet-2B denso / sparse opt-in / FFN P6-ACDC / edge d≥256
P6-HRR / pesquisa L2) + trade-offs + decisões transcendentais D1-D4.*
