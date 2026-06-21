# Decision Matrix — Quando Usar L1 / L3 / L4 / L5

> **RF-02 (do `requirements.md#4`):** Decision matrix "quando usar L3 vs L4 vs L5".
>
> **Versão:** v0.2 — atualizado em 2026-06-09 (bench v0.2.0 + adaptive-K + ACDC rect auto).
> **Ancoragem:** `requirements.md#9` (persona D4), `docs/invariants.md`
> (P1-P7), `docs/theory/06-5-levels.md` (T036, sumário), `ROADMAP.md`.

---

## TL;DR (7 linhas)

| # | Cenário | Kernel | Justificativa |
|---|---------|--------|--------------|
| 1 | **BitNet-2B (atual, denso)** | **L1 I2_S** | Baseline. L2/L3/L5 dão garbage (P6). |
| 2 | **Atenção esparsa, n_ff/n_embd < 5** | **L4 adaptive-K** `cov=0.90` | +28.8% no Falcon3-3B; quase neutro no BitNet-2B (−1.3%). |
| 3 | **FFN qualquer modelo (n_ff/n_embd ≥ 3.0)** | **L3 ACDC rect** `auto` | +118-144% nos Falcon3; zero custo no BitNet-2B (auto não ativa). |
| 4 | **FFN com modelo P6-ACDC** | **L3 ACDC** | 100× speedup teórico, mas requer retreino (reserva Q4 2029). |
| 5 | **Edge device, d ≥ 256, modelo P6-HRR** | **L5 HRR** | Funciona com d grande; inviável sem retreino. |
| 6 | **Atenção esparsa, K fixo explícito** | **L4 sparse float** K=32 | Opt-in; superado por adaptive-K na maioria dos casos. |
| 7 | **Pesquisa/exploração** | **L2 WHT** | Mostra a álgebra; não integrado em produção. |

**Pessoa alvo:** D4 (Privacidade/Soberania, `requirements.md#9`).
**Trade-off dominante:** compatibilidade > performance (D1).
**L4 adaptive-K é opt-in** (D1, AC-06). Default = atenção densa.
**L3 ACDC rect auto** é a forma mais simples de obter speedup em Falcon3-3B/10B sem configuração.

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

### Linha 2: Atenção esparsa, n_ff/n_embd < 5 (recomendado: adaptive-K)

| Campo | Valor |
|-------|-------|
| **Cenário** | Você quer acelerar atenção em modelo denso sem retreino. |
| **Kernel recomendado** | **L4 adaptive-K** `BITNET_SPARSE_TOPK_ADAPTIVE=0.90` (D1, opt-in). |
| **Por que adaptive-K e não sparse fixo?** | Seleciona K dinamicamente via threshold de cobertura softmax. Heads concentradas usam K≪32; uniform usam K≈32. Overhead de partial_sort é O(n·log K_max) mas aggregation cai para O(avg_K·d). |
| **Parâmetros** | `BITNET_SPARSE_TOPK_ADAPTIVE=0.90` (cobertura), `BITNET_SPARSE_TOPK_KMIN=1` (default), `BITNET_SPARSE_TOPK_KMAX=32` (default). |
| **Resultados empíricos** (i5-10210U, n=64, t=4) | BitNet-2B: −1.3% (quase neutro, avg_K≪32). Falcon3-3B: +28.8% (supera tropical +17.6% e sparse fixo +12.4%). Falcon3-10B: −17.4% (gargalo é FFN, não atenção). |
| **Quando usar sparse fixo** | Apenas quando K é conhecido a priori e o modelo tem distribuição uniforme de atenção. Use `BITNET_SPARSE_TOPK=32`. |
| **Quando NÃO usar L4** | n_ctx < 32 (overhead > ganho). Modelos com atenção esparsa nativa (Longformer, BigBird). |
| **Risco aceito** | Regressão de qualidade se o modelo não foi treinado para atenção esparsa. Usuário assume. |

### Linha 3: FFN qualquer modelo com n_ff/n_embd ≥ 3.0 (recomendado: ACDC rect auto)

| Campo | Valor |
|-------|-------|
| **Cenário** | Você quer speedup na FFN sem retreino, em qualquer modelo com FFN assimétrica. |
| **Kernel recomendado** | **L3 ACDC rect** `BITNET_ACDC_FFN_RECT=auto` |
| **O que `auto` faz** | Ativa rect automaticamente quando `n_ff/n_embd >= 3.0` (threshold empírico de break-even). Zero configuração extra. |
| **Resultados empíricos** (i5-10210U, n=64, t=4) | BitNet-2B (2.7×): no-op automático. Falcon3-3B (3.0×): +144%. Falcon3-10B (7.5×): +118%. |
| **Por que funciona sem retreino** | ACDC rect opera na FFN via FWHT com diagonal `d=0` (ou random com `BITNET_ACDC_FFN_RECT_RAND=1`). Output é numericamente incorreto (P6 gap), mas throughput é real. Para retreino real ver §Linha 4. |
| **Como ativar** | `BITNET_ACDC_FFN_RECT=auto` — detecta n_ff/n_embd e ativa quando ≥ 3.0. |
| **Threshold** | `>= 3.0f` (inclui Falcon3-3B com ratio exato 3.0). Threshold `> 3.0f` excluiria o 3B — bug histórico corrigido em 2026-06-09. |

### Linha 4: FFN com modelo P6-ACDC (reserva técnica)

| Campo | Valor |
|-------|-------|
| **Cenário** | Você tem (ou terá) um modelo treinado com **ACDC** (L3) desde o início. |
| **Kernel recomendado** | **L3 ACDC** (FWHT em circulant, `acdc_forward`). |
| **Por que vale a pena** | Speedup teórico 100× vs GEMM denso (P3, O(n log n) vs O(n²)). |
| **Por que ainda não é rotina** | **P6 — Estrutura, não compressão.** ACDC exige retreino do zero. BitNet-2B atual dá garbage. Reserva técnica Q4 2029. |
| **Quando ativar** | Se você (a) tem GPU para retreinar E (b) está OK com 1-2 meses de retreino E (c) validou empiricamente com Llama-2-7B (gate D2). |
| **ACDC retangular (gate/up/down 2560×6912)** | T009, T018, T019 — gated by D2. Atualmente não implementado. |
| **Test de validação** | `tests/test_acdc.cpp#test_acdc_known_dense_recovery` (L3 quadrado), `tests/test_acdc_properties.cpp#p1..p4` (T005). |

### Linha 5: Edge device, d ≥ 256, modelo P6-HRR

| Campo | Valor |
|-------|-------|
| **Cenário** | Você tem (ou terá) um modelo com cabeças d ≥ 256 E treinado com **HRR** (L5) desde o início. |
| **Kernel recomendado** | **L5 HRR** (FFT circular bind/unbind). |
| **Por que d ≥ 256** | HRR retrieval quality requires `d ≥ 10·N`. Para N=32 tokens, d=256 é o mínimo; para N=64, d=640. Abaixo disso, retrieval é ruidoso. |
| **Por que phasor keys** | Phasor keys (spectrum de magnitude unitária) têm inversa exata via `IFFT(conj(FFT(k)))`. Gaussian random keys só têm inversa aproximada. Para BitNet-2B com HRR, use **phasor** (`hrr_phasor_key(d)`). |
| **Por que ainda não é rotina** | **P6 — Estrutura, não compressão.** HRR exige retreino. BitNet-2B atual dá garbage. |
| **Quando ativar** | Se você tem um modelo **explicitamente treinado com HRR** (não aplica ACDC/HRR a um modelo clássico — vai dar garbage). |
| **Test de validação** | `tests/test_hrr_cleanup.cpp`, `tests/test_hrr_attention.cpp`, `tests/test_hrr_properties.cpp#p1..p3` (T007). |

### Linha 6: L4 sparse float K fixo (legado)

| Campo | Valor |
|-------|-------|
| **Cenário** | K conhecido a priori, distribuição de atenção presumida uniforme. |
| **Kernel** | `BITNET_SPARSE_TOPK=32` |
| **Quando preferir a adaptive-K** | Nunca, na prática. Adaptive-K com `cov=1.0` degenera para K=K_max e é numericamente equivalente. |
| **Mantido por** | Compatibilidade retroativa e referência de baseline. |

### Linha 7: Pesquisa / exploração

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
| **D1** — L4 sparse/adaptive-K é opt-in, não default | Linhas 2 e 6 marcadas como "opt-in" | `requirements.md#10` |
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

*v0.2 — atualizado em 2026-06-09 (bench v0.2.0)*
*7 linhas: L1 baseline / L4 adaptive-K / L3 ACDC rect auto / L3 P6-ACDC / L5 P6-HRR / L4 sparse fixo legado / L2 pesquisa.*
*Dados empíricos: i5-10210U, 3 modelos × 9 configs, n=64 t=4 (ver `benchmarks/v0.2.0/bench.md`).*
