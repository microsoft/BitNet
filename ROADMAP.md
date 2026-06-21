# ROADMAP — BitNet CPU-Universal

> Roadmap **público** do fork, separado em 3 seções por horizonte temporal
> e compromisso. **Versão:** v0.2.2 — atualizado em 2026-06-09 (S7: T015/T016/T020-T023/T028 concluídos).
> **Ancoragem:** `requirements.md#8` (marcos M1-M5) e
> `.reversa/scout/gap-analysis.md`.
>
> **Persona-alvo:** D4 (Privacidade/Soberania) — ver `requirements.md#9`.
> Toda decisão aqui é influenciada por essa persona.

---

## ⏰ Reavaliações agendadas (Q4 2029)

> Esta seção é a primeira coisa a ser vista. Marca o **compromisso público**
> de reavaliar reservas técnicas em data específica. Próxima: **Q4 2029**.

| Data | Item | Gatilho | Ação esperada |
|------|------|---------|---------------|
| **Q4 2029** | **RF-06** (scaffolding fine-tuning ACDC) | LR-02 (D3) | Decidir: sobe para média / baixa definitiva / removido. Ver `requirements.md#10` (LR-02) |
| **Q4 2029** | **D-01`** (P6 retreino, LAC-01 🟡) | LR-02 + LR-01 | Reabrir clarificação sobre P6. Decidir se sobe para prioridade ou é aposentado. Ver `requirements.md#9` (D-01`) |
| **Q4 2029** | **D2 trigger** (Llama-2-7B smoke test) | LR-01 | Se ainda não executado, reavaliar viabilidade. Se impossível (sem GPU), aposentar e marcar como "diferencial permanente". Ver `requirements.md#10` (LR-01) |
| **Q4 2029** | **Persona D4** (LR-03) | Mudança de mercado/regulamentação | Se regulamentação europeia de IA / HIPAA / LGPD mudar significativamente, reabrir clarificação. Ver `requirements.md#10` (LR-03) |

**Compromisso:** em **outubro de 2029**, abrir nova rodada de `/reversa-clarify`
para reavaliar estes 4 itens. Resultado alimentará v0.3+ do roadmap.

---

## Resumo executivo (TL;DR)

| Seção | Horizonte | Status | Compromisso |
|-------|-----------|--------|-------------|
| **1. Atual** | v0.1 (curto prazo) | ✅ Pronto para release | Núcleo algébrico, persona D4, decision matrix, 11/13 ACs verdes |
| **2. Reserva técnica** | Reavaliação Q4 2029 | 📋 Documentado, não priorizado | RF-06 (finetune_acdc.py), retreino P6 |
| **3. Fora de escopo** | Indefinido | ❌ Nunca | GPU kernels, cloud, telemetria |

**Diferencial competitivo:** inferência 1.58-bit **CPU-only**, **local-first**,
**sem CUDA, sem cloud, sem telemetria** — para a persona D4 (saúde,
jurídico, financeiro, privacidade individual).

---

## 1. Atual (v0.1)

> O que está **em desenvolvimento** ou **pronto** agora. Tudo aqui tem
> commit hash ou ações atômicas rastreáveis em `_reversa_forward/001-trilha-rigor-produto/`.

### 1.1. Núcleo algébrico (L1-L5)

| Nível | Operação | Status | Localização | Tests |
|-------|----------|--------|-------------|-------|
| **L1 I2_S** | Ternary GEMM x86/ARM | ✅ Pronto | `src/ggml-bitnet-mad.cpp` | 9/9 ctest |
| **L2 WHT** | Walsh-Hadamard decomposition (zero mult) | ✅ Pronto | `src/ggml-bitnet-wht.cpp` | `test_wht` |
| **L3 ACDC** | Adaptive Circulant Diagonal Conv (FWHT) | ✅ Pronto | `src/ggml-bitnet-fwht.cpp` | `test_acdc` + `test_acdc_properties` (T005) |
| **L4 tropical** | (max,+) semiring, top-K argmax | ✅ Pronto (opt-in) | `src/ggml-bitnet-tropical.cpp` | `test_tropical` + `test_l4_sparse_properties` (T006) |
| **L5 HRR** | Holographic Reduced Representations (FFT) | ✅ Pronto (opt-in) | `src/ggml-bitnet-hrr.cpp` | `test_hrr_*` + `test_hrr_properties` (T007) |
| **L6 RAG** | CPU-RAG flat-index ANN (inner-product + adaptive-K) | ✅ Standalone (opt-in) | `src/ggml-bitnet-rag.cpp` | `test_rag_retrieval` (4/4) |

**Invariantes P1-P7** estão documentadas em `docs/invariants.md` (T013).
**P6 (Estrutura, não compressão)** é a tese central: L3 e L5 **não são
métodos de compressão**; são arquiteturas de treinamento (ver §2).

### 1.2. Features de produto (v0.1)

| Feature | RF | Status | Marco |
|---------|-----|--------|-------|
| Property-based tests (1000+ inputs) | RF-01 | ✅ Fase 2 | M1 |
| Decision matrix "quando usar L1-L5" | RF-02 | ✅ T015 (2026-06-09 v0.2) | M2 |
| Cross-validação C ↔ Python | RF-03 | ✅ Fase 2 (T011) | M2 |
| L4 sparse float opt-in | RF-05 | ✅ Comportamento + Doxygen (T017) | M2 |
| **L4 adaptive-K opt-in** | RF-05b | ✅ 2026-06-09 `BITNET_SPARSE_TOPK_ADAPTIVE` | M2 |
| **L3 ACDC rect auto** | RF-05c | ✅ 2026-06-09 `BITNET_ACDC_FFN_RECT=auto` | M2 |
| Bench sistemático + publicação | RF-07 | ✅ T020/T028 (2026-06-09 bench v0.2.0) | M5 |
| Persona D4 (Privacidade/Soberania) | D4 | ✅ `requirements.md#9` | M5 |
| Air-gapped boot (sem rede) | AC-11 | ✅ T010 (Fase 2) | M5 |
| Documentação persona D4 | AC-12 | ✅ T021-T023 (2026-06-09 v0.2) | M5 |
| Hardware-compatibility matrix | AC-13 | ✅ T016 (2026-06-09 v0.2) | M5 |

### 1.3. Métricas de qualidade (RNF-01, RNF-02)

- **ctest:** 15/15 verde (default CI), 16/16 com `-DBITNET_ENABLE_ACDC_RECT=ON`; ≥ 50 subtests (RNF-01) ✅
- **Performance:** baseline L1 dentro de ±2 % em `n=128, t=4` (RNF-02) ✅
- **Documentação:** pt-BR (RNF-03) ✅
- **Patches:** patches vendored em `patches/llama.cpp/` (RNF-04) ✅

### 1.4. Marcos restantes (v0.1)

| Marco | Status | O que resta |
|-------|--------|-------------|
| M1 (Hardening matemático) | ✅ **Concluído** (2026-06-09) | T013 ✅, T015 ✅, T029 ✅ (D2=DIFERENCIAL confirmado — `investigation-d2-result.md`) |
| M2 (Decision matrix) | ✅ **Concluído** (2026-06-09) | T015 ✅, T020 ✅, RF-05b/c ✅ |
| M3 (ACDC retangular) | 🚧 Pausado | D2 resolvido (DIFERENCIAL); agora gateado apenas por **P6** (retreino Q4 2029); T009/T018/T019 `[ ]` por design |
| M5 (Produto) | ✅ **Concluído** (2026-06-09) | T021-T023 ✅, T016 ✅, T028 ✅ |

---

## 2. Reserva técnica (reavaliação Q4 2029)

> O que está **documentado conceitualmente** mas **não priorizado** agora.
> Tudo aqui tem uma **data de reavaliação** e um **gatilho explícito**
> para reativação. **Nada é abandonado** — é diferido com rastreabilidade.

### 2.1. RF-06: Scaffolding de fine-tuning ACDC (`utils/finetune_acdc.py`)

**Status:** 📋 Documentado, **não priorizado**.

**O que é:** Loop em PyTorch que treina **apenas a diagonal d*** de cada
GEMV FFN, mantendo W frozen. Roda em CPU ou GPU. Estimativa: 1-2 dias
de A100, ~500 linhas.

**Por que é reserva:** A validação empírica dos kernels L3 (ACDC) e L5
(HRR) **exige P6 (retreino)**, que é explicitamente fora do escopo
CPU-only (NO-02). Sem retreino, BitNet-2B dá garbage com L2/L3/L5
(documentado em `docs/findings-cpu-universal.md#5`).

**Decisão D3 (esclarecimento, 2026-06-06):** "Explícito > implícito;
reavaliação periódica > ambição imediata." O scaffolding existe
conceitualmente, sem código. Reavaliação: **Q4 2029**.

**Gatilho para reativação:**
1. **GPU disponível** no ambiente de desenvolvimento, **E**
2. **Demanda de comunidade** documentada (issue aberta, PR upstream
   relacionado, ou menção em release notes de outro projeto).

**Ação quando reativar:** Criar `utils/finetune_acdc.py` (PyTorch) com
smoke test mínimo (`--smoke` flag), conforme AC-09 do `requirements.md#6`.

**Risco aceito:** Documentação sem código é mais fácil de esquecer
que código documentado. Mitigação: este ROADMAP.md é linked do README.md
e revisado em cada release.

### 2.2. M3 (ACDC retangular, FFN) — gateado por P6

**Status:** 🚧 **Diferencial** (não bloqueador). Gate D2 (T029) resolvido
em 2026-06-09 — resultado: **DIFERENCIAL**. Agora gateado exclusivamente
por **P6** (retreino GPU, Q4 2029).

**O que é:** Estender `acdc_project(d, W, n)` para matrizes m×n com
m ≠ n. Para BitNet-2B, isso cobre FFN (gate/up 2560×6912, down
6912×2560). Sem esta extensão, ACDC fica restrito a ~30 % das matrizes
do modelo (apenas attention QKV/O, que são 1280×1280 ou 2560×1280).

**Resultado D2 (T029, 2026-06-09):** Llama-2-7B testado em fp16 nativo
(13.5 GB GGUF) e Q4_K_M: `RECT=auto` = no-op correto (ratio 2.69 <
threshold 3.0); `RECT=1` = garbage (P6 gap, opt-in explícito). A falha é
**esperada e documentada** — o modo `=1` sem retreino é research-only.
`=auto` é seguro em produção (Falcon3-3B **+51.7%**, Falcon3-10B **+179%**
confirmados). Ver `investigation-d2-result.md`.

**Ações T009, T018, T019:** Pausadas por P6 (não por D2). Ativar
quando GPU + demanda de comunidade (mesmo gatilho de §2.1).

**Gatilho de reativação:** GPU disponível no ambiente de dev **E**
demanda de comunidade documentada. Reavaliação: **Q4 2029**.

### 2.3. P6 (Estrutura, não compressão) — validação empírica

**Status:** 🟡 Tese matemática comprovada (`docs/theory/03-acdc-structured-layers.md`),
validação empírica pendente (exige P6 retraining, que está em §2.1).

**O que é:** Demonstrar que ACDC (L3) e HRR (L5), **quando treinados
com a arquitetura desde o início**, atingem a paridade com transformers
clássicos em CPU-only, com speedup de 10-100×. Sem retreino, ACDC é
uma aproximação de ordem `O(1/n)` (não atinge paridade).

**Dívida D-01 → D-01`:** Dívida consciente com plano de pagamento
(reavaliação Q4 2029).

**Gatilho:** Mesmo de §2.1.

---

## 3. Fora de escopo (nunca)

> O que o fork **NÃO** faz, **NÃO** pretende fazer, e **NÃO** aceita
> como contribuição. Tudo aqui viola uma restrição fundadora ou a
> persona D4.

### 3.1. GPU kernels (NO-02)

**Status:** ❌ Nunca. **Restrição fundadora** do fork (CLAUDE.md,
ADR-003 se existente).

**Por que:** A persona D4 (laptop corporativo padrão, hardware legado)
**é incompatível** com GPU dedicado. Hardware GPU dedicado é caro,
requer drivers proprietários (CUDA, ROCm), e quebra a portabilidade
"roda em qualquer x86_64 com AVX2 (post-2013) ou ARM64 com NEON".

**Política:** PR que adicione código GPU é **rejeitado** sem review.
Issues sugerindo GPU são fechadas com link para esta seção.

### 3.2. Cloud deployment (NO-07)

**Status:** ❌ Nunca. Persona D4 assume uso **local single-user**.

**Por que:** A persona D4 exige que **nenhum dado saia do dispositivo
local**. Cloud deployment, mesmo com criptografia, é incompatível com
essa restrição. Servidor OpenAI-compat (`run_inference_server.py`)
permanece **desabilitado por padrão** e **não documentado** na persona D4
(ver `requirements.md#12`).

**Política:** PR que adicione deploy cloud, sync, multi-tenant, ou
qualquer abstração de servidor é rejeitado.

### 3.3. Telemetria de qualquer tipo (NO-06)

**Status:** ❌ Nunca. Por padrão, o binário não envia nenhum dado a
nenhum endpoint. Qualquer instrumentação nova deve ser opt-in, explícita
e justificada pela persona D4.

**Por que:** Telemetria viola a premissa fundamental da persona D4
(privacidade/soberania). Mesmo telemetria "anônima" é um vetor de
vazamento de uso que pode ser correlacionado com IP, timing, etc.

**Política:** PR que adicione código de telemetria (HTTP POST, log de
métricas remoto, analytics) é rejeitado. Auditoria NO-06 (T031) é
rodada como parte do CI.

**Auditoria atual:** `grep -rn "telemetry\|upload_data\|send_metrics"
src/ utils/ run_inference*.py` retorna 0 hits (ver `verification-report.md`
gerado por T033).

### 3.4. Mudança no formato GGUF ou no conversor HF → GGUF (NO-03)

**Status:** ❌ Nunca. O fork **consome** GGUF, não **produz** uma
variante.

**Por que:** GGUF é o formato canônico de BitNet. Mudar o formato
quebraria interoperabilidade com BitNet-2B e HuggingFace ecosystem. O
fork é uma **engine de inferência**, não um novo formato de modelo.

**Política:** PR que modifique o parser GGUF ou o conversor
`convert-helper-bitnet.py` é rejeitado (a menos que seja bugfix
localizado).

### 3.5. Integração com llama.cpp upstream como dependência (NO-04)

**Status:** ❌ Nunca. Submodule permanece inalterado. Mudanças vão em
`patches/llama.cpp/0N-*.patch` com sentinel idempotente em
`scripts/apply-dispatch-patches.sh`.

**Por que:** Persona D4 exige **dependências mínimas**. A integração
com upstream como dep traria cadeia de fornecedores (CIs, releases,
breaking changes) que a persona D4 não tolera.

**Política:** O submodule é read-only exceto para patches explícitos
via `apply-dispatch-patches.sh`.

---

## Reavaliações agendadas

> Lembretes visíveis no topo do ROADMAP para evitar esquecimento.
> Ver `SESSION_SUMMARY.md` para histórico de revisões.

| Data | Gatilho | Quem | O que |
|------|---------|------|-------|
| **Q4 2029** | Reavaliação periódica (LR-02, D3) | Mantenedor do fork | Reabrir `/reversa-clarify` sobre RF-06 (finetune_acdc.py) e M3 (T009/T018/T019). GPU + demanda de comunidade = gatilho de reativação. |
| **Q1 2027** | Próxima release minor (v0.2) | Mantenedor | Revisar §1 (Atual) e mover itens para §2 (Reserva) ou §3 (Fora) conforme apropriado. Candidatos: bench em mais hardware, ARM64, Windows. |
| **Sob demanda** | Mudança de persona ou regulamentação | Mantenedor | Se persona D4 mudar (LR-03) ou nova regulamentação (LGPD, EU AI Act, etc.), reabrir `/reversa-clarify`. |
| **Imediato (v0.1.0)** | Release tag | Mantenedor | Criar tag `v0.1.0`, push 6 commits locais, abrir PR upstream `microsoft/BitNet`. Ver `NEXT_STEPS.md`. |

**Mecanismo de reminder:** Este ROADMAP.md é linked do README.md
principal. Revisões de release checam este arquivo. (Ver R-07 do
`roadmap.md` da feature 001.)

---

## Como usar este ROADMAP

- **Se você é um contribuidor:** comece por §1.1 (Núcleo algébrico) e
  §1.4 (Marcos restantes). Suas PRs devem respeitar §3 (Fora de escopo).
- **Se você é um usuário (persona D4):** §1.1 lista o que funciona hoje.
  §1.2 lista as features de produto. §1.3 dá as métricas de qualidade.
- **Se você é um mantenedor:** §2 (Reserva) e o final "Reavaliações
  agendadas" são seus checkpoints. Não deixe §2 virar "abandonado" sem
  mover formalmente para §3.

---

## Referências cruzadas

- **Análise reversa:** `_reversa_sdd/architecture.md`, `_reversa_sdd/domain.md`
- **Síntese de princípios:** `.reversa/scout/principles.md` (7 princípios)
- **Decisões fundadoras:** `_reversa_sdd/adrs/001-007`
- **Findings consolidados:** `docs/findings-cpu-universal.md` (5 níveis, 4 bugs, 50 subtests)
- **Invariantes P1-P7:** `docs/invariants.md` (T013)
- **Decision matrix:** `docs/decision-matrix.md` (T015)
- **Hardware-compatibility:** `docs/hardware-compatibility.md` (T016)
- **Requirements:** `_reversa_forward/001-trilha-rigor-produto/requirements.md`
- **Roadmap da feature:** `_reversa_forward/001-trilha-rigor-produto/roadmap.md`
- **Actions:** `_reversa_forward/001-trilha-rigor-produto/actions.md`
- **Persona D4 (origem):** `requirements.md#9`

---

*v0.2.3 — atualizado em 2026-06-09: §2.2 M3 refletindo D2 resolvido (T029 DIFERENCIAL); reavaliação imediata v0.1.0 adicionada.*
*v0.2.2 — atualizado em 2026-06-09: M1/M2/M5 ✅; M3 atualizado; RF-05b/c adicionados.*
*v0.2 — atualizado por T035 em 2026-06-06T23:59:00Z*
*v0.1 — gerado por T014 em 2026-06-06T21:15:00Z*
