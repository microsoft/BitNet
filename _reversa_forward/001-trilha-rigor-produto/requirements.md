# Requirements — `001-trilha-rigor-produto`

> **Feature:** Trilha de rigor teórico e fundamental para que BitNet CPU-Universal se mantenha categórico quanto aos fundamentos matemáticos e ainda assim evolua até se tornar um produto viável.
>
> **Argumento original:** "trilha de rigor teórico e fundamental para que BitNet se mantenha rígido e categórico quanto aos fundamentos matemáticos e ainda assim possa evoluir até se tornar um produto viável"
>
> **Gerado em:** 2026-06-06
> **Agente:** reversa-requirements + reversa-clarify
> **Ancoragem:** `_reversa_sdd/` (análise reversa prévia) + `.reversa/scout/` (síntese de princípios e gaps)
> **Idioma:** pt-BR
> **Versão:** 2 (pós-clarify: 4 dúvidas resolvidas, persona D4 adicionada, 3 ACs novos)

---

## 1. Visão

BitNet CPU-Universal já tem **5 níveis algébricos** (L1 I2_S, L2 WHT, L3 ACDC, L4 tropical, L5 HRR) que demonstram a tese de "inferência CPU via álgebra esquecida" no plano matemático. A cobertura de testes é sólida (9/9 ctest, 50 subtests, `docs/findings-cpu-universal.md`).

O gap entre o estado atual e um **produto viável** é de governança, não de código. Precisamos de um conjunto explícito de:

- **Invariantes matemáticas** que nenhum PR pode violar
- **Critérios de aceitação** que diferenciam "demo acadêmica" de "ferramenta que alguém usa em produção"
- **Marcos verificáveis** que tornam o progresso em direção ao produto mensurável

Esta feature é meta: ela **estabelece a trilha**, não implementa kernels. Entregas concretas virão como sub-features filhas (ex: property-based tests, decision matrix, finetune scaffold).

---

## 2. Contexto e Motivação

**Achado do `docs/findings-cpu-universal.md`:** os kernels L2/L3/L5 dão output garbage em BitNet-2B porque o modelo **não foi treinado com essas arquiteturas**. A tese matemática é correta (provada em `docs/theory/`), mas o caminho até validar empiricamente exige P6 (retreino GPU), que está explicitamente fora de escopo.

**Tensão central a resolver:**
- **Rigor matemático** exige provas, contra-exemplos, invariantes formais, cobertura ampla
- **Produto viável** exige uma feature drop-in que funciona HOJE em BitNet-2B, sem GPU

A trilha precisa entregar as duas coisas sem comprometer nenhuma: kernels matematicamente sólidos + um caminho de adoção que não exige retreino.

**Restrições inegociáveis (vindas de `_reversa_sdd/`):**
- CPU only — GPU proibida (decisão fundadora, ver ADR-003 se existente ou `CLAUDE.md`)
- Llama.cpp como backend (ADR-001, `bitnet-cpp` como nome do conda env)
- Clang ≥ 18 obrigatório (ADR-002) — GCC tolerado com `-fpermissive`
- Não tocar `3rdparty/llama.cpp` exceto via patches vendored em `patches/llama.cpp/`
- Não modificar `_reversa_sdd/` nem `.reversa/context/` (imutáveis)

---

## 3. Princípios Matemáticos Inegociáveis (Invariantes)

Cada PR que toque código algébrico (`src/ggml-bitnet-*.cpp`, `utils/extract_*.py`, `utils/codegen_*.py`) deve preservar estas invariantes. **Quebrar uma = bloquear o PR**, não documentar depois.

### P1 — Fechos formais dos kernels são verificáveis

Cada kernel algébrico tem um **fecho matemático documentado**: para QUAL classe de entrada ele é exato, e para QUAL classe ele é aproximação. Documentado em:
- `include/ggml-bitnet-*.h` (cabeçalho público de cada kernel)
- `docs/theory/0[1-5]-*.md` (prova + limit error)

Invariante prática: para todo kernel algébrico novo ou modificado, existe um **test de contra-exemplo exato** (não só teste aleatório). Exemplo: `test_acdc.cpp#test_acdc_exact_recovery` valida que para `W = H·diag(d)·H`, o d* extraído é exato (erro = 0, energia = 1.0).

### P2 — Especificação > Implementação

Toda especificação matemática vive em **dois lugares canônicos** e em mais nenhum:
1. `docs/theory/0X-*.md` (formal, com prova)
2. `test_<kernel>.cpp` (executável, com asserção)

Se uma das duas diverge da outra, o test vence (assume-se que o test está correto e a prosa está errada). Isso é o oposto da prática comum e foi explicitamente validado em S2.4: o bug "ACDC fwht_i8_to_i32 normalization" só foi pego porque atualizamos o test, não a prosa.

Invariante prática: o `ctest` é a especificação. Mudar a prosa sem mudar o test é permitido (atualização de doc); mudar o test sem mudar a prosa é um **red flag** que exige review.

### P3 — Níveis não compartilham butterflies

P3 dos princípios transversais (`.reversa/scout/principles.md:32-50`): WHT (L2), FWHT (L3), FFT (L5) **não compartilham uma API butterfly comum**. A tentação de DRY-ificar leva a bugs sutis onde um kernel usa o butterfly do outro. Documentado no header `include/ggml-bitnet-common.h`. Invariante: cada kernel tem sua própria implementação de butterfly, sem dependência cruzada de funções internas.

### P4 — ACDC é unnormalized (sem 1/n²)

P4 dos princípios: `acdc_forward(x) = H · (d · (H · x))` SEM fatores de 1/n². O bug S2.4 introduziu um stray `1/n²` que violou esta invariante e foi pego por `test_acdc.cpp#test_acdc_known_dense_recovery`. Invariante: todo `acdc_*` (forward, gemv, project) é unnormalized.

### P5 — Escala do ACDC é lockada no primeiro call

O cache K_i8 (`include/ggml-bitnet-kv-cache.h`) locka a escala de quantização no primeiro call. Isso é uma decisão de design, não um bug: lockar a escala garante que o ranking top-K permanece estável entre decode steps. Se um novo call trouxer keys com magnitude maior, a escala não se ajusta — keys saturam em ±127. Trade-off documentado: simplicidade de cache > precisão marginal.

Invariante: `bitnet_kv_i8_cache_get` nunca recaulcula `k_scale` depois do primeiro call por slot. Validado em `test_kv_i8_cache.cpp#test_incremental_only_new`.

### P6 — Strided head loop NÃO é thread-safe em GQA > 1

Lição aprendida em S2c.5: o bug "double free or corruption" foi causado por múltiplas threads (de strided head loop) compartilhando o mesmo `kv_h` (devido a GQA: n_head=20, n_head_kv=5, gqa=4). Invariante: **toda estrutura de dados particionada por (layer, head) precisa de sincronização explícita em modelos com GQA > 1**, ou de prova formal de que threads disjuntas escrevem nela. O `pthread_mutex` por slot do cache K_i8 é o padrão atual.

### P7 — Diffs matemáticos precisam de tests de contra-exemplo exato

`docs/findings-cpu-universal.md#bug-4-acdc-energy-formula`: o bug "energia = n vs n²" só foi pego porque `test_acdc_exact_recovery` usava `W = H·diag(d)·H` como contra-exemplo exato. Sem esse padrão, o bug teria passado com energia "razoável" (0.125) sem disparar alerta. Invariante: cada kernel algébrico tem pelo menos um **test de contra-exemplo exato** (input conhecido → output conhecido bit-a-bit, não estatístico).

---

## 4. Requisitos Funcionais

### RF-01: Property-based tests para todos os kernels algébricos

Substituir (ou complementar) os testes de valor fixo por testes baseados em propriedades, gerando 100+ inputs aleatórios por run. Cada kernel declara suas invariantes (ex: `||d*|| ≤ ||W||/n` para ACDC, `H·W·H = n²·diag(d)` para W diagonal-via-H) e o test verifica.

**Prioridade:** Alta. Sem isso, regressões sutis passam (caso documentado em S2.4).

### RF-02: Decision matrix "quando usar L3 vs L4 vs L5"

Documento `docs/decision-matrix.md` que diz, em uma página, **quando cada kernel é recomendado**. Baseado em:
- `docs/findings-cpu-universal.md` (dados empíricos)
- `.reversa/scout/gap-analysis.md` (estado consolidado)
- Princípios P3 (não compartilhar butterflies) e P6 (estrutura, não compressão)

Tabela esperada:

| Cenário | Kernel | Justificativa |
|---------|--------|---------------|
| BitNet-2B (atual, denso) | L1 I2_S | Baseline; L2/L3/L5 dão garbage |
| Atenção esparsa em modelo denso | **L4 sparse float** | Única opção que funciona sem retreino |
| FFN com modelo P6-ACDC | L3 ACDC | 100× speedup teórico, mas requer P6 |
| Edge device, d ≥ 256, modelo P6-HRR | L5 HRR | Funciona com d grande; inviável em d=128 |
| Pesquisa/exploração | L2 WHT | Mostra a álgebra; não integrado em produção |

**Prioridade:** Alta. Reduz a curva de aprendizado de novos contribuidores.

### RF-03: Cross-validação C ↔ Python

Para cada kernel com versão Python (`utils/extract_acdc_diagonal.py`, scripts de benchmark), gerar **seeds idênticas** e verificar que o resultado do C e do Python batem bit-a-bit (com tolerância de ponto flutuante). Implementar como `test_cross_validation.cpp` ou script Python que orquestra.

**Prioridade:** Média. Catches divergence between research code and production code.

### RF-04: ACDC para matrizes retangulares (Caminho A++)

Estender `acdc_project(d, W, n)` para matrizes m×n com m ≠ n. Para BitNet-2B isso cobre FFN (`gate_proj, up_proj` são 2560×6912, `down_proj` é 6912×2560). Sem essa extensão, ACDC fica restrito a 30% das matrizes do modelo.

**Classificação inicial (esclarecimento D2):** "diferencial, não bloqueador". Esta classificação é **condicional** e deve ser reavaliada empiricamente. Gatilho de reclassificação para "bloqueador imediato": executar inferência fim-a-fim com Llama-2-7B (modelo popular, não-BitNet, fp16) através do pipeline BitNet; se a falha no FFN impedir geração de texto coerente (perplexity > 100 ou output repetitivo/incoerente em prompt simples), RF-04 vira bloqueador e M3 é movido para curto-prazo. Caso contrário, permanece diferencial.

**Prioridade:** Média condicional.

### RF-05: L4 sparse float como caminho opt-in

Mover `sparse_attention_float` de "variante experimental" para "caminho L4 disponível, opt-in" via env var `BITNET_SPARSE_TOPK` ou flag CLI `--attn sparse`. **Default permanece attention denso** (esclarecimento D1). Documentar em `docs/decision-matrix.md` que sparse float é o L4 recomendado para BitNet-2B (mais rápido que tropical a n ≥ 32, sem int8, sem cache, mais simples), mas o usuário **assume o risco** de regressão ao habilitar uma otimização para a qual o modelo pode não estar preparado (modelos não-treinados para atenção esparsa podem degradar qualidade).

Esta escolha preserva compatibilidade com a maioria dos modelos existentes (D1: "comportamento default deve preservar a compatibilidade"). Atende o princípio P6 (estrutura, não compressão): não impomos ao usuário uma otimização estrutural sem consentimento explícito.

**Prioridade:** Alta (mas conservadora: opt-in, não default).

### RF-06: Scaffolding de fine-tuning ACDC (reserva técnica)

`utils/finetune_acdc.py` (PyTorch): loop que treina **só a diagonal** d* de cada GEMV FFN, mantendo W frozen. Roda em CPU ou GPU. **Não executar P6** (retreino completo), só deixar o código pronto para quando a GPU aparecer. Estimativa: 1-2 dias de A100, 500 linhas.

**Classificação (esclarecimento D3):** tratado como **reserva técnica** — o código existe mas não é prioridade atual. Deve ser explicitamente documentado em `ROADMAP.md` (ou seção equivalente em `README.md`) com:
- Status: "disponível, mas não priorizado"
- Marco de reavaliação: **Q4 2029** (ou a próxima data revisável escolhida pelo time)
- Critério para reativar: GPU disponível no ambiente de desenvolvimento + demanda de comunidade documentada (issue aberta ou PR upstream relacionado)

Esta decisão preserva o fork como CPU-only sem fingir que P6 está em andamento, e dá um sinal claro para contribuidores externos sobre o status da feature.

**Prioridade:** Baixa (reserva).

### RF-07: Script de benchmark público (BitNet-CPU leaderboard)

`utils/bench_publish.py`: roda o bench sistemático e produz um JSON+Markdown que pode ser commitado e versionado. Permite tracking de performance ao longo do tempo e comparação com baselines (transformers equivalentes em CPU).

**Prioridade:** Baixa. Marketing técnico, não bloqueia.

---

## 5. Requisitos Não-Funcionais

### RNF-01: Cobertura de testes permanece ≥ 9/9 ctest, 50/50 subtests

Cada nova feature **adiciona** testes, nunca remove. Cobertura por kernel: pelo menos 1 test de contra-exemplo exato (RNF derivado de P7).

### RNF-02: Performance não regride

Cada PR mantém o baseline L1 dentro de ±2 % em `n=128, t=4` (BitNet-2B, `utils/cpu_universal_benchmark.py`). Se um PR regredir, ou otimiza de volta ou justifica a regressão (ex: novo kernel é mais lento mas mais correto).

### RNF-03: Documentação em pt-BR

Prose explicativa (não comentários de código) em português. Comentários de código em inglês (padrão da indústria). Esta é a convenção do projeto desde a fundação.

### RNF-04: Não tocar `3rdparty/llama.cpp` exceto via patches vendored

Submodule permanece inalterado. Mudanças vão em `patches/llama.cpp/0N-*.patch` com sentinel idempotente em `scripts/apply-dispatch-patches.sh`. Já implementado (S1), manter.

---

## 6. Critérios de Aceitação para "Produto Viável"

Um release do BitNet CPU-Universal é considerado "produto viável" (e pode ir para upstream PR / Hugging Face) quando **TODOS** estes critérios são satisfeitos:

| # | Critério | Verificação |
|---|----------|-------------|
| AC-01 | ctest passa 9/9 com ≥ 50 subtests, runtime < 1s | `ctest --output-on-failure` |
| AC-02 | Pelo menos 1 kernel algébrico (L3 ACDC ou L4 sparse) tem property-based tests com 1000+ inputs | `tests/test_*_properties.cpp` (a criar) |
| AC-03 | `docs/decision-matrix.md` existe e tem tabela de quando usar o quê | Inspeção visual |
| AC-04 | `docs/findings-cpu-universal.md` cobre os 5 níveis, 4 bugs, 50 subtests | Já existe (S2e) |
| AC-05 | Bench sistemático commitado em `benchmarks/v0.1.0/` mostra baseline L1 vs L3 vs L4 com números | `utils/bench_publish.py` (a criar) |
| AC-06 | L4 sparse float é o caminho de atenção default quando `BITNET_SPARSE_TOPK` está setado | Code review do dispatch |
| AC-07 | Patches vendored em `patches/llama.cpp/` aplicam via `apply-dispatch-patches.sh` em clone fresh | CI step |
| AC-08 | ACDC cobre matrizes retangulares (FFN) — *bloqueador condicional* (gated por trigger de reclassificação empírica via Llama-2-7B; ver RF-04) | `test_acdc_rect.cpp` (a criar) |
| AC-09 | Scaffolding de fine-tuning ACDC existe e roda em smoke test — *reserva técnica* (RF-06; reavaliação Q4 2029) | `utils/finetune_acdc.py --smoke` |
| AC-10 | Documento `docs/theory/06-5-levels.md` resume os 5 níveis em uma página | Já parcialmente existe em `mathematical-foundations.md` |
| AC-11 | Binário roda em ambiente air-gapped (sem rede) sem crash, sem warning de telemetria, sem tentativa de download (D4 persona: privacidade/soberania) | `tests/test_air_gapped_boot.sh` (a criar) |
| AC-12 | Documentação e exemplos usam cenário "single user, single laptop, sem rede" como caso canônico (D4) | Inspeção visual de `docs/`, `examples/`, `README.md` |
| AC-13 | Compatibilidade declarada com CPUs pré-AVX2 (x86_64) e ARM64 com NEON, com degradação aceitável documentada (D4 hardware-alvo) | Tabela em `docs/hardware-compatibility.md` |

**Limiar mínimo para "produto viável"**: AC-01 a AC-07 verdes. AC-08 a AC-10 são "diferenciais" que tornam o produto competitivo, com AC-08 podendo ser reclassificado como bloqueador (trigger em RF-04) e AC-09 mantido como reserva técnica (RF-06).

---

## 7. Ancoragem em Artefatos Pré-Existentes

Esta feature **não** inventa princípios. Ela codifica e torna verificáveis princípios que já estão documentados em:

| Princípio | Fonte primária | Fonte derivada |
|-----------|----------------|----------------|
| P1 Shannon floor | `docs/theory/01-ternary-algebra.md:5-24` | `.reversa/scout/principles.md:18-26` |
| P2 Identidade algébrica | `docs/theory/00-index.md:44-72` | `.reversa/scout/principles.md:28-37` |
| P3 Hierarquia de custo | `docs/mathematical-foundations.md:18-28` | `.reversa/scout/principles.md:39-50` |
| P4 Mínimo irredutível | `docs/theory/03-acdc-structured-layers.md:65-87` | `.reversa/scout/principles.md:52-60` |
| P5 Tropical | `docs/theory/04-tropical-algebra.md:56-105` | `.reversa/scout/principles.md:62-71` |
| P6 Estrutura, não compressão | `docs/theory/03-acdc-structured-layers.md:159-189` | `.reversa/scout/principles.md:73-82` |
| P7 FFT como cola | `docs/theory/02-wht-decomposition.md:50-64` | `.reversa/scout/principles.md:84-93` |
| 4 bugs encontrados | `docs/findings-cpu-universal.md#2-bugs-reais-encontrados` | (S2) commits `cdce725`, `ed6fbde`, `ec2a654`, `fcf1d4d` |
| 5 níveis algébricos | `docs/mathematical-foundations.md:30-200` | `docs/findings-cpu-universal.md#1-os-5-níveis-algébricos` |
| 16 domain rules | `_reversa_sdd/domain.md` | `.reversa/scout/principle-code-map.json` |
| 7 ADRs | `_reversa_sdd/adrs/001-007` | `.reversa/context/surface.json` |
| Gap P6 (retreino GPU) | `.reversa/scout/gap-analysis.md` | `docs/findings-cpu-universal.md#5-por-que-a-tese-não-validou` |

---

## 8. Marcos Verificáveis (Milestones)

Não ordenados por dependência técnica, mas por **valor de produto**:

- **M1: Hardening matemático (curto prazo, 2-3 semanas)**
  - RF-01 property-based tests
  - Documentar invariantes P1-P7 em `docs/invariants.md`
  - RNF-01 ctest 9/9 + 50+ subtests

- **M2: Decision matrix (curto prazo, 1 semana)**
  - RF-02 `docs/decision-matrix.md`
  - RF-05 L4 sparse float como opt-in (não default) — D1

- **M3: ACDC retangular (médio prazo, 1-2 meses) — bloqueador condicional**
  - RF-04 ACDC para FFN, mas classificação "diferencial" até trigger de reclassificação (test com Llama-2-7B) — D2
  - RNF-02 performance não regride
  - Property tests cobrindo FFN shapes
  - Se trigger D2 dispara, M3 vira curto-prazo

- **M4: Validação empírica (reserva técnica, reavaliação Q4 2029)**
  - RF-06 scaffolding de fine-tuning como reserva explícita
  - (Fora de escopo deste fork) P6 retraining real
  - Critério de reativação: GPU no ambiente de dev + demanda de comunidade

- **M5: Produto (médio prazo, paralelo a M1-M3)**
  - AC-01 a AC-07 verdes
  - PR upstream aberto em `ggerganov/llama.cpp`
  - HF integration `AutoModel.from_pretrained(attention="sparse")`

---

## 9. Persona Alvo

> Definida em sessão `/reversa-clarify` (2026-06-06). Esta persona governa todas as decisões de produto daqui em diante: o que documentar, como documentar, o que priorizar, o que postergar.

### Desenvolvedores de Privacidade e Soberania de Dados

**Definição.** Usuários que exigem que **nenhum dado saia do dispositivo local**, mas que **não podem arcar com o custo** de servidores GPU locais.

**Perfil profissional e demográfico:**
- Setores **regulamentados**: saúde (LGPD/HIPAA), jurídico (sigilo profissional), financeiro (compliance BCB/GLBA)
- Usuários finais preocupados com privacidade que desejam rodar **assistentes pessoais** ou **analisadores de documentos** em laptops corporativos padrão ou hardware legado
- Idiomas prioritários: pt-BR, en-US (documentação bilíngue quando útil)

**Hardware-alvo:**
- Laptops corporativos comuns: Intel i5/i7 de 6ª geração em diante, 8-16 GB RAM
- Hardware legado: qualquer x86_64 com AVX2 (post-2013) ou ARM64 com NEON
- Sem placa de vídeo dedicada; sem acesso a clusters; sem internet obrigatória após instalação

**Diferencial competitivo (do ponto de vista da persona):**
- Arquitetura 1.58 bits (ternária: -1, 0, +1) **elimina a dependência de CUDA** e bibliotecas GPU proprietárias
- Execução **nativa em CPUs x86 e ARM** com dependências mínimas (libstdc++, libgomp, sem CUDA, sem ROCm)
- Modelo **inteiro off-line** após download inicial do GGUF: nenhuma chamada externa, nenhuma telemetria, nenhum cloud round-trip
- Footprint de RAM previsível (BitNet-2B a 4-bit KV cache cabe em 4-5 GB)

**Implicações para o produto:**

1. **Documentação e exemplos** devem focar no cenário "single user, single laptop, sem rede". Não há persona "cluster GPU" no produto.
2. **Marketing técnico** deve enfatizar "sem CUDA, sem GPU, sem cloud" como headline (vs. llama.cpp upstream que assume GPU disponível).
3. **Critérios de aceitação** devem incluir verificações de que o binário roda sem internet (AC-11: smoke test de boot em ambiente air-gapped).
4. **Compatibilidade de hardware** é um vetor de aceitação: testar em laptop com CPU pré-AVX2 e documentar degradação aceitável (não crash).
5. **Trade-offs de qualidade vs. privacidade** sempre pendem para privacidade: preferimos "modelo menor que cabe no dispositivo" a "modelo maior que requer cloud".
6. **Telemetria é proibida** por padrão. Qualquer instrumentação nova deve ser opt-in e documentada como tal (alinhado com P6 — estrutura, não compressão: o sistema respeita a integridade do dispositivo, não o espreme).

### Casos de uso canônicos (ilustrativos, não-exaustivos)

| Caso de uso | Persona | Como BitNet CPU-Universal atende |
|-------------|---------|----------------------------------|
| Médico analisa prontuários em laptop de consultório, sem internet | Saúde (regulamentado) | L1 I2_S + sparse opt-in; ar local; zero telemetria |
| Advogado resume petição inicial em escritório de advocacia pequeno | Jurídico (regulamentado) | L1 I2_S; roda em laptop com 8 GB RAM; sem dependência externa |
| Analista financeiro categoriza despesas em workstation bancária restrita | Financeiro (regulamentado) | L1 I2_S; auditável (modelo determinístico); sem upload de dados sensíveis |
| Pesquisador universitário roda BitNet-2B em máquina institucional bloqueada | Acadêmico (privacidade) | L1 I2_S + L4 sparse opt-in para experimentação; sem CUDA disponível |
| Entusiasta roda BitNet-2B em laptop de 2018 | Hobbyista (privacidade) | L1 I2_S; performance aceitável; sem upgrades de hardware necessários |

---

## 10. Esclarecimentos

> Sessão de clarificação realizada em **2026-06-06** via `/reversa-clarify`. Quatro dúvidas foram resolvidas e integradas in-place no documento. Os marcadores `[DÚVIDA]` foram removidos.

### Sessão 2026-06-06

- **Q (D1):** L4 sparse float deve ser o caminho default L4 mesmo sem env var?
  **R:** **Não.** O comportamento default preserva compatibilidade com a maioria dos modelos existentes. O attention denso permanece como padrão. O modo sparse é **opt-in** via env var `BITNET_SPARSE_TOPK` ou flag `--attn sparse`. O usuário assume o risco de regressão ao ativar uma otimização para a qual o modelo pode não estar preparado. Reflete em RF-05 e AC-06.

- **Q (D2):** ACDC para matrizes retangulares (FFN gate/up/down 2560×6912) deve ser bloqueador do v0.1?
  **R:** **Classificação condicional com trigger empírico.** Inicialmente classificado como "diferencial". Deve-se executar um **teste de inferência com um modelo popular** (ex: Llama-2-7B) através do pipeline BitNet. **Se a falha no FFN impedir a geração de texto coerente**, a classificação deve ser **atualizada imediatamente para "Bloqueador"** e a implementação de RF-04 priorizada (M3 movido para curto-prazo). Reflete em RF-04, AC-08 e M3.

- **Q (D3):** Quando (e se) o scaffolding RF-06 (finetune_acdc.py) vira prioridade?
  **R:** **Reserva técnica com marco de reavaliação.** Atualizar o `README.md` ou criar um `ROADMAP.md` para explicitar que o scaffolding existe apenas como **reserva técnica**, sem prioridade atual. Definir um **marco revisável**: reavaliação em **Q4 2029** (ou a próxima data revisável escolhida pelo time). Reflete em RF-06, AC-09, M4 e NO-01.

- **Q (D4):** Quem é o usuário primário do BitNet CPU-Universal como produto?
  **R:** **Desenvolvedores de Privacidade e Soberania de Dados.** Usuários que exigem que nenhum dado saia do dispositivo local, mas que não podem arcar com o custo de servidores GPU locais. Perfil: setores regulamentados (saúde, jurídico, financeiro) e usuários finais preocupados com privacidade que desejam rodar assistentes pessoais ou analisadores de documentos em laptops corporativos padrão ou hardware legado. **Diferencial:** a arquitetura de 1.58 bits (ternária: -1, 0, +1) elimina a necessidade de bibliotecas pesadas de CUDA, permitindo execução nativa em CPUs x86 e ARM com dependências mínimas. Adiciona seção `## 9. Persona Alvo` e impacta todos os critérios de aceitação, marketing e exemplos.

### Mudanças aplicadas

- RF-04, RF-05, RF-06 reescritos com classificações e justificativas
- AC-08 e AC-09 marcados como "bloqueador condicional" e "reserva técnica" respectivamente
- M3 e M4 atualizados com triggers e datas de reavaliação
- Nova seção `## 9. Persona Alvo` com perfil, hardware-alvo e casos de uso
- Nova seção `## 10. Esclarecimentos` (esta)
- Seção `## 11. Lacunas Residuais` (abaixo) substitui `## 9. Pendências e Dúvidas`

---

## 11. Lacunas Residuais

Após a clarificação, **não há mais dúvidas abertas**. As únicas entradas monitoradas (que podem gerar nova rodada de clarificação no futuro) são:

- **LR-01 (D2 trigger):** Monitorar se o teste empírico com Llama-2-7B é executado e qual é o resultado. Se Llama-2-7B é executado com sucesso e FFN não é bloqueador, RF-04 permanece diferencial. Caso contrário, reabrir clarificação.
  - **Status T034 (2026-06-06, Fase 5):** T029 não executado. Razões: (1) Llama-2-7B não está no ambiente de dev (~13 GB, sem GPU, sem download autorizado pelo maintainer); (2) NO-02 veda GPU; (3) P6 é reserva técnica (Q4 2029). **Decisão:** manter T009/T018/T019/T029 como pausa indefinida. RF-04 permanece "diferencial" por design. `tests/CMakeLists.txt:270-287` deixa `test_acdc_rect` opt-in via `-DBITNET_ENABLE_ACDC_RECT=ON` (default OFF) — gate é hardware-side, não código-side. Próxima reavaliação: quando mantenedor com acesso a Llama-2-7B + autorização para download de 13 GB estiver disponível.
- **LR-02 (D3 reavaliação):** No Q4 2029, reabrir clarificação sobre RF-06 (scaffolding de fine-tuning). Decidir se sobe para prioridade média, baixa definitiva, ou é removido.
- **LR-03 (D4 persona):** Se a persona alvo mudar (ex: novo mercado, nova regulamentação), reabrir clarificação. A persona atual é forte mas específica; um movimento de mercado (ex: regulamentação europeia de IA) pode exigir revisão.

---

## 12. Não-Objetivos (Out of Scope)

Para deixar o escopo claro, esta feature **NÃO** cobre:

- **NO-01**: P6 retraining real (retreino completo do BitNet com arquitetura ACDC). Só o scaffolding, e como **reserva técnica** (esclarecimento D3; reavaliação Q4 2029).
- **NO-02**: GPU kernels. Restrição fundadora do fork. A persona D4 (privacidade/soberania) reforça esta restrição: hardware GPU dedicado é incompatível com o caso de uso "laptop corporativo padrão".
- **NO-03**: Mudança no formato GGUF ou no conversor HuggingFace → GGUF.
- **NO-04**: Integração com llama.cpp upstream como dependência. Patches vendored permanecem. Compatibilidade com persona D4: dependências mínimas obrigam a minimizar cadeia de fornecedores.
- **NO-05**: Sub-features filhas. Esta é a feature-mãe; cada RF vira uma sub-feature independente com seu próprio ciclo forward (requirements → plan → to-do → coding).
- **NO-06**: Telemetria de qualquer tipo. Por padrão, o binário não envia nenhum dado a nenhum endpoint. Qualquer instrumentação nova deve ser opt-in, explícita e justificada pela persona D4.
- **NO-07**: Cloud deployment, API server, multi-tenant. Persona D4 assume uso local single-user; server-side está fora do escopo.

---

## 13. Referências Cruzadas

- **Análise reversa**: `_reversa_sdd/` (16 domain rules, 7 ADRs, 4 state machines)
- **Síntese de princípios**: `.reversa/scout/principles.md` (7 princípios transversais)
- **Mapeamento princípio→código**: `.reversa/scout/principle-code-map.json`
- **Análise de gaps**: `.reversa/scout/gap-analysis.md`
- **Findings consolidados**: `docs/findings-cpu-universal.md` (5 níveis, 4 bugs, 50 subtests, bench)
- **Histórico de sessões**: `SESSION_SUMMARY.md` (S1, S2, S2b, S2c, S2d, S2e)
- **CLAUDE.md do projeto**: `/home/peder/Projetos/BitNet/CLAUDE.md` (restrições, build, kernels)
- **Persona D4 (origem)**: `/reversa-clarify` em 2026-06-06, usuário-resposta #4

---

## 10. Não-Objetivos (Out of Scope)

Para deixar o escopo claro, esta feature **NÃO** cobre:

- **NO-01**: P6 retraining real (retreino completo do BitNet com arquitetura ACDC). Só o scaffolding.
- **NO-02**: GPU kernels. Restrição fundadora do fork.
- **NO-03**: Mudança no formato GGUF ou no conversor HuggingFace → GGUF.
- **NO-04**: Integração com llama.cpp upstream como dependência. Patches vendored permanecem.
- **NO-05**: Sub-features filhas. Esta é a feature-mãe; cada RF vira uma sub-feature independente com seu próprio ciclo forward (requirements → plan → to-do → coding).

---

## 11. Referências Cruzadas

- **Análise reversa**: `_reversa_sdd/` (16 domain rules, 7 ADRs, 4 state machines)
- **Síntese de princípios**: `.reversa/scout/principles.md` (7 princípios transversais)
- **Mapeamento princípio→código**: `.reversa/scout/principle-code-map.json`
- **Análise de gaps**: `.reversa/scout/gap-analysis.md`
- **Findings consolidados**: `docs/findings-cpu-universal.md` (5 níveis, 4 bugs, 50 subtests, bench)
- **Histórico de sessões**: `SESSION_SUMMARY.md` (S1, S2, S2b, S2c, S2d, S2e)
- **CLAUDE.md do projeto**: `/home/peder/Projetos/BitNet/CLAUDE.md` (restrições, build, kernels)

---

*requirements.md v2 — gerado por reversa-requirements + reversa-clarify em 2026-06-06*
