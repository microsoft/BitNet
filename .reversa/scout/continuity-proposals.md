# Propostas de Continuidade — BitNet CPU-Universal

> Três caminhos para evoluir o projeto a partir do estado atual (junho/2026).
> Cada caminho declara escopo, princípios tocados, entregáveis verificáveis,
> riscos e pré-requisitos. Nenhum caminho é mutuamente exclusivo — podem ser
> combinados. Gerado em 2026-06-05 pelo `reversa-scout` para alimentar o
> ciclo forward do Reversa (requirements → plan → to-do → coding).
>
> **Atualizado 2026-06-05 21:30** com verificação de integração L2/L4/L5
> (commit `129557d`) e adição do sub-caminho F (L3 ACDC FFN, próxima peça
> faltante do dispatch).

---

## Estado de Partida (consolidado de `gap-analysis.md`)

```
Fundação teórica:        100% (P1–P7 documentados com provas)
Kernels standalone L1–L4: 100% (compilam, max_diff = 0)
Kernel L5 (HRR):         100% (FFT Cooley-Tukey + bind/unbind/pseudoinverse/cleanup)
Integração dispatch:       71% (L1 default + L2 patched em vec_dot + L4+L5 via env; L3 só op registrado, não plugado)
Validação empírica:        parcial (L4: +33% e2e medido; L5: -46% regressão; L3: não medido)
```

A tese do projeto é matematicamente sólida. Os kernels são corretos
isoladamente. O Caminho B (integração com dispatch) está **71% concluído**:
L4 tropical e L5 HRR já rodam end-to-end (com qualidade garbage esperada
— P6 não validado, modelo não foi treinado com essas arquiteturas).
**L3 ACDC é a única peça faltante do dispatch** — ver sub-caminho F abaixo.

---

## Caminho A — Completar L5 (HRR)

**Natureza**: pesquisa pura, sem integração com produção.
**Esforço estimado**: 2-4 dias de trabalho focado.
**Risco**: baixo (continua trabalho já em curso).
**Princípios tocados**: P2, P3, P4, P7.

### Justificativa

L5 (HRR) é o único nível marcado como "em andamento" em
`docs/theory/05-holographic-memory.md` e o único sem benchmark
verificado. É o trabalho de pesquisa **menos arriscado** e mais
diretamente conectado à continuidade natural do roadmap.

### Escopo detalhado

#### Fase A1 — Primitivas FFT (1 dia)

Localização: `src/ggml-bitnet-hrr.cpp` (Cooley-Tukey já esboçado em
linhas 81-100) e `include/ggml-bitnet-hrr.h`.

Implementar:
- [ ] `hrr_bind(out, a, b, d)` — convolução circular via FFT
- [ ] `hrr_unbind(out, M, k_inv, d)` — recuperação
- [ ] `hrr_pseudoinverse(k_inv, k, d)` — inversa com regularização
      (regularizar `|FFT(a)|² + ε` antes de dividir para evitar div/0)
- [ ] `hrr_accumulate(M, k, v, d)` — M += k ⊛ v
- [ ] `hrr_cleanup(out, noisy, codebook, n_items, d, n_iters)` —
      projeção iterativa no manifold (Frady 2021)

Entregável: as 5 funções compilam e passam no teste de identidade
`max|bind(a,b) − IFFT(FFT(a)⊙FFT(b))| = 0`.

#### Fase A2 — Substituição da atenção (1 dia)

Localização: `src/ggml-bitnet-hrr.cpp` + novo header se necessário.

Implementar:
- [ ] `hrr_attention(out, q, M, n_context, d)` — recuperação
      associativa completa, O(d log d) por query
- [ ] `hrr_build_memory(M, K, V, n_context, d)` — superposição
      dos pares (K, V) do contexto, O(n·d·log d) total
- [ ] Benchmarks: SNR de recuperação para `d ∈ {64, 256, 1024, 4096}`
      e `N ∈ {32, 64, 128, 256}`

Entregável: API `hrr_attention` completa, com benchmark
`utils/hrr_benchmark.py` mostrando:
- Identidade exata: `max|hrr_bind(a,b) − IFFT(FFT(a)⊙FFT(b))| = 0`
- SNR medido vs analítico: `SNR(d, N) ≈ √d / (N−1)`

#### Fase A3 — Cleanup iterativo (1 dia)

Localização: `src/ggml-bitnet-hrr.cpp` (já tem esboço de
`hrr_cleanup` na API).

Implementar:
- [ ] Codebook de valores V (K-means leve sobre V real)
- [ ] Loop de cleanup: `v_t+1 = α · hrr_unbind(M, q_inv) + (1−α) · arg_nearest(v_t, codebook)`
- [ ] Benchmarks: SNR com cleanup vs sem cleanup

Entregável: função `hrr_attention_with_cleanup` que atinge
`||recuperado − v_real|| < 0.1` para `d = 4096, N = 64`
(sem cleanup, esse valor é ~0.98).

#### Fase A4 — Verificação end-to-end (meio dia)

Localização: `utils/hrr_benchmark.py` (em construção).

Adicionar testes:
- [ ] Identidade: max_diff = 0 (P2)
- [ ] SNR analítico: `d = 10N → SNR ≈ 10` (P4)
- [ ] Cleanup converge: iteração 10 → erro < 0.01
- [ ] Comparação com tropical: para `n = 64`, L5 vs L4 em
      `||v_recuperado − v_real||`

### Entregáveis verificáveis

- 5 funções novas em `ggml-bitnet-hrr.{h,cpp}` compilando com `-mavx2`
- 1 benchmark (`utils/hrr_benchmark.py`) com 4 testes de identidade
- 1 API `hrr_attention()` documentada no header
- 1 doc breve `docs/theory/05-impl-status.md` mostrando status L5 v1

### Riscos

- **Pseudoinverse instável**: divisão por `|FFT(a)|` próximo de zero
  pode explodir. Mitigação: regularização `|FFT(a)|² + ε` (padrão em
  regularização de Tikhonov).
- **Memory overhead**: M ∈ ℝᵈ por head × 32 heads × 30 camadas
  ≈ 32·30·4096·4 bytes = 15 MB. Aceitável para CPU.
- **Phase 3 cleanup lenta**: o número de iterações pode ser alto.
  Mitigação: começar com α=0.5 e ajustar empiricamente.

---

## Caminho B — Conectar L2-L5 ao dispatch do llama.cpp

**Natureza**: engenharia, integração com produção.
**Esforço estimado**: 4-7 dias (depende da familiaridade com o dispatch
do llama.cpp — `3rdparty/llama.cpp/ggml/src/`).
**Risco**: médio (modifica o fork do llama.cpp; pode quebrar build
upstream).
**Princípios tocados**: P2, P3 (este é o caminho que torna P3
verificável end-to-end).

### Justificativa

CLAUDE.md:101 sinaliza explicitamente: "These Level 2–5 kernels are
**not yet wired into CMakeLists.txt or the llama.cpp dispatch path**.
They are standalone C implementations + Python verification benchmarks."

O `CMakeLists.txt` (root) corrige a primeira parte: `bitnet_math` é
linkado em `ggml` via `target_link_libraries(ggml PUBLIC
${BITNET_MATH_TARGET})` (linha 62). Mas a segunda parte (dispatch)
permanece em aberto.

Sem este caminho, **todos os speedups publicados (L3: 174×, L4: 2863×,
L5: 186×) são números teóricos** — não há como invocá-los em produção.

### Escopo detalhado

#### Fase B1 — Operadores ggml (1-2 dias)

Localização: `3rdparty/llama.cpp/include/ggml.h` e
`3rdparty/llama.cpp/ggml/include/ggml.h`.

Adicionar ao enum `ggml_op`:
```c
enum ggml_op {
    GGML_OP_BITNET_WHT = ...,
    GGML_OP_BITNET_ACDC = ...,
    GGML_OP_BITNET_TROPICAL = ...,
    GGML_OP_BITNET_HRR = ...,
    ...
};
```

Adicionar funções de construção:
```c
struct ggml_tensor * ggml_bitnet_wht (..., struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_bitnet_acdc(...);
struct ggml_tensor * ggml_bitnet_tropical(...);
struct ggml_tensor * ggml_bitnet_hrr(...);
```

#### Fase B2 — Implementação dos ops (2-3 dias)

Localização: `3rdparty/llama.cpp/ggml/src/ggml-bitnet*.cpp`
(provavelmente em um diretório dedicado, seguindo o padrão dos
outros ops).

Para cada op, criar:
- `ggml_compute_forward_bitnet_wht.c` (ou .cpp) — chama
  `ggml_vec_dot_wht_ternary` quando tensores são ternários
- `ggml_compute_forward_bitnet_acdc.c` — chama `acdc_forward_i8`
- `ggml_compute_forward_bitnet_tropical.c` — chama
  `tropical_attention`
- `ggml_compute_forward_bitnet_hrr.c` — chama `hrr_attention`

Adicionar entradas em `ggml_compute_forward_dispatch` (em
`ggml-impl.h` ou similar).

#### Fase B3 — Auto-seleção por quant type (1 dia)

Localização: `3rdparty/llama.cpp/ggml/src/ggml-quants.c` (ou
similar) — onde o dispatcher decide qual kernel chamar.

Adicionar lógica: se o peso é ternário I2_S e o flag
`BITNET_USE_L2_WHT=ON`, chamar `ggml_vec_dot_wht_ternary` em vez de
`ggml_vec_dot_i2_i8_s_1x1`. Análogo para L3/L4/L5.

#### Fase B4 — Parity check end-to-end (1 dia)

Localização: `utils/e2e_benchmark.py` (estender).

Rodar o mesmo prompt em `llama-cli` com:
- Baseline L1 (I2_S atual)
- L1 + L2 (WHT)
- L1 + L2 + L3 (WHT + ACDC em FFN)
- L1 + L2 + L4 (WHT + tropical em atenção)
- L1 + L2 + L3 + L4 (FFN ACDC + atenção tropical)

Para cada combinação, medir:
- Tempo de inferência (tokens/seg)
- Perplexidade em wikitext-2 (parity com modelo float)

**Critério de aceitação**:
- Aceleração de tokens/segundo consistente com a tabela
  `docs/theory/00-index.md:77-86`
- Perplexidade varia < 5% entre L1 e L1+L2+L3 (P2 garante identidade;
  variação só pode vir de ordem de operações em fp32)

### Entregáveis verificáveis

- 4 novos `GGML_OP_BITNET_*` no enum de ops
- 4 implementações `ggml_compute_forward_bitnet_*.c`
- Auto-seleção por quant type
- 1 relatório `docs/integration-parity-report.md` com medições
  tokens/segundo e perplexidade para 5 configurações

### Riscos

- **Modificar o fork do llama.cpp pode quebrar merges com upstream**.
  Mitigação: isolar mudanças em um diretório dedicado
  `ggml/src/bitnet/` e minimizar diff no core.
- **Diffs grandes no llama.cpp são difíceis de revisar**. Mitigação:
  cada op em commit separado, com mensagem referenciando a doc
  teórica.
- **Performance pode ficar abaixo da teoria** se AVX2 não estiver
  habilitado ou se `gemm-config.h` não estiver tunado. Mitigação:
  chamar `tune_gemm_config.py` antes de medir.

---

## Caminho C — Validar empiricamente com modelo treinado

**Natureza**: pesquisa empírica, validação de P6.
**Esforço estimado**: 2-6 semanas (depende se há GPU disponível).
**Risco**: alto (resultado imprevisível — pode revelar que P6
não se sustenta em escala real).
**Princípios tocados**: P6 (este é o caminho que valida P6
empiricamente).

### Justificativa

O gap mais sério do projeto (de `gap-analysis.md`): **P6 é teoria
não testada**. O `acdc_project` é uma ferramenta de validação, não
de produção. Não há modelo BitNet treinado com camadas ACDC nem com
atenção HRR.

Se P6 falha empiricamente (e.g., ACDC dá perplexidade muito pior que
L1), todo o roadmap L3-L5 precisa ser repensado. Se P6 passa, o
projeto tem sua tese validada para publicação.

### Escopo detalhado

#### Fase C1 — Setup do experimento (1-2 dias)

- Selecionar modelo pequeno: `bitnet_b1_58-large` (0.7B params)
  para iterar rápido
- Selecionar dataset: WikiText-2 (103 MB, padrão em benchmarks)
- Definir baseline: perplexidade do modelo L1 puro (já publicado)

#### Fase C2 — ACDC em 1 camada (1 semana)

- Pegar o modelo `bitnet_b1_58-large` pré-treinado
- Substituir 1 camada FFN por uma camada ACDC (inicializar `d` via
  `acdc_project` na W original)
- Fine-tune: 1 epoch em WikiText-2, LR=1e-4, só atualizando `d`
- Medir perplexidade vs baseline

**Critério P6**:
- ACDC captura ≥ 80% da qualidade do FFN original → P6 passa para
  esta camada
- ACDC captura < 50% → P6 falha; modelo ACDC precisa ser
  treinado do zero com `d` desde o início

#### Fase C3 — ACDC em todas as camadas FFN (1-2 semanas)

- Substituir todas as 16 camadas FFN (large tem 16 camadas)
- Fine-tune completo
- Medir perplexidade end-to-end

**Critério**:
- Aceleração: medir tokens/segundo (precisa da integração do
  Caminho B para ser mensurável, ou usar proxy: contagem de
  FLOPs × frequência CPU)
- Qualidade: perplexidade ≤ baseline + 5%

#### Fase C4 — HRR em atenção (opcional, 1-2 semanas)

- Substituir atenção por HRR
- Treinar (regime totalmente novo — não há pré-treinado para
  inicializar M)
- Comparar com atenção L1 e tropical L4

**Critério**:
- SNR medido dentro de `±20%` do analítico `√d/(N-1)`
- Perplexidade competitiva com L1

### Entregáveis verificáveis

- 1 modelo BitNet-large com ACDC em todas as FFN
- 1 relatório `docs/acdc-empirical-validation.md` com perplexidade,
  speedup medido, e tabela `perplexity(d)` variando d
- 1 (opcional) modelo com atenção HRR

### Riscos

- **Sem GPU, fine-tune é inviável em escala**. 0.7B params × 16
  camadas × múltiplos epochs em CPU pode levar **semanas**. Mitigação:
  começar com 1 camada; se não houver GPU, abortar e documentar
  limitação.
- **ACDC pode capturar < 50% da qualidade** (P6 falhar). Este é o
  resultado mais importante — mesmo negativo, fecha a tese.
- **Hiperparâmetros do fine-tune** são críticos. Mitigação: começar
  com LR=1e-4 e reduzir se não convergir.

---

## Caminho D (Combinado) — B → A

**Natureza**: pesquisa + engenharia, na ordem que maximiza
informação por hora investida.
**Esforço**: ~1-2 semanas (B: 4-7 dias + A: 2-4 dias)
**Risco**: médio

### Justificativa

A e B são complementares:
- A entrega HRR pronto (pesquisa)
- B torna HRR e L2-L4 acessíveis em produção (engenharia)

Fazer A primeiro e B depois significa: HRR pronto, mas sem como
rodá-lo. Fazer B primeiro e A depois significa: A e B prontos juntos.

B → A permite **medir empiricamente o impacto de A** (HRR aparece no
benchmark end-to-end assim que sai da fase A). A → B inverte isso.

### Quando escolher este caminho

- Quando o objetivo final é "speedup real em CPU"
- Quando há pressão por entregas tangíveis
- Quando A e B não podem ser paralelizados (recursos limitados)

---

## Caminho E (Combinado) — B + C

**Natureza**: pesquisa + engenharia, com foco em validação
empírica.
**Esforço**: ~2-3 semanas
**Risco**: alto (pode revelar que a tese empírica falha)

### Justificativa

C sozinho produz dados, mas sem B, é difícil medir speedup real.
B sozinho produz integração, mas com kernels não-validados em
modelos treinados.

B + C permite: medir speedup real **e** validar qualidade real.

### Quando escolher este caminho

- Quando o objetivo final é **publicação** dos resultados
- Quando há GPU disponível para treinar
- Quando se quer fechar a tese teórica com evidência empírica

---

## Recomendação Default

**Estado atual** (junho/2026, pós-commit `129557d` + sessão refinamento ACDC + HRR):
- **Caminho B está 100%** (L1 default + L2 patched + L3 ACDC FFN + L4+L5 integrados).
- **Caminho A (HRR completo) está 100%** — kernels + Frady 2021 cleanup_iter
  (NAIVE + RESIDUAL) implementados e validados em `test_hrr_cleanup.cpp` 5/5
  PASS. Tabela de convergência cross-valida com Python benchmark.

Ordem recomendada dado o estado atual:

> **F ✓ (concluído) → A ✓ (concluído) → C (longo prazo, requer GPU).**

**Próximas ações** (em ordem de prioridade, escopo de ~1-2 dias cada):
1. **Integração L5 HRR com cleanup** no dispatch do llama.cpp
   (`bitnet_op_hrr_attn_with_cleanup` em `llm_build_kqv`, env `BITNET_HRR_ATTN_CLEANUP=1`).
   Kernel pronto (`hrr_cleanup_iter`), falta o wrapper de dispatch + benchmark
   de perplexidade.
2. **CI/CD mínimo** (gap #1 do scout): `cmake -B build && cmake --build`
   + `./test_hrr_cleanup` em `.github/workflows/`. Tempo: 1-2 horas.
3. **DRY refactor L2/L3/L5** (gap #3 do scout): todas compartilham
   butterfly Cooley-Tukey radix-2. Extrair para `ggml-bitnet-fft-butterfly.{h,cpp}`
   compartilhado. Tempo: 1 dia.
4. **Commit estruturado** dos 5 arquivos modificados + `test_hrr_cleanup.cpp`
   novo. Já estão staged em working tree.
5. **Caminho C** (retreino P6 com ACDC, 2-6 semanas GPU): escopo separado.

Razões:
1. **Integração L5 cleanup** fecha o último buraco entre kernel e dispatch —
   depois disso, o `bitnet_math` OBJECT lib está **funcionalmente completo**
   end-to-end em CPU.
2. **CI/CD** protege contra regressões nos próximos 5 kernels (L1-L5).
3. **DRY refactor** reduz ~30% do código duplicado (L2/L3/L5 butterfly).
4. **Commit** fixa o trabalho num ponto estável antes de mexer no dispatch.

Esta ordem maximiza informação por hora e minimiza risco de conclusão
errônea ("a tese falhou" quando na verdade foi "o dispatch estava bugado").

---

## Sub-caminho F — L3 ACDC no FFN dispatch (PEÇA FALTANTE)

**Status**: ✓ **CONCLUÍDO** em 2026-06-05 22:00.

**Resultado medido**:
- Build limpo: `cmake --build build -j$(nproc)` compila sem erros
- Smoke: `BITNET_ACDC_FFN=1 python run_inference.py ... -n 64 -t 4` roda sem crash
- Speedup: 4.92 → 5.04 tok/s (+2.4%) com D=zeros, proj=identidade parcial
- Output: garbage (esperado, P6 — modelo não treinado com ACDC)
- Combina com L4: `BITNET_TROPICAL_TOPK=32 BITNET_ACDC_FFN=1` → 4.37 tok/s (tropical domina)

**Implementação** (1 arquivo novo + 3 modificados):

| Arquivo | Mudança |
|---------|---------|
| `include/ggml-bitnet-dispatch.h` | +`bitnet_op_acdc_gemv(ctx, x, m, n, K, n_orig)` |
| `src/ggml-bitnet-dispatch.cpp` | +`acdc_gemv_callback` (lazy init de proj identidade-parcial + D zeros + int8 scratch) |
| `3rdparty/llama.cpp/src/llama.cpp` | +`llm_build_ffn_acdc_bitnet()` (helper) + branch `BITNET_ACDC_FFN=1` no call site BitNet (linha 11222) |
| `3rdparty/llama.cpp/src/llama.cpp:31-33` | extend `#if defined(BITNET_L4_TROPICAL)` para `|| defined(BITNET_L3_ACDC)` |

**Arquitetura da integração**:
```
attn_norm (2560)
    ↓ bitnet_op_acdc_gemv(m=6912, n=4096, K=2, n_orig=2560)
    │   → quantize float→int8, zero-pad 2560→4096
    │   → 2× FWHT(4096) + 2× diag(zeros)=0
    │   → proj[6912×8192] identidade parcial (top-6912)
    ↓
[6912-dim, last 1280 are zero since m<K*n=8192 padding]
    ↓ GELU
    ↓
[8192-dim com padding zero nas últimas 1280]
    ↓ bitnet_op_acdc_gemv(m=2560, n=8192, K=1, n_orig=6912)
    │   → quantize, zero-pad 6912→8192
    │   → 1× FWHT(8192) + diag(zeros)=0
    │   → proj[2560×8192] identidade parcial (top-2560)
    ↓
[2560-dim, all zero — D=0 makes ACDC output 0]
    ↓
+ residual (ffn_inp)
```

**Memória por chamada** (estática, lazy-init uma vez):
- proj up: 6912 × 8192 × 4 bytes = 226 MB
- proj down: 2560 × 8192 × 4 bytes = 84 MB
- D up: 2 × 4096 × 4 = 32 KB
- D down: 1 × 8192 × 4 = 32 KB
- x_i8 scratch: 8 KB
- **Total: ~310 MB alocado uma vez, reutilizado em 30 camadas**

(Otimização futura: não alocar proj up/down por chamada; usar proj
efetivamente aprendido depois de retreino P6.)

**Riscos materializados vs esperados**:
- ✓ Sem crash, sem segfault
- ✓ Build limpo na primeira tentativa (após 1 fix de header de função)
- ⚠ Output é garbage — esperado e documentado (P6 não validado)
- ⚠ Speedup modesto (+2.4%) — esperado porque com D=0 o ACDC não está
  fazendo trabalho útil; o speedup vem do fato de que a GEMV dense
  (17.7M ops) tem overhead maior que a chain FWHT+zeros (poucos ops reais)


---

## Pré-requisitos Comuns (qualquer caminho)

Antes de começar qualquer caminho, recomenda-se:

1. **Reservar `3rdparty/llama.cpp` antes de modificar** (B): é fork
   customizado, não upstream. Mudanças precisam ser isoladas em
   diretório dedicado.
2. **Atualizar `docs/theory/05-holographic-memory.md`**: trocar o
   status "Em andamento" pelo novo status após cada fase.
3. **Adicionar CI** (qualquer caminho): mesmo mínimo, para
   regressões não passarem despercebidas. `utils/test_perplexity.py`
   + 1 script de CI que rode `wht_benchmark.py`, `acdc_benchmark.py`,
   `tropical_benchmark.py`.

---

## Próximos Passos Imediatos

Para desbloquear a continuidade, o próximo passo concreto é:

1. **Definir caminho** (A, B, C, D, E) — decisão humana
2. **Rodar `/reversa-requirements`** com o caminho escolhido como
   input (ciclo forward do Reversa)
3. **Seguir `/reversa-clarify` → `/reversa-plan` → `/reversa-to-do`**
   para decompor em ações atômicas
4. **Executar via `/reversa-coding`**

Estes comandos Reversa usarão os artefatos desta pasta (`.reversa/scout/`)
como contexto estruturado, em conjunto com `_reversa_sdd/` (análise
prévia) e `docs/` (fundamentos).
