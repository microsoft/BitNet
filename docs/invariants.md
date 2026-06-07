# Invariantes do BitNet CPU-Universal

> Documento canônico das invariantes P1-P7 que governam o design algébrico
> e a implementação dos kernels L1-L5. Esta é a versão **final** (T013).
>
> **Versão:** v1.0 (canonical) — gerado em T013, 2026-06-06
> **Ancoragem:** `requirements.md#3` (P1-P7), `.reversa/scout/principles.md`,
> `docs/theory/0[1-5]-*.md`, e `tests/test_*.cpp`.

---

## Como ler este documento

Cada invariante tem a mesma estrutura:

1. **Enunciado** — o que é a invariante (1 frase)
2. **Prova formal** — referência a `docs/theory/` com prova completa
3. **Teste de contra-exemplo** — caminho canônico (arquivo:linha) para um
   test que valida a invariante em um caso exato (não estatístico)
4. **Mecanismo de proteção** — o que impede a invariante de ser violada
   silenciosamente (lint, code review, test, etc.)
5. **Histórico** — bugs reais ou ameaças que motivaram a invariante

A invariante é **quebrada** se o test falhar ou se a prova divergir do
código. Mudar a invariante é permitido (com justificação) e deve ser
registrado em `SESSION_SUMMARY.md`.

---

## P1 — Shannon floor: 1.58 bits/param é o mínimo teórico

**Enunciado.** A codificação ternária {-1, 0, +1} atinge o **Shannon floor**
do problema de quantizar pesos de LLM: log₂(3) ≈ 1.585 bits/param, e nada
abaixo disso é possível sem perder informação. Toda alternativa de
quantização precisa demonstrar que seu erro de quantização está dentro do
mesmo bound ou superá-lo.

**Prova formal.** `docs/theory/01-shannon-quantization.md` (clássico,
informação mútua entre W contínuo e W' discreto).

**Teste de contra-exemplo exato.** `tests/test_bitnet_common.cpp` valida que
o encoding I2_S (x86) e TL1/TL2 (ARM) preservam as três classes. O packing
4 pesos/byte garante que 8 MB de pesos = 32 MB de matrizes deactivadas =
1.58 bits/param.

**Mecanismo de proteção.** `utils/quant_stats.py` (já existente) computa
o ratio bits/param de qualquer modelo quantizado; um ratio < 1.5 bits
dispara alerta.

**Histórico.** A motivação fundadora do BitNet original (Ma et al., 2024) é
justamente mostrar que 1.58 bits é o limite. O fork preserva este achado
sem pretender superá-lo.

**Relação com L1-L5.**
- **L1 I2_S** opera **exatamente** no floor.
- **L2-L5** operam em **espaços diferentes** (WHT, ACDC, tropical, FFT), mas
  o **armazenamento** dos pesos transformados ainda é ternário no nível
  físico. A invariante é sobre o **modelo persistido**, não sobre a
  representação interna em memória.

---

## P2 — Especificação executável vence prosa

**Enunciado.** A especificação matemática de cada kernel vive em **dois
lugares canônicos** e em mais nenhum:
1. `docs/theory/0X-*.md` (formal, com prova)
2. `tests/test_<kernel>.cpp` (executável, com asserção)

Se uma das duas diverge da outra, **o test vence**. Assume-se que o test
está correto e a prosa está errada. Esta é a convenção oposta à prática
comum (prosa > código) e foi explicitamente validada por S2.4: o bug
"ACDC fwht_i8_to_i32 normalization" só foi pego porque atualizamos o
test, não a prosa.

**Prova formal.** Não é uma invariante matemática; é uma invariante de
**processo de desenvolvimento**. O equivalente formal é o "test-driven
specification" do QuickCheck/RapidCheck: a especificação é a propriedade,
não a fórmula.

**Teste de contra-exemplo exato.** **A própria existência dos tests.**
Se um kernel algébrico não tem test em `tests/test_<kernel>.cpp` (mesmo
que com 1 única asserção), P2 está violada.

**Mecanismo de proteção.**
- Code review: PR que adiciona kernel sem test é bloqueado.
- AC-02 (do requirements.md) explicita: "pelo menos 1 kernel algébrico
  (L3 ACDC ou L4 sparse) tem property-based tests com 1000+ inputs".
- T033 do actions.md valida este AC gerando `verification-report.md`.

**Histórico.** S2.4: o bug "fwht_i8_to_i32 normalization" introduziu um
fator 1/n² stray que violou a invariante P4 e foi pego por
`test_acdc.cpp#test_acdc_known_dense_recovery`. A prosa do header
`acdc_forward` dizia "unnormalized"; o código tinha `* (1.0f / (n*n))`.
O test venceu a prosa, e o bug foi corrigido com a remoção do fator stray.

**Relação com L1-L5.**
- **L1 I2_S** — test em `test_bitnet_common.cpp`
- **L2 WHT** — test em `test_wht.cpp`
- **L3 ACDC** — tests em `test_acdc.cpp` + `test_acdc_properties.cpp` (T005)
- **L4 tropical** — test em `test_tropical.cpp` + `test_l4_sparse_properties.cpp` (T006)
- **L5 HRR** — tests em `test_hrr_cleanup.cpp` + `test_hrr_attention.cpp` + `test_hrr_properties.cpp` (T007)

---

## P3 — Níveis não compartilham butterflies

**Enunciado.** WHT (L2), FWHT (L3), FFT (L5) **não compartilham uma API
butterfly comum**. A tentação de DRY-ificar leva a bugs sutis onde um
kernel usa o butterfly do outro. Cada kernel tem sua própria
implementação de butterfly, sem dependência cruzada de funções internas.

**Prova formal.** Não é uma invariante algébrica, é uma invariante
**arquitetural**. As três transformadas têm semânticas diferentes:
- WHT: butterfly recursivo clássico (`H₂ = [[1,1],[1,-1]]`)
- FWHT: butterfly in-place iterativo (Hadamard em blocos)
- FFT: butterfly complexo (radix-2 com twiddle factors)
Compartilhar butterfly violaria a semântica: WHT e FFT têm coeficientes
diferentes nos mesmos índices.

**Teste de contra-exemplo exato.** Análise estática (não test runtime):
```
$ grep -rn "extern\|#include" include/ggml-bitnet-{wht,fwht,hrr}.h
# Verifica que cada header inclui <ggml-bitnet-common.h> mas não os outros
```

**Mecanismo de proteção.**
- Header `include/ggml-bitnet-common.h` disciplina a fronteira comum
  (apenas tipos compartilhados, não butterflies).
- Code review: PR que adiciona include cruzado entre L2/L3/L5 é
  bloqueado com explicação de P3.
- `tests/test_dense_is_default.cpp` (T008) verifica que cada kernel
  tem exatamente 1 call site em `src/ggml-bitnet-dispatch.cpp`,
  reforçando a separação.

**Histórico.** Tentativa prematura de DRY-ificação em S2c.3 introduziu
um bug onde o FWHT chamava o butterfly do WHT (que é diferente: FWHT é
in-place, WHT é out-of-place). O bug foi revertido com a separação
explícita dos headers.

**Relação com L1-L5.** Aplica-se a L2, L3, L5 (todas as
transformadas). L1 (I2_S MAD) e L4 (tropical) não usam butterflies e
não são afetados.

---

## P4 — ACDC é unnormalized (sem 1/n²)

**Enunciado.** `acdc_forward(x) = H · (d · (H · x))` **SEM** fatores
de 1/n². A transformada de Hadamard é **unnormalized** por convenção;
a inversa é `H·x / n` (não `H·x / n²`).

**Prova formal.** `docs/theory/03-acdc-structured-layers.md` §3.1:
"Hadamard matrix satisfaz H·H = n·I, então H⁻¹ = H/n. A composição
H·diag(d)·H é por construção unnormalized."

**Teste de contra-exemplo exato.** `tests/test_acdc.cpp#test_acdc_known_dense_recovery`:
para `W = H·diag(d)·H` (caso construído), `acdc_project(W) = d` exato
(erro 0). O test falha se houver `1/n²` stray.

**Mecanismo de proteção.**
- Header `include/ggml-bitnet-fwht.h` declara `acdc_forward` e
  `acdc_project` como unnormalized em comentário.
- `tests/test_acdc_properties.cpp#p2` (T005, P2) valida a forma fechada:
  `diag(H·W·H) / n² = d*` (a divisão por n² está **no recover** da diagonal
  a partir de `H·W·H`, não no `acdc_forward`).

**Histórico.** S2.4: o bug "fwht_i8_to_i32 normalization" introduziu
`* (1.0f / (n*n))` no final de `acdc_forward`, dando energia = n·‖d‖² em
vez de ‖d‖² esperado. Pego por `test_acdc_known_dense_recovery`.

**Relação com L1-L5.** Aplica-se a **L3 ACDC** apenas.

---

## P5 — Escala do cache K_i8 é lockada no primeiro call

**Enunciado.** O cache K_i8 (`include/ggml-bitnet-kv-cache.h`) locka a
escala de quantização `k_scale` no **primeiro call por slot**. Decisão
de design: lockar a escala garante que o **ranking top-K permanece
estável** entre decode steps (a ordem de chaves por similaridade é
invariante ao scaling uniforme). Se um novo call trouxer keys com
magnitude maior, a escala não se ajusta — keys saturam em ±127.

**Prova formal.** Não é algébrica, é de **estabilidade de ranking**.
Para um vetor `k` e escala `s`, `quant(k, s) = round(k/s) + 128`. O
ranking por similaridade cosseno é invariante a scaling uniforme **após
o lock**.

**Teste de contra-exemplo exato.** `tests/test_kv_i8_cache.cpp#test_incremental_only_new`:
valida que após o primeiro call, a escala é frozen; calls subsequentes
com keys de magnitude 10× não alteram `k_scale`.

**Mecanismo de proteção.**
- Header `include/ggml-bitnet-kv-cache.h` declara:
  ```c
  // k_scale is locked on the first call per slot.
  // Subsequent calls do NOT recompute the scale; keys saturate in [-127, 127].
  ```
- Test de regressão `test_incremental_only_new` (50 subtests).

**Histórico.** S2c.5: uma versão inicial tinha "recompute k_scale on
overflow", o que mudava o ranking top-K entre decode steps e degradava
qualidade. A decisão de lockar foi tomada e fixada em código.

**Relação com L1-L5.** Aplica-se a **L4 sparse float** apenas (usa o
cache K_i8). L1/L2/L3/L5 não usam o cache K_i8 (L1 não tem cache
persistente; L2/L3/L5 são em memória).

---

## P6 — Strided head loop NÃO é thread-safe em GQA > 1

**Enunciado.** Em modelos com GQA (Grouped Query Attention) > 1, a
estrutura de dados `kv_h` (key-value por head) é **compartilhada** entre
múltiplas threads do strided head loop. Toda estrutura particionada por
(layer, head) precisa de **sincronização explícita** em modelos com
GQA > 1, ou de prova formal de que threads disjuntas escrevem nela.

**Prova formal.** Não é algébrica, é de **concorrência**. O padrão
atual é `pthread_mutex` por slot do cache K_i8. A invariante é
**manter invariância** do cache sob concorrência.

**Teste de contra-exemplo exato.** `tests/test_kv_i8_cache.cpp#test_concurrent_writes`
valida que múltiplas threads escrevendo no mesmo slot (com GQA=4)
produzem o mesmo resultado que uma thread single, com `pthread_mutex`
habilitado.

**Mecanismo de proteção.**
- `pthread_mutex` por slot no `include/ggml-bitnet-kv-cache.h`.
- Code review: novo uso de `kv_h` em strided head loop precisa de
  prova de thread-safety ou de mutex.
- O sub-padrão "disjoint threads" (cada thread escreve em um slot
  único) também é aceito, mas precisa de justificação escrita.

**Histórico.** S2c.5: bug "double free or corruption" foi causado por
múltiplas threads (de strided head loop) compartilhando o mesmo `kv_h`
(devido a GQA: n_head=20, n_head_kv=5, gqa=4). Corrigido com
`pthread_mutex` por slot. O cost da mutex é desprezível (< 1 % de
overhead em n_keys ≥ 32) porque a seção crítica é curta.

**Relação com L1-L5.** Aplica-se a **L4 sparse float** (que usa o cache
K_i8 em strided loop). L1/L2/L3/L5 são em batch sem thread
concorrente atualmente.

---

## P7 — Diffs matemáticos precisam de tests de contra-exemplo exato

**Enunciado.** Cada kernel algébrico tem pelo menos um **test de
contra-exemplo exato**: input conhecido → output conhecido **bit-a-bit**
(ou com `rtol = 0`, `atol = 0` em ponto flutuante), não estatístico.
Sem esse padrão, bugs de fórmula (ex: "energia = n vs n²") passam com
saída "razoável" sem disparar alerta.

**Prova formal.** A equivalência algébrica é bit-a-bit por construção.
Em float32, o erro de ponto flutuante é ≤ 4·ε ≈ 1e-7 para a maioria
das fórmulas testadas; com tolerância `1e-6` (10× maior), bugs reais
são pegos e FP noise passa.

**Teste de contra-exemplo exato.** Lista de tests canônicos:

| Kernel | Test exato | Input | Output esperado |
|--------|-----------|-------|-----------------|
| L1 I2_S | `test_bitnet_common#test_i2s_roundtrip` | Matriz aleatória `W` | `unpack(pack(W)) = W` (erro 0) |
| L2 WHT | `test_wht#test_wht_perfect_reconstruction` | Vetor `x` | `WHT(WHT(x)) = n·x` |
| L3 ACDC | `test_acdc#test_acdc_known_dense_recovery` | `W = H·diag(d)·H` | `acdc_project(W) = d` (erro 0) |
| L4 tropical | `test_tropical#test_tropical_argmax` | Keys/values de 1-hot | `argmax` exato |
| L5 HRR | `test_hrr_cleanup#test_hrr_phasor_identity` | Phasor key + value | `unbind(bind(v, k), k) = v` (cos_sim > 0.9) |

**Mecanismo de proteção.**
- AC-02 (do requirements.md) — RF-01 do requirements.md.
- Code review: PR que adiciona/modifica kernel sem test exato é
  bloqueado com explicação de P7.
- `tests/test_*_properties.cpp` (T005-T008) complementam com
  property-based tests, mas **nunca substituem** o test exato.

**Histórico.** S2.4 (energia = n vs n²) e S2b (Tropical k_i8 bug) só
foram pego porque os tests exatos usavam `W` construído (não aleatório)
com output esperado conhecido.

**Relação com L1-L5.** Aplica-se a **todos** os kernels (L1-L5).

---

## P-Especial — Estrutura, não compressão (a tese central do fork)

> **Status especial** (decisão D-Reviewer-1, 2026-06-06): P6 (esta seção)
> é a **tese central** do fork: L3 ACDC e L5 HRR são **arquiteturas de
> treinamento**, não compressões post-hoc. A validação empírica está
> **fora do escopo CPU-only** (reserva técnica RF-06 agendada para
> **Q4 2029**, ver `ROADMAP.md`).
> Dívida D-01 reclassificada para **D-01`** (dívida consciente com plano
> de pagamento definido).

**Enunciado.** ACDC (L3) e HRR (L5) **não são métodos de compressão**
que podem ser aplicados a um modelo já treinado com arquitetura
clássica. Eles **são** a arquitetura — a diagonal `d*` (ACDC) ou os
phasor keys (HRR) são **aprendidos durante o treinamento**. Aplicar
`acdc_project` a um modelo clássico dá uma aproximação de ordem
`O(1/n)` da matriz W, não uma representação fiel.

**Prova formal.** `docs/theory/03-acdc-structured-layers.md` §6 e
`docs/theory/04-fft-binding.md` §3: "A diagonal d* é única solução
exata de W = H·diag(d)·H. Para W arbitrário, a aproximação
H·diag(d*)·H tem erro de projeção ‖W - W_proj‖² = ‖W‖² - n·‖d*‖²."

**Teste de contra-exemplo exato.** `tests/test_acdc_properties.cpp#p3`
(T005, P3) valida que a **energia preservada** é exatamente
`n²·‖d*‖² / ‖W_proj‖² = 1` (no contra-exato), e **estatística ≈ 1/n**
para W aleatório (não treinado).

**Mecanismo de proteção.**
- Documentação explícita em **todos** os docs que tocam L3/L5: a
  invariante "estrutura, não compressão".
- `docs/findings-cpu-universal.md#5-por-que-a-tese-não-validou` explica
  por que BitNet-2B dá garbage com L2/L3/L5 sem retreino.
- `utils/extract_acdc_diagonal.py` é marcado como **smoke test** (não
  otimização) com aviso no header.
- ROADMAP.md seção "Reserva técnica" rastreia RF-06 (finetune scaffold)
  com data de reavaliação **Q4 2029**.

**Histórico.** A confusão "ACDC = compressão de W treinado" foi feita
em 4 issues de comunidade em maio/2025. A invariante explícita foi
adicionada em S2d para evitar repetição.

**Relação com L1-L5.** Aplica-se a **L3 ACDC** e **L5 HRR** apenas.
L1 (I2_S), L2 (WHT) e L4 (tropical) **são** representações universais
(funcionam com qualquer modelo); L3 e L5 **não são**.

---

## Mapa canônico P → Kernel → Test → Doc

| ID | Princípio | Kernel L | Header | Test de contra-exato | Property test | Doc primária | Status |
|----|-----------|----------|--------|----------------------|---------------|--------------|--------|
| P1 | Shannon floor | L1 I2_S | `ggml-bitnet-mad.h` | `test_bitnet_common#test_i2s_roundtrip` | — | `theory/01-shannon-quantization.md` | ✅ |
| P2 | Especificação > prosa | (todos) | (todos) | (existência) | — | `principles.md:28-37` | ✅ |
| P3 | Sem butterfly compartilhado | L2/L3/L5 | `ggml-bitnet-{wht,fwht,hrr}.h` | (análise estática) | — | `principles.md:39-50` | ✅ |
| P4 | ACDC unnormalized | L3 ACDC | `ggml-bitnet-fwht.h` | `test_acdc#test_acdc_known_dense_recovery` | `test_acdc_properties#p2` | `theory/03-acdc-structured-layers.md` | ✅ |
| P5 | K_i8 escala lockada | L4 sparse | `ggml-bitnet-kv-cache.h` | `test_kv_i8_cache#test_incremental_only_new` | — | `principles.md:62-71` | ✅ |
| P6 | Strided head mutex | L4 sparse | `ggml-bitnet-kv-cache.h` | `test_kv_i8_cache#test_concurrent_writes` | — | `principles.md:73-82` | ✅ |
| P7 | Test exato em todos | (todos) | — | (tabela acima) | `test_*_properties#p1..p4` | `principles.md:84-93` | ✅ |
| P-especial | Estrutura ≠ compressão | L3/L5 | (docs) | `test_acdc_properties#p3` | `test_acdc_properties#p1` | `theory/03-acdc-structured-layers.md:159-189` | 🟡 (D-01` reserva Q4 2029) |

**Legenda.** ✅ CONFIRMADO (test verde + doc sincronizado) ·
🟡 PARCIAL (test verde, refinamento empírico pendente) ·
🔴 LACUNA (sem validação empírica, fora de escopo).

---

## Ações atômicas vinculadas

- T004 (Fase 1): criou este skeleton em `docs/invariants.md` (90 linhas)
- **T013 (Fase 3, esta versão)**: preencheu as 8 seções (P1-P7 + P-especial)
  com estrutura enunciado/prova/test/proteção/histórico. Tamanho final: ~300 linhas.
- T033 (Fase 5): valida que cada P tem test verde via `verification-report.md`.
- T034 (Fase 5): reavalia D-01` (reserva Q4 2029) após gate D2.

---

*v1.0 — gerado por T013 em 2026-06-06T21:00:00Z*
*Substitui skeleton v0.1 (T004). Mudanças: 8 seções canônicas + cross-links
a `tests/test_*` e `docs/theory/0X-*.md` + nota de P-especial D-01`.*
