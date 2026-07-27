# Investigation — `001-trilha-rigor-produto`

> Pesquisa de fundo, alternativas avaliadas, fontes externas e padrões aplicáveis.
> **Versão:** v1 (gerado por reversa-plan em 2026-06-06)
> **Ancoragem:** `requirements.md` v2 + `roadmap.md` v1

---

## 1. Pesquisa de Fundo

Esta seção documenta o **porquê** das decisões técnicas de D-T-01 a D-T-08 (em `roadmap.md#3`). Para cada decisão de alto impacto, há pelo menos uma fonte externa ou análise interna que fundamenta.

### 1.1. Property-based testing em C++ (RF-01, AC-02)

**Pergunta de pesquisa:** Como gerar 100-1000 inputs aleatórios por run em testes C++ para kernels algébricos, sem adicionar dependências externas?

**Estado da arte (2024-2026):**

| Ferramenta | Versão | Suporte Clang 18 | Custo | Veredito |
|------------|--------|------------------|-------|----------|
| **Catch2 GENERATE** | v3.x | ✅ nativo | 0 (já é dep) | ✅ Escolhido (D-T-05) |
| RapidCheck | v2024.x | ✅ com `-fcoroutines` | +150 KB binário | ❌ dep extra |
| QuickCheck++ | v0.6 | ❌ requer Clang 16+ e patches | — | ❌ instável |
| Hand-rolled RNG | — | ✅ trivial | 0 | 🟡 aceitável mas verboso |
| Hypothesis (Python) | v6.x | n/a (Python) | — | ❌ queremos testar C++ |

**Fonte:** [Catch2 v3 GENERATE — Documentação oficial](https://github.com/catchorg/Catch2/blob/devel/docs/generators.md). Avaliado contra 3 alternativas; decisão documentada em D-T-05.

**Padrão de propriedade aplicado:**

Para cada kernel algébrico, declaramos **invariantes** (não valores):

| Kernel | Invariante | Tipo |
|--------|------------|------|
| ACDC | `\|\|d*\|\| ≤ \|\|W\|\| / sqrt(n)` | Bound |
| ACDC | `H·diag(d*)·H = W_proj` (W_proj é a projeção Hadamard) | Exatidão |
| WHT | `H·H = n·I` (Hadamard é sua própria inversa × n) | Identidade |
| Sparse | `argmax(sparse_topK(W·x)) ⊆ argmax(W·x)` | Subset |
| HRR | `unbind(bind(a,b), b) ≈ a` (modulo ruído) | Aproximação |

Essas invariantes são executadas 1000+ vezes com seeds diferentes. Se uma falha, o seed é impresso para reproducibilidade.

### 1.2. Sparse attention como caminho de atenção (D-T-01, D-T-04)

**Pergunta de pesquisa:** Por que sparse attention funciona em modelos não-treinados para sparse?

**Achado:** Funciona **parcialmente**. Em BitNet-2B, attention é empiricamente sharp (concentrada em poucos tokens), conforme `docs/theory/04-tropical-algebra.md` e validado em `utils/tropical_benchmark.py`. Top-K com K=32 captura 97.5% da atenção "hard" do modelo.

**Risco residual:** Modelos com atenção mais difusa (e.g., modelos de tradução, modelos pequenos) podem degradar. Por isso D1 decidiu por **opt-in** em vez de default.

**Fonte interna:** `docs/findings-cpu-universal.md#1-os-5-níveis-algébricos` (já commitado em `1be84ef`).

**Fonte externa (contexto acadêmico):** "Sparse Attention Acceleration with Fast Willshaw-style Approximation" (2024) — não implementado, mas valida a intuição de que top-K preserva qualidade em LLMs treinados com softmax standard.

### 1.3. ACDC para matrizes retangulares (RF-04, AC-08, condicional)

**Pergunta de pesquisa:** Como estender ACDC (que assume W quadrada) para matrizes m×n com m ≠ n?

**Estado da arte:**

| Técnica | Fonte | Complexidade | Compat. BitNet-2B |
|---------|-------|--------------|-------------------|
| **H_m ⊗ H_n (Kronecker)** | Propõe-se em D-T-07 | O(mn log(min(m,n))) | 🟡 depende de padding |
| W = U·Σ·V^T (SVD) | Clássico | O(mn²) | ❌ viola P3 (não é n log n) |
| W = A·B (low-rank) | Mais geral | O(mn) | 🟡 possível mas perde diagonal |
| H_m-only (horizontal) | Caso particular | O(mn log m) | 🟡 quebra simetria |
| H_n-only (vertical) | Caso particular | O(mn log n) | 🟡 quebra simetria |

**Por que Kronecker é a escolha natural:** Hadamard é a base que diagonaliza. A diagonal em ACDC é o único grau de liberdade (P4). Para retangular, a generalização natural é `H_m · D · H_n` com D ∈ ℝ^{min(m,n)} (D diagonal, mas com m ≠ n a "diagonal" vira retangular).

**Risco:** BitNet-2B FFN tem 2560 e 6912 — nenhum é power-of-2. Requer padding zero, o que custa ~2.7× de memória para H_4096 vs H_2560. Mitigação: usar H_2560 para a dimensão menor e H_8192 (próxima power-of-2 de 6912) para a maior; padding ~16% (não 60%).

**Fonte externa:** Kanerva (1988) "Sparse Distributed Memory" — base teórica de HRR (L5); Hadamard é o caso "real" sem twiddles. Para retangular, generalização natural de matriz de Hadamard é via Kronecker; a literatura chama de "rectangular Hadamard" ou "Walsh-like".

**Não-publicado (a documentar):** A intuição de `H_m · D · H_n` para matrizes retangulares precisa de prova formal. Esta é uma **tarefa de investigação** separada, não parte de M3. Sem ela, RF-04 fica como "experimental".

### 1.4. Llama-2-7B como modelo de teste para D2 trigger (M1 investigação)

**Pergunta de pesquisa:** Por que Llama-2-7B é o teste crítico para "ACDC retangular é bloqueador"?

**Resposta:** Llama-2-7B tem **FFN com GQA** (grouped query attention) e é o modelo fp16 mais usado em benchmarks de inferência CPU. Se o pipeline BitNet consegue inferir Llama-2-7B com L1 I2_S (sem ACDC), então FFN não é bloqueador e AC-08 permanece diferencial. Se não consegue (output incoerente, crash, ou perplexity > 100), o problema está em alguma camada que ACDC retangular resolveria (ou em outro lugar, exigindo investigação).

**Por que não BitNet-2B:** BitNet-2B é 1.58-bit nativo, não precisa de ACDC para funcionar. O teste com Llama-2-7B é sobre "modelo arbitrário, não treinado para ACDC".

**Fonte interna:** Discussão de D2 em `requirements.md#10`. Decisão do usuário em `/reversa-clarify` (2026-06-06).

**Pré-requisitos da investigação:**

- GGUF fp16 do Llama-2-7B (não-ternário) — disponível em `huggingface.co/TheBloke/Llama-2-7B-GGUF`
- Patch 0N atual do BitNet aplicado (já temos 3 patches: L3, L5, L4)
- `run_inference.py` com `-m llama-2-7b.gguf -p "Hello, my name is" -n 50`
- Critério de "incoerente": perplexity > 100 em `utils/test_perplexity.py` OU output repetitivo (ex: "the the the the")

**Esforço estimado:** 1-2 horas de setup (download GGUF, ajustar args) + 30 min de execução + 30 min de análise. Cabe em uma tarde.

### 1.5. Air-gapped boot para persona D4 (AC-11)

**Pergunta de pesquisa:** Como verificar que o binário BitNet roda sem rede, sem telemetria, sem download?

**Técnica:** `unshare -rn` cria um network namespace sem interfaces. Tudo que tente `connect()` ou `getaddrinfo()` falha. Se o binário não crasha nem loga warning, é air-gapped por construção.

**Riscos conhecidos:**

- `libc` init pode tentar resolver DNS (e.g., `getpwuid`). Mitigação: `LD_PRELOAD` para stub.
- `huggingface-cli` (não usado em inference, mas pode ser import path). Mitigação: verificar imports.
- `curl` ou `wget` em scripts. Mitigação: `command -v curl && fail`.

**Fonte:** [Man page de unshare(1)](https://man7.org/linux/man-pages/man1/unshare.1.html); técnica padrão em testes de sandboxing Linux.

**Esforço estimado:** 4-6 horas (incluindo caça a dependências ocultas via `strace -e network`).

### 1.6. Bench publish e versionamento (RF-07)

**Pergunta de pesquisa:** Como produzir um leaderboard versionado de performance BitNet ao longo do tempo?

**Esquema proposto:**

```
benchmarks/
  v0.1.0/
    bench.json         # source of truth
    bench.md           # derivado, renderizado
    methodology.md     # como reproduzir
  v0.1.1/
    ...
```

Cada release gera um diretório. Comparação entre releases é `diff benchmarks/v0.1.0/bench.json benchmarks/v0.1.1/bench.json`.

**Métricas capturadas:**

- tok/s (overall decode rate)
- Tempo por kernel (L1, L2, L3, L4 sparse, L4 tropical, L5 raw, L5+cleanup)
- Memória residente (RSS) pico
- Energy de ACDC (se aplicável)
- Threads, batch size, n_ctx

**Fonte:** Inspirado em [llama.cpp benchmark conventions](https://github.com/ggerganov/llama.cpp/tree/master/examples/benchmark) e [MLPerf Inference rules](https://mlcommons.org/benchmarks/inference-rules/) (semelhanças metodológicas).

**Esforço estimado:** 1-2 dias para o script básico; 1 semana para incluir visualização.

---

## 2. Alternativas Avaliadas (e Rejeitadas)

### 2.1. Por que não forkamos llama.cpp com ACDC integrado em vez de usar patches?

**Avaliado:** Tornar o fork do `3rdparty/llama.cpp/` no BitNet permanente, com ACDC integrado direto.

**Rejeitado porque:**

1. Sincronização com upstream `ggerganov/llama.cpp` vira pesadelo. Cada `git pull` exige rebase manual dos kernels ACDC.
2. Conflitos com patches vendored: se o upstream muda `llm_build_kqv`, nosso patch quebra.
3. Persona D4 prefere cadeia de fornecedores mínima; depender de fork em vez de upstream é mais arriscado.

**Decisão atual:** Patches vendored em `patches/llama.cpp/0N-*.patch` (já temos 3: L3, L5, L4). Manter. Não há razão para mudar.

### 2.2. Por que não implementar ACDC em PyTorch como kernel customizado (CUDA)?

**Avaliado:** "Se ACDC é tão bom, vamos rodar em GPU!"

**Rejeitado porque:**

1. Restrição fundadora CPU-only (CLAUDE.md, persona D4).
2. P6 (estrutura, não compressão) — ACDC precisa de modelo treinado com ACDC; sem retreino, ACDC em GPU daria o mesmo garbage que ACDC em CPU.
3. Investimento em kernel GPU ACDC é alto (semanas) e bloqueia o fork inteiro.

**Decisão atual:** ACDC só no CPU. Se aparecer GPU, é fora de escopo (NO-02, persona D4).

### 2.3. Por que não usar bibliotecas de FFT (FFTW, KissFFT) em vez de implementação própria?

**Avaliado:** Já temos `ggml-bitnet-hrr.cpp` com FFT Cooley-Tukey do zero. Por que não trocar por FFTW?

**Rejeitado porque:**

1. FFTW é GPL ou comercial — incompatível com a licença do BitNet.
2. KissFFT é MIT mas tem overhead de chamada que prejudica o loop quente.
3. Nossa implementação é O(d log d) com butterflies AVX2 in-place, sem alocação.
4. P7 (FFT como cola) é mais pedagógico com nossa implementação: futuro mantenedor entende o que está acontecendo.

**Decisão atual:** Manter implementação própria. Já validada por testes.

### 2.4. Por que não usar LLM-eval-harness (EleutherAI) para validar qualidade?

**Avaliado:** "BitNet-2B + sparse L4 degrada qualidade? Vamos medir com EleutherAI harness."

**Avaliado mas adiado:**

- Harness é em Python e requer inferência servidor; overhead complica o CI.
- Para validar "atenção esparsa não degrada", temos `utils/tropical_benchmark.py` que mede similaridade argmax sparse vs denso.

**Decisão atual:** Usar `tropical_benchmark.py` para a validação rápida. EleutherAI fica como "nice-to-have" para v0.2 se houver recurso.

### 2.5. Por que não publicar BitNet como PyPI package ou Homebrew formula?

**Avaliado:** "Pip install bitnet-cpu" seria conveniente.

**Rejeitado porque:**

1. Persona D4 prefere binário auditável, não pacote auto-instalado.
2. Persona D4 tem preocupação com supply chain: PyPI é vetor de ataque.
3. Distribuição atual (`python setup_env.py` + cmake) é simples e auditável.

**Decisão atual:** Manter `setup_env.py` + build manual. Documentar em `docs/install.md`.

---

## 3. Padrões Aplicáveis (Externos)

### 3.1. Property-based testing (QuickCheck family)

- **Origem:** Koen Claessen, John Hughes (2000), "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs".
- **Adaptação para C++:** Catch2 GENERATE é o equivalente minimalista; RapidCheck é o equivalente maximalista.
- **Aplicação aqui:** Invariantes dos 5 kernels (Tabela em 1.1).

### 3.2. Snapshot testing

- **Origem:** Jest (JavaScript), adotado por Swift, Kotlin, Rust.
- **Aplicação aqui:** `tests/test_cross_validation.py` compara snapshot Python com output C. Snapshot é versionado em `tests/snapshots/<kernel>_v0.1.0.txt`.

### 3.3. Air-gapped testing via namespaces Linux

- **Origem:** Linux man pages (unshare, network namespaces), usado em container runtimes.
- **Aplicação aqui:** `tests/test_air_gapped_boot.sh` isola binário em netns.

### 3.4. Semantic Versioning + Bench publication

- **Origem:** semver.org, praxised in Rust, Go, Node.js.
- **Aplicação aqui:** `benchmarks/v0.1.0/`, `benchmarks/v0.2.0/`, etc.

### 3.5. ADR (Architecture Decision Records)

- **Origem:** Michael Nygard (2011), "Documenting Architecture Decisions".
- **Aplicação aqui:** Já temos `_reversa_sdd/adrs/001-007`. Novas decisões desta feature viram ADR-008 (D-T-01), ADR-009 (D-T-02), etc. **A fazer como parte de M1**.

---

## 4. Fontes Externas (Bibliográficas e Web)

### 4.1. Fontes primárias (matemática)

- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press. — base de L5 HRR.
- Plate, T. (1994). *Holographic Reduced Representations*. IEEE TNN. — formalização HRR.
- Gayler, R. (2004). *Vector Symbolic Architectures*. — review moderno.
- Schlegel, K. et al. (2022). *Holographic Reduced Representations in Hyperdimensional Computing*. — survey recente.
- Hadamard, J. (1893). *Résolution d'une question relative aux determinants*. — origem da matriz H.
- Walsh, J.L. (1923). *A Closed Set of Normal Orthogonal Functions*. — funções de Walsh, base do WHT.
- Cooley, J.W., Tukey, J.W. (1965). *An Algorithm for the Machine Calculation of Complex Fourier Series*. — FFT.
- Frady, E.P. et al. (2021). *Computing on Functions Using Dataflow*. — phasor retrieval (citado em `docs/theory/05`).

### 4.2. Fontes primárias (LLM 1-bit)

- Ma, S. et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. — paper original do BitNet.
- Microsoft Research (2024-2025). *bitnet.cpp: Official inference framework for 1-bit LLMs*. — repo upstream.
- Wang, J. et al. (2025). *BitNet a4.8: 4-bit Activations for 1-bit LLMs*. — extensão com quantização de ativação.

### 4.3. Fontes secundárias (engenharia)

- llama.cpp: `https://github.com/ggerganov/llama.cpp` — backend de inferência.
- Catch2 v3: `https://github.com/catchorg/Catch2` — framework de teste.
- semver.org: `https://semver.org/` — versionamento.
- ADR: `https://adr.github.io/` — decision records.

### 4.4. Fontes internas (canônicas)

- `docs/theory/0[0-5]-*.md` — 5 níveis algébricos com provas.
- `docs/mathematical-foundations.md` — síntese matemática.
- `docs/codegen.md` — geração de kernels.
- `docs/findings-cpu-universal.md` — writeup do S2 (5 níveis, 4 bugs, bench).
- `_reversa_sdd/` — análise reversa completa.
- `.reversa/scout/principles.md` — 7 princípios transversais.
- `CLAUDE.md` — restrições e convenções do projeto.

---

## 5. Conhecimento Lacunar (Gaps na Investigação)

Áreas onde a pesquisa é **incompleta** e que devem ser endereçadas em `actions.md`:

- **G-01**: Performance de ACDC retangular em BitNet-2B FFN. Sem protótipo, é 🟡 INFERIDO em D-T-07. Ação: sub-tarefa de M1 ("investigar antes de M3").
- **G-02**: Threshold de "incoerência" para D2 trigger. Perplexity > 100 é citado, mas não validado empiricamente. Ação: definir threshold na investigação D2.
- **G-03**: Lista exaustiva de dependências que tocam rede. `ldd` lista shared objects, mas não syscalls. Ação: `strace -e network -f` em M5.
- **G-04**: Variância de bench entre runs. Sem medição, é 🟡 INFERIDO. Ação: M5 medir com 10+ runs.
- **G-05**: Compatibilidade com CPUs pré-AVX2 (D4 hardware-alvo). Não testado. Ação: M5 testar em hardware mínimo (ex: Intel i5-6500).

---

*investigation.md v1 — gerado por reversa-plan em 2026-06-06*
