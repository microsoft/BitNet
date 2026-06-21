# Próximos Passos — BitNet CPU-Universal

> **Data:** 2026-06-09
> **Estado atual:** v0.1 completo — 13/13 ACs ✅, M1/M2/M5 ✅, T029 ✅
> **Fundamentos:** ctest 15/15 | cross-val 3/3 | property tests 10/10 | air-gapped ✅
> **Referências:** `ROADMAP.md`, `verification-report.md` v2.0, `investigation-d2-result.md`

---

## Passo 1 — Release v0.1.0 (imediato)

**O que fazer:**

```bash
# 1. Push dos 6 commits locais
git push origin main

# 2. Tag de release
git tag -a v0.1.0 -m "BitNet CPU-Universal v0.1.0

Inferência 1.58-bit local-first — L1-L5 algébrico, persona D4.

Destaques:
- 5 níveis algébricos: WHT (L2), ACDC (L3), tropical/sparse (L4), HRR (L5)
- BITNET_ACDC_FFN_RECT=auto: Falcon3-10B +179%, Falcon3-3B +51.7%
- BITNET_SPARSE_TOPK_ADAPTIVE=0.90: BitNet-2B +14.9%
- Air-gapped boot (unshare -rn) ✓
- 15/15 ctest, 13/13 ACs verificados
- Persona D4: médico/jurídico/financeiro offline (examples/)
"
git push origin v0.1.0
```

**Critério de done:** tag visível em `github.com/<user>/BitNet/releases`.

---

## Passo 2 — PR upstream `microsoft/BitNet` (curto prazo)

**O que fazer:**

1. Criar fork público de `microsoft/BitNet` no GitHub
2. Branch `feature/cpu-universal-l2-l5`
3. Cherry-pick seletivo dos commits L1-L5 (excluir commits de doc D4 privados)
4. Abrir PR com:
   - Título: `[RFC] CPU-only algebraic dispatch: WHT/ACDC/tropical/HRR (L2-L5)`
   - Body: link para `docs/findings-cpu-universal.md`, tabela de speedups, benchmark
   - Labels: `enhancement`, `performance`, `cpu-only`

**Candidatos a cherry-pick** (commits que fazem sentido upstream):
| Commit | O que leva upstream |
|--------|-------------------|
| ACDC FFN dispatch | `src/ggml-bitnet-fwht.cpp` + patch |
| adaptive-K sparse | `src/ggml-bitnet-tropical.cpp` + patch |
| HRR phasor | `src/ggml-bitnet-hrr.cpp` + patch |
| test suite L3/L4/L5 | `tests/test_acdc_properties.cpp` etc. |

**O que NÃO vai upstream:** docs D4 privados, `examples/medical_offline.md` etc., `investigation-d2-result.md`.

---

## Passo 3 — Benchmarks em mais hardware (curto prazo, Q3 2026)

**Motivação:** os resultados atuais são de um único hardware (Intel i5-10210U).
Para upstream e para a persona D4 ser credível, precisamos de:

| Hardware | Por quê |
|----------|---------|
| ARM64 (Apple M1/M2 ou Raspberry Pi 5) | NEON path, persona D4 mobile |
| AMD Ryzen (Zen 3+, AVX2) | Desktop corporativo padrão |
| Intel Celeron / Atom (hardware legado) | Caso extremo da persona D4 |
| Windows 11 (WSL2 e nativo) | Ambiente corporativo mais comum |

**Como:** `utils/cpu_universal_benchmark.py --model <gguf> --n 128 --threads <nproc> --keep-running`
Gravar output em `benchmarks/v0.1.x/bench-<hw>.json` e commitar.

---

## Passo 4 — ARM64 NEON path (médio prazo, Q4 2026)

**Contexto:** os kernels L3/L4/L5 têm path AVX2 validado. ARM64 (NEON)
usa fallback escalar — funciona, mas não é otimizado.

**O que fazer:**
- `src/ggml-bitnet-fwht.cpp`: adicionar `#ifdef __ARM_NEON` com intrinsics NEON para WHT/ACDC
- `src/ggml-bitnet-tropical.cpp`: NEON path para sparse_attention_float
- Testar em Raspberry Pi 5 ou Apple M1 via CI cross-compile

**Estimativa:** 3-5 dias (similar ao trabalho AVX2 original).

**Gate:** hardware ARM64 disponível para teste.

---

## Passo 5 — Refinamento de thresholds (médio prazo, Q4 2026)

**Contexto:** o threshold `BITNET_ACDC_FFN_RECT=auto` usa `n_ff/n_embd >= 3.0`
baseado em observação empírica com 3 modelos. Com mais modelos, podemos calibrar melhor.

**Modelos a testar:**
- Mistral-7B (n_ff/n_embd = 14336/4096 = 3.50 — deve ativar)
- Llama-3.1-8B (n_ff/n_embd = 14336/4096 = 3.50 — deve ativar)
- Phi-3-mini (n_ff/n_embd = 8192/3072 = 2.67 — deve ser no-op)
- Gemma-2-2B (n_ff/n_embd = 9216/2304 = 4.0 — deve ativar)

**Output esperado:** tabela de compatibilidade em `docs/hardware-compatibility.md`
com coluna "ACDC_RECT=auto ativado".

---

## Passo 6 — M3: ACDC retangular com retreino (longo prazo, Q4 2029)

**Contexto:** T009/T018/T019 pausados por P6 (retreino GPU). Gate D2
resolvido — não é bloqueador, mas sem retreino `RECT=1` é garbage por design.

**Gatilho de reativação:** GPU disponível + demanda de comunidade.

**O que fazer quando reativar:**
1. `src/ggml-bitnet-fwht.cpp`: `acdc_project_rect(W, m, n)` — Kronecker `H_m ⊗ H_n`
2. `utils/extract_acdc_diagonal.py`: shapes retangulares (sidecar `.npz`)
3. `tests/test_acdc_rect.cpp`: ativar via `-DBITNET_ENABLE_ACDC_RECT=ON`
4. `utils/finetune_acdc.py`: loop PyTorch treinando só `d*`, W frozen
5. Rodar fine-tune em BitNet-2B (~1-2 dias A100), medir perplexity vs baseline

---

## Passo 7 — Comunidade (contínuo)

**Ações imediatas:**
- Criar `CONTRIBUTING.md` com: política de PR, como rodar testes, restrições §3 do ROADMAP
- Abrir issues de "good first issue" para: bench em novo hardware, doc em inglês, ARM64 NEON
- Adicionar badge de CI no README (`.github/workflows/ci.yml` já existe)

**Canais sugeridos:**
- `Discussions` no GitHub para perguntas D4
- `issues` para bugs e feature requests
- Mencionar em: HuggingFace forums, r/LocalLLaMA, llama.cpp Discord

---

## Resumo executivo

| # | Passo | Prazo | Esforço | Impacto |
|---|-------|-------|---------|---------|
| 1 | **Release v0.1.0 + push** | Hoje | 5 min | Alto — torna o trabalho público |
| 2 | **PR upstream microsoft/BitNet** | Semana 1 | 2-4h | Alto — visibilidade, feedback |
| 3 | **Bench mais hardware** | Q3 2026 | 1-2 dias | Médio — credibilidade D4 |
| 4 | **ARM64 NEON path** | Q4 2026 | 3-5 dias | Médio — persona D4 mobile |
| 5 | **Calibrar threshold auto** | Q4 2026 | 1-2 dias | Baixo — refinamento |
| 6 | **M3 ACDC retangular** | Q4 2029 | 1 semana | Alto (futuro) — exige GPU |
| 7 | **Comunidade** | Contínuo | Baixo | Alto — sustentabilidade |

**Bloqueadores para Passo 1:** nenhum — 6 commits locais prontos para push.
**Bloqueadores para Passo 2:** nenhum técnico — decisão de abertura pública.
**Bloqueadores para Passo 6:** GPU + autorização de retreino.

---

*Gerado em 2026-06-09 após auditoria integral: 13/13 ACs ✅, T029 ✅, bench 3 modelos ✅.*
*Próxima revisão: v0.1.0 release (Passo 1) ou Q1 2027 (revisão minor).*
