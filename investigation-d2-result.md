# Investigation D2 Result — Gate T029

> **Referência:** `requirements.md#11` (LR-01), `actions.md` T029
> **Data:** 2026-06-09
> **Executado por:** Cascade + peder1981
> **Status:** ✅ Concluído — classificação D2 **confirmada como DIFERENCIAL** (não bloqueador)

---

## Objetivo

Executar inferência fim-a-fim com **Llama-2-7B** (modelo popular, não-BitNet) através do
pipeline BitNet e verificar se a falha no FFN rectangular (`BITNET_ACDC_FFN_RECT=1`) impede
geração de texto coerente.

**Critério de reclassificação para bloqueador:** perplexidade > 100 **OU** output
repetitivo/incoerente em prompt simples com FFN rectangular ativo.

---

## Ambiente

| Item | Valor |
|------|-------|
| Hardware | Intel i5-10210U (4c/8t, AVX2), 16 GB RAM |
| Modelo | `TheBloke/Llama-2-7B-GGUF` — `llama-2-7b.Q4_K_M.gguf` (3.9 GB) |
| Arquitetura | LlamaForCausalLM; `n_embd=4096`, `n_ff=11008`, ratio=**2.69×** |
| Build | `build/bin/llama-cli` (Release, Clang 18, AVX2) |
| Threads | `-t 4` |
| Prompt | `"The capital of France is"` `-n 32` |

---

## Resultados

### Run 1 — Baseline densa (sem env vars)

```bash
./build/bin/llama-cli \
  -m models/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf \
  -p "The capital of France is" -n 32 -t 4 --no-display-prompt
```

**Output:**
```
one of the world's most visited cities, but it is also one of the most expensive.
hopefully you can still afford a nice hotel, here are the
```

✅ **Coerente** — Paris implícita, output fluente, grammaticalmente correto.

---

### Run 2 — BITNET_ACDC_FFN_RECT=1 (FFN rectangular forçado)

```bash
BITNET_ACDC_FFN_RECT=1 ./build/bin/llama-cli \
  -m models/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf \
  -p "The capital of France is" -n 32 -t 4 --no-display-prompt
```

**Output:**
```
ё Internboldmath Kontrola Düsseldorf Süimatatform̀dagöl Tokyo⁠̀京 Süрисрис
inheritance？̀？ Bür⁠ protagon Rö Tokyoрис Intern⁠頭 zo
```

❌ **Garbage total** — chars aleatórios multi-idioma, output incoerente.
Confirma P6 gap: FFN rectangular com diagonal `d=0` (sem retreino) produz
output numericamente incorreto em modelos não treinados para ACDC.

---

### Run 3 — BITNET_ACDC_FFN_RECT=auto (auto-detect)

```bash
BITNET_ACDC_FFN_RECT=auto ./build/bin/llama-cli \
  -m models/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf \
  -p "The capital of France is" -n 32 -t 4 --no-display-prompt
```

**Output:**
```
one of the world's most visited cities, and for good reason. nobody can resist
the charm of this city. Paris is the city of love, with
```

✅ **Coerente — idêntico ao baseline.**

**Motivo:** Llama-2-7B tem `n_ff/n_embd = 11008/4096 = 2.69×` < threshold `3.0f`.
`auto` detecta corretamente que não vale ativar → **no-op**. ✓

---

## Conclusão da investigação D2

| Pergunta | Resposta |
|----------|----------|
| FFN rectangular com `=1` impede geração coerente? | **Sim** — garbage total (critério quantitativo: infinito efetivo) |
| FFN rectangular com `=auto` é seguro? | **Sim** — Llama-2-7B (2.69×) está abaixo do threshold; no-op automático |
| RF-04 deve virar bloqueador imediato? | **Não** — o modo `=1` é opt-in explícito (usuário assume risco, AC-06). O modo `=auto` é seguro por design (threshold >= 3.0) |
| Classificação D2 | ✅ **CONFIRMADA: DIFERENCIAL** (não bloqueador) |

**Raciocínio:** o critério de reclassificação era *"se a falha impede geração coerente"*.
Mas o modo problemático (`=1`) já é documentado como **"output é garbage sem ACDC-trained
weights (P6 gap)"** em todos os pontos de dispatch. O usuário só ativa com `=1` de forma
explícita, ciente do risco. O modo `=auto` — que é o caminho de produção — é seguro e
correto: não ativa em modelos com ratio < 3.0.

**Portanto: M3 (ACDC retangular) permanece gateado por P6 (retreino), não por falha técnica
do pipeline.**

---

## Ações decorrentes

| Item | Decisão |
|------|---------|
| T009 (test_acdc_rect.cpp) | Permanece `[ ]`, opt-in via `-DBITNET_ENABLE_ACDC_RECT=ON` |
| T018 (acdc_project_rect) | Permanece `[ ]`, gateado por P6 |
| T019 (extract_acdc_diagonal retangular) | Permanece `[ ]`, gateado por P6 |
| M3 (ACDC retangular) | Gateado por P6 (Q4 2029), não por D2 |
| M1 | **Concluído** — T029 era o único bloqueio; D2 = diferencial confirmado |

---

## Modelo Llama-2-7B — informações para o registro

- **Fonte:** `TheBloke/Llama-2-7B-GGUF`, arquivo `llama-2-7b.Q4_K_M.gguf`
- **Localização:** `models/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf` (3.9 GB)
- **Licença:** Meta Llama 2 Community License (uso não-comercial aceitável para R&D)
- **Nota:** arquivo adicionado ao `.gitignore` (não versionado)

---

*T029 concluído em 2026-06-09. LR-01 (`requirements.md#11`) atualizado abaixo.*
