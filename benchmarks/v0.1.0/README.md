# Benchmarks v0.1.0

> Diretório canônico para benchmarks do release v0.1.0.
> Esta pasta é versionada no git; os arquivos JSON e Markdown aqui
> representam o **baseline oficial** da v0.1.0 para referência futura.

---

## Status atual (2026-06-06)

Os arquivos `bench.json` e `bench.md` ainda **não foram gerados** porque
a geração exige um **modelo real** (BitNet-2B ou similar) e a execução
demora ~3-5 min por configuração × 6 configurações ≈ 30 min.

**Para gerar (manualmente, em hardware real):**

```bash
# 1. Ativar env
conda activate bitnet-cpp
cd BitNet

# 2. Gerar bench completo
python utils/bench_publish.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --json benchmarks/v0.1.0/bench.json \
  --md benchmarks/v0.1.0/bench.md

# 3. Verificar
cat benchmarks/v0.1.0/bench.json
cat benchmarks/v0.1.0/bench.md

# 4. Commitar
git add benchmarks/v0.1.0/
git commit -m "bench(v0.1.0): systematic L1-L5 benchmark"
```

**Quando commitar:** após **cada release minor** (v0.1.0, v0.2.0, ...).
A comparação entre `bench.json` de releases consecutivos revela regressões
de performance e progresso dos kernels algébricos.

---

## Arquivos

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `README.md` | ✅ Este arquivo | Como gerar e usar o bench |
| `methodology.md` | ✅ Stub | Metodologia canônica (veja abaixo) |
| `bench.json` | ⏳ Pendente | JSON canônico (gerado por `bench_publish.py`) |
| `bench.md` | ⏳ Pendente | Markdown derivado (gerado por `bench_publish.py`) |

---

## Cross-references

- **`utils/bench_publish.py`** — Gerador (T020)
- **`utils/cpu_universal_benchmark.py`** — Script de bench base
- **`docs/decision-matrix.md`** (T015) — Interpretação dos números
- **`docs/hardware-compatibility.md`** (T016) — Hardware testado
- **AC-05** (`requirements.md#6`) — Critério de aceitação "bench sistemático commitado"

---

*v0.1 — gerado por T030 (Fase 4: Integração) em 2026-06-06*
*Estrutura criada. JSON/MD pendentes de geração em hardware real.*
