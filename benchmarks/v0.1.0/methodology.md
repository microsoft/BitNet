# Methodology — BitNet CPU-Universal Benchmarks v0.1.0

> Metodologia canônica para reprodução dos benchmarks v0.1.0. Este
> documento é **source of truth** para interpretação dos números em
> `bench.json` / `bench.md`.

---

## 1. Hardware

**Capturado automaticamente** por `utils/bench_publish.py` via
`platform.processor()`, `/proc/cpuinfo` e `/proc/meminfo`. Cada bench
JSON inclui a seção `hardware` com:

- `cpu_model` — string do `/proc/cpuinfo` (Linux) ou equivalente
- `cpu_count_logical` — `os.cpu_count()`
- `ram_mb` — `MemTotal` de `/proc/meminfo` em MB
- `platform` — `platform.platform()` (Linux, Darwin, Windows, etc.)
- `machine` — `platform.machine()` (x86_64, aarch64, etc.)
- `python_version` — versão do Python usado para gerar

**Mínimo aceitável** (persona D4, `requirements.md#9`):
- CPU: x86_64 com AVX2 (post-2013) ou ARM64 com NEON
- RAM: 8 GB mínimo, 16 GB recomendado
- Disco: ~2 GB livres (modelo + cache)

Ver [`docs/hardware-compatibility.md`](../../docs/hardware-compatibility.md)
para matriz CPU → modo.

---

## 2. Modelo

**Para v0.1.0:** `microsoft/BitNet-b1.58-2B-4T` (2.4B params, formato
GGUF i2_s). Modelo pequeno o suficiente para caber em hardware D4
(~4.5 GB RAM com KV cache 4-bit).

**Download:**
```bash
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
  --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

**Não usar** outros modelos em v0.1.0 (mude apenas com nova versão
do benchmark). Comparações entre modelos diferentes são enganosas.

---

## 3. Configurações medidas

6 configurações, cada uma medida independentemente. Ordem:

| # | Nome | Env vars adicionais | Esperado |
|---|------|---------------------|----------|
| 1 | L1 baseline (I2_S GEMV) | (nenhuma) | tok/s = 100 % de referência |
| 2 | L3 ACDC FFN | `BITNET_ACDC_FFN=1` | tok/s varia; output garbage (P6) |
| 3 | L4 Tropical top-K=32 | `BITNET_TROPICAL_TOPK=32` | tok/s tipicamente > 100 % |
| 4 | L4 Sparse float top-K=32 | `BITNET_SPARSE_TOPK=32` | tok/s tipicamente > 100 % |
| 5 | L5 HRR raw | `BITNET_HRR_ATTN=1`, `BITNET_HRR_ATTN_CLEANUP=0` | tok/s varia; output garbage (P6) |
| 6 | L5 HRR + cleanup 8 | `BITNET_HRR_ATTN=1`, `BITNET_HRR_ATTN_CLEANUP=8` | tok/s menor que L5 raw; output garbage (P6) |

**L2 WHT** é patched in `vec_dot` (always on); já incluído no L1 baseline.

**Atenção (P6):** configurações L3 e L5 produzem **output garbage** em
BitNet-2B porque o modelo não foi treinado com essas arquiteturas.
Os números medidos são **apenas overhead de kernel**, não qualidade.
Para qualidade, é necessário retreino (reserva Q4 2029).

---

## 4. Prompt e número de tokens

**Padrão:** `"The capital of France is"` (simples, não-induz-bias).
**Tokens gerados:** 64 (default; ajustável com `-n`).
**Threads:** 4 (default; ajustável com `-t`).

```bash
python run_inference.py \
  -m models/.../ggml-model-i2_s.gguf \
  -p "The capital of France is" \
  -n 64 -t 4
```

**Por que esse prompt:** tokens de saída são **completamente determinísticos**
dado o modelo e a seed, então variabilidade entre runs vem **apenas do
overhead de kernel**, não de criatividade. Ideal para comparar throughput.

**Por que 64 tokens:** mínimo razoável para `llama-cli` emitir o "tokens
per second" final no log. Menos tokens (16-32) dão variância alta.

**Por que 4 threads:** baseline D4 (laptop corporativo 4-cores, ex: i5).

---

## 5. Métrica

**Wall-clock tok/s** (tokens por segundo, end-to-end). Lido do log
`llama-cli` que imprime:

```
eval time =    X ms /    N runs   (  Y ms per token,   Z,WW tokens per second)
       total time = ... (    K,KK tokens per second)
```

**Pegamos a última menção de "tokens per second"** (overall rate, não
per-token). `utils/bench_publish.py:run_with_env` faz isso via regex
`r"(\d+[.,]\d+)\s*tokens per second"`.

**Tolerância:** run-to-run, esperar ±5 % de variância. Bench
significativo requer 3+ runs; `bench_publish.py` faz 1 run por
configuração (suficiente para v0.1.0; refine em v0.2.0).

---

## 6. Execução

### 6.1. Ambiente isolado

```bash
# Máquina parada: nenhum outro processo pesado rodando
# (Chrome, Docker, etc.) — bench é CPU-bound.
sudo systemctl stop docker  # se aplicável
# Fechar apps que possam usar CPU
```

### 6.2. Thermal

Rodar **uma** configuração por vez, esperar **30s** entre runs para o
CPU resfriar. Bench em laptop sem cooling pad pode ter thermal throttling
que não é reproduzível.

### 6.3. Sequência

```bash
# 1. Baseline primeiro
python run_inference.py ... # L1
# 2. Esperar 30s
sleep 30
# 3. Próxima
BITNET_ACDC_FFN=1 python run_inference.py ... # L3
# 4. etc.
```

**Por que sequencial e não paralelo:** queremos medir kernel isolado.
Cores em paralelo dariam falsa impressão de speedup (na verdade é só
multithreading).

### 6.4. Saída

`utils/bench_publish.py` gera:

- `bench.json` — canônico, source of truth. **Não editar manualmente.**
- `bench.md` — derivado. Gerado a partir de `bench.json`. **Não editar.**

Se precisar mudar a metodologia, mude `methodology.md` (este arquivo),
NÃO os JSON/MD. Re-rodar `bench_publish.py` regenera ambos.

---

## 7. Versionamento

Cada `bench.json` inclui `schema_version` (atualmente `"0.1.0"`) e
`timestamp_utc` (ISO 8601). Comparações entre versões:

```bash
# diff de schema entre duas versões
diff <(jq '.hardware' v0.1.0/bench.json) <(jq '.hardware' v0.2.0/bench.json)

# diff de tok/s
diff <(jq '.rows[] | "\(.id): \(.tok_per_sec)"' v0.1.0/bench.json) \
     <(jq '.rows[] | "\(.id): \(.tok_per_sec)"' v0.2.0/bench.json)
```

**Política de regressão (RNF-02):** baseline L1 não pode regredir mais
que 2 % entre releases. Se regredir, investigar antes de commitar
`bench.json`. Outras configurações podem variar (kernel experimental).

---

## 8. Limitações conhecidas

1. **1 run por configuração.** Variância run-to-run não é capturada.
   Para ±erro, rodar N vezes e reportar média ± desvio.
2. **Modelo único (BitNet-2B).** Comparações com outros modelos exigem
   nova versão de benchmark.
3. **Sem L2 separado.** L2 WHT é patched in `vec_dot`; medir isolado
   requer patch adicional.
4. **L3 e L5 dão garbage** (P6). Números são overhead, não qualidade.

---

*v0.1 — gerado por T030 (Fase 4: Integração) em 2026-06-06*
*Methodology canônica. Source of truth para interpretação de bench.json/bench.md.*
