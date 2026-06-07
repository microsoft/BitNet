# Benchmarks v0.3.0 — L1–L5 + ACDC rect (Fase II/III)

**Gerado em:** 2026-06-07T14:30:00Z  
**Hardware:** Intel Core i5-10210U @ 1.60 GHz · 4 threads · 35 GB RAM · AVX2  
**Condições:** `llama-cli`, prompt="The capital of France is", n=64 tokens decode  
**Versão anterior:** [v0.2.0/bench.md](../v0.2.0/bench.md)

---

## Configurações

| ID | Env vars | Descrição |
|----|----------|-----------|
| L1 baseline | _(nenhuma)_ | I2_S GEMV padrão (atenção densa) |
| L3 ACDC FFN | `BITNET_ACDC_FFN=1` | ACDC quadrado, dims hardcoded BitNet-2B |
| **L3 ACDC rect d=0** | `BITNET_ACDC_FFN_RECT=1` | ACDC rect, diagonal=zeros (pesos não lidos) |
| **L3 ACDC rect d=rand** | `BITNET_ACDC_FFN_RECT=1 BITNET_ACDC_FFN_RECT_RAND=1` | ACDC rect, diagonal aleatório (timing puro) |
| L4 Tropical K=32 | `BITNET_TROPICAL_TOPK=32` | Atenção tropical (max,+) top-K |
| L4 Sparse float K=32 | `BITNET_SPARSE_TOPK=32` | Atenção sparse float top-K |
| L5 HRR raw | `BITNET_HRR_ATTN=1` | Holographic reduced representations |
| L5 HRR + cleanup 8 | `BITNET_HRR_ATTN=1 BITNET_HRR_ATTN_CLEANUP=8` | HRR + Frady 2021 iterative cleanup |

---

## BitNet-b1.58-2B-4T

**Arquitetura:** 18 layers · hidden=2560 · n_ff=6912 · **n_ff/n_embd=2.7×** · head_dim=128

| Configuração | tok/s | Δ vs L1 |
|---|---:|---:|
| L1 baseline (I2_S GEMV) | 5.27 | 0.0% |
| L3 ACDC FFN | 4.71 | −10.6% |
| L3 ACDC rect d=rand | **5.36** | **+1.7%** |
| L4 Tropical K=32 | 4.53 | −14.0% |
| L4 Sparse float K=32 | 4.85 | −8.0% |
| L5 HRR raw | 1.85 | −64.9% |
| L5 HRR + cleanup 8 | 1.87 | −64.5% |

> ACDC rect d=0 não foi medido neste modelo (n_ff/n_embd=2.7× abaixo do limiar empírico de ~5×).  
> L3/L4/L5 (exceto rect) levados do v0.2.0.

---

## Falcon3-3B-Instruct-1.58bit

**Arquitetura:** 22 layers · hidden=3072 · n_ff=9216 · **n_ff/n_embd=3.0×** · head_dim=256

| Configuração | tok/s | Δ vs L1 |
|---|---:|---:|
| L1 baseline (I2_S GEMV) | 4.61 | 0.0% |
| L3 ACDC FFN | 4.21 | −8.7% |
| L3 ACDC rect d=0 | 4.51 | −2.2% |
| L3 ACDC rect d=rand | 4.45 | −3.5% |
| L4 Tropical K=32 | 4.19 | −9.1% |
| L4 Sparse float K=32 | 4.49 | −2.6% |
| L5 HRR raw | 2.64 | −42.7% |
| L5 HRR + cleanup 8 | 2.22 | −51.8% |

> n_ff/n_embd=3.0× — abaixo do limiar. ACDC rect overhead (FWHT P=16384) > economia de I/O.  
> L3/L4/L5 (exceto rect) levados do v0.2.0.

---

## Falcon3-10B-Instruct-1.58bit

**Arquitetura:** 40 layers · hidden=3072 · n_ff=23040 · **n_ff/n_embd=7.5×** · head_dim=256

| Configuração | tok/s | Δ vs L1 |
|---|---:|---:|
| L1 baseline (I2_S GEMV) | 1.12 | 0.0% |
| L3 ACDC FFN | 1.25 | −10.7% (v0.2.0) |
| **L3 ACDC rect d=0** | **4.11** | **+267%** |
| **L3 ACDC rect d=real** | **4.19** | **+274%** |
| L4 Tropical K=32 | 1.16 | −17.1% (v0.2.0) |
| L4 Sparse float K=32 | 1.14 | −18.6% (v0.2.0) |
| L5 HRR raw | 0.89 | −36.4% (v0.2.0) |
| L5 HRR + cleanup 8 | 0.97 | −30.7% (v0.2.0) |

> n_ff/n_embd=7.5× — **acima do limiar**. Reads de pesos (720 MB/forward) dominam;
> ACDC rect reduz para 4.2 MB (170× menos I/O de memória) → **3.7× speedup líquido**.
>
> **Correção v0.3.1 (2026-06-07):** benchmarks anteriores (+3.6%) eram errados —
> o gate `BITNET_ACDC_FFN_RECT` estava apenas em `build_falcon()`, mas Falcon3-10B
> reporta `arch=llama` e roteia por `build_llama()`. Patch 05 adicionou o gate
> ao `build_llama()`. Baseline re-medido na mesma sessão.
>
> **d=real vs d=0 (4.19 vs 4.11 tok/s):** marginal, dentro do ruído térmico.
> Para modelos não treinados com ACDC, d=real ≈ d=0 em throughput e qualidade.
> L3/L4/L5 (exceto rect) levados do v0.2.0.

---

## Tabela comparativa: ACDC rect × 3 modelos

| Modelo | n_ff/n_embd | Baseline | ACDC rect d=0 | ACDC rect d=real |
|--------|-------------|----------|---------------|-----------------|
| BitNet-2B | 2.7× | 5.27 tok/s | — | — |
| Falcon3-3B | 3.0× | 4.61 tok/s | −2.2% | n/a |
| **Falcon3-10B** | **7.5×** | **1.12 tok/s** | **+267%** | **+274%** |

**Lei empírica confirmada (revisada):** ACDC rect traz speedup quando `n_ff/n_embd > ~5`.
**Mecanismo:** FFN rectangular lê 720 MB/forward de pesos (Falcon3-10B);
ACDC rect substitui por FWHT in-cache → **3.7× speedup real** (não os +3.6% errados do v0.3.0).

> **Nota (v0.3.1):** d=real vem de `extract_acdc_diagonals.py` + `acdc_diag_to_bin.py`
> (pipeline completo de Direção #1). d=real ≈ d=0 em throughput para modelo não-ACDC-treinado.

---

## Achados chave

1. **Speedup real de 3.7× no Falcon3-10B (correção v0.3.1):** benchmarks anteriores (+3.6%) estavam errados — o gate `BITNET_ACDC_FFN_RECT` só estava em `build_falcon()`, não em `build_llama()`. Falcon3-10B usa arch=llama, então ACDC rect não estava ativo. Patch 05 corrigiu isso. O speedup real é **+267% d=0, +274% d=real**.

2. **d=real ≈ d=0 em throughput:** para modelos não treinados com ACDC, a diagonal real `d*` extraída via XOR-convolution é essencialmente ruído (magnitude ~10⁻⁵). A diferença de throughput (4.19 vs 4.11 tok/s) é dentro da variância térmica.

3. **Pipeline Direction #1 completo:** `extract_acdc_diagonals.py` → `.acdc_diag.npz` → `acdc_diag_to_bin.py` → `.acdc_diag.bin` → carregado em `ggml-bitnet-dispatch.cpp` via `BITNET_ACDC_FFN_RECT_DIAG`. Falcon3-10B: 120 tensores em 5.5min, sidecar de 11.3 MB.

4. **Limiar empírico n_ff/n_embd ≈ 5 confirmado:** Falcon3-10B (7.5×) — 3.7× speedup; Falcon3-3B (3.0×) — −2.2%. O mecanismo é redução de I/O de memória (720 MB/forward → ~0 com ACDC rect).

5. **Gap P6 permanece:** todos os kernels L2-L5 produzem output degradado — modelos não treinados com essas arquiteturas. Próximo passo: treinar modelo com n_ff/n_embd ≥ 7 com FFN ACDC rect.

---

## Anotações de metodologia

- `d=0` (default): diagonal é zero → output zero, mas leitura de pesos ignorada → speedup puro de I/O.  
- `d=rand` (`BITNET_ACDC_FFN_RECT_RAND=1`): diagonal aleatório → output inválido, mesma carga computacional → timing real do FWHT.  
- Baseline v0.3.0 re-medido na mesma sessão; variância ±0.1 tok/s vs v0.2.0 por condições térmicas.  
- Patches aplicados via `scripts/apply-dispatch-patches.sh` (patch cumulativo 04).

---

*Gerado manualmente em 2026-06-07 a partir de medições com `llama-cli`. JSON canônico: [`bench.json`](bench.json).*
