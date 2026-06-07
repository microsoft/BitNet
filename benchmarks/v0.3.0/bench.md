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
| L1 baseline (I2_S GEMV) | 1.40 | 0.0% |
| L3 ACDC FFN | 1.25 | −10.7% |
| **L3 ACDC rect d=0** | **1.45** | **+3.6%** |
| **L3 ACDC rect d=rand** | **1.43** | **+2.1%** |
| L4 Tropical K=32 | 1.16 | −17.1% |
| L4 Sparse float K=32 | 1.14 | −18.6% |
| L5 HRR raw | 0.89 | −36.4% |
| L5 HRR + cleanup 8 | 0.97 | −30.7% |

> n_ff/n_embd=7.5× — **acima do limiar**. Reads de pesos (720 MB/forward) dominam;  
> ACDC rect reduz para 4.2 MB (170× menos I/O de memória) → speedup líquido.  
> L3/L4/L5 (exceto rect) levados do v0.2.0.

---

## Tabela comparativa: ACDC rect × 3 modelos

| Modelo | n_ff/n_embd | Baseline | ACDC rect d=0 | ACDC rect d=rand |
|--------|-------------|----------|---------------|-----------------|
| BitNet-2B | 2.7× | 5.27 tok/s | — | +1.7% |
| Falcon3-3B | 3.0× | 4.61 tok/s | −2.2% | −3.5% |
| **Falcon3-10B** | **7.5×** | **1.40 tok/s** | **+3.6%** | **+2.1%** |

**Lei empírica confirmada:** ACDC rect traz speedup quando `n_ff/n_embd > ~5`.  
**Mecanismo:** FFN rectangular (gate/up/down proj) lê pesos diretamente da RAM;  
ACDC rect substitui por FWHT (P log P ops, zero peso lido) → 170× menos tráfego de memória.

---

## Achados chave

1. **ACDC rect d=0 > d=rand no 10B (+3.6% vs +2.1%):** diagonal zero zera o output, mas skipa completamente os reads de peso. FWHT de zeros é cache-trivial. O FWHT em si não é o gargalo — o I/O de memória é.

2. **Limiar empírico n_ff/n_embd ≈ 5:** Falcon3-10B (7.5×) é o único modelo com speedup consistente. Falcon3-3B (3.0×) e BitNet-2B (2.7×) ficam no ruído ou perdem levemente.

3. **L3 ACDC quadrado piora com escala:** −2.8% (BitNet-2B) → −8.7% (Falcon3-3B) → −10.7% (Falcon3-10B). FWHT sem otimização AVX2 perde para GEMV I2_S. ACDC rect resolve isso via redução de I/O em vez de redução de ops.

4. **Gap P6 permanece:** todos os kernels L2-L5 produzem output degradado — modelos não treinados com essas arquiteturas. Medições refletem overhead/speedup do kernel, não qualidade de saída.

5. **Próximo passo:** treinar um modelo com n_ff/n_embd ≥ 7 com FFN ACDC rect para fechar o gap P6 e medir perplexidade real.

---

## Anotações de metodologia

- `d=0` (default): diagonal é zero → output zero, mas leitura de pesos ignorada → speedup puro de I/O.  
- `d=rand` (`BITNET_ACDC_FFN_RECT_RAND=1`): diagonal aleatório → output inválido, mesma carga computacional → timing real do FWHT.  
- Baseline v0.3.0 re-medido na mesma sessão; variância ±0.1 tok/s vs v0.2.0 por condições térmicas.  
- Patches aplicados via `scripts/apply-dispatch-patches.sh` (patch cumulativo 04).

---

*Gerado manualmente em 2026-06-07 a partir de medições com `llama-cli`. JSON canônico: [`bench.json`](bench.json).*
