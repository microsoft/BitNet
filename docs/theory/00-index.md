# Fundamentos Teóricos: CPU Universal LLM

> **Hipótese central**: a inferência de LLMs de grande porte no CPU pode atingir
> a velocidade da GPU não por paralelismo de hardware, mas por eliminação algébrica
> das operações de ponto flutuante — substituindo multiplicações por adições, e
> adições por comparações, descendo a hierarquia de custo computacional.

---

## A Hierarquia de Custo Operacional

```
Multiplicação float32   ~4–5 ciclos
Adição float32          ~1 ciclo
Comparação              ~0.3 ciclos
XOR / AND de bits       ~0.1 ciclos
```

Cada nível desta pesquisa substitui operações mais caras por mais baratas:

| Nível | Operação eliminada | Substituída por | Documento |
|-------|-------------------|-----------------|-----------|
| 1 | Float weights | Pesos ternários {-1,0,+1} | [01-ternary-algebra.md](01-ternary-algebra.md) |
| 2 | Multiplicações em GEMV | Adições condicionais (WHT) | [02-wht-decomposition.md](02-wht-decomposition.md) |
| 3 | O(n²) GEMV | O(n log n) FWHT + diagonal | [03-acdc-structured-layers.md](03-acdc-structured-layers.md) |
| 4 | O(n²) atenção + exp | Comparações top-K (tropical) | [04-tropical-algebra.md](04-tropical-algebra.md) |
| 5 | Atenção O(n²) completa | Memória holográfica O(n log n) | [05-holographic-memory.md](05-holographic-memory.md) |

---

## Estado de Implementação

```
Nível 0  fp16 baseline                      [referência — não implementado aqui]
Nível 1  Ternary quantization (BitNet)      [✓ herdado — src/ggml-bitnet-mad.cpp]
Nível 2  WHT decomposition zero-mul         [✓ DONE — src/ggml-bitnet-wht.cpp]
Nível 3  FWHT + ACDC O(n log n)            [✓ DONE — src/ggml-bitnet-fwht.cpp]
Nível 4  Tropical attention (max,+)         [✓ DONE — src/ggml-bitnet-tropical.cpp]
Nível 5  Holographic Reduced Representations [→ EM ANDAMENTO]
```

---

## Conexões Entre os Níveis

```
GEMV padrão: y = W·x   W ∈ ℝ^{m×n},  O(mn) multiplicações

       ┌─ Nível 1 ──────────────────────────────────────────────────┐
       │  W ternário: w ∈ {-1,0,+1}                                 │
       │  Multiplicação → skip/±1 → ainda O(mn) ops, mas 0 muls    │
       └────────────────────────────────────────────────────────────┘
                ↓
       ┌─ Nível 2 ──────────────────────────────────────────────────┐
       │  WHT decomposition: y[i] = Σ_{w=+1} x[j] - Σ_{w=-1} x[j] │
       │  SIMD: cmpeq + and + sub → zero multiplicações, O(mn)     │
       └────────────────────────────────────────────────────────────┘
                ↓
       ┌─ Nível 3 ──────────────────────────────────────────────────┐
       │  ACDC: W = H·diag(d)·H (Hadamard estruturado)             │
       │  y = H·(d⊙(H·x))  — 2 FWHTs + n muls → O(n log n)        │
       └────────────────────────────────────────────────────────────┘
                ↓
       ┌─ Nível 4 ──────────────────────────────────────────────────┐
       │  Atenção tropical: softmax(QKᵀ/√d) → (max,+) semiring     │
       │  Top-K via argmax → O(n) comparações por token            │
       └────────────────────────────────────────────────────────────┘
                ↓
       ┌─ Nível 5 ──────────────────────────────────────────────────┐
       │  Memória holográfica: Q/K/V → binding via FFT              │
       │  Atenção = recuperação associativa O(n log n)              │
       └────────────────────────────────────────────────────────────┘
```

---

## Budget Operacional — BitNet-2B (30 camadas, seq=2048)

| Pipeline | Ops/token | vs fp16 |
|----------|-----------|---------|
| fp16 baseline | ~847B | 1× |
| Nível 1 (ternário) | ~424B | 2× |
| Nível 2 (WHT, zero muls) | ~424B adds | 2× real, 4× effective |
| Nível 3 (ACDC FFN) | ~17B | ~50× |
| Nível 4 (+Tropical attn) | ~3B | ~280× |
| Nível 5 (+Holográfico) | ~500M | ~1700× |

---

## Referências Fundamentais

- Kanerva (1988). *Sparse Distributed Memory*. MIT Press.
- Walsh (1923). "A closed set of normal orthogonal functions." *Am. J. Math.*
- Hadamard (1893). "Résolution d'une question relative aux déterminants."
- Le et al. (2013). "Fastfood — Approximating Kernel Expansions in Loglinear Time." *ICML.*
- Plate (1994). *Holographic Reduced Representations*. PhD thesis, Toronto.
- Maclagan & Sturmfels (2015). *Introduction to Tropical Geometry*. AMS.
- Zhang et al. (2018). "Tropical Geometry of Deep Neural Networks." *ICML.*
- Ma et al. (2024). "The Era of 1-bit LLMs." arXiv:2402.17764.
- Bengio et al. (2013). "Estimating or Propagating Gradients Through Stochastic Neurons." arXiv:1308.3432.
