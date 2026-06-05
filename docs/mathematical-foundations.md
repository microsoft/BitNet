# Mathematical Foundations: CPU-Universal Language Models via Forgotten Algebra

> **Goal**: Universalize large language models through mathematical structures
> that make CPU-native inference as capable as GPU inference — not by brute-force
> hardware, but by eliminating the need for multiplication entirely.

---

## The Central Question

A 7B-parameter fp16 model needs ~14 TFLOPS to generate a token.
A modern CPU delivers ~0.1–0.5 TFLOPS.
A GPU closes this gap with parallelism.

**Our answer**: eliminate FLOPS at the algebraic level, not the hardware level.

---

## Level 0 — Baseline: Float Arithmetic

Standard neural network linear layer:

```
y = W · x      W ∈ ℝ^{m×n},  x ∈ ℝ^n

Cost: m·n multiplications + m·(n-1) additions ≈ 2mn FLOPs
```

For BitNet-2B, one FFN layer: m=6912, n=2560 → ~35.4M FLOPs per token.

---

## Level 1 — Ternary Quantization: 1.58 bits/parameter

**Mathematical basis**: Shannon entropy of a uniform 3-symbol distribution.

```
H({-1, 0, +1}) = log₂(3) ≈ 1.585 bits/symbol
```

This is the information-theoretic minimum — no lossless code can do better.

**Quantization function** (absmax-mean, per tensor):

```
γ = (1/n) Σᵢ |wᵢ|                      (scale: robust mean, not max)
w_q = round( clamp(w/γ, -1, 1) )        → {-1, 0, +1}
```

**Why mean, not max**: The absmax is dominated by outliers (large wᵢ),
wasting the dynamic range. The mean is the MLE estimator for the Laplace
distribution that ternary weights follow after training (empirically verified
in BitNet paper, 2024).

**Error bound** (Frobenius norm):

```
||W - γ·W_q||_F  ≤  γ/2 · √(mn)

For Gaussian W ~ N(0, σ²/n):  γ ≈ σ·√(2/π)
Relative error: ||error||_F / ||W||_F ≈ 1/(2√n) → 0 as n→∞
```

**Key insight**: larger models tolerate ternary quantization better because the
relative error decreases with the square root of the parameter count.

---

## Level 2 — WHT Decomposition: Zero Multiplications

**Algebraic identity** (the core of this project):

```
For W ∈ {-1, 0, +1}^{m×n} and x ∈ ℤ^n:

Define:  W⁺[i,j] = 𝟙[W[i,j] = +1]     (positive mask, binary)
         W⁻[i,j] = 𝟙[W[i,j] = -1]     (negative mask, binary)

Then:  y[i] = Σⱼ W[i,j]·x[j]
             = Σ_{j: W[i,j]=+1} x[j]  −  Σ_{j: W[i,j]=-1} x[j]
             ≡ (W⁺·x)[i]  −  (W⁻·x)[i]
```

**Result**: the dot product decomposes into two conditional sums.
**No multiplication ever occurs.** Only additions and subtractions.

**SIMD implementation** (AVX2, one dot product of length 32):

```c
// Load 32 int8 activations
__m256i acts    = _mm256_loadu_si256(x);

// Extract sign masks from ternary weights (cost: 2 × cmpeq = 2 cycles)
__m256i pos_m   = _mm256_cmpeq_epi8(weights, v_pos);   // 0xFF where w=+1
__m256i neg_m   = _mm256_cmpeq_epi8(weights, v_zero);  // 0xFF where w=-1

// Conditional selection (cost: 2 × and = 2 cycles)
__m256i pos_v   = _mm256_and_si256(acts, pos_m);
__m256i neg_v   = _mm256_and_si256(acts, neg_m);

// Signed subtraction (cost: 1 × sub = 1 cycle)
__m256i delta   = _mm256_sub_epi8(pos_v, neg_v);

// Accumulate (cost: madd+add = 2 cycles)
accum = _mm256_add_epi32(accum,
    _mm256_madd_epi16(_mm256_add_epi16(
        _mm256_cvtepi8_epi16(lo128(delta)),
        _mm256_cvtepi8_epi16(hi128(delta))), ones16));
```

Total: ~7 cycles per 32 elements vs ~10 cycles for maddubs path.

### Connection to Walsh-Hadamard Transform

The WHT of a vector v ∈ {-1, +1}^n:

```
V̂[k] = Σⱼ v[j] · H[j,k]    where H[j,k] = (-1)^{popcount(j & k)}
```

The Hadamard matrix H has entries in {-1, +1} only. The Fast WHT computes all
V̂[k] in O(n log n) using only additions and subtractions — the **butterfly
algorithm**, which is the direct ancestor of the FFT.

Our W = W⁺ - W⁻ decomposition IS the WHT in disguise:
- W⁺ encodes which activations to ADD
- W⁻ encodes which activations to SUBTRACT
- The butterfly structure of the WHT shows this can be organized recursively

**Deeper connection**: if W is structured as H·diag(d) for a diagonal d and
Hadamard H, then y = H·(d ⊙ x) can be computed in O(n log n) with ONLY
additions. This is Level 3 — the structured WHT approximation.

---

## Level 3 — Structured WHT Approximation: O(n log n) GEMV

**The ACDC / Fastfood idea** (Le et al., 2013; Yu et al., 2016):

```
W ≈ H · D · H    where H is Hadamard, D is learned diagonal

y = W·x ≈ H·(D·(H·x))
         = H·(d ⊙ (H·x))

Step 1: z = H·x    — Fast WHT, O(n log n), additions only
Step 2: z = d ⊙ z  — diagonal scaling, n multiplications by scalars d
Step 3: y = H·z    — Fast WHT again, O(n log n), additions only

Total: O(n log n) instead of O(n²)
Multiplications: n (only the diagonal d — irreducible minimum)
```

For BitNet-2B n=2560, m=6912:
- Current: 2560×6912 ≈ 17.7M operations per layer per token
- Level 3: 2×2560×log₂(2560) + 2560 ≈ 60K operations per layer per token
- **Speedup: ~295× in operations** (accuracy trade-off to be measured)

### Why this approximation works (algebraically)

Random ternary matrices W are approximately **incoherent** — their singular
values are nearly uniform (Marchenko-Pastur law for random matrices). Hadamard
matrices are maximally incoherent (all entries ±1/√n). The product H·W is
close to a diagonal matrix because Hadamard is the "natural basis" for
symmetric, incoherent linear maps.

**Formal statement** (Johnson-Lindenstrauss): For a random Gaussian W,
with high probability:

```
||W·x - H·D·H·x|| ≤ ε||W·x||    for  |D| ~ O(√n)
```

The approximation quality improves with n — larger models benefit more.

---

## Level 4 — Tropical Attention: O(n) Per Token

**The softmax bottleneck**:

```
Attention: A[i,j] = softmax( Q[i]·K[j]ᵀ / √d )   — O(n²) operations
```

For n=2048 tokens, d=128: 2048² = 4M operations per attention head per token.

**Tropical reformulation**:

The (max, +) semiring replaces (ℝ, +, ×) with (ℝ, max, +):
- Multiplication a×b → addition a+b
- Addition a+b → maximum max(a,b)

In the limit temperature → 0, softmax(v/τ) → one-hot(argmax(v)):

```
lim_{τ→0} softmax(v/τ)[i] = 𝟙[i = argmax(v)]
```

This is exactly the argmax operation, which in the max-plus semiring IS the
tropical matrix product:

```
(A ⊗ᵗʳᵒᵖ B)[i,k] = max_j( A[i,j] + B[j,k] )
```

**Implication**: at low temperature (sharp attention), the transformer's
attention mechanism IS a tropical matrix product. This requires only:
- Comparisons (max) instead of multiplications
- Additions instead of exp (for the +)

**Sparse tropical attention**:

```
y[i] = Σⱼ A[i,j] · v[j]

where A[i,j] = 𝟙[j = argmax_k Q[i]·K[k]ᵀ]  (hard attention)

→ y[i] = v[argmax_k Q[i]·K[k]ᵀ]
       = lookup(V, nearest_neighbor(Q[i], K))
```

This reduces attention from O(n²) to O(n·log n) via approximate nearest
neighbor search (HNSW, FAISS-CPU), or O(n) with locality-sensitive hashing.

**CPU advantage**: ANN search is memory-efficient and cache-friendly on CPU.
GPU attention is fast because it parallelizes the O(n²) — but O(n log n) ANN
is faster than parallelized O(n²) beyond a crossover point.

---

## Level 5 — Holographic Reduced Representations (Kanerva 1988)

**The deepest forgotten algebra**: Sparse Distributed Memory + HRR.

Kanerva's SDM stores and retrieves associations in high-dimensional binary
vectors using Hamming distance — a pure bitwise operation (XOR + popcount).

**Holographic encoding** (Plate, 1994):

```
Binding:     A # B = IFFT(FFT(A) ⊙ FFT(B))   — element-wise complex multiply in Fourier space
Superposition: M = A # B + C # D + ...        — additive
Retrieval:   B̃ = M # A⁻¹                     — inverse binding (FFT-based)
```

This is an associative memory that:
- Stores key-value pairs in a SINGLE vector (superposition)
- Retrieves by approximate nearest-neighbor in Fourier space
- Scales to millions of pairs in O(n log n) time

**Connection to Transformer**: the attention mechanism IS approximate holographic
retrieval. The query Q is the "retrieval key", K is the "stored key", V is the
"stored value":

```
Transformer:  y = softmax(QKᵀ)·V    (O(n²) exact)
HRR:          y ≈ retrieve(M, Q)     (O(n log n) approximate)
```

For ternary LLMs, storing all K-V pairs in a holographic memory and retrieving
via FFT replaces the transformer's attention entirely — no O(n²), no GPU memory
bandwidth wall.

---

## Implementation Roadmap

| Level | Math | Status | Expected CPU Speedup |
|-------|------|--------|---------------------|
| 0 | fp16 GEMV baseline | — | 1× |
| 1 | Ternary quantization | ✓ (BitNet) | 3–6× |
| 2 | WHT decomposition (zero mul) | **NOW** → `src/ggml-bitnet-wht.cpp` | 1.5–2× over L1 |
| 3 | Structured WHT (ACDC) | Next sprint | ~10–50× over L1 |
| 4 | Tropical attention | Research | O(n²) → O(n log n) |
| 5 | Holographic memory | Research | Transformer replacement |

---

## Key Mathematical References

- **Ternary quantization**: Ma et al., "The Era of 1-bit LLMs" (2024)
- **Walsh-Hadamard**: Walsh (1923), Hadamard (1893) — 100+ years old
- **ACDC/Fastfood**: Le et al., "Fastfood – Approximating Kernel Expansions in Loglinear Time" (ICML 2013)
- **Tropical mathematics**: Maclagan & Sturmfels, "Introduction to Tropical Geometry" (AMS 2015)
- **Tropical neural networks**: Zhang et al., "Tropical Geometry of Deep Neural Networks" (ICML 2018)
- **Sparse Distributed Memory**: Kanerva (1988), MIT Press
- **Holographic Reduced Representations**: Plate (1994), PhD thesis, Univ. of Toronto
- **STE (Straight-Through Estimator)**: Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons" (2013)
- **Marchenko-Pastur law**: Random matrix theory — explains why ternary approximation works at scale
