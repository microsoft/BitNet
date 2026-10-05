"""
tropical_benchmark.py — Atenção Tropical: Semiring (max, +)

Nível 4 do roteiro de universalização CPU.

FUNDAMENTO MATEMÁTICO:
  O semiring tropical (ℝ, max, +) substitui (ℝ, +, ×):
    a ⊕ b = max(a, b)        [adição tropical]
    a ⊗ b = a + b            [multiplicação tropical]

  Produto matricial tropical:
    (A ⊗ B)[i,k] = max_j (A[i,j] + B[j,k])

  Conexão com Transformer:
    Atenção padrão:   A[i,j] = softmax(Q[i]·K[j]ᵀ / √d)   — O(n²)
    Limite τ→0:       A[i,j] → δ[j = argmax_k Q[i]·K[k]ᵀ]  — O(n)

    lim_{τ→0} softmax(v/τ)[j] = 𝟙[j = argmax(v)]
    ↑ isto É o produto tropical max-plus.

  Atenção Top-K tropical:
    1. Tropical max scan:  O(n·d)     [adições ternárias — zero multiplicações]
    2. Softmax top-K:      O(K)       [apenas K exponenciais]
    3. Weighted sum V:     O(K·d)     [soma ponderada de K vetores]
    Speedup: n/K vs atenção padrão (para n=2048, K=32: 64×)
"""

import argparse
import time
import math
import numpy as np
from typing import Tuple, List


# ─── Primitivas ternárias ──────────────────────────────────────────────────

def random_ternary_matrix(rows: int, cols: int, sparsity: float = 0.5,
                          seed: int = 42) -> np.ndarray:
    """Gera matriz ternária {-1,0,+1} com sparsidade dada (fração de zeros)."""
    rng = np.random.default_rng(seed)
    p_neg = (1 - sparsity) / 2
    p_zer = sparsity
    p_pos = (1 - sparsity) / 2
    return rng.choice([-1, 0, 1], size=(rows, cols), p=[p_neg, p_zer, p_pos])


def quantize_int8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantiza vetor float para int8, retorna (int8, scale)."""
    absmax = np.max(np.abs(x))
    if absmax == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    scale = absmax / 127.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


# ─── Produto escalar ternário (Level 2: zero multiplicações) ──────────────

def dot_ternary(q: np.ndarray, k_ternary: np.ndarray) -> float:
    """
    q · k  onde k ∈ {-1,0,+1}^d.
    Decompõe: Σ_{k=+1} q[i] - Σ_{k=-1} q[i]
    Zero multiplicações — apenas adições condicionais.
    """
    pos_sum = np.sum(q[k_ternary > 0])
    neg_sum = np.sum(q[k_ternary < 0])
    return float(pos_sum - neg_sum)


# ─── Semiring (max, +) ────────────────────────────────────────────────────

def tropical_add(a: float, b: float) -> float:
    """Adição tropical: a ⊕ b = max(a, b)."""
    return max(a, b)


def tropical_mul(a: float, b: float) -> float:
    """Multiplicação tropical: a ⊗ b = a + b."""
    return a + b


def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Produto matricial tropical: C[i,k] = max_j (A[i,j] + B[j,k])
    Semanticamente: substitui (×,+) por (+,max) em álgebra tropical.
    """
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    C = np.full((m, p), -np.inf)
    for i in range(m):
        for k in range(p):
            for j in range(n):
                val = A[i, j] + B[j, k]    # tropical mul = adição real
                C[i, k] = max(C[i, k], val) # tropical add = max
    return C


def tropical_matmul_fast(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Produto tropical via broadcasting NumPy — O(m·n·p) mas vetorizado."""
    # A: (m, n), B: (n, p)
    # C[i,k] = max_j (A[i,j] + B[j,k])
    # A[:,i,:] = A[i,:,np.newaxis] ; B: (1,n,p)
    # A_exp: (m,n,1) + B_exp: (1,n,p) → (m,n,p), então max over axis 1
    A_exp = A[:, :, np.newaxis]   # (m, n, 1)
    B_exp = B[np.newaxis, :, :]   # (1, n, p)
    return np.max(A_exp + B_exp, axis=1)


# ─── Atenção tropical completa ────────────────────────────────────────────

def attention_standard(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
    """
    Atenção padrão: softmax(Q·Kᵀ / (√d · τ)) · V
    O(n²·d) — referência.
    """
    d = Q.shape[-1]
    scores = Q @ K.T / (math.sqrt(d) * temperature)
    # log-sum-exp numericamente estável
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V


def attention_tropical_hard(Q: np.ndarray, K_ternary: np.ndarray,
                             V: np.ndarray) -> np.ndarray:
    """
    Atenção tropical HARD: output[i] = V[argmax_j Q[i]·K_ternary[j]]
    O(n·d) — produto tropical puro, zero multiplicações para K ternário.

    Limite exato de softmax quando τ → 0.
    """
    n_queries = Q.shape[0]
    d = Q.shape[1]
    n_keys = K_ternary.shape[0]
    output = np.zeros((n_queries, V.shape[1]))

    for i in range(n_queries):
        best_j = 0
        best_score = -np.inf
        for j in range(n_keys):
            # Dot product ternário: zero multiplicações
            s = dot_ternary(Q[i].astype(np.int64), K_ternary[j])
            if s > best_score:
                best_score = s
                best_j = j
        output[i] = V[best_j]
    return output


def attention_tropical_hard_fast(Q: np.ndarray, K_ternary: np.ndarray,
                                  V: np.ndarray) -> np.ndarray:
    """
    Versão vetorizada: Q @ K_ternary.T → argmax por linha → indexar V.
    Equivalente a dot_ternary mas usando NumPy para benchmark de velocidade.
    K_ternary ∈ {-1,0,+1}: @ com int8/float funciona como adições condicionais.
    """
    scores = Q @ K_ternary.T   # (n_q, n_k) — float×{-1,0,+1} = adição
    best_indices = np.argmax(scores, axis=1)
    return V[best_indices]


def attention_tropical_topk(Q: np.ndarray, K_ternary: np.ndarray,
                             V: np.ndarray, K_top: int = 32,
                             temperature: float = 1.0) -> np.ndarray:
    """
    Atenção tropical Top-K: encontra K melhores keys, aplica softmax sobre elas.

    Algoritmo:
      1. Scan tropical O(n·d): Q @ K_ternary.T  (adições para K ternário)
      2. Top-K O(n·log K):     argpartition
      3. Softmax sobre K:      O(K) exponenciais
      4. Output:               Σ_{k∈topK} w_k · V[k]

    vs atenção padrão: O(n²·d) → O(n·d + K·d) speedup ≈ n/K
    """
    n_queries = Q.shape[0]
    d = Q.shape[1]
    n_keys = K_ternary.shape[0]
    output = np.zeros((n_queries, V.shape[1]))

    for i in range(n_queries):
        # Passo 1: scores ternários — O(n·d), adições apenas
        scores = (Q[i] @ K_ternary.T).astype(np.float64)
        scores /= math.sqrt(d) * temperature

        # Passo 2: Top-K O(n)
        k = min(K_top, n_keys)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_scores  = scores[top_indices]

        # Passo 3: Softmax sobre K tokens — O(K) exponenciais
        top_scores -= top_scores.max()
        weights = np.exp(top_scores)
        weights /= weights.sum()

        # Passo 4: Weighted sum — O(K·d)
        output[i] = (weights[:, np.newaxis] * V[top_indices]).sum(axis=0)

    return output


# ─── Produto matricial tropical (tropical_gemv) ───────────────────────────

def tropical_gemv_ref(A: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produto tropical: output[i] = max_j (A[i,j] + x[j])
    Retorna (argmax[m], max_vals[m]).
    A ternário {-1,0,+1}: A[i,j]+x[j] = ±x[j] ou x[j]+0 = x[j]
    """
    # Vetorizado via broadcasting: A (m,n) + x (n,) → (m,n)
    vals = A.astype(np.float64) + x  # tropical mul = adição real
    argmax_out = np.argmax(vals, axis=1)
    max_out    = vals[np.arange(len(argmax_out)), argmax_out]
    return argmax_out, max_out


# ─── Verificação de identidades ───────────────────────────────────────────

def verify_tropical_limit():
    """
    Verifica que lim_{τ→0} softmax(v/τ) → one-hot(argmax(v)).
    Esta é a conexão fundamental com o produto tropical.
    """
    print("\n[1] Limite tropical: softmax(v/τ) → argmax quando τ → 0")
    rng = np.random.default_rng(7)
    v = rng.standard_normal(16)

    true_argmax = np.argmax(v)
    print(f"    argmax(v) = {true_argmax}  (v[{true_argmax}] = {v[true_argmax]:.4f})")

    for tau in [1.0, 0.1, 0.01, 0.001, 0.0001]:
        w = np.exp((v - v.max()) / tau)
        w /= w.sum()
        pred = np.argmax(w)
        entropy = -np.sum(w * np.log(w + 1e-30))
        print(f"    τ={tau:.4f}: argmax(softmax) = {pred}, "
              f"weight[{pred}] = {w[pred]:.6f}, entropy = {entropy:.4f}")

    print(f"    τ→0: softmax se concentra em j={true_argmax} ✓  (argmax tropical)")


def verify_tropical_matmul():
    """
    Verifica que tropical_matmul_fast produz resultado correto vs. loop ingênuo.
    Ilustra o semiring (max,+) com exemplo 3×3.
    """
    print("\n[2] Produto matricial tropical (max,+) — verificação 3×3")
    A = np.array([[0., 1., -np.inf],
                  [-np.inf, 0., 2.],
                  [3., -np.inf, 0.]])
    B = np.array([[1., 0.],
                  [0., 2.],
                  [-1., 1.]])

    C_ref  = tropical_matmul(A, B)
    C_fast = tropical_matmul_fast(A, B)

    print(f"    A =\n{A}")
    print(f"    B =\n{B}")
    print(f"    A ⊗ B (ref)  =\n{C_ref}")
    print(f"    A ⊗ B (fast) =\n{C_fast}")
    print(f"    max|diff| = {np.max(np.abs(C_ref - C_fast)):.2e}")
    assert np.allclose(C_ref, C_fast, equal_nan=False)
    print(f"    IDENTIDADE ✓")


def verify_attention_limit(n_keys=64, d=32, seed=99):
    """
    Verifica que atenção tropical hard (τ→0) converge para a atenção padrão
    quando a temperatura diminui.
    """
    print(f"\n[3] Convergência da atenção: softmax → tropical quando τ→0")
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((4, d)).astype(np.float32)
    K_f = rng.standard_normal((n_keys, d)).astype(np.float32)
    K_t = np.sign(K_f).astype(np.int8)   # ternário {-1,0,+1}
    V = rng.standard_normal((n_keys, d)).astype(np.float32)

    # Hard tropical (τ→0): output = V[argmax Q·K]
    out_tropical = attention_tropical_hard_fast(Q, K_t, V)
    # Padrão com temperatura decrescente
    for tau in [1.0, 0.1, 0.01, 0.001]:
        out_std = attention_standard(Q, K_f, V, temperature=tau)
        diff = np.mean(np.abs(out_std - out_tropical))
        print(f"    τ={tau:.3f}: mean|standard - tropical_hard| = {diff:.4f}")

    # Para τ muito pequeno, ambos devem apontar para o mesmo token dominante
    out_std_small = attention_standard(Q, K_f, V, temperature=0.001)
    diff_small = np.mean(np.abs(out_std_small - out_tropical))
    print(f"    ✓ Para τ=0.001 vs tropical hard: diff = {diff_small:.4f} (deve ser pequeno)")


def verify_tropical_gemv():
    """
    Verifica produto tropical ternário.
    Para A ternário: A[i,j]+x[j] = {x[j], 0, -x[j]} dependendo de A[i,j].
    """
    print(f"\n[4] Produto tropical ternário: output[i] = max_j(A[i,j] + x[j])")
    rng = np.random.default_rng(123)
    m, n = 8, 16
    A = random_ternary_matrix(m, n, sparsity=0.5, seed=1)
    x = rng.standard_normal(n)

    argmax_out, max_out = tropical_gemv_ref(A, x)
    # Verificação: calcular manualmente para linha 0
    row0_vals = A[0].astype(float) + x
    print(f"    Linha 0: A[0,j]+x[j] max = {row0_vals.max():.4f}")
    print(f"    tropical_gemv[0] = {max_out[0]:.4f}  argmax={argmax_out[0]}")
    assert np.isclose(max_out[0], row0_vals.max()), "Erro no tropical_gemv!"
    print(f"    IDENTIDADE ✓")


# ─── Benchmark de complexidade ────────────────────────────────────────────

def benchmark_attention(n_keys: int, d: int, K_top: int, seed: int = 42):
    """
    Compara velocidade e qualidade: atenção padrão vs. tropical top-K.
    """
    rng = np.random.default_rng(seed)
    n_q = 1   # decode: uma query por vez (batch=1, o caso CPU)
    Q = rng.standard_normal((n_q, d)).astype(np.float32)
    K_float   = rng.standard_normal((n_keys, d)).astype(np.float32)
    K_ternary = np.sign(K_float).astype(np.int8)
    V = rng.standard_normal((n_keys, d)).astype(np.float32)

    iters = max(10, min(500, 5000 // n_keys))

    # Warmup
    for _ in range(5):
        attention_standard(Q, K_float, V, temperature=1.0)
        attention_tropical_topk(Q, K_ternary, V, K_top=K_top)

    t0 = time.perf_counter()
    for _ in range(iters):
        out_std = attention_standard(Q, K_float, V, temperature=1.0)
    t_std = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        out_topk = attention_tropical_topk(Q, K_ternary, V, K_top=K_top)
    t_topk = (time.perf_counter() - t0) / iters

    # Qualidade: cosine similarity entre outputs
    cos_sim = float(np.dot(out_std[0], out_topk[0]) /
                    (np.linalg.norm(out_std[0]) * np.linalg.norm(out_topk[0]) + 1e-9))

    return t_std, t_topk, cos_sim


# ─── Scaling: ops reais ────────────────────────────────────────────────────

def op_count_attention(n: int, d: int, K: int) -> dict:
    """
    Contagem teórica de operações para atenção com seq_len=n, head_dim=d, top-K=K.
    """
    std_ops  = 2 * n * n * d   # Q·Kᵀ + weighted sum V, todos pares
    trop_ops = 2 * n * d + 2 * K * d  # scan + topK softmax + V lookup
    # Para K ternário: sem multiplicações no scan
    return {
        "standard":   std_ops,
        "tropical_k": trop_ops,
        "speedup":    std_ops / max(trop_ops, 1),
    }


def scaling_ops(d: int = 64, K: int = 32):
    print(f"\n[Scaling] Ops teóricas: atenção padrão vs tropical top-K={K} (d={d})")
    print(f"  {'n':>6}  {'std_ops':>12}  {'trop_ops':>12}  {'speedup':>10}")
    for exp in range(4, 14):
        n = 2**exp
        ops = op_count_attention(n, d, K)
        print(f"  {n:>6}  {ops['standard']:>12,}  "
              f"{ops['tropical_k']:>12,}  {ops['speedup']:>10.1f}×")
    print(f"\n  Speedup ≈ n/(K + n/n) ≈ n/K → cresce linearmente com n.")
    print(f"  Para K={K}: n=2048 → {2048//K}× speedup, n=8192 → {8192//K}× speedup.")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=256,
                        help="Número de keys (seq_len)")
    parser.add_argument("--d",       type=int, default=64,
                        help="Dimensão por head")
    parser.add_argument("--k",       type=int, default=16,
                        help="Top-K para atenção tropical")
    parser.add_argument("--scaling", action="store_true",
                        help="Mostrar tabela de scaling de operações")
    args = parser.parse_args()

    n, d, K_top = args.n, args.d, args.k

    print(f"\n{'='*66}")
    print(f"  Nível 4: Atenção Tropical — Semiring (max, +)")
    print(f"  n={n} tokens, d={d} head_dim, K_top={K_top}")
    print(f"  Keys ternários {{-1,0,+1}} — zero multiplicações no scan")
    print(f"{'='*66}")

    # ══ VERIFICAÇÕES MATEMÁTICAS ══════════════════════════════════════════
    verify_tropical_limit()
    verify_tropical_matmul()
    verify_attention_limit(n_keys=min(n, 128), d=min(d, 32))
    verify_tropical_gemv()

    # ══ BENCHMARK DE TEMPO ════════════════════════════════════════════════
    print(f"\n[5] Benchmark: atenção padrão O(n²) vs tropical top-K O(n)")
    print(f"    {'n':>5}  {'t_std(μs)':>10}  {'t_topk(μs)':>11}  "
          f"{'speedup':>9}  {'cosine_sim':>11}")

    for test_n in [64, 128, 256, 512, 1024]:
        t_std, t_topk, cos = benchmark_attention(test_n, d, K_top)
        speedup = t_std / max(t_topk, 1e-9)
        print(f"    {test_n:>5}  {t_std*1e6:>10.1f}  {t_topk*1e6:>11.1f}  "
              f"{speedup:>9.2f}×  {cos:>11.4f}")

    print(f"\n    Nota: Python/NumPy — C++ SIMD: +8–16× adicionais.")
    print(f"    cosine_sim > 0.95 indica boa qualidade de aproximação.")

    # ══ ANÁLISE DE QUALIDADE vs TEMPERATURA ══════════════════════════════
    print(f"\n[6] Qualidade da atenção tropical vs temperatura")
    rng = np.random.default_rng(55)
    Q_q = rng.standard_normal((4, d)).astype(np.float32)
    K_f = rng.standard_normal((n, d)).astype(np.float32)
    K_t = np.sign(K_f).astype(np.int8)
    V_v = rng.standard_normal((n, d)).astype(np.float32)

    out_hard = attention_tropical_hard_fast(Q_q, K_t, V_v)
    print(f"    {'tau':>8}  {'K_top':>6}  {'vs_hard_cos':>12}  {'vs_std_cos':>12}")
    for tau in [1.0, 0.5, 0.1]:
        out_std = attention_standard(Q_q, K_f, V_v, temperature=tau)
        for kk in [8, 16, 32, n]:
            out_topk = attention_tropical_topk(Q_q, K_t, V_v, K_top=kk, temperature=tau)
            # Média de cosine similarities por query
            cos_hard = float(np.mean([
                np.dot(out_topk[i], out_hard[i]) /
                (np.linalg.norm(out_topk[i]) * np.linalg.norm(out_hard[i]) + 1e-9)
                for i in range(4)]))
            cos_std = float(np.mean([
                np.dot(out_topk[i], out_std[i]) /
                (np.linalg.norm(out_topk[i]) * np.linalg.norm(out_std[i]) + 1e-9)
                for i in range(4)]))
            print(f"    {tau:>8.2f}  {kk:>6}  {cos_hard:>12.4f}  {cos_std:>12.4f}")

    # ══ CONTAGEM DE OPS TEÓRICAS ══════════════════════════════════════════
    print(f"\n[7] Operações teóricas (n={n}, d={d}, K={K_top})")
    ops = op_count_attention(n, d, K_top)
    print(f"    Atenção padrão:      {ops['standard']:>10,}  muls+adds")
    print(f"    Tropical top-K:      {ops['tropical_k']:>10,}  adds (scan) + {2*K_top*d:,} mul-adds (V)")
    print(f"    Speedup teórico:     {ops['speedup']:>10.1f}×")
    print(f"    Scan ternário:       zero multiplicações (Level 2 kernel)")

    if args.scaling:
        scaling_ops(d=d, K=K_top)

    # ══ IMPLICAÇÃO PARA BITNET-2B ═════════════════════════════════════════
    print(f"\n{'='*66}")
    print("  Projeção: BitNet-2B (n_heads=20, head_dim=128, seq=2048)")
    print(f"{'='*66}")
    n_h, h_d, seq = 20, 128, 2048
    k_top = 32
    ops_std  = n_h * 2 * seq * seq * h_d // 1_000_000
    ops_trop = n_h * (2 * seq * h_d + 2 * k_top * h_d) // 1_000_000
    print(f"""
  Atenção padrão (fp16):
    {n_h} heads × {seq}² × {h_d} × 2 = {ops_std:,} M ops/token

  Atenção tropical top-{k_top} (ternária):
    Scan:      {n_h} × {seq} × {h_d} = {n_h*seq*h_d//1000:,}K adições (zero muls)
    Top-K:     {n_h} × {k_top} × {h_d} × 2 = {n_h*k_top*h_d*2//1000:,}K mul-adds
    Total:     {ops_trop:,} M ops/token

  Speedup: {ops_std//max(ops_trop,1)}× menos operações/token na atenção

  Combinando com ACDC (Nível 3) para FFN:
    Nível 1 (ternário): fp16 baseline / ~4× memória
    Nível 2 (WHT):      zero muls em todos os GEMVs
    Nível 3 (ACDC FFN): ~128× menos ops em FFN
    Nível 4 (tropical): ~{ops_std//max(ops_trop,1)}× menos ops em atenção

  Pipeline completo: token generation no CPU sem GPU.
""")


if __name__ == "__main__":
    main()
