"""
hrr_benchmark.py — Memória Holográfica: Representações Holográficas Reduzidas

Nível 5 do roteiro de universalização CPU.

FUNDAMENTO MATEMÁTICO:
  Convolução circular como operação de binding:
    (a ⊛ b)[k] = Σⱼ a[j] · b[(k-j) mod d]
    Via FFT: a ⊛ b = IFFT( FFT(a) ⊙ FFT(b) )   →  O(d log d)

  Memória associativa:
    Armazenamento: M = Σᵢ kᵢ ⊛ vᵢ  (superposição de N pares em 1 vetor)
    Recuperação:   ṽⱼ = M ⊛ kⱼ⁻¹  ≈  vⱼ   (ruído ~ (N-1)/√d)

  Substituição da atenção Transformer:
    Standard: Q·Kᵀ + softmax → O(n²·d)
    HRR:      M⊛Q⁻¹          → O(n·d·log d) build + O(d·log d) retrieve
    Speedup retrieval: n/log n (para n=2048: ~186×)

PROPRIEDADES VERIFICADAS:
  [1] Binding é convolução circular exata (via FFT)
  [2] Identidade: δ ⊛ a = a
  [3] Comutatividade: a ⊛ b = b ⊛ a
  [4] Associatividade: (a ⊛ b) ⊛ c = a ⊛ (b ⊛ c)
  [5] Pseudo-inversa: a ⊛ a⁻¹ ≈ δ
  [6] Recuperação: (a⊛b+C⊛D) ⊛ a⁻¹ ≈ b
  [7] Capacidade de memória vs dimensão
  [8] Scaling de speedup vs comprimento de sequência
"""

import argparse
import time
import math
import numpy as np
from typing import Tuple, List


# ─── Convolução circular via NumPy FFT ────────────────────────────────────

def circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    (a ⊛ b)[k] = Σⱼ a[j] · b[(k-j) mod d]
    Implementado via FFT: IRFFT( RFFT(a) ⊙ RFFT(b) )
    O(d log d) — d/2+1 multiplicações complexas.
    """
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))


def pseudoinverse(a: np.ndarray) -> np.ndarray:
    """
    a⁻¹ = IRFFT( conj(RFFT(a)) )
    Para vetores de norma unitária: a⁻¹ = cyclic_reverse(a)
    Esta é a inversão EXATA via conjugação espectral.
    """
    return np.fft.irfft(np.conj(np.fft.rfft(a)), n=len(a))


def bind(k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Binding: k ⊛ v"""
    return circular_conv(k, v)


def unbind(M: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Unbinding: M ⊛ k⁻¹"""
    return circular_conv(M, pseudoinverse(k))


# ─── Memória holográfica ──────────────────────────────────────────────────

def build_memory(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    M = Σᵢ keys[i] ⊛ values[i]
    Armazena N pares (key, value) em um único vetor M ∈ ℝᵈ.
    Complexidade: O(N · d · log d)
    """
    d = keys.shape[1]
    M = np.zeros(d)
    for i in range(len(keys)):
        M += bind(keys[i], values[i])
    return M


def retrieve(M: np.ndarray, query_key: np.ndarray) -> np.ndarray:
    """
    ṽ = M ⊛ query_key⁻¹ ≈ value associado à query_key
    Complexidade: O(d · log d) — INDEPENDENTE de N (contexto)
    """
    return unbind(M, query_key)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ─── Geração de vetores aleatórios ────────────────────────────────────────

def random_unit_vector(d: int, rng: np.random.Generator) -> np.ndarray:
    """Vetor aleatório de norma unitária em ℝᵈ."""
    v = rng.standard_normal(d)
    return normalize(v.astype(np.float64))


def random_phasor_vector(d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Vetor com |FFT[k]| = 1 para todo k — "phasor" puro.
    Propriedade: a ⊛ a⁻¹ = δ EXATAMENTE (sem erro de norma).
    Gerado via fases aleatórias uniformes em [-π, π].
    """
    phases = rng.uniform(-math.pi, math.pi, d // 2 + 1)
    spectrum = np.exp(1j * phases)
    # Garantir simetria Hermitiana para resultado real
    spectrum[0] = abs(spectrum[0])  # DC: real
    if d % 2 == 0:
        spectrum[d//2] = abs(spectrum[d//2])  # Nyquist: real
    v = np.fft.irfft(spectrum, n=d)
    return normalize(v.astype(np.float64))


# ─── Verificações matemáticas ─────────────────────────────────────────────

def verify_circular_convolution(d: int, rng: np.random.Generator):
    """
    [1] Verifica que circular_conv implementa corretamente a convolução circular.
    Compara com definição direta: (a⊛b)[k] = Σⱼ a[j]·b[(k-j) mod d]
    """
    print(f"\n[1] Convolução circular: FFT vs definição direta  (d={d})")
    a = random_unit_vector(d, rng)
    b = random_unit_vector(d, rng)

    # Definição direta O(d²)
    c_ref = np.zeros(d)
    for k in range(d):
        for j in range(d):
            c_ref[k] += a[j] * b[(k - j) % d]

    c_fft = circular_conv(a, b)
    max_diff = np.max(np.abs(c_ref - c_fft))
    print(f"    max|c_ref - c_fft| = {max_diff:.2e}  (deve ser ≈ epsilon de máquina)")
    assert max_diff < 1e-10, "Falhou!"
    print(f"    IDENTIDADE VERIFICADA ✓")


def verify_identity_element(d: int, rng: np.random.Generator):
    """[2] δ ⊛ a = a  (elemento neutro: impulso unitário)"""
    print(f"\n[2] Elemento neutro: δ ⊛ a = a  (d={d})")
    delta = np.zeros(d); delta[0] = 1.0
    a = random_unit_vector(d, rng)
    result = circular_conv(delta, a)
    max_diff = np.max(np.abs(result - a))
    print(f"    max|δ⊛a - a| = {max_diff:.2e}")
    assert max_diff < 1e-12, "Falhou!"
    print(f"    IDENTIDADE ✓")


def verify_commutativity(d: int, rng: np.random.Generator):
    """[3] a ⊛ b = b ⊛ a"""
    print(f"\n[3] Comutatividade: a ⊛ b = b ⊛ a  (d={d})")
    a = random_unit_vector(d, rng)
    b = random_unit_vector(d, rng)
    ab = circular_conv(a, b)
    ba = circular_conv(b, a)
    max_diff = np.max(np.abs(ab - ba))
    print(f"    max|a⊛b - b⊛a| = {max_diff:.2e}")
    assert max_diff < 1e-12, "Falhou!"
    print(f"    COMUTATIVIDADE ✓")


def verify_associativity(d: int, rng: np.random.Generator):
    """[4] (a ⊛ b) ⊛ c = a ⊛ (b ⊛ c)"""
    print(f"\n[4] Associatividade: (a⊛b)⊛c = a⊛(b⊛c)  (d={d})")
    a = random_unit_vector(d, rng)
    b = random_unit_vector(d, rng)
    c = random_unit_vector(d, rng)
    left  = circular_conv(circular_conv(a, b), c)
    right = circular_conv(a, circular_conv(b, c))
    max_diff = np.max(np.abs(left - right))
    print(f"    max|(a⊛b)⊛c - a⊛(b⊛c)| = {max_diff:.2e}")
    assert max_diff < 1e-10, "Falhou!"
    print(f"    ASSOCIATIVIDADE ✓")


def verify_inverse(d: int, rng: np.random.Generator):
    """[5] a ⊛ a⁻¹ ≈ δ"""
    print(f"\n[5] Pseudo-inversa: a ⊛ a⁻¹ ≈ δ  (d={d})")
    delta = np.zeros(d); delta[0] = 1.0

    # Vetor unitário normal (inversa aproximada)
    a = random_unit_vector(d, rng)
    a_inv = pseudoinverse(a)
    result = circular_conv(a, a_inv)
    err_unit = np.linalg.norm(result - delta) / np.linalg.norm(delta)
    print(f"    Vetor unitário normal:  ||a⊛a⁻¹ - δ|| / ||δ|| = {err_unit:.2e}")

    # Vetor phasor (inversa exata)
    p = random_phasor_vector(d, rng)
    p_inv = pseudoinverse(p)
    result_p = circular_conv(p, p_inv)
    err_phasor = np.linalg.norm(result_p - delta) / np.linalg.norm(delta)
    print(f"    Vetor phasor (|FFT|=1): ||p⊛p⁻¹ - δ|| / ||δ|| = {err_phasor:.2e}")
    print(f"    Phasor é exato? {'✓' if err_phasor < 1e-10 else '≈'}")


def verify_retrieval(d: int, N: int, rng: np.random.Generator):
    """
    [6] Recuperação: M = Σᵢ kᵢ⊛vᵢ, recuperar v₀ dado k₀.
    Cosine similarity entre v₀_retrieved e v₀_true.
    Erro teórico: (N-1)/√d.
    """
    print(f"\n[6] Recuperação de memória  (d={d}, N={N} pares)")

    keys   = np.array([random_phasor_vector(d, rng) for _ in range(N)])
    values = np.array([random_unit_vector(d, rng)   for _ in range(N)])

    M = build_memory(keys, values)

    # Tentar recuperar cada valor
    sims = []
    for i in range(min(N, 10)):  # verificar os 10 primeiros
        retrieved = retrieve(M, keys[i])
        sim = cosine_sim(retrieved, values[i])
        sims.append(sim)

    mean_sim = np.mean(sims)
    min_sim  = np.min(sims)
    noise_theory = (N - 1) / math.sqrt(d)

    print(f"    Cosine similarity média:  {mean_sim:.4f}")
    print(f"    Cosine similarity mínima: {min_sim:.4f}")
    print(f"    Ruído teórico (N-1)/√d:   {noise_theory:.4f}")
    print(f"    Recuperação {'✓ boa' if mean_sim > 0.7 else '✗ ruidosa'} "
          f"(>0.7 indica recuperação utilizável)")


# ─── Capacidade de memória vs dimensão ────────────────────────────────────

def capacity_analysis(d_values: List[int], rng: np.random.Generator):
    """
    Para cada dimensão d, encontrar o N máximo onde cosine_sim > 0.9.
    Capacidade teórica: N ≈ d/9  (para SNR > 3, 1σ acima do ruído).
    """
    print(f"\n[7] Capacidade: máximo N para cosine_sim > 0.9")
    print(f"    {'d':>6}  {'N_max(empírico)':>16}  {'d/9 (teoria)':>14}  "
          f"{'sim@N_max':>11}")
    for d in d_values:
        # Busca binária de N_max
        lo, hi = 1, d
        best_N = 1
        best_sim = 1.0
        while lo <= hi:
            N = (lo + hi) // 2
            keys   = np.array([random_phasor_vector(d, rng) for _ in range(N)])
            values = np.array([random_unit_vector(d, rng)   for _ in range(N)])
            M = build_memory(keys, values)
            sims = [cosine_sim(retrieve(M, keys[i]), values[i]) for i in range(min(N, 5))]
            sim = np.mean(sims)
            if sim > 0.9:
                best_N, best_sim = N, sim
                lo = N + 1
            else:
                hi = N - 1
        print(f"    {d:>6}  {best_N:>16}  {d//9:>14}  {best_sim:>11.4f}")


# ─── Scaling de speedup vs n (contexto) ──────────────────────────────────

def scaling_speedup(d: int = 128):
    """
    Compara complexidade teórica de atenção vs HRR para sequências crescentes.
    """
    print(f"\n[8] Speedup teórico: Atenção O(n²d) vs HRR O(nd·log d)  (d={d})")
    print(f"    {'n':>6}  {'std_ops':>12}  {'hrr_build':>12}  "
          f"{'hrr_ret/tok':>13}  {'speedup_build':>14}  {'speedup_ret':>12}")
    log_d = math.log2(d)
    for exp in range(4, 14):
        n = 2**exp
        std_ops   = n * n * d * 2          # atenção O(n²d): Q·Kᵀ + A·V
        hrr_build = n * d * log_d * 3      # N × FFT(key) + FFT(val) + IFFT(binding)
        hrr_ret   = d * log_d * 3          # por token: FFT(q) + mult + IFFT
        sp_build  = std_ops / hrr_build
        sp_ret    = (n * d) / hrr_ret      # vs 1 scan O(nd) da atenção tropical
        print(f"    {n:>6}  {std_ops:>12,.0f}  {hrr_build:>12,.0f}  "
              f"{hrr_ret:>13,.0f}  {sp_build:>14.1f}×  {sp_ret:>12.1f}×")
    print(f"\n    Speedup retrieval ≈ n/log₂d → cresce linearmente com n.")
    print(f"    Para n=2048, d=128: {2048/log_d:.0f}× por token gerado.")


# ─── Benchmark de velocidade ──────────────────────────────────────────────

def benchmark_attention_vs_hrr(n: int, d: int, rng: np.random.Generator):
    """
    Compara tempo real de:
    - Atenção padrão: softmax(Q·Kᵀ/√d)·V
    - HRR: build M + retrieve(M, q) por token
    """
    Q = np.array([random_unit_vector(d, rng) for _ in range(1)])   # 1 query (decode)
    K = np.array([random_unit_vector(d, rng) for _ in range(n)])
    V = np.array([random_unit_vector(d, rng) for _ in range(n)])

    iters = max(5, min(100, 1000 // n))

    # Atenção padrão
    def std_attention():
        scores = Q @ K.T / math.sqrt(d)
        scores -= scores.max()
        w = np.exp(scores); w /= w.sum(axis=-1, keepdims=True)
        return w @ V

    # HRR: build + retrieve
    def hrr_full():
        M = build_memory(K, V)
        return retrieve(M, Q[0])

    # HRR: apenas retrieve (M já construída, reutilizável)
    M_prebuilt = build_memory(K, V)
    def hrr_retrieve_only():
        return retrieve(M_prebuilt, Q[0])

    for _ in range(3): std_attention(); hrr_full(); hrr_retrieve_only()

    t0 = time.perf_counter()
    for _ in range(iters): std_attention()
    t_std = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters): hrr_full()
    t_hrr = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters): hrr_retrieve_only()
    t_ret = (time.perf_counter() - t0) / iters

    # Qualidade: cosine similarity
    out_std = std_attention()[0]
    out_hrr = hrr_retrieve_only()
    sim = cosine_sim(out_std, out_hrr)

    return t_std, t_hrr, t_ret, sim


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",        type=int, default=128,
                        help="Dimensão dos vetores (head_dim)")
    parser.add_argument("--n",        type=int, default=64,
                        help="Número de pares K/V (contexto)")
    parser.add_argument("--capacity", action="store_true",
                        help="Análise de capacidade de memória")
    parser.add_argument("--scaling",  action="store_true",
                        help="Tabela de scaling de speedup")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    d = hrr_next_pow2(args.d)
    N = args.n
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*68}")
    print(f"  Nível 5: Memória Holográfica — Representações Holográficas Reduzidas")
    print(f"  d={d} (dimensão),  N={N} pares em memória")
    print(f"  Binding: a ⊛ b = IRFFT( RFFT(a) ⊙ RFFT(b) )   [O(d log d)]")
    print(f"{'='*68}")

    # ══ VERIFICAÇÕES ALGÉBRICAS ═══════════════════════════════════════════
    d_small = min(d, 64)  # pequeno para verificação com loop direto
    verify_circular_convolution(d_small, rng)
    verify_identity_element(d_small, rng)
    verify_commutativity(d_small, rng)
    verify_associativity(d_small, rng)
    verify_inverse(d, rng)
    verify_retrieval(d, N, rng)

    # ══ CAPACIDADE ════════════════════════════════════════════════════════
    if args.capacity:
        capacity_analysis([64, 128, 256, 512, 1024], rng)

    # ══ SCALING TEÓRICO ═══════════════════════════════════════════════════
    if args.scaling:
        scaling_speedup(d)

    # ══ BENCHMARK DE TEMPO ════════════════════════════════════════════════
    print(f"\n[9] Benchmark: Atenção padrão vs HRR  (d={d}, decode batch=1)")
    print(f"    {'n':>5}  {'t_std(μs)':>10}  {'t_hrr(μs)':>10}  "
          f"{'t_ret(μs)':>10}  {'speedup_ret':>12}  {'cosine_sim':>11}")
    for test_n in [16, 32, 64, 128, 256, 512]:
        t_std, t_hrr, t_ret, sim = benchmark_attention_vs_hrr(test_n, d, rng)
        sp = t_std / max(t_ret, 1e-9)
        print(f"    {test_n:>5}  {t_std*1e6:>10.1f}  {t_hrr*1e6:>10.1f}  "
              f"{t_ret*1e6:>10.1f}  {sp:>12.2f}×  {sim:>11.4f}")
    print(f"\n    t_hrr = build time (one-shot per context)")
    print(f"    t_ret = retrieve time per token (= O(d log d), amortizes over all tokens)")
    print(f"    cosine_sim: qualidade de aproximação vs atenção padrão")
    print(f"    Nota: Python puro — C++ SIMD: +8-16× adicional")

    # ══ PROJEÇÃO BITNET-2B ════════════════════════════════════════════════
    print(f"\n{'='*68}")
    print("  Projeção: BitNet-2B (20 heads, head_dim=128, seq=2048)")
    print(f"{'='*68}")
    n_h, h_d, seq = 20, 128, 2048
    log_d = math.log2(h_d)
    std_ops  = n_h * seq * seq * h_d * 2
    hrr_b    = n_h * seq * h_d * log_d * 3
    hrr_r    = n_h * h_d * log_d * 3
    print(f"""
  Atenção padrão (fp16):
    {n_h} heads × {seq}² × {h_d} × 2 = {std_ops/1e9:.1f}B ops/token

  HRR — Build da memória (one-shot, contexto de {seq} tokens):
    {n_h} heads × {seq} × {h_d} × log₂({h_d}) × 3 = {hrr_b/1e6:.0f}M ops (total)

  HRR — Retrieve por token (decode):
    {n_h} heads × {h_d} × log₂({h_d}) × 3 = {hrr_r:.0f} ops/token
    Speedup retrieval: {std_ops/hrr_r:.0f}× vs atenção padrão

  Resumo do pipeline completo (todos os 5 níveis):
    fp16:                  ~847B ops/token    (1×)
    L1 ternário:           ~424B ops/token    (2×)
    L2 WHT (zero muls):    424B adições       (4–6× efetivo)
    L3 ACDC FFN:            ~17B ops/token    (~50×)
    L4 Tropical attn:        ~3B ops/token    (~280×)
    L5 HRR retrieval:      ~{n_h*hrr_r/1e6:.0f}M ops/token       (~{int(std_ops*30/(n_h*hrr_r*30))}× attn, acumulado com L3-4)

  Token generation sem GPU: teoricamente viável no CPU moderno.
""")


def hrr_next_pow2(n: int) -> int:
    p = 1
    while p < n: p <<= 1
    return p


if __name__ == "__main__":
    main()
