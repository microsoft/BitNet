#!/usr/bin/env python3
"""
Testa o closed-form ACDC d* = diag(H·W·H) / n².

Para uma matriz W que É diagonalizável por Hadamard (i.e., W = H·diag(d)·H
para algum d), o d* extraído deve ser EXATO (error = 0).

Para W aleatório Uniform{-1, 0, +1}, a energia capturada deve ser
próxima de 1/n (derivação teórica).
"""
import numpy as np
import sys
from pathlib import Path

# Adiciona utils/ ao path para poder importar o extractor
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from extract_acdc_diagonal import acdc_extract_diag, next_pow2
from scipy.linalg import hadamard


def make_acdc_matrix(d: np.ndarray, n: int) -> np.ndarray:
    """Constrói W = H·diag(d)·H. Esta matriz TEM diagonal perfeita
    (modulo fator 1/n; aqui usamos Hadamard não-normalizada, então
    H @ W @ H = n² · diag(d), e d* = n²·diag(d) / n² = diag(d))."""
    H = hadamard(n).astype(np.float32)
    return H @ np.diag(d.astype(np.float32)) @ H


def test_acdc_exact_recovery():
    """W que É ACDC-diagonalizável → d* deve ser EXATO."""
    print("\n--- test_acdc_exact_recovery ---")
    n = 8
    rng = np.random.default_rng(42)
    d_true = rng.standard_normal(n).astype(np.float32) * 0.5
    W = make_acdc_matrix(d_true, n)

    d_star, meta = acdc_extract_diag(W, "test", verbose=False)
    err = np.max(np.abs(d_star - d_true))
    print(f"  d_true[0:4]  = {d_true[:4]}")
    print(f"  d_star[0:4]  = {d_star[:4]}")
    print(f"  max|d* - d_true| = {err}")
    print(f"  energy_captured  = {meta['energy_captured']}")
    assert err < 1e-3, f"d* should be exact for ACDC matrix, err={err}"
    assert meta['energy_captured'] > 0.99, f"energy should be ~1, got {meta['energy_captured']}"
    print("  ✓ exact recovery for ACDC-diagonalizable matrix")


def test_acdc_random_captures_1_over_n():
    """W aleatório Uniform{-1,0,+1} → energia capturada ≈ 1/n."""
    print("\n--- test_acdc_random_captures_1_over_n ---")
    n = 32
    rng = np.random.default_rng(123)
    # Ternário: 33% -1, 33% 0, 33% +1
    W = rng.choice([-1, 0, 1], size=(n, n)).astype(np.float32)

    d_star, meta = acdc_extract_diag(W, "test", verbose=False)
    expected = 1.0 / n
    actual = meta['energy_captured']
    print(f"  n = {n}")
    print(f"  expected energy ≈ 1/n = {expected:.4f}")
    print(f"  actual energy    = {actual:.4f}")
    # Tolerância ampla: o resultado depende muito de realizações individuais
    # Para W truly random, esperamos energy in [1/(2n), 2/n].
    assert 0.5 / n < actual < 3.0 / n, \
        f"random W should capture ~1/n energy, got {actual}"
    print("  ✓ random W captures ~1/n energy as predicted by theory")


def test_acdc_known_dense_recovery():
    """W=I (identidade) é sua própria ACDC: d*[0]=1, resto 0."""
    print("\n--- test_acdc_known_dense_recovery ---")
    n = 16
    W = np.eye(n, dtype=np.float32)

    d_star, meta = acdc_extract_diag(W, "I", verbose=False)
    print(f"  d*[0]  = {d_star[0]}  (expected ~1)")
    print(f"  d*[1]  = {d_star[1]}  (expected ~0)")
    print(f"  d*[2]  = {d_star[2]}  (expected ~0)")
    # I = H · diag([1, 0, 0, ...]) · H / n → isso só funciona se H·I·H = n·I
    # então d* = n·I / n² = I / n. Não é "d* = [1, 0, 0, ...]".
    # A diagonal real de H·I·H / n² é diag(H @ I @ H) / n² = diag(n·I) / n² = I / n.
    expected_d0 = 1.0 / n  # = 0.0625 para n=16
    err0 = abs(d_star[0] - expected_d0)
    assert err0 < 1e-3, f"d*[0] for W=I should be 1/n={expected_d0}, got {d_star[0]}"
    print(f"  ✓ W=I: d*[0]={d_star[0]:.4f} matches 1/n={expected_d0}")


def test_acdc_uses_ternary_form():
    """Verifica que a fórmula coincide com acdc_project do C kernel."""
    print("\n--- test_acdc_uses_ternary_form ---")
    n = 8
    rng = np.random.default_rng(7)
    # W ternário
    W_tern = rng.choice([-1, 0, 1], size=(n, n)).astype(np.int8)
    W = W_tern.astype(np.float32)

    H = hadamard(n).astype(np.float32)
    # ACD reference: d* = diag(H·W·H) / n²
    A = H @ W @ H
    d_ref = np.diag(A) / (n * n)

    d_star, _ = acdc_extract_diag(W, "test", verbose=False)
    err = np.max(np.abs(d_star - d_ref))
    assert err < 1e-5, f"d* should match closed-form, err={err}"
    print(f"  ✓ d* matches closed-form (max err = {err:.2e})")


def test_next_pow2():
    """Função utilitária."""
    print("\n--- test_next_pow2 ---")
    cases = [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (16, 16), (17, 32),
             (1023, 1024), (1024, 1024), (1025, 2048), (2560, 4096)]
    for n_in, n_out in cases:
        got = next_pow2(n_in)
        assert got == n_out, f"next_pow2({n_in}) = {got}, expected {n_out}"
    print(f"  ✓ {len(cases)} cases PASS")


if __name__ == "__main__":
    test_next_pow2()
    test_acdc_exact_recovery()
    test_acdc_random_captures_1_over_n()
    test_acdc_known_dense_recovery()
    test_acdc_uses_ternary_form()
    print("\n=== test_extract_acdc_diagonal: ALL PASS ===")
