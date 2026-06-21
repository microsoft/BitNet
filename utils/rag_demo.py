#!/usr/bin/env python3
"""
rag_demo.py — CPU-RAG reference demo (Direção E, Level 6)

Demonstrates the same flat-index ANN algorithm as ggml-bitnet-rag.cpp using
NumPy.  No model download required; all operations run CPU-only.

Usage (numpy path — no build needed):
    python utils/rag_demo.py

Usage (ctypes path — requires shared library):
    cmake -B build -DBITNET_L6_RAG=ON -DBITNET_RAG_SHARED=ON
    cmake --build build --target bitnet_rag
    python utils/rag_demo.py --lib build/lib/libbitnet_rag.so

Algorithm (matches ggml-bitnet-rag.cpp exactly):
    score(q, doc) = (q · doc) / sqrt(d)
    top-K:  partial sort by score, descending
    adaptive K: cumulative softmax threshold (same as tropical_adaptive_k)
"""

import argparse
import ctypes
import os
import sys
import time
import numpy as np


# ─── NumPy reference implementation (always available) ────────────────────

class RagStoreNumpy:
    """Pure-NumPy RAG flat-index store.  Matches the C API in ggml-bitnet-rag.h."""

    def __init__(self, d: int):
        self.d = d
        self.embeddings: list[np.ndarray] = []

    def add(self, embedding: np.ndarray) -> int:
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        assert len(emb) == self.d, f"dim mismatch: got {len(emb)}, expected {self.d}"
        doc_id = len(self.embeddings)
        self.embeddings.append(emb.copy())
        return doc_id

    def _score_all(self, query: np.ndarray) -> np.ndarray:
        if not self.embeddings:
            return np.empty(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).ravel()
        E = np.stack(self.embeddings)                    # [n, d]
        inv_sqrt_d = 1.0 / np.sqrt(float(self.d))
        return (E @ q) * inv_sqrt_d                      # [n] dot products

    def retrieve_topk(self, query: np.ndarray, k: int):
        scores = self._score_all(query)
        n = len(scores)
        K = min(k, n)
        if K == 0:
            return [], []
        # argpartition + sort for top-K (same complexity as std::partial_sort)
        if K < n:
            part = np.argpartition(scores, -K)[-K:]
        else:
            part = np.arange(n)
        order = np.argsort(-scores[part])
        ids = part[order].tolist()
        sc  = scores[part[order]].tolist()
        return ids, sc

    def retrieve_adaptive(self, query: np.ndarray,
                          coverage: float = 0.90,
                          k_min: int = 1,
                          k_max: int = 32):
        scores = self._score_all(query)
        n = len(scores)
        K_limit = min(k_max, n)
        k_min = max(1, min(k_min, K_limit))

        # Partial sort: top K_limit
        if K_limit < n:
            part = np.argpartition(scores, -K_limit)[-K_limit:]
        else:
            part = np.arange(n)
        order = np.argsort(-scores[part])
        top_ids    = part[order]
        top_scores = scores[top_ids]

        # Cumulative softmax
        s_max = top_scores[0]
        w = np.exp(top_scores - s_max)
        w_norm = w / w.sum()
        cum = np.cumsum(w_norm)

        K_chosen = K_limit
        if coverage < 1.0:
            exceed = np.where(cum >= coverage)[0]
            if len(exceed) > 0:
                K_chosen = int(exceed[0]) + 1
        K_chosen = max(k_min, K_chosen)

        return top_ids[:K_chosen].tolist(), top_scores[:K_chosen].tolist()


# ─── ctypes bridge (optional — needs libbitnet_rag.so) ────────────────────

class RagStoreCTypes:
    """ctypes wrapper around ggml-bitnet-rag C API."""

    def __init__(self, lib_path: str, capacity: int, d: int):
        self._lib = ctypes.CDLL(lib_path)
        self.d = d
        self._setup_prototypes()
        self._ptr = self._lib.rag_store_create(capacity, d)
        if not self._ptr:
            raise RuntimeError("rag_store_create returned NULL")

    def _setup_prototypes(self):
        lib = self._lib
        vp  = ctypes.c_void_p
        f   = ctypes.c_float
        i   = ctypes.c_int
        fp  = ctypes.POINTER(ctypes.c_float)
        ip  = ctypes.POINTER(ctypes.c_int)

        lib.rag_store_create.restype  = vp
        lib.rag_store_create.argtypes = [i, i]
        lib.rag_store_free.restype    = None
        lib.rag_store_free.argtypes   = [vp]
        lib.rag_store_add.restype     = i
        lib.rag_store_add.argtypes    = [vp, fp]
        lib.rag_retrieve_topk.restype = i
        lib.rag_retrieve_topk.argtypes = [vp, fp, i, ip, fp]
        lib.rag_retrieve_adaptive.restype  = i
        lib.rag_retrieve_adaptive.argtypes = [vp, fp, f, i, i, ip, fp]
        lib.rag_store_n_docs.restype  = i
        lib.rag_store_n_docs.argtypes = [vp]

    def add(self, embedding: np.ndarray) -> int:
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        return self._lib.rag_store_add(
            self._ptr, emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def retrieve_topk(self, query: np.ndarray, k: int):
        q   = np.ascontiguousarray(query, dtype=np.float32)
        ids = (ctypes.c_int   * k)()
        sc  = (ctypes.c_float * k)()
        n = self._lib.rag_retrieve_topk(
            self._ptr, q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k, ids, sc)
        return list(ids[:n]), list(sc[:n])

    def retrieve_adaptive(self, query: np.ndarray,
                          coverage: float = 0.90,
                          k_min: int = 1,
                          k_max: int = 32):
        q   = np.ascontiguousarray(query, dtype=np.float32)
        ids = (ctypes.c_int   * k_max)()
        sc  = (ctypes.c_float * k_max)()
        n = self._lib.rag_retrieve_adaptive(
            self._ptr, q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(coverage), k_min, k_max, ids, sc)
        return list(ids[:n]), list(sc[:n])

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            self._lib.rag_store_free(self._ptr)


# ─── Demo ─────────────────────────────────────────────────────────────────

def run_demo(store_cls, **kwargs):
    rng = np.random.default_rng(0xB177E742)
    d, N = 256, 1000

    print(f"\n{'═'*60}")
    print(f"  CPU-RAG Demo — {store_cls.__name__}")
    print(f"  {N} docs × d={d}, dtype=float32")
    print(f"{'═'*60}")

    # Build corpus
    corpus = rng.standard_normal((N, d)).astype(np.float32)
    # Normalize for cosine-like ranking
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8

    if store_cls is RagStoreCTypes:
        store = store_cls(kwargs['lib_path'], capacity=N + 1, d=d)
    else:
        store = store_cls(d=d)

    t0 = time.perf_counter()
    for i in range(N):
        store.add(corpus[i])
    t_index = time.perf_counter() - t0
    print(f"  Indexed {N} docs in {t_index*1000:.2f} ms")

    # Fixed-K retrieval: query = doc[42] → should be rank-0
    target = 42
    t0 = time.perf_counter()
    ids, sc = store.retrieve_topk(corpus[target], k=5)
    t_topk = time.perf_counter() - t0
    ok = ids[0] == target
    print(f"\n  Fixed-K (k=5) — query = doc[{target}]:")
    print(f"    ids={ids}, scores={[f'{s:.4f}' for s in sc]}")
    print(f"    rank-0 correct: {'YES ✓' if ok else 'NO ✗'}  ({t_topk*1000:.3f} ms)")

    # Adaptive-K: concentrated query (exact doc) → small K
    ids_a, sc_a = store.retrieve_adaptive(corpus[target],
                                          coverage=0.90, k_min=1, k_max=32)
    print(f"\n  Adaptive-K (coverage=0.90, k_min=1, k_max=32):")
    print(f"    K chosen={len(ids_a)}, top_id={ids_a[0]}, score={sc_a[0]:.4f}")

    # Throughput: 100 random queries
    queries = corpus[rng.integers(0, N, size=100)]
    t0 = time.perf_counter()
    for q in queries:
        store.retrieve_topk(q, k=10)
    t_batch = time.perf_counter() - t0
    print(f"\n  Throughput: 100 queries × k=10 → {t_batch*1000:.1f} ms total "
          f"({t_batch/100*1000:.2f} ms/query)")
    print()


def main():
    ap = argparse.ArgumentParser(description="CPU-RAG Direção E demo")
    ap.add_argument("--lib", default=None,
                    help="path to libbitnet_rag.so (ctypes path; omit for numpy)")
    args = ap.parse_args()

    # Always run numpy reference
    run_demo(RagStoreNumpy)

    if args.lib:
        if not os.path.exists(args.lib):
            print(f"[WARN] shared library not found: {args.lib}", file=sys.stderr)
            print("       Build with: cmake -B build -DBITNET_L6_RAG=ON "
                  "-DBITNET_RAG_SHARED=ON && cmake --build build --target bitnet_rag")
        else:
            run_demo(RagStoreCTypes, lib_path=args.lib)
    else:
        print("Tip: run with --lib build/lib/libbitnet_rag.so to benchmark the C kernel.")


if __name__ == "__main__":
    main()
