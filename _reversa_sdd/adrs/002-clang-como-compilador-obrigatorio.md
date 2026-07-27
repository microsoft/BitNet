# ADR-002: Clang como compilador obrigatório

**Status:** Aceito (com exceção para Android/ARM64)  
**Data:** ~2024-03-01 (commit inicial)  
**Confiança:** 🟢 CONFIRMADO

---

## Contexto

Os kernels BitNet gerados (TL1/TL2) usam extensões SIMD avançadas (AVX2, NEON) e templates C++ complexos. O projeto precisa garantir compatibilidade de compilação.

## Decisão

Forçar Clang/Clang++ via CMake:
```python
run_command(["cmake", ..., "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"])
```

Commits relacionados:
- `c9e752c` — Fix build error with GCC by forcing Clang compiler in CMake on android/aarch64
- `141ddfd` — Fix compiler errors on GCC (adicionou `-fpermissive`)
- `9d37b86` — Add GCC to compiler check

## Alternativas consideradas

- **GCC por padrão:** Testado, mas produziu erros de compilação em extensões SIMD específicas dos kernels gerados. Suporte adicionado com `-fpermissive` como workaround.
- **MSVC no Windows:** Suportado via flag `-T ClangCL` no CMake para Windows (usa Clang-CL toolchain).

## Consequências

**Positivas:**
- Comportamento mais previsível com intrínsecas SIMD
- Melhor otimização de código gerado com `@torch.compile`-style patterns

**Negativas:**
- Clang é pré-requisito que pode não estar instalado por padrão (especialmente em ambientes CI sem imagem específica)
- Windows usa ClangCL especificamente, não Clang puro
