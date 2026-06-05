# ADR-005: Três formatos de quantização CPU distintos por arquitetura

**Status:** Aceito  
**Data:** Inicial: ~2024-03-01 (I2_S); TL1/TL2 adicionados em `4c736e3` (fev 2025)  
**Confiança:** 🟢 CONFIRMADO

---

## Contexto

A inferência CPU eficiente de modelos BitNet requer explorar as capacidades específicas de cada arquitetura de processador. ARM64 tem instruções NEON diferentes do x86 AVX2, e os padrões de acesso de memória ótimos diferem.

## Decisão

Três formatos distintos com kernels especializados:

| Formato | Plataforma | Método | Performance relativa |
|---------|-----------|--------|---------------------|
| `I2_S` | arm64 + x86 | MAD com SIMD genérico | Baseline |
| `TL1` | arm64 only | LUT com NEON específico | > I2_S em ARM |
| `TL2` | x86_64 only | LUT com AVX2 específico | > I2_S em x86 |

Commits chave:
- `112f853` (nov 2025) — I2S kernels para weight+activation parallel em Intel e ARM
- `4c736e3` (fev 2025) — commit paper code com TL1/TL2 kernels

## Alternativas consideradas

- **Único formato universal (I2_S):** Mais simples, mas deixa performance na mesa por não explorar LUTs e instruções específicas.
- **Formato por modelo em vez de por arquitetura:** Considerado implicitamente — os BM/BK/bm variam por modelo dentro de cada formato.

## Consequências

**Positivas:**
- Performance máxima para cada arquitetura alvo
- I2_S como fallback robusto para ambas as arquiteturas

**Negativas:**
- Três pipelines de conversão distintos a manter
- Usuários devem escolher o formato correto para sua plataforma
- Código de geração de kernel duplicado (codegen_tl1.py vs codegen_tl2.py com lógica similar)
