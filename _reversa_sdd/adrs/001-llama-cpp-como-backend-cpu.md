# ADR-001: Usar llama.cpp como backend de inferência CPU

**Status:** Aceito  
**Data:** ~2024-03-01 (commit inicial `6cfd883`)  
**Confiança:** 🟢 CONFIRMADO

---

## Contexto

O BitNet precisa de um runtime de inferência para CPU que suporte modelos GGUF quantizados e seja suficientemente extensível para adicionar tipos de quantização customizados (I2_S, TL1, TL2).

## Decisão

Usar llama.cpp como runtime de inferência CPU, estendendo-o com kernels BitNet customizados via patches ao submodule `3rdparty/llama.cpp`.

## Alternativas consideradas

- **Implementação do zero:** Daria controle total, mas exigiria reimplementar sampling, context management, modelo architecture, etc.
- **PyTorch no CPU:** Possível, mas sem as otimizações de inferência de baixo nível do llama.cpp.
- **ONNX Runtime:** Mais difícil de estender com tipos de quantização customizados.

## Consequências

**Positivas:**
- Herda otimizações de inferência maduras do llama.cpp (scheduling, KV cache, batching)
- Suporte nativo a GGUF e múltiplas arquiteturas
- API CLI (`llama-cli`, `llama-server`) disponível imediatamente

**Negativas:**
- Acoplamento ao ciclo de release do llama.cpp (necessidade de atualizar submodule)
- Complexidade de manter fork/patch de código C++ de terceiros
- Evidenciado por múltiplos commits de "update submodule" no histórico
