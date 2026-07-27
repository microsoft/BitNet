# ADR-003: Dual-model GPU: modelo separado para prefill (fp16) e decode (int2)

**Status:** Aceito  
**Data:** 2025-05-15 (commit `154c92b` — Init gpu branch)  
**Confiança:** 🟡 INFERIDO

---

## Contexto

Na inferência de LLMs há duas fases com características distintas:
- **Prefill:** Processa todos os tokens do prompt de uma vez. Alto paralelismo, executa uma única vez por request.
- **Decode:** Gera tokens um a um. Baixo paralelismo (batch=1 tipicamente), executa centenas/milhares de vezes.

O pipeline GPU do BitNet precisa otimizar ambas as fases.

## Decisão

Manter dois modelos Transformer em memória GPU simultaneamente:
- `prefill_model`: usa `BitLinear` (pesos ternários em BF16, sem kernel CUDA customizado)
- `decode_model`: usa `BitLinearKernel` (pesos INT2 comprimidos + kernel CUDA int8×int2)

```python
model_args_prefill = fast.ModelArgs(use_kernel=False)
model_args_decode = fast.ModelArgs(use_kernel=True)
```

## Alternativas consideradas

- **Único modelo INT2 para tudo:** Mais simples, mas o kernel CUDA int8×int2 pode ter menor acurácia numérica no prefill onde os cálculos em batch grande são mais sensíveis.
- **Único modelo FP16 para tudo:** Máxima acurácia, mas muito mais lento no decode (sem benefício da quantização int2).
- **Trocar o modelo dinamicamente:** Evitaria uso duplo de memória, mas adicionaria latência de troca e complexidade.

## Consequências

**Positivas:**
- Máxima acurácia no prefill (relevante para compreensão do prompt)
- Máxima velocidade no decode (relevante para latência de geração)
- CUDA Graphs são estáveis pois cada modelo tem shapes fixas

**Negativas:**
- ~2× mais uso de memória GPU vs. single-model
- Dois arquivos de checkpoint separados a manter (`model_state_fp16.pt` e `model_state_int2.pt`)
- Complexidade no pipeline de conversão de checkpoint
