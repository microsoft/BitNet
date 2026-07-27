# ADR-004: CUDA Graphs para eliminação de overhead no loop de decode

**Status:** Aceito (com escape hatch)  
**Data:** 2025-05-15 (commit `154c92b` — Init gpu branch)  
**Confiança:** 🟢 CONFIRMADO

---

## Contexto

No loop de decode auto-regressivo, cada iteração executa um único passo forward no Transformer. Com batch=1 e tokens individuais, o overhead de lançamento de kernels CUDA (latência de scheduler, transferências de parâmetros) pode dominar o tempo de computação, especialmente em modelos menores.

## Decisão

Usar CUDA Graphs (`torch.cuda.CUDAGraph`) para capturar a sequência de kernels do prefill e do decode, permitindo replay zero-overhead:

```python
self._prefill_cuda_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(self._prefill_cuda_graph, **recording_kwargs):
    self._prefill_logits = self.prefill_model.forward_with_attn_bias(...)

def replay(tokens, seq_lens=None):
    self._prefill_inputs[0].copy_(tokens)  # apenas atualiza dados
    self._prefill_cuda_graph.replay()       # replay sem overhead de launch
    return self._prefill_logits
```

**Constraint imposto pela decisão:** Shapes dos tensors devem ser fixas. Isso força padding de prompts para `prompt_length` fixo.

**Escape hatch:** `NO_CUDA_GRAPHS` env var desabilita para debugging.

## Alternativas consideradas

- **`torch.compile` (Inductor):** Compilação JIT que otimiza grafos computacionais. Menos controle explícito sobre shapes, mas mais automático. Usado para `top_p` e `BitLinear.quant_input`.
- **Execução eager PyTorch:** Mais flexível (shapes variáveis), mas alto overhead por kernel launch no decode.

## Consequências

**Positivas:**
- Redução dramática de latência no decode (overhead de ~µs por kernel → zero)
- Throughput (tokens/s) significativamente maior

**Negativas:**
- Shapes fixas obrigam padding de prompts — usuários com prompts longos recebem comportamento silenciosamente incorreto se `prompt_length` for muito curto
- Debugging difícil (stacks de erro não informativas durante replay)
- Workaround necessário para watchdog CUDA em PyTorch ≥2.1 (`capture_error_mode="thread_local"`)
- Aquecimento (warm-up) necessário antes de capturar o grafo (extra latência de inicialização)
