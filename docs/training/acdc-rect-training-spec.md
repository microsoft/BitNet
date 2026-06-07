# Spec de Treinamento — ACDCLite (ACDC Rect, Direção A)

> **Status:** Spec aprovada, implementação pendente (Q4 2029 gate per ROADMAP.md §2)
> **Propósito:** Fechar o P6 gap — kernels L3 ACDC rect só produzem output correto
> em modelos treinados com a arquitetura ACDC. Esta spec define o que treinar,
> como treinar, e como verificar que o modelo está integrado aos kernels C.
> **Constraint hard:** CPU-only inference. Treinamento pode usar GPU; inferência nunca.

---

## 1. O Problema (P6 Gap)

Os kernels L3 (ACDC rect) e L5 (HRR) implementados nos níveis 3 e 5 funcionam
corretamente como operações matemáticas, mas produzem output sem sentido quando
aplicados ao BitNet-2B:

```
L3 ACDC sobre BitNet-2B:    speedup +0.6%, output diverge da baseline
L5 HRR sobre BitNet-2B:     speedup -69 %, output garbage
```

A razão é matemática, não um bug de implementação. Para W aleatório
(distribuição BitNet ternária), a aproximação ACDC captura apenas ~1/n da
energia de W:

```
E_ACDC = ||H·diag(d*)·H||² / ||W||²  ≈  1/n   ≈ 0.02% para n=4096
```

**A única solução é treinar com ACDC como arquitetura**, não como aproximação
post-hoc. O diagonal `d` é o único parâmetro da camada — aprendido por
backprop, não extraído de W pré-treinado.

---

## 2. Condição de Speedup (Por que n_ff/n_embd ≥ 7)

O speedup do ACDC rect em relação ao GEMV denso depende do ratio:

```
r = n_ff / n_embd

ACDC rect:  2 × P × log₂(P) adições  (P = next_pow2(max(n_embd, n_ff)))
Dense GEMV: n_embd × n_ff adições

Speedup = (n_embd × n_ff) / (2 × P × log₂(P))
```

Para n_embd=1024 e n_ff variando:

| n_ff  | r    | P     | ACDC ops | Dense ops | Speedup |
|-------|------|-------|----------|-----------|---------|
| 1024  | 1.0× | 1024  | 20480    | 1.05M     | 51×     |
| 2048  | 2.0× | 2048  | 45056    | 2.10M     | 47×     |
| 4096  | 4.0× | 4096  | 98304    | 4.19M     | 43×     |
| 7168  | 7.0× | 8192  | 229376   | 7.34M     | **32×** |
| 10240 | 10×  | 16384 | 458752   | 10.49M    | 23×     |

O speedup diminui conforme r aumenta (P "pula" para a próxima potência de 2,
mas n_ff × n_embd cresce linearmente). O ponto ótimo de custo-benefício é
**r ≈ 7 (n_ff ≈ 7 × n_embd)**: large FFN (alta capacidade) com speedup >30×
vs GEMV denso para o mesmo tamanho de modelo.

Valores de r < 5 dão speedup >40× mas modelos com FFN estreito têm menor
capacidade (regressão de qualidade no pretraining). Valores de r > 10 têm
speedup <25× e P dobra de tamanho (overhead de padding).

**Constraint hard desta spec:** r ≥ 7.

---

## 3. Arquitetura do Modelo — ACDCLite-1B

### 3.1 Dimensões

| Parâmetro          | Valor  | Justificativa                                |
|--------------------|--------|----------------------------------------------|
| `n_embd`           | 1024   | Balanceia expressividade vs ops              |
| `n_heads`          | 16     | head_dim = 64 (SIMD-friendly para AVX2)     |
| `n_kv_heads`       | 4      | GQA 4:1 (reduz KV cache em 4×)              |
| `n_ff`             | 7168   | ≈ 7 × n_embd = 7.0 (dentro do gatilho)      |
| `P_acdc`           | 8192   | `next_pow2(7168)` = 8192 (padding overhead minimal) |
| `n_ff / P_acdc`    | 7/8    | Razão de utilização de P                    |
| `n_layers`         | 24     | Profundidade típica de modelos ~1B           |
| `vocab_size`       | 32000  | Llama-2 SentencePiece BPE                   |
| `context_len`      | 4096   | Suficiente para CPU decode                  |
| `rope_base`        | 10000  | RoPE padrão Llama                           |

### 3.2 Contagem de Parâmetros

| Componente            | Params (M)  | Formato       | Inferência   |
|-----------------------|-------------|---------------|--------------|
| Token embedding       | 32.8M       | fp32/bf16     | lookup       |
| Attention Q (×24)     | 25.2M       | 1.58b ternary | I2_S GEMV L1 |
| Attention K (×24)     | 6.3M        | 1.58b ternary | I2_S GEMV L1 |
| Attention V (×24)     | 6.3M        | 1.58b ternary | I2_S GEMV L1 |
| Attention O (×24)     | 25.2M       | 1.58b ternary | I2_S GEMV L1 |
| **FFN gate diagonal** | **0.20M**   | fp32          | ACDC rect L3 |
| **FFN up diagonal**   | **0.20M**   | fp32          | ACDC rect L3 |
| **FFN down diagonal** | **0.20M**   | fp32          | ACDC rect L3 |
| LayerNorm (×48)       | 0.10M       | fp32          | scalar       |
| LM head (shared emb)  | —           | tied          | lookup       |
| **Total**             | **~96M**    |               |              |

O modelo equivalente denso (mesmas dimensões, FFN não-ACDC) teria:
`96M + 24 × 2 × 1024 × 7168 ≈ 448M params` — o ACDC rect economiza 352M
parâmetros de FFN sem perda de capacidade expressiva (quando treinado corretamente).

> **Nota sobre "300M":** o target original "~300M" referia-se à capacidade
> equivalente (comparable a modelos densos de 300-450M), não ao count real.
> ACDCLite-1B tem 96M params reais mas ACDC FFN da largura de um 448M modelo.

### 3.3 Estrutura da Camada FFN ACDC Rect

Cada camada FFN usa **dois blocos ACDC rect** (gate × up projection como SiLU-gated):

```python
# Pseudocódigo da camada FFN ACDC rect (equivalente Llama SwiGLU)
def ffn_acdc_rect(x: Tensor[n_embd], 
                  d_gate: Tensor[P_acdc],
                  d_up: Tensor[P_acdc], 
                  d_down: Tensor[P_acdc]) -> Tensor[n_embd]:
    
    # x ∈ ℝ^{n_embd}, P = 8192 = next_pow2(7168)
    x_pad = pad(x, P_acdc)              # zero-pad para potência de 2
    
    # Gate projection: ACDC rect  n_embd → n_ff
    gate = fwht(x_pad)                  # H · x_pad  (zero muls)
    gate = gate * d_gate                # diagonal scaling (n_embd muls)
    gate = fwht(gate)[:n_ff]            # H · gate, truncate para n_ff
    gate = silu(gate)                   # ativação
    
    # Up projection: ACDC rect  n_embd → n_ff
    up = fwht(x_pad)                    # reutilizar (cache)
    up = up * d_up
    up = fwht(up)[:n_ff]
    
    # Element-wise product (SiLU-gated)
    hidden = gate * up                  # ∈ ℝ^{n_ff}
    
    # Down projection: ACDC rect  n_ff → n_embd
    h_pad = pad(hidden, P_acdc)
    h_pad = fwht(h_pad)
    h_pad = h_pad * d_down
    out = fwht(h_pad)[:n_embd]          # truncate de P para n_embd
    
    return out
```

**Grad das diagonais** (diferenciável, sem truque):
```
∂L/∂d_gate[k] = (H · x_pad)[k] · (∂L/∂gate_scaled)[k]  (chain rule simples)
```

### 3.4 Atenção (Mantida Padrão BitNet Ternário)

A atenção não é modificada — usa I2_S GEMV L1 (ternary + avx2 via llama.cpp).
Os pesos Q/K/V/O são quantizados em 1.58b na carga do checkpoint. RoPE padrão.

Esta escolha isola o P6 gap: apenas FFN usa ACDC; atenção permanece em L1.
Isso permite comparar qualidade diretamente com BitNet-2B no mesmo plano.

---

## 4. Treinamento

### 4.1 Dataset

| Dataset          | Tokens | Proporção | Justificativa                           |
|------------------|--------|-----------|------------------------------------------|
| FineWeb-Edu      | 200B   | 40%       | Alta qualidade web, educacional          |
| The Stack v2     | 80B    | 16%       | Código (melhora raciocínio estrutural)   |
| Wikipedia EN+PT  | 20B    | 4%        | Factual, diverso                         |
| OpenWebText2     | 40B    | 8%        | Cobertura web geral                      |
| Books3           | 60B    | 12%       | Longa dependência contextual             |
| C4               | 100B   | 20%       | Complemento web                          |
| **Total**        | **500B** | 100%    | Chinchilla-optimal para 96M params       |

Chinchilla scaling: ~500B tokens é near-optimal para 96M params (C_opt ≈ 20 × N).

### 4.2 Tokenizador

Llama-2 SentencePiece BPE, vocab=32000. Já usado no BitNet-2B — permite
comparação direta de perplexidade em benchmarks padrão.

### 4.3 Configuração de Treinamento

```yaml
# training_config.yaml
model:
  architecture: acdc_lite
  n_embd: 1024
  n_heads: 16
  n_kv_heads: 4
  n_ff: 7168
  n_layers: 24
  vocab_size: 32000
  context_len: 4096
  rope_base: 10000

optimizer:
  type: adamw
  lr: 3e-4
  lr_schedule: cosine_with_warmup
  warmup_steps: 2000
  min_lr: 3e-5
  weight_decay: 0.1
  grad_clip: 1.0
  beta1: 0.9
  beta2: 0.95

quantization:
  attention_weights: 1.58bit  # BitNet ternary, per-row absmax
  ffn_diagonals: fp32          # diagonais ACDC em float32 (96M total)
  activations: bf16            # computação em bf16

batch:
  global_batch_tokens: 4194304   # 4M tokens/step (estável para 96M params)
  micro_batch_size: 2            # por GPU
  gradient_accumulation: varies  # dependendo do hardware

training:
  total_tokens: 500_000_000_000  # 500B
  eval_interval: 1000            # steps
  save_interval: 5000            # steps
  eval_datasets: [wikitext103, lambada]

hardware:
  # Treinamento: GPU (qualquer; especificação mínima abaixo)
  min_gpu_memory: 24GB           # para micro_batch=2
  recommended: 8× A100 80GB     # ~72h de treinamento
  # Inferência: CPU ONLY (hard constraint)
```

### 4.4 Inicialização dos Diagonais ACDC

Os diagonais `d_gate`, `d_up`, `d_down` são inicializados para preservar
a variância de ativação de entrada (evitar colapso na primeira iteração):

```python
# Inicialização dos diagonais (equivalente a identidade com ruído)
std_init = (1.0 / math.sqrt(P_acdc)) * 0.1
d_gate = torch.ones(P_acdc) + torch.randn(P_acdc) * std_init
d_up   = torch.ones(P_acdc) + torch.randn(P_acdc) * std_init
d_down = torch.ones(P_acdc) * (1.0 / P_acdc) + torch.randn(P_acdc) * std_init
```

A inicialização de `d_down` com `1/P_acdc` compensa o fator de escala da
FWHT não-normalizada (o IRFFT da biblioteca é normalizado, mas o FWHT de
treinamento em PyTorch precisa da normalização manual).

### 4.5 Implementação do Backward (PyTorch)

O FWHT não tem implementação nativa no PyTorch — usar `torch.fft.fft` como
proxy (identical butterfly structure, complex version):

```python
import torch
import torch.nn.functional as F

def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform via FFT (differentiable)."""
    n = x.shape[-1]
    assert (n & (n-1)) == 0, "n deve ser potência de 2"
    # Alternativa: scipy.linalg.hadamard para n pequeno,
    # ou implementação butterfly manual para autograd
    result = x.clone()
    h = 1
    while h < n:
        result = result.view(*result.shape[:-1], n // (2*h), 2*h)
        a, b = result[..., :h], result[..., h:]
        result = torch.cat([a + b, a - b], dim=-1)
        result = result.view(*result.shape[:-2], n)
        h *= 2
    return result

class ACDCRectLayer(torch.nn.Module):
    def __init__(self, n_embd: int, n_ff: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_ff   = n_ff
        self.P      = 1 << (n_ff - 1).bit_length()  # next_pow2(n_ff)
        
        self.d_gate = torch.nn.Parameter(torch.ones(self.P))
        self.d_up   = torch.nn.Parameter(torch.ones(self.P))
        self.d_down = torch.nn.Parameter(torch.ones(self.P) / self.P)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Pad input to P
        x_pad = F.pad(x, (0, self.P - D))              # [B, T, P]
        x_h   = hadamard_transform(x_pad)               # H·x_pad
        
        # Gate + Up (reuse x_h)
        gate = hadamard_transform(x_h * self.d_gate)[..., :self.n_ff]
        up   = hadamard_transform(x_h * self.d_up  )[..., :self.n_ff]
        hidden = F.silu(gate) * up                      # [B, T, n_ff]
        
        # Down projection
        h_pad = F.pad(hidden, (0, self.P - self.n_ff))
        out   = hadamard_transform(
                    hadamard_transform(h_pad) * self.d_down
                )[..., :D]
        return out
```

---

## 5. Verificação P6 (Como Saber que o Gap Está Fechado)

O P6 gap está fechado quando o modelo ACDCLite-1B treinado:
1. Produz output finito e não-divergente com os kernels C L3 ACDC
2. A perplexidade no checkpoint convertido ≤ perplexidade de referência + 2 pontos

### 5.1 Pipeline de Conversão (Checkpoint → GGUF ACDC)

```bash
# 1. Treinar e salvar checkpoint PyTorch
# (outputs: acdc_lite_24L.pt + tokenizer)

# 2. Exportar diagonais ACDC para .npz
python utils/export_acdc_diagonals.py \
    --checkpoint acdc_lite_24L.pt \
    --out models/acdc_lite/acdc_diagonals.npz

# 3. Exportar atenção BitNet para GGUF (weights ternários)
python utils/convert_acdc_to_gguf.py \
    --checkpoint acdc_lite_24L.pt \
    --acdc-diags models/acdc_lite/acdc_diagonals.npz \
    --out models/acdc_lite/ggml-model-i2_s.gguf

# 4. Rodar inferência com kernels L3
python run_inference.py \
    -m models/acdc_lite/ggml-model-i2_s.gguf \
    -p "The capital of France is" -n 50 -t 4 \
    --attn dense --ffn acdc_rect
```

### 5.2 Critério de Aceitação P6

| Teste | Critério | Método |
|-------|----------|--------|
| P6-A: Output finito | max(|output|) < 100 | Verificar no primeiro forward pass |
| P6-B: Perplexidade | PPL(wikitext103) ≤ ref_dense + 2.0 | `python utils/test_perplexity.py` |
| P6-C: Throughput L3 > L1 | tokens/s com L3 ≥ tokens/s com L1 | `python utils/e2e_benchmark.py` |
| P6-D: ACDC energy > 0.5 | energia capturada por d* ≥ 50% de W | `utils/extract_acdc_diagonal.py` |

P6-D é a checagem matemática central: para um modelo treinado com ACDC,
`acdc_project(W)` deve capturar ≥ 50% da energia (vs ~0.04% no BitNet-2B).
Isso confirma que o modelo efetivamente aprendeu na base de Hadamard.

### 5.3 Script de Verificação

```python
# utils/verify_p6.py — roda após converter o checkpoint
import numpy as np
from utils.extract_acdc_diagonal import extract_diagonal

def verify_p6(model_dir: str, threshold: float = 0.5):
    """Verifica que o modelo tem energia ACDC ≥ threshold."""
    diags = np.load(f"{model_dir}/acdc_diagonals.npz")
    energies = []
    for key in diags.keys():
        if key.startswith('_'):
            continue
        d = diags[key]           # diagonal extraída
        energy = np.sum(d**2)    # energia da projeção ACDC
        energies.append(energy)
    
    mean_energy = np.mean(energies)
    ok = mean_energy >= threshold
    print(f"[P6] ACDC energy: {mean_energy:.4f} (threshold: {threshold})")
    print(f"[P6] {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok
```

---

## 6. Sequência de Implementação

### Fase 0 — Pré-requisitos (já prontos)

- [x] Kernel C `acdc_forward_rect_f32` (`src/ggml-bitnet-fwht.cpp`)
- [x] Kernel C `acdc_forward_rect_i8` (int8 input variant)
- [x] Kernel C `acdc_project_rect` (diagnóstico de energia)
- [x] Teste `test_acdc_rect.cpp` (valida kernels rect)
- [x] Script `utils/extract_acdc_diagonal.py` (extração de d*)

### Fase 1 — Modelo PyTorch (2-4 semanas)

- [ ] `models/acdc_lite/modeling_acdc.py` — `ACDCRectLayer` + modelo completo
- [ ] `models/acdc_lite/config.py` — `ACDCLiteConfig` (24L, 1024d, 7168ff)
- [ ] `models/acdc_lite/train.py` — loop de treinamento com DataLoader
- [ ] `models/acdc_lite/dataset.py` — streaming de FineWeb-Edu + C4
- [ ] Smoke test: treinar 1B tokens, verificar PPL decresce monotonamente

### Fase 2 — Conversão e Integração (1-2 semanas)

- [ ] `utils/export_acdc_diagonals.py` — exporta d* do checkpoint PyTorch
- [ ] `utils/convert_acdc_to_gguf.py` — gera GGUF com atenção L1 + FFN ACDC
- [ ] Patch mínimo em `src/ggml-bitnet-dispatch.cpp` para rotear FFN → L3
- [ ] Teste de roundtrip: PyTorch output == kernel C output (max_diff < 1e-3)

### Fase 3 — P6 Validation (1 semana)

- [ ] `utils/verify_p6.py` — script de verificação automática
- [ ] Executar 4 critérios P6-A/B/C/D
- [ ] Atualizar `docs/findings-cpu-universal.md` com resultados reais
- [ ] Atualizar ROADMAP.md: mover D-01` de "reserva" para "concluído"

---

## 7. Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|---------------|-----------|
| Instabilidade no treinamento (gradients divergem na FWHT) | Média | Gradient clipping agressivo (0.5), LR warmup longo (4000 steps), init conservador de d_down |
| Qualidade inferior ao modelo denso equivalente (PPL muito alto) | Alta | Usar K=2 blocos ACDC por camada em vez de 1 (dobra capacity) |
| n_ff não-multiplo de P (padding waste) | Baixa | n_ff=7168 → P=8192, utilização=87.5% (aceitável) |
| Tempo de treinamento proibitivo sem GPU | Certeza | GPU obrigatória para Fase 1/2; CPU só para inferência |
| Tokenizador incompatível | Baixa | Llama-2 BPE usado no BitNet-2B — compatível diretamente |

---

## 8. Referências e Baseamento no Codebase

| Conceito | Arquivo de referência | Linha/Seção |
|----------|----------------------|-------------|
| Kernel rect forward | `include/ggml-bitnet-fwht.h` | `acdc_forward_rect_f32` |
| ACDC invariant crítico | `CLAUDE.md` | "Critical ACDC invariant" |
| P6 gap | `docs/findings-cpu-universal.md` | §1.3 (L3) |
| Speedup rect | `docs/findings-cpu-universal.md` | §1.3 (benchmarks Falcon3) |
| Extração d* | `utils/extract_acdc_diagonal.py` | Completo |
| acdc_project_rect | `include/ggml-bitnet-fwht.h` | `acdc_project_rect` |
| Test rect | `test_acdc_rect.cpp` | Completo |

---

*Última atualização: 2026-06-07 — Direção A spec completa.*
*Implementação: aguarda disponibilidade de GPU ou decisão de parceria de compute.*
