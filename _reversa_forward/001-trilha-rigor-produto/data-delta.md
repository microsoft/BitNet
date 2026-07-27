# Data Delta — `001-trilha-rigor-produto`

> Diff conceitual sobre o modelo extraído em `_reversa_sdd/`. Para esta feature, **o modelo de dados é majoritariamente estável**; mudanças só aparecem em vertentes futuras (v0.2+).
>
> **Versão:** v1 (gerado por reversa-plan em 2026-06-06)
> **Ancoragem:** `requirements.md` v2 + `roadmap.md` v1 + `_reversa_sdd/data-dictionary.md`

---

## 1. Estado Atual do Modelo de Dados

### 1.1. Modelo primário: GGUF

O BitNet CPU-Universal consome modelos no formato **GGUF** (GPT-Generated Unified Format), produzido por:
- HF safetensors → `convert-hf-to-gguf-bitnet.py` → GGUF F32
- GGUF F32 → `llama-quantize` → GGUF I2_S (x86) / TL1 (ARM) / TL2 (x86 LUT)

**Estrutura interna do GGUF (campos relevantes para esta feature):**

| Campo | Tipo | Descrição | Afetado por esta feature? |
|-------|------|-----------|---------------------------|
| `general.architecture` | string | "bitnet" | ❌ |
| `general.name` | string | "BitNet-b1.58-2B-4T" | ❌ |
| `bitnet.quantization.type` | string | "i2_s" / "tl1" / "tl2" | ❌ |
| `tensor.token_embd.weight` | tensor F32/I2S | Embedding (2560 × 128000) | ❌ |
| `tensor.<layer>.attn_q.weight` | tensor I2S | Q projection | ❌ |
| `tensor.<layer>.attn_k.weight` | tensor I2S | K projection (com GQA, n_head_kv < n_head) | ❌ |
| `tensor.<layer>.attn_v.weight` | tensor I2S | V projection | ❌ |
| `tensor.<layer>.attn_output.weight` | tensor I2S | O projection | ❌ |
| `tensor.<layer>.ffn_gate.weight` | tensor I2S | FFN gate (2560 × 6912) | ❌ (v0.1) / 🟡 (v0.2 se D2 dispara) |
| `tensor.<layer>.ffn_up.weight` | tensor I2S | FFN up (2560 × 6912) | ❌ (v0.1) / 🟡 (v0.2 se D2 dispara) |
| `tensor.<layer>.ffn_down.weight` | tensor I2S | FFN down (6912 × 2560) | ❌ (v0.1) / 🟡 (v0.2 se D2 dispara) |
| `tensor.<layer>.ffn_norm.weight` | tensor F32 | LayerNorm | ❌ |
| `tensor.output.weight` | tensor F32 | LM head | ❌ |

**Conclusão:** para v0.1 desta feature, **nenhum campo GGUF é alterado, removido ou adicionado**. O modelo BitNet-2B existente continua sendo lido sem modificação.

### 1.2. Modelo secundário: sidecars Python

Para a extração de ACDC diagonais (commit `fcf1d4d`, Phase A):

| Arquivo | Formato | Conteúdo | Afetado? |
|---------|---------|----------|----------|
| `<model>_acdc_diagonals.npz` | NumPy savez | `{layer_name: d_array}` | ❌ gerado por `utils/extract_acdc_diagonal.py`; não usado em inferência |
| `<model>_acdc_diagonals.json` | JSON | Metadados: shapes, energia por matriz, hash de validação | ❌ sidecar do npz; para auditoria |

Esses sidecars existem mas **não são lidos pelo pipeline de inferência** (P6: estrutura, não compressão). Servem apenas para validar a tese ACDC em análise offline.

### 1.3. Modelo terciário: artefatos de build

| Arquivo | Formato | Conteúdo | Afetado? |
|---------|---------|----------|----------|
| `include/bitnet-lut-kernels.h` | C header | Kernels gerados para o modelo (TL1/TL2) | ❌ |
| `build/bin/llama-cli` | ELF binary | Executável | ❌ |
| `build/bin/llama-server` | ELF binary | Servidor HTTP OpenAI-compat | ❌ |
| `build_tests/test_*` | ELF binary | Testes C++ | 🟡 serão adicionados novos (M1, M3) |

---

## 2. Mudanças para v0.1 (D-T-01 a D-T-04, M1-M2)

### 2.1. Mudanças NO modelo de dados

**Nenhuma.**

A feature v0.1 adiciona:
- Testes C++ novos (em `tests/test_*_properties.cpp`)
- Documentos novos (em `docs/decision-matrix.md`, `docs/invariants.md`, `ROADMAP.md`, `examples/*.md`)
- Script de bench novo (em `utils/bench_publish.py`)

Nenhum desses **muda o formato de leitura do modelo GGUF** nem introduz campos novos no GGUF.

### 2.2. Mudanças NO pipeline de inferência (binário, não dados)

| Mudança | Tipo | Comportamento |
|---------|------|---------------|
| L4 sparse opt-in (D-T-01) | Comportamental | Default attention denso preservado. `BITNET_SPARSE_TOPK=32` ou `--attn sparse` ativa. |
| Documentação persona D4 | Cosmético | README e exemplos. Não afeta binário. |
| Test air-gapped (AC-11) | Test infra | `tests/test_air_gapped_boot.sh` é um script que roda o binário em netns; não muda o binário. |

### 2.3. Compatibilidade retroativa

✅ **Garantida.** Um modelo GGUF gerado antes desta feature continua funcionando idêntico. A feature é puramente aditiva.

---

## 3. Mudanças para v0.2 (CONDICIONAL — D2 trigger)

**Se a investigação D2** (M1, sub-tarefa "testar Llama-2-7B") **disparar** o trigger "ACDC retangular vira bloqueador", então:

### 3.1. Extensão do GGUF (ou sidecar dedicado)

**Opção A: sidecar .npz (preferida, retrocompatível)**

Adiciona-se ao lado de `ggml-model-i2_s.gguf` um arquivo `ggml-model-i2_s.acdc.npz`:

```
ggml-model-i2_s.gguf        # original, inalterado
ggml-model-i2_s.acdc.npz    # novo, contém d* por FFN matrix
  ├── 'blk.0.ffn_gate'      # d ∈ ℝ^2560
  ├── 'blk.0.ffn_up'        # d ∈ ℝ^2560
  ├── 'blk.0.ffn_down'      # d ∈ ℝ^2560
  ├── 'blk.1.ffn_gate'
  ├── ...
  └── 'blk.29.ffn_down'
```

**Vantagem:** Retrocompatível. Modelos sem sidecar usam FFN denso (atual). Modelos com sidecar usam ACDC retangular.

**Desvantagem:** Dois arquivos para distribuir.

**Opção B: extensão GGUF com nova seção (não retrocompatível)**

Adicionar ao GGUF uma seção `acdc.diagonals` (formato customizado). Mais limpo, mas exige regenerar GGUF e não carrega em versões antigas do loader.

### 3.2. Schema do sidecar (opção A, recomendada)

```python
# Formato: NumPy savez
{
    'blk.0.ffn_gate': np.ndarray(shape=(2560,), dtype=np.float32),
    'blk.0.ffn_up':   np.ndarray(shape=(2560,), dtype=np.float32),
    'blk.0.ffn_down': np.ndarray(shape=(6912,), dtype=np.float32),  # min(m,n) = 2560 na verdade
    # ... 30 camadas × 3 matrizes = 90 diagonais
}
```

**Shape da diagonal `d`:** `min(m, n)` (a menor dimensão). Para BitNet-2B:
- gate/up (2560 × 6912): d ∈ ℝ^2560
- down (6912 × 2560): d ∈ ℝ^2560

### 3.3. Pipeline de geração do sidecar

```bash
# Gera o sidecar a partir de um GGUF existente (Hadamard projection, energy validation)
python utils/extract_acdc_diagonal.py \
    --input models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    --output models/BitNet-b1.58-2B-4T/ggml-model-i2_s.acdc.npz \
    --json-sidecar models/BitNet-b1.58-2B-4T/ggml-model-i2_s.acdc.json
```

(Pipeline parcial já existe: `utils/extract_acdc_diagonal.py` commit `fcf1d4d`; só precisa estender para retangular.)

### 3.4. Migração

**Não há migração de dados de usuário** porque a feature é local-first (persona D4). O usuário baixa o GGUF e o sidecar, coloca ambos em `models/`, e o loader detecta o sidecar e ativa ACDC.

**Não há migração de modelo**: o GGUF original é preservado. O sidecar é adicional.

### 3.5. Compatibilidade

- Loader sem suporte a ACDC: ignora o sidecar, usa FFN denso (atual).
- Loader com suporte a ACDC: detecta o sidecar, valida shapes, ativa FFN ACDC.
- Modelo sem sidecar em loader com suporte: usa FFN denso (fallback gracioso).

---

## 4. Mudanças para v0.3 (RESERVA — D3 reavaliação Q4 2029)

**Não implementada em v0.1.** Apenas documentada como reserva.

Se a reavaliação Q4 2029 decidir retomar o scaffolding `utils/finetune_acdc.py`:

### 4.1. Formato de checkpoint de fine-tuning

```
<model>_acdc_finetuned/
  ├── config.json              # hiperparâmetros: lr, n_epochs, layer_subset
  ├── acdc_diagonals/          # 90 .npy files, uma por GEMV FFN
  │   ├── blk.0.ffn_gate.d.npy
  │   ├── blk.0.ffn_up.d.npy
  │   └── ...
  ├── training_log.jsonl       # uma linha por epoch: {loss, val_loss, lr}
  └── smoke_test_report.json   # perplexity antes/depois, tempo de execução
```

### 4.2. Conversão checkpoint → GGUF

```bash
# Pseudo-código (NÃO IMPLEMENTADO em v0.1)
python utils/finetune_acdc.py \
    --base-model models/BitNet-b1.58-2B-4T/ggml-model-f32.gguf \
    --output models/BitNet-b1.58-2B-4T-acdc/ \
    --epochs 1 --lr 1e-4 --layers 0..29

python utils/convert_acdc_finetuned_to_gguf.py \
    --input models/BitNet-b1.58-2B-4T-acdc/acdc_diagonals/ \
    --output models/BitNet-b1.58-2B-4T-acdc/ggml-model-acdc.gguf
```

### 4.3. Implicação para o formato GGUF

Introduz-se uma nova variante de quantização: `i2_s_acdc` (ou nome similar). O GGUF passa a ter campos:
- `bitnet.quantization.type` = "i2_s_acdc" (em vez de "i2_s")
- `tensor.<layer>.ffn_*.acdc_diagonal` = tensor F32 (a diagonal d* treinada)

**Não retrocompatível**: o loader precisa saber interpretar a nova variante. Documentar em `docs/gguf-extension.md` (a criar).

---

## 5. Resumo de Compatibilidade

| Versão | Compatível com versões anteriores do loader? | Compatível com modelos antigos? | Notas |
|--------|----------------------------------------------|--------------------------------|-------|
| v0.1 | ✅ | ✅ | Aditivo; sem mudança de modelo |
| v0.2 (condicional) | ✅ (modelo sem sidecar = fallback denso) | ✅ (modelo antigo = FFN denso) | Sidecar é opcional |
| v0.3 (reserva) | ❌ (nova variante `i2_s_acdc`) | ❌ (precisa de GGUF `i2_s_acdc`) | Requer loader atualizado; é "modelo novo", não "modelo antigo" |

---

## 6. Resumo Executivo

**Para v0.1 (esta iteração):**

- ✅ **Zero mudança no modelo de dados** (GGUF inalterado)
- ✅ **Zero migração de dados**
- ✅ **100% retrocompatível**
- ✅ **Sem novos formatos**

**Para v0.2 (condicional ao trigger D2):**

- 🟡 **Sidecar .npz** (retrocompatível, opcional)
- 🟡 **Pipeline de extração estendido** (estende `utils/extract_acdc_diagonal.py`)
- 🟡 **Sem migração de dados** (sidecar é gerado a partir de GGUF existente)

**Para v0.3 (reserva, Q4 2029):**

- 🔴 **Nova variante GGUF** (`i2_s_acdc`)
- 🔴 **Pipeline de fine-tuning** (PyTorch, requer GPU)
- 🔴 **Não retrocompatível** (mas é "modelo novo", não "atualização")

---

*data-delta.md v1 — gerado por reversa-plan em 2026-06-06*
