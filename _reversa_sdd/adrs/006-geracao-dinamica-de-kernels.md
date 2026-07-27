# ADR-006: Geração dinâmica de código C++ de kernel por modelo/plataforma

**Status:** Aceito  
**Data:** ~2024-03-01 (commit inicial com `utils/codegen_tl1.py`)  
**Confiança:** 🟢 CONFIRMADO

---

## Contexto

Os kernels GEMM TL1/TL2 têm parâmetros de tiling (BM, BK, bm) que devem ser escolhidos empiricamente para maximizar utilização de cache por modelo. Esses parâmetros diferem entre modelos (bitnet-large, bitnet-3B, Llama3-8B) e arquiteturas (ARM64 vs x86).

## Decisão

Gerar código C++ especializado em tempo de setup via scripts Python (`utils/codegen_tl1.py`, `utils/codegen_tl2.py`). O código gerado é salvo em `include/bitnet-lut-kernels.h` e incluído na compilação seguinte.

```python
# Parâmetros de exemplo para bitnet_b1_58-3B em ARM64
run_command([sys.executable, "utils/codegen_tl1.py",
    "--model", "bitnet_b1_58-3B",
    "--BM", "160,320,320",
    "--BK", "64,128,64",
    "--bm", "32,64,32"])
```

**Preset mechanism:** Para modelos conhecidos, existe `preset_kernels/{model}/bitnet-lut-kernels-tl1.h` com parâmetros já validados, pulando a geração (`--use-pretuned`).

## Alternativas consideradas

- **Parâmetros configuráveis em runtime:** Elimina recompilação, mas impede otimizações de compilador via loop unrolling e inlining dos valores fixos.
- **Biblioteca única com todos os parâmetros:** Aumentaria tamanho do binário; o compilador não poderia especializar o código.
- **Auto-tuning em runtime (como TVM, MLIR):** Mais sofisticado mas muito mais complexo de implementar e manter.

## Consequências

**Positivas:**
- Compilador pode fazer unrolling e inlining total dos loops internos com valores conhecidos em tempo de compilação
- Cada modelo tem kernel literalmente especializado para suas dimensões
- Pode usar preset para pular recompilação em modelos conhecidos

**Negativas:**
- Recompilação necessária quando modelo muda
- `utils/tune_gemm_config.py` necessário para obter parâmetros ótimos para novos modelos
- Adicionar novo modelo requer: tunagem de parâmetros + adição ao codegen + adição ao setup_env.py
- `NotImplementedError` para modelos não suportados em vez de degradação graciosa
