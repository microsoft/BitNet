# patches/llama.cpp/

Patches de dispatch do BitNet CPU-Universal sobre o submodule `3rdparty/llama.cpp`.

## Por que este diretório existe

O submodule `3rdparty/llama.cpp` aponta para o fork
[`Eddie-Wang1120/llama.cpp`](https://github.com/Eddie-Wang1120/llama.cpp.git)
na branch `merge-dev`. Em algum momento entre 2025-06-05 e 2026-06-05, a
branch foi reescrita (force-push), fazendo com que os commits que
adicionei com a integração do BitNet CPU-Universal ficassem **órfãos** —
eles existem no object DB local mas não são mais acessíveis por ref
alguma no remoto.

CI clones fresh não conseguem buscá-los, então os patches de
dispatch do L3 ACDC, L5 HRR cleanup e L4 TROPICAL K_I8 cache
ficaram **inacessíveis** em qualquer clone novo do fork.

## Solução

Esta pasta contém os três patches de dispatch exportados via
`git diff` a partir do working tree local. O script
`scripts/apply-dispatch-patches.sh` os aplica em ordem (L3 → L5 → L4
— L5 e L4 dependem do guard `#if` e do bloco tropical que L3
adiciona) após o `git submodule update --init`.

## Patches

| Arquivo | Linhas | O que faz |
|---------|--------|-----------|
| `01-L3-ACDC-FFN-dispatch.patch` | 162 | Adiciona `llm_build_ffn_acdc_bitnet` e o branch `BITNET_ACDC_FFN=1` no call site FFN BitNet-específico; estende o guard `#if` para incluir `BITNET_L3_ACDC`; adiciona include `ggml-bitnet-dispatch.h` |
| `02-L5-HRR-cleanup-dispatch.patch` | 16 | Adiciona branch `BITNET_HRR_ATTN_CLEANUP=N` no call site KQV BitNet-específico; estende o guard `#if` para incluir `BITNET_L5_HRR` |
| `03-L4-TROPICAL-KI8-cache.patch` | 12 | Adiciona include `ggml-bitnet-kv-cache.h` e a chamada `bitnet_kv_i8_cache_set_layer(il)` antes do `bitnet_op_tropical_attn` (Phase C: cache de K_i8 incremental para eliminar re-quantização de K a cada decode step) |

## Aplicação

Automática no CI (GitHub Actions), manual localmente:

```bash
# aplicar
./scripts/apply-dispatch-patches.sh

# só verificar
./scripts/apply-dispatch-patches.sh --check

# reverter (cleanup)
./scripts/apply-dispatch-patches.sh --reverse
```

O script é **idempotente**: detecta se os patches já estão aplicados
via sentinela (string característica que o patch adiciona) e sai
com sucesso sem reaplicar.

## Pontos de atenção

- Os patches foram gerados contra `merge-dev` em `1f86f05` (commit
  atual da branch no fork upstream). Se a branch for reescrita
  novamente, este diretório precisa ser regenerado.
- Os patches são **acumulativos**: L5 assume que L3 já foi aplicado;
  L4 assume que L3 já foi aplicado (precisa do bloco tropical e do
  guard `#if BITNET_L4_TROPICAL`). O script aplica nessa ordem
  automaticamente.
- Os patches NÃO tocam `include/ggml-bitnet-dispatch.h` nem
  `src/ggml-bitnet-dispatch.cpp` — esses arquivos vivem no repo
  principal (`include/`, `src/`).
