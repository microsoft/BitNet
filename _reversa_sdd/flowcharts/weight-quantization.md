# Fluxograma — Quantização de Pesos

> Reversa Archaeologist | 2026-05-03

## Pipeline de conversão de checkpoint GPU

```mermaid
flowchart TD
    A([model_state.pt]) --> B[torch.load - weights_only=True]
    B --> C{Para cada tensor}

    C -->|*.wqkv.weight| D[Dividir em wq, wk, wv\npela dimensão da atenção]
    D --> D1[quant_weight_int8 para cada\ns = 1/mean abs\nround.clamp -1..1 → int8]
    D1 --> D2[Concatenar wq+wk+wv\nscales = wa,wb,wc,zero]
    D2 --> D3[convert_int8_to_int2\n→ int2_result]
    D2 --> D4[quant_weight_fp16 para cada\nround.clamp -1..1 / s → bf16]
    D4 --> D5[Concatenar → fp16_result]

    C -->|*.w13.weight| E[Dividir em w1, w3\npela dim ffn]
    E --> E1[quant_weight_int8 para cada]
    E1 --> E2[Concatenar w1+w3\nscales = w1,w3,zero,zero]
    E2 --> E3[convert_int8_to_int2\n→ int2_result]
    E2 --> E4[quant_weight_fp16 para cada\n→ fp16_result]

    C -->|*.w2 ou *.wo| F[quant_weight_int8\nscale = s,zero,zero,zero]
    F --> F1[convert_int8_to_int2\n→ int2_result]
    F --> F2[quant_weight_fp16\n→ fp16_result]

    C -->|demais\nembeddings, norms| G[Cópia direta\n→ ambos os resultados]

    D3 --> H([model_state_int2.pt])
    E3 --> H
    F1 --> H
    G --> H

    D5 --> I([model_state_fp16.pt])
    E4 --> I
    F2 --> I
    G --> I
```

## Empacotamento para GPU: `convert_weight_int8_to_int2`

```mermaid
flowchart LR
    A([weight int8\n{-1, 0, +1}]) --> B[+2 shift\n→ {1, 2, 3}]
    B --> C[permutate_weight_fastest\nReordena blocos 16×32\npara layout WMMA shared mem]
    C --> D[compress_int2_to_int8\n4 valores de 2 bits\npor byte via bitwise OR]
    D --> E[interleave_weight_int8\nReinterpreta como int32\nreordena bits internos\npara padrão WMMA]
    E --> F[reshape → N × K//4]
    F --> G([weight empacotado\nint8])
```

## Quantização I2_S para CPU: `quantize_i2_s`

```mermaid
flowchart TD
    A([float32 tensor]) --> B[Encontrar max absoluto\n→ i2_scale]
    B --> C{Para cada elemento}
    C -->|abs x < 1e-6| D[q8 = 1 zero]
    C -->|x × scale > 0| E[q8 = 2 positivo]
    C -->|x × scale < 0| F[q8 = 0 negativo]
    D --> G[Empacotar q8 → 2 bits]
    E --> G
    F --> G
    G --> H{arquitetura}
    H -->|x86 QK=128| I[Agrupamento de 32 por grupo\n4 grupos por byte\nshift: 6-2×group_idx]
    H -->|ARM QK=64| J[Agrupamento de 16 por grupo\n4 grupos por byte\nshift: 6-2×group_idx]
    I --> K[Armazenar scale float32\nao final dos dados]
    J --> K
    K --> L([I2_S empacotado])
```
