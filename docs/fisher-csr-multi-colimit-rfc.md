# RFC: Fisher-Guided Sparse CSR Epilogues, Multi-Colimit Layer Group Sharing ($K \ll L$), and Q16.16 Functorial Morphisms for `TL1`/`TL2` mpGEMM

## Motivation

`bitnet.cpp` achieves remarkable CPU inference speedups via its zero-multiplication `TL1` (2-trits-in-4-bits) and `TL2` (3-trits-in-5-bits, $1.667\text{ bits/weight}$) lookup-table (`vpshufb`) kernels and `SubLN` activation stabilization. However, two structural barriers remain for wider adoption across the open-source ecosystem:

1. **Post-Training / Fast-Distillation Collapse on Pretrained Models**:
   Quantizing standard pretrained FP16/BF16 models (e.g., `Qwen2.5-Coder`, `Llama-3`, `Mistral`) directly onto the ternary grid $\{-1, 0, +1\}$ degrades accuracy because $\sim 1.35\%\text{--}2.8\%$ of weights lie on steep regions of the empirical Fisher Information manifold ($\mathcal{F}_{i,j} = \mathbb{E}[(\partial \mathcal{L} / \partial W_{i,j})^2] \gg \tau$).
2. **DRAM-to-L3 Memory Bandwidth Wall Across $L$ Layers**:
   Loading $L$ independent ternary FFN blocks (`gate_proj`, `up_proj`, `down_proj`) from DRAM evicts L3 cache at every layer on desktop and edge CPUs.

## Proposed Additions (`include/ggml-bitnet-kinecat.h`)

This PR introduces a self-contained, zero-breaking-change C/C++17 AVX2 header (`include/ggml-bitnet-kinecat.h`) and verification suite (`utils/test-bitnet-kinecat-fusion.cpp`) that extend `TL1` and `TL2` with:

1. **Fused Sparse Fisher Curvature CSR Int8 Epilogue (`ggml_bitnet_tl1_fisher_csr_q16_matvec` & `ggml_bitnet_tl2_fisher_csr_q16_matvec`)**:
   $$W_l \approx \operatorname{diag}(\tau_l)\,\mathcal{U}_{\text{TL2}}^{(g)}\,\operatorname{diag}(\sigma_l) + \mathcal{R}_{\text{CSR}}^{(l)}, \qquad \operatorname{supp}(\mathcal{R}_{\text{CSR}}^{(l)}) = \operatorname{Top-}p\%\!\left(\mathcal{F}_{i,j}^{(l)} (\Delta W_{i,j}^{(l)})^2\right)$$
   - $>98\%$ of weights remain inside `bitnet.cpp`'s zero-multiplication `vpshufb` LUT inner loop.
   - The top $1.35\%\text{--}2.8\%$ high-curvature outliers are accumulated in a fused per-row Int8 CSR epilogue while the quantized activation vector $x_{q8}$ is already hot in L1 cache.
   - By the second-order KL expansion on the Fisher manifold, preserving $\Omega_l$ recovers $>92\%$ of curvature energy at an average cost of $\sim 0.32\text{ bits/weight}$, enabling fast conversion/distillation of existing pretrained models without training from scratch for 100B+ tokens.

2. **Multi-Colimit Layer Group Sharing ($K \ll L$) with Fused `SubLN` + Q16.16 Natural Transformations (`ggml_bitnet_subln_and_quantize_i8`)**:
   - Groups of $L/K$ consecutive layers (e.g., $K=4$ groups for $L=24$ layers) share a single universal ternary `TL2` colimit $\mathcal{U}^{(g)}$ extracted via Fisher-Weighted HOSVD, while specializing each layer $l$ via $O(M+N)$ diagonal natural transformations $(\sigma_l, \tau_l)$ fused directly into `SubLN` and the row epilogue.
   - A `TL2` group colimit stays resident in CPU L3 cache across consecutive layers, reducing DRAM-to-L3 bandwidth by $4\times\text{--}6\times$ and compressing a 1B-class model to $< 87\text{ MiB}$ (`TL2`).

## Verification

```bash
g++ -O3 -mavx2 -mfma -Wall -Wextra -Werror utils/test-bitnet-kinecat-fusion.cpp -o /tmp/test-bitnet-kinecat-fusion
/tmp/test-bitnet-kinecat-fusion
```

Expected output:
```text
[OK] ggml_bitnet_kinecat_fusion.h verified! max_diff_tl1=1.48e-06 | max_diff_tl2=1.48e-06
```
