# Verification Report: Entropy-Adaptive KV Cache Claims

**Date:** 2026-03-19
**PR:** microsoft/BitNet #497
**Author:** SCJedi
**Status:** Correction required -- several claims are model-specific, not general

## What Was Claimed

PR #497 made three central claims:

1. **KV cache is 68.6% of total memory** for BitNet inference
2. **49.8% of attention heads are sink/focused** (entropy < 1.5 bits)
3. **2-5x compression achieves ~96% quality retention**

These claims were based on profiling `bitnet_b1_58-large` (~0.7B parameters), which uses **multi-head attention (MHA)** with 16 KV heads.

## What We Measured

We ran verification against the flagship model that most users of this repository will actually use: **BitNet-b1.58-2B-4T** (2.4B parameters). This model uses **Grouped Query Attention (GQA)** with a 4:1 ratio -- 20 query heads but only 5 KV heads.

### Architecture Comparison

| Parameter | bitnet_b1_58-large (PR basis) | BitNet-2B-4T (verified) |
|-----------|-------------------------------|------------------------|
| Parameters | ~0.7B | ~2.4B |
| Layers | 24 | 30 |
| Query heads | 16 | 20 |
| **KV heads** | **16 (MHA)** | **5 (GQA 4:1)** |
| Head dimension | 96 | 128 |
| Max context | 2048 | 4096 |

The GQA architecture is the critical difference. It reduces KV cache size by 4x relative to MHA, which fundamentally changes the memory profile.

### Claim 1: KV Cache Memory Fraction

**Original claim: 68.6%**

| Model | Seq Length | Weights (MB) | KV Cache (MB) | KV % |
|-------|-----------|-------------|--------------|------|
| bitnet_b1_58-large | 2048 | 153.6 | 302.0 | **65.4%** |
| BitNet-2B-4T | 512 | 476.5 | 39.3 | **7.6%** |
| BitNet-2B-4T | 1024 | 476.5 | 78.6 | **14.0%** |
| BitNet-2B-4T | 2048 | 476.5 | 157.3 | **24.4%** |
| BitNet-2B-4T | 4096 | 476.5 | 314.6 | **38.7%** |

**Verdict:** The 68.6% figure is approximately correct for bitnet_b1_58-large (we calculate 65.4%). However, for BitNet-2B-4T, KV cache is only 24.4% at seq_len=2048, rising to 38.7% at max context (4096). The claim does not generalize across architectures.

### Claim 2: Sink/Focused Head Prevalence

**Original claim: 49.8%**

Measured across 600 attention heads (30 layers x 20 heads) on 20 diverse calibration texts:

| Category | Count | Percentage |
|----------|-------|------------|
| Sink (entropy < 0.5) | 21 | 3.5% |
| Focused (0.5 - 1.5) | 53 | 8.8% |
| Moderate (1.5 - 2.5) | 386 | 64.3% |
| Mixed (2.5 - 3.5) | 139 | 23.2% |
| Diffuse (> 3.5) | 1 | 0.2% |
| **Sink + Focused** | **74** | **12.3%** |

**Verdict:** Only 12.3% of heads in BitNet-2B-4T are sink/focused, compared to 49.8% in bitnet_b1_58-large. The entropy distribution is model-dependent. The larger model has learned more distributed attention patterns.

Entropy statistics:
- Mean: 2.402 bits (std: 0.785)
- Range: 0.001 to 4.063 bits
- Coefficient of variation: 0.327

### Claim 3: Compression Quality

**Original claim: ~96% at 2-5x compression**

Measured via cosine similarity between full and compressed attention patterns:

| Compression | Attention Preservation | Budget Range |
|------------|----------------------|-------------|
| 2x | **99.0%** | [0.355, 0.710] |
| 5x | **96.2%** | [0.142, 0.284] |
| 10x | **90.8%** | [0.071, 0.142] |

**Verdict:** Quality retention holds up and is actually slightly better than claimed. At 5x compression, we measure 96.2% attention pattern preservation. The entropy-adaptive allocation strategy works as designed.

**Caveat:** These numbers measure attention pattern similarity, not end-to-end generation quality. True quality assessment requires benchmarking with the C++ implementation on downstream tasks.

## Why The Numbers Differ

The discrepancy has a single root cause: **architecture**.

`bitnet_b1_58-large` uses multi-head attention (MHA), where every query head has its own KV head. This means:
- KV cache scales linearly with the number of attention heads
- With 1.58-bit weights, the KV cache (in float16) quickly dominates memory
- Many heads can specialize into sink/focused patterns because each head is independent

`BitNet-b1.58-2B-4T` uses grouped query attention (GQA 4:1), where 4 query heads share 1 KV head. This means:
- KV cache is 4x smaller relative to model size
- KV heads serve multiple query patterns, which pushes them toward moderate entropy
- The memory savings from GQA overlap with what entropy-adaptive eviction provides

This is not a flaw in the technique. It is a factual correction about which models benefit most.

## What Holds Up

1. **The technique is architecturally sound.** Attention heads in BitNet models exhibit significant entropy variation (CV = 0.327), confirming that uniform KV cache budgets are suboptimal.

2. **Compression quality is real.** 96.2% attention preservation at 5x compression on a model with only 12.3% sink/focused heads is a strong result. It means the adaptive allocation works even when the entropy distribution is less extreme.

3. **The implementation is correct.** GQA support was already built into the code. The calibration script handles grouped heads properly.

4. **Sink preservation is critical.** Our verification confirms the finding that position-0 sink heads exist in BitNet-2B-4T (3.5% of heads), and these must be preserved during eviction.

## What Does Not Transfer

1. **The 68.6% memory fraction** applies only to MHA models at long context. For GQA models, KV cache is a smaller fraction of total memory (24-39% for BitNet-2B-4T depending on context length).

2. **The 49.8% sink/focused figure** is specific to bitnet_b1_58-large. Other models will have different entropy distributions.

3. **Total memory savings projections** (34% at 2x, 55% at 5x, 62% at 10x) were calculated for bitnet_b1_58-large and do not apply to GQA models.

## Where Entropy-Adaptive KV Cache Remains Valuable

1. **Long-context inference.** Even with GQA, KV cache grows linearly with sequence length. At 4096 tokens, it is already 38.7% of memory for BitNet-2B-4T. At 8K-16K context (increasingly common), it would dominate again.

2. **MHA models.** Any BitNet model using multi-head attention (like bitnet_b1_58-large) gets the full benefit.

3. **Edge deployment.** When running on memory-constrained devices, even a 24% memory component is worth compressing, especially if it enables fitting the model at all.

4. **Future models.** The technique is architecture-agnostic. As BitNet models scale to larger context windows, KV cache compression becomes increasingly relevant regardless of GQA.

## Methodology

The verification was performed by loading the actual BitNet-b1.58-2B-4T bf16 weights (Q/K/V/O projections) into a Llama model skeleton. The attention computation (QK^T/sqrt(d), softmax, V multiplication) is identical between Llama and BitNet architectures, so **attention patterns are accurate**. However, MLP outputs and final logits are incorrect due to BitNet's custom components (attn_sub_norm, ffn_sub_norm, relu2 activation), so absolute perplexity numbers are not meaningful.

The full raw data, per-layer entropy heatmaps, and detailed methodology are in `tools/KV_VERIFICATION_REPORT.md`.

Verification script: `tools/verify_kv_savings.py`

## Recommended PR Updates

1. Qualify the 68.6% claim as specific to bitnet_b1_58-large with MHA
2. Note that BitNet-2B-4T (GQA) has a 24-39% KV cache fraction depending on context
3. State that entropy distributions are model-dependent
4. Keep the compression quality claims (they hold up)
5. Add a note that GQA models benefit less but still benefit at long context

## Conclusion

The entropy-adaptive KV cache technique works. The compression quality is confirmed. The specific numbers in the PR description are accurate for the model they were measured on (bitnet_b1_58-large), but they do not generalize to the GQA-based BitNet-2B-4T model that most users will encounter. The PR description should be updated to reflect this.

Our findings matter more than being right about specific numbers.
