#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

# Allow using the local gguf-py if present
if "NO_LOCAL_GGUF" not in os.environ:
    _local_gguf = Path(__file__).parent / "gguf-py"
    if _local_gguf.exists():
        sys.path.insert(1, str(_local_gguf))
import gguf

logger = logging.getLogger("convert-bitnet-embedding")

# Supported architectures: model_type -> gguf arch name
SUPPORTED_ARCHS = {
    "qwen3":       "qwen3",
    "gemma3_text": "gemma3",
}

# ---------------------------------------------------------------------------
# Tensor name mapping: HuggingFace -> GGUF
# ---------------------------------------------------------------------------

def build_tensor_name_map(n_layers: int, arch: str) -> dict[str, str]:
    """Build HF tensor name -> GGUF tensor name mapping."""
    mapping: dict[str, str] = {
        "embed_tokens.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
    }

    for i in range(n_layers):
        pfx = f"layers.{i}"
        blk = f"blk.{i}"

        mapping.update({
            # Layer norms
            f"{pfx}.input_layernorm.weight":           f"{blk}.attn_norm.weight",

            # Self-attention projections
            f"{pfx}.self_attn.q_proj.weight":           f"{blk}.attn_q.weight",
            f"{pfx}.self_attn.k_proj.weight":           f"{blk}.attn_k.weight",
            f"{pfx}.self_attn.v_proj.weight":           f"{blk}.attn_v.weight",
            f"{pfx}.self_attn.o_proj.weight":           f"{blk}.attn_output.weight",

            # QK head norms
            f"{pfx}.self_attn.q_norm.weight":           f"{blk}.attn_q_norm.weight",
            f"{pfx}.self_attn.k_norm.weight":           f"{blk}.attn_k_norm.weight",

            # Per-projection input norms (BitNet-specific)
            f"{pfx}.self_attn.q_proj.norm.weight":      f"{blk}.attn_q_norm_in.weight",
            f"{pfx}.self_attn.k_proj.norm.weight":      f"{blk}.attn_k_norm_in.weight",
            f"{pfx}.self_attn.v_proj.norm.weight":      f"{blk}.attn_v_norm_in.weight",
            f"{pfx}.self_attn.o_proj.norm.weight":      f"{blk}.attn_output_norm_in.weight",

            # MLP projections
            f"{pfx}.mlp.gate_proj.weight":              f"{blk}.ffn_gate.weight",
            f"{pfx}.mlp.up_proj.weight":                f"{blk}.ffn_up.weight",
            f"{pfx}.mlp.down_proj.weight":              f"{blk}.ffn_down.weight",

            # Per-projection input norms for MLP (BitNet-specific)
            f"{pfx}.mlp.gate_proj.norm.weight":         f"{blk}.ffn_gate_norm_in.weight",
            f"{pfx}.mlp.up_proj.norm.weight":           f"{blk}.ffn_up_norm_in.weight",
            f"{pfx}.mlp.down_proj.norm.weight":         f"{blk}.ffn_down_norm_in.weight",
        })

        if arch == "qwen3":
            mapping[f"{pfx}.post_attention_layernorm.weight"] = f"{blk}.ffn_norm.weight"
        elif arch == "gemma3_text":
            mapping.update({
                f"{pfx}.post_attention_layernorm.weight":   f"{blk}.post_attention_norm.weight",
                f"{pfx}.pre_feedforward_layernorm.weight":  f"{blk}.ffn_norm.weight",
                f"{pfx}.post_feedforward_layernorm.weight": f"{blk}.post_ffw_norm.weight",
            })

    return mapping


# ---------------------------------------------------------------------------
# Tokenizer handling
# ---------------------------------------------------------------------------

def get_vocab_base_pre(tokenizer, arch: str) -> str:
    # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
    # is specific for the BPE pre-tokenizer used by the model
    # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
    # use in llama.cpp to implement the same pre-tokenizer

    chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n\U0001f680 (normal) \U0001f636‍\U0001f32b️ (multiple emojis concatenated) ✅ \U0001f999\U0001f999 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច\U0001f601 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

    chktok = tokenizer.encode(chktxt)
    chkhsh = sha256(str(chktok).encode()).hexdigest()

    logger.debug(f"chktok: {chktok}")
    logger.debug(f"chkhsh: {chkhsh}")

    res = None

    if arch == "qwen3":
        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c":
            # ref: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
            res = "qwen2"
        if chkhsh == "855d9fb74bb0b28ce2305e9cd037ff6d8c798f18d19381ddfc14bea3dc9c002f":
            # ref: multilingual-e5-0.6b-260311 (Qwen3 tokenizer variant)
            res = "qwen2"
    elif arch == "gemma3_text":
        if chkhsh == "fcb6bf9f20f6c40fa4aa4f7f99607bd6c106ca2348efdacacdca8152e59dcfe9":
            # ref: multilingual-e5-270m-260311 (Gemma3 tokenizer)
            res = "default"
        if chkhsh == "a8594e3edff7c29c003940395316294b2c623571571fc8d3d2d6571f5571cbe6":
            # ref: google/gemma-2-9b
            res = "default"

    if res is None:
        logger.warning("\n")
        logger.warning("**************************************************************************************")
        logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
        logger.warning("**          There are 2 possible reasons for this:")
        logger.warning("**          - the model has not been added to convert_hf_to_gguf_update.py yet")
        logger.warning("**          - the pre-tokenization config has changed upstream")
        logger.warning("**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.")
        logger.warning("** ref:     https://github.com/ggml-org/llama.cpp/pull/6920")
        logger.warning("**")
        logger.warning(f"** chkhsh:  {chkhsh}")
        logger.warning("**************************************************************************************")
        logger.warning("\n")
        raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

    logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
    logger.debug(f"chkhsh: {chkhsh}")

    return res


def _does_token_look_special(token: str) -> bool:
    """Check if a token looks like a special token (e.g., <|...|>, <...>)."""
    if not token:
        return False
    # Matches patterns like <|endoftext|>, <s>, </s>, [CLS], [SEP], etc.
    if token.startswith(("<|", "<", "[")) and token.endswith(("|>", ">", "]")):
        return True
    return False


def set_vocab(gguf_writer: gguf.GGUFWriter, dir_model: Path, hparams: dict, arch: str):
    """Set tokenizer vocab.

    - Qwen3: BPE tokenizer (tokenizer.ggml.model = "gpt2")
    - Gemma3: SPM-compatible tokenizer from tokenizer.json (tokenizer.ggml.model = "llama")
      Gemma uses SentencePiece-style tokenization with ▁ space prefix and byte fallback.
      Using "llama" model type ensures llama.cpp uses the correct SPM pre-tokenizer
      instead of the BPE regex-based pre-tokenizer which breaks CJK tokenization.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(dir_model)
    vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))

    if arch == "gemma3_text":
        _set_vocab_gemma3(gguf_writer, dir_model, tokenizer, vocab_size)
    else:
        _set_vocab_bpe(gguf_writer, dir_model, tokenizer, vocab_size, arch)


def _set_vocab_bpe(gguf_writer: gguf.GGUFWriter, dir_model: Path,
                   tokenizer, vocab_size: int, arch: str):
    """Set BPE vocab (for Qwen3)."""
    tokpre = get_vocab_base_pre(tokenizer, arch)

    tokens: list[str] = []
    toktypes: list[int] = []

    reverse_vocab = {id_: tok for tok, id_ in tokenizer.vocab.items()}
    added_vocab = tokenizer.get_added_vocab()

    added_tokens_decoder = tokenizer.added_tokens_decoder

    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
        elif reverse_vocab[i] in added_vocab:
            token = reverse_vocab[i]

            # Only encode-decode non-normalized tokens (matching llama.cpp upstream)
            if not added_tokens_decoder[i].normalized:
                token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))

            if added_tokens_decoder[i].special or _does_token_look_special(token):
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")
                toktypes.append(gguf.TokenType.USER_DEFINED)

            tokens.append(token)
        else:
            tokens.append(reverse_vocab[i])
            toktypes.append(gguf.TokenType.NORMAL)

    gguf_writer.add_tokenizer_model("gpt2")
    gguf_writer.add_tokenizer_pre(tokpre)
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(dir_model, load_merges=True)

    if arch == "qwen3":
        # Override EOS token: PyTorch tokenizer appends <|endoftext|> (151643) as the
        # sentence-end marker, not <|im_end|> (151645). For last-token pooling to work
        # correctly, llama.cpp must append the same token.
        special_vocab.special_token_ids["eos"] = 151643

    special_vocab.add_to_gguf(gguf_writer)

    if arch == "qwen3":
        # Embedding models need EOS token appended for last-token pooling
        gguf_writer.add_add_eos_token(True)


def _set_vocab_gemma3(gguf_writer: gguf.GGUFWriter, dir_model: Path,
                      tokenizer, vocab_size: int):
    """Set SPM-compatible vocab for Gemma3.

    Gemma's tokenizer is SentencePiece-based (BPE variant with ▁ space prefix
    and byte fallback). We read tokenizer.json to extract vocab and compute
    BPE merge scores, then write as tokenizer.ggml.model = "llama" so llama.cpp
    uses the SPM code path (correct pre-tokenizer behavior for CJK etc.).

    Score assignment:
      - BPE merge results get scores derived from merge rank (lower rank = higher score)
      - Single-char / byte tokens get score 0
      - Special / added tokens get score -1000
    """
    tokenizer_json_file = dir_model / "tokenizer.json"
    if not tokenizer_json_file.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {dir_model}")

    with open(tokenizer_json_file, encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    bpe_vocab = tokenizer_json["model"]["vocab"]  # token_str -> token_id
    bpe_merges = tokenizer_json["model"].get("merges", [])

    # Build merge result -> rank mapping for score computation
    # merge_scores[result_token] = -rank (lower rank = earlier merge = higher priority)
    merge_scores: dict[str, float] = {}
    for rank, merge in enumerate(bpe_merges):
        if isinstance(merge, list):
            result = "".join(merge)
        else:
            parts = merge.split(" ", 1)
            result = "".join(parts)
        if result not in merge_scores:
            merge_scores[result] = -float(rank)

    # Build token arrays
    reverse_vocab = {v: k for k, v in bpe_vocab.items()}
    added_tokens_decoder = tokenizer.added_tokens_decoder

    tokens: list[bytes] = []
    scores: list[float] = []
    toktypes: list[int] = []

    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]".encode("utf-8"))
            scores.append(-10000.0)
            toktypes.append(gguf.TokenType.UNUSED)
            continue

        token_str = reverse_vocab[i]
        token_bytes = token_str.encode("utf-8")

        # Determine token type
        if i in added_tokens_decoder:
            tok_data = added_tokens_decoder[i]
            if tok_data.special or _does_token_look_special(token_str):
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                toktypes.append(gguf.TokenType.USER_DEFINED)
            scores.append(-1000.0)
        elif token_str.startswith("<0x") and token_str.endswith(">") and len(token_str) == 6:
            # Byte token: <0xHH>
            toktypes.append(gguf.TokenType.BYTE)
            scores.append(0.0)
        elif token_str == "<unk>":
            toktypes.append(gguf.TokenType.UNKNOWN)
            scores.append(0.0)
        else:
            toktypes.append(gguf.TokenType.NORMAL)
            # Score from merge rank, or 0 for single-char tokens
            scores.append(merge_scores.get(token_str, 0.0))

        tokens.append(token_bytes)

    gguf_writer.add_tokenizer_model("llama")
    gguf_writer.add_tokenizer_pre("default")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)
    gguf_writer.add_add_space_prefix(False)

    special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
    special_vocab.add_to_gguf(gguf_writer)


# ---------------------------------------------------------------------------
# GGUF metadata
# ---------------------------------------------------------------------------

def set_gguf_parameters(gguf_writer: gguf.GGUFWriter, hparams: dict, dir_model: Path, ftype: int):
    gguf_writer.add_name(dir_model.name)

    n_layers = hparams["num_hidden_layers"]
    n_embd = hparams["hidden_size"]
    n_head = hparams["num_attention_heads"]
    n_head_kv = hparams.get("num_key_value_heads", n_head)
    n_ff = hparams["intermediate_size"]

    gguf_writer.add_block_count(n_layers)
    gguf_writer.add_context_length(hparams.get("max_position_embeddings", 32768))
    gguf_writer.add_embedding_length(n_embd)
    gguf_writer.add_feed_forward_length(n_ff)
    gguf_writer.add_head_count(n_head)
    gguf_writer.add_head_count_kv(n_head_kv)
    gguf_writer.add_vocab_size(hparams["vocab_size"])

    head_dim = hparams.get("head_dim", n_embd // n_head)
    gguf_writer.add_rope_dimension_count(head_dim)
    gguf_writer.add_key_length(head_dim)
    gguf_writer.add_value_length(head_dim)

    if hparams.get("rope_theta") is not None:
        gguf_writer.add_rope_freq_base(hparams["rope_theta"])
    if hparams.get("rms_norm_eps") is not None:
        gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

    gguf_writer.add_file_type(ftype)

    # Pooling type for embedding models
    # Try to read from modules.json / 1_Pooling/config.json (sentence-transformers convention)
    pooling_type = None
    module_path = dir_model / "modules.json"
    if module_path.is_file():
        with open(module_path, encoding="utf-8") as f:
            modules = json.load(f)
        for mod in modules:
            if mod["type"].endswith("Pooling"):
                pooling_path = dir_model / mod["path"] / "config.json"
                if pooling_path.is_file():
                    with open(pooling_path, encoding="utf-8") as f:
                        pooling = json.load(f)
                    if pooling.get("pooling_mode_mean_tokens"):
                        pooling_type = gguf.PoolingType.MEAN
                    elif pooling.get("pooling_mode_cls_token"):
                        pooling_type = gguf.PoolingType.CLS
                    elif pooling.get("pooling_mode_lasttoken"):
                        pooling_type = gguf.PoolingType.LAST
                break
    if pooling_type is None:
        # Default to MEAN pooling for embedding models
        logger.info("  No pooling config found, defaulting to MEAN pooling")
        pooling_type = gguf.PoolingType.MEAN
    gguf_writer.add_pooling_type(pooling_type)

    logger.info(f"  n_layers={n_layers}, n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, n_ff={n_ff}, head_dim={head_dim}")


# ---------------------------------------------------------------------------
# Tensor iteration from safetensors
# ---------------------------------------------------------------------------

def iter_tensors(dir_model: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (name, tensor) from safetensors files."""
    from safetensors import safe_open

    safetensor_files = sorted(dir_model.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files in {dir_model}")

    for sf_path in safetensor_files:
        logger.info(f"Loading {sf_path.name}")
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


# ---------------------------------------------------------------------------
# I2_S ternary packing (platform-independent)
# ---------------------------------------------------------------------------
#
# I2_S format (from dequantize_row_i2_s in ggml-quants.c):
#   - Every 128 values form a block, packed into 32 bytes
#   - Each byte stores 4 values at positions [0*32+gp, 1*32+gp, 2*32+gp, 3*32+gp]
#     where gp is the byte index within the 32-byte group
#   - Encoding per byte: c0=(b>>6)&3, c1=(b>>4)&3, c2=(b>>2)&3, c3=(b>>0)&3
#   - Value mapping: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0
#   - Scale is stored as a separate tensor (tensor_name + "_scale")

def quantize_to_i2_s(w: np.ndarray) -> np.ndarray:
    """Quantize float weights to ternary and pack into I2_S layout.

    Uses the same quantization as BitLinear weight_quant_minmax():
        scale = 1.0 / mean(|w|)
        q = round(w * scale).clamp(-1, 1)
        dequant = q / scale = q * mean(|w|)

    The I2_S format is self-contained: packed ternary bytes followed by a f32 scale
    appended at the end of the data buffer.

    Args:
        w: float weight tensor of shape (M, K)

    Returns:
        packed_data: uint8 array containing I2_S packed bytes + scale (as 4 trailing bytes)
    """
    M, K = w.shape
    n = M * K
    w_flat = w.flatten().astype(np.float32)

    # BitLinear weight_quant_minmax: scale = 1/mean(|w|), then round & clamp
    abs_mean = np.mean(np.abs(w_flat))
    abs_mean = max(abs_mean, 1e-5)
    inv_scale = 1.0 / abs_mean
    q_float = np.round(w_flat * inv_scale).clip(-1, 1)  # ternary: {-1, 0, 1}

    # scale for dequantization = abs_mean (i.e., dequant = q * abs_mean)
    scale = np.float32(abs_mean)

    # Map ternary {-1, 0, 1} -> I2_S encoding {0, 1, 2}
    #   -1 -> 0,  0 -> 1,  +1 -> 2
    q = np.ones(n, dtype=np.uint8)  # default to 1 (zero)
    q[q_float > 0.5] = 2    # +1 -> 2
    q[q_float < -0.5] = 0   # -1 -> 0

    # Pack into I2_S layout: 128-value blocks, interleaved into 32 bytes
    # Pad to multiple of 128
    pad_len = (128 - n % 128) % 128
    if pad_len:
        q = np.pad(q, (0, pad_len), constant_values=1)

    n_padded = len(q)
    n_blocks = n_padded // 128

    q = q.reshape(n_blocks, 4, 32)

    # Pack: byte = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3
    packed = (q[:, 0, :].astype(np.uint8) << 6) | \
             (q[:, 1, :].astype(np.uint8) << 4) | \
             (q[:, 2, :].astype(np.uint8) << 2) | \
             (q[:, 3, :].astype(np.uint8))

    packed = packed.reshape(-1).astype(np.uint8)

    # I2_S format: packed_bytes + 32-byte aligned tail (scale in first 4 bytes of tail)
    # Total size = n_elements / 4 + 32  (as defined in ggml.c)
    packed_size = n // 4
    total_size = packed_size + 32
    result = np.zeros(total_size, dtype=np.uint8)
    result[:len(packed)] = packed[:packed_size]
    # Write scale as float32 at offset packed_size
    result[packed_size:packed_size+4] = np.frombuffer(scale.tobytes(), dtype=np.uint8)

    return result


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert bitnet-embeddings (Qwen3/Gemma3) to GGUF")
    parser.add_argument("model", type=Path, help="Model directory")
    parser.add_argument("--outfile", type=Path, default=None, help="Output GGUF file")
    parser.add_argument("--outtype", choices=["f32", "f16", "i2_s"], default="f16",
                        help="Output type: f32, f16, or i2_s (ternary quantized)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dir_model = args.model
    if not dir_model.is_dir():
        logger.error(f"{dir_model} is not a directory")
        sys.exit(1)

    # Default output filename
    if args.outfile is None:
        suffix = {"f32": "-f32", "f16": "-f16", "i2_s": "-f16-new-i2_s"}[args.outtype]
        args.outfile = dir_model / f"{dir_model.name}{suffix}.gguf"

    # Load config
    with open(dir_model / "config.json") as f:
        hparams = json.load(f)

    arch = hparams.get("model_type", "")
    if arch not in SUPPORTED_ARCHS:
        logger.error(f"Unsupported model_type '{arch}'. Supported: {list(SUPPORTED_ARCHS.keys())}")
        sys.exit(1)

    gguf_arch = SUPPORTED_ARCHS[arch]
    n_layers = hparams["num_hidden_layers"]

    # Determine ftype
    if args.outtype == "f32":
        ftype = 0  # GGML F32
    elif args.outtype == "f16":
        ftype = 1  # GGML F16
    else:  # i2_s
        ftype = 40  # LLAMA_FTYPE_MOSTLY_I2_S

    logger.info(f"Converting {dir_model.name} (arch={arch}) to GGUF ({args.outtype})")

    # Create GGUF writer
    gguf_writer = gguf.GGUFWriter(str(args.outfile), gguf_arch)

    # Set parameters
    set_gguf_parameters(gguf_writer, hparams, dir_model, ftype)

    # Set vocab
    logger.info("Setting tokenizer/vocab...")
    set_vocab(gguf_writer, dir_model, hparams, arch)

    # Build tensor name map
    tensor_map = build_tensor_name_map(n_layers, arch)

    # Process tensors
    logger.info("Processing tensors...")
    tensor_count = 0
    for hf_name, data_torch in iter_tensors(dir_model):
        # Skip tensors we don't need
        if hf_name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
            continue

        # Strip "model." prefix if present
        name = hf_name
        if name.startswith("model."):
            name = name[len("model."):]

        # Look up GGUF name
        gguf_name = tensor_map.get(name)
        if gguf_name is None:
            logger.warning(f"Skipping unmapped tensor: {hf_name}")
            continue

        old_dtype = data_torch.dtype

        # Convert bf16 -> f32 first (bf16 not directly supported by gguf)
        if data_torch.dtype == torch.bfloat16:
            data_torch = data_torch.to(torch.float32)

        data = data_torch.squeeze().numpy()
        n_dims = len(data.shape)
        data_shape = data.shape

        # Determine if this is a linear weight suitable for ternary quantization
        is_norm = gguf_name.endswith("_norm.weight") or gguf_name.endswith("_norm_in.weight")
        is_embed = gguf_name == "token_embd.weight"
        is_linear_weight = n_dims == 2 and not is_norm and not is_embed
        suit_i2 = is_linear_weight

        if args.outtype == "i2_s" and suit_i2:
            # --- I2_S ternary packing (scale embedded in data) ---
            packed = quantize_to_i2_s(data)
            data_qtype = gguf.GGMLQuantizationType.I2_S

            shape_str = f"{{{', '.join(str(n) for n in reversed(data_shape))}}}"
            logger.info(f"  {gguf_name}: {list(data_shape)} {old_dtype} -> I2_S, shape = {shape_str}")

            gguf_writer.add_tensor(gguf_name, packed, raw_shape=data_shape, raw_dtype=data_qtype)
            tensor_count += 1

        elif args.outtype in ("f16", "i2_s") and (is_linear_weight or is_embed):
            # 2D weight tensors (linear + embedding) -> f16
            data = data.astype(np.float16)
            logger.info(f"  {gguf_name}: {list(data_torch.shape)} {old_dtype} -> float16")
            gguf_writer.add_tensor(gguf_name, data)
            tensor_count += 1

        else:
            # norms, 1D tensors
            # Gemma3 RMSNorm uses (1+w)*x instead of w*x; preprocess w -> w+1
            # so llama.cpp's standard RMSNorm produces correct results.
            # NOTE: *_norm_in weights are BitLinear standard RMSNorm (initialized ~1.0),
            # NOT Gemma3RMSNorm (initialized ~0.0), so they must NOT get +1.
            is_gemma3_native_norm = (arch == "gemma3_text" and is_norm
                                     and not gguf_name.endswith("_norm_in.weight"))
            if is_gemma3_native_norm:
                data = data.astype(np.float32) + 1.0
                logger.info(f"    [Gemma3 norm offset] {gguf_name}: applied w = w + 1")

            if args.outtype in ("f16", "i2_s"):
                data = data.astype(np.float16)
                logger.info(f"  {gguf_name}: {list(data_torch.shape)} {old_dtype} -> float16")
            else:
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                logger.info(f"  {gguf_name}: {list(data_torch.shape)} {old_dtype} -> float32")
            gguf_writer.add_tensor(gguf_name, data)
            tensor_count += 1

    logger.info(f"Total tensors written: {tensor_count}")

    # Note: output.weight (lm_head) is skipped for embedding models —
    # it is not needed (no token generation) and saves ~297MB for this model.

    # Write GGUF
    logger.info(f"Writing to {args.outfile}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()
