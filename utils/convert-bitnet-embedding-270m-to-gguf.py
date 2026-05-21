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

logger = logging.getLogger("convert-bitnet-embedding-270m")

# ---------------------------------------------------------------------------
# Tensor name mapping: HuggingFace -> GGUF
# ---------------------------------------------------------------------------

def build_tensor_name_map(n_layers: int) -> dict[str, str]:
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
            f"{pfx}.post_attention_layernorm.weight":   f"{blk}.post_attention_norm.weight",
            f"{pfx}.pre_feedforward_layernorm.weight":  f"{blk}.ffn_norm.weight",
            f"{pfx}.post_feedforward_layernorm.weight": f"{blk}.post_ffw_norm.weight",

            # Self-attention projections
            f"{pfx}.self_attn.q_proj.weight":           f"{blk}.attn_q.weight",
            f"{pfx}.self_attn.k_proj.weight":           f"{blk}.attn_k.weight",
            f"{pfx}.self_attn.v_proj.weight":           f"{blk}.attn_v.weight",
            f"{pfx}.self_attn.o_proj.weight":           f"{blk}.attn_output.weight",

            # QK head norms (Gemma3)
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

    return mapping


# ---------------------------------------------------------------------------
# Tokenizer handling (BPE for Gemma3)
# ---------------------------------------------------------------------------

def get_vocab_base_pre(tokenizer) -> str:
    chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n\U0001f680 (normal) \U0001f636‍\U0001f32b️ (multiple emojis concatenated) ✅ \U0001f999\U0001f999 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច\U0001f601 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

    chktok = tokenizer.encode(chktxt)
    chkhsh = sha256(str(chktok).encode()).hexdigest()

    logger.debug(f"chktok: {chktok}")
    logger.debug(f"chkhsh: {chkhsh}")

    res = None

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
        logger.warning(f"** chkhsh:  {chkhsh}")
        logger.warning("**************************************************************************************")
        logger.warning("\n")
        raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

    logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
    return res


def _does_token_look_special(token: str) -> bool:
    if not token:
        return False
    if token.startswith(("<|", "<", "[")) and token.endswith(("|>", ">", "]")):
        return True
    return False


def set_vocab(gguf_writer: gguf.GGUFWriter, dir_model: Path, hparams: dict):
    """Set BPE vocab for Gemma3."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(dir_model)
    vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))

    tokpre = get_vocab_base_pre(tokenizer)

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

def quantize_to_i2_s(w: np.ndarray) -> np.ndarray:
    """Quantize float weights to ternary and pack into I2_S layout.

    Uses the same quantization as BitLinear weight_quant_minmax():
        scale = 1.0 / mean(|w|)
        q = round(w * scale).clamp(-1, 1)
        dequant = q / scale = q * mean(|w|)

    Args:
        w: float weight tensor of shape (M, K)

    Returns:
        packed_data: uint8 array containing I2_S packed bytes + scale (as 4 trailing bytes)
    """
    M, K = w.shape
    n = M * K
    w_flat = w.flatten().astype(np.float32)

    abs_mean = np.mean(np.abs(w_flat))
    abs_mean = max(abs_mean, 1e-5)
    inv_scale = 1.0 / abs_mean
    q_float = np.round(w_flat * inv_scale).clip(-1, 1)

    scale = np.float32(abs_mean)

    # Map ternary {-1, 0, 1} -> I2_S encoding {0, 1, 2}
    q = np.ones(n, dtype=np.uint8)
    q[q_float > 0.5] = 2
    q[q_float < -0.5] = 0

    # Pack into I2_S layout: 128-value blocks, interleaved into 32 bytes
    pad_len = (128 - n % 128) % 128
    if pad_len:
        q = np.pad(q, (0, pad_len), constant_values=1)

    n_padded = len(q)
    n_blocks = n_padded // 128

    q = q.reshape(n_blocks, 4, 32)

    packed = (q[:, 0, :].astype(np.uint8) << 6) | \
             (q[:, 1, :].astype(np.uint8) << 4) | \
             (q[:, 2, :].astype(np.uint8) << 2) | \
             (q[:, 3, :].astype(np.uint8))

    packed = packed.reshape(-1).astype(np.uint8)

    packed_size = n // 4
    total_size = packed_size + 32
    result = np.zeros(total_size, dtype=np.uint8)
    result[:len(packed)] = packed[:packed_size]
    result[packed_size:packed_size+4] = np.frombuffer(scale.tobytes(), dtype=np.uint8)

    return result


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert bitnet-embeddings-270m (Gemma3) to GGUF")
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

    arch = hparams.get("model_type", "gemma3_text")
    assert arch == "gemma3_text", f"Expected gemma3_text architecture, got {arch}"

    n_layers = hparams["num_hidden_layers"]

    # Determine ftype
    if args.outtype == "f32":
        ftype = 0  # GGML F32
    elif args.outtype == "f16":
        ftype = 1  # GGML F16
    else:  # i2_s
        ftype = 40  # LLAMA_FTYPE_MOSTLY_I2_S

    logger.info(f"Converting {dir_model.name} to GGUF ({args.outtype})")

    # Create GGUF writer
    gguf_writer = gguf.GGUFWriter(str(args.outfile), "gemma3")

    # Set parameters
    set_gguf_parameters(gguf_writer, hparams, dir_model, ftype)

    # Set vocab
    logger.info("Setting tokenizer/vocab...")
    set_vocab(gguf_writer, dir_model, hparams)

    # Build tensor name map
    tensor_map = build_tensor_name_map(n_layers)

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

    # Write GGUF
    logger.info(f"Writing to {args.outfile}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()
