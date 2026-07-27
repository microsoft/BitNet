#!/usr/bin/env python3
"""
Entropy Calibration Script for BitNet KV Cache Compression
===========================================================

Generates per-head attention entropy profiles for use with entropy-adaptive
KV cache compression in BitNet/llama.cpp.

Usage:
    python calibrate_entropy.py --model 1bitLLM/bitnet_b1_58-large --output entropy_config.json
    python calibrate_entropy.py --model microsoft/bitnet-b1.58-2B-4T --output entropy_config.json

The output JSON file is consumed by entropy_kv_cache.cpp at inference startup.

Requirements:
    pip install torch transformers accelerate

Runtime: ~2-5 minutes on CPU for a 0.7B model, ~30 seconds for calibration only.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_model(model_id: str, device: str = "cpu"):
    """Load a HuggingFace model with attention output enabled."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Try loading with eager attention (needed for output_attentions)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=device if device != "cpu" else None,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    except Exception:
        # Fallback without specifying attention implementation
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s")

    return model, tokenizer


def get_calibration_texts(n_sequences: int = 20, max_length: int = 128) -> list[str]:
    """Get calibration texts. Uses a mix of patterns to capture diverse attention."""
    texts = [
        # Factual/encyclopedic
        "The transformer architecture was introduced in 2017 by Vaswani et al. in the paper "
        "'Attention Is All You Need'. It replaced recurrent neural networks for sequence modeling "
        "tasks and became the foundation for models like BERT, GPT, and their successors.",

        "Python is a high-level programming language known for its readability and versatility. "
        "It supports multiple programming paradigms including procedural, object-oriented, and "
        "functional programming. Python's standard library provides tools for many tasks.",

        "The solar system consists of eight planets orbiting the Sun. The inner planets are "
        "Mercury, Venus, Earth, and Mars. The outer planets are Jupiter, Saturn, Uranus, and "
        "Neptune. Pluto was reclassified as a dwarf planet in 2006.",

        # Narrative
        "Once upon a time, in a small village nestled between two mountains, there lived a "
        "young inventor who dreamed of building a machine that could fly. Every day, she would "
        "study the birds that soared above the valley, sketching their wing patterns.",

        "The detective examined the room carefully. The window was open, a chair was overturned, "
        "and there were muddy footprints leading from the door to the desk. The safe was empty. "
        "Someone had been here within the last hour.",

        # Technical
        "In machine learning, the key-value cache stores previously computed attention keys and "
        "values during autoregressive generation. This avoids redundant computation but consumes "
        "significant memory, especially for long sequences and large batch sizes.",

        "The HTTP protocol defines methods for client-server communication. GET requests retrieve "
        "data, POST requests submit data, PUT requests update existing resources, and DELETE "
        "requests remove resources. Each request includes headers and an optional body.",

        # Conversational
        "User: What is the capital of France?\nAssistant: The capital of France is Paris. It is "
        "the largest city in France and serves as the country's political, economic, and cultural "
        "center. Paris is known for landmarks like the Eiffel Tower and the Louvre Museum.",

        # Code-like
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    "
        "for i in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n"
        "# Test the function\nfor i in range(10):\n    print(f'fib({i}) = {fibonacci(i)}')",

        # Lists and structure
        "Steps to make a cup of tea: 1. Boil water in a kettle. 2. Place a tea bag in a cup. "
        "3. Pour the boiling water over the tea bag. 4. Let it steep for 3-5 minutes. "
        "5. Remove the tea bag. 6. Add milk and sugar to taste. 7. Stir and enjoy.",

        # Mathematical
        "The quadratic formula gives the solutions of ax^2 + bx + c = 0 as "
        "x = (-b +/- sqrt(b^2 - 4ac)) / (2a). When the discriminant b^2 - 4ac is positive, "
        "there are two real solutions. When it equals zero, there is one repeated root.",

        # Scientific
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight. "
        "The process occurs in two stages: light-dependent reactions in the thylakoid membranes "
        "and the Calvin cycle in the stroma of chloroplasts.",

        # Historical
        "The Industrial Revolution began in Britain in the late 18th century. Key inventions "
        "included the spinning jenny, the steam engine, and the power loom. These innovations "
        "transformed manufacturing from cottage industries to factory-based production.",

        # Dialogue
        "Alice: Did you finish the report?\nBob: Almost. I need to add the charts.\n"
        "Alice: The deadline is tomorrow. Need any help?\nBob: Could you review the introduction? "
        "I want to make sure the methodology section is clear.",

        # Abstract reasoning
        "If all roses are flowers and all flowers need water, then all roses need water. "
        "This is an example of a syllogism, a form of deductive reasoning where a conclusion "
        "follows necessarily from two premises.",

        # News-like
        "Global temperatures continued to rise in 2025, with the average surface temperature "
        "reaching 1.3 degrees Celsius above pre-industrial levels. Climate scientists emphasized "
        "the urgency of reducing greenhouse gas emissions to limit further warming.",

        # Medical
        "The human heart beats approximately 100,000 times per day, pumping about 7,500 liters "
        "of blood through a network of blood vessels that would stretch approximately 96,000 "
        "kilometers if laid end to end.",

        # Legal
        "The contract shall be governed by the laws of the State of California. Any disputes "
        "arising from this agreement shall be resolved through binding arbitration in accordance "
        "with the rules of the American Arbitration Association.",

        # Philosophical
        "Descartes' famous statement 'I think, therefore I am' established the certainty of "
        "one's own existence as a thinking being. This became the foundation of modern Western "
        "philosophy and the starting point for his method of systematic doubt.",

        # Instructions
        "To install the software: First, download the installer from the official website. "
        "Run the executable and follow the on-screen instructions. Select the installation "
        "directory and choose which components to install. Click 'Install' to begin.",
    ]

    # Return requested number of sequences
    return texts[:n_sequences]


def compute_head_entropy(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 128,
    device: str = "cpu",
) -> dict:
    """
    Compute per-head attention entropy across calibration texts.

    Returns a dict mapping "L{layer}_H{head}" to average entropy (bits).
    """
    print(f"Running calibration on {len(texts)} sequences...")
    t0 = time.time()

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

    # Accumulate entropy per head
    entropy_sum = np.zeros((n_layers, n_heads))
    entropy_count = np.zeros((n_layers, n_heads))

    for text_idx, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        if seq_len < 4:
            continue

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        # Sample entropy at representative positions
        positions = [
            seq_len // 4,
            seq_len // 2,
            3 * seq_len // 4,
            seq_len - 1,
        ]

        for layer_idx, attn_weights in enumerate(outputs.attentions):
            # attn_weights shape: [batch, n_heads, seq_len, seq_len]
            for head_idx in range(n_heads):
                for pos in positions:
                    if pos < 1:
                        continue
                    # Get attention distribution for this query position
                    row = attn_weights[0, head_idx, pos, : pos + 1]
                    row = row.float().cpu().numpy()

                    # Compute entropy: H = -sum(p * log2(p))
                    row_pos = row[row > 1e-10]
                    if len(row_pos) > 0:
                        entropy = -(row_pos * np.log2(row_pos)).sum()
                        entropy_sum[layer_idx, head_idx] += entropy
                        entropy_count[layer_idx, head_idx] += 1

        if (text_idx + 1) % 5 == 0:
            print(f"  Processed {text_idx + 1}/{len(texts)} sequences...")

    # Average entropy per head
    mask = entropy_count > 0
    entropy_avg = np.zeros_like(entropy_sum)
    entropy_avg[mask] = entropy_sum[mask] / entropy_count[mask]

    # Build output dict
    head_entropies = {}
    for l in range(n_layers):
        for h in range(n_heads):
            key = f"L{l}_H{h}"
            head_entropies[key] = float(entropy_avg[l, h])

    elapsed = time.time() - t0
    print(f"Calibration complete in {elapsed:.1f}s")

    return head_entropies


def build_config(
    model,
    head_entropies: dict,
    model_id: str,
) -> dict:
    """Build the full config JSON structure."""
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    n_ctx = getattr(config, "max_position_embeddings", 2048)
    hidden_size = config.hidden_size
    head_dim = hidden_size // n_heads

    # Compute entropy stats
    values = list(head_entropies.values())
    entropy_stats = {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "cv": float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0,
        "range_ratio": float(np.max(values) / np.min(values)) if np.min(values) > 1e-10 else 0.0,
    }

    # Classify head types
    head_types = {"sink": 0, "focused": 0, "moderate": 0, "mixed": 0, "diffuse": 0}
    for v in values:
        if v < 0.5:
            head_types["sink"] += 1
        elif v < 1.5:
            head_types["focused"] += 1
        elif v < 3.0:
            head_types["moderate"] += 1
        elif v < 4.0:
            head_types["mixed"] += 1
        else:
            head_types["diffuse"] += 1

    result = {
        "format_version": 1,
        "model": model_id,
        "n_layer": n_layers,
        "n_head": n_heads,
        "n_head_kv": n_kv_heads,
        "n_ctx": n_ctx,
        "model_info": {
            "hidden_size": hidden_size,
            "num_hidden_layers": n_layers,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv_heads,
            "max_position_embeddings": n_ctx,
            "head_dim": head_dim,
            "gqa_ratio": n_heads / n_kv_heads,
            "is_gqa": n_heads != n_kv_heads,
        },
        "calibration": {
            "n_sequences": 20,
            "n_sample_positions": 4,
            "method": "mean_entropy_at_representative_positions",
        },
        "entropy_stats": entropy_stats,
        "head_types": head_types,
        "head_entropies": head_entropies,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate attention entropy for KV cache compression"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="HuggingFace model ID (e.g., 1bitLLM/bitnet_b1_58-large)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="entropy_config.json",
        help="Output JSON file path (default: entropy_config.json)",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=20,
        help="Number of calibration sequences (default: 20)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for calibration (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--from-results",
        type=str,
        default=None,
        help="Skip calibration, load from existing results JSON (e.g., bitnet_kv_results.json)",
    )

    args = parser.parse_args()

    if args.from_results is None and args.model is None:
        print("Error: either --model or --from-results is required")
        sys.exit(1)

    if args.from_results:
        # Load from existing calibration results
        print(f"Loading calibration data from {args.from_results}")
        with open(args.from_results) as f:
            results = json.load(f)

        if "head_entropies" not in results:
            print("Error: no 'head_entropies' in results file")
            sys.exit(1)

        # Build config from existing data
        config = {
            "format_version": 1,
            "model": results.get("model_info", {}).get("model_id", "unknown"),
            "n_layer": results.get("model_info", {}).get("num_hidden_layers", 0),
            "n_head": results.get("model_info", {}).get("num_attention_heads", 0),
            "n_head_kv": results.get("model_info", {}).get("num_key_value_heads", 0),
            "n_ctx": results.get("model_info", {}).get("max_position_embeddings", 2048),
            "model_info": results.get("model_info", {}),
            "entropy_stats": results.get("entropy_stats", {}),
            "head_types": results.get("head_types", {}),
            "head_entropies": results["head_entropies"],
        }

        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Config written to {output_path}")
        n_heads_total = len(results["head_entropies"])
        print(f"  {n_heads_total} heads profiled")
        if "entropy_stats" in results:
            stats = results["entropy_stats"]
            print(f"  Entropy: mean={stats.get('mean', 0):.3f}, "
                  f"range=[{stats.get('min', 0):.3f}, {stats.get('max', 0):.3f}], "
                  f"CV={stats.get('cv', 0):.3f}")
        return

    # Full calibration
    model, tokenizer = load_model(args.model, args.device)
    texts = get_calibration_texts(args.n_sequences, args.max_length)
    head_entropies = compute_head_entropy(
        model, tokenizer, texts, args.max_length, args.device
    )
    config = build_config(model, head_entropies, args.model)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig written to {output_path}")
    print(f"  {len(head_entropies)} heads profiled")
    stats = config["entropy_stats"]
    print(f"  Entropy: mean={stats['mean']:.3f}, "
          f"range=[{stats['min']:.3f}, {stats['max']:.3f}], "
          f"CV={stats['cv']:.3f}")
    print(f"\nTo use with BitNet inference:")
    print(f"  llama-cli -m model.gguf --entropy-config {output_path} --kv-budget 0.5")


if __name__ == "__main__":
    main()
