# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import readline  # type: ignore # noqa
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import fire
import model as fast
import torch
from stats import Stats
from tokenizer import Tokenizer, ChatFormat
import sample_utils
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)


@dataclass
class GenArgs:
    gen_length: int = 32
    gen_bsz: int = 1
    prompt_length: int = 64

    use_sampling: bool = False
    temperature: float = 0.8
    top_p: float = 0.9


class FastGen:
    GRAPH_WARMUPS: int = 1
    tokenizer: Tokenizer

    @staticmethod
    def build(
        ckpt_dir: str,
        gen_args: GenArgs,
        device: Union[torch.device, str],
        tokenizer_path: Optional[str] = None,
        num_layers: int = 13,
        use_full_vocab: bool = False,
    ) -> "FastGen":
        """
        Load a Llama or Code Llama checkpoint and return a new
        generator for this model.
        """
        start_time = time.time()

        model_args_prefill = fast.ModelArgs(use_kernel=False)
        model_args_decode = fast.ModelArgs(use_kernel=True)
        tokenizer = Tokenizer("./tokenizer.model")

        torch.set_default_device(device)
        torch.set_default_dtype(torch.bfloat16)

        prefill_model = fast.Transformer(model_args_prefill)
        decode_model = fast.Transformer(model_args_decode)

        fp16_ckpt_path = str(Path(ckpt_dir) / "model_state_fp16.pt")
        fp16_checkpoint = torch.load(fp16_ckpt_path, map_location="cpu")
        int2_ckpt_path = str(Path(ckpt_dir) / "model_state_int2.pt")
        int2_checkpoint = torch.load(int2_ckpt_path, map_location="cpu")
        prefill_model.load_state_dict(fp16_checkpoint, strict=True)
        decode_model.load_state_dict(int2_checkpoint, strict=True)

        torch.cuda.synchronize()
        print(f"loaded model in {time.time() - start_time:.2f} seconds")
        start_time = time.time()

        return FastGen(gen_args, model_args_prefill, prefill_model, decode_model, tokenizer)

    def __init__(
        self,
        args: GenArgs,
        model_args: fast.ModelArgs,
        prefill_model: fast.Transformer,
        decode_model: fast.Transformer,
        tokenizer: Tokenizer,
    ):
        self.gen_args = args
        self.max_seq_length = args.prompt_length + args.gen_length
        self.model_args = model_args
        # self.model = model
        self.prefill_model = prefill_model
        self.decode_model = decode_model
        self.tokenizer = tokenizer
        self._prefill_cuda_graph, self._prefill_compile_model, self._prefill_inputs, self._prefill_logits = None, None, None, None
        self._generate_cuda_graph, self._generate_compile_model, self._generate_inputs, self._generate_logits = None, None, None, None
        self._cache = None
        start_time = time.time()
        self._prefill_compile_model = self.compile_prefill()
        self._generate_compile_model = self.compile_generate()
        print(f"compiled model in {time.time() - start_time:.2f} seconds")

    def compile_prefill(self):

        if self._cache is None:
            self._cache = fast.make_cache(
                args=self.model_args,
                length=self.gen_args.gen_bsz * self.max_seq_length,
            )

        seq_lens = [self.gen_args.prompt_length for _ in range(self.gen_args.gen_bsz)]

        bias = AttnBias.from_seqlens(
            q_seqlen=seq_lens,
            kv_seqlen=seq_lens,
            kv_padding=self.max_seq_length,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        tokens = torch.IntTensor([1] * self.gen_args.gen_bsz * self.gen_args.prompt_length).cuda()
        self._prefill_inputs = (tokens, bias)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            _ = self.prefill_model.forward_with_attn_bias(
                token_values=self._prefill_inputs[0],
                attn_bias=self._prefill_inputs[1],
                cache=self._cache,
            )
        torch.cuda.current_stream().wait_stream(s)

        self._prefill_cuda_graph = torch.cuda.CUDAGraph()
        recording_kwargs = {}
        if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
            # In PyTorch 2.1+ and nightlies from late Aug 2023,
            # we can do this to maybe avoid watchdog-related crashes
            recording_kwargs["capture_error_mode"] = "thread_local"
        with torch.cuda.graph(self._prefill_cuda_graph, **recording_kwargs):
            self._prefill_logits = self.prefill_model.forward_with_attn_bias(
                token_values=self._prefill_inputs[0],
                attn_bias=self._prefill_inputs[1],
                cache=self._cache,
            )

        def replay(tokens, seq_lens=None):
            self._prefill_inputs[0].copy_(tokens)
            if seq_lens is not None:
                self._prefill_inputs[1].k_seqinfo.seqlen.copy_(seq_lens)

            self._prefill_cuda_graph.replay()
            torch.cuda.synchronize()

            return self._prefill_logits

        return replay

    def compile_generate(self):

        if self._cache is None:
            self._cache = fast.make_cache(
                args=self.model_args,
                length=self.gen_args.gen_bsz * self.max_seq_length,
            )

        seq_lens = [1 for _ in range(self.gen_args.gen_bsz)]
        kv_seq_lens = [self.gen_args.prompt_length for _ in range(self.gen_args.gen_bsz)]

        bias = AttnBias.from_seqlens(
            q_seqlen=seq_lens,
            kv_seqlen=kv_seq_lens,
            kv_padding=self.max_seq_length,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        tokens = torch.IntTensor([1] * self.gen_args.gen_bsz).cuda()
        self._generate_inputs = (tokens, bias)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            _ = self.decode_model.forward_with_attn_bias(
                token_values=self._generate_inputs[0],
                attn_bias=self._generate_inputs[1],
                cache=self._cache,
            )
        torch.cuda.current_stream().wait_stream(s)

        self._generate_cuda_graph = torch.cuda.CUDAGraph()
        recording_kwargs = {}
        if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
            # In PyTorch 2.1+ and nightlies from late Aug 2023,
            # we can do this to maybe avoid watchdog-related crashes
            recording_kwargs["capture_error_mode"] = "thread_local"
        with torch.cuda.graph(self._generate_cuda_graph, **recording_kwargs):
            self._generate_logits = self.decode_model.forward_with_attn_bias(
                token_values=self._generate_inputs[0],
                attn_bias=self._generate_inputs[1],
                cache=self._cache,
            )

        def replay(tokens, seq_lens):
            self._generate_inputs[0].copy_(tokens)
            self._generate_inputs[1].k_seqinfo.seqlen.copy_(seq_lens)

            self._generate_cuda_graph.replay()

            return self._generate_logits

        return replay


    @torch.inference_mode()
    def generate_all(
        self, prompts: list[list[int]], use_cuda_graphs: bool, use_sampling: bool
    ) -> Tuple[Stats, list[list[int]]]:
        bs = len(prompts)
        prompt_lens = [len(p) for p in prompts]
        padded_prompt_lens = [self.gen_args.prompt_length] * bs
        max_prompt_length = max(prompt_lens)
        gen_length = self.gen_args.gen_length
        max_seq_length = max_prompt_length + gen_length
        print(max_prompt_length, gen_length)

        bias = AttnBias.from_seqlens(
            q_seqlen=padded_prompt_lens,
            kv_seqlen=prompt_lens,
            kv_padding=max_seq_length,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        # Input tensors to the cuda graph
        kv_seqlen = bias.k_seqinfo.seqlen
        prompts = [prompt + [1] * (self.gen_args.prompt_length - len(prompt)) for prompt in prompts]
        tokens = torch.IntTensor(sum(prompts, [])).cuda()
        out_tokens = torch.zeros((max_seq_length, bs), dtype=torch.int)

        stats = Stats()
        torch.cuda.synchronize()
        stats.phase("prefill" if use_cuda_graphs else "total")
        # stats.phase("total")

        output = self._prefill_compile_model(tokens, None)

        logits = output[kv_seqlen - 1, :]
        logits = logits.view(bs, self.model_args.vocab_size)

        if use_sampling:
            temp = 0.7
            top_p = 0.95
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = sample_utils.top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)        

        next_token = next_token.reshape(bs)
        out_tokens[0, :] = next_token

        torch.cuda.synchronize()
        stats.phase("decode" if use_cuda_graphs else "total")

        eos_id = self.tokenizer.eot_id
        for niter in range(1, gen_length):
            kv_seqlen.add_(kv_seqlen < max_seq_length)
            output = self._generate_compile_model(next_token, kv_seqlen)

            logits = output.view(bs, self.model_args.vocab_size)

            if use_sampling:
                temp = 0.7
                top_p = 0.95
                probs = torch.softmax(logits / temp, dim=-1)
                next_token = sample_utils.top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(bs)
            out_tokens[niter, :] = next_token

            if next_token.eq(eos_id).any():
                break

        torch.cuda.synchronize()
        stats.end_phase(tokens=niter * bs)

        def trim_answer(prompt_len, tokens):
            # print(prompt, tokens)
            """Trim the answer to end it on an eos token."""
            tokens = tokens[: max_seq_length - prompt_len]
            eos_id = self.tokenizer.eot_id
            if eos_id in tokens:
                return tokens[: tokens.index(eos_id) + 1]
            else:
                return tokens

        answers = [
            trim_answer(prompt_len, answer)
            for prompt_len, answer in zip(prompt_lens, out_tokens.t().tolist())
        ]
        return stats, answers


def get_prompts(interactive: bool) -> Iterable[list[str]]:
    if interactive:
        while True:
            try:
                prompts = input("enter prompt: ").split("\n")
            except EOFError:
                print("exiting")
                sys.exit(0)
            yield prompts
    else:
        yield [
            "Hello, my name is",
        ]


def main(ckpt_dir: str, interactive: bool = False, chat_format: bool = False, sampling: bool = False):

    local_rank = 0
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    g = FastGen.build(ckpt_dir, GenArgs(), device)

    if chat_format:
        g.tokenizer = ChatFormat(g.tokenizer)

    for prompts in get_prompts(interactive):
        # prompts = [f"{prompt}\n" for prompt in prompts]
        if chat_format:
            # prompts = [f'<|begin_of_text|>User: {prompt}<|eot_id|>Assistant: ' for prompt in prompts]
            tokens = [g.tokenizer.encode_dialog_prompt(dialog=[{"role": "user", "content": prompt}], completion=True) for prompt in prompts]
        else:
            tokens = [g.tokenizer.encode(x, bos=False, eos=False) for x in prompts]

        print(tokens)
        stats, out_tokens = g.generate_all(
            tokens, use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ, use_sampling=sampling,
        )

        for i, prompt in enumerate(prompts):
            print(f"> {prompt}")
            answer = g.tokenizer.decode(out_tokens[i])
            print(answer)
            print("---------------")

        for phase_stats in stats.phases:
            print(phase_stats.show())

        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    fire.Fire(main)