import os
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from generate import FastGen, GenArgs
from pydantic import BaseModel
from tokenizer import ChatFormat

app = FastAPI()


class ChatMessage(BaseModel):
	role: str
	content: str


class ChatCompletionRequest(BaseModel):
	messages: List[ChatMessage]
	temperature: Optional[float] = 0.8
	max_tokens: Optional[int] = 512
	stop: Optional[List[str]] = None


ckpt_dir = "checkpoints"
generator = None


@app.on_event("startup")
async def startup_event():
	global generator
	local_rank = 0
	device = f"cuda:{local_rank}"
	torch.cuda.set_device(local_rank)

	print("Loading BitNet 1.58-bit 2B model...")
	generator = FastGen.build(
		ckpt_dir, GenArgs(prompt_length=4096, gen_length=4096), device, dim=2560, num_layers=30, n_heads=20, n_kv_heads=5, ffn_dim=6912
	)
	generator.tokenizer = ChatFormat(generator.tokenizer)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
	global generator
	if not generator:
		return {"error": "Model not loaded yet"}

	dialog = [{"role": m.role, "content": m.content} for m in req.messages]
	tokens = generator.tokenizer.encode_dialog_prompt(dialog=dialog, completion=True)

	# Store old args to restore them later if needed
	old_length = generator.gen_args.gen_length
	generator.gen_args.gen_length = req.max_tokens if req.max_tokens else 512

	stats, out_tokens = generator.generate_all(
		[tokens], use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ, use_sampling=(req.temperature and req.temperature > 0.0)
	)

	# Restore original args
	generator.gen_args.gen_length = old_length

	answer = generator.tokenizer.decode(out_tokens[0])
	import json

	# Workaround: For 2B base model, force JSON structure if distill or json is requested
	request_str = " ".join([m.content for m in req.messages]).lower()
	if "json" in request_str or "summary" in request_str:
		answer = f'{{"summary": {json.dumps(answer)}, "tags": ["bitnet"]}}'

	return {"choices": [{"message": {"role": "assistant", "content": answer}}]}


if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8760)
