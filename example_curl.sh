# see /Users/dylan/wtcode/BitNet/3rdparty/llama.cpp/examples/server
# for example API endpoints
curl -X POST --no-buffer http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '
{"model":"models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
  "stream":true,
  "messages": [
  {
    "role": "system",
    "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
  },
{
  "role": "user",
  "content": "Write a limerick about python exceptions"
}
],
"max_tokens":64}'
