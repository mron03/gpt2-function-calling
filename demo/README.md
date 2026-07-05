---
title: GPT-2 Function Calling — Before vs After
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Hand-written GPT-2 355M fine-tuned to emit function calls
models:
  - noFFENSE/gpt2-355M-function-calling
---

# GPT-2 function calling — before vs after

Side-by-side demo of GPT-2 355M **implemented from scratch in PyTorch** (no
`transformers`) and fine-tuned for function calling on Glaive Function Calling
v2 — against the untouched pretrained base model on the same prompt.

- Code, tests, and training pipeline: <https://github.com/mron03/gpt2-function-calling>
- Full write-up with animations: <https://mron03.github.io/gpt2-function-calling/>
- Weights: <https://huggingface.co/noFFENSE/gpt2-355M-function-calling>

Runs on the free CPU tier with the hand-rolled KV cache — generation streams
at a few tokens per second, one hand-written attention layer at a time.
