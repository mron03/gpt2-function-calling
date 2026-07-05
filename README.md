# GPT-2 Function Calling — from scratch

GPT-2 implemented **from scratch in PyTorch** and fine-tuned to do **function calling**: given a JSON function schema and a user request, the model decides whether to call the function and emits a parseable `<functioncall>` JSON payload with the right arguments.

No Hugging Face `transformers`, no PEFT — the transformer, the weight loading from OpenAI's original TF checkpoints, the training loop, and the evaluation harness are all hand-written. Based on the methodology of Sebastian Raschka's *Build a Large Language Model From Scratch* (ch. 7), extended from instruction tuning to structured function calling.

```text
###SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "get_current_weather",
    "description": "Get the current weather for a location",
    "parameters": { ... "location": {"type": "string"} ... }
}
###USER: What's the weather like in Almaty right now?
###ASSISTANT: <functioncall> {"name": "get_current_weather", "arguments": '{"location": "Almaty"}'}   ← model output
```

## Results

GPT-2 **355M** fine-tuned for 1 epoch on the full Glaive Function Calling v2 dataset (112,960 samples, Kaggle T4, ~9.4 h). Evaluated on 200 held-out test samples (84 of which have a function call in the ground truth); metrics are strict — an unparseable prediction counts as an error in every row.

| Metric | Value |
|---|---|
| Function-call parse rate | **86.9%** |
| Function name accuracy | **85.7%** |
| Argument keys accuracy | **79.8%** |
| Exact match (name + all argument values) | **75.0%** |

Of the 84 expected calls, the model produced a parseable call with the correct function name in all but one of the parsed cases. The main failure mode isn't hallucinated functions — it's the model replying conversationally ("Sure, let me calculate that for you") instead of emitting the call, plus occasional garbled JSON on long nested arguments. Full per-sample outputs: `gpt2fc-eval` writes `results/eval-results.json`; the summary is committed as [`results/metrics-355M.json`](results/metrics-355M.json).

![Training loss](results/loss-355M.png)

Loss is cross-entropy over **response tokens only** — prompt and padding tokens are masked out with `ignore_index=-100`. Validation loss drops from 2.53 (pretrained baseline) to ~0.13.

### Before / after fine-tuning

Same prompt (the weather example above), same greedy decoding — only the weights differ:

| | Model output |
|---|---|
| **Base GPT-2 355M** | `{` followed by 128 tokens of whitespace — a web-text predictor sees a JSON blob and keeps padding it, with no concept of the dialog format, the tool, or when to stop |
| **Fine-tuned 355M** | `###ASSISTANT: <functioncall> {"name": "get_current_weather", "arguments": '{"location": "Almaty"}'}` — then a clean end-of-text |

One epoch of fine-tuning teaches the *protocol*: answer as the assistant, treat the schema as a callable tool, pull the arguments out of the user's sentence, and stop.

## Project structure

The repo is organized around the three stages of the project:

```text
src/gpt2fc/
├── model/          1️⃣  Manual GPT-2 implementation
│   ├── attention.py       causal multi-head attention
│   ├── layers.py          LayerNorm, GELU (tanh approx.), FeedForward, TransformerBlock
│   ├── gpt2.py            GPTModel: embeddings → N blocks → LayerNorm → LM head
│   └── pretrained.py      download OpenAI TF checkpoints + map weights onto the model
├── training/       2️⃣  Fine-tuning pipeline
│   ├── data.py            Glaive loading & 85/10/5 split, prompt formatting,
│   │                      pre-tokenized Dataset, collate fn with loss masking
│   └── train.py           CLI: fine-tune GPT-2 (124M–1.5B) on Glaive
└── inference/      3️⃣  Inference & evaluation
    ├── generate.py        greedy / temperature / top-k decoding
    ├── parser.py          robust <functioncall> JSON extraction
    ├── evaluate.py        CLI: benchmark on the test split
    └── chat.py            CLI: single-turn demo with any function schema

tests/              25 unit tests (model, data pipeline, parser, decoding, config) — no network, no weights
notebooks/          the learning journey: data exploration → architecture → training
cloud/              self-contained Kaggle/Colab training & eval notebooks + Azure ML job
```

## How it works

**Model.** GPT-2 architecture written in ~200 lines of PyTorch: learned token + positional embeddings, pre-norm transformer blocks (causal multi-head attention → residual, GELU feed-forward → residual), final LayerNorm, untied LM head. Weights for the 124M–1.5B checkpoints are transferred tensor-by-tensor from OpenAI's original TensorFlow checkpoints, with shape checks on every assignment.

**Data.** Each Glaive sample is a function schema (`system`) plus a dialog (`chat`). `format_entry` rewrites role markers to `###SYSTEM:` / `###USER:` / `###ASSISTANT:` sentinels and splits at the first assistant turn: everything before is the prompt, the assistant turn is the training target. The collate function pads to batch max length, appends EOS, shifts targets by one, and masks prompt + padding tokens so the loss teaches only the assistant's behavior.

**Training.** Plain AdamW (`lr=5e-5`, `weight_decay=0.1`) full fine-tune, batch size 8, sequences capped at 512 tokens. Trained on Kaggle's free T4 (also reproducible on Colab or Azure ML — see `cloud/`).

**Evaluation.** The tricky part is that Glaive (and therefore the fine-tuned model) emits *almost*-JSON: `{"name": "f", "arguments": '{"k": "v"}'}` — the arguments object is wrapped in single quotes. The parser finds the payload span by brace-depth counting (regexes fail on nested objects), then unwraps the single-quoted arguments blob before `json.loads`. On the ground-truth test split this parses **100%** of function calls, vs ~4% for a naive `json.loads`.

## Quick start

```bash
uv venv --python=python3.10 && source .venv/bin/activate
uv pip install -e ".[train,dev]"
pytest                       # 25 tests, runs in ~2s
```

**Demo** (downloads nothing; needs a fine-tuned checkpoint in `checkpoints/`):

```bash
gpt2fc-chat --checkpoint checkpoints/gpt2-355M-function-calling.pth \
    --user "What's the weather like in Almaty right now?"
```

```text
###ASSISTANT: <functioncall> {"name": "get_current_weather", "arguments": '{"location": "Almaty"}'}

Parsed function call:
{ "name": "get_current_weather", "arguments": { "location": "Almaty" } }
```

You can pass any schema: `--schema '{"name": "book_flight", "parameters": {...}}'`.

**Train** (auto-downloads the pretrained GPT-2 weights on first run):

```bash
# local smoke test (--device cpu: PyTorch MPS backward kernels can emit NaN grads)
gpt2fc-train --data-slice 200 --num-epochs 1 --batch-size 2 --allowed-max-length 256 --device cpu

# full run (CUDA GPU recommended; see cloud/ for Kaggle/Colab/Azure recipes)
gpt2fc-train --model-size 355M --num-epochs 1 --batch-size 8 --data-slice -1
```

Inference and evaluation run fine on Apple Silicon (MPS) — only training gradients hit the MPS kernel bug (verified via `torch.autograd.set_detect_anomaly`: `LinearBackward0` returns NaN; CPU and CUDA are unaffected).

**Evaluate**:

```bash
gpt2fc-eval --checkpoint checkpoints/gpt2-355M-function-calling.pth \
    --model-size 355M --num-samples 200 --output-json results/eval-results.json
```

## What I learned / limitations

- A 355M model trained for one epoch learns the *format* essentially perfectly (near-100% parseable calls) and generalizes to unseen schemas and argument values — but it inherits its training distribution's quirks: it reproduces Glaive's single-quoted arguments style, and it refuses requests (e.g. flight booking) that Glaive's assistants habitually refused.
- Evaluation infrastructure matters as much as the model: a parser bug made a well-trained model look completely broken (4% vs 100% parse rate on identical outputs).
- Decoding is deliberately simple (no KV cache, batch size 1) — generation is O(n²); a KV cache is the natural next optimization.
- Single-turn only: the model sees `SYSTEM + USER → ASSISTANT`. Multi-turn function calling (with `FUNCTION RESPONSE` turns) is future work.

## Acknowledgments

- Architecture and training methodology: [Sebastian Raschka — *Build a Large Language Model From Scratch*](https://github.com/rasbt/LLMs-from-scratch) (Apache 2.0)
- Dataset: [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- Pretrained weights: OpenAI GPT-2 TF checkpoints

Licensed under [Apache 2.0](LICENSE).
