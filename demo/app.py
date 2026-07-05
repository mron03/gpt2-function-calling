"""Gradio demo: base GPT-2 355M vs the function-calling fine-tune, side by side.

Runs on the free CPU tier of Hugging Face Spaces. Both checkpoints are plain
PyTorch state_dicts for the hand-written GPTModel in gpt2fc — no TensorFlow,
no transformers.
"""

import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from gpt2fc.config import EOS_TOKEN_ID, get_model_config
from gpt2fc.inference.generate import get_tokenizer
from gpt2fc.inference.parser import extract_functioncall
from gpt2fc.model import GPTModel

WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO", "noFFENSE/gpt2-355M-function-calling")

torch.set_num_threads(os.cpu_count() or 2)

DEFAULT_SCHEMA = json.dumps(
    {
        "name": "get_current_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city, e.g. San Francisco"}
            },
            "required": ["location"],
        },
    },
    indent=2,
)

EXAMPLES = [
    ["What's the weather like in Almaty right now?", DEFAULT_SCHEMA],
    [
        "I need to convert 100 US dollars to euros",
        json.dumps(
            {
                "name": "convert_currency",
                "description": "Convert an amount from one currency to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "from_currency": {"type": "string"},
                        "to_currency": {"type": "string"},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
            indent=2,
        ),
    ],
    [
        "Set a reminder to call my mom tomorrow at 6pm",
        json.dumps(
            {
                "name": "create_reminder",
                "description": "Create a reminder for a specific time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "time": {"type": "string"},
                    },
                    "required": ["task", "time"],
                },
            },
            indent=2,
        ),
    ],
    ["Hey, how are you today?", DEFAULT_SCHEMA],
]


def load_model(filename):
    path = hf_hub_download(repo_id=WEIGHTS_REPO, filename=filename)
    model = GPTModel(get_model_config("355M"))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

print("Loading checkpoints (first start downloads ~3.2 GB)...")
FINETUNED = load_model("gpt2-355M-function-calling.pth")
BASE = load_model("gpt2-355M-base.pth")
TOKENIZER = get_tokenizer()
print("Ready.")


def build_prompt(schema_str, user_message):
    return (
        "###SYSTEM: You are a helpful assistant with access to the following functions. "
        f"Use them if required -\n{schema_str}\n"
        f"###USER: {user_message}"
    )


@torch.no_grad()
def stream_generate(model, prompt, max_new_tokens):
    """Greedy decoding, yielding the decoded continuation as it grows."""
    idx = torch.tensor(TOKENIZER.encode(prompt, allowed_special={"<|endoftext|>"})).unsqueeze(0)
    prompt_len = len(prompt)
    context = model.context_length
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context:])
        next_idx = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_idx.item() == EOS_TOKEN_ID:
            break
        idx = torch.cat((idx, next_idx), dim=1)
        yield TOKENIZER.decode(idx.squeeze(0).tolist())[prompt_len:]


def run(user_message, schema_str, max_new_tokens):
    if not user_message.strip():
        raise gr.Error("Type a message first.")
    try:
        json.loads(schema_str)
    except json.JSONDecodeError as e:
        raise gr.Error(f"Schema is not valid JSON: {e}")

    prompt = build_prompt(schema_str, user_message)
    ft_out, base_out, parsed = "", "", ""

    for ft_out in stream_generate(FINETUNED, prompt, max_new_tokens):
        yield ft_out, parsed, base_out
    fc = extract_functioncall(ft_out)
    parsed = json.dumps(fc, indent=2) if fc else "(no function call parsed — conversational reply)"
    yield ft_out, parsed, base_out

    for base_out in stream_generate(BASE, prompt, max_new_tokens):
        yield ft_out, parsed, base_out
    if not base_out.strip():
        base_out = "(only whitespace — the base model pads the JSON blob forever)"
    yield ft_out, parsed, base_out


with gr.Blocks(title="GPT-2 function calling — before vs after") as demo:
    gr.Markdown(
        "# GPT-2, from scratch, learns to call functions\n"
        "GPT-2 355M implemented in raw PyTorch and fine-tuned on "
        "[Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2). "
        "Give it a function schema and a request — the fine-tuned model emits a structured call, "
        "the untouched base model shows why that isn't free. "
        "[Code](https://github.com/mron03/gpt2-function-calling) · "
        "[write-up](https://mron03.github.io/gpt2-function-calling/)\n\n"
        "*Free CPU Space: expect ~1–2 tokens/s, and the fine-tuned model streams first. "
        "The models share weights with nothing — no transformers, no KV cache, just the loop.*"
    )
    with gr.Row():
        with gr.Column():
            user_message = gr.Textbox(label="Your message", placeholder="What's the weather like in Almaty right now?")
            schema = gr.Code(label="Function schema (JSON)", value=DEFAULT_SCHEMA, language="json", lines=12)
            max_tokens = gr.Slider(16, 96, value=64, step=8, label="Max new tokens")
            btn = gr.Button("Generate with both models", variant="primary")
        with gr.Column():
            ft_box = gr.Textbox(label="Fine-tuned 355M", lines=5)
            parsed_box = gr.Textbox(label="Parsed function call", lines=6)
            base_box = gr.Textbox(label="Base GPT-2 355M (no fine-tuning)", lines=5)
    gr.Examples(examples=EXAMPLES, inputs=[user_message, schema])
    btn.click(run, inputs=[user_message, schema, max_tokens], outputs=[ft_box, parsed_box, base_box])
    user_message.submit(run, inputs=[user_message, schema, max_tokens], outputs=[ft_box, parsed_box, base_box])

demo.launch()
