"""Gradio demo: base GPT-2 355M vs the function-calling fine-tune, side by side.

Runs on the free CPU tier of Hugging Face Spaces. Both checkpoints are plain
PyTorch state_dicts for the hand-written GPTModel in gpt2fc — no TensorFlow,
no transformers. Decoding uses the KV cache, so each step feeds one token.
"""

import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from gpt2fc.config import EOS_TOKEN_ID, get_model_config
from gpt2fc.inference.generate import get_tokenizer
from gpt2fc.inference.parser import extract_functioncall
from gpt2fc.model import GPTModel, KVCache

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


def validate_schemas(schema_str):
    """Accept one JSON schema or several stacked ones (Glaive lists multiple
    functions as concatenated JSON objects separated by a blank line)."""
    decoder = json.JSONDecoder()
    text = schema_str.strip()
    if not text:
        raise gr.Error("Schema is empty.")
    idx = 0
    while idx < len(text):
        try:
            _, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            raise gr.Error(f"Schema is not valid JSON: {e}")
        idx = end
        while idx < len(text) and text[idx].isspace():
            idx += 1


def build_prompt(schema_str, user_message):
    return (
        "###SYSTEM: You are a helpful assistant with access to the following functions. "
        f"Use them if required -\n{schema_str}\n"
        f"###USER: {user_message}"
    )


def preview_prompt(user_message, schema_str):
    return build_prompt(schema_str, user_message.strip() or "<your message>")


@torch.no_grad()
def stream_generate(model, prompt, max_new_tokens):
    """KV-cached greedy decoding, yielding the decoded continuation as it grows."""
    ids = TOKENIZER.encode(prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor(ids).unsqueeze(0)
    cache = KVCache()
    logits = model(idx[:, -model.context_length:], kv_cache=cache)
    generated = []
    for _ in range(max_new_tokens):
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_id.item() == EOS_TOKEN_ID:
            break
        generated.append(next_id.item())
        yield TOKENIZER.decode(generated)
        if cache.size >= model.context_length:
            break
        logits = model(next_id, kv_cache=cache)


def run(user_message, schema_str, max_new_tokens):
    if not user_message.strip():
        raise gr.Error("Type a message first.")
    validate_schemas(schema_str)

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
        "GPT-2 355M implemented in raw PyTorch (no `transformers`) and fine-tuned on "
        "[Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2). "
        "Describe what you want — the fine-tuned model emits a structured function call, "
        "while the untouched base model shows what fine-tuning is for. "
        "[Code](https://github.com/mron03/gpt2-function-calling) · "
        "[write-up](https://mron03.github.io/gpt2-function-calling/)"
    )
    with gr.Row():
        with gr.Column():
            user_message = gr.Textbox(
                label="1 · Your message",
                placeholder="What's the weather like in Almaty right now?",
            )
            with gr.Accordion("2 · Function schema — edit it, invent your own tool", open=False):
                schema = gr.Code(value=DEFAULT_SCHEMA, language="json", lines=14, label="JSON schema")
                gr.Markdown(
                    "*You can list **several** tools: stack JSON objects separated by a blank "
                    "line, like the training data does. Tip: the model favors the first one, "
                    "so put the most relevant tool on top.*"
                )
            with gr.Accordion("3 · The exact prompt the model receives", open=True):
                prompt_view = gr.Textbox(
                    value=preview_prompt("", DEFAULT_SCHEMA),
                    lines=10,
                    max_lines=16,
                    show_label=False,
                    interactive=False,
                )
                gr.Markdown(
                    "*This full text — role sentinels, schema and all — is what gets tokenized "
                    "and fed to both models. They were trained to continue it with an "
                    "`###ASSISTANT:` turn.*"
                )
            max_tokens = gr.Slider(16, 128, value=64, step=8, label="Max new tokens")
            btn = gr.Button("Generate with both models", variant="primary")
        with gr.Column():
            ft_box = gr.Textbox(label="✅ Fine-tuned 355M", lines=5)
            parsed_box = gr.Textbox(label="Parsed function call", lines=7)
            base_box = gr.Textbox(label="❌ Base GPT-2 355M (no fine-tuning)", lines=5)
            gr.Markdown(
                "*Free CPU hardware — a few tokens per second. The fine-tuned model streams "
                "first; the base model follows on the identical prompt.*"
            )

    user_message.change(preview_prompt, inputs=[user_message, schema], outputs=prompt_view)
    schema.change(preview_prompt, inputs=[user_message, schema], outputs=prompt_view)
    btn.click(run, inputs=[user_message, schema, max_tokens], outputs=[ft_box, parsed_box, base_box])
    user_message.submit(run, inputs=[user_message, schema, max_tokens], outputs=[ft_box, parsed_box, base_box])

demo.launch()
