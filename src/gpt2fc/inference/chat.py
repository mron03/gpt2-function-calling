"""Single-turn demo: give the model a function schema and a user message,
get back an assistant turn (typically a <functioncall> block).

Example:
    python -m gpt2fc.inference.chat \
        --checkpoint checkpoints/gpt2-355M-function-calling.pth \
        --user "What's the weather like in Almaty?"
"""

import argparse
import json

from gpt2fc.config import MODEL_CONFIGS, get_device
from gpt2fc.inference.generate import get_tokenizer, load_finetuned_model, run_inference
from gpt2fc.inference.parser import extract_functioncall

DEFAULT_SCHEMA = {
    "name": "get_current_weather",
    "description": "Get the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city, e.g. San Francisco"
            }
        },
        "required": ["location"]
    }
}


def build_prompt(schema, user_message):
    schema_str = json.dumps(schema, indent=4) if isinstance(schema, dict) else schema
    return (
        "###SYSTEM: You are a helpful assistant with access to the following functions. "
        f"Use them if required -\n{schema_str}\n"
        f"###USER: {user_message}"
    )


def main():
    parser = argparse.ArgumentParser(description="Demo the fine-tuned function-calling model on a single prompt")
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--model-size",     default="355M", choices=list(MODEL_CONFIGS))
    parser.add_argument("--user",           required=True, help="User message")
    parser.add_argument("--schema",         default=None, help="Function schema as a JSON string (default: weather example)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--device",         default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_finetuned_model(args.checkpoint, args.model_size, device)
    tokenizer = get_tokenizer()

    schema = json.loads(args.schema) if args.schema else DEFAULT_SCHEMA
    prompt = build_prompt(schema, args.user)

    print(prompt)
    reply = run_inference(model, tokenizer, prompt, args.max_new_tokens, device, temperature=args.temperature)
    print(reply)

    fc = extract_functioncall(reply)
    if fc is not None:
        print("\nParsed function call:")
        print(json.dumps(fc, indent=2))


if __name__ == "__main__":
    main()
